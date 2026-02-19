"""
Example Generator - Select relevant few-shot examples for LLM prompts.

Features:
- Multi-strategy example selection (keywords, tables, patterns)
- Diversity selection to avoid redundant examples
- Schema validation before returning examples
- Scoring and ranking by relevance
"""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

from .knowledge_store import KnowledgeStore
from .query_analyzer import QueryPattern, QueryExample

logger = logging.getLogger(__name__)


@dataclass
class ScoredExample:
    """An example with its relevance score."""
    example: Dict
    score: float
    match_reasons: List[str]


class ExampleGenerator:
    """
    Generates relevant few-shot examples for LLM prompts.

    Uses multiple strategies to find the best examples:
    1. Table overlap - examples using the same tables
    2. Keyword matching - examples with similar keywords
    3. Pattern matching - examples of similar query type
    4. Intent similarity - similar aggregations, filters, etc.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        semantic_config: Optional[Dict] = None
    ):
        """
        Initialize the generator.

        Args:
            knowledge_store: KnowledgeStore instance
            semantic_config: Optional semantic config for validation
        """
        self.knowledge_store = knowledge_store
        self.semantic_config = semantic_config or {}
        self._known_tables: Set[str] = set()
        self._load_known_tables()

    def _load_known_tables(self):
        """Load known tables from semantic config."""
        schemas = self.semantic_config.get("schemas", {})
        for schema_name, schema_data in schemas.items():
            tables = schema_data.get("tables", {})
            for table_name in tables.keys():
                self._known_tables.add(table_name.lower())

    def refresh_schema(self, semantic_config: Dict):
        """Refresh schema knowledge."""
        self.semantic_config = semantic_config
        self._known_tables.clear()
        self._load_known_tables()

    def get_relevant_examples(
        self,
        user_query: str,
        tables: List[str],
        query_intent: Optional[Dict] = None,
        max_examples: int = 3,
        validate_schema: bool = True
    ) -> List[Dict]:
        """
        Get the most relevant examples for a user query.

        Args:
            user_query: The natural language query from user
            tables: List of resolved table names
            query_intent: Optional extracted query intent (from EntityExtractor)
            max_examples: Maximum examples to return
            validate_schema: Whether to validate examples against current schema

        Returns:
            List of example dicts, most relevant first
        """
        scored_examples: List[ScoredExample] = []

        # Strategy 1: Get examples by table overlap
        if tables:
            table_examples = self.knowledge_store.get_examples_by_tables(
                tables, limit=max_examples * 2
            )
            for ex in table_examples:
                score, reasons = self._score_by_tables(ex, tables)
                if score > 0:
                    scored_examples.append(ScoredExample(ex, score, reasons))

        # Strategy 2: Get examples by keywords from query
        keywords = self._extract_query_keywords(user_query)
        if keywords:
            keyword_examples = self.knowledge_store.get_examples_by_keywords(
                keywords, limit=max_examples * 2
            )
            for ex in keyword_examples:
                # Check if already scored
                if not any(se.example["id"] == ex["id"] for se in scored_examples):
                    score, reasons = self._score_by_keywords(ex, keywords)
                    if score > 0:
                        scored_examples.append(ScoredExample(ex, score, reasons))
                else:
                    # Boost existing score
                    for se in scored_examples:
                        if se.example["id"] == ex["id"]:
                            keyword_score, keyword_reasons = self._score_by_keywords(ex, keywords)
                            se.score += keyword_score * 0.5
                            se.match_reasons.extend(keyword_reasons)

        # Strategy 3: Get examples by query type/pattern
        if query_intent:
            intent_type = self._infer_query_type(user_query, query_intent)
            type_examples = self.knowledge_store.get_examples_by_type(
                intent_type, limit=max_examples
            )
            for ex in type_examples:
                if not any(se.example["id"] == ex["id"] for se in scored_examples):
                    score, reasons = self._score_by_intent(ex, query_intent, intent_type)
                    if score > 0:
                        scored_examples.append(ScoredExample(ex, score, reasons))
                else:
                    # Boost existing score
                    for se in scored_examples:
                        if se.example["id"] == ex["id"]:
                            se.score += 0.2
                            se.match_reasons.append(f"Same query type: {intent_type}")

        # Sort by score
        scored_examples.sort(key=lambda x: x.score, reverse=True)

        # Select diverse examples
        selected = self._select_diverse(scored_examples, max_examples)

        # Validate against current schema if requested
        if validate_schema and self._known_tables:
            selected = [
                ex for ex in selected
                if self._validate_example(ex)
            ]

        # Convert to final format
        return [
            {
                **se.example,
                "relevance_score": se.score,
                "match_reasons": se.match_reasons,
            }
            for se in selected[:max_examples]
        ]

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract keywords from user query for matching."""
        # Stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'for', 'by', 'to', 'in', 'of',
            'with', 'show', 'get', 'list', 'find', 'all', 'from', 'query',
            'give', 'me', 'please', 'want', 'need', 'can', 'you', 'i',
            'what', 'how', 'where', 'when', 'who', 'which', 'that', 'this'
        }

        words = re.findall(r'\b[a-z]+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _score_by_tables(
        self,
        example: Dict,
        target_tables: List[str]
    ) -> Tuple[float, List[str]]:
        """Score example by table overlap."""
        example_tables = set(t.lower() for t in example.get("tables", []))
        target_set = set(t.lower() for t in target_tables)

        if not example_tables or not target_set:
            return 0.0, []

        # Calculate Jaccard similarity
        intersection = len(example_tables & target_set)
        union = len(example_tables | target_set)

        if union == 0:
            return 0.0, []

        score = intersection / union
        reasons = []

        if intersection > 0:
            matching = example_tables & target_set
            reasons.append(f"Tables: {', '.join(matching)}")

        # Bonus for exact match
        if example_tables == target_set:
            score += 0.3
            reasons.append("Exact table match")

        return min(score, 1.0), reasons

    def _score_by_keywords(
        self,
        example: Dict,
        target_keywords: List[str]
    ) -> Tuple[float, List[str]]:
        """Score example by keyword overlap."""
        example_keywords = set(k.lower() for k in example.get("keywords", []))
        target_set = set(k.lower() for k in target_keywords)

        if not example_keywords or not target_set:
            return 0.0, []

        intersection = len(example_keywords & target_set)

        if intersection == 0:
            return 0.0, []

        # Score based on proportion of target keywords matched
        score = intersection / len(target_set) * 0.5

        matching = example_keywords & target_set
        reasons = [f"Keywords: {', '.join(list(matching)[:3])}"]

        return score, reasons

    def _score_by_intent(
        self,
        example: Dict,
        query_intent: Dict,
        intent_type: str
    ) -> Tuple[float, List[str]]:
        """Score example by intent similarity."""
        score = 0.0
        reasons = []

        # Check query type match
        if example.get("query_type") == intent_type:
            score += 0.3
            reasons.append(f"Query type: {intent_type}")

        # Check for aggregation match
        example_pattern = example.get("pattern", {})
        has_aggregation = bool(query_intent.get("aggregations", []))
        example_has_agg = bool(example_pattern.get("aggregations", []))

        if has_aggregation and example_has_agg:
            score += 0.2
            reasons.append("Both have aggregations")

        # Check for time filter match
        has_time = bool(query_intent.get("time_filters", []))
        example_has_time = example_pattern.get("has_time_filter", False)

        if has_time and example_has_time:
            score += 0.2
            reasons.append("Both have time filters")

        # Check for join match
        has_joins = len(query_intent.get("join_hints", [])) > 0
        example_has_joins = len(example_pattern.get("joins", [])) > 0

        if has_joins and example_has_joins:
            score += 0.2
            reasons.append("Both have joins")

        return score, reasons

    def _infer_query_type(
        self,
        query: str,
        query_intent: Dict
    ) -> str:
        """Infer the query type from query and intent."""
        query_lower = query.lower()

        # Check for aggregation keywords
        agg_keywords = ['count', 'sum', 'total', 'average', 'avg', 'how many']
        has_agg = any(kw in query_lower for kw in agg_keywords)

        # Check for join indicators
        join_keywords = ['and their', 'with their', 'join', 'along with', 'together with']
        has_join = any(kw in query_lower for kw in join_keywords)

        # Check entities
        entities = query_intent.get("entities", [])
        if isinstance(entities, dict):
            entities = entities.get("entities", [])

        dataset_count = sum(
            1 for e in entities
            if isinstance(e, dict) and e.get("type") == "dataset"
        )

        # Determine type
        if has_join or dataset_count > 1:
            if has_agg:
                return "join_aggregation"
            return "join"
        elif has_agg:
            return "aggregation"
        else:
            return "simple"

    def _select_diverse(
        self,
        scored: List[ScoredExample],
        max_count: int
    ) -> List[ScoredExample]:
        """
        Select diverse examples to avoid redundancy.

        Tries to pick examples that:
        - Have different query types
        - Use different table combinations
        - Cover different patterns
        """
        if len(scored) <= max_count:
            return scored

        selected = []
        seen_types = set()
        seen_table_combos = set()

        for se in scored:
            query_type = se.example.get("query_type", "simple")
            tables = tuple(sorted(se.example.get("tables", [])))

            # Check diversity
            is_new_type = query_type not in seen_types
            is_new_tables = tables not in seen_table_combos

            # Always take high scorers, otherwise check diversity
            if se.score >= 0.7 or is_new_type or is_new_tables:
                selected.append(se)
                seen_types.add(query_type)
                if tables:
                    seen_table_combos.add(tables)

            if len(selected) >= max_count:
                break

        return selected

    def _validate_example(self, example: ScoredExample) -> bool:
        """Check if example is still valid against current schema."""
        tables = example.example.get("tables", [])

        if not tables:
            return True  # No tables to validate

        if not self._known_tables:
            return True  # No schema to validate against

        # Check if all tables exist
        return all(t.lower() in self._known_tables for t in tables)

    def format_examples_for_prompt(
        self,
        examples: List[Dict],
        include_explanation: bool = True
    ) -> str:
        """
        Format examples for inclusion in LLM prompt.

        Args:
            examples: List of example dicts
            include_explanation: Whether to include match reasons

        Returns:
            Formatted string for prompt
        """
        if not examples:
            return ""

        lines = ["SIMILAR SUCCESSFUL QUERIES:"]

        for i, ex in enumerate(examples, 1):
            title = ex.get("title", "Query")
            sql = ex.get("sql", "")

            lines.append(f"\nExample {i}: {title}")

            if include_explanation and ex.get("match_reasons"):
                reasons = ", ".join(ex["match_reasons"][:2])
                lines.append(f"(Relevant because: {reasons})")

            lines.append(f"```sql\n{sql}\n```")

        return "\n".join(lines)

    def get_learned_patterns_for_tables(
        self,
        tables: List[str]
    ) -> str:
        """
        Get learned patterns/hints for specific tables.

        Returns a string describing common patterns seen with these tables,
        including joins, aggregations, filters, and other useful context.
        """
        if not tables:
            return ""

        examples = self.knowledge_store.get_examples_by_tables(tables, limit=30)

        if not examples:
            return ""

        # Aggregate patterns
        common_aggregations = {}  # agg -> count
        common_group_by = {}  # column -> count
        common_filters = {}  # column -> count
        join_counts = {}  # (left, right, type) -> count
        has_time_filter_count = 0
        query_types = {}

        for ex in examples:
            pattern = ex.get("pattern", {})
            if isinstance(pattern, str):
                try:
                    import json as json_mod
                    pattern = json_mod.loads(pattern)
                except Exception:
                    pattern = {}

            # Track query types
            qt = pattern.get("query_type", ex.get("query_type", "simple"))
            query_types[qt] = query_types.get(qt, 0) + 1

            # Track aggregations
            for agg in pattern.get("aggregations", []):
                common_aggregations[agg] = common_aggregations.get(agg, 0) + 1

            # Track group by columns
            for col in pattern.get("group_by", []):
                common_group_by[col] = common_group_by.get(col, 0) + 1

            # Track filter columns
            for f in pattern.get("filters", []):
                col = f.get("column", "") if isinstance(f, dict) else str(f)
                if col:
                    common_filters[col] = common_filters.get(col, 0) + 1

            # Track joins
            for j in pattern.get("joins", []):
                if isinstance(j, dict):
                    left = j.get("left_table", "")
                    right = j.get("right_table", "")
                    jtype = j.get("join_type", "INNER")
                    key = (left, right, jtype)
                    join_counts[key] = join_counts.get(key, 0) + 1

            # Track time filters
            if pattern.get("has_time_filter"):
                has_time_filter_count += 1

        # Build the learned patterns string
        lines = [f"LEARNED PATTERNS FOR {', '.join(tables)}:"]

        # Query type distribution
        if query_types:
            top_types = sorted(query_types.items(), key=lambda x: -x[1])[:3]
            type_str = ", ".join(f"{qt} ({cnt})" for qt, cnt in top_types)
            lines.append(f"- Query types seen: {type_str}")

        # Most common aggregations
        if common_aggregations:
            top_aggs = sorted(common_aggregations.items(), key=lambda x: -x[1])[:4]
            agg_str = ", ".join(f"{agg}" for agg, _ in top_aggs)
            lines.append(f"- Common aggregations: {agg_str}")

        # Most common group by columns
        if common_group_by:
            top_groups = sorted(common_group_by.items(), key=lambda x: -x[1])[:4]
            group_str = ", ".join(f"{col}" for col, _ in top_groups)
            lines.append(f"- Common GROUP BY columns: {group_str}")

        # Most common filter columns
        if common_filters:
            top_filters = sorted(common_filters.items(), key=lambda x: -x[1])[:4]
            filter_str = ", ".join(f"{col}" for col, _ in top_filters)
            lines.append(f"- Commonly filtered columns: {filter_str}")

        # Common joins
        if join_counts:
            top_joins = sorted(join_counts.items(), key=lambda x: -x[1])[:3]
            for (left, right, jtype), cnt in top_joins:
                if left and right:
                    lines.append(f"- Common join: {left} {jtype} JOIN {right}")

        # Time filter frequency
        total = len(examples)
        if has_time_filter_count > 0:
            pct = int(has_time_filter_count * 100 / total)
            if pct >= 30:
                lines.append(f"- Time filters used in {pct}% of queries")

        return "\n".join(lines) if len(lines) > 1 else ""
