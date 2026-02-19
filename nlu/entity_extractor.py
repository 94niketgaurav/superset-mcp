"""
Extract structured entities from natural language queries.
Uses LLM for robust extraction with fallback to regex patterns.
"""
import re
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict


@dataclass
class ExtractedEntity:
    """An entity extracted from the query."""
    name: str                           # e.g., "assets", "events"
    entity_type: str                    # "dataset", "column", "filter", "aggregation"
    original_text: str                  # Original text span
    confidence: float                   # 0.0 - 1.0
    modifiers: List[str] = field(default_factory=list)  # e.g., ["recent", "active"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedJoinHint:
    """A join relationship hint extracted from the query."""
    left_entity: str                    # e.g., "assets"
    right_entity: str                   # e.g., "events"
    join_key_hint: Optional[str]        # e.g., "assetId" if mentioned
    relationship: str                   # "has", "belongs_to", "linked_to"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedTimeFilter:
    """A time filter extracted from the query."""
    period: str                         # "last_week", "last_month", "custom"
    column_hint: Optional[str]          # e.g., "created_at" if mentioned
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedFilter:
    """A WHERE clause filter extracted from the query."""
    column: str                         # e.g., "tenant"
    operator: str                       # "=", "like", "in", etc.
    value: str                          # e.g., "lcmc"
    original_text: str                  # Original text span

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedStateFilter:
    """A state/status filter extracted from the query (e.g., 'not completed')."""
    state: str                          # e.g., "completed", "pending", "assigned"
    negated: bool                       # True if "not completed", False if "completed"
    original_text: str                  # Original text span
    target_entity: Optional[str] = None # Entity this applies to (e.g., "assignment")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryIntent:
    """Complete parsed intent from a natural language query."""
    entities: List[ExtractedEntity]
    join_hints: List[ExtractedJoinHint]
    time_filters: List[ExtractedTimeFilter]
    aggregations: List[str]             # ["count", "sum", "avg"]
    limit: Optional[int]
    raw_query: str
    columns_mentioned: List[str] = field(default_factory=list)
    filters: List[ExtractedFilter] = field(default_factory=list)
    state_filters: List[ExtractedStateFilter] = field(default_factory=list)  # e.g., "not completed"
    extracted_schema: Optional[str] = None  # Schema name if mentioned (e.g., "jeeves")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "join_hints": [j.to_dict() for j in self.join_hints],
            "time_filters": [t.to_dict() for t in self.time_filters],
            "filters": [f.to_dict() for f in self.filters],
            "state_filters": [s.to_dict() for s in self.state_filters],
            "aggregations": self.aggregations,
            "limit": self.limit,
            "raw_query": self.raw_query,
            "extracted_schema": self.extracted_schema,
            "columns_mentioned": self.columns_mentioned,
        }


class EntityExtractor:
    """
    Extract entities from natural language queries.

    Supports LLM-based extraction with regex fallback.
    """

    EXTRACTION_PROMPT = '''Analyze this data query and extract structured information.

Query: "{query}"

Extract:
1. Dataset/table references (nouns that likely refer to data tables)
2. Column references (specific field names like "assetId", "created_at")
3. Join relationships (how entities relate - "X and Y", "X's Y", "X with Y", "by X")
4. Time filters (date ranges, periods like "last week", "since January")
5. Aggregations (count, sum, average, total, etc.)
6. Limit/pagination hints (top N, first N, limit N)

Return ONLY valid JSON with this structure:
{{
  "entities": [
    {{"name": "entity_name", "type": "dataset", "original": "original text", "confidence": 0.9, "modifiers": []}}
  ],
  "columns_mentioned": ["assetId", "created_at"],
  "join_hints": [
    {{"left": "entity1", "right": "entity2", "key_hint": "assetId", "relationship": "linked_to"}}
  ],
  "time_filters": [
    {{"period": "last_week", "column_hint": null}}
  ],
  "aggregations": ["count"],
  "limit": null
}}
'''

    # Stop words to exclude from entity detection
    STOP_WORDS = frozenset({
        # Articles and determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'such',
        'both', 'each', 'every', 'either', 'neither', 'any', 'some', 'all',
        # Pronouns
        'me', 'my', 'our', 'their', 'its', 'your', 'his', 'her',
        'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'how',
        # Conjunctions and prepositions
        'and', 'or', 'but', 'nor', 'for', 'with', 'from', 'to',
        'in', 'on', 'at', 'by', 'about', 'above', 'after', 'before',
        'below', 'into', 'over', 'under', 'through', 'during', 'until',
        'if', 'then', 'else', 'so', 'yet', 'as', 'than',
        # Auxiliary verbs and common verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'done', 'doing',
        'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall', 'will',
        # Negations
        'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere',
        # Query action words
        'show', 'get', 'find', 'list', 'give', 'fetch', 'retrieve', 'display',
        'want', 'need', 'please', 'help', 'tell',
        # Time-related (handled separately)
        'last', 'first', 'next', 'previous', 'recent', 'current',
        'week', 'weeks', 'month', 'months', 'day', 'days', 'year', 'years',
        'hour', 'hours', 'minute', 'minutes', 'second', 'seconds', 'quarter', 'quarters',
        'today', 'yesterday', 'tomorrow', 'ago', 'since', 'now', 'date', 'time',
        # Month names (handled by time patterns)
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        # SQL and database keywords
        'data', 'query', 'table', 'tables', 'database', 'schema',
        'join', 'joined', 'joining', 'select', 'using', 'between',
        'top', 'limit', 'count', 'sum', 'average', 'avg', 'total',
        'max', 'min', 'group', 'order', 'sort', 'filter', 'filtered',
        # State/status words (often filter values, not tables)
        'completed', 'pending', 'active', 'inactive', 'enabled', 'disabled',
        'due', 'overdue', 'expired', 'assigned', 'unassigned',
        # Adverbs and misc
        'only', 'just', 'also', 'still', 'even', 'ever', 'again',
        'many', 'much', 'more', 'most', 'less', 'least', 'few',
        'details', 'information', 'info', 'based', 'related',
    })

    # Regex patterns for various extractions - comprehensive date/time patterns
    TIME_PATTERNS = [
        # Relative periods
        (r'last\s+(\d+)?\s*(day|week|month|year|hour|quarter)s?', 'relative'),
        (r'past\s+(\d+)?\s*(day|week|month|year|hour|quarter)s?', 'relative'),
        (r'previous\s+(\d+)?\s*(day|week|month|year|quarter)s?', 'relative'),
        (r'recent\s+(\d+)?\s*(day|week|month|year)s?', 'relative'),
        (r'in\s+the\s+last\s+(\d+)?\s*(day|week|month|year)s?', 'relative'),
        # Current period
        (r'this\s+(day|week|month|year|quarter)', 'current'),
        (r'current\s+(day|week|month|year|quarter)', 'current'),
        # Specific days
        (r'\btoday\b', 'today'),
        (r'\byesterday\b', 'yesterday'),
        (r'\btomorrow\b', 'tomorrow'),
        # Since patterns
        (r'since\s+(\d{4}-\d{2}-\d{2})', 'since'),
        (r'since\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'since'),
        (r'since\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?', 'since_month'),
        (r'from\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?', 'since_month'),
        (r'after\s+(\d{4}-\d{2}-\d{2})', 'after'),
        (r'before\s+(\d{4}-\d{2}-\d{2})', 'before'),
        # Date ranges
        (r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', 'range'),
        (r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', 'range'),
        # Month/year patterns (full names)
        (r'in\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?', 'in_month'),
        # Month/year patterns (abbreviated names)
        (r'in\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:\s+(\d{4}))?', 'in_month'),
        # Due date in month patterns
        (r'due\s+(?:date\s+)?(?:is\s+)?in\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)', 'due_in_month'),
        (r'in\s+(\d{4})', 'in_year'),
        (r'q([1-4])\s+(\d{4})?', 'quarter'),
        # Informal patterns
        (r'(\d+)\s*(day|week|month|year)s?\s+ago', 'ago'),
        (r'created\s+(today|yesterday|recently|last\s+\w+)', 'created'),
        (r'updated\s+(today|yesterday|recently|last\s+\w+)', 'updated'),
    ]

    AGGREGATION_PATTERNS = [
        r'\b(count|sum|average|avg|min|max|total)\b',
    ]

    JOIN_PATTERNS = [
        (r"(\w+)\s+and\s+(?:their\s+)?(\w+)", None),           # "assets and events"
        (r"(\w+)'s\s+(\w+)", None),                             # "user's orders"
        (r"(\w+)\s+with\s+(?:their\s+)?(\w+)", None),          # "users with purchases"
        (r"(\w+)\s+(?:joined|linked|connected)\s+(?:to|with)\s+(\w+)", None),
        (r"join(?:ing)?\s+(\w+)\s+(?:and|with|to)\s+(\w+)", None),
        (r"(\w+)\s+by\s+(\w+)", "join_key"),                   # "events by assetId" - captures join key
        (r"(\w+)\s+on\s+(\w+)", "join_key"),                   # "join on assetId"
    ]

    # Pattern to detect column-like identifiers (camelCase, snake_case, with Id/ID suffix)
    COLUMN_PATTERN = re.compile(
        r'\b([a-z]+(?:[A-Z][a-z]*)+|[a-z]+_[a-z_]+|[a-z]+(?:Id|ID|Uuid|UUID))\b'
    )

    # Filter patterns - detect "of column value", "where column = value", etc.
    FILTER_PATTERNS = [
        # "of tenant lcmc" or "for tenant lcmc"
        (r'\b(?:of|for|with)\s+(\w+)\s+(["\']?)(\w+)\2(?:\s|$)', 'of_pattern'),
        # "where tenant = 'lcmc'" or "where tenant is lcmc"
        (r'\bwhere\s+(\w+)\s*(?:=|is|equals?)\s*["\']?(\w+)["\']?', 'where_pattern'),
        # "tenant = lcmc" or "tenant is lcmc"
        (r'\b(\w+)\s*(?:=|is|equals?)\s*["\']?(\w+)["\']?(?:\s|$)', 'equals_pattern'),
        # "by tenant lcmc" (filtering context, not grouping)
        (r'\bby\s+(\w+)\s+(["\']?)(\w+)\2(?:\s|$)', 'by_pattern'),
        # "tenant: lcmc" or "tenant:lcmc"
        (r'\b(\w+):\s*["\']?(\w+)["\']?', 'colon_pattern'),
    ]

    # State/status filter patterns (detect "not completed", "pending", etc.)
    STATE_FILTER_PATTERNS = [
        # Negated states: "not completed", "haven't finished", "have not done"
        (r'\b(?:not|never|haven\'t|hasn\'t|have\s+not|has\s+not)\s+(completed?|finished?|done|submitted|assigned|approved|reviewed|started|active)\b', True),
        # Positive states: "is completed", "are pending", "completed"
        (r'\b(?:is|are|was|were)\s+(completed?|finished?|done|submitted|assigned|approved|reviewed|started|pending|active|inactive|overdue)\b', False),
        # Adjective-like state before noun: "pending assignments", "active users", "overdue tasks"
        (r'\b(pending|active|inactive|overdue|expired|archived|completed|approved)\s+\w+', False),
        # Standalone negative: "incomplete", "unfinished", "unassigned"
        (r'\b(incomplete|unfinished|unassigned|unapproved|unreviewed|unsubmitted)\b', True),
        # Standalone positive: just the state word after "who/which/that"
        (r'\b(?:who|which|that)\s+(?:have\s+)?(?:not\s+)?(completed?|finished?|done|submitted|assigned|approved)\b', None),  # None = check for "not" in match
    ]

    # Patterns for extracting relationships like "users who have X" or "X whose Y"
    RELATIONSHIP_PATTERNS = [
        # "users who have (not) completed the assignment"
        (r'(\w+)\s+who\s+(?:have\s+)?(?:not\s+)?(completed?|finished?|done|submitted)\s+(?:the\s+)?(\w+)', 'has_action'),
        # "users whose assignment is/are"
        (r'(\w+)\s+whose\s+(\w+)', 'possessive'),
        # "assignment whose due date"
        (r'(\w+)\s+whose\s+(\w+)\s+(\w+)', 'possessive_attr'),
    ]

    # Common column names that should NOT be treated as table names
    COMMON_COLUMN_NAMES = frozenset({
        'tenant', 'status', 'type', 'name', 'id', 'email', 'created', 'updated',
        'department', 'category', 'level', 'priority', 'state', 'region',
        'country', 'city', 'date', 'time', 'value', 'count', 'total', 'amount',
        'price', 'quantity', 'description', 'title', 'label', 'tag', 'role',
        'permission', 'active', 'enabled', 'deleted', 'archived', 'version'
    })

    # Patterns to detect schema references in queries
    SCHEMA_PATTERNS = [
        r'\bin\s+(\w+)\s+schema\b',           # "in jeeves schema"
        r'\bfrom\s+(\w+)\s+schema\b',         # "from jeeves schema"
        r'\bschema\s+(\w+)\b',                # "schema jeeves"
        r'\bin\s+(\w+)\b(?=\s+(?:for|from|give|show|get))',  # "in jeeves for/from..."
        r'\bfrom\s+(\w+)\s+(?:for|give|show|get)',           # "from jeeves for..."
        r'\bin\s+(\w+)$',                     # "in jeeves" at end of query
    ]

    def __init__(self, llm_client: Optional[Callable[[str], str]] = None,
                 known_columns: Optional[set] = None,
                 known_tables: Optional[set] = None,
                 known_schemas: Optional[set] = None):
        """
        Initialize the entity extractor.

        Args:
            llm_client: Optional callable that takes a prompt and returns LLM response.
                       If not provided, uses regex-based extraction only.
            known_columns: Optional set of known column names from the schema.
            known_tables: Optional set of known table names from the schema.
            known_schemas: Optional set of known schema names (e.g., 'jeeves', 'dexit').
        """
        self.llm_client = llm_client
        self.known_columns = known_columns or set()
        self.known_tables = known_tables or set()
        self.known_schemas = known_schemas or set()

    def extract(self, query: str) -> QueryIntent:
        """
        Extract entities from a natural language query.

        Uses LLM if available, falls back to regex patterns.

        Args:
            query: Natural language query string

        Returns:
            QueryIntent with extracted entities, joins, filters, etc.
        """
        if self.llm_client:
            try:
                return self._extract_with_llm(query)
            except Exception as e:
                # Log error and fall back to regex
                import warnings
                warnings.warn(f"LLM extraction failed, using regex fallback: {e}")

        return self._extract_with_regex(query)

    def _extract_with_llm(self, query: str) -> QueryIntent:
        """Extract using LLM."""
        prompt = self.EXTRACTION_PROMPT.format(query=query)
        response = self.llm_client(prompt)

        # Parse JSON from response
        # Handle potential markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code fence
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        data = json.loads(response)

        entities = [
            ExtractedEntity(
                name=e.get("name", ""),
                entity_type=e.get("type", "dataset"),
                original_text=e.get("original", e.get("name", "")),
                confidence=float(e.get("confidence", 0.7)),
                modifiers=e.get("modifiers", [])
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]

        join_hints = [
            ExtractedJoinHint(
                left_entity=j.get("left", ""),
                right_entity=j.get("right", ""),
                join_key_hint=j.get("key_hint"),
                relationship=j.get("relationship", "linked_to")
            )
            for j in data.get("join_hints", [])
            if j.get("left") and j.get("right")
        ]

        time_filters = [
            ExtractedTimeFilter(
                period=t.get("period", ""),
                column_hint=t.get("column_hint"),
                start_date=t.get("start"),
                end_date=t.get("end")
            )
            for t in data.get("time_filters", [])
            if t.get("period")
        ]

        # Also extract schema and state filters using regex (LLM doesn't know about these)
        query_lower = query.lower()
        extracted_schema = self._extract_schema(query_lower)
        state_filters = self._extract_state_filters(query_lower)

        # Enhance join hints with relationship patterns
        relationship_hints = self._extract_relationship_hints(query_lower)
        join_hints.extend(relationship_hints)

        return QueryIntent(
            entities=entities,
            join_hints=join_hints,
            time_filters=time_filters,
            aggregations=data.get("aggregations", []),
            limit=data.get("limit"),
            raw_query=query,
            columns_mentioned=data.get("columns_mentioned", []),
            state_filters=state_filters,
            extracted_schema=extracted_schema
        )

    def _extract_with_regex(self, query: str) -> QueryIntent:
        """Extract using regex patterns (fallback)."""
        query_lower = query.lower()

        # Extract schema reference FIRST (before filter detection)
        extracted_schema = self._extract_schema(query_lower)

        # Extract potential column names first (before lowercasing everything)
        columns_mentioned = self._extract_columns(query)

        # Extract filters FIRST (before entity detection)
        # Pass extracted_schema so it's not treated as a filter value
        filters = self._extract_filters(query, query_lower, exclude_values={extracted_schema} if extracted_schema else None)

        # Get filter column names to exclude from entity detection
        filter_columns = {f.column.lower() for f in filters}
        filter_values = {f.value.lower() for f in filters}

        # Also exclude schema name from entity detection
        exclude_from_entities = filter_columns | filter_values
        if extracted_schema:
            exclude_from_entities.add(extracted_schema)

        # Extract potential dataset names (excluding filter columns/values and schema)
        entities = self._extract_dataset_entities(query_lower, exclude_from_entities)

        # Extract join hints
        join_hints = self._extract_join_hints(query, query_lower)

        # Update join hints with column mentions
        join_hints = self._enhance_join_hints_with_columns(join_hints, columns_mentioned)

        # Extract time filters
        time_filters = self._extract_time_filters(query_lower)

        # Extract state filters (e.g., "not completed", "pending")
        state_filters = self._extract_state_filters(query_lower)

        # Extract relationship-based join hints (e.g., "users who have... assignment")
        relationship_hints = self._extract_relationship_hints(query_lower)
        join_hints.extend(relationship_hints)

        # Extract aggregations
        aggregations = list(set(re.findall(self.AGGREGATION_PATTERNS[0], query_lower)))

        # Extract limit
        limit = self._extract_limit(query_lower)

        return QueryIntent(
            entities=entities,
            join_hints=join_hints,
            time_filters=time_filters,
            aggregations=aggregations,
            limit=limit,
            raw_query=query,
            columns_mentioned=columns_mentioned,
            filters=filters,
            state_filters=state_filters,
            extracted_schema=extracted_schema
        )

    def _extract_schema(self, query_lower: str) -> Optional[str]:
        """Extract schema name from the query if mentioned."""
        for pattern in self.SCHEMA_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                potential_schema = match.group(1)
                # Only return if it's a known schema name
                if potential_schema in self.known_schemas:
                    return potential_schema

        # Also check for any known schema name mentioned directly
        # (handles cases like "assets in jeeves" where jeeves is at end)
        for schema in self.known_schemas:
            if f' {schema}' in query_lower or f' {schema} ' in query_lower:
                # Make sure it's not part of a table name or column
                if schema not in self.known_tables and schema not in self.known_columns:
                    return schema

        return None

    def _extract_columns(self, query: str) -> List[str]:
        """Extract column-like identifiers from query."""
        matches = self.COLUMN_PATTERN.findall(query)
        return list(set(matches))

    def _extract_dataset_entities(self, query_lower: str, exclude_words: set = None) -> List[ExtractedEntity]:
        """Extract potential dataset/table references."""
        exclude_words = exclude_words or set()

        # Common table name suffixes that indicate a compound table name
        TABLE_SUFFIXES = {'details', 'detail', 'info', 'data', 'metadata', 'history',
                         'log', 'logs', 'items', 'records', 'entries',
                         'attribute', 'attributes', 'settings', 'config', 'stats'}

        # Split into words
        words = re.findall(r'\b([a-z][a-z_]*[a-z])\b', query_lower)

        entities = []
        seen = set()
        used_indices = set()  # Track indices used in compound phrases

        # Skip compound extraction if query mentions "join" - user wants separate tables
        has_join = 'join' in query_lower

        # First, try to find compound phrases ONLY when:
        # 1. Second word is a table suffix
        # 2. Query doesn't mention "join" (which implies separate tables)
        if not has_join:
            for i in range(len(words) - 1):
                if i in used_indices:
                    continue
                word1 = words[i]
                word2 = words[i + 1]

                # Skip stop words
                if word1 in self.STOP_WORDS or word2 in self.STOP_WORDS:
                    continue

                # Only create compound if second word is a common table suffix
                if word2 not in TABLE_SUFFIXES:
                    continue

                # Create compound name (underscore-joined)
                compound = f"{word1}_{word2}"

                if compound not in seen and compound not in exclude_words:
                    seen.add(compound)
                    used_indices.add(i)
                    used_indices.add(i + 1)

                    entities.append(ExtractedEntity(
                        name=compound,
                        entity_type="dataset",
                        original_text=f"{word1} {word2}",
                        confidence=0.8,  # Higher confidence for compound phrases
                        modifiers=[]
                    ))

        # Then extract single words that weren't part of compounds
        for i, word in enumerate(words):
            if i in used_indices:
                continue
            if word in self.STOP_WORDS:
                continue
            if len(word) < 3:
                continue
            if word in seen:
                continue
            if word in exclude_words:
                continue
            # Skip common suffixes when standalone (they're not table names by themselves)
            if word in TABLE_SUFFIXES:
                continue
            # Skip if this looks like a column name (known or common)
            # BUT don't skip if it's also a known table name
            if word in self.known_tables:
                pass  # Always include known table names
            elif word in self.COMMON_COLUMN_NAMES or word in self.known_columns:
                continue

            seen.add(word)

            # Boost confidence for plural nouns (likely table names)
            confidence = 0.6 if word.endswith('s') else 0.5

            entities.append(ExtractedEntity(
                name=word,
                entity_type="dataset",
                original_text=word,
                confidence=confidence,
                modifiers=[]
            ))

        return entities

    def _extract_filters(self, query: str, query_lower: str, exclude_values: Optional[set] = None) -> List[ExtractedFilter]:
        """Extract WHERE clause filters from the query.

        Args:
            query: Original query string
            query_lower: Lowercased query string
            exclude_values: Optional set of values to exclude (e.g., schema names)
        """
        filters = []
        seen_columns = set()
        exclude_values = exclude_values or set()

        for pattern, pattern_type in self.FILTER_PATTERNS:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                groups = match.groups()
                column = None
                value = None

                if pattern_type in ('of_pattern', 'by_pattern'):
                    # Groups: (column, optional_quote, value)
                    column = groups[0]
                    value = groups[2] if len(groups) > 2 else groups[1]
                elif pattern_type in ('where_pattern', 'equals_pattern', 'colon_pattern'):
                    # Groups: (column, value)
                    column = groups[0]
                    value = groups[1]

                # Validate: column should look like a column name, not a table name
                if column and value:
                    column_lower = column.lower()
                    value_lower = value.lower()

                    # Skip if column is a stop word
                    if column_lower in self.STOP_WORDS:
                        continue
                    # Skip if value is a stop word (likely not a filter)
                    if value_lower in self.STOP_WORDS:
                        continue
                    # Skip if value is a schema name (not a filter value)
                    if value_lower in exclude_values or value_lower in self.known_schemas:
                        continue
                    # Accept if column is in known columns or common column names
                    is_likely_column = (
                        column_lower in self.known_columns or
                        column_lower in self.COMMON_COLUMN_NAMES or
                        self._looks_like_column(column)
                    )
                    if is_likely_column and column_lower not in seen_columns:
                        seen_columns.add(column_lower)
                        filters.append(ExtractedFilter(
                            column=column,
                            operator="=",
                            value=value,
                            original_text=match.group(0)
                        ))

        return filters

    def _extract_join_hints(self, query: str, query_lower: str) -> List[ExtractedJoinHint]:
        """Extract join relationship hints."""
        join_hints = []

        for pattern, hint_type in self.JOIN_PATTERNS:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if len(match) >= 2:
                    left = match[0]
                    right = match[1]

                    # Skip if either is a stop word
                    if left in self.STOP_WORDS or right in self.STOP_WORDS:
                        continue

                    key_hint = None
                    if hint_type == "join_key":
                        # The second match might be a column name
                        key_hint = right if self._looks_like_column(right) else None
                        if key_hint:
                            continue  # Don't add as join hint if it's a column

                    join_hints.append(ExtractedJoinHint(
                        left_entity=left,
                        right_entity=right,
                        join_key_hint=key_hint,
                        relationship="linked_to"
                    ))

        # Deduplicate
        seen = set()
        unique_hints = []
        for hint in join_hints:
            key = (hint.left_entity, hint.right_entity)
            if key not in seen:
                seen.add(key)
                unique_hints.append(hint)

        return unique_hints

    def _enhance_join_hints_with_columns(
        self,
        join_hints: List[ExtractedJoinHint],
        columns: List[str]
    ) -> List[ExtractedJoinHint]:
        """Add column hints to join hints if columns match entity names."""
        if not columns:
            return join_hints

        enhanced = []
        for hint in join_hints:
            # Look for columns that might be join keys
            for col in columns:
                col_lower = col.lower()
                # Check if column contains entity name (e.g., assetId contains "asset")
                if hint.left_entity in col_lower or hint.right_entity in col_lower:
                    hint = ExtractedJoinHint(
                        left_entity=hint.left_entity,
                        right_entity=hint.right_entity,
                        join_key_hint=col,
                        relationship=hint.relationship
                    )
                    break
            enhanced.append(hint)

        return enhanced

    def _looks_like_column(self, text: str) -> bool:
        """Check if text looks like a column name."""
        # camelCase or has underscore or ends with Id/ID
        return bool(re.match(r'^[a-z]+([A-Z][a-z]*)+$', text) or
                   '_' in text or
                   text.endswith('id') or
                   text.endswith('Id'))

    def _extract_time_filters(self, query_lower: str) -> List[ExtractedTimeFilter]:
        """Extract time-related filters."""
        time_filters = []

        for pattern, period_type in self.TIME_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                time_filters.append(ExtractedTimeFilter(
                    period=period_type,
                    column_hint=None,
                    start_date=None,
                    end_date=None
                ))
                break  # Only capture first time filter

        return time_filters

    def _extract_limit(self, query_lower: str) -> Optional[int]:
        """Extract limit/top N hints."""
        patterns = [
            r'\b(?:top|first|limit)\s+(\d+)\b',
            r'\b(\d+)\s+(?:rows?|records?|results?)\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))

        return None

    def _extract_state_filters(self, query_lower: str) -> List[ExtractedStateFilter]:
        """Extract state/status filters like 'not completed', 'pending', etc."""
        state_filters = []
        seen_states = set()

        for pattern, is_negated in self.STATE_FILTER_PATTERNS:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                state = match.group(1)

                # Normalize state name
                state_normalized = state.rstrip('ed')  # complete/completed -> complet
                if state_normalized in seen_states:
                    continue
                seen_states.add(state_normalized)

                # Determine if negated
                negated = is_negated
                if negated is None:
                    # Check if "not" appears before the state in the match
                    negated = 'not' in match.group(0)

                # Try to find target entity (what is completed/pending)
                target_entity = None
                # Look for "the X" after the state word
                target_match = re.search(
                    rf'{re.escape(match.group(0))}\s+(?:the\s+)?(\w+)',
                    query_lower
                )
                if target_match:
                    potential_target = target_match.group(1)
                    if potential_target not in self.STOP_WORDS:
                        target_entity = potential_target

                state_filters.append(ExtractedStateFilter(
                    state=state,
                    negated=negated,
                    original_text=match.group(0),
                    target_entity=target_entity
                ))

        return state_filters

    def _extract_relationship_hints(self, query_lower: str) -> List[ExtractedJoinHint]:
        """Extract join hints from relationship patterns like 'users who have... assignment'."""
        join_hints = []
        seen_pairs = set()

        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                groups = match.groups()

                if rel_type == 'has_action' and len(groups) >= 3:
                    # Pattern: (subject, action, object) e.g., (users, completed, assignment)
                    left = groups[0]
                    right = groups[2]
                elif rel_type in ('possessive', 'possessive_attr') and len(groups) >= 2:
                    # Pattern: (owner, owned) e.g., (users, assignment) or (assignment, due, date)
                    left = groups[0]
                    right = groups[1]
                else:
                    continue

                # Skip stop words
                if left in self.STOP_WORDS or right in self.STOP_WORDS:
                    continue

                # Avoid duplicates
                pair_key = (left, right)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                join_hints.append(ExtractedJoinHint(
                    left_entity=left,
                    right_entity=right,
                    join_key_hint=None,
                    relationship="has" if rel_type == 'has_action' else "belongs_to"
                ))

        return join_hints