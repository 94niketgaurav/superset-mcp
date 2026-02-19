"""
Query Analyzer - Parse SQL and extract patterns using sqlglot.

Features:
- Dialect-aware SQL parsing (StarRocks, PostgreSQL, MySQL)
- Extract tables, joins, filters, aggregations
- Generate intent signatures for similarity matching
- Normalize SQL for deduplication
- Validate queries against current schema
"""
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple

try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    sqlglot = None
    exp = None

logger = logging.getLogger(__name__)


@dataclass
class JoinInfo:
    """Information about a JOIN in a query."""
    left_table: str
    right_table: str
    join_type: str          # "inner", "left", "right", "cross"
    on_clause: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FilterInfo:
    """Information about a WHERE filter."""
    column: str
    operator: str           # "=", "!=", ">", "<", "like", "in", etc.
    value_type: str         # "literal", "column", "subquery"
    is_time_filter: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class QueryPattern:
    """Extracted pattern from a SQL query."""
    query_type: str               # "simple", "join", "aggregation", "subquery", "cte"
    tables: List[str]             # Tables referenced
    joins: List[JoinInfo]         # Join relationships
    filters: List[FilterInfo]     # WHERE conditions
    aggregations: List[str]       # COUNT, SUM, AVG, etc.
    group_by: List[str]           # GROUP BY columns
    order_by: List[str]           # ORDER BY columns
    has_time_filter: bool         # Has date/time filtering
    has_limit: bool               # Has LIMIT clause
    limit_value: Optional[int] = None
    complexity_score: float = 0.0 # 0-1 (simple to complex)
    intent_signature: str = ""    # Hash for similarity matching

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "joins": [j.to_dict() for j in self.joins],
            "filters": [f.to_dict() for f in self.filters],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QueryPattern":
        data["joins"] = [JoinInfo(**j) for j in data.get("joins", [])]
        data["filters"] = [FilterInfo(**f) for f in data.get("filters", [])]
        return cls(**data)


@dataclass
class QueryExample:
    """A complete query example for few-shot learning."""
    title: str                    # Natural language title
    description: str              # What the query does
    sql: str                      # The SQL
    normalized_sql: str           # Literals replaced with placeholders
    pattern: QueryPattern         # Analyzed pattern
    keywords: List[str]           # Keywords for matching
    schema: Optional[str] = None  # Schema context
    dialect: str = "starrocks"    # SQL dialect
    source: str = "saved_query"   # Where it came from
    source_id: Optional[int] = None
    is_valid: bool = True         # Still valid against current schema
    last_validated: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "pattern": self.pattern.to_dict() if self.pattern else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QueryExample":
        pattern_data = data.pop("pattern", None)
        pattern = QueryPattern.from_dict(pattern_data) if pattern_data else None
        return cls(**data, pattern=pattern)


class QueryAnalyzer:
    """
    Analyzes SQL queries to extract patterns and metadata.

    Uses sqlglot for dialect-aware parsing.
    """

    # Date/time column patterns
    TIME_COLUMN_PATTERNS = [
        r'.*date.*', r'.*time.*', r'.*created.*', r'.*updated.*',
        r'.*_at$', r'.*_on$', r'.*timestamp.*', r'.*modified.*'
    ]

    # Keywords that suggest aggregation
    AGGREGATION_KEYWORDS = ['count', 'sum', 'avg', 'average', 'total', 'min', 'max', 'group']

    # Keywords that suggest time filtering
    TIME_KEYWORDS = ['today', 'yesterday', 'week', 'month', 'year', 'last', 'recent', 'past']

    def __init__(self, dialect: str = "starrocks", semantic_config: Optional[Dict] = None):
        """
        Initialize the analyzer.

        Args:
            dialect: SQL dialect for parsing (starrocks, postgres, mysql)
            semantic_config: Optional semantic config for schema validation
        """
        self.dialect = dialect
        self.semantic_config = semantic_config or {}
        self._known_tables: Set[str] = set()
        self._known_columns: Dict[str, Set[str]] = {}  # table -> columns
        self._load_schema_from_config()

    def _load_schema_from_config(self):
        """Load known tables and columns from semantic config."""
        schemas = self.semantic_config.get("schemas", {})
        for schema_name, schema_data in schemas.items():
            tables = schema_data.get("tables", {})
            for table_name, table_data in tables.items():
                self._known_tables.add(table_name.lower())
                columns = table_data.get("columns", {})
                if isinstance(columns, dict):
                    self._known_columns[table_name.lower()] = set(
                        col.lower() for col in columns.keys()
                    )

    def analyze(self, sql: str, title: str = "", description: str = "") -> Optional[QueryExample]:
        """
        Analyze a SQL query and extract its pattern.

        Args:
            sql: The SQL query to analyze
            title: Human-readable title
            description: Description of what the query does

        Returns:
            QueryExample with extracted pattern, or None if parsing fails
        """
        if not SQLGLOT_AVAILABLE:
            logger.warning("sqlglot not available, using basic analysis")
            return self._analyze_basic(sql, title, description)

        try:
            pattern = self._parse_sql(sql)
            normalized = self._normalize_sql(sql)
            keywords = self._extract_keywords(title, description, pattern)

            # Generate intent signature
            pattern.intent_signature = self._generate_signature(pattern)

            # Calculate complexity
            pattern.complexity_score = self._calculate_complexity(pattern)

            return QueryExample(
                title=title,
                description=description,
                sql=sql,
                normalized_sql=normalized,
                pattern=pattern,
                keywords=keywords,
                dialect=self.dialect,
                is_valid=True,
            )

        except Exception as e:
            logger.warning(f"Failed to analyze SQL: {e}")
            return self._analyze_basic(sql, title, description)

    def _parse_sql(self, sql: str) -> QueryPattern:
        """Parse SQL using sqlglot and extract pattern."""
        parsed = sqlglot.parse_one(sql, dialect=self.dialect)

        # Extract tables
        tables = []
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            if table_name:
                tables.append(table_name)

        # Extract joins
        joins = []
        for join in parsed.find_all(exp.Join):
            join_info = JoinInfo(
                left_table="",  # Parent table
                right_table=join.this.name if hasattr(join.this, 'name') else str(join.this),
                join_type=str(join.kind).lower() if join.kind else "inner",
                on_clause=str(join.args.get("on")) if join.args.get("on") else None,
            )
            joins.append(join_info)

        # Extract aggregations
        aggregations = []
        for agg in parsed.find_all(exp.AggFunc):
            agg_name = type(agg).__name__.upper()
            aggregations.append(agg_name)

        # Extract GROUP BY
        group_by = []
        for group in parsed.find_all(exp.Group):
            for expr in group.expressions:
                if hasattr(expr, 'name'):
                    group_by.append(expr.name)
                else:
                    group_by.append(str(expr))

        # Extract ORDER BY
        order_by = []
        for order in parsed.find_all(exp.Order):
            for expr in order.expressions:
                if hasattr(expr, 'this') and hasattr(expr.this, 'name'):
                    order_by.append(expr.this.name)

        # Extract filters
        filters = []
        has_time_filter = False
        for where in parsed.find_all(exp.Where):
            filter_info, is_time = self._extract_filter_info(where)
            filters.extend(filter_info)
            if is_time:
                has_time_filter = True

        # Check for LIMIT
        has_limit = False
        limit_value = None
        for limit in parsed.find_all(exp.Limit):
            has_limit = True
            if hasattr(limit.this, 'this'):
                try:
                    limit_value = int(limit.this.this)
                except (ValueError, TypeError):
                    pass

        # Determine query type
        query_type = self._determine_query_type(
            tables, joins, aggregations, group_by, parsed
        )

        return QueryPattern(
            query_type=query_type,
            tables=tables,
            joins=joins,
            filters=filters,
            aggregations=list(set(aggregations)),
            group_by=group_by,
            order_by=order_by,
            has_time_filter=has_time_filter,
            has_limit=has_limit,
            limit_value=limit_value,
        )

    def _extract_filter_info(self, where_expr) -> Tuple[List[FilterInfo], bool]:
        """Extract filter information from WHERE clause."""
        filters = []
        has_time_filter = False

        def process_expr(expr):
            nonlocal has_time_filter

            if isinstance(expr, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                left = expr.left
                right = expr.right

                col_name = ""
                if hasattr(left, 'name'):
                    col_name = left.name
                elif hasattr(left, 'this') and hasattr(left.this, 'name'):
                    col_name = left.this.name

                # Determine operator
                op_map = {
                    exp.EQ: "=",
                    exp.NEQ: "!=",
                    exp.GT: ">",
                    exp.GTE: ">=",
                    exp.LT: "<",
                    exp.LTE: "<=",
                }
                operator = op_map.get(type(expr), "=")

                # Determine value type
                value_type = "literal"
                if isinstance(right, exp.Column):
                    value_type = "column"
                elif isinstance(right, exp.Subquery):
                    value_type = "subquery"

                # Check if time filter
                is_time = any(
                    re.match(pattern, col_name.lower())
                    for pattern in self.TIME_COLUMN_PATTERNS
                )
                if is_time:
                    has_time_filter = True

                filters.append(FilterInfo(
                    column=col_name,
                    operator=operator,
                    value_type=value_type,
                    is_time_filter=is_time,
                ))

            elif isinstance(expr, exp.Like):
                col_name = expr.this.name if hasattr(expr.this, 'name') else ""
                filters.append(FilterInfo(
                    column=col_name,
                    operator="like",
                    value_type="literal",
                    is_time_filter=False,
                ))

            elif isinstance(expr, exp.In):
                col_name = expr.this.name if hasattr(expr.this, 'name') else ""
                filters.append(FilterInfo(
                    column=col_name,
                    operator="in",
                    value_type="literal",
                    is_time_filter=False,
                ))

            elif isinstance(expr, (exp.And, exp.Or)):
                for child in expr.args.values():
                    if isinstance(child, list):
                        for c in child:
                            process_expr(c)
                    else:
                        process_expr(child)

        process_expr(where_expr.this)
        return filters, has_time_filter

    def _determine_query_type(
        self,
        tables: List[str],
        joins: List[JoinInfo],
        aggregations: List[str],
        group_by: List[str],
        parsed: Any
    ) -> str:
        """Determine the type of query."""
        # Check for CTE
        if list(parsed.find_all(exp.CTE)):
            return "cte"

        # Check for subquery
        subqueries = list(parsed.find_all(exp.Subquery))
        if subqueries:
            return "subquery"

        # Check for joins
        if joins or len(tables) > 1:
            if aggregations or group_by:
                return "join_aggregation"
            return "join"

        # Check for aggregation
        if aggregations or group_by:
            return "aggregation"

        return "simple"

    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL by replacing literals with placeholders.

        This helps with deduplication - queries that differ only by
        literal values will have the same normalized form.
        """
        if not SQLGLOT_AVAILABLE:
            # Basic regex-based normalization
            normalized = re.sub(r"'[^']*'", "?", sql)
            normalized = re.sub(r"\b\d+\b", "?", normalized)
            return normalized

        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)

            # Replace all literals with placeholders
            for literal in parsed.find_all(exp.Literal):
                literal.replace(exp.Placeholder())

            return parsed.sql(dialect=self.dialect)
        except Exception:
            # Fallback to regex
            normalized = re.sub(r"'[^']*'", "?", sql)
            normalized = re.sub(r"\b\d+\b", "?", normalized)
            return normalized

    def _extract_keywords(
        self,
        title: str,
        description: str,
        pattern: QueryPattern
    ) -> List[str]:
        """
        Extract keywords for similarity matching.

        Keywords come from:
        - Title words (excluding stop words)
        - Table names
        - Aggregation types
        - Query type indicators
        """
        keywords = set()

        # Extract from title
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'for', 'by', 'to', 'in', 'of',
            'with', 'show', 'get', 'list', 'find', 'all', 'from', 'query'
        }
        title_words = re.findall(r'\b[a-z]+\b', title.lower())
        for word in title_words:
            if word not in stop_words and len(word) > 2:
                keywords.add(word)

        # Add table names
        for table in pattern.tables:
            keywords.add(table.lower())
            # Also add singular/plural variants
            if table.endswith('s'):
                keywords.add(table[:-1].lower())
            else:
                keywords.add(f"{table}s".lower())

        # Add aggregation indicators
        for agg in pattern.aggregations:
            keywords.add(agg.lower())

        # Add query type
        keywords.add(pattern.query_type)

        # Add time-related keywords if time filter present
        if pattern.has_time_filter:
            keywords.add("time")
            keywords.add("date")

        return list(keywords)

    def _generate_signature(self, pattern: QueryPattern) -> str:
        """
        Generate a signature hash for the query pattern.

        Queries with similar signatures have similar structure.
        """
        sig_parts = [
            pattern.query_type,
            ",".join(sorted(pattern.tables)),
            ",".join(sorted(pattern.aggregations)),
            str(len(pattern.joins)),
            str(pattern.has_time_filter),
            str(len(pattern.group_by) > 0),
        ]
        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()[:12]

    def _calculate_complexity(self, pattern: QueryPattern) -> float:
        """
        Calculate a complexity score for the query.

        Score ranges from 0 (simple) to 1 (complex).
        """
        score = 0.0

        # Base score by query type
        type_scores = {
            "simple": 0.1,
            "aggregation": 0.3,
            "join": 0.4,
            "join_aggregation": 0.6,
            "subquery": 0.7,
            "cte": 0.8,
        }
        score = type_scores.get(pattern.query_type, 0.2)

        # Add for joins
        score += min(len(pattern.joins) * 0.1, 0.3)

        # Add for filters
        score += min(len(pattern.filters) * 0.05, 0.2)

        # Add for aggregations
        score += min(len(pattern.aggregations) * 0.05, 0.15)

        # Add for GROUP BY
        if pattern.group_by:
            score += 0.1

        return min(score, 1.0)

    def _analyze_basic(
        self,
        sql: str,
        title: str,
        description: str
    ) -> Optional[QueryExample]:
        """
        Basic analysis without sqlglot (fallback).

        Uses regex patterns to extract basic information.
        """
        sql_upper = sql.upper()
        sql_lower = sql.lower()

        # Extract tables using regex
        table_pattern = r'FROM\s+([`"\']?)(\w+)\1'
        join_pattern = r'JOIN\s+([`"\']?)(\w+)\1'

        tables = []
        for match in re.finditer(table_pattern, sql, re.IGNORECASE):
            tables.append(match.group(2))
        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            tables.append(match.group(2))

        # Check for aggregations
        aggregations = []
        for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']:
            if f'{agg}(' in sql_upper:
                aggregations.append(agg)

        # Check for GROUP BY
        has_group_by = 'GROUP BY' in sql_upper

        # Check for time filter
        has_time_filter = any(
            kw in sql_lower for kw in
            ['created_at', 'updated_at', 'date', 'timestamp', 'current_date', 'now()']
        )

        # Determine query type
        has_join = 'JOIN' in sql_upper
        query_type = "simple"
        if 'WITH' in sql_upper:
            query_type = "cte"
        elif has_join and aggregations:
            query_type = "join_aggregation"
        elif has_join:
            query_type = "join"
        elif aggregations:
            query_type = "aggregation"

        pattern = QueryPattern(
            query_type=query_type,
            tables=tables,
            joins=[],
            filters=[],
            aggregations=aggregations,
            group_by=[],
            order_by=[],
            has_time_filter=has_time_filter,
            has_limit='LIMIT' in sql_upper,
        )

        pattern.intent_signature = self._generate_signature(pattern)
        pattern.complexity_score = self._calculate_complexity(pattern)

        keywords = self._extract_keywords(title, description, pattern)

        # Basic normalization
        normalized = re.sub(r"'[^']*'", "?", sql)
        normalized = re.sub(r"\b\d+\b", "?", normalized)

        return QueryExample(
            title=title,
            description=description,
            sql=sql,
            normalized_sql=normalized,
            pattern=pattern,
            keywords=keywords,
            dialect=self.dialect,
        )

    def validate_against_schema(self, example: QueryExample) -> Tuple[bool, List[str]]:
        """
        Validate a query example against the current schema.

        Checks if all tables and columns still exist.

        Args:
            example: QueryExample to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if not self._known_tables:
            # No schema loaded, assume valid
            return True, []

        # Check tables
        for table in example.pattern.tables:
            if table.lower() not in self._known_tables:
                issues.append(f"Table '{table}' not found in current schema")

        # Check columns in filters
        for f in example.pattern.filters:
            if f.column:
                # Try to find which table the column belongs to
                found = False
                for table in example.pattern.tables:
                    table_cols = self._known_columns.get(table.lower(), set())
                    if f.column.lower() in table_cols:
                        found = True
                        break
                # Don't flag as issue if we can't verify
                # (column might be from a table we know about)

        return len(issues) == 0, issues

    def refresh_schema(self, semantic_config: Dict):
        """
        Refresh the schema knowledge from semantic config.

        Call this when the schema has been updated.
        """
        self.semantic_config = semantic_config
        self._known_tables.clear()
        self._known_columns.clear()
        self._load_schema_from_config()
