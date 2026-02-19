"""
SQL validation before execution.

Validates:
1. Basic syntax (balanced parentheses, quotes)
2. Dangerous operations (if READ_ONLY)
3. Column references (warnings for missing columns)
4. Best practices (SELECT *, missing LIMIT)
5. Dialect-specific syntax (StarRocks vs PostgreSQL)
"""
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


# Dialect-specific syntax patterns
DIALECT_PATTERNS = {
    "starrocks": {
        # StarRocks uses: INTERVAL 7 DAY (no quotes around number+unit)
        "interval_correct": r"INTERVAL\s+\d+\s+(DAY|WEEK|MONTH|YEAR|HOUR|MINUTE|SECOND)S?",
        # PostgreSQL style intervals are WRONG for StarRocks
        "interval_wrong": [
            (r"INTERVAL\s+'[^']+'\s*$", "StarRocks uses INTERVAL N UNIT (e.g., INTERVAL 7 DAY), not INTERVAL 'N units'"),
            (r"INTERVAL\s+'\d+\s+(day|week|month|year|hour|minute|second)s?'", "StarRocks uses INTERVAL N UNIT (e.g., INTERVAL 7 DAY), not INTERVAL '7 days'"),
            (r"NOW\(\)\s*-\s*INTERVAL", "StarRocks prefers DATE_SUB(NOW(), INTERVAL N UNIT) over NOW() - INTERVAL"),
            (r"CURRENT_DATE\s*-\s*INTERVAL", "StarRocks prefers DATE_SUB(CURDATE(), INTERVAL N UNIT) over CURRENT_DATE - INTERVAL"),
        ],
        "date_functions": ["NOW()", "CURDATE()", "DATE_SUB", "DATE_ADD", "DATEDIFF", "DATE_FORMAT"],
    },
    "postgresql": {
        # PostgreSQL uses: INTERVAL '7 days' (quoted)
        "interval_correct": r"INTERVAL\s+'[^']+'",
        # StarRocks style intervals are WRONG for PostgreSQL
        "interval_wrong": [
            (r"INTERVAL\s+\d+\s+DAY(?!S)", "PostgreSQL uses INTERVAL 'N days', not INTERVAL N DAY"),
            (r"DATE_SUB\s*\(", "PostgreSQL uses date - INTERVAL '...', not DATE_SUB()"),
            (r"CURDATE\s*\(\)", "PostgreSQL uses CURRENT_DATE, not CURDATE()"),
        ],
        "date_functions": ["NOW()", "CURRENT_DATE", "CURRENT_TIMESTAMP", "DATE_TRUNC", "AGE"],
    }
}


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def can_execute(self) -> bool:
        """Check if SQL can be executed (no blocking errors)."""
        return self.is_valid

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "can_execute": self.can_execute
        }


class SQLValidator:
    """
    Validate SQL queries before execution.

    Checks:
    - Basic syntax (balanced parens, quotes)
    - Dangerous operations blocked in READ_ONLY mode
    - Column references against known schemas
    - SQL best practices
    """

    # Operations that modify data
    DANGEROUS_KEYWORDS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "GRANT", "REVOKE", "MERGE", "UPSERT", "REPLACE"
    ]

    # Keywords that should start a valid query
    VALID_QUERY_STARTS = ["SELECT", "WITH", "--", "/*", "EXPLAIN"]

    def __init__(
        self,
        datasets: Optional[List[Dict]] = None,
        read_only: bool = True,
        dialect: str = "starrocks"
    ):
        """
        Initialize validator.

        Args:
            datasets: List of dataset dicts with columns for reference checking
            read_only: If True, block write operations
            dialect: Database dialect for syntax validation ("starrocks" or "postgresql")
        """
        self.datasets = datasets or []
        self.read_only = read_only
        self.dialect = dialect.lower() if dialect else "starrocks"
        self._build_column_index()

    def _build_column_index(self):
        """Build index of all available columns."""
        self.columns_by_table: Dict[str, Set[str]] = {}
        self.all_columns: Set[str] = set()
        self.all_tables: Set[str] = set()

        for ds in self.datasets:
            table_name = ds.get("table_name", "").lower()
            self.all_tables.add(table_name)

            columns = {
                c.get("column_name", "").lower()
                for c in ds.get("columns", [])
                if c.get("column_name")
            }
            self.columns_by_table[table_name] = columns
            self.all_columns.update(columns)

    def _check_syntax(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Basic syntax validation.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Check for empty query
        sql_stripped = sql.strip()
        if not sql_stripped:
            errors.append("SQL query is empty")
            return False, errors

        # Check for valid query start
        sql_upper = sql_stripped.upper()
        valid_start = any(
            sql_upper.startswith(start)
            for start in self.VALID_QUERY_STARTS
        )
        if not valid_start:
            errors.append(
                f"Query should start with SELECT or WITH, "
                f"found: {sql_upper[:30]}..."
            )

        # Check balanced parentheses
        paren_count = sql.count('(') - sql.count(')')
        if paren_count != 0:
            errors.append(
                f"Unbalanced parentheses: "
                f"{'missing' if paren_count > 0 else 'extra'} "
                f"{abs(paren_count)} closing paren(s)"
            )

        # Check for unclosed quotes (basic check)
        # Remove escaped quotes first
        sql_no_escaped = sql.replace("\\'", "").replace('\\"', "")

        single_quotes = len(re.findall(r"'", sql_no_escaped))
        if single_quotes % 2 != 0:
            errors.append("Unclosed single quote detected")

        double_quotes = len(re.findall(r'"', sql_no_escaped))
        if double_quotes % 2 != 0:
            errors.append("Unclosed double quote detected")

        # Check for common typos
        common_typos = [
            (r'\bFROM\s+FROM\b', "Duplicate FROM keyword"),
            (r'\bWHERE\s+WHERE\b', "Duplicate WHERE keyword"),
            (r'\bAND\s+AND\b', "Duplicate AND keyword"),
            (r'\bOR\s+OR\b', "Duplicate OR keyword"),
            (r'\bSELECT\s+SELECT\b', "Duplicate SELECT keyword"),
        ]
        for pattern, message in common_typos:
            if re.search(pattern, sql_upper):
                errors.append(message)

        return len(errors) == 0, errors

    def _check_dangerous_operations(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Check for dangerous/write operations.

        Returns:
            Tuple of (is_safe, list of errors)
        """
        if not self.read_only:
            return True, []

        errors = []
        sql_upper = sql.upper()

        for keyword in self.DANGEROUS_KEYWORDS:
            # Use word boundary to avoid matching "SELECTABLE" etc.
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, sql_upper):
                errors.append(
                    f"Dangerous operation '{keyword}' not allowed in READ_ONLY mode"
                )

        return len(errors) == 0, errors

    def _check_column_references(
        self,
        sql: str,
        expected_tables: Optional[List[str]] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check if referenced columns exist.

        This is a heuristic check - full SQL parsing would be more accurate.

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        if not self.columns_by_table:
            return True, errors, warnings

        # Extract table.column references
        qualified_refs = re.findall(r'\b(\w+)\.(\w+)\b', sql)

        # Track aliases from FROM/JOIN clauses
        alias_pattern = r'\b(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)\b'
        aliases: Dict[str, str] = {}
        for match in re.finditer(alias_pattern, sql, re.IGNORECASE):
            table = match.group(1).lower()
            alias = match.group(2).lower()
            aliases[alias] = table

        for table_ref, col_ref in qualified_refs:
            table_lower = table_ref.lower()
            col_lower = col_ref.lower()

            # Skip if it's a known function or keyword
            sql_functions = {
                'date', 'time', 'timestamp', 'cast', 'extract', 'count',
                'sum', 'avg', 'min', 'max', 'coalesce', 'nullif', 'case',
                'when', 'then', 'else', 'end', 'over', 'partition', 'row_number'
            }
            if table_lower in sql_functions:
                continue

            # Resolve alias to actual table
            actual_table = aliases.get(table_lower, table_lower)

            # Check if table is in our known tables
            if actual_table in self.columns_by_table:
                if col_lower not in self.columns_by_table[actual_table]:
                    warnings.append(
                        f"Column '{col_ref}' not found in table '{actual_table}'"
                    )

        return True, errors, warnings

    def _check_select_star(self, sql: str) -> List[str]:
        """Warn about SELECT *."""
        warnings = []
        if re.search(r'\bSELECT\s+\*', sql, re.IGNORECASE):
            warnings.append(
                "SELECT * is used - consider selecting specific columns "
                "for better performance and clarity"
            )
        return warnings

    def _check_limit(self, sql: str) -> List[str]:
        """Warn if no LIMIT clause."""
        warnings = []
        sql_upper = sql.upper()

        # Don't warn for CTEs, subqueries that might have LIMIT elsewhere
        # or aggregations that return single rows
        has_limit = bool(re.search(r'\bLIMIT\s+\d+', sql_upper))
        has_top = bool(re.search(r'\bTOP\s+\d+', sql_upper))
        has_fetch = bool(re.search(r'\bFETCH\s+(?:FIRST|NEXT)\s+\d+', sql_upper))

        if not (has_limit or has_top or has_fetch):
            # Check if it's an aggregation-only query
            is_aggregation_only = (
                bool(re.search(r'\bGROUP\s+BY\b', sql_upper)) or
                (bool(re.search(r'\bCOUNT\s*\(', sql_upper)) and
                 not bool(re.search(r'\bFROM\b.*\bFROM\b', sql_upper)))
            )

            if not is_aggregation_only:
                warnings.append(
                    "No LIMIT clause found - consider adding one to prevent "
                    "large result sets"
                )

        return warnings

    def _check_dialect_syntax(self, sql: str) -> Tuple[List[str], List[str]]:
        """
        Check for dialect-specific syntax issues.

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        dialect_rules = DIALECT_PATTERNS.get(self.dialect, {})
        wrong_patterns = dialect_rules.get("interval_wrong", [])

        sql_upper = sql.upper()

        for pattern, message in wrong_patterns:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                errors.append(f"Dialect error ({self.dialect}): {message}")

        return errors, warnings

    def _check_join_conditions(self, sql: str) -> List[str]:
        """Check for potential join issues."""
        suggestions = []
        sql_upper = sql.upper()

        # Check for JOIN without ON
        joins_without_on = re.findall(
            r'\b(LEFT|RIGHT|INNER|OUTER|FULL|CROSS)?\s*JOIN\s+\w+\s+(?!ON\b)',
            sql_upper
        )
        if joins_without_on:
            # Could be USING clause or CROSS JOIN
            if 'USING' not in sql_upper and 'CROSS JOIN' not in sql_upper:
                suggestions.append(
                    "JOIN without ON clause detected - ensure join conditions are specified"
                )

        # Check for CROSS JOIN (intentional cartesian product)
        if 'CROSS JOIN' in sql_upper:
            suggestions.append(
                "CROSS JOIN detected - this creates a cartesian product. "
                "Ensure this is intentional."
            )

        return suggestions

    def _check_ambiguous_columns(self, sql: str) -> List[str]:
        """Check for potentially ambiguous column references."""
        warnings = []

        if not self.columns_by_table or len(self.columns_by_table) < 2:
            return warnings

        # Find unqualified column references in SELECT
        select_match = re.search(
            r'\bSELECT\s+(.*?)\s+FROM\b',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        if not select_match:
            return warnings

        select_clause = select_match.group(1)

        # Extract unqualified column names
        # This is a simplified check - doesn't handle all SQL syntax
        words = re.findall(r'\b([a-zA-Z_]\w*)\b', select_clause)

        # Skip keywords and functions
        skip_words = {
            'as', 'distinct', 'case', 'when', 'then', 'else', 'end',
            'count', 'sum', 'avg', 'min', 'max', 'coalesce', 'cast',
            'and', 'or', 'not', 'null', 'true', 'false', 'is', 'in',
            'like', 'between', 'over', 'partition', 'order', 'by'
        }

        for word in words:
            word_lower = word.lower()
            if word_lower in skip_words:
                continue

            # Check if this column exists in multiple tables
            tables_with_column = [
                table for table, cols in self.columns_by_table.items()
                if word_lower in cols
            ]

            if len(tables_with_column) > 1:
                warnings.append(
                    f"Column '{word}' exists in multiple tables: "
                    f"{', '.join(tables_with_column)}. "
                    f"Consider qualifying it (e.g., table.{word})"
                )

        return warnings

    def validate(
        self,
        sql: str,
        expected_tables: Optional[List[str]] = None,
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate a SQL query.

        Args:
            sql: The SQL query to validate
            expected_tables: List of table names expected to be used
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []

        # Syntax check
        syntax_ok, syntax_errors = self._check_syntax(sql)
        errors.extend(syntax_errors)

        # Dangerous operations check
        safe_ok, danger_errors = self._check_dangerous_operations(sql)
        errors.extend(danger_errors)

        # Dialect-specific syntax check
        dialect_errors, dialect_warnings = self._check_dialect_syntax(sql)
        errors.extend(dialect_errors)
        warnings.extend(dialect_warnings)

        # Column reference check
        if self.columns_by_table:
            col_ok, col_errors, col_warnings = self._check_column_references(
                sql, expected_tables
            )
            errors.extend(col_errors)
            warnings.extend(col_warnings)

        # Best practice warnings
        warnings.extend(self._check_select_star(sql))
        warnings.extend(self._check_limit(sql))
        warnings.extend(self._check_ambiguous_columns(sql))

        # Suggestions
        suggestions.extend(self._check_join_conditions(sql))

        # In strict mode, warnings become errors
        if strict:
            errors.extend(warnings)
            warnings = []

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def quick_validate(self, sql: str) -> bool:
        """
        Quick validation - just checks for blocking issues.

        Args:
            sql: SQL to validate

        Returns:
            True if SQL can be executed
        """
        result = self.validate(sql)
        return result.is_valid