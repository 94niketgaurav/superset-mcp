"""
Column analyzer for detecting patterns, types, and join candidates.

Analyzes:
- Column data types and patterns (UUID, integer ID, email, etc.)
- Sample values for overlap detection
- Foreign key relationships based on value matching
"""
import re
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class ColumnPattern:
    """Detected pattern in a column."""
    pattern_type: str
    confidence: float
    regex: Optional[str] = None
    description: Optional[str] = None


class ColumnAnalyzer:
    """
    Analyze columns to detect patterns and joinability.

    Detects patterns like:
    - UUID (various formats)
    - Integer IDs (sequential, random)
    - Timestamps/Dates
    - Email addresses
    - URLs
    - Foreign key references
    """

    # Pattern definitions
    PATTERNS = {
        "uuid": {
            "regex": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            "description": "UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"
        },
        "uuid_no_dashes": {
            "regex": r"^[0-9a-f]{32}$",
            "description": "UUID without dashes"
        },
        "integer_id": {
            "regex": r"^\d+$",
            "description": "Integer ID"
        },
        "email": {
            "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "description": "Email address"
        },
        "url": {
            "regex": r"^https?://",
            "description": "URL"
        },
        "iso_date": {
            "regex": r"^\d{4}-\d{2}-\d{2}$",
            "description": "ISO date (YYYY-MM-DD)"
        },
        "iso_datetime": {
            "regex": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",
            "description": "ISO datetime"
        },
        "phone": {
            "regex": r"^[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}$",
            "description": "Phone number"
        },
        "ip_address": {
            "regex": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
            "description": "IPv4 address"
        },
        "slug": {
            "regex": r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
            "description": "URL slug"
        },
        "code": {
            "regex": r"^[A-Z]{2,4}[-_]?\d{3,}$",
            "description": "Code format (e.g., SKU-123)"
        }
    }

    # Column name patterns that suggest ID columns
    ID_COLUMN_PATTERNS = [
        r"^id$",
        r"_id$",
        r"Id$",
        r"ID$",
        r"^uuid$",
        r"_uuid$",
        r"^guid$",
        r"_guid$",
        r"_pk$",
        r"^pk$",
        r"_key$",
        r"_ref$",
        r"_fk$",
    ]

    def __init__(self):
        """Initialize the column analyzer."""
        self._compiled_patterns = {
            name: re.compile(info["regex"], re.IGNORECASE)
            for name, info in self.PATTERNS.items()
        }
        self._id_patterns = [re.compile(p, re.IGNORECASE) for p in self.ID_COLUMN_PATTERNS]

    def detect_pattern(self, values: List[Any]) -> Optional[ColumnPattern]:
        """
        Detect the primary pattern in a list of values.

        Args:
            values: Sample values from the column

        Returns:
            ColumnPattern if detected, None otherwise
        """
        if not values:
            return None

        # Filter out None/null values
        str_values = [str(v) for v in values if v is not None and str(v).strip()]

        if not str_values:
            return None

        # Check each pattern
        pattern_matches: Dict[str, int] = {}

        for value in str_values:
            for pattern_name, regex in self._compiled_patterns.items():
                if regex.match(value):
                    pattern_matches[pattern_name] = pattern_matches.get(pattern_name, 0) + 1

        if not pattern_matches:
            return None

        # Find best matching pattern
        total_values = len(str_values)
        best_pattern = max(pattern_matches.items(), key=lambda x: x[1])
        pattern_name, match_count = best_pattern
        confidence = match_count / total_values

        if confidence >= 0.8:  # At least 80% match
            info = self.PATTERNS[pattern_name]
            return ColumnPattern(
                pattern_type=pattern_name,
                confidence=confidence,
                regex=info["regex"],
                description=info["description"]
            )

        return None

    def is_id_column(self, column_name: str, values: Optional[List[Any]] = None) -> bool:
        """
        Check if a column appears to be an ID column.

        Args:
            column_name: Name of the column
            values: Optional sample values for additional analysis

        Returns:
            True if column appears to be an ID column
        """
        # Check name patterns
        for pattern in self._id_patterns:
            if pattern.search(column_name):
                return True

        # Check values if provided
        if values:
            pattern = self.detect_pattern(values)
            if pattern and pattern.pattern_type in ("uuid", "uuid_no_dashes", "integer_id"):
                return True

        return False

    def compute_value_hash(self, values: List[Any], sample_size: int = 100) -> str:
        """
        Compute a hash of sample values for matching columns.

        This helps identify columns that might contain the same data
        across different tables (potential join candidates).

        Args:
            values: Values from the column
            sample_size: Number of values to use for hashing

        Returns:
            Hash string representing the value distribution
        """
        if not values:
            return ""

        # Sort and take a sample
        str_values = sorted([str(v) for v in values if v is not None][:sample_size])

        # Create hash
        value_str = "|".join(str_values)
        return hashlib.md5(value_str.encode()).hexdigest()[:16]

    def compute_value_overlap(
        self,
        values1: List[Any],
        values2: List[Any]
    ) -> float:
        """
        Compute the overlap ratio between two sets of values.

        This helps identify columns that contain related data.

        Args:
            values1: Values from first column
            values2: Values from second column

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not values1 or not values2:
            return 0.0

        set1 = set(str(v) for v in values1 if v is not None)
        set2 = set(str(v) for v in values2 if v is not None)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        smaller_set = min(len(set1), len(set2))

        return intersection / smaller_set if smaller_set > 0 else 0.0

    def analyze_column(
        self,
        column_name: str,
        column_type: str,
        values: List[Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a column.

        Args:
            column_name: Name of the column
            column_type: Database type of the column
            values: Sample values from the column

        Returns:
            Dict with analysis results
        """
        analysis = {
            "column_name": column_name,
            "column_type": column_type,
            "is_id_column": self.is_id_column(column_name, values),
            "pattern": None,
            "value_hash": None,
            "statistics": {}
        }

        if values:
            # Detect pattern
            pattern = self.detect_pattern(values)
            if pattern:
                analysis["pattern"] = {
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "description": pattern.description
                }

            # Compute value hash
            analysis["value_hash"] = self.compute_value_hash(values)

            # Basic statistics
            non_null = [v for v in values if v is not None]
            analysis["statistics"] = {
                "sample_size": len(values),
                "non_null_count": len(non_null),
                "null_count": len(values) - len(non_null),
                "distinct_count": len(set(str(v) for v in non_null))
            }

            # String-specific statistics
            if non_null and all(isinstance(v, str) for v in non_null[:10]):
                lengths = [len(str(v)) for v in non_null]
                analysis["statistics"]["avg_length"] = sum(lengths) / len(lengths)
                analysis["statistics"]["min_length"] = min(lengths)
                analysis["statistics"]["max_length"] = max(lengths)

        return analysis

    def find_join_candidates(
        self,
        source_col: str,
        source_values: List[Any],
        target_columns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find potential join candidates for a source column.

        Args:
            source_col: Name of the source column
            source_values: Values from the source column
            target_columns: List of target columns with their values
                           Each dict should have: column_name, table_name, values

        Returns:
            List of potential join candidates with confidence scores
        """
        candidates = []
        source_hash = self.compute_value_hash(source_values)

        for target in target_columns:
            target_name = target.get("column_name", "")
            target_table = target.get("table_name", "")
            target_values = target.get("values", [])

            if not target_values:
                continue

            # Compute overlap
            overlap = self.compute_value_overlap(source_values, target_values)

            if overlap > 0.1:  # At least 10% overlap
                target_hash = self.compute_value_hash(target_values)
                hash_match = source_hash == target_hash

                confidence = overlap
                if hash_match:
                    confidence = min(1.0, confidence + 0.2)

                candidates.append({
                    "table_name": target_table,
                    "column_name": target_name,
                    "overlap_ratio": overlap,
                    "hash_match": hash_match,
                    "confidence": confidence
                })

        # Sort by confidence
        candidates.sort(key=lambda x: x["confidence"], reverse=True)

        return candidates

    def infer_foreign_key(
        self,
        column_name: str,
        table_names: List[str]
    ) -> Optional[Dict[str, str]]:
        """
        Infer potential foreign key reference from column name.

        Args:
            column_name: Name of the column (e.g., "user_id", "assetId")
            table_names: List of known table names

        Returns:
            Dict with table and column if inference successful
        """
        # Normalize column name to extract entity
        from joins.normalizer import ColumnNormalizer

        entity, key_type = ColumnNormalizer.extract_entity_from_fk(column_name)

        if not entity:
            return None

        # Look for matching table
        entity_lower = entity.lower()
        entity_variants = [
            entity_lower,
            entity_lower + "s",  # Plural
            entity_lower.rstrip("s"),  # Singular
        ]

        for table_name in table_names:
            table_lower = table_name.lower()
            if table_lower in entity_variants:
                return {
                    "table": table_name,
                    "column": key_type or "id"
                }

        return None

    def detect_join_patterns(
        self,
        tables: Dict[str, Dict[str, List[str]]]
    ) -> List[Dict[str, Any]]:
        """
        Detect join patterns between tables based on column names and types.

        Args:
            tables: Dict of {table_name: {column_name: [sample_values]}}

        Returns:
            List of detected join patterns with confidence scores
        """
        join_patterns = []
        table_names = list(tables.keys())

        for table_name, columns in tables.items():
            for col_name, values in columns.items():
                # Check if this looks like a FK column
                fk_info = self.infer_foreign_key(col_name, table_names)

                if fk_info and fk_info["table"] != table_name:
                    # Found potential FK relationship
                    target_table = fk_info["table"]
                    target_col = fk_info["column"]

                    # Verify the target column exists
                    if target_table in tables:
                        target_cols = tables[target_table]
                        # Look for matching column
                        for potential_pk in [target_col, "id", f"{target_table}_id", f"{target_table}Id"]:
                            if potential_pk in target_cols:
                                # Check value overlap if we have samples
                                overlap = 0.0
                                if values and target_cols.get(potential_pk):
                                    overlap = self.compute_value_overlap(
                                        values, target_cols[potential_pk]
                                    )

                                confidence = 0.7  # Base confidence for name match
                                if overlap > 0.3:
                                    confidence = min(0.95, confidence + overlap * 0.3)

                                join_patterns.append({
                                    "left_table": table_name,
                                    "left_column": col_name,
                                    "right_table": target_table,
                                    "right_column": potential_pk,
                                    "join_type": "inner",
                                    "confidence": confidence,
                                    "detection_method": "fk_naming_convention",
                                    "value_overlap": overlap
                                })
                                break

        # Detect cross-table joins based on common column names
        common_id_patterns = ["user_id", "asset_id", "tenant", "organization_id", "created_by"]
        for pattern in common_id_patterns:
            tables_with_col = []
            for table_name, columns in tables.items():
                for col_name in columns.keys():
                    if col_name.lower() == pattern.lower() or col_name.lower().replace("_", "") == pattern.lower().replace("_", ""):
                        tables_with_col.append((table_name, col_name))

            # Create joins between tables with same column
            for i, (t1, c1) in enumerate(tables_with_col):
                for t2, c2 in tables_with_col[i+1:]:
                    if t1 != t2:
                        join_patterns.append({
                            "left_table": t1,
                            "left_column": c1,
                            "right_table": t2,
                            "right_column": c2,
                            "join_type": "inner",
                            "confidence": 0.65,
                            "detection_method": "common_column_name"
                        })

        # Remove duplicates and sort by confidence
        seen = set()
        unique_patterns = []
        for jp in join_patterns:
            key = tuple(sorted([
                f"{jp['left_table']}.{jp['left_column']}",
                f"{jp['right_table']}.{jp['right_column']}"
            ]))
            if key not in seen:
                seen.add(key)
                unique_patterns.append(jp)

        unique_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        return unique_patterns

    def analyze_table_columns(
        self,
        table_name: str,
        columns: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze all columns of a table.

        Args:
            table_name: Name of the table
            columns: Dict of {column_name: {type, values, ...}}

        Returns:
            Analysis results for the table
        """
        result = {
            "table_name": table_name,
            "column_count": len(columns),
            "primary_key_candidates": [],
            "foreign_key_candidates": [],
            "columns": {}
        }

        for col_name, col_info in columns.items():
            col_type = col_info.get("type", "string")
            values = col_info.get("values", [])

            analysis = self.analyze_column(col_name, col_type, values)
            result["columns"][col_name] = analysis

            # Check for PK candidate
            if analysis["is_id_column"]:
                if col_name.lower() in ["id", f"{table_name}_id", f"{table_name}id"]:
                    result["primary_key_candidates"].append({
                        "column": col_name,
                        "confidence": 0.95
                    })
                elif "_id" in col_name.lower() or "id" in col_name.lower():
                    # Might be FK
                    result["foreign_key_candidates"].append({
                        "column": col_name,
                        "pattern": analysis.get("pattern")
                    })

        return result