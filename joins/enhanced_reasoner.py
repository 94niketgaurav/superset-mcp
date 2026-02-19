"""
Enhanced join reasoning with multi-strategy detection.

Strategies (in priority order):
1. YAML relationships (confidence: 1.0)
2. Normalized column matches (confidence: 0.75-0.85)
3. FK → PK pattern detection (confidence: 0.65-0.80)
4. Type compatibility boost
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from .normalizer import ColumnNormalizer


@dataclass
class JoinColumn:
    """Represents a column with join-relevant metadata."""
    dataset_id: int
    table_name: str
    column_name: str
    column_type: str
    normalized_name: str
    is_primary_key: bool
    fk_entity: Optional[str]        # If FK, what entity does it reference
    fk_key_type: Optional[str]      # Type of key (id, uuid, etc.)


@dataclass
class JoinSuggestion:
    """A suggested join between two datasets."""
    left_dataset_id: int
    left_table: str
    left_column: str
    right_dataset_id: int
    right_table: str
    right_column: str
    join_type: str                  # "inner", "left", "right"
    confidence: float               # 0.0 - 1.0
    reason: str
    type_compatible: bool


class EnhancedJoinReasoner:
    """
    Advanced join detection using multiple strategies.

    Handles:
    - camelCase/snake_case normalization
    - FK → PK pattern detection (assetId → assets.id)
    - Table name correlation
    - Type compatibility scoring
    - Pre-defined relationships from YAML
    """

    # Type compatibility matrix - types that can be joined
    TYPE_GROUPS: Dict[str, List[str]] = {
        "integer": ["INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "INT4", "INT8",
                   "SERIAL", "BIGSERIAL", "MEDIUMINT"],
        "string": ["VARCHAR", "CHAR", "TEXT", "STRING", "NVARCHAR", "NCHAR", "NTEXT",
                  "CHARACTER VARYING", "LONGTEXT", "MEDIUMTEXT"],
        "uuid": ["UUID", "GUID", "UNIQUEIDENTIFIER"],
        "float": ["FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL", "DOUBLE PRECISION",
                 "MONEY", "SMALLMONEY"],
        "date": ["DATE", "DATETIME", "TIMESTAMP", "TIMESTAMPTZ", "DATETIME2"],
        "boolean": ["BOOLEAN", "BOOL", "BIT"],
    }

    def __init__(
        self,
        datasets: List[Dict],
        relationships: Optional[List[Dict]] = None,
        table_name_lookup: Optional[Dict[str, int]] = None
    ):
        """
        Initialize with dataset metadata.

        Args:
            datasets: List of dataset dicts with columns
            relationships: Optional pre-defined relationships from YAML
            table_name_lookup: Optional mapping of table names to dataset IDs
        """
        self.datasets = {ds["id"]: ds for ds in datasets}
        self.relationships = relationships or []
        self._build_table_lookup(table_name_lookup)
        self._index_columns()

    def _build_table_lookup(self, provided_lookup: Optional[Dict[str, int]]):
        """Build table name to dataset ID lookup."""
        if provided_lookup:
            self.table_to_dataset_id = provided_lookup
        else:
            self.table_to_dataset_id = {}
            for ds_id, ds in self.datasets.items():
                table_name = ds.get("table_name", "").lower()
                self.table_to_dataset_id[table_name] = ds_id
                # Also add singular/plural variants
                if table_name.endswith('s'):
                    self.table_to_dataset_id[table_name[:-1]] = ds_id
                else:
                    self.table_to_dataset_id[table_name + 's'] = ds_id

    def _index_columns(self):
        """Build column indexes for fast lookup."""
        self.columns_by_dataset: Dict[int, List[JoinColumn]] = {}
        self.pk_columns: Dict[int, List[JoinColumn]] = {}
        self.fk_columns: Dict[int, List[JoinColumn]] = {}
        self.columns_by_normalized: Dict[str, List[JoinColumn]] = {}

        for ds_id, ds in self.datasets.items():
            columns = []
            table_name = ds.get("table_name", "")

            for col in ds.get("columns", []):
                col_name = col.get("column_name", "")
                if not col_name:
                    continue

                col_type = col.get("type", "") or ""
                normalized = ColumnNormalizer.normalize_for_comparison(col_name)
                is_pk = col_name.lower() in ("id", "uuid", "pk", "guid")
                fk_entity, fk_key_type = ColumnNormalizer.extract_entity_from_fk(col_name)

                jc = JoinColumn(
                    dataset_id=ds_id,
                    table_name=table_name,
                    column_name=col_name,
                    column_type=col_type,
                    normalized_name=normalized,
                    is_primary_key=is_pk,
                    fk_entity=fk_entity,
                    fk_key_type=fk_key_type
                )
                columns.append(jc)

                # Index by normalized name
                if normalized not in self.columns_by_normalized:
                    self.columns_by_normalized[normalized] = []
                self.columns_by_normalized[normalized].append(jc)

                # Index PKs and FKs
                if is_pk:
                    if ds_id not in self.pk_columns:
                        self.pk_columns[ds_id] = []
                    self.pk_columns[ds_id].append(jc)

                if fk_entity:
                    if ds_id not in self.fk_columns:
                        self.fk_columns[ds_id] = []
                    self.fk_columns[ds_id].append(jc)

            self.columns_by_dataset[ds_id] = columns

    def _types_compatible(self, type1: str, type2: str) -> bool:
        """
        Check if two column types are compatible for joining.

        Args:
            type1: First column type
            type2: Second column type

        Returns:
            True if types are compatible
        """
        if not type1 or not type2:
            return True  # Unknown types - assume compatible

        t1_upper = type1.upper()
        t2_upper = type2.upper()

        # Exact match
        if t1_upper == t2_upper:
            return True

        # Check type groups
        for group, types in self.TYPE_GROUPS.items():
            t1_in = any(t in t1_upper for t in types)
            t2_in = any(t in t2_upper for t in types)
            if t1_in and t2_in:
                return True

        return False

    def _check_yaml_relationships(
        self,
        ds1_id: int,
        ds2_id: int
    ) -> List[JoinSuggestion]:
        """Check for pre-defined relationships in YAML."""
        suggestions = []

        for rel in self.relationships:
            left_id = rel.get("left_dataset_id")
            right_id = rel.get("right_dataset_id")

            # Check if this relationship involves our datasets
            if not ((left_id == ds1_id and right_id == ds2_id) or
                   (left_id == ds2_id and right_id == ds1_id)):
                continue

            # Handle column specification (could be 'on' list or direct columns)
            on_spec = rel.get("on", [])
            if isinstance(on_spec, list) and on_spec:
                left_col = on_spec[0].get("left", rel.get("left_column"))
                right_col = on_spec[0].get("right", rel.get("right_column"))
            else:
                left_col = rel.get("left_column")
                right_col = rel.get("right_column")

            # Swap if needed to match ds1/ds2 order
            if left_id == ds2_id:
                left_col, right_col = right_col, left_col

            if not left_col or not right_col:
                continue

            suggestions.append(JoinSuggestion(
                left_dataset_id=ds1_id,
                left_table=self.datasets[ds1_id].get("table_name", ""),
                left_column=left_col,
                right_dataset_id=ds2_id,
                right_table=self.datasets[ds2_id].get("table_name", ""),
                right_column=right_col,
                join_type=rel.get("type", rel.get("join_type", "inner")),
                confidence=1.0,  # YAML relationships are definitive
                reason="defined in relationships.yaml",
                type_compatible=True
            ))

        return suggestions

    def _find_normalized_matches(
        self,
        ds1_id: int,
        ds2_id: int
    ) -> List[JoinSuggestion]:
        """Find columns that match after normalization."""
        suggestions = []
        cols1 = self.columns_by_dataset.get(ds1_id, [])
        cols2 = self.columns_by_dataset.get(ds2_id, [])

        # Build lookup for ds2
        ds2_by_normalized: Dict[str, List[JoinColumn]] = {}
        for c in cols2:
            if c.normalized_name not in ds2_by_normalized:
                ds2_by_normalized[c.normalized_name] = []
            ds2_by_normalized[c.normalized_name].append(c)

        for c1 in cols1:
            # Only consider ID-like columns for this strategy
            if not ColumnNormalizer.is_id_column(c1.column_name):
                continue

            matches = ds2_by_normalized.get(c1.normalized_name, [])
            for c2 in matches:
                if not ColumnNormalizer.is_id_column(c2.column_name):
                    continue

                type_compat = self._types_compatible(c1.column_type, c2.column_type)

                # Score based on exact vs normalized match
                if c1.column_name == c2.column_name:
                    conf = 0.85 if type_compat else 0.70
                    reason = "exact column name match"
                else:
                    conf = 0.78 if type_compat else 0.63
                    reason = f"normalized match ({c1.column_name} ↔ {c2.column_name})"

                suggestions.append(JoinSuggestion(
                    left_dataset_id=ds1_id,
                    left_table=self.datasets[ds1_id].get("table_name", ""),
                    left_column=c1.column_name,
                    right_dataset_id=ds2_id,
                    right_table=self.datasets[ds2_id].get("table_name", ""),
                    right_column=c2.column_name,
                    join_type="inner",
                    confidence=conf,
                    reason=reason,
                    type_compatible=type_compat
                ))

        return suggestions

    def _find_fk_to_pk_matches(
        self,
        ds1_id: int,
        ds2_id: int
    ) -> List[JoinSuggestion]:
        """
        Find FK → PK relationships based on naming patterns.

        Looks for patterns like:
        - assetId in events → assets.id
        - userId in orders → users.id
        """
        suggestions = []

        # Check FKs in ds1 → PKs in ds2
        suggestions.extend(self._find_fk_pk_direction(ds1_id, ds2_id))

        # Check FKs in ds2 → PKs in ds1
        suggestions.extend(self._find_fk_pk_direction(ds2_id, ds1_id, reverse=True))

        return suggestions

    def _find_fk_pk_direction(
        self,
        fk_ds_id: int,
        pk_ds_id: int,
        reverse: bool = False
    ) -> List[JoinSuggestion]:
        """Find FK → PK matches in one direction."""
        suggestions = []

        fks = self.fk_columns.get(fk_ds_id, [])
        pks = self.pk_columns.get(pk_ds_id, [])

        if not fks or not pks:
            return suggestions

        pk_table_name = self.datasets[pk_ds_id].get("table_name", "").lower()

        # Normalize table name for matching
        pk_table_normalized = ColumnNormalizer.normalize_for_comparison(pk_table_name)
        # Also try singular form
        pk_table_singular = pk_table_normalized.rstrip('s')
        # And the raw lowercase
        pk_table_variants = {pk_table_normalized, pk_table_singular, pk_table_name}

        for fk in fks:
            if not fk.fk_entity:
                continue

            fk_entity_normalized = ColumnNormalizer.normalize_for_comparison(fk.fk_entity)

            # Check if FK entity matches table name
            if fk_entity_normalized not in pk_table_variants:
                # Also check if fk_entity is a prefix of the table name
                if not pk_table_normalized.startswith(fk_entity_normalized):
                    continue

            for pk in pks:
                type_compat = self._types_compatible(fk.column_type, pk.column_type)
                conf = 0.80 if type_compat else 0.65

                # Determine join direction and type
                if reverse:
                    suggestions.append(JoinSuggestion(
                        left_dataset_id=pk_ds_id,
                        left_table=self.datasets[pk_ds_id].get("table_name", ""),
                        left_column=pk.column_name,
                        right_dataset_id=fk_ds_id,
                        right_table=self.datasets[fk_ds_id].get("table_name", ""),
                        right_column=fk.column_name,
                        join_type="inner",
                        confidence=conf,
                        reason=f"FK pattern: {fk.column_name} → {pk_table_name}.{pk.column_name}",
                        type_compatible=type_compat
                    ))
                else:
                    suggestions.append(JoinSuggestion(
                        left_dataset_id=fk_ds_id,
                        left_table=self.datasets[fk_ds_id].get("table_name", ""),
                        left_column=fk.column_name,
                        right_dataset_id=pk_ds_id,
                        right_table=self.datasets[pk_ds_id].get("table_name", ""),
                        right_column=pk.column_name,
                        join_type="left",  # FK typically means left join
                        confidence=conf,
                        reason=f"FK pattern: {fk.column_name} → {pk_table_name}.{pk.column_name}",
                        type_compatible=type_compat
                    ))

        return suggestions

    def _find_by_join_key_hint(
        self,
        ds1_id: int,
        ds2_id: int,
        join_key_hint: str
    ) -> List[JoinSuggestion]:
        """Find join using a specific column name hint."""
        suggestions = []

        cols1 = self.columns_by_dataset.get(ds1_id, [])
        cols2 = self.columns_by_dataset.get(ds2_id, [])

        hint_normalized = ColumnNormalizer.normalize_for_comparison(join_key_hint)

        # Find matching columns in each dataset
        c1_match = None
        c2_match = None

        for c in cols1:
            if ColumnNormalizer.columns_match(c.column_name, join_key_hint):
                c1_match = c
                break

        for c in cols2:
            if ColumnNormalizer.columns_match(c.column_name, join_key_hint):
                c2_match = c
                break

        if c1_match and c2_match:
            type_compat = self._types_compatible(c1_match.column_type, c2_match.column_type)
            suggestions.append(JoinSuggestion(
                left_dataset_id=ds1_id,
                left_table=self.datasets[ds1_id].get("table_name", ""),
                left_column=c1_match.column_name,
                right_dataset_id=ds2_id,
                right_table=self.datasets[ds2_id].get("table_name", ""),
                right_column=c2_match.column_name,
                join_type="inner",
                confidence=0.90 if type_compat else 0.75,
                reason=f"explicit join key hint: {join_key_hint}",
                type_compatible=type_compat
            ))

        return suggestions

    def suggest_joins(
        self,
        dataset_ids: List[int],
        join_key_hints: Optional[List[str]] = None
    ) -> List[JoinSuggestion]:
        """
        Suggest joins for a list of datasets.

        Args:
            dataset_ids: List of dataset IDs to find joins between
            join_key_hints: Optional list of column names mentioned as join keys

        Returns:
            List of JoinSuggestion sorted by confidence
        """
        all_suggestions: List[JoinSuggestion] = []
        join_key_hints = join_key_hints or []

        # Check all pairs
        for i, ds1_id in enumerate(dataset_ids):
            for ds2_id in dataset_ids[i + 1:]:
                # Priority 1: YAML relationships
                yaml_joins = self._check_yaml_relationships(ds1_id, ds2_id)
                if yaml_joins:
                    all_suggestions.extend(yaml_joins)
                    continue  # Skip heuristics if YAML defines the relationship

                # Priority 2: Explicit join key hints
                for hint in join_key_hints:
                    hint_joins = self._find_by_join_key_hint(ds1_id, ds2_id, hint)
                    all_suggestions.extend(hint_joins)

                # Priority 3: Normalized column matches
                all_suggestions.extend(self._find_normalized_matches(ds1_id, ds2_id))

                # Priority 4: FK → PK patterns
                all_suggestions.extend(self._find_fk_to_pk_matches(ds1_id, ds2_id))

        # Deduplicate and sort by confidence
        seen: Set[tuple] = set()
        unique: List[JoinSuggestion] = []

        for s in sorted(all_suggestions, key=lambda x: x.confidence, reverse=True):
            key = (s.left_dataset_id, s.left_column, s.right_dataset_id, s.right_column)
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique

    def get_best_join(
        self,
        ds1_id: int,
        ds2_id: int,
        join_key_hint: Optional[str] = None
    ) -> Optional[JoinSuggestion]:
        """
        Get the best join suggestion between two datasets.

        Args:
            ds1_id: First dataset ID
            ds2_id: Second dataset ID
            join_key_hint: Optional specific column to use for join

        Returns:
            Best JoinSuggestion or None if no join found
        """
        hints = [join_key_hint] if join_key_hint else []
        suggestions = self.suggest_joins([ds1_id, ds2_id], hints)
        return suggestions[0] if suggestions else None