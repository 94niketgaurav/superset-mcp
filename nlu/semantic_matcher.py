"""
Semantic dataset matching for natural language to SQL query building.

This module provides fast table/column lookup and join path discovery
using a schema-based configuration file.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    from rapidfuzz import fuzz
    USE_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    USE_RAPIDFUZZ = False


@dataclass
class ColumnInfo:
    """Column metadata for query building."""
    name: str
    column_type: str
    is_pk: bool = False
    is_fk: bool = False
    references: Optional[str] = None


@dataclass
class TableInfo:
    """Table metadata for query building."""
    name: str
    schema: str
    dataset_id: int
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    database_id: Optional[int] = None
    is_virtual: bool = False  # True for Superset virtual datasets (SQL-based views)
    description: Optional[str] = None
    physical_tables: List[str] = field(default_factory=list)  # Underlying physical tables for virtual datasets

    @property
    def primary_keys(self) -> List[str]:
        return [c.name for c in self.columns.values() if c.is_pk]

    @property
    def foreign_keys(self) -> List[Tuple[str, str]]:
        return [(c.name, c.references) for c in self.columns.values() if c.is_fk and c.references]


@dataclass
class JoinPath:
    """A join between two tables."""
    left_table: str
    right_table: str
    left_column: str
    right_column: str
    join_type: str = "inner"
    confidence: float = 1.0


@dataclass
class MatchResult:
    """Result of matching an entity to a table."""
    table_name: str
    schema: str
    dataset_id: int
    score: float
    match_reason: str
    table_info: Optional[TableInfo] = None


class SemanticMatcher:
    """
    Semantic matcher for table/column lookup and join discovery.

    Loads table definitions, join patterns, and synonyms from a JSON config file.
    """

    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "semantic_config.json"
    )

    TABLE_PREFIXES = ["tbl_", "t_", "vw_", "view_", "dim_", "fact_", "stg_", "raw_", "src_"]
    TABLE_SUFFIXES = ["_table", "_view", "_data", "_v1", "_v2", "_v3", "_staging", "_raw"]

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the semantic matcher.

        Args:
            config_path: Path to config JSON. Uses default if not provided.
        """
        self.config_path = config_path or self.CONFIG_PATH
        self.config: Dict = {}

        # Core lookups
        self.tables: Dict[str, TableInfo] = {}
        self.aliases: Dict[str, str] = {}  # alias -> table_name
        self.synonyms: Dict[str, List[str]] = {}
        self.join_patterns: Dict[str, List[JoinPath]] = {}
        self.schemas: Dict[str, str] = {}  # schema_name -> description
        self.database_names: Dict[int, str] = {}

        self._load_config()

    def _load_config(self) -> None:
        """Load and index configuration from JSON."""
        if not os.path.exists(self.config_path):
            return

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Load prefixes/suffixes
        self.TABLE_PREFIXES = self.config.get("table_prefixes", self.TABLE_PREFIXES)
        self.TABLE_SUFFIXES = self.config.get("table_suffixes", self.TABLE_SUFFIXES)

        # Load databases
        for db_id_str, db_info in self.config.get("databases", {}).items():
            try:
                self.database_names[int(db_id_str)] = db_info.get("name", f"Database {db_id_str}")
            except ValueError:
                pass

        # Load synonyms
        self.synonyms = self.config.get("synonyms", {})

        # Load schemas and tables
        for schema_name, schema_data in self.config.get("schemas", {}).items():
            self.schemas[schema_name] = schema_data.get("description", "")

            for table_name, table_data in schema_data.get("tables", {}).items():
                columns: Dict[str, ColumnInfo] = {}
                for col_name, col_data in table_data.get("columns", {}).items():
                    columns[col_name] = ColumnInfo(
                        name=col_name,
                        column_type=col_data.get("type", "string"),
                        is_pk=col_data.get("is_pk", False),
                        is_fk=col_data.get("is_fk", False),
                        references=col_data.get("references")
                    )

                table_info = TableInfo(
                    name=table_name,
                    schema=schema_name,
                    dataset_id=table_data.get("dataset_id", 0),
                    columns=columns,
                    aliases=table_data.get("aliases", []),
                    database_id=table_data.get("database_id", 1),
                    is_virtual=table_data.get("is_virtual", False),
                    description=table_data.get("description"),
                    physical_tables=table_data.get("physical_tables", [])
                )

                table_key = table_name.lower()
                self.tables[table_key] = table_info

                # Index aliases
                for alias in table_info.aliases:
                    self.aliases[alias.lower()] = table_key

        # Load join patterns
        for jp in self.config.get("join_patterns", []):
            on_clause = jp.get("on", "")
            parts = on_clause.split("=")
            if len(parts) == 2:
                left_col = parts[0].strip()
                right_col = parts[1].strip()

                join_path = JoinPath(
                    left_table=jp.get("left", ""),
                    right_table=jp.get("right", ""),
                    left_column=left_col,
                    right_column=right_col,
                    join_type=jp.get("type", "inner")
                )

                left_key = join_path.left_table.lower()
                right_key = join_path.right_table.lower()

                # Forward mapping
                if left_key not in self.join_patterns:
                    self.join_patterns[left_key] = []
                self.join_patterns[left_key].append(join_path)

                # Reverse mapping
                if right_key not in self.join_patterns:
                    self.join_patterns[right_key] = []
                self.join_patterns[right_key].append(JoinPath(
                    left_table=join_path.right_table,
                    right_table=join_path.left_table,
                    left_column=right_col,
                    right_column=left_col,
                    join_type=join_path.join_type
                ))

    def _normalize_name(self, name: str) -> str:
        """Normalize a table name by removing common prefixes/suffixes."""
        name = name.lower()
        for prefix in self.TABLE_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        for suffix in self.TABLE_SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        return name

    def _string_similarity(self, a: str, b: str) -> float:
        """Calculate string similarity."""
        if USE_RAPIDFUZZ:
            return fuzz.ratio(a.lower(), b.lower()) / 100.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _get_synonyms(self, term: str) -> Set[str]:
        """Get all synonyms for a term."""
        term_lower = term.lower()
        result = {term_lower}

        # Direct lookup
        if term_lower in self.synonyms:
            result.update(self.synonyms[term_lower])

        # Reverse lookup
        for key, syns in self.synonyms.items():
            if term_lower in syns:
                result.add(key)
                result.update(syns)

        # Singular/plural
        if term_lower.endswith('s'):
            result.add(term_lower[:-1])
        else:
            result.add(term_lower + 's')

        if term_lower.endswith('ies'):
            result.add(term_lower[:-3] + 'y')
        elif term_lower.endswith('y'):
            result.add(term_lower[:-1] + 'ies')

        return result

    def match_table(self, entity: str, min_score: float = 0.4, fuzzy_min_score: float = 0.7,
                    exclude_virtual: bool = False, deprioritize_virtual: bool = True) -> List[MatchResult]:
        """
        Match an entity name to tables.

        Priority order:
        1. Exact match (score: 1.0)
        2. Alias match (score: 0.98)
        3. Synonym match (score: 0.95)
        4. Prefix match (score: 0.7-0.9)
        5. Suffix match (score: 0.65-0.8)
        6. Contains match (score: 0.5-0.65)
        7. Fuzzy match (only if no semantic matches found, requires fuzzy_min_score)

        Virtual datasets (Superset SQL-based views) are deprioritized by default since they
        cannot be joined with physical tables in raw SQL queries.

        Args:
            entity: The entity/table name to match (e.g., "assets", "user")
            min_score: Minimum similarity score to include
            fuzzy_min_score: Minimum score for fuzzy matches (default 0.7)
            exclude_virtual: If True, completely exclude virtual datasets from results
            deprioritize_virtual: If True, reduce score of virtual datasets by 0.3

        Returns:
            List of MatchResult sorted by score descending
        """
        entity_lower = entity.lower()
        entity_normalized = self._normalize_name(entity_lower)
        synonyms = self._get_synonyms(entity_lower)
        semantic_results: List[MatchResult] = []
        fuzzy_results: List[MatchResult] = []

        # Check aliases first
        if entity_lower in self.aliases:
            actual_table = self.aliases[entity_lower]
            if actual_table in self.tables:
                ti = self.tables[actual_table]
                # Skip or deprioritize virtual datasets
                if ti.is_virtual:
                    if exclude_virtual:
                        pass  # Skip this table entirely
                    else:
                        alias_score = 0.98 - (0.3 if deprioritize_virtual else 0)
                        semantic_results.append(MatchResult(
                            table_name=ti.name,
                            schema=ti.schema,
                            dataset_id=ti.dataset_id,
                            score=alias_score,
                            match_reason="alias match (virtual dataset)",
                            table_info=ti
                        ))
                else:
                    semantic_results.append(MatchResult(
                        table_name=ti.name,
                        schema=ti.schema,
                        dataset_id=ti.dataset_id,
                        score=0.98,
                        match_reason="alias match",
                        table_info=ti
                    ))

        # Match against all tables
        for table_key, table_info in self.tables.items():
            # Skip if already added as alias
            if any(r.table_name.lower() == table_key for r in semantic_results):
                continue

            # Skip virtual datasets if exclude_virtual is set
            if table_info.is_virtual and exclude_virtual:
                continue

            normalized = self._normalize_name(table_key)
            score = 0.0
            reason = ""
            is_semantic = True

            # Exact match
            if table_key == entity_lower or normalized == entity_normalized:
                score = 1.0
                reason = "exact match"

            # Synonym match
            elif table_key in synonyms or normalized in synonyms:
                score = 0.95
                reason = "synonym match"

            # Prefix match
            elif table_key.startswith(entity_lower) or normalized.startswith(entity_normalized):
                ratio = len(entity_lower) / len(table_key) if table_key else 0
                score = 0.7 + (0.2 * ratio)
                reason = f"prefix match ({entity_lower}*)"

            # Suffix match
            elif table_key.endswith(entity_lower) or normalized.endswith(entity_normalized):
                ratio = len(entity_lower) / len(table_key) if table_key else 0
                score = 0.65 + (0.15 * ratio)
                reason = f"suffix match (*{entity_lower})"

            # Contains match
            elif entity_lower in table_key or entity_normalized in normalized:
                ratio = len(entity_lower) / len(table_key) if table_key else 0
                score = 0.5 + (0.15 * ratio)
                reason = f"contains '{entity_lower}'"

            # Fuzzy match (separate list, only used if no semantic matches)
            else:
                is_semantic = False
                best_sim = 0.0
                for syn in synonyms:
                    sim = max(
                        self._string_similarity(syn, table_key),
                        self._string_similarity(syn, normalized)
                    )
                    if sim > best_sim:
                        best_sim = sim
                if best_sim >= fuzzy_min_score:
                    score = best_sim
                    reason = f"fuzzy match ({best_sim:.2f})"

            if score >= min_score:
                # Deprioritize virtual datasets (reduce score by 0.3)
                final_score = score
                final_reason = reason
                if table_info.is_virtual and deprioritize_virtual:
                    final_score = max(0.1, score - 0.3)
                    final_reason = f"{reason} (virtual dataset)"

                result = MatchResult(
                    table_name=table_info.name,
                    schema=table_info.schema,
                    dataset_id=table_info.dataset_id,
                    score=round(final_score, 3),
                    match_reason=final_reason,
                    table_info=table_info
                )
                if is_semantic:
                    semantic_results.append(result)
                else:
                    fuzzy_results.append(result)

        # Return semantic matches if any, otherwise fuzzy matches
        if semantic_results:
            semantic_results.sort(key=lambda r: r.score, reverse=True)
            return semantic_results[:10]
        else:
            fuzzy_results.sort(key=lambda r: r.score, reverse=True)
            return fuzzy_results[:10]

    def get_table(self, name: str) -> Optional[TableInfo]:
        """Get table info by name or alias."""
        name_lower = name.lower()
        if name_lower in self.tables:
            return self.tables[name_lower]
        if name_lower in self.aliases:
            return self.tables.get(self.aliases[name_lower])
        return None

    def get_column(self, table_name: str, column_name: str) -> Optional[ColumnInfo]:
        """Get column info from a table."""
        table = self.get_table(table_name)
        if table:
            return table.columns.get(column_name)
        return None

    def get_columns(self, table_name: str) -> List[str]:
        """Get all column names for a table."""
        table = self.get_table(table_name)
        return list(table.columns.keys()) if table else []

    def find_tables_with_column(self, column_name: str) -> List[TableInfo]:
        """
        Find all tables that have a specific column.

        Args:
            column_name: The column name to search for

        Returns:
            List of TableInfo for tables containing this column
        """
        column_lower = column_name.lower()
        results = []
        for table_info in self.tables.values():
            for col in table_info.columns.keys():
                if col.lower() == column_lower:
                    results.append(table_info)
                    break
        return results

    def table_has_column(self, table_name: str, column_name: str) -> bool:
        """Check if a table has a specific column."""
        table = self.get_table(table_name)
        if table:
            return any(c.lower() == column_name.lower() for c in table.columns.keys())
        return False

    def get_join_path(self, from_table: str, to_table: str) -> Optional[JoinPath]:
        """Find direct join path between tables."""
        from_lower = from_table.lower()
        to_lower = to_table.lower()

        # Resolve aliases
        if from_lower in self.aliases:
            from_lower = self.aliases[from_lower]
        if to_lower in self.aliases:
            to_lower = self.aliases[to_lower]

        joins = self.join_patterns.get(from_lower, [])
        for jp in joins:
            if jp.right_table.lower() == to_lower:
                return jp
        return None

    def find_join_path(self, tables: List[str], max_depth: int = 3) -> List[JoinPath]:
        """
        Find join paths connecting multiple tables.

        Args:
            tables: List of table names to connect
            max_depth: Maximum intermediate tables to explore

        Returns:
            List of JoinPath connecting the tables
        """
        if len(tables) < 2:
            return []

        # Resolve table names (aliases, matching)
        resolved = []
        for t in tables:
            t_lower = t.lower()
            if t_lower in self.aliases:
                resolved.append(self.aliases[t_lower])
            elif t_lower in self.tables:
                resolved.append(t_lower)
            else:
                matches = self.match_table(t, min_score=0.7)
                if matches:
                    resolved.append(matches[0].table_name.lower())

        if len(resolved) < 2:
            return []

        result: List[JoinPath] = []
        connected = {resolved[0]}
        remaining = set(resolved[1:])

        while remaining and len(result) < max_depth * len(tables):
            found = False
            for table in list(remaining):
                for conn_table in connected:
                    jp = self.get_join_path(conn_table, table)
                    if jp:
                        result.append(jp)
                        connected.add(table)
                        remaining.remove(table)
                        found = True
                        break
                if found:
                    break

            # Try intermediate tables
            if not found:
                for table in list(remaining):
                    for conn_table in connected:
                        for jp in self.join_patterns.get(conn_table, []):
                            intermediate = jp.right_table.lower()
                            if intermediate not in connected and intermediate != table:
                                jp2 = self.get_join_path(intermediate, table)
                                if jp2:
                                    result.append(jp)
                                    result.append(jp2)
                                    connected.add(intermediate)
                                    connected.add(table)
                                    remaining.remove(table)
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
            if not found:
                break

        return result

    def get_all_tables(self) -> List[TableInfo]:
        """Get all tables."""
        return list(self.tables.values())

    def get_tables_in_schema(self, schema: str) -> List[TableInfo]:
        """Get tables in a schema."""
        return [t for t in self.tables.values() if t.schema == schema]

    def get_schema_description(self, schema: str) -> str:
        """Get schema description."""
        return self.schemas.get(schema, "")

    def get_database_name(self, database_id: int) -> Optional[str]:
        """Get database name by ID."""
        return self.database_names.get(database_id)

    def get_primary_keys(self, table_name: str) -> List[str]:
        """Get primary key columns."""
        table = self.get_table(table_name)
        return table.primary_keys if table else []

    def get_foreign_keys(self, table_name: str) -> List[Dict[str, str]]:
        """Get foreign key columns."""
        table = self.get_table(table_name)
        if not table:
            return []
        return [{"column": fk[0], "references": fk[1]} for fk in table.foreign_keys]

    def get_joinable_columns(self, table_name: str) -> List[Dict]:
        """Get columns usable for joins."""
        table = self.get_table(table_name)
        if not table:
            return []
        result = []
        for col in table.columns.values():
            is_joinable = (
                col.is_pk or col.is_fk or
                col.name.lower().endswith('_id') or
                col.name.lower().endswith('id') or
                col.name.lower() in ('id', 'uuid', 'key')
            )
            if is_joinable:
                result.append({
                    "column": col.name,
                    "type": col.column_type,
                    "is_pk": col.is_pk,
                    "is_fk": col.is_fk,
                    "references": col.references
                })
        return result

    def reload(self) -> None:
        """Reload configuration from file."""
        self.tables.clear()
        self.aliases.clear()
        self.synonyms.clear()
        self.join_patterns.clear()
        self.schemas.clear()
        self.database_names.clear()
        self._load_config()

    def get_statistics(self) -> Dict:
        """Get matcher statistics."""
        total_columns = sum(len(t.columns) for t in self.tables.values())
        total_pks = sum(len(t.primary_keys) for t in self.tables.values())
        total_fks = sum(len(t.foreign_keys) for t in self.tables.values())

        return {
            "total_tables": len(self.tables),
            "total_columns": total_columns,
            "total_primary_keys": total_pks,
            "total_foreign_keys": total_fks,
            "schemas": list(self.schemas.keys()),
            "databases": len(self.database_names),
            "join_patterns": sum(len(v) for v in self.join_patterns.values()) // 2,
            "synonyms": len(self.synonyms),
            "aliases": len(self.aliases)
        }

    def get_database_backend(self, database_id: int) -> str:
        """
        Get the backend/dialect for a specific database.

        Args:
            database_id: The database ID

        Returns:
            Backend name (e.g., "starrocks", "postgresql") or default "starrocks"
        """
        databases = self.config.get("databases", {})
        db_info = databases.get(str(database_id), {})
        return db_info.get("backend", "starrocks")

    def get_table_database_id(self, table_name: str) -> Optional[int]:
        """
        Get the database ID for a specific table.

        Args:
            table_name: The table name

        Returns:
            Database ID or None if not found
        """
        table_info = self.tables.get(table_name.lower())
        if table_info:
            return table_info.database_id
        return None

    def is_table_virtual(self, table_name: str) -> bool:
        """
        Check if a table is a virtual dataset (Superset SQL-based view).

        Virtual datasets cannot be joined with physical tables in raw SQL queries.
        They should only be queried through Superset's explore interface.

        Args:
            table_name: The table name to check

        Returns:
            True if the table is a virtual dataset, False otherwise
        """
        table_info = self.tables.get(table_name.lower())
        if table_info:
            return table_info.is_virtual
        return False

    def get_physical_tables(self, schema: Optional[str] = None) -> List[TableInfo]:
        """
        Get all physical (non-virtual) tables, optionally filtered by schema.

        Args:
            schema: Optional schema name to filter by

        Returns:
            List of TableInfo for physical tables
        """
        tables = self.tables.values()
        if schema:
            tables = [t for t in tables if t.schema == schema]
        return [t for t in tables if not t.is_virtual]

    def get_underlying_physical_tables(self, table_name: str) -> List[TableInfo]:
        """
        Get the underlying physical tables for a virtual dataset.

        Args:
            table_name: The table name (possibly virtual)

        Returns:
            List of physical TableInfo objects. If not virtual, returns the table itself.
        """
        table_info = self.tables.get(table_name.lower())
        if not table_info:
            return []

        if not table_info.is_virtual:
            return [table_info]

        # Get the physical tables for this virtual dataset
        physical = []
        for pt_name in table_info.physical_tables:
            pt_info = self.tables.get(pt_name.lower())
            if pt_info and not pt_info.is_virtual:
                physical.append(pt_info)

        return physical if physical else [table_info]  # Fallback to original if no physical found

    def resolve_tables_for_sql(self, table_names: List[str]) -> Tuple[List[TableInfo], List[str]]:
        """
        Resolve a list of table names, replacing virtual datasets with physical tables.

        Args:
            table_names: List of table names (may include virtual datasets)

        Returns:
            Tuple of (resolved physical tables, warnings about virtual tables)
        """
        resolved = []
        warnings = []
        seen = set()

        for name in table_names:
            table_info = self.tables.get(name.lower())
            if not table_info:
                continue

            if table_info.is_virtual:
                warnings.append(
                    f"'{table_info.name}' is a Superset virtual dataset. "
                    f"Using physical tables: {', '.join(table_info.physical_tables) or 'none defined'}"
                )
                for pt in self.get_underlying_physical_tables(name):
                    if pt.name.lower() not in seen:
                        resolved.append(pt)
                        seen.add(pt.name.lower())
            else:
                if table_info.name.lower() not in seen:
                    resolved.append(table_info)
                    seen.add(table_info.name.lower())

        return resolved, warnings

    def print_summary(self) -> None:
        """Print schema summary."""
        print("\n" + "=" * 60)
        print("SEMANTIC MATCHER - SCHEMA SUMMARY")
        print("=" * 60)

        stats = self.get_statistics()
        print(f"\nTables: {stats['total_tables']}")
        print(f"Columns: {stats['total_columns']}")
        print(f"Join patterns: {stats['join_patterns']}")

        for schema in self.schemas:
            tables = self.get_tables_in_schema(schema)
            print(f"\n[{schema}] {self.schemas[schema]}")
            for table in sorted(tables, key=lambda t: t.name):
                pk_count = len(table.primary_keys)
                fk_count = len(table.foreign_keys)
                print(f"  {table.name}: {len(table.columns)} cols (PK: {pk_count}, FK: {fk_count})")

        print("\n" + "=" * 60)


def create_matcher(config_path: Optional[str] = None) -> SemanticMatcher:
    """Create a semantic matcher from config."""
    return SemanticMatcher(config_path=config_path)


def create_trained_matcher(storage_path: Optional[str] = None) -> SemanticMatcher:
    """Create a semantic matcher (alias for create_matcher)."""
    return SemanticMatcher()
