"""
Semantic Configuration Manager

Manages the JSON configuration file for the semantic matcher.
Provides read/write access to the schema-based table definitions.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


class SemanticConfigManager:
    """
    Manages semantic matcher configuration stored in JSON.

    The configuration (v2.0) includes:
    - databases: Database connection info
    - schemas: Schema definitions with tables and columns
    - join_patterns: Pre-defined join relationships
    - synonyms: Term synonyms for matching
    - table_prefixes/suffixes: For name normalization
    """

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__),
        "semantic_config.json"
    )

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the config manager."""
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Dict = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except json.JSONDecodeError:
                self._config = self._get_default_config()
        else:
            self._config = self._get_default_config()
            self._save_config()

    def _save_config(self) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self._config["last_updated"] = datetime.now().isoformat()

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, default=str)

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "version": "2.0",
            "last_updated": None,
            "table_prefixes": ["tbl_", "t_", "vw_", "view_", "dim_", "fact_", "stg_", "raw_", "src_"],
            "table_suffixes": ["_table", "_view", "_data", "_v1", "_v2", "_v3", "_staging", "_raw"],
            "databases": {},
            "schemas": {},
            "join_patterns": [],
            "synonyms": {}
        }

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    @property
    def table_prefixes(self) -> List[str]:
        return self._config.get("table_prefixes", [])

    @property
    def table_suffixes(self) -> List[str]:
        return self._config.get("table_suffixes", [])

    @property
    def databases(self) -> Dict[str, Dict]:
        return self._config.get("databases", {})

    @property
    def schemas(self) -> Dict[str, Dict]:
        return self._config.get("schemas", {})

    @property
    def join_patterns(self) -> List[Dict]:
        return self._config.get("join_patterns", [])

    @property
    def synonyms(self) -> Dict[str, List[str]]:
        return self._config.get("synonyms", {})

    @property
    def all_synonyms(self) -> Dict[str, List[str]]:
        """Backwards compatible - returns synonyms."""
        return self.synonyms

    @property
    def learned_tables(self) -> List[Dict]:
        """Backwards compatible - flatten schemas to table list."""
        tables = []
        for schema_name, schema_data in self.schemas.items():
            for table_name, table_data in schema_data.get("tables", {}).items():
                tables.append({
                    "dataset_id": table_data.get("dataset_id"),
                    "table_name": table_name,
                    "schema": schema_name,
                    "database_id": 1,
                    "columns": list(table_data.get("columns", {}).keys())
                })
        return tables

    @property
    def learned_databases(self) -> Dict[str, Dict]:
        """Backwards compatible - returns databases."""
        return self.databases

    @property
    def last_updated(self) -> Optional[str]:
        return self._config.get("last_updated")

    def get_table(self, table_name: str) -> Optional[Dict]:
        """Get table config by name."""
        table_lower = table_name.lower()
        for schema_data in self.schemas.values():
            for name, data in schema_data.get("tables", {}).items():
                if name.lower() == table_lower:
                    return data
        return None

    def get_tables_in_schema(self, schema: str) -> List[str]:
        """Get all table names in a schema."""
        schema_data = self.schemas.get(schema, {})
        return list(schema_data.get("tables", {}).keys())

    def add_table(self, schema: str, table_name: str, table_config: Dict) -> None:
        """Add or update a table configuration."""
        if schema not in self._config["schemas"]:
            self._config["schemas"][schema] = {"description": "", "tables": {}}
        self._config["schemas"][schema]["tables"][table_name] = table_config
        self._save_config()

    def add_join_pattern(self, left: str, right: str, on: str, join_type: str = "inner") -> None:
        """Add a join pattern."""
        pattern = {"left": left, "right": right, "on": on, "type": join_type}
        if pattern not in self._config["join_patterns"]:
            self._config["join_patterns"].append(pattern)
            self._save_config()

    def add_synonym(self, term: str, synonyms: List[str]) -> None:
        """Add synonyms for a term."""
        term_lower = term.lower()
        if term_lower not in self._config["synonyms"]:
            self._config["synonyms"][term_lower] = []
        existing = set(self._config["synonyms"][term_lower])
        existing.update([s.lower() for s in synonyms])
        self._config["synonyms"][term_lower] = list(existing)
        self._save_config()

    def update_from_training(
        self,
        tables: List[Dict],
        databases: Dict[str, Dict],
        synonyms: Dict[str, List[str]],
        join_patterns: List[Dict],
        overwrite: bool = True,
        **kwargs
    ) -> None:
        """
        Update config from training results.

        Converts flat table list to schema-based structure.

        Args:
            tables: List of table configurations
            databases: Database info keyed by ID
            synonyms: Synonym mappings
            join_patterns: Join relationships
            overwrite: If True, completely replaces existing data. If False, merges.
        """
        if overwrite:
            # Clear existing data and start fresh
            self._config["databases"] = {}
            self._config["schemas"] = {}
            self._config["synonyms"] = {}
            self._config["join_patterns"] = []

        # Update databases
        self._config["databases"] = databases

        # Group tables by schema
        schema_tables: Dict[str, Dict[str, Dict]] = {}
        for table in tables:
            schema = table.get("schema", "default")
            if schema not in schema_tables:
                schema_tables[schema] = {}

            table_name = table.get("table_name", "")
            if not table_name:
                continue

            columns = {}
            for col in table.get("columns", []):
                if isinstance(col, str):
                    columns[col] = {"type": "string"}
                elif isinstance(col, dict):
                    col_name = col.get("name", "")
                    if col_name:
                        columns[col_name] = {
                            "type": col.get("type", "string"),
                            "is_pk": col.get("is_pk", False),
                            "is_fk": col.get("is_fk", False),
                            "references": col.get("references")
                        }

            schema_tables[schema][table_name] = {
                "dataset_id": table.get("dataset_id"),
                "database_id": table.get("database_id"),
                "columns": columns,
                "aliases": table.get("aliases", [])
            }

        # Replace or update schemas based on overwrite flag
        if overwrite:
            for schema, tables_dict in schema_tables.items():
                self._config["schemas"][schema] = {"description": "", "tables": tables_dict}
        else:
            for schema, tables_dict in schema_tables.items():
                if schema not in self._config["schemas"]:
                    self._config["schemas"][schema] = {"description": "", "tables": {}}
                self._config["schemas"][schema]["tables"].update(tables_dict)

        # Update synonyms
        if overwrite:
            self._config["synonyms"] = synonyms
        else:
            self._config["synonyms"].update(synonyms)

        # Update join patterns
        for jp in join_patterns:
            left = jp.get("left_table", jp.get("left", ""))
            right = jp.get("right_table", jp.get("right", ""))
            left_col = jp.get("left_column", "")
            right_col = jp.get("right_column", "")
            if left and right and left_col and right_col:
                pattern = {
                    "left": left,
                    "right": right,
                    "on": f"{left_col} = {right_col}",
                    "type": jp.get("join_type", "inner")
                }
                if pattern not in self._config["join_patterns"]:
                    self._config["join_patterns"].append(pattern)

        self._save_config()

    def clear(self) -> None:
        """Clear all learned data."""
        self._config = self._get_default_config()
        self._save_config()

    def get_synonyms_for(self, term: str) -> List[str]:
        """Get all synonyms for a term."""
        term_lower = term.lower()
        result = set()

        if term_lower in self.synonyms:
            result.update(self.synonyms[term_lower])

        for key, values in self.synonyms.items():
            if term_lower in values:
                result.add(key)
                result.update(values)

        return list(result)


# Global instance
_config_manager: Optional[SemanticConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> SemanticConfigManager:
    """Get or create the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = SemanticConfigManager(config_path)
    return _config_manager