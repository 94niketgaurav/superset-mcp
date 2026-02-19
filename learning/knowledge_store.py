
"""
Knowledge store for persisting learned dataset and column information.

Stores:
- Dataset metadata (names, schemas, descriptions)
- Column information (types, sample values, statistics)
- Join patterns (learned column relationships)
- Synonyms and aliases discovered from data
"""
import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ColumnKnowledge:
    """Knowledge about a single column."""
    column_name: str
    column_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    fk_references: Optional[str] = None  # "table.column" format
    sample_values: List[Any] = field(default_factory=list)
    distinct_count: Optional[int] = None
    null_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_length: Optional[float] = None  # For string columns
    value_patterns: List[str] = field(default_factory=list)  # Regex patterns found

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ColumnKnowledge":
        return cls(**data)


@dataclass
class DatasetKnowledge:
    """Knowledge about a single dataset/table."""
    dataset_id: int
    table_name: str
    schema: Optional[str] = None
    database_id: Optional[int] = None
    description: Optional[str] = None
    row_count: Optional[int] = None
    columns: Dict[str, ColumnKnowledge] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)  # Alternative names
    related_tables: List[str] = field(default_factory=list)  # Tables often queried together
    common_joins: List[Dict] = field(default_factory=list)  # Learned join patterns
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["columns"] = {k: v.to_dict() if isinstance(v, ColumnKnowledge) else v
                           for k, v in self.columns.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetKnowledge":
        columns = {}
        for col_name, col_data in data.get("columns", {}).items():
            if isinstance(col_data, dict):
                columns[col_name] = ColumnKnowledge.from_dict(col_data)
            else:
                columns[col_name] = col_data
        data["columns"] = columns
        return cls(**data)


@dataclass
class JoinPattern:
    """A learned join pattern between two tables."""
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: str  # inner, left, right
    confidence: float
    usage_count: int = 0
    last_used: Optional[str] = None
    value_overlap_ratio: Optional[float] = None  # How much values overlap

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "JoinPattern":
        return cls(**data)


class KnowledgeStore:
    """
    Persistent store for learned knowledge about datasets.

    Uses SQLite for storage with JSON serialization for complex objects.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the knowledge store.

        Args:
            storage_path: Path to the knowledge database. Defaults to
                         ~/.superset-mcp/knowledge.db
        """
        if storage_path is None:
            home = Path.home()
            storage_dir = home / ".superset-mcp"
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage_path = str(storage_dir / "knowledge.db")

        self.storage_path = storage_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        # Datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id INTEGER PRIMARY KEY,
                table_name TEXT NOT NULL,
                schema TEXT,
                database_id INTEGER,
                knowledge_json TEXT,
                last_updated TEXT,
                UNIQUE(dataset_id)
            )
        """)

        # Create indexes for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_datasets_table_name
            ON datasets(table_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_datasets_database_id
            ON datasets(database_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_datasets_schema
            ON datasets(schema)
        """)

        # Join patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS join_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                left_table TEXT NOT NULL,
                left_column TEXT NOT NULL,
                right_table TEXT NOT NULL,
                right_column TEXT NOT NULL,
                join_type TEXT,
                confidence REAL,
                usage_count INTEGER DEFAULT 0,
                last_used TEXT,
                value_overlap_ratio REAL,
                UNIQUE(left_table, left_column, right_table, right_column)
            )
        """)

        # Synonyms table for table name aliases
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS synonyms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                synonym TEXT NOT NULL,
                source TEXT,  -- 'manual', 'learned', 'inferred'
                confidence REAL DEFAULT 1.0,
                UNIQUE(term, synonym)
            )
        """)

        # Column value patterns for join detection
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS column_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                pattern_type TEXT,  -- 'uuid', 'integer_id', 'email', 'date', etc.
                sample_hash TEXT,  -- Hash of sample values for comparison
                UNIQUE(table_name, column_name)
            )
        """)

        # Metadata table for store information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)

        # Query examples table for few-shot learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,              -- saved_query, chart, chatbot_success
                source_id INTEGER,        -- ID in source system
                title TEXT,
                description TEXT,
                sql TEXT NOT NULL,
                normalized_sql TEXT,      -- Literals replaced with placeholders
                query_type TEXT,          -- simple, join, aggregation, etc.
                tables_json TEXT,         -- JSON list of tables
                pattern_json TEXT,        -- Full QueryPattern as JSON
                keywords_json TEXT,       -- Keywords for matching
                schema_name TEXT,
                database_id INTEGER,
                dialect TEXT DEFAULT 'starrocks',
                complexity_score REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                is_valid INTEGER DEFAULT 1,    -- Still valid against schema
                last_validated TEXT,
                created_at TEXT,
                last_used TEXT,
                UNIQUE(normalized_sql)    -- Deduplicate by normalized form
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_examples_tables
            ON query_examples(tables_json)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_examples_type
            ON query_examples(query_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_examples_schema
            ON query_examples(schema_name)
        """)

        # Query feedback table for tracking success/failure
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                example_id INTEGER,
                natural_language TEXT,    -- Original user query
                generated_sql TEXT,       -- SQL that was generated
                feedback_type TEXT,       -- success, failure, correction
                corrected_sql TEXT,       -- If user corrected the SQL
                error_message TEXT,       -- If query failed
                execution_time_ms INTEGER,
                row_count INTEGER,
                created_at TEXT,
                FOREIGN KEY (example_id) REFERENCES query_examples(id)
            )
        """)

        conn.commit()
        conn.close()

    def save_dataset(self, knowledge: DatasetKnowledge) -> None:
        """Save dataset knowledge to the store."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        knowledge.last_updated = datetime.now().isoformat()
        knowledge_json = json.dumps(knowledge.to_dict())

        cursor.execute("""
            INSERT OR REPLACE INTO datasets
            (dataset_id, table_name, schema, database_id, knowledge_json, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            knowledge.dataset_id,
            knowledge.table_name,
            knowledge.schema,
            knowledge.database_id,
            knowledge_json,
            knowledge.last_updated
        ))

        conn.commit()
        conn.close()

    def get_dataset(self, dataset_id: int) -> Optional[DatasetKnowledge]:
        """Get dataset knowledge by ID."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT knowledge_json FROM datasets WHERE dataset_id = ?",
            (dataset_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            data = json.loads(row[0])
            return DatasetKnowledge.from_dict(data)
        return None

    def get_dataset_by_name(self, table_name: str) -> Optional[DatasetKnowledge]:
        """Get dataset knowledge by table name."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT knowledge_json FROM datasets WHERE LOWER(table_name) = LOWER(?)",
            (table_name,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            data = json.loads(row[0])
            return DatasetKnowledge.from_dict(data)
        return None

    def get_all_datasets(self) -> List[DatasetKnowledge]:
        """Get all dataset knowledge."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("SELECT knowledge_json FROM datasets")
        rows = cursor.fetchall()
        conn.close()

        return [DatasetKnowledge.from_dict(json.loads(row[0])) for row in rows]

    def get_all_table_names(self) -> List[str]:
        """Get all known table names."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("SELECT table_name FROM datasets")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_datasets_by_database(self, database_id: int) -> List[DatasetKnowledge]:
        """Get all datasets for a specific database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT knowledge_json FROM datasets WHERE database_id = ?",
            (database_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [DatasetKnowledge.from_dict(json.loads(row[0])) for row in rows]

    def get_unique_databases(self) -> List[int]:
        """Get list of unique database IDs."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT database_id FROM datasets WHERE database_id IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_unique_schemas(self, database_id: Optional[int] = None) -> List[str]:
        """Get list of unique schemas, optionally filtered by database."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        if database_id is not None:
            cursor.execute(
                "SELECT DISTINCT schema FROM datasets WHERE database_id = ? AND schema IS NOT NULL",
                (database_id,)
            )
        else:
            cursor.execute("SELECT DISTINCT schema FROM datasets WHERE schema IS NOT NULL")

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def save_join_pattern(self, pattern: JoinPattern) -> None:
        """Save or update a join pattern."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO join_patterns
            (left_table, left_column, right_table, right_column,
             join_type, confidence, usage_count, last_used, value_overlap_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(left_table, left_column, right_table, right_column)
            DO UPDATE SET
                confidence = ?,
                usage_count = usage_count + 1,
                last_used = ?,
                value_overlap_ratio = ?
        """, (
            pattern.left_table, pattern.left_column,
            pattern.right_table, pattern.right_column,
            pattern.join_type, pattern.confidence,
            pattern.usage_count, pattern.last_used,
            pattern.value_overlap_ratio,
            # For UPDATE
            pattern.confidence, pattern.last_used, pattern.value_overlap_ratio
        ))

        conn.commit()
        conn.close()

    def get_join_patterns(
        self,
        table_name: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> List[JoinPattern]:
        """Get join patterns, optionally filtered by table."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        if table_name:
            cursor.execute("""
                SELECT left_table, left_column, right_table, right_column,
                       join_type, confidence, usage_count, last_used, value_overlap_ratio
                FROM join_patterns
                WHERE (LOWER(left_table) = LOWER(?) OR LOWER(right_table) = LOWER(?))
                  AND confidence >= ?
                ORDER BY confidence DESC, usage_count DESC
            """, (table_name, table_name, min_confidence))
        else:
            cursor.execute("""
                SELECT left_table, left_column, right_table, right_column,
                       join_type, confidence, usage_count, last_used, value_overlap_ratio
                FROM join_patterns
                WHERE confidence >= ?
                ORDER BY confidence DESC, usage_count DESC
            """, (min_confidence,))

        rows = cursor.fetchall()
        conn.close()

        return [
            JoinPattern(
                left_table=row[0], left_column=row[1],
                right_table=row[2], right_column=row[3],
                join_type=row[4] or "inner",
                confidence=row[5] or 0.5,
                usage_count=row[6] or 0,
                last_used=row[7],
                value_overlap_ratio=row[8]
            )
            for row in rows
        ]

    def add_synonym(
        self,
        term: str,
        synonym: str,
        source: str = "learned",
        confidence: float = 1.0
    ) -> None:
        """Add a synonym for a term."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO synonyms (term, synonym, source, confidence)
            VALUES (?, ?, ?, ?)
        """, (term.lower(), synonym.lower(), source, confidence))

        conn.commit()
        conn.close()

    def get_synonyms(self, term: str) -> List[str]:
        """Get all synonyms for a term."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT synonym FROM synonyms
            WHERE LOWER(term) = LOWER(?)
            ORDER BY confidence DESC
        """, (term,))
        rows = cursor.fetchall()

        # Also check reverse
        cursor.execute("""
            SELECT term FROM synonyms
            WHERE LOWER(synonym) = LOWER(?)
            ORDER BY confidence DESC
        """, (term,))
        rows.extend(cursor.fetchall())

        conn.close()
        return list(set(row[0] for row in rows))

    def save_column_pattern(
        self,
        table_name: str,
        column_name: str,
        pattern_type: str,
        sample_hash: str
    ) -> None:
        """Save column pattern information for join detection."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO column_patterns
            (table_name, column_name, pattern_type, sample_hash)
            VALUES (?, ?, ?, ?)
        """, (table_name, column_name, pattern_type, sample_hash))

        conn.commit()
        conn.close()

    def find_matching_columns(
        self,
        pattern_type: str,
        sample_hash: Optional[str] = None
    ) -> List[Dict]:
        """Find columns with matching patterns or sample hashes."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        if sample_hash:
            cursor.execute("""
                SELECT table_name, column_name, pattern_type, sample_hash
                FROM column_patterns
                WHERE pattern_type = ? OR sample_hash = ?
            """, (pattern_type, sample_hash))
        else:
            cursor.execute("""
                SELECT table_name, column_name, pattern_type, sample_hash
                FROM column_patterns
                WHERE pattern_type = ?
            """, (pattern_type,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "table_name": row[0],
                "column_name": row[1],
                "pattern_type": row[2],
                "sample_hash": row[3]
            }
            for row in rows
        ]

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, value, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_last_training_time(self) -> Optional[str]:
        """Get the timestamp of the last training run."""
        return self.get_metadata("last_training_time")

    def set_last_training_time(self, timestamp: Optional[str] = None) -> None:
        """Set the timestamp of the last training run."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        self.set_metadata("last_training_time", timestamp)

    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge store."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM datasets")
        dataset_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT database_id) FROM datasets WHERE database_id IS NOT NULL")
        database_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT schema) FROM datasets WHERE schema IS NOT NULL")
        schema_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM join_patterns")
        join_pattern_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM synonyms")
        synonym_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM column_patterns")
        column_pattern_count = cursor.fetchone()[0]

        # Count total columns across all datasets
        cursor.execute("SELECT knowledge_json FROM datasets")
        total_columns = 0
        for row in cursor.fetchall():
            try:
                data = json.loads(row[0])
                total_columns += len(data.get("columns", {}))
            except json.JSONDecodeError:
                pass

        conn.close()

        return {
            "datasets": dataset_count,
            "databases": database_count,
            "schemas": schema_count,
            "total_columns": total_columns,
            "join_patterns": join_pattern_count,
            "synonyms": synonym_count,
            "column_patterns": column_pattern_count,
            "last_training": self.get_last_training_time(),
            "storage_path": self.storage_path
        }

    def clear_all(self) -> None:
        """Clear all stored knowledge (use with caution!)."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM datasets")
        cursor.execute("DELETE FROM join_patterns")
        cursor.execute("DELETE FROM synonyms")
        cursor.execute("DELETE FROM column_patterns")
        cursor.execute("DELETE FROM metadata")

        conn.commit()
        conn.close()

    def export_to_json(self, filepath: str) -> None:
        """Export all knowledge to a JSON file."""
        data = {
            "datasets": [ds.to_dict() for ds in self.get_all_datasets()],
            "join_patterns": [jp.to_dict() for jp in self.get_join_patterns(min_confidence=0.0)],
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat()
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def import_from_json(self, filepath: str) -> None:
        """Import knowledge from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ds_data in data.get("datasets", []):
            self.save_dataset(DatasetKnowledge.from_dict(ds_data))

        for jp_data in data.get("join_patterns", []):
            self.save_join_pattern(JoinPattern.from_dict(jp_data))

    # =========================================================================
    # Query Examples Methods (for few-shot learning)
    # =========================================================================

    def save_query_example(
        self,
        title: str,
        sql: str,
        normalized_sql: str,
        pattern_json: str,
        keywords: List[str],
        tables: List[str],
        source: str = "saved_query",
        source_id: Optional[int] = None,
        description: Optional[str] = None,
        schema_name: Optional[str] = None,
        database_id: Optional[int] = None,
        dialect: str = "starrocks",
        query_type: str = "simple",
        complexity_score: float = 0.0
    ) -> Optional[int]:
        """
        Save a query example for few-shot learning.

        Returns the example ID, or None if it's a duplicate.
        """
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO query_examples
                (source, source_id, title, description, sql, normalized_sql,
                 query_type, tables_json, pattern_json, keywords_json,
                 schema_name, database_id, dialect, complexity_score,
                 created_at, is_valid, last_validated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                source,
                source_id,
                title,
                description,
                sql,
                normalized_sql,
                query_type,
                json.dumps(tables),
                pattern_json,
                json.dumps(keywords),
                schema_name,
                database_id,
                dialect,
                complexity_score,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))

            conn.commit()
            example_id = cursor.lastrowid if cursor.rowcount > 0 else None
            return example_id

        finally:
            conn.close()

    def get_examples_by_tables(
        self,
        tables: List[str],
        limit: int = 10,
        valid_only: bool = True
    ) -> List[Dict]:
        """
        Get query examples that use any of the specified tables.

        Args:
            tables: List of table names to match (if empty, returns all examples)
            limit: Maximum number of examples
            valid_only: Only return valid examples

        Returns:
            List of example dicts
        """
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        # Handle empty tables list - return all examples
        if not tables:
            where_clause = "1=1"
            if valid_only:
                where_clause = "is_valid = 1"
            cursor.execute(f"""
                SELECT id, source, source_id, title, description, sql, normalized_sql,
                       query_type, tables_json, pattern_json, keywords_json,
                       schema_name, database_id, dialect, complexity_score,
                       usage_count, success_count, failure_count
                FROM query_examples
                WHERE {where_clause}
                ORDER BY (success_count - failure_count) DESC, usage_count DESC
                LIMIT ?
            """, (limit,))
        else:
            # Build OR conditions for table matching
            conditions = []
            params = []
            for table in tables:
                conditions.append("LOWER(tables_json) LIKE ?")
                params.append(f'%"{table.lower()}"%')

            where_clause = " OR ".join(conditions)
            if valid_only:
                where_clause = f"({where_clause}) AND is_valid = 1"

            cursor.execute(f"""
                SELECT id, source, source_id, title, description, sql, normalized_sql,
                       query_type, tables_json, pattern_json, keywords_json,
                       schema_name, database_id, dialect, complexity_score,
                       usage_count, success_count, failure_count
                FROM query_examples
                WHERE {where_clause}
                ORDER BY (success_count - failure_count) DESC, usage_count DESC
                LIMIT ?
            """, (*params, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_example_dict(row) for row in rows]

    def get_examples_by_keywords(
        self,
        keywords: List[str],
        limit: int = 10,
        valid_only: bool = True
    ) -> List[Dict]:
        """
        Get query examples matching any of the specified keywords.

        Args:
            keywords: List of keywords to match
            limit: Maximum number of examples
            valid_only: Only return valid examples

        Returns:
            List of example dicts
        """
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        conditions = []
        params = []
        for keyword in keywords:
            conditions.append("LOWER(keywords_json) LIKE ?")
            params.append(f'%"{keyword.lower()}"%')

        where_clause = " OR ".join(conditions)
        if valid_only:
            where_clause = f"({where_clause}) AND is_valid = 1"

        cursor.execute(f"""
            SELECT id, source, source_id, title, description, sql, normalized_sql,
                   query_type, tables_json, pattern_json, keywords_json,
                   schema_name, database_id, dialect, complexity_score,
                   usage_count, success_count, failure_count
            FROM query_examples
            WHERE {where_clause}
            ORDER BY (success_count - failure_count) DESC, usage_count DESC
            LIMIT ?
        """, (*params, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_example_dict(row) for row in rows]

    def get_examples_by_type(
        self,
        query_type: str,
        limit: int = 10,
        valid_only: bool = True
    ) -> List[Dict]:
        """Get examples of a specific query type."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        valid_clause = "AND is_valid = 1" if valid_only else ""

        cursor.execute(f"""
            SELECT id, source, source_id, title, description, sql, normalized_sql,
                   query_type, tables_json, pattern_json, keywords_json,
                   schema_name, database_id, dialect, complexity_score,
                   usage_count, success_count, failure_count
            FROM query_examples
            WHERE query_type = ? {valid_clause}
            ORDER BY (success_count - failure_count) DESC, usage_count DESC
            LIMIT ?
        """, (query_type, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_example_dict(row) for row in rows]

    def update_example_usage(
        self,
        example_id: int,
        success: bool
    ) -> None:
        """Update usage statistics for an example."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        if success:
            cursor.execute("""
                UPDATE query_examples
                SET usage_count = usage_count + 1,
                    success_count = success_count + 1,
                    last_used = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), example_id))
        else:
            cursor.execute("""
                UPDATE query_examples
                SET usage_count = usage_count + 1,
                    failure_count = failure_count + 1,
                    last_used = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), example_id))

        conn.commit()
        conn.close()

    def invalidate_example(self, example_id: int, reason: str = "") -> None:
        """Mark an example as invalid (schema changed, etc.)."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE query_examples
            SET is_valid = 0,
                last_validated = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), example_id))

        conn.commit()
        conn.close()

    def revalidate_examples(self, known_tables: Set[str]) -> int:
        """
        Revalidate all examples against current schema.

        Args:
            known_tables: Set of currently known table names

        Returns:
            Number of examples invalidated
        """
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, tables_json FROM query_examples WHERE is_valid = 1")
        rows = cursor.fetchall()

        invalidated = 0
        now = datetime.now().isoformat()

        for row in rows:
            example_id, tables_json = row
            try:
                tables = json.loads(tables_json)
                # Check if all tables still exist
                all_valid = all(t.lower() in known_tables for t in tables)

                if not all_valid:
                    cursor.execute("""
                        UPDATE query_examples
                        SET is_valid = 0, last_validated = ?
                        WHERE id = ?
                    """, (now, example_id))
                    invalidated += 1
                else:
                    cursor.execute("""
                        UPDATE query_examples
                        SET last_validated = ?
                        WHERE id = ?
                    """, (now, example_id))

            except json.JSONDecodeError:
                continue

        conn.commit()
        conn.close()

        return invalidated

    def record_query_feedback(
        self,
        natural_language: str,
        generated_sql: str,
        feedback_type: str,
        example_id: Optional[int] = None,
        corrected_sql: Optional[str] = None,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        row_count: Optional[int] = None
    ) -> None:
        """Record feedback for a query execution."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO query_feedback
            (example_id, natural_language, generated_sql, feedback_type,
             corrected_sql, error_message, execution_time_ms, row_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            example_id,
            natural_language,
            generated_sql,
            feedback_type,
            corrected_sql,
            error_message,
            execution_time_ms,
            row_count,
            datetime.now().isoformat()
        ))

        # Update example statistics if linked
        if example_id:
            if feedback_type == "success":
                cursor.execute("""
                    UPDATE query_examples
                    SET usage_count = usage_count + 1,
                        success_count = success_count + 1
                    WHERE id = ?
                """, (example_id,))
            elif feedback_type == "failure":
                cursor.execute("""
                    UPDATE query_examples
                    SET failure_count = failure_count + 1
                    WHERE id = ?
                """, (example_id,))

        conn.commit()
        conn.close()

    def has_similar_example(self, normalized_sql: str) -> bool:
        """Check if a similar example already exists."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM query_examples
            WHERE normalized_sql = ?
        """, (normalized_sql,))

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def get_example_count(self, valid_only: bool = True) -> int:
        """Get count of query examples."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        if valid_only:
            cursor.execute("SELECT COUNT(*) FROM query_examples WHERE is_valid = 1")
        else:
            cursor.execute("SELECT COUNT(*) FROM query_examples")

        count = cursor.fetchone()[0]
        conn.close()

        return count

    def get_all_examples(
        self,
        valid_only: bool = True,
        limit: int = 1000
    ) -> List[Dict]:
        """Get all query examples."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()

        valid_clause = "WHERE is_valid = 1" if valid_only else ""

        cursor.execute(f"""
            SELECT id, source, source_id, title, description, sql, normalized_sql,
                   query_type, tables_json, pattern_json, keywords_json,
                   schema_name, database_id, dialect, complexity_score,
                   usage_count, success_count, failure_count
            FROM query_examples
            {valid_clause}
            ORDER BY (success_count - failure_count) DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_example_dict(row) for row in rows]

    def _row_to_example_dict(self, row: tuple) -> Dict:
        """Convert a database row to example dict."""
        return {
            "id": row[0],
            "source": row[1],
            "source_id": row[2],
            "title": row[3],
            "description": row[4],
            "sql": row[5],
            "normalized_sql": row[6],
            "query_type": row[7],
            "tables": json.loads(row[8]) if row[8] else [],
            "pattern": json.loads(row[9]) if row[9] else {},
            "keywords": json.loads(row[10]) if row[10] else [],
            "schema_name": row[11],
            "database_id": row[12],
            "dialect": row[13],
            "complexity_score": row[14],
            "usage_count": row[15],
            "success_count": row[16],
            "failure_count": row[17],
        }