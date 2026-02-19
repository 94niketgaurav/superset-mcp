"""
PostgreSQL-based knowledge store for persisting learned dataset and column information.

This is the production-ready store using PostgreSQL instead of SQLite.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from dotenv import load_dotenv
load_dotenv()

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    from psycopg2.pool import ThreadedConnectionPool
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from .knowledge_store import (
    DatasetKnowledge,
    ColumnKnowledge,
    JoinPattern,
    KnowledgeStore as SQLiteKnowledgeStore
)


class PostgresKnowledgeStore:
    """
    PostgreSQL-based persistent store for learned knowledge about datasets.

    Features:
    - Connection pooling for better performance
    - JSONB storage for flexible schema
    - Full-text search capabilities
    - Better concurrent access than SQLite
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10
    ):
        """
        Initialize the PostgreSQL knowledge store.

        Args:
            host: PostgreSQL host (default: from POSTGRES_HOST env or localhost)
            port: PostgreSQL port (default: from POSTGRES_PORT env or 5432)
            database: Database name (default: from POSTGRES_DB env or superset_mcp)
            user: Database user (default: from POSTGRES_USER env or postgres)
            password: Database password (default: from POSTGRES_PASSWORD env)
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
        """
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2 is required. Install with: pip install psycopg2-binary")

        self.host = host or os.environ.get("POSTGRES_HOST", "localhost")
        self.port = port or int(os.environ.get("POSTGRES_PORT", "5432"))
        self.database = database or os.environ.get("POSTGRES_DB", "superset_mcp")
        self.user = user or os.environ.get("POSTGRES_USER", "postgres")
        self.password = password or os.environ.get("POSTGRES_PASSWORD", "")

        self._pool = ThreadedConnectionPool(
            min_connections,
            max_connections,
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )

    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _get_cursor(self, dict_cursor: bool = False):
        """Get a cursor from a pooled connection."""
        with self._get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def save_dataset(self, knowledge: DatasetKnowledge) -> None:
        """Save dataset knowledge to the store."""
        knowledge.last_updated = datetime.now().isoformat()
        knowledge_json = knowledge.to_dict()

        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO superset_mcp.datasets
                (dataset_id, table_name, schema_name, database_id, description,
                 knowledge_json, aliases, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dataset_id) DO UPDATE SET
                    table_name = EXCLUDED.table_name,
                    schema_name = EXCLUDED.schema_name,
                    database_id = EXCLUDED.database_id,
                    description = EXCLUDED.description,
                    knowledge_json = EXCLUDED.knowledge_json,
                    aliases = EXCLUDED.aliases,
                    last_updated = EXCLUDED.last_updated
            """, (
                knowledge.dataset_id,
                knowledge.table_name,
                knowledge.schema,
                knowledge.database_id,
                knowledge.description,
                Json(knowledge_json),
                knowledge.aliases,
                knowledge.last_updated
            ))

            # Also save columns to the columns table for faster queries
            for col_name, col_info in knowledge.columns.items():
                cursor.execute("""
                    INSERT INTO superset_mcp.columns
                    (dataset_id, column_name, column_type, is_nullable,
                     is_primary_key, is_foreign_key, fk_references,
                     distinct_count, null_count, sample_values)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dataset_id, column_name) DO UPDATE SET
                        column_type = EXCLUDED.column_type,
                        is_primary_key = EXCLUDED.is_primary_key,
                        is_foreign_key = EXCLUDED.is_foreign_key,
                        fk_references = EXCLUDED.fk_references,
                        distinct_count = EXCLUDED.distinct_count,
                        null_count = EXCLUDED.null_count,
                        sample_values = EXCLUDED.sample_values
                """, (
                    knowledge.dataset_id,
                    col_name,
                    col_info.column_type,
                    col_info.is_nullable,
                    col_info.is_primary_key,
                    col_info.is_foreign_key,
                    col_info.fk_references,
                    col_info.distinct_count,
                    col_info.null_count,
                    Json(col_info.sample_values) if col_info.sample_values else None
                ))

    def get_dataset(self, dataset_id: int) -> Optional[DatasetKnowledge]:
        """Get dataset knowledge by ID."""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute(
                "SELECT knowledge_json FROM superset_mcp.datasets WHERE dataset_id = %s",
                (dataset_id,)
            )
            row = cursor.fetchone()

            if row:
                return DatasetKnowledge.from_dict(row["knowledge_json"])
            return None

    def get_dataset_by_name(self, table_name: str) -> Optional[DatasetKnowledge]:
        """Get dataset knowledge by table name."""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute(
                "SELECT knowledge_json FROM superset_mcp.datasets WHERE LOWER(table_name) = LOWER(%s)",
                (table_name,)
            )
            row = cursor.fetchone()

            if row:
                return DatasetKnowledge.from_dict(row["knowledge_json"])
            return None

    def get_all_datasets(self) -> List[DatasetKnowledge]:
        """Get all dataset knowledge."""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute("SELECT knowledge_json FROM superset_mcp.datasets")
            rows = cursor.fetchall()
            return [DatasetKnowledge.from_dict(row["knowledge_json"]) for row in rows]

    def get_all_table_names(self) -> List[str]:
        """Get all known table names."""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT table_name FROM superset_mcp.datasets")
            return [row[0] for row in cursor.fetchall()]

    def get_datasets_by_database(self, database_id: int) -> List[DatasetKnowledge]:
        """Get all datasets for a specific database."""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute(
                "SELECT knowledge_json FROM superset_mcp.datasets WHERE database_id = %s",
                (database_id,)
            )
            rows = cursor.fetchall()
            return [DatasetKnowledge.from_dict(row["knowledge_json"]) for row in rows]

    def get_unique_databases(self) -> List[int]:
        """Get list of unique database IDs."""
        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT database_id FROM superset_mcp.datasets WHERE database_id IS NOT NULL"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_unique_schemas(self, database_id: Optional[int] = None) -> List[str]:
        """Get list of unique schemas."""
        with self._get_cursor() as cursor:
            if database_id is not None:
                cursor.execute(
                    "SELECT DISTINCT schema_name FROM superset_mcp.datasets WHERE database_id = %s AND schema_name IS NOT NULL",
                    (database_id,)
                )
            else:
                cursor.execute(
                    "SELECT DISTINCT schema_name FROM superset_mcp.datasets WHERE schema_name IS NOT NULL"
                )
            return [row[0] for row in cursor.fetchall()]

    def save_join_pattern(self, pattern: JoinPattern) -> None:
        """Save or update a join pattern."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO superset_mcp.join_patterns
                (left_table, left_column, right_table, right_column,
                 join_type, confidence, usage_count, last_used, value_overlap_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (left_table, left_column, right_table, right_column)
                DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    usage_count = superset_mcp.join_patterns.usage_count + 1,
                    last_used = EXCLUDED.last_used,
                    value_overlap_ratio = EXCLUDED.value_overlap_ratio
            """, (
                pattern.left_table, pattern.left_column,
                pattern.right_table, pattern.right_column,
                pattern.join_type, pattern.confidence,
                pattern.usage_count, pattern.last_used,
                pattern.value_overlap_ratio
            ))

    def get_join_patterns(
        self,
        table_name: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> List[JoinPattern]:
        """Get join patterns, optionally filtered by table."""
        with self._get_cursor(dict_cursor=True) as cursor:
            if table_name:
                cursor.execute("""
                    SELECT left_table, left_column, right_table, right_column,
                           join_type, confidence, usage_count, last_used, value_overlap_ratio
                    FROM superset_mcp.join_patterns
                    WHERE (LOWER(left_table) = LOWER(%s) OR LOWER(right_table) = LOWER(%s))
                      AND confidence >= %s
                    ORDER BY confidence DESC, usage_count DESC
                """, (table_name, table_name, min_confidence))
            else:
                cursor.execute("""
                    SELECT left_table, left_column, right_table, right_column,
                           join_type, confidence, usage_count, last_used, value_overlap_ratio
                    FROM superset_mcp.join_patterns
                    WHERE confidence >= %s
                    ORDER BY confidence DESC, usage_count DESC
                """, (min_confidence,))

            rows = cursor.fetchall()
            return [
                JoinPattern(
                    left_table=row["left_table"],
                    left_column=row["left_column"],
                    right_table=row["right_table"],
                    right_column=row["right_column"],
                    join_type=row["join_type"] or "inner",
                    confidence=float(row["confidence"] or 0.5),
                    usage_count=row["usage_count"] or 0,
                    last_used=str(row["last_used"]) if row["last_used"] else None,
                    value_overlap_ratio=float(row["value_overlap_ratio"]) if row["value_overlap_ratio"] else None
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
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO superset_mcp.synonyms (term, synonym, source, confidence)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (term, synonym) DO UPDATE SET
                    confidence = EXCLUDED.confidence
            """, (term.lower(), synonym.lower(), source, confidence))

    def get_synonyms(self, term: str) -> List[str]:
        """Get all synonyms for a term."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                SELECT synonym FROM superset_mcp.synonyms
                WHERE LOWER(term) = LOWER(%s)
                ORDER BY confidence DESC
            """, (term,))
            synonyms = [row[0] for row in cursor.fetchall()]

            # Also check reverse
            cursor.execute("""
                SELECT term FROM superset_mcp.synonyms
                WHERE LOWER(synonym) = LOWER(%s)
                ORDER BY confidence DESC
            """, (term,))
            synonyms.extend([row[0] for row in cursor.fetchall()])

            return list(set(synonyms))

    def save_column_pattern(
        self,
        table_name: str,
        column_name: str,
        pattern_type: str,
        sample_hash: str,
        sample_values: Optional[List] = None,
        statistics: Optional[Dict] = None
    ) -> None:
        """Save column pattern information for join detection."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO superset_mcp.column_patterns
                (table_name, column_name, pattern_type, sample_hash, sample_values, statistics, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (table_name, column_name) DO UPDATE SET
                    pattern_type = EXCLUDED.pattern_type,
                    sample_hash = EXCLUDED.sample_hash,
                    sample_values = EXCLUDED.sample_values,
                    statistics = EXCLUDED.statistics,
                    updated_at = EXCLUDED.updated_at
            """, (
                table_name, column_name, pattern_type, sample_hash,
                Json(sample_values) if sample_values else None,
                Json(statistics) if statistics else None,
                datetime.now().isoformat()
            ))

    def find_matching_columns(
        self,
        pattern_type: str,
        sample_hash: Optional[str] = None
    ) -> List[Dict]:
        """Find columns with matching patterns or sample hashes."""
        with self._get_cursor(dict_cursor=True) as cursor:
            if sample_hash:
                cursor.execute("""
                    SELECT table_name, column_name, pattern_type, sample_hash
                    FROM superset_mcp.column_patterns
                    WHERE pattern_type = %s OR sample_hash = %s
                """, (pattern_type, sample_hash))
            else:
                cursor.execute("""
                    SELECT table_name, column_name, pattern_type, sample_hash
                    FROM superset_mcp.column_patterns
                    WHERE pattern_type = %s
                """, (pattern_type,))

            return [dict(row) for row in cursor.fetchall()]

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value."""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute(
                "SELECT value FROM superset_mcp.metadata WHERE key = %s",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                value = row["value"]
                # Return as string for backward compatibility
                if isinstance(value, dict):
                    return json.dumps(value)
                return str(value) if value else None
            return None

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        # Try to parse as JSON, otherwise store as string
        try:
            json_value = json.loads(value)
        except json.JSONDecodeError:
            json_value = value

        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO superset_mcp.metadata (key, value, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
            """, (key, Json(json_value), datetime.now().isoformat()))

    def get_last_training_time(self) -> Optional[str]:
        """Get the timestamp of the last training run."""
        return self.get_metadata("last_training_time")

    def set_last_training_time(self, timestamp: Optional[str] = None) -> None:
        """Set the timestamp of the last training run."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        self.set_metadata("last_training_time", timestamp)

    def save_database_info(self, db_id: int, name: str, backend: str, schemas: List[str]) -> None:
        """Save database information."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO superset_mcp.databases (id, name, backend, schemas, last_synced)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    backend = EXCLUDED.backend,
                    schemas = EXCLUDED.schemas,
                    last_synced = EXCLUDED.last_synced
            """, (db_id, name, backend, schemas, datetime.now().isoformat()))

    def get_database_info(self) -> List[Dict]:
        """Get all database information."""
        with self._get_cursor(dict_cursor=True) as cursor:
            cursor.execute("SELECT * FROM superset_mcp.databases ORDER BY id")
            return [dict(row) for row in cursor.fetchall()]

    def save_query_history(
        self,
        natural_language_query: str,
        generated_sql: str,
        dataset_ids: List[int],
        was_executed: bool = False,
        was_successful: bool = None,
        user_feedback: str = None
    ) -> int:
        """Save a query to history for learning."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO superset_mcp.query_history
                (natural_language_query, generated_sql, dataset_ids,
                 was_executed, was_successful, user_feedback)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                natural_language_query, generated_sql, dataset_ids,
                was_executed, was_successful, user_feedback
            ))
            return cursor.fetchone()[0]

    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge store."""
        with self._get_cursor() as cursor:
            stats = {}

            cursor.execute("SELECT COUNT(*) FROM superset_mcp.datasets")
            stats["datasets"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT database_id) FROM superset_mcp.datasets WHERE database_id IS NOT NULL")
            stats["databases"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT schema_name) FROM superset_mcp.datasets WHERE schema_name IS NOT NULL")
            stats["schemas"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM superset_mcp.columns")
            stats["total_columns"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM superset_mcp.join_patterns")
            stats["join_patterns"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM superset_mcp.synonyms")
            stats["synonyms"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM superset_mcp.column_patterns")
            stats["column_patterns"] = cursor.fetchone()[0]

            stats["last_training"] = self.get_last_training_time()
            stats["storage_path"] = f"postgresql://{self.host}:{self.port}/{self.database}"

            return stats

    def clear_all(self) -> None:
        """Clear all stored knowledge (use with caution!)."""
        with self._get_cursor() as cursor:
            cursor.execute("TRUNCATE superset_mcp.datasets CASCADE")
            cursor.execute("TRUNCATE superset_mcp.join_patterns")
            cursor.execute("TRUNCATE superset_mcp.synonyms")
            cursor.execute("TRUNCATE superset_mcp.column_patterns")
            cursor.execute("TRUNCATE superset_mcp.metadata")
            cursor.execute("TRUNCATE superset_mcp.databases")
            cursor.execute("TRUNCATE superset_mcp.query_history")

    def export_to_json(self, filepath: str) -> None:
        """Export all knowledge to a JSON file."""
        data = {
            "datasets": [ds.to_dict() for ds in self.get_all_datasets()],
            "join_patterns": [jp.to_dict() for jp in self.get_join_patterns(min_confidence=0.0)],
            "databases": self.get_database_info(),
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

    def close(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()


def create_knowledge_store(use_postgres: bool = None) -> Any:
    """
    Factory function to create the appropriate knowledge store.

    Args:
        use_postgres: If True, use PostgreSQL. If None, auto-detect from env.

    Returns:
        PostgresKnowledgeStore or KnowledgeStore (SQLite)
    """
    if use_postgres is None:
        # Auto-detect based on environment
        use_postgres = bool(os.environ.get("POSTGRES_HOST") or os.environ.get("POSTGRES_DB"))

    if use_postgres:
        if not HAS_PSYCOPG2:
            print("WARNING: psycopg2 not installed, falling back to SQLite")
            return SQLiteKnowledgeStore()
        return PostgresKnowledgeStore()

    return SQLiteKnowledgeStore()