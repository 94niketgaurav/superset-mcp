"""
Dataset learner for automatically training the semantic matcher.

Learns:
- All datasets and their metadata
- Column information and sample values
- Join patterns based on column analysis
- Table name synonyms and aliases
- Database and schema organization

Supports:
- Superset API for registered datasets
- SQL queries for physical tables (StarRocks, PostgreSQL, MySQL)
- Lakekeeper (Iceberg REST catalog) for schema metadata fallback
"""
import os
import json
import time
import httpx
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from .knowledge_store import DatasetKnowledge, ColumnKnowledge, JoinPattern
from .column_analyzer import ColumnAnalyzer


# Lakekeeper configuration
LAKEKEEPER_URL = os.environ.get("LAKEKEEPER_URL", "").rstrip("/")
LAKEKEEPER_TOKEN = os.environ.get("LAKEKEEPER_TOKEN", "")
LAKEKEEPER_WAREHOUSE = os.environ.get("LAKEKEEPER_WAREHOUSE", "default")


class LakekeeperClient:
    """
    Client for Lakekeeper (Iceberg REST Catalog) to fetch table metadata.

    Lakekeeper provides a standard REST API for Iceberg table catalogs,
    which can be used as a fallback for schema discovery when SQL queries fail.
    """

    def __init__(self, base_url: str = None, token: str = None, warehouse: str = None):
        """Initialize the Lakekeeper client."""
        self.base_url = base_url or LAKEKEEPER_URL
        self.token = token or LAKEKEEPER_TOKEN
        self.warehouse = warehouse or LAKEKEEPER_WAREHOUSE
        self._http = httpx.Client(timeout=30.0, verify=False)

    @property
    def is_configured(self) -> bool:
        """Check if Lakekeeper is configured."""
        return bool(self.base_url)

    def _headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def list_namespaces(self) -> List[str]:
        """List all namespaces (schemas) in the catalog."""
        if not self.is_configured:
            return []

        try:
            url = f"{self.base_url}/v1/{self.warehouse}/namespaces"
            r = self._http.get(url, headers=self._headers())
            r.raise_for_status()
            data = r.json()
            # Namespaces come as arrays like [["schema1"], ["schema2"]]
            return [ns[0] if isinstance(ns, list) else ns for ns in data.get("namespaces", [])]
        except Exception as e:
            print(f"    Lakekeeper: Failed to list namespaces: {e}")
            return []

    def list_tables(self, namespace: str) -> List[str]:
        """List all tables in a namespace."""
        if not self.is_configured:
            return []

        try:
            url = f"{self.base_url}/v1/{self.warehouse}/namespaces/{namespace}/tables"
            r = self._http.get(url, headers=self._headers())
            r.raise_for_status()
            data = r.json()
            # Tables come as {"identifiers": [{"namespace": [...], "name": "table"}]}
            return [t.get("name") for t in data.get("identifiers", [])]
        except Exception as e:
            print(f"    Lakekeeper: Failed to list tables in {namespace}: {e}")
            return []

    def get_table_schema(self, namespace: str, table_name: str) -> List[Dict]:
        """
        Get column schema for a table.

        Returns a list of column dicts with: column_name, type, is_nullable, etc.
        """
        if not self.is_configured:
            return []

        try:
            url = f"{self.base_url}/v1/{self.warehouse}/namespaces/{namespace}/tables/{table_name}"
            r = self._http.get(url, headers=self._headers())
            r.raise_for_status()
            data = r.json()

            columns = []
            # Iceberg schema format
            schema = data.get("metadata", {}).get("current-schema", {})
            if not schema:
                schema = data.get("schema", {})

            for field_info in schema.get("fields", []):
                col_type = field_info.get("type", "string")
                # Handle complex types
                if isinstance(col_type, dict):
                    col_type = col_type.get("type", "struct")

                columns.append({
                    "column_name": field_info.get("name"),
                    "type": str(col_type),
                    "is_nullable": not field_info.get("required", False),
                    "column_key": "",
                    "is_dttm": any(t in str(col_type).lower() for t in ["date", "time", "timestamp"]),
                    "comment": field_info.get("doc", "")
                })

            return columns

        except Exception as e:
            print(f"    Lakekeeper: Failed to get schema for {namespace}.{table_name}: {e}")
            return []


# Global Lakekeeper client
_lakekeeper_client: Optional[LakekeeperClient] = None


def get_lakekeeper_client() -> LakekeeperClient:
    """Get or create the Lakekeeper client."""
    global _lakekeeper_client
    if _lakekeeper_client is None:
        _lakekeeper_client = LakekeeperClient()
    return _lakekeeper_client


def get_knowledge_store():
    """Get the appropriate knowledge store based on configuration."""
    postgres_host = os.environ.get("POSTGRES_HOST")
    postgres_db = os.environ.get("POSTGRES_DB")

    if postgres_host or postgres_db:
        try:
            from .pg_knowledge_store import PostgresKnowledgeStore
            return PostgresKnowledgeStore()
        except ImportError:
            pass

    from .knowledge_store import KnowledgeStore
    return KnowledgeStore()


@dataclass
class DatabaseInfo:
    """Information about a Superset database."""
    id: int
    name: str
    backend: str
    schemas: List[str] = field(default_factory=list)
    dataset_count: int = 0


@dataclass
class LearningProgress:
    """Progress information during learning."""
    total_datasets: int
    processed_datasets: int
    current_dataset: str
    current_database: str = ""
    current_schema: str = ""
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def progress_percent(self) -> float:
        if self.total_datasets == 0:
            return 100.0
        return (self.processed_datasets / self.total_datasets) * 100


class DatasetLearner:
    """
    Learn dataset information from Superset.

    This class connects to Superset, fetches all datasets,
    analyzes their columns, and stores the knowledge.
    """

    def __init__(
        self,
        knowledge_store = None,
        sample_size: int = 100,
        progress_callback: Optional[Callable[[LearningProgress], None]] = None
    ):
        """
        Initialize the dataset learner.

        Args:
            knowledge_store: Knowledge store to use. Creates default if None.
            sample_size: Number of sample values to fetch per column.
            progress_callback: Optional callback for progress updates.
        """
        self.knowledge_store = knowledge_store or get_knowledge_store()
        self.sample_size = sample_size
        self.progress_callback = progress_callback
        self.column_analyzer = ColumnAnalyzer()
        self._superset_client = None
        self._databases: Dict[int, DatabaseInfo] = {}

    def _get_superset_client(self):
        """Get or create Superset client."""
        if self._superset_client is None:
            # Import here to avoid circular imports
            from mcp_superset_server import client
            self._superset_client = client
        return self._superset_client

    def _login_and_verify(self) -> bool:
        """Login to Superset and verify connection."""
        client = self._get_superset_client()
        try:
            client.login()
            return True
        except Exception as e:
            print(f"ERROR: Failed to login to Superset: {e}")
            return False

    def _fetch_all_databases(self) -> List[DatabaseInfo]:
        """Fetch all databases from Superset."""
        client = self._get_superset_client()

        databases = []
        page = 0
        page_size = 100

        while True:
            params = {"page": page, "page_size": page_size}
            data = client.get("/api/v1/database/", params=params)

            for row in data.get("result", []):
                db_info = DatabaseInfo(
                    id=row.get("id"),
                    name=row.get("database_name", "unknown"),
                    backend=row.get("backend", "unknown")
                )
                databases.append(db_info)
                self._databases[db_info.id] = db_info

            if len(data.get("result", [])) < page_size:
                break
            page += 1

        return databases

    def _fetch_database_schemas(self, database_id: int) -> List[str]:
        """Fetch all schemas for a database."""
        client = self._get_superset_client()
        try:
            data = client.get(f"/api/v1/database/{database_id}/schemas/")
            return data.get("result", [])
        except Exception as e:
            print(f"    Warning: Could not fetch schemas for database {database_id}: {e}")
            return []

    def _fetch_all_datasets(self) -> List[Dict]:
        """Fetch all datasets registered in Superset."""
        client = self._get_superset_client()

        all_datasets = []
        page = 0
        page_size = 100

        while True:
            params = {"page": page, "page_size": page_size}
            data = client.get("/api/v1/dataset/", params=params)

            results = data.get("result", [])
            for row in results:
                db = row.get("database")
                db_id = db.get("id") if isinstance(db, dict) else db
                db_name = db.get("database_name") if isinstance(db, dict) else None

                all_datasets.append({
                    "id": row.get("id"),
                    "database_id": db_id,
                    "database_name": db_name,
                    "schema": row.get("schema"),
                    "table_name": row.get("table_name"),
                    "datasource_name": row.get("datasource_name"),
                    "is_registered": True,  # This is a registered dataset
                })

            if len(results) < page_size:
                break
            page += 1

        return all_datasets

    def _fetch_all_tables_from_database(self, database_id: int, schema: str) -> List[Dict]:
        """Fetch ALL tables from a database schema (not just registered datasets)."""
        client = self._get_superset_client()
        tables = []

        try:
            # Use the tables API to get all tables in the schema
            params = {"q": f"(schema_name:{schema})"}
            data = client.get(f"/api/v1/database/{database_id}/tables/", params=params)

            db_info = self._databases.get(database_id)
            db_name = db_info.name if db_info else f"Database {database_id}"

            for table in data.get("result", []):
                table_name = table.get("value")
                if table_name:
                    tables.append({
                        "id": None,  # No dataset ID - not registered
                        "database_id": database_id,
                        "database_name": db_name,
                        "schema": schema,
                        "table_name": table_name,
                        "table_type": table.get("type", "table"),
                        "is_registered": False,
                    })

        except Exception as e:
            print(f"    Warning: Could not fetch tables for {schema}: {e}")

        return tables

    def _fetch_table_columns(self, database_id: int, schema: str, table_name: str) -> List[Dict]:
        """Fetch columns for a specific table using SQL or API."""
        client = self._get_superset_client()
        columns = []

        # First try the table_metadata API (most reliable)
        try:
            params = {"q": json.dumps({"table_name": table_name, "schema_name": schema})}
            metadata = client.get(f"/api/v1/database/{database_id}/table_metadata/", params=params)
            cols = metadata.get("columns", [])
            if cols:
                for col in cols:
                    col_name = col.get("column_name") or col.get("name", "")
                    col_type = col.get("type") or col.get("type_generic", "VARCHAR")
                    if col_name:
                        columns.append({
                            "column_name": col_name,
                            "type": str(col_type),
                            "is_nullable": col.get("nullable", "YES"),
                            "column_key": "",
                            "is_dttm": any(t in str(col_type).lower() for t in ["date", "time", "timestamp"]),
                        })
                if columns:
                    return columns
        except Exception as e:
            pass  # Fall back to SQL

        # Get database backend to choose right SQL syntax
        db_info = self._databases.get(database_id)
        backend = db_info.backend.lower() if db_info else "unknown"

        # Try different SQL dialects for column information
        sql_queries = []

        if backend in ("starrocks", "mysql", "mariadb"):
            # StarRocks/MySQL uses backticks and DESCRIBE
            sql_queries = [
                f"DESCRIBE `{schema}`.`{table_name}`",
                f"SHOW COLUMNS FROM `{schema}`.`{table_name}`",
                f"SHOW FULL COLUMNS FROM `{schema}`.`{table_name}`",
            ]
        elif backend in ("postgresql", "postgres", "redshift"):
            sql_queries = [
                f"""SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = '{schema}'
                      AND table_name = '{table_name}'
                    ORDER BY ordinal_position""",
            ]
        else:
            # Try all variants
            sql_queries = [
                f"DESCRIBE `{schema}`.`{table_name}`",
                f"SHOW COLUMNS FROM `{schema}`.`{table_name}`",
                f"""SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = '{schema}'
                      AND table_name = '{table_name}'
                    ORDER BY ordinal_position""",
            ]

        for sql in sql_queries:
            try:
                payload = {
                    "database_id": database_id,
                    "sql": sql.strip(),
                    "runAsync": False,
                    "queryLimit": 500,
                }

                result = client.post("/api/v1/sqllab/execute/", payload, needs_csrf=True)

                # Check for errors in response
                if result.get("status") == "error" or result.get("errors"):
                    continue

                # Check for results - handle both sync and async responses
                data = result.get("data", [])
                if not data:
                    # Try to get from query result
                    query_id = result.get("query_id") or result.get("query", {}).get("id")
                    if query_id:
                        import time
                        for _ in range(5):  # Wait up to 5 seconds
                            time.sleep(1)
                            try:
                                query_result = client.get(f"/api/v1/query/{query_id}")
                                if query_result.get("status") == "success":
                                    data = query_result.get("data", [])
                                    break
                            except:
                                pass

                if data:
                    for row in data:
                        if not row:
                            continue

                        # Handle different result formats:
                        # Dict format (StarRocks DESCRIBE): {'Field': 'col', 'Type': 'varchar', ...}
                        # List format: [col_name, col_type, is_nullable, key, ...]
                        # info_schema: column_name, data_type, is_nullable

                        if isinstance(row, dict):
                            # StarRocks returns dict with Field, Type keys
                            col_name = row.get('Field') or row.get('column_name') or row.get('name', '')
                            col_type = row.get('Type') or row.get('data_type') or row.get('type', 'VARCHAR')
                            is_nullable = row.get('Null') or row.get('is_nullable', 'YES')
                            col_key = row.get('Key') or row.get('column_key', '')
                        elif isinstance(row, (list, tuple)) and len(row) >= 1:
                            col_name = str(row[0]) if row[0] else ""
                            col_type = str(row[1]) if len(row) > 1 and row[1] else "VARCHAR"
                            is_nullable = str(row[2]) if len(row) > 2 else "YES"
                            col_key = str(row[3]) if len(row) > 3 else ""
                        else:
                            continue

                        col_name = str(col_name)
                        col_type = str(col_type)

                        if col_name and col_name not in ("Field", "column_name"):  # Skip headers
                            columns.append({
                                "column_name": col_name,
                                "type": col_type,
                                "is_nullable": str(is_nullable),
                                "column_key": str(col_key),
                                "is_dttm": any(t in col_type.lower() for t in ["date", "time", "timestamp"]),
                            })

                    if columns:
                        return columns

            except Exception as e:
                continue  # Try next SQL format

        # Fallback to Lakekeeper if SQL queries failed and Lakekeeper is configured
        if not columns:
            lakekeeper = get_lakekeeper_client()
            if lakekeeper.is_configured:
                print(f"        Trying Lakekeeper for {schema}.{table_name}...", end="")
                lk_columns = lakekeeper.get_table_schema(schema, table_name)
                if lk_columns:
                    columns = lk_columns
                    print(f" found {len(columns)} columns")
                else:
                    print(" no data")

        return columns

    def _fetch_columns_from_lakekeeper_all(self, schema: str) -> Dict[str, List[Dict]]:
        """
        Fetch all table schemas from Lakekeeper for a given namespace.

        This is more efficient than fetching one table at a time.

        Args:
            schema: The namespace/schema to fetch

        Returns:
            Dict mapping table_name -> list of column dicts
        """
        lakekeeper = get_lakekeeper_client()
        if not lakekeeper.is_configured:
            return {}

        result = {}
        tables = lakekeeper.list_tables(schema)

        for table_name in tables:
            columns = lakekeeper.get_table_schema(schema, table_name)
            if columns:
                result[table_name] = columns

        return result

    def _fetch_all_tables_all_schemas(self) -> List[Dict]:
        """Fetch ALL tables from ALL schemas in ALL databases."""
        all_tables = []
        registered_datasets = self._fetch_all_datasets()

        # Create a set of registered table keys for quick lookup
        registered_keys = set()
        for ds in registered_datasets:
            key = f"{ds.get('database_id')}:{ds.get('schema')}:{ds.get('table_name')}".lower()
            registered_keys.add(key)

        # Add registered datasets first
        all_tables.extend(registered_datasets)

        # Now fetch all tables from each database schema
        for db_id, db_info in self._databases.items():
            print(f"    Scanning database: {db_info.name}")

            for schema in db_info.schemas:
                # Skip system schemas
                if schema in ('information_schema', 'pg_catalog', 'sys'):
                    continue

                tables = self._fetch_all_tables_from_database(db_id, schema)

                for table in tables:
                    key = f"{table.get('database_id')}:{table.get('schema')}:{table.get('table_name')}".lower()

                    # Only add if not already registered
                    if key not in registered_keys:
                        all_tables.append(table)
                        registered_keys.add(key)

                print(f"      Schema {schema}: {len(tables)} tables found")

        # Also fetch tables from Lakekeeper (Iceberg catalog) if configured
        lakekeeper = get_lakekeeper_client()
        if lakekeeper.is_configured:
            print(f"    Scanning Lakekeeper catalog...")
            lk_namespaces = lakekeeper.list_namespaces()

            for namespace in lk_namespaces:
                # Skip system namespaces
                if namespace in ('system', 'information_schema'):
                    continue

                lk_tables = lakekeeper.list_tables(namespace)
                lk_added = 0

                for table_name in lk_tables:
                    # Use database_id 0 for Lakekeeper tables (no specific Superset database)
                    key = f"0:{namespace}:{table_name}".lower()

                    if key not in registered_keys:
                        all_tables.append({
                            "id": None,
                            "database_id": 0,  # Lakekeeper tables don't have a Superset database
                            "database_name": "Lakekeeper",
                            "schema": namespace,
                            "table_name": table_name,
                            "table_type": "iceberg",
                            "is_registered": False,
                            "source": "lakekeeper"
                        })
                        registered_keys.add(key)
                        lk_added += 1

                if lk_added > 0:
                    print(f"      Lakekeeper namespace {namespace}: {lk_added} tables added")

        return all_tables

    def _organize_by_database_schema(self, datasets: List[Dict]) -> Dict[int, Dict[str, List[Dict]]]:
        """
        Organize datasets by database_id -> schema -> datasets.

        Returns:
            Dict[database_id, Dict[schema, List[dataset]]]
        """
        organized = defaultdict(lambda: defaultdict(list))

        for ds in datasets:
            db_id = ds.get("database_id") or 0
            schema = ds.get("schema") or "default"
            organized[db_id][schema].append(ds)

        return organized

    def print_database_hierarchy(self, datasets: List[Dict]) -> None:
        """Print a hierarchical view of databases, schemas, and tables."""
        organized = self._organize_by_database_schema(datasets)

        print("\n" + "=" * 60)
        print("SUPERSET DATABASE HIERARCHY")
        print("=" * 60)

        for db_id, schemas in sorted(organized.items()):
            db_info = self._databases.get(db_id)
            db_name = db_info.name if db_info else f"Database {db_id}"
            backend = db_info.backend if db_info else "unknown"

            total_tables = sum(len(tables) for tables in schemas.values())
            print(f"\n📦 {db_name} (ID: {db_id}, Backend: {backend})")
            print(f"   └── {len(schemas)} schema(s), {total_tables} table(s)")

            for schema_name, tables in sorted(schemas.items()):
                print(f"       📁 {schema_name or '(default)'} ({len(tables)} tables)")
                # Show first few tables
                for i, table in enumerate(sorted(tables, key=lambda t: t.get("table_name", ""))):
                    if i < 5:
                        print(f"          └── {table.get('table_name')}")
                    elif i == 5:
                        remaining = len(tables) - 5
                        print(f"          └── ... and {remaining} more")
                        break

        print("\n" + "=" * 60)

    def _fetch_dataset_details(self, dataset_id: int) -> Dict:
        """Fetch detailed dataset information including columns."""
        client = self._get_superset_client()
        data = client.get(f"/api/v1/dataset/{dataset_id}")
        return data.get("result", data)

    def _fetch_sample_values(
        self,
        database_id: int,
        table_name: str,
        column_name: str,
        schema: Optional[str] = None
    ) -> List[Any]:
        """Fetch sample values for a column."""
        client = self._get_superset_client()

        # Build qualified table name
        if schema:
            qualified_table = f'"{schema}"."{table_name}"'
        else:
            qualified_table = f'"{table_name}"'

        # Build sample query
        sql = f"""
            SELECT DISTINCT "{column_name}"
            FROM {qualified_table}
            WHERE "{column_name}" IS NOT NULL
            LIMIT {self.sample_size}
        """

        try:
            payload = {
                "database_id": database_id,
                "sql": sql,
                "runAsync": False,
                "queryLimit": self.sample_size,
            }
            result = client.post("/api/v1/sqllab/execute/", payload, needs_csrf=True)

            # Get results
            client_id = result.get("client_id") or result.get("result", {}).get("client_id")
            if client_id:
                results = client.get(
                    "/api/v1/sqllab/results/",
                    params={"q": json.dumps({"client_id": client_id})}
                )
                data = results.get("data", [])
                return [row[0] if row else None for row in data]

        except Exception as e:
            # Silently fail - sample values are optional
            pass

        return []

    def learn_dataset(
        self,
        dataset_info: Dict,
        fetch_samples: bool = True
    ) -> DatasetKnowledge:
        """
        Learn information about a single dataset or table.

        Args:
            dataset_info: Basic dataset/table info from list
            fetch_samples: Whether to fetch sample values

        Returns:
            DatasetKnowledge object
        """
        dataset_id = dataset_info.get("id")  # May be None for unregistered tables
        table_name = dataset_info.get("table_name", "")
        schema = dataset_info.get("schema")
        database_id = dataset_info.get("database_id")
        is_registered = dataset_info.get("is_registered", True)
        source = dataset_info.get("source", "superset")

        # Fetch columns - use different methods based on source
        raw_columns = []

        if source == "lakekeeper":
            # Table from Lakekeeper catalog - use Lakekeeper API directly
            lakekeeper = get_lakekeeper_client()
            if lakekeeper.is_configured:
                raw_columns = lakekeeper.get_table_schema(schema, table_name)
        elif is_registered and dataset_id:
            # Use dataset API for registered Superset datasets
            details = self._fetch_dataset_details(dataset_id)
            raw_columns = details.get("columns", [])
        else:
            # Use SQL to get columns for unregistered tables (with Lakekeeper fallback)
            raw_columns = self._fetch_table_columns(database_id, schema, table_name)

        # Generate a pseudo dataset_id for unregistered tables
        if not dataset_id:
            # Use hash of database_id:schema:table_name (limited to fit in INT)
            import hashlib
            key = f"{database_id}:{schema}:{table_name}"
            # Use modulo to keep within PostgreSQL integer range (max 2147483647)
            dataset_id = int(hashlib.md5(key.encode()).hexdigest()[:7], 16) % 2000000000

        # Process columns
        columns: Dict[str, ColumnKnowledge] = {}

        for col in raw_columns:
            col_name = col.get("column_name", "")
            if not col_name:
                continue

            col_type = col.get("type", "")
            is_dttm = col.get("is_dttm", False)

            # Fetch sample values if enabled
            sample_values = []
            if fetch_samples and database_id:
                sample_values = self._fetch_sample_values(
                    database_id, table_name, col_name, schema
                )

            # Analyze column
            analysis = self.column_analyzer.analyze_column(col_name, col_type, sample_values)

            # Infer FK references
            fk_ref = None
            if self.column_analyzer.is_id_column(col_name, sample_values):
                known_tables = self.knowledge_store.get_all_table_names()
                inferred = self.column_analyzer.infer_foreign_key(col_name, known_tables)
                if inferred:
                    fk_ref = f"{inferred['table']}.{inferred['column']}"

            columns[col_name] = ColumnKnowledge(
                column_name=col_name,
                column_type=col_type,
                is_primary_key=col_name.lower() in ("id", "uuid", "pk"),
                is_foreign_key=fk_ref is not None,
                fk_references=fk_ref,
                sample_values=sample_values[:20],  # Store limited samples
                distinct_count=analysis["statistics"].get("distinct_count"),
                null_count=analysis["statistics"].get("null_count"),
                value_patterns=[analysis["pattern"]["type"]] if analysis.get("pattern") else []
            )

            # Store column pattern for join detection
            if analysis.get("pattern"):
                self.knowledge_store.save_column_pattern(
                    table_name=table_name,
                    column_name=col_name,
                    pattern_type=analysis["pattern"]["type"],
                    sample_hash=analysis.get("value_hash", "")
                )

        # Create knowledge object
        aliases = self._generate_aliases(table_name)
        knowledge = DatasetKnowledge(
            dataset_id=dataset_id,
            table_name=table_name,
            schema=schema,
            database_id=database_id,
            description=None,  # Not available for unregistered tables
            columns=columns,
            aliases=aliases,
            last_updated=datetime.now().isoformat()
        )

        # Save synonyms for this table
        self._save_table_synonyms(table_name, aliases)

        return knowledge

    def _generate_aliases(self, table_name: str) -> List[str]:
        """Generate common aliases for a table name."""
        aliases = []
        name_lower = table_name.lower()

        # Singular/plural
        if name_lower.endswith('s'):
            aliases.append(name_lower[:-1])
        elif name_lower.endswith('ies'):
            aliases.append(name_lower[:-3] + 'y')
        else:
            aliases.append(name_lower + 's')

        # Without common prefixes
        prefixes = ["tbl_", "t_", "vw_", "dim_", "fact_", "stg_", "raw_"]
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                aliases.append(name_lower[len(prefix):])

        # Split by underscore and use parts
        if '_' in name_lower:
            parts = name_lower.split('_')
            if len(parts) >= 2:
                # First part as alias (e.g., "user_details" -> "user")
                aliases.append(parts[0])
                # Last part as alias (e.g., "user_details" -> "details")
                aliases.append(parts[-1])

        # CamelCase splitting
        import re
        camel_parts = re.findall(r'[A-Z][a-z]+|[a-z]+', table_name)
        if len(camel_parts) >= 2:
            for part in camel_parts:
                if len(part) > 2:
                    aliases.append(part.lower())

        return list(set(aliases))

    def _save_table_synonyms(self, table_name: str, aliases: List[str]) -> None:
        """Save table synonyms to the knowledge store."""
        for alias in aliases:
            if alias != table_name.lower():
                self.knowledge_store.add_synonym(alias, table_name, source="learned", confidence=0.9)

    def learn_synonyms_from_tables(self) -> int:
        """
        Generate and save synonyms based on all learned table names.

        This creates associations like:
        - "asset" -> "assets", "assetversion", "assetpin", etc.
        - "user" -> "users", "user_details", "user_attribute", etc.

        Returns:
            Number of synonyms created
        """
        all_datasets = self.knowledge_store.get_all_datasets()
        table_names = [ds.table_name.lower() for ds in all_datasets]

        synonym_count = 0

        # Group tables by common prefixes
        prefix_groups: Dict[str, List[str]] = defaultdict(list)
        for table in table_names:
            # Extract base name (first part before _ or camelCase)
            import re
            parts = re.split(r'[_]', table)
            if parts:
                base = parts[0]
                if len(base) >= 3:
                    prefix_groups[base].append(table)

            # Also group by camelCase prefix
            camel_parts = re.findall(r'[A-Z][a-z]+|[a-z]+', table)
            if camel_parts:
                base = camel_parts[0].lower()
                if len(base) >= 3:
                    prefix_groups[base].append(table)

        # Save synonyms for related tables
        for base, tables in prefix_groups.items():
            if len(tables) > 1:
                for table in tables:
                    # Create synonym: base -> table
                    self.knowledge_store.add_synonym(base, table, source="learned", confidence=0.85)
                    synonym_count += 1

                    # Also create synonyms between related tables
                    for other_table in tables:
                        if table != other_table:
                            # Lower confidence for cross-table synonyms
                            self.knowledge_store.add_synonym(table, other_table, source="inferred", confidence=0.6)

        return synonym_count

    def learn_join_patterns(self) -> List[JoinPattern]:
        """
        Analyze all datasets to discover join patterns.

        This should be called after learning all datasets.
        """
        patterns = []
        all_datasets = self.knowledge_store.get_all_datasets()

        # Build index of ID columns with their values
        id_columns: List[Dict] = []
        for ds in all_datasets:
            for col_name, col_info in ds.columns.items():
                if col_info.is_primary_key or col_info.is_foreign_key or \
                   self.column_analyzer.is_id_column(col_name):
                    id_columns.append({
                        "table_name": ds.table_name,
                        "column_name": col_name,
                        "values": col_info.sample_values,
                        "is_pk": col_info.is_primary_key,
                        "is_fk": col_info.is_foreign_key,
                        "fk_ref": col_info.fk_references
                    })

        # Find join patterns based on value overlap
        for i, col1 in enumerate(id_columns):
            if not col1["values"]:
                continue

            # Look for explicit FK references
            if col1["fk_ref"]:
                ref_table, ref_col = col1["fk_ref"].split(".", 1)
                patterns.append(JoinPattern(
                    left_table=col1["table_name"],
                    left_column=col1["column_name"],
                    right_table=ref_table,
                    right_column=ref_col,
                    join_type="left",
                    confidence=0.95,
                    value_overlap_ratio=None
                ))
                continue

            # Look for value-based matches
            candidates = self.column_analyzer.find_join_candidates(
                col1["column_name"],
                col1["values"],
                [c for j, c in enumerate(id_columns) if j != i and c["table_name"] != col1["table_name"]]
            )

            for candidate in candidates[:3]:  # Top 3 candidates
                if candidate["confidence"] >= 0.3:
                    patterns.append(JoinPattern(
                        left_table=col1["table_name"],
                        left_column=col1["column_name"],
                        right_table=candidate["table_name"],
                        right_column=candidate["column_name"],
                        join_type="inner" if col1["is_pk"] or candidate.get("is_pk") else "left",
                        confidence=candidate["confidence"],
                        value_overlap_ratio=candidate["overlap_ratio"]
                    ))

        # Save patterns
        for pattern in patterns:
            self.knowledge_store.save_join_pattern(pattern)

        return patterns

    def learn_all(
        self,
        fetch_samples: bool = True,
        learn_joins: bool = True,
        show_hierarchy: bool = True
    ) -> Dict[str, Any]:
        """
        Learn all datasets from Superset.

        This method:
        1. Logs into Superset automatically
        2. Fetches all databases and their schemas
        3. Lists all datasets organized by database/schema
        4. Learns column information and patterns
        5. Discovers join patterns

        Args:
            fetch_samples: Whether to fetch sample values for columns
            learn_joins: Whether to analyze and learn join patterns
            show_hierarchy: Whether to display database hierarchy

        Returns:
            Dict with learning statistics
        """
        start_time = time.time()
        errors = []

        # Step 1: Login to Superset
        print("\n[Step 1] Logging into Superset...")
        if not self._login_and_verify():
            return {
                "datasets_processed": 0,
                "datasets_successful": 0,
                "errors": ["Failed to login to Superset"],
                "join_patterns_found": 0,
                "elapsed_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
        print("  ✓ Successfully authenticated with Superset")

        # Step 2: Fetch all databases
        print("\n[Step 2] Discovering databases...")
        databases = self._fetch_all_databases()
        print(f"  ✓ Found {len(databases)} database(s)")

        for db in databases:
            print(f"    • {db.name} (ID: {db.id}, Backend: {db.backend})")
            # Fetch schemas for each database
            schemas = self._fetch_database_schemas(db.id)
            db.schemas = schemas
            if schemas:
                print(f"      Schemas: {', '.join(schemas[:5])}{'...' if len(schemas) > 5 else ''}")

        # Step 3: Fetch all tables (registered datasets + all database tables)
        print("\n[Step 3] Fetching all tables from all databases...")
        print("  Scanning for registered datasets...")
        registered_datasets = self._fetch_all_datasets()
        print(f"    Found {len(registered_datasets)} registered dataset(s)")

        print("  Scanning for ALL tables in database schemas...")
        all_datasets = self._fetch_all_tables_all_schemas()
        total = len(all_datasets)
        print(f"  ✓ Total tables discovered: {total} ({len(registered_datasets)} registered, {total - len(registered_datasets)} unregistered)")

        # Organize and display hierarchy
        if show_hierarchy:
            self.print_database_hierarchy(all_datasets)

        # Step 4: Learn each dataset
        print("\n[Step 4] Learning dataset metadata and columns...")
        organized = self._organize_by_database_schema(all_datasets)

        progress = LearningProgress(
            total_datasets=total,
            processed_datasets=0,
            current_dataset="",
            errors=[],
            start_time=start_time
        )

        processed_count = 0
        for db_id, schemas in sorted(organized.items()):
            db_info = self._databases.get(db_id)
            db_name = db_info.name if db_info else f"Database {db_id}"

            print(f"\n  Processing database: {db_name}")

            for schema_name, datasets in sorted(schemas.items()):
                print(f"    Schema: {schema_name or '(default)'} ({len(datasets)} tables)")

                for ds_info in datasets:
                    table_name = ds_info.get("table_name", "unknown")
                    progress.current_dataset = table_name
                    progress.current_database = db_name
                    progress.current_schema = schema_name or ""
                    progress.processed_datasets = processed_count

                    if self.progress_callback:
                        self.progress_callback(progress)

                    try:
                        processed_count += 1
                        print(f"      [{processed_count}/{total}] Learning: {table_name}", end="")
                        knowledge = self.learn_dataset(ds_info, fetch_samples=fetch_samples)
                        self.knowledge_store.save_dataset(knowledge)
                        col_count = len(knowledge.columns)
                        print(f" ({col_count} columns)")

                    except Exception as e:
                        error_msg = f"Error learning {db_name}.{schema_name}.{table_name}: {str(e)}"
                        errors.append(error_msg)
                        progress.errors.append(error_msg)
                        print(f" - Error: {e}")

        # Step 5: Learn join patterns
        join_patterns = []
        if learn_joins:
            print("\n[Step 5] Analyzing join patterns...")
            join_patterns = self.learn_join_patterns()
            print(f"  ✓ Found {len(join_patterns)} potential join patterns")

            # Show top join patterns
            if join_patterns:
                print("\n  Top join patterns discovered:")
                for jp in sorted(join_patterns, key=lambda x: x.confidence, reverse=True)[:10]:
                    print(f"    • {jp.left_table}.{jp.left_column} → {jp.right_table}.{jp.right_column} "
                          f"(confidence: {jp.confidence:.2f})")

        # Step 6: Learn synonyms from table names
        print("\n[Step 6] Learning synonyms from table names...")
        synonym_count = self.learn_synonyms_from_tables()
        print(f"  ✓ Created {synonym_count} synonym relationships")

        # Update metadata
        self.knowledge_store.set_last_training_time()

        # Store database info in metadata
        db_summary = {db.id: {"name": db.name, "backend": db.backend, "schemas": db.schemas}
                      for db in databases}
        self.knowledge_store.set_metadata("databases", json.dumps(db_summary))

        elapsed = time.time() - start_time
        stats = {
            "databases_found": len(databases),
            "datasets_processed": total,
            "datasets_successful": total - len(errors),
            "errors": errors,
            "join_patterns_found": len(join_patterns),
            "synonyms_created": synonym_count,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }

        print(f"\n" + "=" * 60)
        print("LEARNING COMPLETE")
        print("=" * 60)
        print(f"  Databases: {stats['databases_found']}")
        print(f"  Tables: {stats['datasets_successful']}/{total}")
        print(f"  Join patterns: {stats['join_patterns_found']}")
        print(f"  Synonyms: {stats['synonyms_created']}")
        print(f"  Errors: {len(errors)}")
        print(f"  Time: {elapsed:.1f}s")

        return stats

    def incremental_learn(self) -> Dict[str, Any]:
        """
        Perform incremental learning - only learn new/updated datasets.

        Compares timestamps to detect changes.
        """
        # Get last training time
        last_training = self.knowledge_store.get_last_training_time()

        # For now, just do full learning
        # TODO: Implement proper incremental learning with change detection
        return self.learn_all(fetch_samples=True, learn_joins=True)

    def export_for_semantic_matcher(self) -> Dict[str, Any]:
        """
        Export learned knowledge in a format suitable for semantic matcher.

        Returns:
            Dict with datasets, synonyms, and join patterns
        """
        all_datasets = self.knowledge_store.get_all_datasets()
        join_patterns = self.knowledge_store.get_join_patterns(min_confidence=0.5)

        # Build dataset list for matcher
        datasets = []
        for ds in all_datasets:
            datasets.append({
                "id": ds.dataset_id,
                "table_name": ds.table_name,
                "schema": ds.schema,
                "database_id": ds.database_id,
                "aliases": ds.aliases,
                "columns": list(ds.columns.keys()),
                "id_columns": [
                    col_name for col_name, col in ds.columns.items()
                    if col.is_primary_key or col.is_foreign_key
                ]
            })

        # Build synonym map
        synonyms = {}
        for ds in all_datasets:
            for alias in ds.aliases:
                if alias not in synonyms:
                    synonyms[alias] = []
                synonyms[alias].append(ds.table_name)

        # Build join map
        joins = {}
        for jp in join_patterns:
            key = f"{jp.left_table}:{jp.right_table}"
            if key not in joins:
                joins[key] = []
            joins[key].append({
                "left_column": jp.left_column,
                "right_column": jp.right_column,
                "join_type": jp.join_type,
                "confidence": jp.confidence
            })

        return {
            "datasets": datasets,
            "synonyms": synonyms,
            "join_patterns": joins,
            "exported_at": datetime.now().isoformat()
        }