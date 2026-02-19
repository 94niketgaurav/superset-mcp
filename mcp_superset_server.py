"""
MCP Superset Server - Central hub for all Superset operations.

This server provides tools for:
1. Training - Learning dataset schemas, joins, synonyms
2. Query Building - Semantic matching, join discovery
3. SQL Generation - LLM-based SQL generation with validation
4. Execution - Running queries and fetching results
5. Charts - Creating and managing visualizations

Usage:
    python mcp_superset_server.py  # Run as MCP server
"""
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

import httpx
import yaml
from fastmcp import FastMCP

# Internal imports
from nlu.entity_extractor import EntityExtractor
from nlu.semantic_matcher import SemanticMatcher, MatchResult, JoinPath
from nlu.table_intelligence import TableIntelligence, get_table_intelligence
from joins.enhanced_reasoner import EnhancedJoinReasoner
from validation.sql_validator import SQLValidator

# ---------------------------
# Config via environment vars
# ---------------------------
SUPERSET_URL = os.environ.get("SUPERSET_URL", "https://insights.314ecorp.com/insights").rstrip("/")
SUPERSET_USERNAME = os.environ.get("SUPERSET_USERNAME", "ng")
SUPERSET_PASSWORD = os.environ.get("SUPERSET_PASSWORD", "12345")
SUPERSET_PROVIDER = os.environ.get("SUPERSET_PROVIDER", "db")

READ_ONLY = os.environ.get("READ_ONLY", "false").lower() == "true"
ALLOWED_DATABASE_IDS = {
    int(x) for x in os.environ.get("ALLOWED_DATABASE_IDS", "").split(",") if x.strip().isdigit()
}

RELATIONSHIPS_PATH = os.environ.get("RELATIONSHIPS_PATH", "relationships.yaml")
ENGINE_HINT = os.environ.get("ENGINE_HINT", "postgresql")

mcp = FastMCP("superset-mcp")


# ---------------------------
# Superset REST client
# ---------------------------
class SupersetClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._access_token: Optional[str] = None
        self._csrf_token: Optional[str] = None
        self._token_ts: float = 0.0
        # Use cookies and disable SSL verification for testing
        self._http = httpx.Client(timeout=60.0, verify=False, cookies=httpx.Cookies())

    def login(self) -> None:
        if self._access_token and (time.time() - self._token_ts) < 60 * 20:
            return

        r = self._http.post(
            f"{self.base_url}/api/v1/security/login",
            headers={"Content-Type": "application/json"},
            json={
                "username": SUPERSET_USERNAME,
                "password": SUPERSET_PASSWORD,
                "provider": SUPERSET_PROVIDER,
                "refresh": True,
            },
        )
        r.raise_for_status()
        data = r.json()
        self._access_token = data["access_token"]
        self._token_ts = time.time()

        csrf = self._http.get(
            f"{self.base_url}/api/v1/security/csrf_token/",
            headers=self._headers(needs_csrf=False),
        )
        csrf.raise_for_status()
        self._csrf_token = csrf.json().get("result")

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"} if self._access_token else {}

    def _headers(self, needs_csrf: bool = False) -> Dict[str, str]:
        h = {"Content-Type": "application/json", **self._auth_headers()}
        if needs_csrf and self._csrf_token:
            h["X-CSRFToken"] = self._csrf_token
        return h

    def get(self, path: str, params: Optional[dict] = None) -> Any:
        self.login()
        r = self._http.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: dict, needs_csrf: bool = True) -> Any:
        self.login()
        r = self._http.post(
            f"{self.base_url}{path}",
            headers=self._headers(needs_csrf=needs_csrf),
            json=payload,
        )
        r.raise_for_status()
        return r.json()


client = SupersetClient(SUPERSET_URL)


# ---------------------------
# Caches
# ---------------------------
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
_dataset_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}
_semantic_matcher: Optional[SemanticMatcher] = None


def _get_semantic_matcher() -> SemanticMatcher:
    """Get or create the global semantic matcher instance."""
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = SemanticMatcher()
    return _semantic_matcher


def _reload_semantic_matcher() -> SemanticMatcher:
    """Reload the semantic matcher from config."""
    global _semantic_matcher
    _semantic_matcher = SemanticMatcher()
    return _semantic_matcher


def _get_all_datasets_cached() -> List[Dict]:
    """Get all datasets with caching."""
    now = time.time()
    if _dataset_cache["data"] and (now - _dataset_cache["timestamp"]) < CACHE_TTL:
        return _dataset_cache["data"]

    all_datasets = []
    page = 0
    page_size = 100

    while True:
        result = list_datasets(search=None, page=page, page_size=page_size)
        all_datasets.extend(result["results"])
        if len(result["results"]) < page_size:
            break
        page += 1

    _dataset_cache["data"] = all_datasets
    _dataset_cache["timestamp"] = now
    return all_datasets


def _clear_dataset_cache() -> None:
    """Clear the dataset cache."""
    _dataset_cache["data"] = None
    _dataset_cache["timestamp"] = 0.0


# ---------------------------
# Helpers
# ---------------------------
def _check_db_allowed(database_id: int) -> None:
    if ALLOWED_DATABASE_IDS and database_id not in ALLOWED_DATABASE_IDS:
        raise ValueError(f"database_id {database_id} is not allowed by ALLOWED_DATABASE_IDS")


def _load_relationships() -> List[dict]:
    if not os.path.exists(RELATIONSHIPS_PATH):
        return []
    with open(RELATIONSHIPS_PATH, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj.get("relationships", []) or []


def _normalize_dataset_result(r: dict) -> dict:
    cols = [
        {
            "column_name": c.get("column_name"),
            "type": c.get("type"),
            "is_dttm": c.get("is_dttm"),
        }
        for c in (r.get("columns") or [])
        if c.get("column_name")
    ]
    return {
        "id": r.get("id"),
        "database_id": r.get("database", {}).get("id") if isinstance(r.get("database"), dict) else r.get("database"),
        "schema": r.get("schema"),
        "table_name": r.get("table_name"),
        "sql": r.get("sql"),
        "columns": cols,
    }


# ===========================================================================
# SECTION 1: TRAINING TOOLS
# ===========================================================================

@mcp.tool
def train_fetch_all_databases() -> Dict[str, Any]:
    """
    Fetch all databases from Superset for training.

    Returns database IDs, names, backends, and schemas.
    """
    databases = []
    page = 0
    page_size = 100

    while True:
        params = {"page": page, "page_size": page_size}
        data = client.get("/api/v1/database/", params=params)

        for row in data.get("result", []):
            db_info = {
                "id": row.get("id"),
                "database_name": row.get("database_name"),
                "backend": row.get("backend"),
            }

            # Fetch schemas for this database
            try:
                schemas_data = client.get(f"/api/v1/database/{db_info['id']}/schemas/")
                db_info["schemas"] = schemas_data.get("result", [])
            except Exception:
                db_info["schemas"] = []

            databases.append(db_info)

        if len(data.get("result", [])) < page_size:
            break
        page += 1

    return {"count": len(databases), "databases": databases}


@mcp.tool
def train_fetch_all_datasets() -> Dict[str, Any]:
    """
    Fetch all registered datasets from Superset for training.

    Returns dataset IDs, table names, schemas, and database IDs.
    """
    all_datasets = []
    page = 0
    page_size = 100

    while True:
        params = {"page": page, "page_size": page_size}
        data = client.get("/api/v1/dataset/", params=params)

        for row in data.get("result", []):
            db = row.get("database")
            all_datasets.append({
                "id": row.get("id"),
                "database_id": db.get("id") if isinstance(db, dict) else db,
                "database_name": db.get("database_name") if isinstance(db, dict) else None,
                "schema": row.get("schema"),
                "table_name": row.get("table_name"),
            })

        if len(data.get("result", [])) < page_size:
            break
        page += 1

    return {"count": len(all_datasets), "datasets": all_datasets}


@mcp.tool
def train_fetch_dataset_columns(dataset_id: int) -> Dict[str, Any]:
    """
    Fetch detailed column information for a dataset.

    Args:
        dataset_id: The dataset ID to fetch columns for

    Returns:
        Dataset metadata including all columns with types
    """
    data = client.get(f"/api/v1/dataset/{dataset_id}")
    r = data.get("result", data)

    columns = []
    for c in r.get("columns", []):
        columns.append({
            "column_name": c.get("column_name"),
            "type": c.get("type"),
            "is_dttm": c.get("is_dttm", False),
            "filterable": c.get("filterable", True),
            "groupby": c.get("groupby", True),
        })

    return {
        "id": r.get("id"),
        "table_name": r.get("table_name"),
        "schema": r.get("schema"),
        "database_id": r.get("database", {}).get("id") if isinstance(r.get("database"), dict) else r.get("database"),
        "sql": r.get("sql"),
        "columns": columns,
    }


@mcp.tool
def train_fetch_sample_values(
    database_id: int,
    table_name: str,
    column_name: str,
    schema: Optional[str] = None,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Fetch sample values for a column to help with join detection.

    Args:
        database_id: Database ID
        table_name: Table name
        column_name: Column to sample
        schema: Schema name (optional)
        sample_size: Number of samples to fetch

    Returns:
        List of distinct sample values
    """
    _check_db_allowed(database_id)

    if schema:
        qualified_table = f'"{schema}"."{table_name}"'
    else:
        qualified_table = f'"{table_name}"'

    sql = f"""
        SELECT DISTINCT "{column_name}"
        FROM {qualified_table}
        WHERE "{column_name}" IS NOT NULL
        LIMIT {sample_size}
    """

    try:
        payload = {
            "database_id": database_id,
            "sql": sql,
            "runAsync": False,
            "queryLimit": sample_size,
        }
        result = client.post("/api/v1/sqllab/execute/", payload, needs_csrf=True)
        data = result.get("data", [])
        samples = [row[0] if row else None for row in data]
        return {"column": column_name, "samples": samples, "count": len(samples)}
    except Exception as e:
        return {"column": column_name, "samples": [], "error": str(e)}


@mcp.tool
def train_reload_semantic_matcher() -> Dict[str, Any]:
    """
    Reload the semantic matcher from config file.

    Call this after training completes to pick up new knowledge.
    """
    matcher = _reload_semantic_matcher()
    stats = matcher.get_statistics()
    return {
        "status": "reloaded",
        "tables": stats["total_tables"],
        "columns": stats["total_columns"],
        "join_patterns": stats["join_patterns"],
        "synonyms": stats["synonyms"]
    }


@mcp.tool
def get_semantic_matcher_stats() -> Dict[str, Any]:
    """
    Get statistics about the current semantic matcher state.
    """
    matcher = _get_semantic_matcher()
    return matcher.get_statistics()


@mcp.tool
def train_from_queries(
    include_saved_queries: bool = True,
    include_charts: bool = True,
    include_datasets: bool = True,
    since_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train the query learning system from existing Superset queries, charts, and datasets.

    This harvests saved queries, charts, and virtual datasets (SQL-based),
    analyzes their SQL patterns, and stores them as examples for few-shot
    learning to improve future SQL generation.

    Args:
        include_saved_queries: Whether to harvest SQL Lab saved queries
        include_charts: Whether to harvest chart configurations
        include_datasets: Whether to harvest virtual datasets with SQL queries
        since_date: Only harvest items modified since this date (ISO format)

    Returns:
        Training statistics including number of examples learned
    """
    from learning import QueryHarvester, QueryAnalyzer, KnowledgeStore

    # Get the semantic config for validation
    matcher = _get_semantic_matcher()
    semantic_config = {"schemas": {}}
    # Build a minimal config from the matcher
    for table in matcher.tables.values():
        schema = table.schema or "default"
        if schema not in semantic_config["schemas"]:
            semantic_config["schemas"][schema] = {"tables": {}}
        semantic_config["schemas"][schema]["tables"][table.name] = {
            "columns": {col: {} for col in table.columns.keys()} if table.columns else {}
        }

    harvester = QueryHarvester(client)
    analyzer = QueryAnalyzer(semantic_config=semantic_config)
    knowledge_store = KnowledgeStore()

    # Step 1: Harvest queries from all sources
    harvested = harvester.harvest_all(
        include_saved_queries=include_saved_queries,
        include_charts=include_charts,
        include_datasets=include_datasets,
        since_date=since_date
    )

    # Step 2: Analyze and store each query
    stats = {
        "harvested": len(harvested),
        "analyzed": 0,
        "stored": 0,
        "duplicates": 0,
        "errors": 0,
        "by_source": {}
    }

    for query in harvested:
        try:
            # Analyze the query
            example = analyzer.analyze(
                sql=query.sql,
                title=query.title,
                description=query.description or ""
            )

            if example is None:
                stats["errors"] += 1
                continue

            stats["analyzed"] += 1

            # Store the example
            example_id = knowledge_store.save_query_example(
                title=example.title,
                sql=example.sql,
                normalized_sql=example.normalized_sql,
                pattern_json=json.dumps(example.pattern.to_dict()) if example.pattern else "{}",
                keywords=example.keywords,
                tables=example.pattern.tables if example.pattern else [],
                source=query.source,
                source_id=query.source_id,
                description=example.description,
                schema_name=query.schema,
                database_id=query.database_id,
                dialect=example.dialect,
                query_type=example.pattern.query_type if example.pattern else "simple",
                complexity_score=example.pattern.complexity_score if example.pattern else 0.0
            )

            if example_id:
                stats["stored"] += 1
                source = query.source
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            else:
                stats["duplicates"] += 1

        except Exception as e:
            stats["errors"] += 1

    # Update metadata
    knowledge_store.set_metadata("last_query_training", datetime.now().isoformat())

    return {
        "status": "completed",
        "statistics": stats,
        "total_examples": knowledge_store.get_example_count()
    }


@mcp.tool
def get_query_examples(
    tables: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    query_type: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get stored query examples for few-shot learning.

    Args:
        tables: Filter by tables used
        keywords: Filter by keywords
        query_type: Filter by query type (simple, join, aggregation, etc.)
        limit: Maximum number of examples

    Returns:
        List of query examples
    """
    from learning import KnowledgeStore

    store = KnowledgeStore()
    examples = []

    if tables:
        examples = store.get_examples_by_tables(tables, limit=limit)
    elif keywords:
        examples = store.get_examples_by_keywords(keywords, limit=limit)
    elif query_type:
        examples = store.get_examples_by_type(query_type, limit=limit)
    else:
        examples = store.get_all_examples(limit=limit)

    return {
        "count": len(examples),
        "examples": examples
    }


@mcp.tool
def get_query_learning_stats() -> Dict[str, Any]:
    """
    Get statistics about the query learning system.

    Returns:
        Statistics about stored examples, feedback, etc.
    """
    from learning import KnowledgeStore

    store = KnowledgeStore()
    example_count = store.get_example_count()
    all_examples = store.get_all_examples(limit=1000, valid_only=False)

    # Calculate stats
    by_type = {}
    by_source = {}
    total_usage = 0
    total_success = 0
    total_failure = 0

    for ex in all_examples:
        query_type = ex.get("query_type", "unknown")
        source = ex.get("source", "unknown")

        by_type[query_type] = by_type.get(query_type, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1

        total_usage += ex.get("usage_count", 0)
        total_success += ex.get("success_count", 0)
        total_failure += ex.get("failure_count", 0)

    return {
        "total_examples": example_count,
        "valid_examples": store.get_example_count(valid_only=True),
        "by_query_type": by_type,
        "by_source": by_source,
        "total_usage": total_usage,
        "total_success": total_success,
        "total_failure": total_failure,
        "success_rate": total_success / max(total_usage, 1),
        "last_training": store.get_metadata("last_query_training")
    }


@mcp.tool
def validate_query_examples() -> Dict[str, Any]:
    """
    Validate stored query examples against the current schema.

    Marks examples as invalid if their tables no longer exist.

    Returns:
        Validation results
    """
    from learning import KnowledgeStore

    matcher = _get_semantic_matcher()
    known_tables = set(t.lower() for t in matcher.tables.keys())

    store = KnowledgeStore()
    invalidated = store.revalidate_examples(known_tables)

    return {
        "known_tables": len(known_tables),
        "examples_invalidated": invalidated,
        "valid_examples": store.get_example_count(valid_only=True)
    }


# ===========================================================================
# SECTION 2: DATA DISCOVERY TOOLS
# ===========================================================================

@mcp.tool
def list_databases() -> Dict[str, Any]:
    """List all Superset databases with their connection info."""
    all_databases = []
    page = 0
    page_size = 100

    while True:
        params = {"page": page, "page_size": page_size}
        data = client.get("/api/v1/database/", params=params)

        for row in data.get("result", []):
            all_databases.append({
                "id": row.get("id"),
                "database_name": row.get("database_name"),
                "backend": row.get("backend"),
                "expose_in_sqllab": row.get("expose_in_sqllab"),
            })

        if len(data.get("result", [])) < page_size:
            break
        page += 1

    return {"count": len(all_databases), "databases": all_databases}


@mcp.tool
def get_database_schemas(database_id: int) -> Dict[str, Any]:
    """Get all schemas for a specific database."""
    _check_db_allowed(database_id)
    data = client.get(f"/api/v1/database/{database_id}/schemas/")
    return {"database_id": database_id, "schemas": data.get("result", [])}


@mcp.tool
def list_datasets(search: Optional[str] = None, page: int = 0, page_size: int = 25) -> Dict[str, Any]:
    """List Superset datasets."""
    params = {"page": page, "page_size": page_size}
    if search:
        params["q"] = json.dumps({"filters": [{"col": "table_name", "opr": "ct", "value": search}]})
    data = client.get("/api/v1/dataset/", params=params)

    results = []
    for row in data.get("result", []):
        db = row.get("database")
        results.append({
            "id": row.get("id"),
            "database_id": db.get("id") if isinstance(db, dict) else db,
            "database_name": db.get("database_name") if isinstance(db, dict) else None,
            "schema": row.get("schema"),
            "table_name": row.get("table_name"),
        })
    return {"count": data.get("count"), "results": results}


@mcp.tool
def get_dataset(dataset_id: int) -> Dict[str, Any]:
    """Get dataset metadata (columns and virtual SQL)."""
    data = client.get(f"/api/v1/dataset/{dataset_id}")
    r = data.get("result", data)
    return _normalize_dataset_result(r)


def _get_datasets_internal(dataset_ids: List[int]) -> List[Dict[str, Any]]:
    """Internal helper to fetch dataset metadata."""
    out = []
    for did in dataset_ids:
        data = client.get(f"/api/v1/dataset/{did}")
        r = data.get("result", data)
        out.append(_normalize_dataset_result(r))
    return out


@mcp.tool
def get_datasets(dataset_ids: List[int]) -> Dict[str, Any]:
    """Bulk fetch dataset metadata for multiple datasets."""
    return {"datasets": _get_datasets_internal(dataset_ids)}


# ===========================================================================
# SECTION 3: SEMANTIC MATCHING TOOLS
# ===========================================================================

def _discover_tables_internal(entity_names: List[str], min_score: float = 0.4, smart_select: bool = True) -> Dict[str, Any]:
    """Internal helper for discovering tables.

    Args:
        entity_names: List of entity names to match
        min_score: Minimum similarity score to include
        smart_select: If True, aggressively auto-select best matches (minimize user prompts)
    """
    matcher = _get_semantic_matcher()
    results = {}

    # Pre-process: Detect compound entity patterns and prioritize combined tables
    entity_names_lower = [e.lower() for e in entity_names]
    compound_replacements = {}

    # If both "user/users" AND "assignment" are mentioned, prioritize "Assignment Usage"
    has_user = any(e in ['user', 'users'] for e in entity_names_lower)
    has_assignment = any('assignment' in e for e in entity_names_lower)
    if has_user and has_assignment:
        # Remove individual user/assignment entities and add compound match
        compound_replacements['Assignment Usage'] = {
            'removes': ['user', 'users', 'assignment'],
            'reason': 'compound_match: user + assignment -> Assignment Usage'
        }

    # First, add compound matches with high priority
    entities_to_skip = set()
    for compound_table, compound_info in compound_replacements.items():
        entities_to_skip.update(compound_info['removes'])
        # Add the compound table as a high-confidence match
        table_info = matcher.get_table(compound_table)
        if table_info:
            results[compound_table] = {
                "candidates": [{
                    "table_name": compound_table,
                    "schema": table_info.schema,
                    "dataset_id": table_info.dataset_id,
                    "score": 0.98,  # High confidence for compound match
                    "match_reason": compound_info['reason']
                }],
                "is_ambiguous": False,
                "recommended": {
                    "table_name": compound_table,
                    "schema": table_info.schema,
                    "dataset_id": table_info.dataset_id,
                    "score": 0.98,
                    "match_reason": compound_info['reason']
                }
            }

    for entity in entity_names:
        # Skip entities that are part of a compound match
        if entity.lower() in entities_to_skip:
            continue

        matches = matcher.match_table(entity, min_score)

        candidates = [
            {
                "table_name": m.table_name,
                "schema": m.schema,
                "dataset_id": m.dataset_id,
                "score": m.score,
                "match_reason": m.match_reason
            }
            for m in matches
        ]

        # Smart selection: Be much more aggressive about auto-selecting
        # Only mark as ambiguous when scores are VERY close AND both high
        is_ambiguous = False
        if not smart_select:
            # Legacy behavior: more conservative
            is_ambiguous = (
                len(matches) >= 2 and
                matches[0].score < 1.0 and
                matches[0].score >= 0.6 and
                matches[1].score >= 0.6 and
                (matches[0].score - matches[1].score) < 0.2
            )
        else:
            # Smart mode: Only ambiguous when truly confusing
            # - Both scores above 0.85 (very high confidence)
            # - Score difference less than 0.05 (almost identical)
            # - Neither is an exact match
            is_ambiguous = (
                len(matches) >= 2 and
                matches[0].score < 1.0 and
                matches[0].score >= 0.85 and
                matches[1].score >= 0.85 and
                (matches[0].score - matches[1].score) < 0.05
            )

        # Always recommend top match if we have any matches with decent score
        recommended = None
        if matches:
            top = matches[0]
            # In smart mode, always recommend if score >= 0.5
            # This covers exact (1.0), alias (0.98), synonym (0.95), prefix (0.65+)
            if smart_select and top.score >= 0.5:
                recommended = {
                    "table_name": top.table_name,
                    "schema": top.schema,
                    "dataset_id": top.dataset_id,
                    "score": top.score,
                    "match_reason": top.match_reason
                }
            elif top.score == 1.0 or (top.score >= 0.7 and not is_ambiguous):
                recommended = {
                    "table_name": top.table_name,
                    "schema": top.schema,
                    "dataset_id": top.dataset_id,
                    "score": top.score
                }

        results[entity] = {
            "candidates": candidates,
            "is_ambiguous": is_ambiguous,
            "recommended": recommended
        }

    return {"matches": results}


@mcp.tool
def discover_tables(
    entity_names: List[str],
    min_score: float = 0.4
) -> Dict[str, Any]:
    """
    Semantically match entity names to tables using trained semantics.

    Uses the semantic matcher loaded from config with:
    - Exact match (score: 1.0)
    - Alias match (score: 0.98)
    - Synonym match (score: 0.95)
    - Prefix/suffix match (score: 0.65-0.9)
    - Fuzzy match (score: varies)

    Args:
        entity_names: List of entity names to match (e.g., ["assets", "events", "users"])
        min_score: Minimum similarity score to include (default 0.4)

    Returns:
        Dict with matches for each entity, including candidates and recommendations
    """
    return _discover_tables_internal(entity_names, min_score)


def _get_table_info_internal(table_name: str) -> Dict[str, Any]:
    """Internal helper for getting table info."""
    matcher = _get_semantic_matcher()
    table = matcher.get_table(table_name)

    if not table:
        return {"found": False, "table_name": table_name}

    columns = []
    for col_name, col_info in table.columns.items():
        columns.append({
            "name": col_name,
            "type": col_info.column_type,
            "is_pk": col_info.is_pk,
            "is_fk": col_info.is_fk,
            "references": col_info.references
        })

    # Get join patterns
    joins_from = matcher.join_patterns.get(table_name.lower(), [])
    join_patterns = [
        {
            "to_table": jp.right_table,
            "on": f"{jp.left_column} = {jp.right_column}",
            "type": jp.join_type
        }
        for jp in joins_from
    ]

    return {
        "found": True,
        "table_name": table.name,
        "schema": table.schema,
        "dataset_id": table.dataset_id,
        "database_id": table.database_id,
        "columns": columns,
        "primary_keys": table.primary_keys,
        "foreign_keys": [{"column": fk[0], "references": fk[1]} for fk in table.foreign_keys],
        "aliases": table.aliases,
        "join_patterns": join_patterns
    }


@mcp.tool
def get_table_info(table_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a table from the semantic matcher.

    Args:
        table_name: The table name to look up

    Returns:
        Table info including columns, primary keys, foreign keys, and join patterns
    """
    return _get_table_info_internal(table_name)


def _find_join_path_internal(table_names: List[str], max_depth: int = 3) -> Dict[str, Any]:
    """Internal helper for finding join paths."""
    matcher = _get_semantic_matcher()
    joins = matcher.find_join_path(table_names, max_depth)

    return {
        "tables": table_names,
        "join_paths": [
            {
                "left_table": jp.left_table,
                "right_table": jp.right_table,
                "left_column": jp.left_column,
                "right_column": jp.right_column,
                "join_type": jp.join_type,
                "confidence": jp.confidence
            }
            for jp in joins
        ],
        "connected": len(joins) >= len(table_names) - 1 if len(table_names) > 1 else True
    }


@mcp.tool
def find_join_path(
    table_names: List[str],
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Find join paths connecting multiple tables using trained semantics.

    Args:
        table_names: List of table names to connect
        max_depth: Maximum intermediate tables to explore

    Returns:
        List of join paths connecting the tables
    """
    return _find_join_path_internal(table_names, max_depth)


@mcp.tool
def suggest_joins(
    dataset_ids: List[int],
    join_key_hints: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Suggest join conditions using enhanced reasoning.

    Features:
    - Uses trained join patterns from semantic config
    - Handles camelCase/snake_case normalization
    - FK → PK pattern detection
    - Type compatibility checking
    - YAML relationship priority

    Args:
        dataset_ids: List of dataset IDs to find joins between
        join_key_hints: Optional column names mentioned as join keys

    Returns:
        Dict with join suggestions sorted by confidence
    """
    meta = _get_datasets_internal(dataset_ids)
    relationships = _load_relationships()

    # Use semantic matcher for additional join hints
    matcher = _get_semantic_matcher()
    table_names = [ds["table_name"] for ds in meta]

    # Get join paths from semantic config
    semantic_joins = matcher.find_join_path(table_names)

    # Build table name lookup
    all_datasets = _get_all_datasets_cached()
    table_lookup = {ds["table_name"].lower(): ds["id"] for ds in all_datasets}

    # Use enhanced reasoner for additional detection
    reasoner = EnhancedJoinReasoner(meta, relationships, table_lookup)
    suggestions = reasoner.suggest_joins(dataset_ids, join_key_hints)

    # Merge with semantic joins (higher priority)
    semantic_suggestions = []
    for jp in semantic_joins:
        # Find dataset IDs for the tables
        left_id = None
        right_id = None
        for ds in meta:
            if ds["table_name"].lower() == jp.left_table.lower():
                left_id = ds["id"]
            if ds["table_name"].lower() == jp.right_table.lower():
                right_id = ds["id"]

        if left_id and right_id:
            semantic_suggestions.append({
                "left_dataset_id": left_id,
                "left_table": jp.left_table,
                "left_column": jp.left_column,
                "right_dataset_id": right_id,
                "right_table": jp.right_table,
                "right_column": jp.right_column,
                "join_type": jp.join_type,
                "confidence": 0.95,  # High confidence from trained patterns
                "reason": "trained join pattern",
                "type_compatible": True
            })

    # Combine and deduplicate
    all_suggestions = semantic_suggestions + [
        {
            "left_dataset_id": s.left_dataset_id,
            "left_table": s.left_table,
            "left_column": s.left_column,
            "right_dataset_id": s.right_dataset_id,
            "right_table": s.right_table,
            "right_column": s.right_column,
            "join_type": s.join_type,
            "confidence": s.confidence,
            "reason": s.reason,
            "type_compatible": s.type_compatible
        }
        for s in suggestions
    ]

    # Deduplicate by (left_table, left_col, right_table, right_col)
    seen = set()
    unique_suggestions = []
    for s in sorted(all_suggestions, key=lambda x: x["confidence"], reverse=True):
        key = (s["left_table"].lower(), s["left_column"].lower(),
               s["right_table"].lower(), s["right_column"].lower())
        if key not in seen:
            seen.add(key)
            unique_suggestions.append(s)

    return {"suggestions": unique_suggestions}


# ===========================================================================
# SECTION 4: QUERY BUILDING TOOLS
# ===========================================================================

def _get_known_columns() -> set:
    """Get all known column names from the semantic matcher."""
    matcher = _get_semantic_matcher()
    columns = set()
    for table_info in matcher.tables.values():
        columns.update(col.lower() for col in table_info.columns.keys())
    return columns


def _get_known_tables() -> set:
    """Get all known table names from the semantic matcher."""
    matcher = _get_semantic_matcher()
    return set(name.lower() for name in matcher.tables.keys())


def _get_known_schemas() -> set:
    """Get all known schema names from the semantic config."""
    matcher = _get_semantic_matcher()
    schemas = set()
    # Get schemas from the matcher's config
    if hasattr(matcher, 'config') and matcher.config:
        # From databases section
        for db_info in matcher.config.get('databases', {}).values():
            if isinstance(db_info, dict) and 'schemas' in db_info:
                schemas.update(s.lower() for s in db_info['schemas'])
        # From schemas section (top-level keys)
        if 'schemas' in matcher.config:
            schemas.update(s.lower() for s in matcher.config['schemas'].keys())
    return schemas


def _get_llm_client_for_extraction():
    """Get an LLM client function for entity extraction."""
    try:
        from llm import get_llm_service
        llm_service = get_llm_service()

        def llm_client(prompt: str) -> str:
            """Simple LLM client that returns JSON string."""
            try:
                # Use the default provider
                if llm_service.config.openai_api_key:
                    client = llm_service._get_openai_client()
                    response = client.chat.completions.create(
                        model=llm_service.config.openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    return response.choices[0].message.content
                elif llm_service.config.claude_api_key:
                    client = llm_service._get_claude_client()
                    response = client.messages.create(
                        model=llm_service.config.claude_model,
                        max_tokens=2048,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
                return None
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
                return None

        return llm_client
    except Exception:
        return None


def _extract_query_entities_internal(query: str, use_llm: bool = True) -> Dict[str, Any]:
    """Internal helper for extracting query entities."""
    known_columns = _get_known_columns()
    known_tables = _get_known_tables()
    known_schemas = _get_known_schemas()

    # Get LLM client for better extraction if enabled
    llm_client = _get_llm_client_for_extraction() if use_llm else None

    extractor = EntityExtractor(
        llm_client=llm_client,
        known_columns=known_columns,
        known_tables=known_tables,
        known_schemas=known_schemas
    )
    intent = extractor.extract(query)

    return {
        "entities": [
            {
                "name": e.name,
                "type": e.entity_type,
                "confidence": e.confidence,
                "modifiers": e.modifiers
            }
            for e in intent.entities
        ],
        "columns_mentioned": intent.columns_mentioned,
        "filters": [
            {
                "column": f.column,
                "operator": f.operator,
                "value": f.value,
                "original_text": f.original_text
            }
            for f in intent.filters
        ],
        "join_hints": [
            {
                "left": j.left_entity,
                "right": j.right_entity,
                "key_hint": j.join_key_hint,
                "relationship": j.relationship
            }
            for j in intent.join_hints
        ],
        "time_filters": [
            {
                "period": t.period,
                "column_hint": t.column_hint
            }
            for t in intent.time_filters
        ],
        "state_filters": [
            {
                "state": s.state,
                "negated": s.negated,
                "target_entity": s.target_entity,
                "original_text": s.original_text
            }
            for s in intent.state_filters
        ],
        "aggregations": intent.aggregations,
        "limit": intent.limit,
        "raw_query": intent.raw_query,
        "extracted_schema": intent.extracted_schema
    }


@mcp.tool
def extract_query_entities(query: str) -> Dict[str, Any]:
    """
    Extract structured entities from a natural language query.

    Analyzes the query and extracts:
    - Dataset/table entities
    - Column references
    - Join hints
    - Time filters
    - Aggregations
    - Limit

    Args:
        query: Natural language query string

    Returns:
        Dict with extracted entities, join_hints, time_filters, aggregations, etc.
    """
    return _extract_query_entities_internal(query)


@mcp.tool
def build_execution_plan(
    query: str,
    auto_select: bool = True  # Changed default to True for smarter auto-selection
) -> Dict[str, Any]:
    """
    Build a complete execution plan from a natural language query.

    This is the main entry point for natural language to SQL conversion.
    It combines entity extraction, semantic table discovery, and join suggestion.

    The system is designed to minimize user prompts and auto-select the best tables
    based on semantic matching, learned patterns, and table intelligence.

    Args:
        query: Natural language query (e.g., "show me assets and events by assetId")
        auto_select: If True (default), automatically select top match to minimize prompts

    Returns:
        Dict with execution plan including:
        - entities: Extracted entities from query
        - table_matches: Matched tables from semantic config
        - has_ambiguity: Whether user input is truly needed (rare)
        - recommended_tables: Auto-selected table names
        - join_paths: Join paths from semantic config
        - ready_for_sql: Whether plan is ready for SQL generation
    """
    # Step 1: Extract entities
    entities_result = _extract_query_entities_internal(query)

    # Get dataset entity names
    entity_names = [
        e["name"] for e in entities_result["entities"]
        if e["type"] == "dataset"
    ]

    # Get column hints
    column_hints = entities_result.get("columns_mentioned", [])
    for jh in entities_result.get("join_hints", []):
        if jh.get("key_hint"):
            column_hints.append(jh["key_hint"])

    # Step 2: Discover tables using semantic matcher with smart selection
    table_matches = {}
    if entity_names:
        # Use smart_select=True for aggressive auto-selection
        discovery_result = _discover_tables_internal(entity_names, smart_select=True)
        table_matches = discovery_result["matches"]

    # Step 2b: Use TableIntelligence to enhance candidates when time filters detected
    time_filters = entities_result.get("time_filters", [])
    if time_filters and table_matches:
        try:
            intelligence = get_table_intelligence()

            # Determine time filter type from query (created, updated, expired, etc.)
            query_lower = query.lower()
            time_filter_type = None
            for keyword in ['created', 'updated', 'modified', 'changed', 'expired', 'archived', 'due']:
                if keyword in query_lower:
                    time_filter_type = keyword
                    break

            # Enhance each entity's candidates with date column info
            for entity, match_data in table_matches.items():
                candidates = match_data.get("candidates", [])
                if candidates:
                    enhanced = intelligence.recommend_tables_for_query(
                        entity=entity,
                        candidates=candidates,
                        has_time_filter=True,
                        time_filter_type=time_filter_type
                    )
                    if enhanced:
                        table_matches[entity]["candidates"] = enhanced
                        # Smart selection: Auto-select the best table with date columns
                        # DON'T mark as ambiguous - just pick the best one
                        enhanced_top = enhanced[0]
                        if enhanced_top.get("has_date_columns"):
                            # Auto-select this table - it's better for time filtering
                            table_matches[entity]["recommended"] = {
                                "table_name": enhanced_top.get("table_name"),
                                "schema": enhanced_top.get("schema"),
                                "score": enhanced_top.get("score", 0.9),
                                "match_reason": enhanced_top.get("match_reason", "Has date columns for time filtering")
                            }
                            # Clear ambiguity flag since we made a smart choice
                            table_matches[entity]["is_ambiguous"] = False
        except Exception:
            # Don't fail if TableIntelligence has issues
            pass

    # Step 3: Check for ambiguity - should be rare with smart selection
    has_ambiguity = any(
        m.get("is_ambiguous", False)
        for m in table_matches.values()
    )

    # Step 4: Get recommended tables - always try to get recommendations
    recommended_tables = []
    for entity, match_data in table_matches.items():
        if match_data.get("recommended"):
            recommended_tables.append(match_data["recommended"]["table_name"])
        elif match_data.get("candidates"):
            # Always auto-select top candidate when auto_select is True
            # This minimizes user prompts
            if auto_select:
                top_candidate = match_data["candidates"][0]
                recommended_tables.append(top_candidate["table_name"])
                # Also set it as recommended for consistency
                match_data["recommended"] = {
                    "table_name": top_candidate["table_name"],
                    "schema": top_candidate.get("schema"),
                    "score": top_candidate.get("score"),
                    "match_reason": top_candidate.get("match_reason", "Auto-selected top match")
                }
                match_data["is_ambiguous"] = False

    # Step 5: Find join paths
    join_paths = []
    if len(recommended_tables) >= 2:
        join_result = _find_join_path_internal(recommended_tables)
        join_paths = join_result["join_paths"]

    # Step 6: Get table info for SQL generation
    tables_info = []
    for table_name in recommended_tables:
        info = _get_table_info_internal(table_name)
        if info.get("found"):
            tables_info.append(info)

    # Get filters from entities result
    filters = entities_result.get("filters", [])
    time_filters = entities_result.get("time_filters", [])
    state_filters = entities_result.get("state_filters", [])
    extracted_schema = entities_result.get("extracted_schema")

    return {
        "query": query,
        "entities": entities_result,
        "table_matches": table_matches,
        "has_ambiguity": has_ambiguity,
        "recommended_tables": recommended_tables,
        "tables_info": tables_info,
        "join_paths": join_paths,
        "column_hints": column_hints,
        "filters": filters,
        "time_filters": time_filters,
        "state_filters": state_filters,
        "extracted_schema": extracted_schema,
        "ready_for_sql": (
            not has_ambiguity and
            len(recommended_tables) > 0
        ),
        "needs_user_input": has_ambiguity
    }


def _fetch_columns_from_superset(database_id: int, schema: str, table_name: str) -> List[Dict]:
    """Fetch columns directly from Superset API as a fallback."""
    try:
        params = {"q": json.dumps({"table_name": table_name, "schema_name": schema})}
        metadata = client.get(f"/api/v1/database/{database_id}/table_metadata/", params=params)
        columns = []
        for col in metadata.get("columns", []):
            col_name = col.get("column_name") or col.get("name", "")
            col_type = col.get("type") or col.get("type_generic", "VARCHAR")
            if col_name:
                columns.append({
                    "name": col_name,
                    "type": str(col_type)
                })
        return columns
    except Exception:
        return []


@mcp.tool
def generate_sql_context(
    tables: List[str],
    query: str,
    join_hints: Optional[List[Dict]] = None,
    extracted_schema: Optional[str] = None,
    state_filters: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Generate context for LLM-based SQL generation.

    This prepares all the schema information needed by an LLM to generate SQL.

    Args:
        tables: List of table names to include
        query: The natural language query
        join_hints: Optional explicit join hints
        extracted_schema: Schema name extracted from the query (e.g., 'jeeves')
        state_filters: Optional state filters (e.g., 'not completed', 'pending')

    Returns:
        Context dict with schemas, joins, and hints for LLM consumption
    """
    matcher = _get_semantic_matcher()

    # Build table schemas
    schemas = []
    for table_name in tables:
        table = matcher.get_table(table_name)
        if table:
            cols = []
            column_list = []

            # Check if we have columns from semantic config
            if table.columns:
                for col_name, col_info in table.columns.items():
                    col_desc = f"{col_name} ({col_info.column_type})"
                    if col_info.is_pk:
                        col_desc += " PRIMARY KEY"
                    if col_info.is_fk and col_info.references:
                        col_desc += f" FK -> {col_info.references}"
                    cols.append(col_desc)
                    column_list.append(col_name)
            else:
                # Fallback: fetch columns dynamically from Superset
                db_id = table.database_id or 1
                fetched_cols = _fetch_columns_from_superset(db_id, table.schema, table.name)
                for col in fetched_cols:
                    cols.append(f"{col['name']} ({col['type']})")
                    column_list.append(col['name'])

            schemas.append({
                "table_name": table.name,
                "schema": table.schema,
                "columns": cols,
                "column_list": column_list
            })

    # Get join paths
    joins = []
    if len(tables) >= 2:
        join_result = _find_join_path_internal(tables)
        for jp in join_result["join_paths"]:
            joins.append(
                f"{jp['left_table']}.{jp['left_column']} {jp['join_type'].upper()} JOIN "
                f"{jp['right_table']}.{jp['right_column']}"
            )

    # Get known schemas
    known_schemas = list(_get_known_schemas())

    return {
        "engine": ENGINE_HINT,
        "user_query": query,
        "tables": schemas,
        "available_joins": joins,
        "join_hints": join_hints or [],
        "known_schemas": known_schemas,
        "extracted_schema": extracted_schema,
        "state_filters": state_filters or [],
        "rules": [
            "Use only the provided table schemas",
            "Qualify all column names with table aliases",
            "Do NOT use SELECT *",
            f"Use {ENGINE_HINT} SQL dialect",
            "Include LIMIT 1000 unless user specifies otherwise",
            "Use proper JOIN syntax with ON clauses",
            "For CTEs, use WITH clause syntax"
        ]
    }


# ===========================================================================
# SECTION 5: VALIDATION & EXECUTION TOOLS
# ===========================================================================

@mcp.tool
def validate_sql(
    sql: str,
    dataset_ids: Optional[List[int]] = None,
    database_id: Optional[int] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate SQL query before execution.

    Checks:
    - Basic syntax (balanced parentheses, quotes)
    - Dangerous operations blocked in READ_ONLY mode
    - Column references (warnings for potentially missing columns)
    - Best practices (SELECT *, missing LIMIT)
    - Dialect-specific syntax (StarRocks vs PostgreSQL)

    Args:
        sql: SQL query to validate
        dataset_ids: Optional dataset IDs for column reference checking
        database_id: Optional database ID to determine dialect for validation
        strict: If True, treat warnings as errors

    Returns:
        Dict with is_valid, errors, warnings, suggestions
    """
    datasets = []
    if dataset_ids:
        datasets = _get_datasets_internal(dataset_ids)

    # Determine dialect from database_id or use global ENGINE_HINT
    dialect = ENGINE_HINT
    if database_id:
        matcher = _get_semantic_matcher()
        dialect = matcher.get_database_backend(database_id)

    validator = SQLValidator(datasets, read_only=READ_ONLY, dialect=dialect)
    result = validator.validate(sql, strict=strict)

    return {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "suggestions": result.suggestions,
        "can_execute": result.can_execute,
        "dialect": dialect
    }


@mcp.tool
def execute_sql(
    database_id: int,
    sql: str,
    schema: Optional[str] = None,
    limit: int = 1000,
    run_async: bool = False,
) -> Dict[str, Any]:
    """
    Execute SQL via Superset SQL Lab and fetch results.

    Args:
        database_id: Database to execute against
        sql: SQL query to execute
        schema: Optional schema context
        limit: Row limit (default 1000)
        run_async: Run asynchronously

    Returns:
        Query results or async status
    """
    _check_db_allowed(database_id)

    if READ_ONLY:
        banned = ["insert ", "update ", "delete ", "drop ", "alter ", "create "]
        low = sql.lower()
        if any(b in low for b in banned):
            raise ValueError("READ_ONLY is enabled; write SQL is blocked.")

    payload = {
        "database_id": database_id,
        "schema": schema,
        "sql": sql,
        "runAsync": run_async,
        "queryLimit": limit,
    }
    started = client.post("/api/v1/sqllab/execute/", payload, needs_csrf=True)

    # Check for errors in execute response
    if started.get("error") or started.get("errors"):
        error_msg = started.get("error") or started.get("errors") or started.get("message", "Unknown error")
        raise ValueError(f"SQL execution failed: {error_msg}")

    client_id = started.get("client_id") or started.get("result", {}).get("client_id")
    if run_async:
        return {"client_id": client_id, "status": "running"}

    # Some responses include data directly (sync mode)
    if started.get("data") is not None:
        columns = started.get("columns", [])
        data = started.get("data", [])
    elif client_id:
        results = client.get("/api/v1/sqllab/results/", params={"q": json.dumps({"client_id": client_id})})
        columns = results.get("columns", [])
        data = results.get("data", [])
    else:
        # No client_id and no direct data - check if there's an error
        raise ValueError(f"SQL execution failed - no client_id returned. Response: {started}")

    return {
        "client_id": client_id,
        "columns": [c.get("name") if isinstance(c, dict) else c for c in columns],
        "data": data,
        "row_count": len(data),
        "status": "completed"
    }


@mcp.tool
def create_saved_query(
    database_id: int,
    sql: str,
    label: str,
    schema: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save a query to Superset's saved queries list.

    This creates a saved query that users can see in SQL Lab's "Saved Queries" tab.
    The user can then open, modify, and run the query from the Superset UI.

    Args:
        database_id: Database ID to associate the query with
        sql: The SQL query text
        label: A name/label for the saved query
        schema: Optional schema context
        description: Optional description of what the query does

    Returns:
        Dict with saved query ID and URL to view it
    """
    _check_db_allowed(database_id)

    payload = {
        "db_id": database_id,
        "sql": sql,
        "label": label,
        "schema": schema or "",
        "description": description or ""
    }

    result = client.post("/api/v1/saved_query/", payload, needs_csrf=True)
    saved_query_id = result.get("id") or result.get("result", {}).get("id")

    # Build URL to view the query in SQL Lab
    base_url = client.base_url.rstrip("/")
    sqllab_url = f"{base_url}/sqllab?savedQueryId={saved_query_id}"

    return {
        "saved_query_id": saved_query_id,
        "label": label,
        "sqllab_url": sqllab_url,
        "message": f"Query saved. Open in SQL Lab: {sqllab_url}"
    }


@mcp.tool
def list_saved_queries(
    database_id: Optional[int] = None,
    search: Optional[str] = None,
    page_size: int = 20
) -> Dict[str, Any]:
    """
    List saved queries in Superset.

    Args:
        database_id: Filter by database ID (optional)
        search: Search term to filter by label (optional)
        page_size: Number of results to return

    Returns:
        List of saved queries with their details
    """
    params = {"page_size": page_size}

    filters = []
    if database_id:
        filters.append({"col": "db_id", "opr": "eq", "value": database_id})
    if search:
        filters.append({"col": "label", "opr": "ct", "value": search})

    if filters:
        params["filters"] = json.dumps(filters)

    result = client.get("/api/v1/saved_query/", params=params)
    queries = result.get("result", [])

    base_url = client.base_url.rstrip("/")
    return {
        "queries": [
            {
                "id": q.get("id"),
                "label": q.get("label"),
                "database_id": q.get("db_id") or q.get("database", {}).get("id"),
                "schema": q.get("schema"),
                "sql_preview": (q.get("sql", "")[:100] + "...") if len(q.get("sql", "")) > 100 else q.get("sql", ""),
                "sqllab_url": f"{base_url}/sqllab?savedQueryId={q.get('id')}"
            }
            for q in queries
        ],
        "count": len(queries)
    }


@mcp.tool
def get_sqllab_url(
    database_id: int,
    sql: Optional[str] = None,
    schema: Optional[str] = None,
    saved_query_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a URL to open SQL Lab with pre-filled query.

    This allows the orchestrator to direct users to SQL Lab with a query ready to run.

    Args:
        database_id: Database ID to use
        sql: SQL query to pre-fill (optional if using saved_query_id)
        schema: Schema to select (optional)
        saved_query_id: ID of a saved query to load (optional)

    Returns:
        Dict with SQL Lab URL that user can open in browser
    """
    base_url = client.base_url.rstrip("/")

    if saved_query_id:
        url = f"{base_url}/sqllab?savedQueryId={saved_query_id}"
    else:
        # Build URL with query parameters
        # Look up database name for the given ID
        import urllib.parse
        db_name = None
        try:
            dbs = list_databases.fn()
            for db in dbs.get("databases", []):
                if db["id"] == database_id:
                    db_name = db["database_name"]
                    break
        except Exception:
            pass

        # Use both dbId and optionally database name
        params = {"dbId": database_id}
        if db_name:
            params["dbname"] = db_name
        if sql:
            params["sql"] = sql
        if schema:
            params["schema"] = schema
        query_string = urllib.parse.urlencode(params)
        url = f"{base_url}/sqllab?{query_string}"

    return {
        "sqllab_url": url,
        "database_id": database_id,
        "instructions": "Open this URL in your browser to view and run the query in SQL Lab"
    }


@mcp.tool
def execute_and_save(
    database_id: int,
    sql: str,
    label: str,
    schema: Optional[str] = None,
    description: Optional[str] = None,
    limit: int = 1000
) -> Dict[str, Any]:
    """
    Execute a SQL query AND save it as a saved query in one operation.

    This is the main function for the orchestrator flow:
    1. Validates and executes the SQL
    2. Saves it as a named query in Superset
    3. Returns results AND a URL to view/modify in SQL Lab

    Args:
        database_id: Database to execute against
        sql: SQL query to execute and save
        label: Name for the saved query
        schema: Optional schema context
        description: Optional description
        limit: Row limit for results

    Returns:
        Dict with execution results, saved query ID, and SQL Lab URL
    """
    _check_db_allowed(database_id)

    if READ_ONLY:
        banned = ["insert ", "update ", "delete ", "drop ", "alter ", "create "]
        low = sql.lower()
        if any(b in low for b in banned):
            raise ValueError("READ_ONLY is enabled; write SQL is blocked.")

    # Execute the query
    payload = {
        "database_id": database_id,
        "schema": schema,
        "sql": sql,
        "runAsync": False,
        "queryLimit": limit,
    }
    started = client.post("/api/v1/sqllab/execute/", payload, needs_csrf=True)

    # Check for errors in execute response
    if started.get("error") or started.get("errors"):
        error_msg = started.get("error") or started.get("errors") or started.get("message", "Unknown error")
        raise ValueError(f"SQL execution failed: {error_msg}")

    client_id = started.get("client_id") or started.get("result", {}).get("client_id")

    # Some responses include data directly (sync mode)
    if started.get("data") is not None:
        columns = started.get("columns", [])
        data = started.get("data", [])
    elif client_id:
        results = client.get("/api/v1/sqllab/results/", params={"q": json.dumps({"client_id": client_id})})
        columns = results.get("columns", [])
        data = results.get("data", [])
    else:
        raise ValueError(f"SQL execution failed - no client_id returned. Response: {started}")

    # Save the query
    save_payload = {
        "db_id": database_id,
        "sql": sql,
        "label": label,
        "schema": schema or "",
        "description": description or ""
    }
    save_result = client.post("/api/v1/saved_query/", save_payload, needs_csrf=True)
    saved_query_id = save_result.get("id") or save_result.get("result", {}).get("id")

    base_url = client.base_url.rstrip("/")
    sqllab_url = f"{base_url}/sqllab?savedQueryId={saved_query_id}"

    return {
        "execution": {
            "client_id": client_id,
            "columns": [c.get("name") if isinstance(c, dict) else c for c in columns],
            "data": data,
            "row_count": len(data),
            "status": "completed"
        },
        "saved_query": {
            "id": saved_query_id,
            "label": label,
            "sqllab_url": sqllab_url
        },
        "message": f"Query executed ({len(data)} rows) and saved. View in SQL Lab: {sqllab_url}"
    }


@mcp.tool
def ensure_same_database(dataset_ids: List[int]) -> Dict[str, Any]:
    """Verify all datasets belong to the same Superset database_id."""
    meta = _get_datasets_internal(dataset_ids)
    dbs = sorted({d["database_id"] for d in meta})
    return {"ok": len(dbs) == 1, "database_ids": dbs}


# ===========================================================================
# SECTION 6: CHART TOOLS
# ===========================================================================

@mcp.tool
def infer_chart_type(
    sql: str,
    query: str,
    result_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Infer appropriate chart type from SQL query and natural language query.

    Args:
        sql: The SQL query
        query: Original natural language query
        result_columns: Optional list of result column names

    Returns:
        Recommended chart type with configuration hints
    """
    sql_lower = sql.lower()
    query_lower = query.lower()

    # Detect aggregations
    has_count = "count(" in sql_lower
    has_sum = "sum(" in sql_lower
    has_avg = "avg(" in sql_lower
    has_group_by = "group by" in sql_lower
    has_order_by = "order by" in sql_lower

    # Detect time columns
    time_keywords = ["date", "time", "created", "updated", "timestamp", "month", "year", "day", "week"]
    has_time = any(kw in sql_lower for kw in time_keywords)

    # Detect chart type keywords in query
    chart_keywords = {
        "bar": ["bar chart", "bar graph", "bars"],
        "line": ["line chart", "line graph", "trend", "over time", "time series"],
        "pie": ["pie chart", "pie graph", "distribution", "breakdown"],
        "table": ["table", "list", "show me", "display"],
        "metric": ["count", "total", "how many", "number of"],
        "scatter": ["scatter", "correlation", "relationship between"],
        "heatmap": ["heatmap", "heat map", "matrix"],
    }

    for chart_type, keywords in chart_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return {
                "chart_type": chart_type,
                "confidence": 0.85,
                "reason": f"Query mentions '{chart_type}'-related keywords"
            }

    # Infer from SQL structure
    if has_count and not has_group_by:
        return {
            "chart_type": "metric",
            "confidence": 0.9,
            "reason": "Single aggregate without GROUP BY"
        }

    if has_group_by and has_time:
        return {
            "chart_type": "line",
            "confidence": 0.85,
            "reason": "Grouped aggregation with time column"
        }

    if has_group_by and (has_count or has_sum or has_avg):
        return {
            "chart_type": "bar",
            "confidence": 0.8,
            "reason": "Grouped aggregation suggests categorical comparison"
        }

    if has_group_by:
        return {
            "chart_type": "bar",
            "confidence": 0.7,
            "reason": "GROUP BY suggests categorical data"
        }

    return {
        "chart_type": "table",
        "confidence": 0.6,
        "reason": "Default to table for raw data"
    }


@mcp.tool
def list_charts(page: int = 0, page_size: int = 25) -> Dict[str, Any]:
    """
    List existing charts in Superset.

    Args:
        page: Page number
        page_size: Items per page

    Returns:
        List of charts with metadata
    """
    params = {"page": page, "page_size": page_size}
    data = client.get("/api/v1/chart/", params=params)

    charts = []
    for row in data.get("result", []):
        charts.append({
            "id": row.get("id"),
            "slice_name": row.get("slice_name"),
            "viz_type": row.get("viz_type"),
            "datasource_id": row.get("datasource_id"),
            "created_on": row.get("created_on"),
            "changed_on": row.get("changed_on"),
        })

    return {"count": data.get("count", 0), "charts": charts}


@mcp.tool
def get_chart(chart_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a chart.

    Args:
        chart_id: The chart ID

    Returns:
        Chart configuration and metadata
    """
    data = client.get(f"/api/v1/chart/{chart_id}")
    result = data.get("result", {})

    return {
        "id": result.get("id"),
        "slice_name": result.get("slice_name"),
        "viz_type": result.get("viz_type"),
        "datasource_id": result.get("datasource_id"),
        "datasource_type": result.get("datasource_type"),
        "params": result.get("params"),
        "query_context": result.get("query_context"),
    }


@mcp.tool
def create_chart(
    chart_name: str,
    viz_type: str,
    datasource_id: int,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new chart in Superset.

    Args:
        chart_name: Name for the chart
        viz_type: Visualization type (e.g., "table", "bar", "line", "pie")
        datasource_id: Dataset ID to use
        params: Chart parameters (metrics, dimensions, filters, etc.)

    Returns:
        Created chart info
    """
    if READ_ONLY:
        return {"error": "READ_ONLY mode enabled, cannot create charts"}

    payload = {
        "slice_name": chart_name,
        "viz_type": viz_type,
        "datasource_id": datasource_id,
        "datasource_type": "table",
        "params": json.dumps(params),
    }

    result = client.post("/api/v1/chart/", payload, needs_csrf=True)
    return {"status": "created", "chart": result}


@mcp.tool
def update_chart(
    chart_id: int,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing chart.

    Args:
        chart_id: Chart ID to update
        updates: Fields to update (slice_name, viz_type, params, etc.)

    Returns:
        Updated chart info
    """
    if READ_ONLY:
        return {"error": "READ_ONLY mode enabled, cannot update charts"}

    # Convert params to JSON string if provided as dict
    if "params" in updates and isinstance(updates["params"], dict):
        updates["params"] = json.dumps(updates["params"])

    result = client.post(f"/api/v1/chart/{chart_id}", updates, needs_csrf=True)
    return {"status": "updated", "chart": result}


# ===========================================================================
# SECTION 7: UTILITY TOOLS
# ===========================================================================

@mcp.tool
def list_relationships(dataset_ids: List[int]) -> Dict[str, Any]:
    """Return relationship hints from relationships.yaml for these datasets."""
    rels = _load_relationships()
    wanted = set(dataset_ids)
    filtered = []
    for r in rels:
        if r.get("left_dataset_id") in wanted and r.get("right_dataset_id") in wanted:
            filtered.append(r)
    return {"relationships": filtered}


@mcp.tool
def get_server_info() -> Dict[str, Any]:
    """Get MCP server information and status."""
    matcher = _get_semantic_matcher()
    stats = matcher.get_statistics()

    # Get available LLM providers
    try:
        from llm import get_llm_service
        llm_service = get_llm_service()
        llm_providers = llm_service.get_available_providers()
    except Exception:
        llm_providers = []

    return {
        "server": "superset-mcp",
        "version": "2.1",
        "superset_url": SUPERSET_URL,
        "read_only": READ_ONLY,
        "engine_hint": ENGINE_HINT,
        "semantic_matcher": {
            "tables": stats["total_tables"],
            "columns": stats["total_columns"],
            "join_patterns": stats["join_patterns"],
            "synonyms": stats["synonyms"],
            "schemas": stats["schemas"]
        },
        "llm_providers": llm_providers,
        "cache_ttl_seconds": CACHE_TTL,
        "timestamp": datetime.now().isoformat()
    }


# ===========================================================================
# SECTION 8: DASHBOARD TOOLS
# ===========================================================================

@mcp.tool
def list_dashboards(page: int = 0, page_size: int = 25) -> Dict[str, Any]:
    """
    List existing dashboards in Superset.

    Args:
        page: Page number (0-indexed)
        page_size: Number of results per page

    Returns:
        List of dashboards with basic info
    """
    data = client.get("/api/v1/dashboard/", params={"page": page, "page_size": page_size})
    dashboards = []
    for d in data.get("result", []):
        dashboards.append({
            "id": d.get("id"),
            "dashboard_title": d.get("dashboard_title"),
            "status": d.get("status"),
            "published": d.get("published"),
            "url": f"{SUPERSET_URL}/superset/dashboard/{d.get('id')}/"
        })
    return {"dashboards": dashboards, "count": data.get("count", len(dashboards))}


@mcp.tool
def get_dashboard(dashboard_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a dashboard.

    Args:
        dashboard_id: The dashboard ID

    Returns:
        Dashboard details including charts and layout
    """
    data = client.get(f"/api/v1/dashboard/{dashboard_id}")
    result = data.get("result", data)

    return {
        "id": result.get("id"),
        "dashboard_title": result.get("dashboard_title"),
        "slug": result.get("slug"),
        "published": result.get("published"),
        "charts": result.get("charts", []),
        "url": f"{SUPERSET_URL}/superset/dashboard/{dashboard_id}/"
    }


@mcp.tool
def create_dashboard(
    dashboard_title: str,
    slug: Optional[str] = None,
    published: bool = False
) -> Dict[str, Any]:
    """
    Create a new dashboard in Superset.

    Args:
        dashboard_title: Title for the dashboard
        slug: URL slug (auto-generated if not provided)
        published: Whether to publish immediately

    Returns:
        Created dashboard info with ID and URL
    """
    if READ_ONLY:
        return {"error": "Cannot create dashboard in READ_ONLY mode"}

    payload = {
        "dashboard_title": dashboard_title,
        "published": published
    }
    if slug:
        payload["slug"] = slug

    data = client.post("/api/v1/dashboard/", payload)

    dashboard_id = data.get("id")
    return {
        "id": dashboard_id,
        "dashboard_title": dashboard_title,
        "url": f"{SUPERSET_URL}/superset/dashboard/{dashboard_id}/",
        "edit_url": f"{SUPERSET_URL}/superset/dashboard/{dashboard_id}/edit"
    }


@mcp.tool
def add_chart_to_dashboard(
    dashboard_id: int,
    chart_id: int
) -> Dict[str, Any]:
    """
    Add an existing chart to a dashboard.

    Args:
        dashboard_id: Target dashboard ID
        chart_id: Chart ID to add

    Returns:
        Updated dashboard info
    """
    if READ_ONLY:
        return {"error": "Cannot modify dashboard in READ_ONLY mode"}

    # Get current dashboard
    dashboard = client.get(f"/api/v1/dashboard/{dashboard_id}")
    result = dashboard.get("result", dashboard)

    # Get current chart IDs
    current_charts = [c.get("id") for c in result.get("charts", [])]
    if chart_id in current_charts:
        return {"message": "Chart already in dashboard", "dashboard_id": dashboard_id}

    # Update dashboard with new chart
    # Note: This uses the position_json to add the chart
    current_charts.append(chart_id)

    # Update via PUT
    data = client.post(
        f"/api/v1/dashboard/{dashboard_id}",
        {"charts": current_charts},
        needs_csrf=True
    )

    return {
        "dashboard_id": dashboard_id,
        "chart_id": chart_id,
        "message": "Chart added to dashboard",
        "url": f"{SUPERSET_URL}/superset/dashboard/{dashboard_id}/"
    }


@mcp.tool
def get_chart_types() -> Dict[str, Any]:
    """
    Get available chart types with descriptions for user selection.

    Returns:
        List of chart types with icons and descriptions
    """
    chart_types = [
        {
            "id": "table",
            "name": "Table",
            "icon": "table",
            "description": "Display data in rows and columns",
            "best_for": "Detailed data inspection, exports"
        },
        {
            "id": "big_number_total",
            "name": "Big Number",
            "icon": "number",
            "description": "Single large metric value",
            "best_for": "KPIs, single important metrics"
        },
        {
            "id": "echarts_timeseries_line",
            "name": "Line Chart",
            "icon": "line-chart",
            "description": "Show trends over time",
            "best_for": "Time series data, trend analysis"
        },
        {
            "id": "echarts_timeseries_bar",
            "name": "Bar Chart",
            "icon": "bar-chart",
            "description": "Compare values across categories",
            "best_for": "Category comparisons, rankings"
        },
        {
            "id": "pie",
            "name": "Pie Chart",
            "icon": "pie-chart",
            "description": "Show proportions of a whole",
            "best_for": "Part-to-whole relationships (< 7 categories)"
        },
        {
            "id": "echarts_area",
            "name": "Area Chart",
            "icon": "area-chart",
            "description": "Filled line chart showing volume",
            "best_for": "Cumulative data, stacked comparisons"
        },
        {
            "id": "dist_bar",
            "name": "Distribution Bar",
            "icon": "bar-chart-horizontal",
            "description": "Horizontal bar chart for distributions",
            "best_for": "Comparing many categories"
        },
        {
            "id": "treemap",
            "name": "Treemap",
            "icon": "treemap",
            "description": "Hierarchical data as nested rectangles",
            "best_for": "Hierarchical proportions"
        },
        {
            "id": "heatmap",
            "name": "Heatmap",
            "icon": "heat-map",
            "description": "Color-coded matrix of values",
            "best_for": "Two-dimensional comparisons, correlations"
        },
        {
            "id": "scatter",
            "name": "Scatter Plot",
            "icon": "scatter-plot",
            "description": "Show relationship between two variables",
            "best_for": "Correlation analysis, outlier detection"
        }
    ]

    return {"chart_types": chart_types}


# ===========================================================================
# SECTION 9: LLM PROVIDER TOOLS
# ===========================================================================

@mcp.tool
def get_llm_providers() -> Dict[str, Any]:
    """
    Get available LLM providers for SQL generation.

    Returns:
        List of configured providers with availability status
    """
    try:
        from llm import get_llm_service
        llm_service = get_llm_service()
        providers = llm_service.get_available_providers()
        return {
            "providers": providers,
            "default": providers[0]["id"] if providers else None
        }
    except Exception as e:
        return {"providers": [], "error": str(e)}


@mcp.tool
def generate_sql_with_llm(
    query: str,
    tables: List[str],
    provider: str = "openai"
) -> Dict[str, Any]:
    """
    Generate SQL using specified LLM provider.

    Args:
        query: Natural language query
        tables: List of table names to use
        provider: LLM provider (openai, claude, gemini)

    Returns:
        Generated SQL with explanation and suggested title
    """
    try:
        from llm import get_llm_service, LLMProvider

        # Build context from semantic matcher
        context = _generate_sql_context_internal(tables, query, [])

        # Get LLM service
        llm_service = get_llm_service()

        # Map provider string to enum
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "claude": LLMProvider.CLAUDE,
            "gemini": LLMProvider.GEMINI
        }
        llm_provider = provider_map.get(provider.lower(), LLMProvider.OPENAI)

        # Generate SQL
        result = llm_service.generate_sql(query, context, llm_provider)

        return {
            "sql": result.sql,
            "explanation": result.explanation,
            "assumptions": result.assumptions,
            "suggested_title": result.suggested_title,
            "confidence": result.confidence,
            "provider": result.provider,
            "model": result.model
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def suggest_chart_for_query(
    sql: str,
    query: str,
    provider: str = "openai"
) -> Dict[str, Any]:
    """
    Suggest best chart type for a SQL query result.

    Args:
        sql: The SQL query
        query: Original natural language query
        provider: LLM provider to use

    Returns:
        Chart type recommendation with alternatives
    """
    try:
        from llm import get_llm_service, LLMProvider

        llm_service = get_llm_service()

        provider_map = {
            "openai": LLMProvider.OPENAI,
            "claude": LLMProvider.CLAUDE,
            "gemini": LLMProvider.GEMINI
        }
        llm_provider = provider_map.get(provider.lower(), LLMProvider.OPENAI)

        # Extract columns from SQL (simple parsing)
        import re
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            columns_str = select_match.group(1)
            columns = [c.strip().split(' ')[-1].split('.')[-1] for c in columns_str.split(',')]
        else:
            columns = ["result"]

        result = llm_service.suggest_chart_type(sql, query, columns, llm_provider)

        return result
    except Exception as e:
        return {"error": str(e), "recommended_type": "table"}


if __name__ == "__main__":
    mcp.run()