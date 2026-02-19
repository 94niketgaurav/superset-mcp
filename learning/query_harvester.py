"""
Query Harvester - Fetch existing queries and charts from Superset.

Harvests:
- Saved queries from SQL Lab
- Chart configurations with SQL/query context
- Extracts SQL and metadata for learning
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class HarvestedQuery:
    """A query harvested from Superset."""
    source: str                         # "saved_query" | "chart" | "dataset"
    source_id: int                      # ID in Superset
    title: str                          # Human-readable title
    sql: str                            # The actual SQL query
    database_id: int
    schema: Optional[str] = None
    description: Optional[str] = None
    tables_used: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    chart_type: Optional[str] = None    # For charts: bar, line, pie, etc.
    datasource_id: Optional[int] = None # For charts: the dataset used
    dataset_name: Optional[str] = None  # For datasets: the dataset/table name
    is_virtual: bool = False            # For datasets: True if SQL-based virtual dataset

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarvestedQuery":
        return cls(**data)


class QueryHarvester:
    """
    Harvests queries and charts from Superset for learning.

    Uses the SupersetClient to fetch data via REST API.
    """

    def __init__(self, superset_client):
        """
        Initialize the harvester.

        Args:
            superset_client: SupersetClient instance for API calls
        """
        self.client = superset_client

    def harvest_all(
        self,
        include_saved_queries: bool = True,
        include_charts: bool = True,
        include_datasets: bool = True,
        since_date: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[HarvestedQuery]:
        """
        Harvest all queries, charts, and datasets from Superset.

        Args:
            include_saved_queries: Whether to harvest saved queries
            include_charts: Whether to harvest charts
            include_datasets: Whether to harvest datasets (virtual datasets with SQL)
            since_date: Only harvest items modified since this date (ISO format)
            progress_callback: Optional callback(stage, current, total)

        Returns:
            List of HarvestedQuery objects
        """
        results = []

        if include_saved_queries:
            logger.info("Harvesting saved queries...")
            saved = self.harvest_saved_queries(since_date, progress_callback)
            results.extend(saved)
            logger.info(f"Harvested {len(saved)} saved queries")

        if include_charts:
            logger.info("Harvesting charts...")
            charts = self.harvest_charts(since_date, progress_callback)
            results.extend(charts)
            logger.info(f"Harvested {len(charts)} charts")

        if include_datasets:
            logger.info("Harvesting datasets...")
            datasets = self.harvest_datasets(since_date, progress_callback)
            results.extend(datasets)
            logger.info(f"Harvested {len(datasets)} datasets with SQL")

        return results

    def harvest_saved_queries(
        self,
        since_date: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[HarvestedQuery]:
        """
        Harvest saved queries from SQL Lab.

        Args:
            since_date: Only harvest queries modified since this date
            progress_callback: Optional callback for progress

        Returns:
            List of HarvestedQuery objects
        """
        results = []
        page = 0
        page_size = 100

        while True:
            # Build query params
            params = {"page": page, "page_size": page_size}

            try:
                data = self.client.get("/api/v1/saved_query/", params=params)
                items = data.get("result", [])

                if not items:
                    break

                for item in items:
                    # Skip if older than since_date
                    if since_date:
                        changed_on = item.get("changed_on")
                        if changed_on and changed_on < since_date:
                            continue

                    sql = item.get("sql", "")
                    if not sql or not sql.strip():
                        continue  # Skip empty queries

                    query = HarvestedQuery(
                        source="saved_query",
                        source_id=item.get("id"),
                        title=item.get("label", "Untitled Query"),
                        sql=sql,
                        database_id=item.get("db_id") or item.get("database", {}).get("id", 0),
                        schema=item.get("schema"),
                        description=item.get("description"),
                        created_by=item.get("created_by", {}).get("username") if isinstance(item.get("created_by"), dict) else None,
                        created_at=item.get("created_on"),
                        last_modified=item.get("changed_on"),
                    )
                    results.append(query)

                if progress_callback:
                    progress_callback("saved_queries", len(results), -1)

                if len(items) < page_size:
                    break  # Last page

                page += 1

            except Exception as e:
                logger.error(f"Error fetching saved queries page {page}: {e}")
                break

        return results

    def harvest_charts(
        self,
        since_date: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[HarvestedQuery]:
        """
        Harvest charts with their SQL/query configuration.

        Args:
            since_date: Only harvest charts modified since this date
            progress_callback: Optional callback for progress

        Returns:
            List of HarvestedQuery objects
        """
        results = []
        page = 0
        page_size = 100

        while True:
            params = {"page": page, "page_size": page_size}

            try:
                data = self.client.get("/api/v1/chart/", params=params)
                items = data.get("result", [])

                if not items:
                    break

                for item in items:
                    # Skip if older than since_date
                    if since_date:
                        changed_on = item.get("changed_on")
                        if changed_on and changed_on < since_date:
                            continue

                    chart_id = item.get("id")
                    if not chart_id:
                        continue

                    # Fetch detailed chart info to get query_context
                    try:
                        chart_detail = self.client.get(f"/api/v1/chart/{chart_id}")
                        chart_data = chart_detail.get("result", chart_detail)
                    except Exception as e:
                        logger.warning(f"Could not fetch chart {chart_id} details: {e}")
                        continue

                    # Extract SQL from chart
                    sql = self._extract_sql_from_chart(chart_data)
                    if not sql:
                        continue  # Skip charts without SQL

                    # Get datasource info
                    datasource_id = chart_data.get("datasource_id")
                    datasource_type = chart_data.get("datasource_type", "table")

                    # Try to get database_id from datasource
                    database_id = 0
                    schema = None
                    if datasource_id and datasource_type == "table":
                        try:
                            ds_data = self.client.get(f"/api/v1/dataset/{datasource_id}")
                            ds = ds_data.get("result", ds_data)
                            database_id = ds.get("database", {}).get("id", 0) if isinstance(ds.get("database"), dict) else ds.get("database_id", 0)
                            schema = ds.get("schema")
                        except Exception:
                            pass

                    query = HarvestedQuery(
                        source="chart",
                        source_id=chart_id,
                        title=chart_data.get("slice_name", "Untitled Chart"),
                        sql=sql,
                        database_id=database_id,
                        schema=schema,
                        description=chart_data.get("description"),
                        chart_type=chart_data.get("viz_type"),
                        datasource_id=datasource_id,
                        created_by=chart_data.get("created_by", {}).get("username") if isinstance(chart_data.get("created_by"), dict) else None,
                        created_at=chart_data.get("created_on"),
                        last_modified=chart_data.get("changed_on"),
                    )
                    results.append(query)

                if progress_callback:
                    progress_callback("charts", len(results), -1)

                if len(items) < page_size:
                    break

                page += 1

            except Exception as e:
                logger.error(f"Error fetching charts page {page}: {e}")
                break

        return results

    def harvest_datasets(
        self,
        since_date: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[HarvestedQuery]:
        """
        Harvest datasets (virtual datasets) that have SQL queries.

        Virtual datasets in Superset are created from SQL queries.
        This method fetches those SQL queries for learning.

        Args:
            since_date: Only harvest datasets modified since this date
            progress_callback: Optional callback for progress

        Returns:
            List of HarvestedQuery objects for datasets with SQL
        """
        results = []
        page = 0
        page_size = 100
        total_datasets = 0
        virtual_count = 0

        while True:
            params = {"page": page, "page_size": page_size}

            try:
                data = self.client.get("/api/v1/dataset/", params=params)
                items = data.get("result", [])
                count_info = data.get("count", 0)
                if count_info and total_datasets == 0:
                    total_datasets = count_info

                if not items:
                    break

                for item in items:
                    # Skip if older than since_date
                    if since_date:
                        changed_on = item.get("changed_on")
                        if changed_on and changed_on < since_date:
                            continue

                    dataset_id = item.get("id")
                    if not dataset_id:
                        continue

                    # Fetch detailed dataset info to get the SQL
                    try:
                        dataset_detail = self.client.get(f"/api/v1/dataset/{dataset_id}")
                        dataset_data = dataset_detail.get("result", dataset_detail)
                    except Exception as e:
                        logger.warning(f"Could not fetch dataset {dataset_id} details: {e}")
                        continue

                    # Check if this is a virtual dataset with SQL
                    sql = dataset_data.get("sql")
                    is_virtual = dataset_data.get("is_sqllab_view", False)

                    # Some datasets may have SQL even if not marked as sqllab_view
                    if not sql:
                        # Try alternate field names
                        sql = dataset_data.get("select_star") or dataset_data.get("query")

                    if not sql or not sql.strip():
                        continue  # Skip datasets without SQL (physical tables)

                    # Get database info
                    database_info = dataset_data.get("database", {})
                    database_id = database_info.get("id", 0) if isinstance(database_info, dict) else dataset_data.get("database_id", 0)

                    # Get table/dataset name
                    table_name = dataset_data.get("table_name", "")
                    schema = dataset_data.get("schema")

                    # Extract columns info for better understanding
                    columns = dataset_data.get("columns", [])
                    column_names = [c.get("column_name") for c in columns if isinstance(c, dict) and c.get("column_name")]

                    # Build description from columns
                    description = dataset_data.get("description", "")
                    if not description and column_names:
                        description = f"Virtual dataset with columns: {', '.join(column_names[:10])}"
                        if len(column_names) > 10:
                            description += f" (+{len(column_names) - 10} more)"

                    # Extract metrics if available
                    metrics = dataset_data.get("metrics", [])
                    metric_info = []
                    for m in metrics:
                        if isinstance(m, dict):
                            metric_name = m.get("metric_name") or m.get("verbose_name")
                            expression = m.get("expression")
                            if metric_name and expression:
                                metric_info.append(f"{metric_name}: {expression}")

                    if metric_info:
                        description += f"\nMetrics: {'; '.join(metric_info[:5])}"

                    virtual_count += 1
                    query = HarvestedQuery(
                        source="dataset",
                        source_id=dataset_id,
                        title=table_name or f"Dataset {dataset_id}",
                        sql=sql,
                        database_id=database_id,
                        schema=schema,
                        description=description,
                        dataset_name=table_name,
                        is_virtual=is_virtual or bool(sql),
                        created_by=dataset_data.get("created_by", {}).get("username") if isinstance(dataset_data.get("created_by"), dict) else None,
                        created_at=dataset_data.get("created_on"),
                        last_modified=dataset_data.get("changed_on"),
                    )
                    results.append(query)

                if progress_callback:
                    progress_callback("datasets", len(results), total_datasets)

                if len(items) < page_size:
                    break  # Last page

                page += 1

            except Exception as e:
                logger.error(f"Error fetching datasets page {page}: {e}")
                break

        logger.info(f"Found {virtual_count} virtual datasets with SQL out of {total_datasets} total datasets")
        return results

    def _extract_sql_from_chart(self, chart_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract SQL from chart configuration.

        Charts can have SQL in multiple places:
        1. query_context -> queries[0] -> extras -> raw_sql
        2. params -> adhoc_filters with SQL expressions
        3. params -> custom_sql
        4. Virtual dataset: the underlying dataset's SQL

        Args:
            chart_data: Chart detail dict from API

        Returns:
            SQL string or None if no SQL found
        """
        sql = None

        # Method 1: Check query_context
        query_context = chart_data.get("query_context")
        if query_context:
            if isinstance(query_context, str):
                try:
                    query_context = json.loads(query_context)
                except json.JSONDecodeError:
                    pass

            if isinstance(query_context, dict):
                queries = query_context.get("queries", [])
                for q in queries:
                    extras = q.get("extras", {})
                    raw_sql = extras.get("raw_sql")
                    if raw_sql:
                        sql = raw_sql
                        break

        # Method 2: Check params for SQL
        params = chart_data.get("params")
        if params and not sql:
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {}

            if isinstance(params, dict):
                # Check for custom SQL in metrics
                metrics = params.get("metrics", [])
                for metric in metrics:
                    if isinstance(metric, dict) and metric.get("expressionType") == "SQL":
                        sql_expression = metric.get("sqlExpression")
                        if sql_expression:
                            # This is just an expression, not a full query
                            pass

                # Check for SQL in adhoc_filters
                adhoc_filters = params.get("adhoc_filters", [])
                for f in adhoc_filters:
                    if isinstance(f, dict) and f.get("expressionType") == "SQL":
                        sql_expr = f.get("sqlExpression")
                        if sql_expr:
                            # These are filter expressions, not full queries
                            pass

        # Method 3: Try to reconstruct SQL from chart type and params
        if not sql and params:
            reconstructed = self._reconstruct_sql_from_params(chart_data, params)
            if reconstructed:
                sql = reconstructed

        return sql

    def _reconstruct_sql_from_params(
        self,
        chart_data: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Optional[str]:
        """
        Reconstruct a SQL query from chart params.

        This attempts to build a SQL query from the chart's:
        - groupby columns
        - metrics (with aggregations)
        - filters
        - orderby

        Args:
            chart_data: Full chart data
            params: Parsed params dict

        Returns:
            Reconstructed SQL or None
        """
        try:
            # Get datasource name
            datasource_name = chart_data.get("datasource_name_text", "")
            if not datasource_name:
                return None

            # Parse datasource name (format: "table_name" or "schema.table_name")
            parts = datasource_name.split(".")
            table_name = parts[-1] if parts else ""
            if not table_name:
                return None

            # Build SELECT columns
            columns = []

            # Add groupby columns
            groupby = params.get("groupby", []) or params.get("columns", [])
            for col in groupby:
                if isinstance(col, str):
                    columns.append(col)
                elif isinstance(col, dict):
                    columns.append(col.get("column_name", col.get("label", "")))

            # Add metrics
            metrics = params.get("metrics", [])
            for metric in metrics:
                if isinstance(metric, str):
                    columns.append(f"COUNT(*)")  # Default for simple metrics
                elif isinstance(metric, dict):
                    agg = metric.get("aggregate", "COUNT")
                    col = metric.get("column", {}).get("column_name", "*")
                    label = metric.get("label", f"{agg}_{col}")
                    columns.append(f"{agg}({col}) AS {label}")

            if not columns:
                columns = ["*"]

            # Build WHERE clause
            where_parts = []
            filters = params.get("adhoc_filters", [])
            for f in filters:
                if isinstance(f, dict) and f.get("expressionType") == "SIMPLE":
                    col = f.get("subject")
                    op = f.get("operator", "==")
                    val = f.get("comparator")
                    if col and val is not None:
                        if isinstance(val, str):
                            where_parts.append(f"{col} {op} '{val}'")
                        else:
                            where_parts.append(f"{col} {op} {val}")

            # Build GROUP BY
            group_by = ", ".join(groupby) if groupby else ""

            # Build ORDER BY
            order_by_parts = []
            order_desc = params.get("order_desc", True)
            if metrics:
                first_metric = metrics[0]
                if isinstance(first_metric, dict):
                    order_col = first_metric.get("label", "")
                    if order_col:
                        order_by_parts.append(f"{order_col} {'DESC' if order_desc else 'ASC'}")

            # Assemble SQL
            sql = f"SELECT {', '.join(columns)} FROM {table_name}"

            if where_parts:
                sql += f" WHERE {' AND '.join(where_parts)}"

            if group_by:
                sql += f" GROUP BY {group_by}"

            if order_by_parts:
                sql += f" ORDER BY {', '.join(order_by_parts)}"

            row_limit = params.get("row_limit", 1000)
            if row_limit:
                sql += f" LIMIT {row_limit}"

            return sql

        except Exception as e:
            logger.debug(f"Could not reconstruct SQL: {e}")
            return None

    def harvest_query_history(
        self,
        database_id: Optional[int] = None,
        since_date: Optional[str] = None,
        limit: int = 500
    ) -> List[HarvestedQuery]:
        """
        Harvest SQL Lab query execution history.

        Note: This requires the query history API to be enabled in Superset.

        Args:
            database_id: Filter by database
            since_date: Only harvest queries executed since this date
            limit: Maximum number of queries to fetch

        Returns:
            List of HarvestedQuery objects
        """
        results = []

        try:
            params = {"page_size": min(limit, 100)}
            if database_id:
                params["q"] = json.dumps({
                    "filters": [{"col": "database_id", "opr": "eq", "value": database_id}]
                })

            data = self.client.get("/api/v1/query/", params=params)
            items = data.get("result", [])

            for item in items:
                if since_date:
                    start_time = item.get("start_time")
                    if start_time and start_time < since_date:
                        continue

                sql = item.get("sql", "")
                if not sql or not sql.strip():
                    continue

                query = HarvestedQuery(
                    source="query_history",
                    source_id=item.get("id"),
                    title=f"Query executed at {item.get('start_time', 'unknown')}",
                    sql=sql,
                    database_id=item.get("database_id", 0),
                    schema=item.get("schema"),
                    created_at=item.get("start_time"),
                    last_modified=item.get("end_time"),
                )
                results.append(query)

                if len(results) >= limit:
                    break

        except Exception as e:
            logger.warning(f"Could not fetch query history: {e}")

        return results
