"""
Unified Query Engine for natural language to SQL conversion.

This module provides the core orchestration logic that can be used by:
- REST API (chatbot/api_server.py)
- CLI tools
- Other integration layers

The QueryEngine is designed to be:
- Stateless (state is passed in/out, not stored internally)
- Reusable (all methods can be called independently)
- Extensible (subclass to customize behavior)

Architecture:
    QueryEngine (this file)
        ├── Query Analysis (entity extraction, table matching)
        ├── Table Resolution (disambiguation, recommendations)
        ├── Join Detection (semantic pattern matching)
        ├── SQL Generation (multi-provider LLM support)
        ├── SQL Validation (syntax, safety, best practices)
        ├── SQL Improvement (critique and repair)
        ├── Execution (query execution, saving)
        └── Visualization (chart type inference)

Usage:
    from query_engine import QueryEngine

    engine = QueryEngine()

    # Full pipeline
    result = engine.process_query(
        "Show me all users with their orders",
        table_selections={"users": "user_details"},  # Optional disambiguation
    )

    # Or step by step
    plan = engine.analyze_query("Show me all users")
    tables = engine.resolve_tables(plan)
    sql_result = engine.generate_sql("Show me all users", tables)
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class QueryStep(Enum):
    """Steps in the query processing pipeline."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    CHOOSING_TABLE = "choosing_table"
    CHOOSING_DATE_SOURCE = "choosing_date_source"
    CONFIRMING_JOINS = "confirming_joins"
    GENERATING_SQL = "generating_sql"
    SQL_READY = "sql_ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class QueryPlan:
    """Result of query analysis."""
    original_query: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    table_matches: Dict[str, Any] = field(default_factory=dict)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    time_filters: List[Dict[str, Any]] = field(default_factory=list)
    state_filters: List[Dict[str, Any]] = field(default_factory=list)
    extracted_schema: Optional[str] = None
    ambiguous_entities: List[str] = field(default_factory=list)

    def needs_user_input(self) -> bool:
        """Check if any entities need user disambiguation."""
        return len(self.ambiguous_entities) > 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JoinSuggestion:
    """A suggested join between tables."""
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: str = "INNER"
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_sql_clause(self) -> str:
        """Generate SQL JOIN clause."""
        return (
            f"{self.join_type} JOIN {self.right_table} "
            f"ON {self.left_table}.{self.left_column} = {self.right_table}.{self.right_column}"
        )


@dataclass
class SQLResult:
    """Result of SQL generation."""
    sql: str
    explanation: str = ""
    assumptions: List[str] = field(default_factory=list)
    suggested_title: str = ""
    confidence: float = 0.0
    provider: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    is_valid: bool
    can_execute: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResult:
    """Result of query execution."""
    success: bool
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    data: List[List[Any]] = field(default_factory=list)
    error: Optional[str] = None
    saved_query_id: Optional[int] = None
    sqllab_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Truncate data for serialization
        if len(result.get("data", [])) > 100:
            result["data"] = result["data"][:100]
            result["data_truncated"] = True
        return result


@dataclass
class ChartSuggestion:
    """Suggested chart configuration."""
    chart_type: str
    confidence: float
    reason: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    group_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryState:
    """
    Complete state of a query session.

    This is passed between methods to track progress.
    Can be serialized/deserialized for REST API session management.
    """
    session_id: str
    step: QueryStep = QueryStep.IDLE
    query: Optional[str] = None
    plan: Optional[QueryPlan] = None
    tables: List[str] = field(default_factory=list)
    joins: List[JoinSuggestion] = field(default_factory=list)
    sql_result: Optional[SQLResult] = None
    validation: Optional[ValidationResult] = None
    execution: Optional[ExecutionResult] = None
    database_id: Optional[int] = None
    schema: Optional[str] = None
    pending_choices: List[Dict[str, Any]] = field(default_factory=list)
    pending_entity: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for REST API."""
        return {
            "session_id": self.session_id,
            "step": self.step.value,
            "query": self.query,
            "tables": self.tables,
            "joins": [j.to_dict() for j in self.joins],
            "sql": self.sql_result.sql if self.sql_result else None,
            "database_id": self.database_id,
            "schema": self.schema,
            "pending_choices": self.pending_choices,
            "pending_entity": self.pending_entity,
            "error": self.error,
        }


# =============================================================================
# Query Engine - Core Class
# =============================================================================

class QueryEngine:
    """
    Core query engine for natural language to SQL conversion.

    This class provides a unified interface for:
    - Query analysis (entity extraction, table matching)
    - SQL generation (using LLM providers)
    - SQL validation and improvement
    - Query execution

    It's designed to be stateless - all state is passed via QueryState.
    """

    def __init__(self, llm_provider: str = None):
        """
        Initialize the query engine.

        Args:
            llm_provider: Preferred LLM provider ('openai', 'claude', 'gemini')
                         If None, uses the default from config.
        """
        self._mcp_tools = None
        self._llm_service = None
        self._knowledge_store = None
        self._example_generator = None
        self._llm_provider = llm_provider
        self._config = None

    # =========================================================================
    # Lazy Loading Helpers
    # =========================================================================

    def _get_config(self):
        """Lazy load config."""
        if self._config is None:
            from config import load_config
            try:
                self._config = load_config()
            except Exception:
                self._config = None
        return self._config

    def _get_mcp_tools(self) -> Dict[str, Any]:
        """Lazy load MCP tools."""
        if self._mcp_tools is None:
            import mcp_superset_server as mcp
            self._mcp_tools = {
                "build_execution_plan": mcp.build_execution_plan.fn,
                "find_join_path": mcp.find_join_path.fn,
                "suggest_joins": mcp.suggest_joins.fn,
                "generate_sql_context": mcp.generate_sql_context.fn,
                "validate_sql": mcp.validate_sql.fn,
                "execute_and_save": mcp.execute_and_save.fn,
                "get_sqllab_url": mcp.get_sqllab_url.fn,
                "create_saved_query": mcp.create_saved_query.fn,
                "get_table_info": mcp.get_table_info.fn,
                "get_dataset": mcp.get_dataset.fn,
                "discover_tables": mcp.discover_tables.fn,
                "infer_chart_type": mcp.infer_chart_type.fn,
                "get_server_info": mcp.get_server_info.fn,
            }
        return self._mcp_tools

    def _get_llm_service(self):
        """Lazy load LLM service."""
        if self._llm_service is None:
            from llm import LLMService, LLMProvider
            self._llm_service = LLMService()
            if self._llm_provider:
                try:
                    provider = LLMProvider(self._llm_provider)
                    self._llm_service.set_provider(provider)
                except ValueError:
                    pass
        return self._llm_service

    def _get_knowledge_store(self):
        """Lazy load knowledge store."""
        if self._knowledge_store is None:
            from learning import KnowledgeStore
            self._knowledge_store = KnowledgeStore()
        return self._knowledge_store

    def _get_example_generator(self):
        """Lazy load example generator."""
        if self._example_generator is None:
            from learning import ExampleGenerator
            from config.semantic_config_manager import get_config_manager

            knowledge_store = self._get_knowledge_store()
            try:
                config_manager = get_config_manager()
                semantic_config = config_manager._config or {}
            except Exception:
                semantic_config = {}

            self._example_generator = ExampleGenerator(knowledge_store, semantic_config)
        return self._example_generator

    # =========================================================================
    # Step 1: Query Analysis
    # =========================================================================

    def analyze_query(self, query: str, auto_select: bool = False) -> QueryPlan:
        """
        Analyze a natural language query.

        Extracts:
        - Entities (dataset/table references)
        - Filters (WHERE conditions)
        - Time filters (date-based conditions)
        - State filters (status conditions like "not completed")
        - Schema references

        Args:
            query: Natural language query
            auto_select: If True, auto-select best matches without user input

        Returns:
            QueryPlan with extracted entities, table matches, filters, etc.
        """
        tools = self._get_mcp_tools()
        plan = tools["build_execution_plan"](query, auto_select=auto_select)

        result = QueryPlan(
            original_query=query,
            entities=plan.get("entities", []),
            table_matches=plan.get("table_matches", {}),
            filters=plan.get("filters", []),
            time_filters=plan.get("time_filters", []),
            state_filters=plan.get("state_filters", []),
            extracted_schema=plan.get("extracted_schema"),
        )

        # Identify ambiguous matches
        for entity, match_data in result.table_matches.items():
            if match_data.get("is_ambiguous", False):
                result.ambiguous_entities.append(entity)

        return result

    # =========================================================================
    # Step 2: Table Resolution
    # =========================================================================

    def resolve_tables(
        self,
        plan: QueryPlan,
        user_selections: Dict[str, str] = None
    ) -> List[str]:
        """
        Resolve table names from the query plan.

        Args:
            plan: QueryPlan from analyze_query()
            user_selections: Dict mapping entity names to selected table names

        Returns:
            List of resolved table names
        """
        user_selections = user_selections or {}
        tables = []

        for entity, match_data in plan.table_matches.items():
            if entity in user_selections:
                tables.append(user_selections[entity])
            elif match_data.get("recommended"):
                tables.append(match_data["recommended"]["table_name"])
            elif match_data.get("candidates"):
                tables.append(match_data["candidates"][0]["table_name"])

        return list(set(tables))

    def get_table_candidates(
        self,
        plan: QueryPlan,
        entity: str
    ) -> List[Dict[str, Any]]:
        """Get candidate tables for an ambiguous entity."""
        match_data = plan.table_matches.get(entity, {})
        return match_data.get("candidates", [])

    def get_tables_with_date_columns(
        self,
        plan: QueryPlan,
        entity: str
    ) -> List[Dict[str, Any]]:
        """Get alternative tables with date columns for time filtering."""
        from nlu.table_intelligence import get_table_intelligence
        intelligence = get_table_intelligence()

        candidates = self.get_table_candidates(plan, entity)
        if not candidates:
            return []

        # Enhance candidates with date column info
        enhanced = intelligence.recommend_tables_for_query(
            entity=entity,
            candidates=candidates,
            has_time_filter=bool(plan.time_filters),
        )

        return [c for c in enhanced if c.get("has_date_columns")]

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed table information including columns and database."""
        tools = self._get_mcp_tools()
        return tools["get_table_info"](table_name)

    def get_database_id_for_table(self, table_name: str) -> Tuple[int, Optional[str]]:
        """Get database_id and schema for a table."""
        tools = self._get_mcp_tools()
        table_info = tools["get_table_info"](table_name)

        database_id = table_info.get("database_id")
        schema = table_info.get("schema")

        if not database_id:
            dataset_id = table_info.get("dataset_id")
            if dataset_id:
                dataset_info = tools["get_dataset"](dataset_id)
                database_id = dataset_info.get("database_id", 1)
                schema = schema or dataset_info.get("schema")

        return database_id or 1, schema

    # =========================================================================
    # Step 3: Join Detection
    # =========================================================================

    def find_joins(
        self,
        tables: List[str],
        max_depth: int = 2
    ) -> List[JoinSuggestion]:
        """
        Find suggested joins between tables using semantic patterns.

        Args:
            tables: List of table names
            max_depth: Maximum join path depth

        Returns:
            List of JoinSuggestion objects
        """
        if len(tables) < 2:
            return []

        tools = self._get_mcp_tools()
        join_result = tools["suggest_joins"](tables)

        suggestions = []
        for join in join_result.get("joins", []):
            suggestions.append(JoinSuggestion(
                left_table=join.get("left_table", ""),
                left_column=join.get("left_column", ""),
                right_table=join.get("right_table", ""),
                right_column=join.get("right_column", ""),
                join_type=join.get("join_type", "INNER"),
                confidence=join.get("confidence", 0.0),
                reason=join.get("reason", ""),
            ))

        return suggestions

    # =========================================================================
    # Step 4: SQL Generation
    # =========================================================================

    def generate_sql(
        self,
        query: str,
        tables: List[str],
        joins: List[JoinSuggestion] = None,
        filters: List[Dict] = None,
        state_filters: List[Dict] = None,
        schema: str = None,
        provider: str = None
    ) -> SQLResult:
        """
        Generate SQL from natural language query using LLM.

        Uses few-shot examples from learned queries when available.

        Args:
            query: Original natural language query
            tables: List of table names
            joins: List of JoinSuggestion objects
            filters: List of filter dicts
            state_filters: List of state filter dicts
            schema: Database schema name
            provider: LLM provider to use

        Returns:
            SQLResult with generated SQL and metadata
        """
        tools = self._get_mcp_tools()
        llm_service = self._get_llm_service()

        # Build context
        join_hints = [j.to_dict() for j in (joins or [])]
        context = tools["generate_sql_context"](
            tables, query, join_hints, schema, state_filters
        )

        # Get few-shot examples from learned queries
        examples = []
        learned_patterns = None
        try:
            example_gen = self._get_example_generator()
            examples = example_gen.get_relevant_examples(
                user_query=query,
                tables=tables,
                max_examples=3
            )
            if tables:
                learned_patterns = example_gen.get_learned_patterns_for_tables(tables)
        except Exception as e:
            logger.debug(f"Could not get examples: {e}")

        # Set provider if specified
        if provider:
            from llm import LLMProvider
            try:
                llm_service.set_provider(LLMProvider(provider))
            except ValueError:
                pass

        # Generate SQL
        result = llm_service.generate_sql(
            query=query,
            context=context,
            examples=examples,
            learned_patterns=learned_patterns
        )

        return SQLResult(
            sql=result.sql,
            explanation=result.explanation,
            assumptions=result.assumptions,
            suggested_title=result.suggested_title,
            confidence=result.confidence,
            provider=result.provider,
            model=result.model,
        )

    def modify_sql(
        self,
        original_sql: str,
        modification_request: str,
        tables: List[str],
        provider: str = None
    ) -> SQLResult:
        """
        Modify existing SQL based on user request.

        Args:
            original_sql: The current SQL
            modification_request: What changes the user wants
            tables: List of table names
            provider: LLM provider to use

        Returns:
            SQLResult with modified SQL
        """
        llm_service = self._get_llm_service()

        if provider:
            from llm import LLMProvider
            try:
                llm_service.set_provider(LLMProvider(provider))
            except ValueError:
                pass

        result = llm_service.modify_sql(
            current_sql=original_sql,
            modification_request=modification_request,
            tables=tables
        )

        return SQLResult(
            sql=result.sql,
            explanation=result.explanation,
            assumptions=result.assumptions,
            suggested_title=result.suggested_title,
            confidence=result.confidence,
            provider=result.provider,
            model=result.model,
        )

    # =========================================================================
    # Step 5: SQL Validation
    # =========================================================================

    def validate_sql(self, sql: str, strict: bool = False) -> ValidationResult:
        """
        Validate SQL syntax and safety.

        Checks:
        - Syntax (balanced parentheses, quotes)
        - Safety (blocks dangerous operations in READ_ONLY mode)
        - Best practices (SELECT *, missing LIMIT)
        - Column references against schema

        Args:
            sql: SQL query to validate
            strict: If True, fail on warnings too

        Returns:
            ValidationResult with errors, warnings, suggestions
        """
        tools = self._get_mcp_tools()
        result = tools["validate_sql"](sql, strict=strict)

        return ValidationResult(
            is_valid=result.get("is_valid", False),
            can_execute=result.get("can_execute", False),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
            suggestions=result.get("suggestions", []),
        )

    # =========================================================================
    # Step 6: SQL Improvement (Gemini Critique)
    # =========================================================================

    def improve_sql(
        self,
        sql: str,
        query: str,
        tables: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Review and improve SQL using Gemini.

        Args:
            sql: SQL to review
            query: Original natural language query
            tables: Tables being queried

        Returns:
            Tuple of (improved_sql, notes)
        """
        config = self._get_config()
        if not config or not config.llm.gemini_api_key:
            return sql, ["Gemini not configured; skipped critique."]

        try:
            from google import genai
        except ImportError:
            return sql, ["google-genai package not installed; skipped critique."]

        tools = self._get_mcp_tools()
        context = tools["generate_sql_context"](tables, query, [], None)

        # Build table descriptions
        tables_desc = []
        for t in context.get("tables", []):
            cols = ", ".join(t.get("columns", []))
            tables_desc.append(f"- {t.get('schema', 'default')}.{t['table_name']}: {cols}")

        prompt = f"""Review and improve this SQL query for {context.get('engine', 'postgresql')}.

User Request: {query}

Available Tables:
{chr(10).join(tables_desc)}

SQL to Review:
{sql}

Check for:
- Incorrect joins or join conditions
- Type mismatches
- Missing filters or GROUP BY
- Ambiguous column references
- SQL syntax errors
- Performance issues (missing LIMIT, SELECT *)

Return JSON only with keys:
- approved (bool): true if SQL is correct, false if modified
- sql (string): the corrected SQL or original if approved
- notes (array of strings): list of changes made or issues found"""

        try:
            gclient = genai.Client(api_key=config.llm.gemini_api_key)
            response = gclient.models.generate_content(
                model=config.llm.gemini_model,
                contents=prompt,
            )

            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            result = json.loads(text)
            return result.get("sql", sql), result.get("notes", [])

        except Exception as e:
            logger.warning(f"Gemini critique failed: {e}")
            return sql, [f"Critique failed: {str(e)}"]

    def repair_sql(
        self,
        sql: str,
        error_message: str
    ) -> Tuple[str, List[str]]:
        """
        Attempt to repair failed SQL using Gemini.

        Args:
            sql: Failed SQL
            error_message: Error from execution

        Returns:
            Tuple of (repaired_sql, fix_notes)
        """
        config = self._get_config()
        if not config or not config.llm.gemini_api_key:
            return sql, ["Gemini not configured; cannot repair."]

        try:
            from google import genai
        except ImportError:
            return sql, ["google-genai package not installed; cannot repair."]

        prompt = f"""The following SQL failed with this error:
Error: {error_message}

SQL:
{sql}

Please fix the SQL to resolve the error. Return JSON with:
- sql (string): the corrected SQL
- fix_notes (array): what was changed"""

        try:
            gclient = genai.Client(api_key=config.llm.gemini_api_key)
            response = gclient.models.generate_content(
                model=config.llm.gemini_model,
                contents=prompt,
            )

            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            result = json.loads(text)
            return result.get("sql", sql), result.get("fix_notes", [])

        except Exception as e:
            logger.warning(f"SQL repair failed: {e}")
            return sql, [f"Repair failed: {str(e)}"]

    # =========================================================================
    # Step 7: Execution
    # =========================================================================

    def execute_sql(
        self,
        sql: str,
        database_id: int,
        schema: str = None,
        limit: int = 1000,
        save: bool = False,
        label: str = None
    ) -> ExecutionResult:
        """
        Execute SQL query.

        Args:
            sql: SQL query to execute
            database_id: Database ID in Superset
            schema: Schema name
            limit: Row limit
            save: Whether to save as a query
            label: Query label if saving

        Returns:
            ExecutionResult with data and metadata
        """
        tools = self._get_mcp_tools()

        try:
            result = tools["execute_and_save"](
                sql=sql,
                database_id=database_id,
                schema=schema,
                query_limit=limit,
                save_query=save,
                label=label
            )

            execution = result.get("execution", {})
            saved = result.get("saved_query", {})

            return ExecutionResult(
                success=True,
                row_count=execution.get("row_count", 0),
                columns=execution.get("columns", []),
                data=execution.get("data", []),
                saved_query_id=saved.get("id"),
                sqllab_url=saved.get("sqllab_url"),
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
            )

    def save_query(
        self,
        sql: str,
        database_id: int,
        label: str,
        schema: str = None,
        description: str = None
    ) -> Dict[str, Any]:
        """Save query to Superset SQL Lab without executing."""
        tools = self._get_mcp_tools()
        return tools["create_saved_query"](
            sql=sql,
            database_id=database_id,
            label=label,
            schema=schema,
            description=description
        )

    def get_sqllab_url(
        self,
        sql: str,
        database_id: int,
        schema: str = None
    ) -> str:
        """Get SQL Lab URL with pre-filled query."""
        tools = self._get_mcp_tools()
        result = tools["get_sqllab_url"](
            sql=sql,
            database_id=database_id,
            schema=schema
        )
        return result.get("url", result.get("sqllab_url", ""))

    # =========================================================================
    # Step 8: Chart Inference
    # =========================================================================

    def infer_chart_type(
        self,
        sql: str,
        query: str,
        columns: List[str]
    ) -> ChartSuggestion:
        """
        Infer appropriate chart type from query and results.

        Args:
            sql: The SQL query
            query: Original natural language query
            columns: Result columns

        Returns:
            ChartSuggestion with recommended visualization
        """
        tools = self._get_mcp_tools()
        result = tools["infer_chart_type"](
            sql=sql,
            query=query,
            result_columns=columns
        )

        return ChartSuggestion(
            chart_type=result.get("chart_type", "table"),
            confidence=result.get("confidence", 0.5),
            reason=result.get("reason", ""),
            x_axis=result.get("x_axis"),
            y_axis=result.get("y_axis"),
            group_by=result.get("group_by"),
        )

    # =========================================================================
    # High-Level Pipeline Methods
    # =========================================================================

    def process_query(
        self,
        query: str,
        table_selections: Dict[str, str] = None,
        confirm_joins: bool = True,
        improve_sql: bool = True,
        execute: bool = False,
        database_id: int = None,
        schema: str = None,
        provider: str = None
    ) -> Dict[str, Any]:
        """
        Process a query through the full pipeline.

        This is a convenience method that runs all steps sequentially.
        For interactive use cases, use individual step methods.

        Args:
            query: Natural language query
            table_selections: Pre-selected tables for disambiguation
            confirm_joins: Whether to include join suggestions
            improve_sql: Whether to run Gemini critique
            execute: Whether to execute the query
            database_id: Database ID (auto-detected if None)
            schema: Schema name
            provider: LLM provider

        Returns:
            Dict with all results from each step
        """
        result = {
            "query": query,
            "steps_completed": [],
        }

        try:
            # Step 1: Analyze
            plan = self.analyze_query(query)
            result["plan"] = plan.to_dict()
            result["steps_completed"].append("analyze")

            if plan.needs_user_input() and not table_selections:
                result["needs_disambiguation"] = True
                result["ambiguous_entities"] = plan.ambiguous_entities
                for entity in plan.ambiguous_entities:
                    result[f"candidates_{entity}"] = self.get_table_candidates(plan, entity)
                return result

            # Step 2: Resolve tables
            tables = self.resolve_tables(plan, table_selections)
            result["tables"] = tables
            result["steps_completed"].append("resolve_tables")

            if not tables:
                result["error"] = "No tables resolved"
                return result

            # Get database info
            if not database_id:
                database_id, detected_schema = self.get_database_id_for_table(tables[0])
                schema = schema or detected_schema or plan.extracted_schema

            result["database_id"] = database_id
            result["schema"] = schema

            # Step 3: Find joins
            joins = []
            if confirm_joins and len(tables) > 1:
                joins = self.find_joins(tables)
                result["joins"] = [j.to_dict() for j in joins]
            result["steps_completed"].append("find_joins")

            # Step 4: Generate SQL
            sql_result = self.generate_sql(
                query=query,
                tables=tables,
                joins=joins,
                filters=plan.filters,
                state_filters=plan.state_filters,
                schema=schema,
                provider=provider
            )
            result["sql"] = sql_result.sql
            result["sql_result"] = sql_result.to_dict()
            result["steps_completed"].append("generate_sql")

            # Step 5: Improve SQL
            if improve_sql:
                improved, notes = self.improve_sql(sql_result.sql, query, tables)
                if improved != sql_result.sql:
                    result["sql"] = improved
                    result["improvement_notes"] = notes
                result["steps_completed"].append("improve_sql")

            # Step 6: Validate
            validation = self.validate_sql(result["sql"])
            result["validation"] = validation.to_dict()
            result["steps_completed"].append("validate")

            # Step 7: Execute (optional)
            if execute and validation.can_execute:
                execution = self.execute_sql(
                    sql=result["sql"],
                    database_id=database_id,
                    schema=schema
                )
                result["execution"] = execution.to_dict()
                result["steps_completed"].append("execute")

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            logger.exception(f"Query processing failed: {e}")

        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def set_llm_provider(self, provider: str):
        """Set the LLM provider ('openai', 'claude', 'gemini')."""
        from llm import LLMProvider
        llm_service = self._get_llm_service()
        llm_service.set_provider(LLMProvider(provider))
        self._llm_provider = provider

    def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server and semantic matcher info."""
        tools = self._get_mcp_tools()
        return tools["get_server_info"]()

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get query learning statistics."""
        knowledge_store = self._get_knowledge_store()
        examples = knowledge_store.get_examples_by_tables([], limit=10000)

        by_source = {}
        by_type = {}
        for ex in examples:
            source = ex.get("source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1
            qtype = ex.get("query_type", "simple")
            by_type[qtype] = by_type.get(qtype, 0) + 1

        return {
            "total_examples": len(examples),
            "by_source": by_source,
            "by_type": by_type,
        }

    def record_feedback(
        self,
        query: str,
        sql: str,
        success: bool,
        error: str = None
    ):
        """Record query feedback for learning."""
        try:
            knowledge_store = self._get_knowledge_store()
            knowledge_store.record_query_feedback(
                natural_language=query,
                generated_sql=sql,
                feedback_type="success" if success else "failure",
                error_message=error
            )
        except Exception as e:
            logger.debug(f"Could not record feedback: {e}")


# =============================================================================
# Singleton Instance
# =============================================================================

_engine_instance = None


def get_query_engine(llm_provider: str = None) -> QueryEngine:
    """Get the global QueryEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = QueryEngine(llm_provider)
    return _engine_instance


def reset_query_engine():
    """Reset the global QueryEngine instance."""
    global _engine_instance
    _engine_instance = None
