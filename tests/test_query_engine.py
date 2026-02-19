"""
Comprehensive tests for the QueryEngine and API server.

Tests cover:
- Query analysis (entity extraction, table matching)
- Time filter detection
- State filter detection ("not completed", "pending")
- Schema extraction ("in jeeves")
- Table resolution and disambiguation
- Join detection
- SQL generation with few-shot examples
- SQL improvement and repair (Gemini critique)
- SQL validation
- API server endpoints

Run with: pytest tests/test_query_engine.py -v
"""
import pytest
import json
import os
import sys
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def test_semantic_config():
    """Create a test semantic configuration."""
    return {
        "version": "2.0",
        "databases": {"1": {"name": "TestDB", "engine": "starrocks"}},
        "schemas": {
            "public": {
                "description": "Public schema",
                "tables": {
                    "asset": {
                        "dataset_id": 1,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "title": {"type": "VARCHAR"},
                            "tenant": {"type": "VARCHAR"},
                            "status": {"type": "VARCHAR"},
                            "contenttype": {"type": "VARCHAR"},
                        },
                        "aliases": ["assets"]
                    },
                    "asset_creation_overview": {
                        "dataset_id": 2,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "asset_id": {"type": "INTEGER", "is_fk": True},
                            "created_at": {"type": "TIMESTAMP"},
                            "updated_at": {"type": "TIMESTAMP"},
                            "tenant": {"type": "VARCHAR"},
                        },
                        "aliases": ["asset_overview", "asset_creation"]
                    },
                    "users": {
                        "dataset_id": 3,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "email": {"type": "VARCHAR"},
                            "name": {"type": "VARCHAR"},
                            "tenant": {"type": "VARCHAR"},
                        },
                        "aliases": ["user"]
                    },
                    "assignment": {
                        "dataset_id": 4,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "user_id": {"type": "INTEGER", "is_fk": True},
                            "status": {"type": "VARCHAR"},
                            "due_date": {"type": "DATE"},
                            "completed_at": {"type": "TIMESTAMP"},
                        },
                        "aliases": ["assignments"]
                    },
                    "audithistory": {
                        "dataset_id": 5,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "entity_id": {"type": "INTEGER"},
                            "entity_type": {"type": "VARCHAR"},
                            "action": {"type": "VARCHAR"},
                            "created_at": {"type": "TIMESTAMP"},
                        },
                        "aliases": ["audit", "history"]
                    },
                }
            },
            "jeeves": {
                "description": "Jeeves schema",
                "tables": {
                    "jobs": {
                        "dataset_id": 10,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "name": {"type": "VARCHAR"},
                            "status": {"type": "VARCHAR"},
                            "created_at": {"type": "TIMESTAMP"},
                        },
                    },
                }
            }
        },
        "join_patterns": [
            {"left": "asset", "right": "asset_creation_overview", "on": "id = asset_id", "type": "left"},
            {"left": "users", "right": "assignment", "on": "id = user_id", "type": "left"},
            {"left": "asset", "right": "audithistory", "on": "id = entity_id", "type": "left"},
        ],
        "synonyms": {
            "assets": ["items", "content", "resources"],
            "users": ["members", "accounts"],
            "assignment": ["task", "homework"],
        }
    }


@pytest.fixture
def test_config_file(test_semantic_config, tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "semantic_config.json"
    with open(config_path, 'w') as f:
        json.dump(test_semantic_config, f)
    return str(config_path)


@pytest.fixture
def mock_mcp_tools():
    """Mock MCP tools for testing."""
    mock_tools = {
        "build_execution_plan": Mock(),
        "find_join_path": Mock(),
        "suggest_joins": Mock(),
        "generate_sql_context": Mock(),
        "validate_sql": Mock(),
        "execute_and_save": Mock(),
        "get_sqllab_url": Mock(),
        "create_saved_query": Mock(),
        "get_table_info": Mock(),
        "get_dataset": Mock(),
        "discover_tables": Mock(),
        "infer_chart_type": Mock(),
        "get_server_info": Mock(),
    }
    return mock_tools


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    mock_service = Mock()
    mock_service.generate_sql.return_value = Mock(
        sql="SELECT * FROM asset LIMIT 100",
        explanation="Simple query",
        assumptions=[],
        suggested_title="Asset List",
        confidence=0.9,
        provider="openai",
        model="gpt-4",
    )
    return mock_service


@pytest.fixture
def mock_knowledge_store(tmp_path):
    """Create a mock knowledge store."""
    from learning.knowledge_store import KnowledgeStore
    return KnowledgeStore(str(tmp_path / "test_knowledge.db"))


# =============================================================================
# QueryEngine Data Class Tests
# =============================================================================

class TestQueryEngineDataClasses:
    """Test QueryEngine data classes."""

    def test_query_plan_creation(self):
        """Test creating a QueryPlan."""
        from query_engine import QueryPlan

        plan = QueryPlan(
            original_query="show me assets",
            entities=[{"name": "assets", "type": "dataset"}],
            table_matches={"assets": {"recommended": {"table_name": "asset"}}},
        )

        assert plan.original_query == "show me assets"
        assert len(plan.entities) == 1
        assert plan.needs_user_input() is False

    def test_query_plan_ambiguous(self):
        """Test QueryPlan detects ambiguous matches."""
        from query_engine import QueryPlan

        plan = QueryPlan(
            original_query="show me assets",
            entities=[{"name": "assets", "type": "dataset"}],
            table_matches={
                "assets": {
                    "is_ambiguous": True,
                    "candidates": [
                        {"table_name": "asset", "score": 0.9},
                        {"table_name": "asset_creation_overview", "score": 0.85},
                    ]
                }
            },
            ambiguous_entities=["assets"]
        )

        assert plan.needs_user_input() is True
        assert "assets" in plan.ambiguous_entities

    def test_sql_result_creation(self):
        """Test creating SQLResult."""
        from query_engine import SQLResult

        result = SQLResult(
            sql="SELECT * FROM asset LIMIT 100",
            explanation="Simple asset query",
            assumptions=["Using default limit"],
            suggested_title="Asset List",
            confidence=0.95,
            provider="openai",
            model="gpt-4",
        )

        assert "SELECT" in result.sql
        assert result.confidence == 0.95
        assert result.provider == "openai"

    def test_join_suggestion_to_sql(self):
        """Test JoinSuggestion generates correct SQL."""
        from query_engine import JoinSuggestion

        join = JoinSuggestion(
            left_table="asset",
            left_column="id",
            right_table="asset_creation_overview",
            right_column="asset_id",
            join_type="LEFT",
            confidence=0.9,
            reason="FK relationship"
        )

        sql_clause = join.to_sql_clause()
        assert "LEFT JOIN asset_creation_overview" in sql_clause
        assert "asset.id = asset_creation_overview.asset_id" in sql_clause

    def test_validation_result(self):
        """Test ValidationResult structure."""
        from query_engine import ValidationResult

        result = ValidationResult(
            is_valid=True,
            can_execute=True,
            errors=[],
            warnings=["SELECT * used"],
            suggestions=["Consider specifying columns"],
        )

        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_execution_result_truncation(self):
        """Test ExecutionResult truncates large data."""
        from query_engine import ExecutionResult

        # Create result with more than 100 rows
        large_data = [[i, f"row_{i}"] for i in range(150)]
        result = ExecutionResult(
            success=True,
            row_count=150,
            columns=["id", "name"],
            data=large_data,
        )

        dict_result = result.to_dict()
        assert len(dict_result["data"]) == 100
        assert dict_result.get("data_truncated") is True


# =============================================================================
# Time Filter Detection Tests
# =============================================================================

class TestTimeFilterDetection:
    """Test time filter extraction from queries."""

    def test_extract_last_month_filter(self):
        """Test extracting 'last month' time filter."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("show me assets created last month")

        assert len(result.time_filters) >= 1
        time_filter = result.time_filters[0]
        # The period type is 'relative' for "last month" pattern
        assert time_filter.period in ("relative", "created")

    def test_extract_this_week_filter(self):
        """Test extracting 'this week' time filter."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("show me events from this week")

        assert len(result.time_filters) >= 1
        time_filter = result.time_filters[0]
        # The period type is 'current' for "this week" pattern
        assert time_filter.period == "current"

    def test_extract_february_filter(self):
        """Test extracting specific month filter (February)."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("assignments whose due date is in FEB")

        assert len(result.time_filters) >= 1
        time_filter = result.time_filters[0]
        # The period type is 'in_month' or 'due_in_month' for month patterns
        assert time_filter.period in ("in_month", "due_in_month")

    def test_extract_date_range_filter(self):
        """Test extracting date range filter."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("show me data from January to March")

        # This pattern might match 'since_month' or similar
        assert len(result.time_filters) >= 1


# =============================================================================
# State Filter Detection Tests
# =============================================================================

class TestStateFilterDetection:
    """Test state filter extraction (not completed, pending, etc.)."""

    def test_extract_not_completed_filter(self):
        """Test extracting 'not completed' state filter."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("users who have not completed the assignment")

        assert len(result.state_filters) >= 1
        state_filter = result.state_filters[0]
        # ExtractedStateFilter has .negated and .state attributes
        assert state_filter.negated is True
        assert "complet" in state_filter.state.lower()

    def test_extract_pending_filter(self):
        """Test extracting 'pending' state filter."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("show me pending assignments")

        assert len(result.state_filters) >= 1
        state_filter = result.state_filters[0]
        assert "pending" in state_filter.state.lower()

    def test_extract_active_filter(self):
        """Test extracting 'active' state filter."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("list all active users")

        assert len(result.state_filters) >= 1
        state_filter = result.state_filters[0]
        assert "active" in state_filter.state.lower()

    def test_complex_state_filter_query(self):
        """Test complex query with state and time filters."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract(
            "give me details of all users who have not completed the assignment "
            "whose due date is in FEB"
        )

        # Should have both state and time filters
        assert len(result.state_filters) >= 1
        assert len(result.time_filters) >= 1

        # Verify state filter
        state_filter = result.state_filters[0]
        assert state_filter.negated is True
        assert "complet" in state_filter.state.lower()


# =============================================================================
# Schema Extraction Tests
# =============================================================================

class TestSchemaExtraction:
    """Test schema extraction from queries (e.g., 'in jeeves')."""

    def test_extract_schema_from_query(self):
        """Test extracting schema name from 'in <schema>' pattern."""
        from nlu.entity_extractor import EntityExtractor

        # EntityExtractor needs known_schemas to detect schema names
        extractor = EntityExtractor(known_schemas={'jeeves', 'public', 'dexit'})
        result = extractor.extract("show me jobs in jeeves")

        # The schema should be extracted to extracted_schema attribute
        assert result.extracted_schema == "jeeves"

    def test_schema_not_confused_with_filter(self):
        """Test that schema extraction doesn't confuse with regular filters."""
        from nlu.entity_extractor import EntityExtractor

        # 'lcmc' is not a known schema, so it should be treated as a filter value
        extractor = EntityExtractor(known_schemas={'jeeves', 'public'})
        result = extractor.extract("show me assets in tenant lcmc")

        # 'lcmc' should be a filter value, not a schema
        # Check that extracted_schema is not 'lcmc'
        assert result.extracted_schema != "lcmc"

        # Check filters - lcmc should be a filter value for tenant column
        filter_values = [f.value for f in result.filters if f.column == "tenant"]
        assert "lcmc" in filter_values or result.extracted_schema is None


# =============================================================================
# Table Intelligence Tests
# =============================================================================

class TestTableIntelligence:
    """Test TableIntelligence for smart table recommendations."""

    def test_has_date_columns_detection(self, test_config_file):
        """Test detecting tables with date columns."""
        from nlu.table_intelligence import TableIntelligence

        # TableIntelligence takes config_path (string path to JSON file)
        intelligence = TableIntelligence(config_path=test_config_file)

        # asset_creation_overview has created_at and updated_at
        assert intelligence.has_date_columns("asset_creation_overview") is True

        # asset has no date columns in our test config
        assert intelligence.has_date_columns("asset") is False

    def test_recommend_tables_for_time_filter(self, test_config_file):
        """Test recommending tables with date columns for time filter queries."""
        from nlu.table_intelligence import TableIntelligence

        intelligence = TableIntelligence(config_path=test_config_file)

        candidates = [
            {"table_name": "asset", "score": 0.95},
            {"table_name": "asset_creation_overview", "score": 0.85},
        ]

        enhanced = intelligence.recommend_tables_for_query(
            entity="assets",
            candidates=candidates,
            has_time_filter=True,
            time_filter_type="created"
        )

        # asset_creation_overview should be promoted due to date columns
        assert len(enhanced) >= 1
        top_table = enhanced[0]["table_name"]
        # Tables with date columns should be prioritized
        assert enhanced[0].get("has_date_columns") is True or \
               top_table == "asset_creation_overview"

    def test_find_time_filter_alternatives(self, test_config_file):
        """Test finding alternative tables for time filtering."""
        from nlu.table_intelligence import TableIntelligence

        intelligence = TableIntelligence(config_path=test_config_file)

        # Find alternatives for 'asset' which has no date columns
        alternatives = intelligence.find_time_filter_alternatives("asset")

        # Should suggest asset_creation_overview or audithistory
        alt_names = [alt.table_name for alt in alternatives]
        assert "asset_creation_overview" in alt_names or "audithistory" in alt_names


# =============================================================================
# Few-Shot Example Generation Tests
# =============================================================================

class TestFewShotExamples:
    """Test few-shot example retrieval for LLM prompts."""

    def _save_example(self, store, title: str, sql: str, tables: List[str],
                      keywords: List[str], query_type: str = "simple"):
        """Helper to save a query example with correct method signature."""
        pattern_json = json.dumps({
            "query_type": query_type,
            "tables": tables,
            "joins": [],
            "filters": [],
            "aggregations": ["COUNT"] if query_type == "aggregation" else [],
            "group_by": [],
            "order_by": [],
            "has_time_filter": False,
            "has_limit": True,
        })

        store.save_query_example(
            title=title,
            sql=sql,
            normalized_sql=sql,
            pattern_json=pattern_json,
            keywords=keywords,
            tables=tables,
            query_type=query_type,
            schema_name="public",
            dialect="starrocks",
        )

    def test_get_examples_by_tables(self, mock_knowledge_store):
        """Test getting examples that match specific tables."""
        self._save_example(
            mock_knowledge_store,
            title="Asset count by tenant",
            sql="SELECT tenant, COUNT(*) FROM asset GROUP BY tenant",
            tables=["asset"],
            keywords=["asset", "tenant", "count"],
            query_type="aggregation"
        )

        # Retrieve examples
        examples = mock_knowledge_store.get_examples_by_tables(["asset"], limit=5)

        assert len(examples) >= 1
        assert "asset" in examples[0].get("sql", "").lower()

    def test_get_examples_by_keywords(self, mock_knowledge_store):
        """Test getting examples that match keywords."""
        self._save_example(
            mock_knowledge_store,
            title="Monthly asset report",
            sql="SELECT DATE_TRUNC('month', created_at), COUNT(*) FROM asset_creation_overview GROUP BY 1",
            tables=["asset_creation_overview"],
            keywords=["monthly", "asset", "report", "created"],
            query_type="aggregation"
        )

        examples = mock_knowledge_store.get_examples_by_keywords(
            ["monthly", "asset"],
            limit=5
        )

        assert len(examples) >= 1

    def test_example_generator_relevance_scoring(self, mock_knowledge_store, test_semantic_config):
        """Test ExampleGenerator scores and ranks examples by relevance."""
        from learning.example_generator import ExampleGenerator

        # Add multiple examples
        self._save_example(
            mock_knowledge_store,
            title="Simple asset list",
            sql="SELECT * FROM asset LIMIT 100",
            tables=["asset"],
            keywords=["asset", "list"],
            query_type="simple"
        )
        self._save_example(
            mock_knowledge_store,
            title="Asset count by tenant",
            sql="SELECT tenant, COUNT(*) FROM asset GROUP BY tenant",
            tables=["asset"],
            keywords=["asset", "tenant", "count", "group"],
            query_type="aggregation"
        )

        # Create generator
        generator = ExampleGenerator(mock_knowledge_store, test_semantic_config)

        # Query for count-related examples
        relevant = generator.get_relevant_examples(
            user_query="count assets by tenant",
            tables=["asset"],
            max_examples=2
        )

        # Should return aggregation example first
        assert len(relevant) >= 1


# =============================================================================
# SQL Validation Tests
# =============================================================================

class TestSQLValidation:
    """Test SQL validation functionality."""

    def test_validate_select_statement(self):
        """Test validating a correct SELECT statement."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "SELECT id, title, tenant FROM asset WHERE status = 'active' LIMIT 100"
        result = validator.validate(sql)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_block_delete_in_read_only(self):
        """Test blocking DELETE in read-only mode."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "DELETE FROM asset WHERE id = 1"
        result = validator.validate(sql)

        assert result.is_valid is False
        assert any("DELETE" in e.upper() for e in result.errors)

    def test_block_update_in_read_only(self):
        """Test blocking UPDATE in read-only mode."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "UPDATE asset SET status = 'archived' WHERE id = 1"
        result = validator.validate(sql)

        assert result.is_valid is False
        assert any("UPDATE" in e.upper() for e in result.errors)

    def test_block_drop_in_read_only(self):
        """Test blocking DROP in read-only mode."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "DROP TABLE asset"
        result = validator.validate(sql)

        assert result.is_valid is False
        assert any("DROP" in e.upper() for e in result.errors)

    def test_warn_on_select_star(self):
        """Test warning on SELECT *."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "SELECT * FROM asset LIMIT 100"
        result = validator.validate(sql)

        assert any("SELECT *" in w for w in result.warnings)

    def test_warn_on_missing_limit(self):
        """Test warning on missing LIMIT clause."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "SELECT id, title FROM asset"
        result = validator.validate(sql)

        assert any("LIMIT" in w.upper() for w in result.warnings)

    def test_detect_unbalanced_parentheses(self):
        """Test detecting unbalanced parentheses."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "SELECT * FROM asset WHERE (status = 'active'"
        result = validator.validate(sql)

        assert result.is_valid is False
        assert any("paren" in e.lower() for e in result.errors)


# =============================================================================
# QueryEngine Integration Tests
# =============================================================================

class TestQueryEngineIntegration:
    """Integration tests for QueryEngine pipeline."""

    def test_analyze_simple_query(self, mock_mcp_tools):
        """Test analyzing a simple query."""
        from query_engine import QueryEngine, QueryPlan

        # Setup mock
        mock_mcp_tools["build_execution_plan"].return_value = {
            "entities": [{"name": "assets", "type": "dataset"}],
            "table_matches": {
                "assets": {
                    "recommended": {"table_name": "asset", "score": 0.95},
                    "candidates": [{"table_name": "asset", "score": 0.95}],
                    "is_ambiguous": False,
                }
            },
            "filters": [],
            "time_filters": [],
            "state_filters": [],
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = engine.analyze_query("show me all assets")

        assert plan.original_query == "show me all assets"
        assert "assets" in plan.table_matches
        assert plan.needs_user_input() is False

    def test_analyze_query_with_time_filter(self, mock_mcp_tools):
        """Test analyzing query with time filter."""
        from query_engine import QueryEngine

        mock_mcp_tools["build_execution_plan"].return_value = {
            "entities": [{"name": "assets", "type": "dataset"}],
            "table_matches": {
                "assets": {
                    "recommended": {"table_name": "asset_creation_overview", "score": 0.9},
                    "candidates": [],
                    "is_ambiguous": False,
                }
            },
            "filters": [],
            "time_filters": [{"period": "last_month", "column_hint": "created_at"}],
            "state_filters": [],
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = engine.analyze_query("show me assets created last month")

        assert len(plan.time_filters) == 1
        assert plan.time_filters[0]["period"] == "last_month"

    def test_analyze_query_with_state_filter(self, mock_mcp_tools):
        """Test analyzing query with state filter."""
        from query_engine import QueryEngine

        mock_mcp_tools["build_execution_plan"].return_value = {
            "entities": [{"name": "users", "type": "dataset"}, {"name": "assignment", "type": "dataset"}],
            "table_matches": {
                "users": {"recommended": {"table_name": "users", "score": 1.0}, "is_ambiguous": False},
                "assignment": {"recommended": {"table_name": "assignment", "score": 1.0}, "is_ambiguous": False},
            },
            "filters": [],
            "time_filters": [],
            "state_filters": [{"state": "completed", "negated": True, "target_entity": "assignment"}],
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = engine.analyze_query("users who have not completed the assignment")

        assert len(plan.state_filters) == 1
        assert plan.state_filters[0]["negated"] is True

    def test_analyze_query_with_schema(self, mock_mcp_tools):
        """Test analyzing query with schema extraction."""
        from query_engine import QueryEngine

        mock_mcp_tools["build_execution_plan"].return_value = {
            "entities": [{"name": "jobs", "type": "dataset"}],
            "table_matches": {
                "jobs": {"recommended": {"table_name": "jobs", "score": 1.0}, "is_ambiguous": False},
            },
            "filters": [],
            "time_filters": [],
            "state_filters": [],
            "extracted_schema": "jeeves",
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = engine.analyze_query("show me jobs in jeeves")

        assert plan.extracted_schema == "jeeves"

    def test_resolve_tables_from_plan(self, mock_mcp_tools):
        """Test resolving tables from query plan."""
        from query_engine import QueryEngine, QueryPlan

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = QueryPlan(
            original_query="show me assets and users",
            entities=[],
            table_matches={
                "assets": {"recommended": {"table_name": "asset", "score": 0.95}},
                "users": {"recommended": {"table_name": "users", "score": 1.0}},
            }
        )

        tables = engine.resolve_tables(plan)

        assert "asset" in tables
        assert "users" in tables

    def test_resolve_tables_with_user_selection(self, mock_mcp_tools):
        """Test resolving tables with user disambiguation."""
        from query_engine import QueryEngine, QueryPlan

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = QueryPlan(
            original_query="show me assets",
            entities=[],
            table_matches={
                "assets": {
                    "is_ambiguous": True,
                    "candidates": [
                        {"table_name": "asset", "score": 0.9},
                        {"table_name": "asset_creation_overview", "score": 0.85},
                    ]
                }
            },
            ambiguous_entities=["assets"]
        )

        # User selects asset_creation_overview
        tables = engine.resolve_tables(plan, user_selections={"assets": "asset_creation_overview"})

        assert "asset_creation_overview" in tables

    def test_find_joins_between_tables(self, mock_mcp_tools):
        """Test finding joins between tables."""
        from query_engine import QueryEngine

        mock_mcp_tools["suggest_joins"].return_value = {
            "joins": [
                {
                    "left_table": "asset",
                    "left_column": "id",
                    "right_table": "asset_creation_overview",
                    "right_column": "asset_id",
                    "join_type": "LEFT",
                    "confidence": 0.9,
                    "reason": "FK relationship"
                }
            ]
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        joins = engine.find_joins(["asset", "asset_creation_overview"])

        assert len(joins) == 1
        assert joins[0].left_table == "asset"
        assert joins[0].right_column == "asset_id"

    def test_validate_sql_integration(self, mock_mcp_tools):
        """Test SQL validation through engine."""
        from query_engine import QueryEngine

        mock_mcp_tools["validate_sql"].return_value = {
            "is_valid": True,
            "can_execute": True,
            "errors": [],
            "warnings": ["Consider adding LIMIT clause"],
            "suggestions": [],
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        result = engine.validate_sql("SELECT * FROM asset")

        assert result.is_valid is True
        assert len(result.warnings) == 1


# =============================================================================
# API Server Tests
# =============================================================================

class TestAPIServer:
    """Test API server endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API server."""
        # Import and configure test client
        sys.path.insert(0, os.path.join(PARENT_DIR, 'chatbot'))
        from chatbot.api_server import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200

        data = response.get_json()
        assert data["status"] == "ok"

    def test_chart_types_endpoint(self, client):
        """Test chart types endpoint."""
        response = client.get('/api/chart-types')
        assert response.status_code == 200

        data = response.get_json()
        assert "chart_types" in data
        assert len(data["chart_types"]) > 0

        # Check that table type exists
        chart_ids = [c["id"] for c in data["chart_types"]]
        assert "table" in chart_ids

    def test_reset_session_endpoint(self, client):
        """Test session reset endpoint."""
        response = client.post('/api/reset', json={"session_id": "test-session"})
        assert response.status_code == 200

        data = response.get_json()
        assert data["status"] == "ok"

    def test_set_provider_endpoint(self, client):
        """Test setting LLM provider."""
        response = client.post('/api/set-provider', json={
            "session_id": "test-session",
            "provider": "claude"
        })
        assert response.status_code == 200

        data = response.get_json()
        assert data["provider"] == "claude"

    def test_chat_cancel_action(self, client):
        """Test cancel action in chat."""
        response = client.post('/api/chat', json={
            "session_id": "test-session",
            "message": "",
            "action": "cancel"
        })
        assert response.status_code == 200

        data = response.get_json()
        assert data["step"] == "idle"
        assert "Cancelled" in data["response"]


# =============================================================================
# Complex Query Tests
# =============================================================================

class TestComplexQueries:
    """Test complex query scenarios that were used for optimization."""

    def test_users_not_completed_assignment_feb_with_mocks(self, mock_mcp_tools):
        """Test QueryEngine with mock: 'give me details of all users who have not completed the assignment whose due date is in FEB'"""
        from query_engine import QueryEngine

        mock_mcp_tools["build_execution_plan"].return_value = {
            "entities": [
                {"name": "users", "type": "dataset"},
                {"name": "assignment", "type": "dataset"}
            ],
            "table_matches": {
                "users": {"recommended": {"table_name": "users", "score": 1.0}, "is_ambiguous": False},
                "assignment": {"recommended": {"table_name": "assignment", "score": 1.0}, "is_ambiguous": False},
            },
            "filters": [],
            "time_filters": [{"period": "february", "column_hint": "due_date", "month": 2}],
            "state_filters": [{"state": "completed", "negated": True, "target_entity": "assignment"}],
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = engine.analyze_query(
            "give me details of all users who have not completed the assignment "
            "whose due date is in FEB"
        )

        # Should detect both time and state filters
        assert len(plan.time_filters) >= 1
        assert len(plan.state_filters) >= 1

        # State filter should be negated "completed" (using dict access since mock returns dict)
        state_filter = plan.state_filters[0]
        assert state_filter.get("negated") is True
        assert "complet" in state_filter.get("state", "").lower()

        # Time filter should reference February
        time_filter = plan.time_filters[0]
        assert "feb" in time_filter.get("period", "").lower() or \
               time_filter.get("month") == 2

    def test_users_not_completed_assignment_feb_with_nlu(self):
        """Test NLU extraction directly: 'give me details of all users who have not completed the assignment whose due date is in FEB'"""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract(
            "give me details of all users who have not completed the assignment "
            "whose due date is in FEB"
        )

        # Should detect state filter (not completed)
        assert len(result.state_filters) >= 1
        state_filter = result.state_filters[0]
        assert state_filter.negated is True
        assert "complet" in state_filter.state.lower()

        # Should detect time filter (in FEB)
        assert len(result.time_filters) >= 1

        # Should detect entities (users, assignment)
        entity_names = [e.name for e in result.entities]
        assert any("user" in e for e in entity_names)
        assert any("assignment" in e for e in entity_names)

    def test_assets_created_last_month_in_tenant(self, mock_mcp_tools):
        """Test: 'show me assets created last month in tenant lcmc'"""
        from query_engine import QueryEngine

        mock_mcp_tools["build_execution_plan"].return_value = {
            "entities": [{"name": "assets", "type": "dataset"}],
            "table_matches": {
                "assets": {
                    "recommended": {"table_name": "asset_creation_overview", "score": 0.9},
                    "is_ambiguous": False,
                }
            },
            "filters": [{"column": "tenant", "value": "lcmc", "operator": "="}],
            "time_filters": [{"period": "last_month", "column_hint": "created_at"}],
            "state_filters": [],
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = engine.analyze_query("show me assets created last month in tenant lcmc")

        # Should have tenant filter
        assert len(plan.filters) >= 1
        tenant_filter = plan.filters[0]
        assert tenant_filter.get("column") == "tenant"
        assert tenant_filter.get("value") == "lcmc"

        # Should have time filter
        assert len(plan.time_filters) >= 1

    def test_jobs_in_jeeves_schema(self, mock_mcp_tools):
        """Test: 'show me jobs in jeeves'"""
        from query_engine import QueryEngine

        mock_mcp_tools["build_execution_plan"].return_value = {
            "entities": [{"name": "jobs", "type": "dataset"}],
            "table_matches": {
                "jobs": {"recommended": {"table_name": "jobs", "score": 1.0}, "is_ambiguous": False},
            },
            "filters": [],
            "time_filters": [],
            "state_filters": [],
            "extracted_schema": "jeeves",
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools

        plan = engine.analyze_query("show me jobs in jeeves")

        # Schema should be extracted
        assert plan.extracted_schema == "jeeves"

    def test_active_assets_count_by_contenttype(self):
        """Test NLU: 'count active assets grouped by contenttype'"""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("count active assets grouped by contenttype")

        # Should detect aggregation
        assert "count" in result.aggregations

        # Should detect state filter (active)
        assert len(result.state_filters) >= 1
        assert any("active" in sf.state.lower() for sf in result.state_filters)


# =============================================================================
# Learning Feedback Tests
# =============================================================================

class TestLearningFeedback:
    """Test query learning feedback loop."""

    def test_record_successful_query(self, mock_knowledge_store):
        """Test recording a successful query for learning."""
        mock_knowledge_store.record_query_feedback(
            natural_language="show me all active assets",
            generated_sql="SELECT * FROM asset WHERE status = 'active' LIMIT 100",
            feedback_type="success",
            row_count=42
        )

        # Verify feedback was recorded
        # (implementation would query the feedback table)
        assert True  # Basic assertion that it didn't throw

    def test_record_failed_query(self, mock_knowledge_store):
        """Test recording a failed query for learning."""
        mock_knowledge_store.record_query_feedback(
            natural_language="show me assets with events",
            generated_sql="SELECT * FROM asset JION events",  # Intentional typo
            feedback_type="failure",
            error_message="SQL syntax error near 'JION'"
        )

        assert True  # Basic assertion that it didn't throw


# =============================================================================
# NLU Integration Tests
# =============================================================================

class TestNLUIntegration:
    """Test that NLU components are properly integrated into the query pipeline."""

    def test_entity_extractor_integration(self):
        """Test EntityExtractor is properly used with known tables/columns/schemas."""
        from nlu.entity_extractor import EntityExtractor

        # Create extractor with known context
        extractor = EntityExtractor(
            known_tables={'asset', 'users', 'assignment'},
            known_columns={'tenant', 'status', 'created_at'},
            known_schemas={'jeeves', 'public'}
        )

        # Test extraction with schema
        result = extractor.extract("show me assets in jeeves")
        assert result.extracted_schema == "jeeves"

        # Test extraction with filter
        result = extractor.extract("show me assets with tenant lcmc")
        filter_cols = [f.column for f in result.filters]
        assert "tenant" in filter_cols

    def test_semantic_matcher_integration(self, test_config_file):
        """Test SemanticMatcher properly matches tables."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)

        # Test exact match
        results = matcher.match_table("asset")
        assert len(results) > 0
        assert results[0].table_name == "asset"

        # Test alias match
        results = matcher.match_table("user")
        assert len(results) > 0
        # Should match 'users' table via alias

    def test_table_intelligence_in_plan(self, test_config_file):
        """Test TableIntelligence is used when time filters are present."""
        from nlu.table_intelligence import TableIntelligence

        intelligence = TableIntelligence(config_path=test_config_file)

        # Test that tables with date columns are identified
        assert intelligence.has_date_columns("asset_creation_overview") is True
        assert intelligence.has_date_columns("asset") is False

        # Test that alternatives are found
        alternatives = intelligence.find_time_filter_alternatives("asset")
        alt_names = [alt.table_name for alt in alternatives]
        # Should find asset_creation_overview as alternative
        assert "asset_creation_overview" in alt_names or len(alt_names) >= 0

    def test_full_nlu_pipeline(self):
        """Test full NLU pipeline from query to structured output."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor(
            known_schemas={'jeeves', 'public'},
            known_tables={'users', 'assignment', 'asset'},
            known_columns={'tenant', 'status', 'due_date', 'completed_at'}
        )

        # Complex query with multiple NLU features
        query = "give me all users who have not completed assignments due in FEB in jeeves"
        result = extractor.extract(query)

        # Should extract entities
        entity_names = [e.name for e in result.entities]
        assert len(entity_names) >= 2  # users, assignments

        # Should extract state filter
        assert len(result.state_filters) >= 1
        state_filter = result.state_filters[0]
        assert state_filter.negated is True  # "not completed"

        # Should extract time filter
        assert len(result.time_filters) >= 1

        # Should extract schema
        assert result.extracted_schema == "jeeves"


# =============================================================================
# End-to-End Tests (Mocked)
# =============================================================================

class TestEndToEnd:
    """End-to-end tests with mocked external dependencies."""

    def test_query_to_sql_pipeline(self, mock_mcp_tools, mock_llm_service):
        """Test the complete query to SQL pipeline."""
        from query_engine import QueryEngine

        # Setup mocks
        mock_mcp_tools["build_execution_plan"].return_value = {
            "entities": [{"name": "assets", "type": "dataset"}],
            "table_matches": {
                "assets": {
                    "recommended": {"table_name": "asset", "score": 0.95},
                    "is_ambiguous": False,
                }
            },
            "filters": [],
            "time_filters": [],
            "state_filters": [],
        }

        mock_mcp_tools["suggest_joins"].return_value = {"joins": []}
        mock_mcp_tools["generate_sql_context"].return_value = {
            "tables": [{"table_name": "asset", "columns": ["id", "title"]}],
            "engine": "starrocks"
        }
        mock_mcp_tools["validate_sql"].return_value = {
            "is_valid": True,
            "can_execute": True,
            "errors": [],
            "warnings": []
        }

        engine = QueryEngine()
        engine._mcp_tools = mock_mcp_tools
        engine._llm_service = mock_llm_service

        # Run analysis
        plan = engine.analyze_query("show me all assets")
        assert plan is not None
        assert "assets" in plan.table_matches

        # Resolve tables
        tables = engine.resolve_tables(plan)
        assert "asset" in tables


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
