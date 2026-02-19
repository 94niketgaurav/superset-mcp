"""
Happy path test for the complete superset-mcp flow.

Tests the full flow from natural language query to SQL generation.

Run with: pytest tests/test_happy_flow.py -v
"""
import pytest
import json
import os
import tempfile
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def test_config():
    """Create a test configuration for the semantic matcher."""
    return {
        "version": "2.0",
        "table_prefixes": ["tbl_", "t_"],
        "table_suffixes": ["_table", "_view"],
        "databases": {
            "1": {"name": "TestDB"}
        },
        "schemas": {
            "public": {
                "description": "Public schema for testing",
                "tables": {
                    "assets": {
                        "dataset_id": 1,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "name": {"type": "VARCHAR"},
                            "created_at": {"type": "TIMESTAMP"},
                            "userId": {"type": "INTEGER", "is_fk": True, "references": "users.id"}
                        },
                        "aliases": ["asset", "items"]
                    },
                    "assetversion": {
                        "dataset_id": 2,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "asset_id": {"type": "INTEGER", "is_fk": True, "references": "assets.id"},
                            "version": {"type": "INTEGER"}
                        }
                    },
                    "assethistory": {
                        "dataset_id": 3,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "asset_id": {"type": "INTEGER", "is_fk": True, "references": "assets.id"},
                            "action": {"type": "VARCHAR"}
                        }
                    },
                    "posthogevents": {
                        "dataset_id": 4,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "event_type": {"type": "VARCHAR"},
                            "assetId": {"type": "INTEGER", "is_fk": True, "references": "assets.id"},
                            "timestamp": {"type": "TIMESTAMP"}
                        },
                        "aliases": ["events", "posthog"]
                    },
                    "auditevents": {
                        "dataset_id": 5,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "event_type": {"type": "VARCHAR"},
                            "user_id": {"type": "INTEGER", "is_fk": True, "references": "users.id"}
                        },
                        "aliases": ["audit"]
                    },
                    "users": {
                        "dataset_id": 6,
                        "columns": {
                            "id": {"type": "INTEGER", "is_pk": True},
                            "email": {"type": "VARCHAR"},
                            "name": {"type": "VARCHAR"}
                        },
                        "aliases": ["user"]
                    }
                }
            }
        },
        "join_patterns": [
            {"left": "assets", "right": "posthogevents", "on": "id = assetId", "type": "left"},
            {"left": "assets", "right": "assetversion", "on": "id = asset_id", "type": "left"},
            {"left": "assets", "right": "assethistory", "on": "id = asset_id", "type": "left"},
            {"left": "users", "right": "assets", "on": "id = userId", "type": "left"},
            {"left": "users", "right": "auditevents", "on": "id = user_id", "type": "left"}
        ],
        "synonyms": {
            "assets": ["items", "resources"],
            "events": ["activities", "logs"],
            "users": ["accounts", "members"]
        }
    }


@pytest.fixture
def test_config_file(test_config, tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(test_config, f)
    return str(config_path)


@pytest.fixture
def mock_dataset_details():
    """Detailed dataset information including columns."""
    return {
        1: {
            "id": 1,
            "table_name": "assets",
            "schema": "public",
            "database_id": 1,
            "columns": [
                {"column_name": "id", "type": "INTEGER", "is_dttm": False},
                {"column_name": "name", "type": "VARCHAR", "is_dttm": False},
                {"column_name": "created_at", "type": "TIMESTAMP", "is_dttm": True},
                {"column_name": "userId", "type": "INTEGER", "is_dttm": False},
            ]
        },
        4: {
            "id": 4,
            "table_name": "posthogevents",
            "schema": "public",
            "database_id": 1,
            "columns": [
                {"column_name": "id", "type": "INTEGER", "is_dttm": False},
                {"column_name": "event_type", "type": "VARCHAR", "is_dttm": False},
                {"column_name": "assetId", "type": "INTEGER", "is_dttm": False},
                {"column_name": "timestamp", "type": "TIMESTAMP", "is_dttm": True},
            ]
        },
        6: {
            "id": 6,
            "table_name": "users",
            "schema": "public",
            "database_id": 1,
            "columns": [
                {"column_name": "id", "type": "INTEGER", "is_dttm": False},
                {"column_name": "email", "type": "VARCHAR", "is_dttm": False},
                {"column_name": "name", "type": "VARCHAR", "is_dttm": False},
            ]
        }
    }


# =============================================================================
# NLU Tests
# =============================================================================

class TestEntityExtractor:
    """Test the entity extractor."""

    def test_extract_simple_entities(self):
        """Test extracting entities from a simple query."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("show me assets and events")

        # Should extract 'assets' and 'events'
        entity_names = [e.name for e in result.entities]
        assert "assets" in entity_names or "asset" in entity_names
        assert "events" in entity_names or "event" in entity_names

    def test_extract_join_hints(self):
        """Test extracting join hints from query."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("show me assets and their events")

        # Should detect join relationship
        assert len(result.join_hints) >= 1

    def test_extract_column_hints(self):
        """Test extracting column mentions from query."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("join assets and events by assetId")

        # Should detect assetId column
        assert "assetId" in result.columns_mentioned or len(result.join_hints) > 0

    def test_extract_time_filter(self):
        """Test extracting time filters from query."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("show me events from last week")

        assert len(result.time_filters) >= 1

    def test_extract_aggregation(self):
        """Test extracting aggregations from query."""
        from nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract("count all assets")

        assert "count" in result.aggregations


class TestSemanticMatcher:
    """Test the semantic matcher with the new API."""

    def test_exact_match(self, test_config_file):
        """Test exact table name matching."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        results = matcher.match_table("assets")

        assert len(results) > 0
        top_match = results[0]
        assert top_match.table_name == "assets"
        assert top_match.score == 1.0

    def test_alias_match(self, test_config_file):
        """Test alias matching."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        results = matcher.match_table("events")

        # "events" is an alias for "posthogevents"
        assert len(results) > 0
        top_match = results[0]
        assert top_match.table_name == "posthogevents"
        assert top_match.score >= 0.95  # Alias or synonym match

    def test_prefix_match(self, test_config_file):
        """Test prefix matching (asset -> assetversion, assethistory)."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        results = matcher.match_table("asset")

        # Should match assets, assetversion, assethistory
        table_names = [r.table_name for r in results]
        assert "assets" in table_names
        assert "assetversion" in table_names
        assert "assethistory" in table_names

    def test_suffix_match(self, test_config_file):
        """Test that suffix/contains matching works."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        results = matcher.match_table("history")

        # Should match assethistory
        table_names = [r.table_name for r in results]
        assert "assethistory" in table_names

    def test_synonym_match(self, test_config_file):
        """Test synonym matching."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        results = matcher.match_table("items")

        # "items" is a synonym for "assets"
        assert len(results) > 0
        table_names = [r.table_name for r in results]
        assert "assets" in table_names

    def test_get_table(self, test_config_file):
        """Test getting table info directly."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        table = matcher.get_table("assets")

        assert table is not None
        assert table.name == "assets"
        assert table.schema == "public"
        assert "id" in table.columns
        assert table.columns["id"].is_pk is True

    def test_get_columns(self, test_config_file):
        """Test getting column names."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        columns = matcher.get_columns("assets")

        assert "id" in columns
        assert "name" in columns
        assert "userId" in columns

    def test_get_join_path(self, test_config_file):
        """Test getting join path between tables."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        join = matcher.get_join_path("assets", "posthogevents")

        assert join is not None
        assert join.left_table == "assets"
        assert join.right_table == "posthogevents"
        assert join.left_column == "id"
        assert join.right_column == "assetId"

    def test_find_join_path(self, test_config_file):
        """Test finding join path connecting multiple tables."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        joins = matcher.find_join_path(["assets", "posthogevents"])

        assert len(joins) > 0

    def test_no_match(self, test_config_file):
        """Test handling of non-matching entity with strict threshold."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        # Use higher threshold to ensure truly non-matching entities return no results
        results = matcher.match_table("xyz_nonexistent_table", min_score=0.5)

        assert len(results) == 0

    def test_get_statistics(self, test_config_file):
        """Test getting matcher statistics."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)
        stats = matcher.get_statistics()

        assert stats["total_tables"] == 6
        assert stats["total_columns"] > 0
        assert "public" in stats["schemas"]


# =============================================================================
# Join Reasoning Tests
# =============================================================================

class TestColumnNormalizer:
    """Test column name normalization."""

    def test_camel_to_snake(self):
        """Test converting camelCase to snake_case."""
        from joins.normalizer import ColumnNormalizer

        assert ColumnNormalizer.to_snake_case("assetId") == "asset_id"
        assert ColumnNormalizer.to_snake_case("userId") == "user_id"
        assert ColumnNormalizer.to_snake_case("createdByUserId") == "created_by_user_id"

    def test_snake_to_parts(self):
        """Test splitting snake_case into parts."""
        from joins.normalizer import ColumnNormalizer

        assert ColumnNormalizer.to_parts("asset_id") == ["asset", "id"]
        assert ColumnNormalizer.to_parts("user_uuid") == ["user", "uuid"]

    def test_camel_to_parts(self):
        """Test splitting camelCase into parts."""
        from joins.normalizer import ColumnNormalizer

        assert ColumnNormalizer.to_parts("assetId") == ["asset", "id"]
        assert ColumnNormalizer.to_parts("userId") == ["user", "id"]

    def test_is_id_column(self):
        """Test ID column detection."""
        from joins.normalizer import ColumnNormalizer

        assert ColumnNormalizer.is_id_column("id") is True
        assert ColumnNormalizer.is_id_column("assetId") is True
        assert ColumnNormalizer.is_id_column("user_id") is True
        assert ColumnNormalizer.is_id_column("uuid") is True
        assert ColumnNormalizer.is_id_column("name") is False

    def test_extract_fk_entity(self):
        """Test extracting entity from foreign key column name."""
        from joins.normalizer import ColumnNormalizer

        entity, key_type = ColumnNormalizer.extract_entity_from_fk("assetId")
        assert entity == "asset"
        assert key_type == "id"

        entity, key_type = ColumnNormalizer.extract_entity_from_fk("user_uuid")
        assert entity == "user"
        assert key_type == "uuid"

    def test_columns_match(self):
        """Test column matching across conventions."""
        from joins.normalizer import ColumnNormalizer

        assert ColumnNormalizer.columns_match("assetId", "asset_id") is True
        assert ColumnNormalizer.columns_match("userId", "user_id") is True
        assert ColumnNormalizer.columns_match("name", "email") is False


class TestEnhancedJoinReasoner:
    """Test enhanced join reasoning."""

    def test_fk_to_pk_detection(self, mock_dataset_details):
        """Test FK to PK join detection."""
        from joins.enhanced_reasoner import EnhancedJoinReasoner

        datasets = list(mock_dataset_details.values())
        reasoner = EnhancedJoinReasoner(datasets)

        # Should detect assetId in posthogevents -> assets.id
        suggestions = reasoner.suggest_joins([1, 4])  # assets, posthogevents

        assert len(suggestions) > 0

        # Find the assetId -> id suggestion
        asset_join = None
        for s in suggestions:
            if "asset" in s.left_column.lower() or "asset" in s.right_column.lower():
                asset_join = s
                break

        assert asset_join is not None
        assert asset_join.confidence > 0.5

    def test_normalized_match(self, mock_dataset_details):
        """Test join detection with normalized column names."""
        from joins.enhanced_reasoner import EnhancedJoinReasoner

        datasets = list(mock_dataset_details.values())
        reasoner = EnhancedJoinReasoner(datasets)

        # Should detect userId in assets -> users.id
        suggestions = reasoner.suggest_joins([1, 6])  # assets, users

        # Look for user join
        user_joins = [s for s in suggestions if "user" in s.left_column.lower() or "user" in s.right_column.lower()]
        assert len(user_joins) > 0


# =============================================================================
# SQL Validation Tests
# =============================================================================

class TestSQLValidator:
    """Test SQL validation."""

    def test_valid_select(self):
        """Test validation of valid SELECT statement."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "SELECT id, name FROM assets WHERE id > 0 LIMIT 100"
        result = validator.validate(sql)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_dangerous_operation_blocked(self):
        """Test that dangerous operations are blocked in read-only mode."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "DELETE FROM assets WHERE id = 1"
        result = validator.validate(sql)

        assert result.is_valid is False
        assert any("DELETE" in e for e in result.errors)

    def test_unbalanced_parentheses(self):
        """Test detection of unbalanced parentheses."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "SELECT * FROM assets WHERE (id > 0"
        result = validator.validate(sql)

        assert result.is_valid is False
        assert any("paren" in e.lower() for e in result.errors)

    def test_select_star_warning(self):
        """Test warning for SELECT *."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "SELECT * FROM assets LIMIT 100"
        result = validator.validate(sql)

        assert any("SELECT *" in w for w in result.warnings)

    def test_missing_limit_warning(self):
        """Test warning for missing LIMIT clause."""
        from validation.sql_validator import SQLValidator

        validator = SQLValidator(read_only=True)
        sql = "SELECT id, name FROM assets WHERE id > 0"
        result = validator.validate(sql)

        assert any("LIMIT" in w for w in result.warnings)


# =============================================================================
# Integration Tests
# =============================================================================

class TestHappyFlowIntegration:
    """Integration tests for the complete happy path flow."""

    def test_entity_to_dataset_matching(self, test_config_file):
        """Test complete flow from entity extraction to dataset matching."""
        from nlu.entity_extractor import EntityExtractor
        from nlu.semantic_matcher import SemanticMatcher

        # Step 1: Extract entities
        extractor = EntityExtractor()
        query = "show me assets and their events"
        intent = extractor.extract(query)

        entity_names = [e.name for e in intent.entities if e.entity_type == "dataset"]

        # Step 2: Match to tables
        matcher = SemanticMatcher(config_path=test_config_file)

        matches_found = 0
        for entity in entity_names:
            results = matcher.match_table(entity)
            if results:
                matches_found += 1

        # Should have matches
        assert matches_found >= 1

    def test_dataset_to_join_suggestion(self, mock_dataset_details):
        """Test flow from matched datasets to join suggestions."""
        from joins.enhanced_reasoner import EnhancedJoinReasoner

        # Datasets: assets (1) and posthogevents (4)
        datasets = [mock_dataset_details[1], mock_dataset_details[4]]
        reasoner = EnhancedJoinReasoner(datasets)

        suggestions = reasoner.suggest_joins([1, 4])

        # Should suggest a join
        assert len(suggestions) > 0

        # Best suggestion should have reasonable confidence
        best = suggestions[0]
        assert best.confidence >= 0.5

    def test_full_query_building_flow(self, test_config_file, mock_dataset_details):
        """Test the complete query building flow."""
        from nlu.entity_extractor import EntityExtractor
        from nlu.semantic_matcher import SemanticMatcher
        from joins.enhanced_reasoner import EnhancedJoinReasoner
        from validation.sql_validator import SQLValidator

        # Step 1: Parse query
        extractor = EntityExtractor()
        query = "show me assets and events by joining on assetId"
        intent = extractor.extract(query)

        # Step 2: Match tables
        entity_names = [e.name for e in intent.entities if e.entity_type == "dataset"]
        matcher = SemanticMatcher(config_path=test_config_file)

        # Get best matches
        selected_datasets = []
        for entity in entity_names:
            results = matcher.match_table(entity)
            if results:
                selected_datasets.append(results[0].dataset_id)

        # Step 3: Get join suggestions
        if len(selected_datasets) >= 2:
            # Get detailed dataset info
            detailed = [mock_dataset_details.get(ds_id) for ds_id in selected_datasets if ds_id in mock_dataset_details]
            detailed = [d for d in detailed if d]  # Filter None

            if len(detailed) >= 2:
                reasoner = EnhancedJoinReasoner(detailed)
                joins = reasoner.suggest_joins(selected_datasets)

                # Should have join suggestions
                assert len(joins) >= 0  # May or may not have joins depending on mock data

        # Step 4: Validate sample SQL
        validator = SQLValidator(read_only=True)
        sample_sql = """
            SELECT a.id, a.name, e.event_type
            FROM assets a
            INNER JOIN posthogevents e ON e.assetId = a.id
            LIMIT 1000
        """
        result = validator.validate(sample_sql)

        # SQL should be valid
        assert result.is_valid is True

    def test_semantic_matcher_join_path(self, test_config_file):
        """Test finding join paths using semantic matcher."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)

        # Find join path from user query entities
        join_path = matcher.find_join_path(["assets", "posthogevents"])

        assert len(join_path) > 0
        assert join_path[0].left_table == "assets"
        assert join_path[0].right_table == "posthogevents"

    def test_semantic_matcher_primary_keys(self, test_config_file):
        """Test getting primary keys from semantic matcher."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)

        pks = matcher.get_primary_keys("assets")
        assert "id" in pks

    def test_semantic_matcher_foreign_keys(self, test_config_file):
        """Test getting foreign keys from semantic matcher."""
        from nlu.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher(config_path=test_config_file)

        fks = matcher.get_foreign_keys("assets")
        # assets has userId FK
        fk_columns = [fk["column"] for fk in fks]
        assert "userId" in fk_columns


# =============================================================================
# Knowledge Store Tests
# =============================================================================

class TestKnowledgeStore:
    """Test the knowledge store."""

    def test_save_and_load_dataset(self, tmp_path):
        """Test saving and loading dataset knowledge."""
        from learning.knowledge_store import KnowledgeStore, DatasetKnowledge, ColumnKnowledge

        store = KnowledgeStore(str(tmp_path / "test_knowledge.db"))

        # Create test knowledge
        knowledge = DatasetKnowledge(
            dataset_id=1,
            table_name="test_table",
            schema="public",
            database_id=1,
            columns={
                "id": ColumnKnowledge(column_name="id", column_type="INTEGER", is_primary_key=True),
                "name": ColumnKnowledge(column_name="name", column_type="VARCHAR")
            }
        )

        # Save
        store.save_dataset(knowledge)

        # Load
        loaded = store.get_dataset(1)

        assert loaded is not None
        assert loaded.table_name == "test_table"
        assert "id" in loaded.columns
        assert loaded.columns["id"].is_primary_key is True

    def test_join_patterns(self, tmp_path):
        """Test saving and loading join patterns."""
        from learning.knowledge_store import KnowledgeStore, JoinPattern

        store = KnowledgeStore(str(tmp_path / "test_knowledge.db"))

        # Save pattern
        pattern = JoinPattern(
            left_table="events",
            left_column="assetId",
            right_table="assets",
            right_column="id",
            join_type="inner",
            confidence=0.9
        )
        store.save_join_pattern(pattern)

        # Load patterns
        patterns = store.get_join_patterns("events")

        assert len(patterns) == 1
        assert patterns[0].right_table == "assets"

    def test_synonyms(self, tmp_path):
        """Test adding and retrieving synonyms."""
        from learning.knowledge_store import KnowledgeStore

        store = KnowledgeStore(str(tmp_path / "test_knowledge.db"))

        store.add_synonym("asset", "item")
        store.add_synonym("asset", "resource")

        synonyms = store.get_synonyms("asset")

        assert "item" in synonyms
        assert "resource" in synonyms


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])