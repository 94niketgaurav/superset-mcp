"""
NLU (Natural Language Understanding) module for query analysis.

This module provides:
- Entity extraction from natural language queries
- Semantic table matching for SQL query building
"""

from .entity_extractor import EntityExtractor, QueryIntent, ExtractedEntity
from .semantic_matcher import (
    SemanticMatcher,
    MatchResult,
    TableInfo,
    ColumnInfo,
    JoinPath,
    create_matcher,
    create_trained_matcher,
)

__all__ = [
    "EntityExtractor",
    "QueryIntent",
    "ExtractedEntity",
    "SemanticMatcher",
    "MatchResult",
    "TableInfo",
    "ColumnInfo",
    "JoinPath",
    "create_matcher",
    "create_trained_matcher",
]