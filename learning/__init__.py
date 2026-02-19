"""
Learning module for auto-training the semantic matcher.

This module provides:
- Automatic dataset and column discovery
- Sample value analysis for join detection
- Knowledge persistence and daily refresh
- Query harvesting and analysis for few-shot learning
- Example generation for LLM prompts
"""

from .knowledge_store import KnowledgeStore, DatasetKnowledge, ColumnKnowledge
from .dataset_learner import DatasetLearner
from .column_analyzer import ColumnAnalyzer
from .query_harvester import QueryHarvester, HarvestedQuery
from .query_analyzer import QueryAnalyzer, QueryPattern, QueryExample
from .example_generator import ExampleGenerator

__all__ = [
    "KnowledgeStore",
    "DatasetKnowledge",
    "ColumnKnowledge",
    "DatasetLearner",
    "ColumnAnalyzer",
    "QueryHarvester",
    "HarvestedQuery",
    "QueryAnalyzer",
    "QueryPattern",
    "QueryExample",
    "ExampleGenerator",
]