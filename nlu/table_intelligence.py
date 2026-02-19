"""
Table Intelligence Module - Generic understanding of table structures and relationships.

This module builds a knowledge graph of tables from the semantic config to:
1. Index tables by their columns (especially date/time columns)
2. Track table relationships (joins, foreign keys)
3. Recommend optimal tables based on query requirements
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TableMetadata:
    """Metadata about a table."""
    name: str
    schema: str
    database_id: int
    columns: List[str]
    date_columns: List[str]  # Columns that can be used for time filtering
    primary_keys: List[str]
    foreign_keys: Dict[str, str]  # column -> referenced_table.column
    aliases: List[str]
    description: str = ""


@dataclass
class TableRelationship:
    """A relationship between two tables."""
    left_table: str
    right_table: str
    left_column: str
    right_column: str
    relationship_type: str  # 'fk', 'join_pattern', 'name_similarity'
    confidence: float


@dataclass
class TableRecommendation:
    """A recommended table for a query."""
    table_name: str
    score: float
    reason: str
    has_date_columns: bool
    date_columns: List[str]
    related_to: Optional[str] = None  # If this is an alternative to another table
    join_path: Optional[List[dict]] = None  # How to join if needed


class TableIntelligence:
    """
    Builds and queries a knowledge graph of tables.

    This provides intelligent table recommendations based on:
    - Query requirements (time filters, columns needed)
    - Table relationships
    - Column availability
    """

    # Patterns to identify date/time columns
    DATE_COLUMN_PATTERNS = [
        r'.*date.*', r'.*time.*', r'.*created.*', r'.*updated.*',
        r'.*timestamp.*', r'.*_at$', r'.*_on$', r'.*modified.*',
        r'.*changed.*', r'.*expired.*', r'.*archived.*'
    ]

    def __init__(self, config_path: str = "config/semantic_config.json"):
        """Initialize with semantic config."""
        self.config_path = config_path
        self.tables: Dict[str, TableMetadata] = {}
        self.relationships: List[TableRelationship] = []
        self.date_column_index: Dict[str, List[str]] = {}  # table -> date columns
        self.table_by_alias: Dict[str, str] = {}  # alias -> table name
        self.related_tables: Dict[str, Set[str]] = {}  # table -> related tables

        self._load_config()
        self._build_indexes()

    def _load_config(self):
        """Load and parse the semantic config."""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load semantic config: {e}")
            self.config = {}

    def _is_date_column(self, column_name: str) -> bool:
        """Check if a column name suggests it's a date/time column."""
        col_lower = column_name.lower()
        return any(re.match(pattern, col_lower) for pattern in self.DATE_COLUMN_PATTERNS)

    def _build_indexes(self):
        """Build all indexes from the config."""
        self._index_tables()
        self._index_relationships()
        self._index_aliases()
        self._build_related_tables()

    def _index_tables(self):
        """Index all tables and their metadata."""
        schemas = self.config.get('schemas', {})

        for schema_name, schema_data in schemas.items():
            tables = schema_data.get('tables', {})

            for table_name, table_info in tables.items():
                columns_data = table_info.get('columns', {})

                # Extract column names
                if isinstance(columns_data, dict):
                    columns = list(columns_data.keys())
                    # Extract PKs and FKs
                    pks = [col for col, info in columns_data.items()
                           if isinstance(info, dict) and info.get('is_pk')]
                    fks = {col: info.get('references')
                           for col, info in columns_data.items()
                           if isinstance(info, dict) and info.get('is_fk') and info.get('references')}
                else:
                    columns = []
                    pks = []
                    fks = {}

                # Identify date columns
                date_columns = [col for col in columns if self._is_date_column(col)]

                # Get aliases
                aliases = table_info.get('aliases', [])

                metadata = TableMetadata(
                    name=table_name,
                    schema=schema_name,
                    database_id=table_info.get('database_id', 1),
                    columns=columns,
                    date_columns=date_columns,
                    primary_keys=pks,
                    foreign_keys=fks,
                    aliases=aliases,
                    description=table_info.get('description', '')
                )

                self.tables[table_name.lower()] = metadata

                # Index date columns
                if date_columns:
                    self.date_column_index[table_name.lower()] = date_columns

    def _index_relationships(self):
        """Index table relationships from join patterns and FKs."""
        # From join_patterns in config
        join_patterns = self.config.get('join_patterns', [])

        for pattern in join_patterns:
            if isinstance(pattern, dict):
                tables = pattern.get('tables', [])
                condition = pattern.get('condition', '')

                if len(tables) >= 2:
                    # Parse condition to extract columns
                    # Format: "table1.col1 = table2.col2"
                    match = re.match(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', condition)
                    if match:
                        left_table, left_col, right_table, right_col = match.groups()
                        self.relationships.append(TableRelationship(
                            left_table=left_table.lower(),
                            right_table=right_table.lower(),
                            left_column=left_col,
                            right_column=right_col,
                            relationship_type='join_pattern',
                            confidence=0.9
                        ))

        # From foreign keys in table definitions
        for table_name, metadata in self.tables.items():
            for col, ref in metadata.foreign_keys.items():
                if ref and '.' in ref:
                    ref_table, ref_col = ref.split('.', 1)
                    self.relationships.append(TableRelationship(
                        left_table=table_name,
                        right_table=ref_table.lower(),
                        left_column=col,
                        right_column=ref_col,
                        relationship_type='fk',
                        confidence=1.0
                    ))

        # Infer relationships from naming conventions
        self._infer_relationships_from_names()

    def _infer_relationships_from_names(self):
        """Infer relationships from column/table naming patterns."""
        # Common patterns: assetId -> asset, userId -> user, etc.
        for table_name, metadata in self.tables.items():
            for col in metadata.columns:
                col_lower = col.lower()

                # Check for id patterns: asset_id, assetId, assetid
                for pattern in [r'^(\w+)_id$', r'^(\w+)id$', r'^(\w+)Id$']:
                    match = re.match(pattern, col_lower)
                    if match:
                        potential_table = match.group(1).lower()
                        if potential_table in self.tables and potential_table != table_name:
                            # Check if relationship doesn't already exist
                            exists = any(
                                r.left_table == table_name and r.right_table == potential_table
                                for r in self.relationships
                            )
                            if not exists:
                                self.relationships.append(TableRelationship(
                                    left_table=table_name,
                                    right_table=potential_table,
                                    left_column=col,
                                    right_column='id',
                                    relationship_type='name_similarity',
                                    confidence=0.7
                                ))

    def _index_aliases(self):
        """Build alias -> table mapping."""
        # From table aliases
        for table_name, metadata in self.tables.items():
            self.table_by_alias[table_name] = table_name
            for alias in metadata.aliases:
                self.table_by_alias[alias.lower()] = table_name

        # From synonyms in config
        synonyms = self.config.get('synonyms', {})
        for alias, targets in synonyms.items():
            if isinstance(targets, list):
                for target in targets:
                    if target.lower() in self.tables:
                        self.table_by_alias[alias.lower()] = target.lower()
                        break

    def _build_related_tables(self):
        """Build a map of related tables for quick lookup."""
        for rel in self.relationships:
            # Add bidirectional relationships
            if rel.left_table not in self.related_tables:
                self.related_tables[rel.left_table] = set()
            if rel.right_table not in self.related_tables:
                self.related_tables[rel.right_table] = set()

            self.related_tables[rel.left_table].add(rel.right_table)
            self.related_tables[rel.right_table].add(rel.left_table)

    def get_table_metadata(self, table_name: str) -> Optional[TableMetadata]:
        """Get metadata for a table by name or alias."""
        table_name_lower = table_name.lower()

        # Direct lookup
        if table_name_lower in self.tables:
            return self.tables[table_name_lower]

        # Alias lookup
        if table_name_lower in self.table_by_alias:
            return self.tables.get(self.table_by_alias[table_name_lower])

        return None

    def has_date_columns(self, table_name: str) -> bool:
        """Check if a table has date/time columns."""
        metadata = self.get_table_metadata(table_name)
        return metadata is not None and len(metadata.date_columns) > 0

    def get_date_columns(self, table_name: str) -> List[str]:
        """Get date columns for a table."""
        metadata = self.get_table_metadata(table_name)
        return metadata.date_columns if metadata else []

    def find_tables_with_date_columns(self, related_to: Optional[str] = None) -> List[Tuple[str, List[str]]]:
        """Find all tables that have date columns, optionally filtered by relation."""
        results = []

        for table_name, date_cols in self.date_column_index.items():
            if related_to:
                # Check if this table is related to the specified table
                related_to_lower = related_to.lower()
                if related_to_lower in self.related_tables:
                    if table_name not in self.related_tables[related_to_lower]:
                        continue
                else:
                    continue

            results.append((table_name, date_cols))

        return results

    def find_time_filter_alternatives(self, base_table: str) -> List[TableRecommendation]:
        """
        Find alternative tables that can satisfy time filtering requirements.

        This is the key intelligence function that:
        1. Finds tables related to base_table that have date columns
        2. Finds tables with similar names that have date columns
        3. Ranks them by relevance
        """
        base_table_lower = base_table.lower()
        recommendations = []

        # Strategy 1: Find direct alternatives (tables with same base name + date columns)
        # e.g., asset -> asset_creation_overview, asset_usage_unified
        base_name = base_table_lower.split('_')[0]  # Get root name

        for table_name, metadata in self.tables.items():
            if table_name == base_table_lower:
                continue

            # Check if table name contains the base name and has date columns
            if base_name in table_name and metadata.date_columns:
                # Prefer tables with 'creation', 'history', 'log' in name
                priority_keywords = ['creation', 'history', 'log', 'audit', 'overview']
                score = 0.8
                for kw in priority_keywords:
                    if kw in table_name:
                        score = 0.95
                        break

                recommendations.append(TableRecommendation(
                    table_name=metadata.name,
                    score=score,
                    reason=f"Related table with date columns ({', '.join(metadata.date_columns[:2])})",
                    has_date_columns=True,
                    date_columns=metadata.date_columns,
                    related_to=base_table
                ))

        # Strategy 2: Find related tables via relationships
        if base_table_lower in self.related_tables:
            for related_table in self.related_tables[base_table_lower]:
                if related_table in self.date_column_index:
                    metadata = self.tables.get(related_table)
                    if metadata and metadata not in [r.table_name.lower() for r in recommendations]:
                        # Find the join path
                        join_path = self._find_join_path(base_table_lower, related_table)

                        recommendations.append(TableRecommendation(
                            table_name=metadata.name,
                            score=0.75,
                            reason=f"Can join for date filtering via {metadata.date_columns[0]}",
                            has_date_columns=True,
                            date_columns=metadata.date_columns,
                            related_to=base_table,
                            join_path=join_path
                        ))

        # Strategy 3: Check common audit/history tables
        common_date_tables = ['audithistory', 'auditlog', 'audit_history', 'entitygraph']
        for audit_table in common_date_tables:
            if audit_table in self.tables and audit_table not in [r.table_name.lower() for r in recommendations]:
                metadata = self.tables[audit_table]
                if metadata.date_columns:
                    recommendations.append(TableRecommendation(
                        table_name=metadata.name,
                        score=0.6,
                        reason=f"Audit/history table with timestamps ({metadata.date_columns[0]})",
                        has_date_columns=True,
                        date_columns=metadata.date_columns,
                        related_to=base_table
                    ))

        # Sort by score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)

        return recommendations

    def _find_join_path(self, from_table: str, to_table: str) -> Optional[List[dict]]:
        """Find the join path between two tables."""
        for rel in self.relationships:
            if (rel.left_table == from_table and rel.right_table == to_table) or \
               (rel.right_table == from_table and rel.left_table == to_table):
                return [{
                    'left_table': rel.left_table,
                    'right_table': rel.right_table,
                    'left_column': rel.left_column,
                    'right_column': rel.right_column,
                    'join_type': 'LEFT'
                }]
        return None

    def recommend_tables_for_query(
        self,
        entity: str,
        candidates: List[dict],
        has_time_filter: bool = False,
        time_filter_type: str = None  # 'created', 'updated', 'expired', etc.
    ) -> List[dict]:
        """
        Recommend tables for a query, prioritizing based on requirements.

        Args:
            entity: The entity being queried (e.g., "assets")
            candidates: Current candidate tables from semantic matcher
            has_time_filter: Whether the query has time filtering requirements

        Returns:
            Reordered and enhanced candidates list
        """
        if not has_time_filter:
            return candidates

        # Find the best candidate that has date columns
        enhanced_candidates = []
        seen_tables = set()

        # First, check if any candidates have date columns
        for candidate in candidates:
            table_name = candidate.get('table_name', '').lower()
            metadata = self.get_table_metadata(table_name)

            if metadata and metadata.date_columns:
                # This candidate has date columns - boost its score
                enhanced = candidate.copy()
                enhanced['score'] = max(candidate.get('score', 0), 0.95)
                enhanced['match_reason'] = f"Has date columns ({', '.join(metadata.date_columns[:2])})"
                enhanced['has_date_columns'] = True
                enhanced['date_columns'] = metadata.date_columns
                enhanced_candidates.append(enhanced)
                seen_tables.add(table_name)

        # Find alternatives for candidates without date columns
        for candidate in candidates:
            table_name = candidate.get('table_name', '').lower()

            if table_name in seen_tables:
                continue

            metadata = self.get_table_metadata(table_name)

            if metadata and not metadata.date_columns:
                # Find time-filter alternatives
                alternatives = self.find_time_filter_alternatives(table_name)

                for alt in alternatives[:2]:  # Top 2 alternatives
                    if alt.table_name.lower() not in seen_tables:
                        enhanced_candidates.append({
                            'table_name': alt.table_name,
                            'score': alt.score,
                            'match_reason': alt.reason,
                            'has_date_columns': True,
                            'date_columns': alt.date_columns,
                            'alternative_for': table_name
                        })
                        seen_tables.add(alt.table_name.lower())

            # Add original candidate too (lower priority if no date columns)
            enhanced = candidate.copy()
            if table_name not in seen_tables:
                if metadata and not metadata.date_columns:
                    enhanced['match_reason'] = f"{candidate.get('match_reason', '')} (no date columns)"
                enhanced_candidates.append(enhanced)
                seen_tables.add(table_name)

        # Calculate relevance score based on:
        # 1. Has date columns
        # 2. Date column matches query intent (created, updated, expired, etc.)
        # 3. Original match score
        def calculate_relevance(candidate):
            has_dates = candidate.get('has_date_columns', False)
            if not has_dates:
                return (0, candidate.get('score', 0))

            date_cols = candidate.get('date_columns', [])
            intent_match = 0

            # Check if date columns match the filter type intent
            if time_filter_type:
                filter_lower = time_filter_type.lower()
                for col in date_cols:
                    col_lower = col.lower()
                    # Direct match: "created" matches "created_at", "createdon", etc.
                    if filter_lower in col_lower:
                        intent_match = 2
                        break
                    # Semantic match: "created" also matches "timestamp" for general time
                    if filter_lower in ['created', 'create'] and any(x in col_lower for x in ['timestamp', '_at', 'date']):
                        intent_match = max(intent_match, 1)

            return (1, intent_match, candidate.get('score', 0))

        # Sort by calculated relevance
        enhanced_candidates.sort(key=calculate_relevance, reverse=True)

        return enhanced_candidates


# Global instance
_intelligence: Optional[TableIntelligence] = None


def get_table_intelligence() -> TableIntelligence:
    """Get or create the global TableIntelligence instance."""
    global _intelligence
    if _intelligence is None:
        _intelligence = TableIntelligence()
    return _intelligence