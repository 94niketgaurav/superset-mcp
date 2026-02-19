-- Migration: 001_initial_schema
-- Description: Create initial schema for superset-mcp knowledge store
-- Created: 2024-01-15

-- Create schema for superset-mcp
CREATE SCHEMA IF NOT EXISTS superset_mcp;

-- Migration tracking table
CREATE TABLE IF NOT EXISTS superset_mcp.migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Datasets table - stores metadata about Superset datasets
CREATE TABLE IF NOT EXISTS superset_mcp.datasets (
    dataset_id INTEGER PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    schema_name VARCHAR(255),
    database_id INTEGER,
    database_name VARCHAR(255),
    description TEXT,
    row_count INTEGER,
    knowledge_json JSONB,
    aliases TEXT[],
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_datasets_table_name ON superset_mcp.datasets(table_name);
CREATE INDEX IF NOT EXISTS idx_datasets_database_id ON superset_mcp.datasets(database_id);
CREATE INDEX IF NOT EXISTS idx_datasets_schema_name ON superset_mcp.datasets(schema_name);
CREATE INDEX IF NOT EXISTS idx_datasets_knowledge_json ON superset_mcp.datasets USING GIN (knowledge_json);

-- Join patterns table - stores learned join relationships
CREATE TABLE IF NOT EXISTS superset_mcp.join_patterns (
    id SERIAL PRIMARY KEY,
    left_table VARCHAR(255) NOT NULL,
    left_column VARCHAR(255) NOT NULL,
    right_table VARCHAR(255) NOT NULL,
    right_column VARCHAR(255) NOT NULL,
    join_type VARCHAR(50) DEFAULT 'inner',
    confidence DECIMAL(5,4) DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP WITH TIME ZONE,
    value_overlap_ratio DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(left_table, left_column, right_table, right_column)
);

CREATE INDEX IF NOT EXISTS idx_join_patterns_left_table ON superset_mcp.join_patterns(left_table);
CREATE INDEX IF NOT EXISTS idx_join_patterns_right_table ON superset_mcp.join_patterns(right_table);
CREATE INDEX IF NOT EXISTS idx_join_patterns_confidence ON superset_mcp.join_patterns(confidence DESC);

-- Synonyms table - stores table name aliases and synonyms
CREATE TABLE IF NOT EXISTS superset_mcp.synonyms (
    id SERIAL PRIMARY KEY,
    term VARCHAR(255) NOT NULL,
    synonym VARCHAR(255) NOT NULL,
    source VARCHAR(50) DEFAULT 'learned',  -- 'manual', 'learned', 'inferred'
    confidence DECIMAL(5,4) DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(term, synonym)
);

CREATE INDEX IF NOT EXISTS idx_synonyms_term ON superset_mcp.synonyms(term);
CREATE INDEX IF NOT EXISTS idx_synonyms_synonym ON superset_mcp.synonyms(synonym);

-- Column patterns table - stores column value patterns for join detection
CREATE TABLE IF NOT EXISTS superset_mcp.column_patterns (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100),  -- 'uuid', 'integer_id', 'email', 'date', etc.
    sample_hash VARCHAR(64),    -- Hash of sample values for comparison
    sample_values JSONB,        -- Store sample values for analysis
    statistics JSONB,           -- Column statistics (distinct count, null count, etc.)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(table_name, column_name)
);

CREATE INDEX IF NOT EXISTS idx_column_patterns_table ON superset_mcp.column_patterns(table_name);
CREATE INDEX IF NOT EXISTS idx_column_patterns_type ON superset_mcp.column_patterns(pattern_type);

-- Metadata table - stores configuration and state
CREATE TABLE IF NOT EXISTS superset_mcp.metadata (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Databases table - stores Superset database information
CREATE TABLE IF NOT EXISTS superset_mcp.databases (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    backend VARCHAR(100),
    schemas TEXT[],
    dataset_count INTEGER DEFAULT 0,
    last_synced TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Query history table - stores executed queries for learning
CREATE TABLE IF NOT EXISTS superset_mcp.query_history (
    id SERIAL PRIMARY KEY,
    natural_language_query TEXT,
    generated_sql TEXT,
    dataset_ids INTEGER[],
    was_executed BOOLEAN DEFAULT FALSE,
    was_successful BOOLEAN,
    user_feedback VARCHAR(50),  -- 'approved', 'rejected', 'modified'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_history_created ON superset_mcp.query_history(created_at DESC);

-- Record this migration
INSERT INTO superset_mcp.migrations (version, name)
VALUES ('001', 'initial_schema')
ON CONFLICT (version) DO NOTHING;