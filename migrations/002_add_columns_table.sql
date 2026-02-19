-- Migration: 002_add_columns_table
-- Description: Add dedicated columns table for better querying
-- Created: 2024-01-15

-- Columns table - normalized column information for faster queries
CREATE TABLE IF NOT EXISTS superset_mcp.columns (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES superset_mcp.datasets(dataset_id) ON DELETE CASCADE,
    column_name VARCHAR(255) NOT NULL,
    column_type VARCHAR(100),
    is_nullable BOOLEAN DEFAULT TRUE,
    is_primary_key BOOLEAN DEFAULT FALSE,
    is_foreign_key BOOLEAN DEFAULT FALSE,
    fk_references VARCHAR(255),  -- "table.column" format
    distinct_count INTEGER,
    null_count INTEGER,
    sample_values JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_id, column_name)
);

CREATE INDEX IF NOT EXISTS idx_columns_dataset_id ON superset_mcp.columns(dataset_id);
CREATE INDEX IF NOT EXISTS idx_columns_name ON superset_mcp.columns(column_name);
CREATE INDEX IF NOT EXISTS idx_columns_is_pk ON superset_mcp.columns(is_primary_key) WHERE is_primary_key = TRUE;
CREATE INDEX IF NOT EXISTS idx_columns_is_fk ON superset_mcp.columns(is_foreign_key) WHERE is_foreign_key = TRUE;

-- Record this migration
INSERT INTO superset_mcp.migrations (version, name)
VALUES ('002', 'add_columns_table')
ON CONFLICT (version) DO NOTHING;