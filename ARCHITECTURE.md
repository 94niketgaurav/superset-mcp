# Superset-MCP Architecture Documentation

## Overview

Superset-MCP is an intelligent MCP (Model Context Protocol) server that enables natural language to SQL conversion for Apache Superset. It features auto-learning capabilities to improve over time.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                                                                             │
│   "show me assets and events by joining on assetId for last week"          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR LAYER                                 │
│                           (orchestrator.py)                                  │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │   Step 1    │  │   Step 2    │  │   Step 3    │  │   Step 4    │       │
│   │   Extract   │─▶│   Match     │─▶│   Suggest   │─▶│  Generate   │       │
│   │  Entities   │  │  Datasets   │  │   Joins     │  │    SQL      │       │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                              │               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │               │
│   │   Step 7    │  │   Step 6    │  │   Step 5    │         │               │
│   │   Execute   │◀─│   Confirm   │◀─│  Validate   │◀────────┘               │
│   └─────────────┘  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MCP SERVER LAYER                                  │
│                        (mcp_superset_server.py)                              │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                         MCP TOOLS                                    │   │
│   │                                                                      │   │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │   │
│   │  │ extract_query_   │  │ discover_        │  │ suggest_enhanced_│  │   │
│   │  │ entities()       │  │ datasets()       │  │ joins()          │  │   │
│   │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │   │
│   │                                                                      │   │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │   │
│   │  │ validate_sql()   │  │ build_execution_ │  │ execute_sql()    │  │   │
│   │  │                  │  │ plan()           │  │                  │  │   │
│   │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │   │
│   │                                                                      │   │
│   │  + Original tools: list_datasets, get_datasets, ensure_same_db...   │   │
│   └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│     NLU MODULE        │ │    JOINS MODULE       │ │  VALIDATION MODULE    │
│     (nlu/)            │ │    (joins/)           │ │  (validation/)        │
│                       │ │                       │ │                       │
│ ┌───────────────────┐ │ │ ┌───────────────────┐ │ │ ┌───────────────────┐ │
│ │ EntityExtractor   │ │ │ │ ColumnNormalizer  │ │ │ │ SQLValidator      │ │
│ │ - Extract nouns   │ │ │ │ - camelCase       │ │ │ │ - Syntax check    │ │
│ │ - Detect joins    │ │ │ │ - snake_case      │ │ │ │ - Safety check    │ │
│ │ - Time filters    │ │ │ │ - FK detection    │ │ │ │ - Best practices  │ │
│ └───────────────────┘ │ │ └───────────────────┘ │ │ └───────────────────┘ │
│                       │ │                       │ │                       │
│ ┌───────────────────┐ │ │ ┌───────────────────┐ │ └───────────────────────┘
│ │ SemanticMatcher   │ │ │ │EnhancedJoinReasoner│
│ │ - Exact match     │ │ │ │ - YAML priority   │ │
│ │ - Fuzzy match     │ │ │ │ - FK→PK patterns  │ │
│ │ - Learned aliases │ │ │ │ - Type checking   │ │
│ └───────────────────┘ │ │ └───────────────────┘ │
└───────────────────────┘ └───────────────────────┘
            │                         │
            └────────────┬────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LEARNING MODULE                                     │
│                          (learning/)                                         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       KnowledgeStore                                 │   │
│   │                    (~/.superset-mcp/knowledge.db)                    │   │
│   │                                                                      │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│   │   │  Datasets   │  │Join Patterns│  │  Synonyms   │  │  Column   │  │   │
│   │   │  + Columns  │  │ left→right  │  │  term→alias │  │  Patterns │  │   │
│   │   │  + Samples  │  │ confidence  │  │  learned    │  │  hash     │  │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│   │ DatasetLearner  │  │ ColumnAnalyzer  │  │ Training        │            │
│   │ - Fetch all     │  │ - Detect types  │  │ Pipeline        │            │
│   │ - Learn schema  │  │ - Sample values │  │ (daily cron)    │            │
│   │ - Store info    │  │ - Join hints    │  │                 │            │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM LAYER                                          │
│                                                                             │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐       │
│   │        OpenAI (GPT-4)        │    │      Google Gemini           │       │
│   │                             │    │                             │       │
│   │   - SQL Generation          │    │   - SQL Critique            │       │
│   │   - Handle schema context   │    │   - Validation              │       │
│   │   - Follow engine dialect   │    │   - Improvement suggestions │       │
│   └─────────────────────────────┘    └─────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SUPERSET API LAYER                                    │
│                        (SupersetClient)                                      │
│                                                                             │
│   Authentication │ Dataset API │ SQL Lab API │ Results API                  │
│                                                                             │
│                              ▼                                               │
│                     Apache Superset                                          │
│                   (External Service)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. NLU Module (`nlu/`)

**Purpose**: Natural Language Understanding for query parsing

#### EntityExtractor (`entity_extractor.py`)
- Extracts dataset entities from natural language
- Detects join hints ("assets and events", "user's orders")
- Identifies time filters ("last week", "since January")
- Extracts aggregations ("count", "sum", "average")
- Column mention detection ("by assetId")

#### SemanticMatcher (`semantic_matcher.py`)
- Matches entity names to actual Superset datasets
- Multiple matching strategies:
  1. Exact match (score: 1.0)
  2. Learned alias (score: 0.98)
  3. Learned synonym (score: 0.96)
  4. Built-in synonym (score: 0.95)
  5. Prefix match (score: 0.7-0.9)
  6. Suffix match (score: 0.65-0.8)
  7. Contains match (score: 0.5-0.65)
  8. Fuzzy match (variable)
- Ambiguity detection for user prompting
- Auto-reload from knowledge store

### 2. Joins Module (`joins/`)

**Purpose**: Intelligent join detection and column normalization

#### ColumnNormalizer (`normalizer.py`)
- Converts between naming conventions:
  - `assetId` ↔ `asset_id` ↔ `AssetId`
  - `userId` ↔ `user_id` ↔ `UserId`
- Extracts entity from FK names: `assetId` → `("asset", "id")`
- ID column detection

#### EnhancedJoinReasoner (`enhanced_reasoner.py`)
- Multi-strategy join detection:
  1. YAML relationships (confidence: 1.0)
  2. Normalized column matches (confidence: 0.75-0.85)
  3. FK → PK patterns (confidence: 0.65-0.80)
- Type compatibility checking
- Uses learned join patterns from knowledge store

### 3. Validation Module (`validation/`)

**Purpose**: SQL validation before execution

#### SQLValidator (`sql_validator.py`)
- Syntax checking (balanced parens, quotes)
- Dangerous operation blocking (READ_ONLY mode)
- Column reference validation
- Best practice warnings (SELECT *, missing LIMIT)
- Join condition checking

### 4. Learning Module (`learning/`)

**Purpose**: Auto-learning and knowledge persistence

#### KnowledgeStore (`knowledge_store.py`)
- SQLite-based persistent storage
- Stores:
  - Dataset metadata and columns
  - Join patterns with confidence
  - Synonyms and aliases
  - Column patterns (for value matching)
- Export/import to JSON

#### DatasetLearner (`dataset_learner.py`)
- Fetches all datasets from Superset
- Analyzes column types and patterns
- Fetches sample values
- Discovers join patterns via value overlap
- Stores knowledge for semantic matcher

#### ColumnAnalyzer (`column_analyzer.py`)
- Detects column patterns (UUID, integer ID, email, etc.)
- Computes value hashes for matching
- Calculates value overlap between columns
- Infers FK relationships from names

### 5. MCP Server (`mcp_superset_server.py`)

**Purpose**: FastMCP server exposing tools for Claude/MCP clients

#### New Tools:
| Tool | Purpose |
|------|---------|
| `extract_query_entities` | NLU parsing of natural language |
| `discover_datasets` | Semantic dataset matching |
| `suggest_enhanced_joins` | Advanced join detection |
| `validate_sql` | Pre-execution validation |
| `build_execution_plan` | Complete NL→SQL planning |

#### Original Tools:
- `list_datasets` - Search datasets
- `get_dataset` / `get_datasets` - Fetch metadata
- `ensure_same_database` - Database validation
- `list_relationships` - YAML relationships
- `suggest_joins` - Basic join heuristics
- `execute_sql` - Run SQL queries

### 6. Orchestrator (`orchestrator.py`)

**Purpose**: Interactive workflow coordination

#### Steps:
1. **Extract & Discover** - Parse query, match datasets
2. **Resolve Ambiguity** - User selects from candidates
3. **Confirm Joins** - Show and confirm join suggestions
4. **Generate SQL** - LLM generates SQL
5. **Improve SQL** - Gemini reviews/improves
6. **Validate** - Check for issues
7. **Execute** - Run after user confirmation

## Data Flow

### Query Processing Flow

```
User Query: "show me assets and events by assetId"
                │
                ▼
    ┌───────────────────────┐
    │  EntityExtractor      │
    │  - entities: [assets, │
    │    events]            │
    │  - columns: [assetId] │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  SemanticMatcher      │
    │  - assets → assets    │
    │    (score: 1.0)       │
    │  - events → [posthog  │
    │    events, audit      │
    │    events] (ambiguous)│
    └───────────────────────┘
                │
                ▼ (user selects posthogevents)
    ┌───────────────────────┐
    │  EnhancedJoinReasoner │
    │  - posthogevents.     │
    │    assetId →          │
    │    assets.id          │
    │  - confidence: 0.82   │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  OpenAI SQL Draft     │
    │  SELECT a.id, e.*     │
    │  FROM assets a        │
    │  JOIN posthogevents e │
    │  ON e.assetId = a.id  │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Gemini SQL Review    │
    │  - approved: true     │
    │  - notes: [...]       │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  SQLValidator         │
    │  - is_valid: true     │
    │  - warnings: [LIMIT]  │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Execute SQL          │
    │  → Results            │
    └───────────────────────┘
```

### Learning Flow

```
                           ┌────────────────────┐
                           │  Training Trigger  │
                           │  (daily cron)      │
                           └─────────┬──────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DatasetLearner                              │
│                                                                  │
│  1. Fetch all datasets from Superset API                        │
│     └─▶ GET /api/v1/dataset/ (paginated)                        │
│                                                                  │
│  2. For each dataset:                                            │
│     a. Fetch detailed metadata                                   │
│        └─▶ GET /api/v1/dataset/{id}                             │
│     b. Fetch sample values for ID columns                        │
│        └─▶ POST /api/v1/sqllab/execute/                         │
│     c. Analyze columns (type, pattern, FK references)            │
│     d. Generate aliases (singular/plural)                        │
│     e. Store in KnowledgeStore                                   │
│                                                                  │
│  3. Discover join patterns:                                      │
│     a. Index all ID columns with their values                    │
│     b. Compute value overlap between columns                     │
│     c. Infer FK→PK relationships from column names               │
│     d. Store join patterns with confidence scores                │
└─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      KnowledgeStore                              │
│                   (SQLite database)                              │
│                                                                  │
│  Tables:                                                         │
│  ├─ datasets: id, table_name, schema, knowledge_json             │
│  ├─ join_patterns: left_table, left_col, right_table, right_col │
│  ├─ synonyms: term, synonym, source, confidence                  │
│  └─ column_patterns: table, column, pattern_type, sample_hash   │
└─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SemanticMatcher                                │
│                                                                  │
│  On initialization or reload_knowledge():                        │
│  - Load datasets from store                                      │
│  - Load learned synonyms                                         │
│  - Load learned aliases                                          │
│  - Load learned join patterns                                    │
│  - Build matching indexes                                        │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
superset-mcp/
├── config.py                    # Secure configuration
├── .env.example                 # Environment template
├── mcp_superset_server.py       # MCP server with 12 tools
├── orchestrator.py              # Interactive workflow
├── train_semantic_matcher.py    # Training pipeline
├── requirements.txt             # Dependencies
│
├── nlu/                         # Natural Language Understanding
│   ├── __init__.py
│   ├── entity_extractor.py      # Query parsing
│   └── semantic_matcher.py      # Dataset matching
│
├── joins/                       # Join Detection
│   ├── __init__.py
│   ├── normalizer.py            # Column normalization
│   └── enhanced_reasoner.py     # Join reasoning
│
├── validation/                  # SQL Validation
│   ├── __init__.py
│   └── sql_validator.py         # Validation logic
│
├── learning/                    # Auto-Learning
│   ├── __init__.py
│   ├── knowledge_store.py       # Persistent storage
│   ├── dataset_learner.py       # Learning logic
│   └── column_analyzer.py       # Column analysis
│
└── tests/                       # Test Suite
    ├── __init__.py
    └── test_happy_flow.py       # Integration tests
```

## Security Considerations

1. **No Hardcoded Credentials**: All secrets via environment variables
2. **READ_ONLY Mode**: Blocks INSERT, UPDATE, DELETE by default
3. **Database Whitelist**: Optional ALLOWED_DATABASE_IDS
4. **SQL Validation**: Pre-execution safety checks
5. **User Confirmation**: Interactive mode requires approval

## Performance Optimizations

1. **Dataset Caching**: 5-minute TTL for dataset list
2. **Knowledge Store**: SQLite with indexes
3. **Fuzzy Matching**: rapidfuzz when available (10x faster)
4. **Incremental Learning**: Only fetch changed datasets (TODO)