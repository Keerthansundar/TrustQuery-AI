-- ==============================================================================
-- 03_rag_app.sql: Retrieval Chunks & Application Memory / Audit Log Tables
-- ==============================================================================

-- Document chunks table with 1536-dim OpenAI embeddings for vector search
CREATE TABLE IF NOT EXISTS rag.documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536),
    content_hash TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Cosine distance index for exact search or IVFFlat/HNSW (exact cosine search used initially)
CREATE INDEX IF NOT EXISTS idx_rag_documents_content_hash ON rag.documents(content_hash);

-- Multi-turn session memory
CREATE TABLE IF NOT EXISTS app.sessions (
    session_id TEXT NOT NULL,
    turn_idx INT NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    question TEXT NOT NULL,
    standalone TEXT,
    sql TEXT,
    result_summary JSONB,
    answer TEXT,
    PRIMARY KEY (session_id, turn_idx)
);

CREATE INDEX IF NOT EXISTS idx_app_sessions_session_id ON app.sessions(session_id, turn_idx);

-- Execution & security audit log
CREATE TABLE IF NOT EXISTS app.query_log (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    intent TEXT,
    standalone TEXT,
    sql TEXT,
    ok BOOLEAN,
    attempts INT,
    latency_ms NUMERIC,
    fallback_used BOOLEAN,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_app_query_log_session_ts ON app.query_log(session_id, ts);
