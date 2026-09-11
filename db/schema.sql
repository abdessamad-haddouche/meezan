-- Project Meezan — SQLite schema (docs/FRD.md, Section 5).
-- Immutable snapshots: rows are never UPDATEd after a sweep completes,
-- except sweep_runs.status / total_cost_usd while running, and
-- ideas.has_deep_pass / deep-pass columns when a Deep Dive runs (Section 7.2).

CREATE TABLE sweep_runs (
    id                   INTEGER PRIMARY KEY,
    category             TEXT NOT NULL,
    market               TEXT NOT NULL,
    triggered_at         TEXT NOT NULL,          -- ISO 8601
    status               TEXT NOT NULL,           -- running|completed|failed
    total_cost_usd       REAL NOT NULL DEFAULT 0,
    scoring_model_version TEXT NOT NULL DEFAULT 'v1',
    notes                TEXT
);

CREATE TABLE ideas (
    id                   INTEGER PRIMARY KEY,
    sweep_run_id         INTEGER NOT NULL REFERENCES sweep_runs(id),
    name                 TEXT NOT NULL,
    description          TEXT,                    -- plain-English, 1 paragraph
    category             TEXT NOT NULL,
    market               TEXT NOT NULL,
    seed_or_suggested    TEXT NOT NULL,            -- seed|suggested
    composite_score      REAL,                     -- 0-10, deterministic
    composite_confidence REAL,                     -- 0-1
    margin_min_pct       REAL,
    margin_max_pct       REAL,
    cost_min_usd         INTEGER,
    cost_max_usd         INTEGER,
    complexity           INTEGER,                  -- 1-5
    research_priority    TEXT,                      -- research_this_week|revisit_later|deprioritize
    has_deep_pass        INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL
);

CREATE TABLE scores (
    id                   INTEGER PRIMARY KEY,
    idea_id              INTEGER NOT NULL REFERENCES ideas(id),
    dimension            TEXT NOT NULL,   -- market_demand|competitive_intensity|margin|
                                           -- cost_to_start|complexity|regulatory_friction|trend_momentum
    value                REAL NOT NULL,   -- 1-10 normalized
    confidence           REAL NOT NULL,   -- 0-1
    evidence_level       TEXT NOT NULL,   -- sufficient|insufficient
    rationale            TEXT,
    comparable_products  TEXT,            -- JSON array, mainly for margin
    pass_type            TEXT NOT NULL    -- cheap|deep
);

CREATE TABLE evidence (
    id                   INTEGER PRIMARY KEY,
    idea_id              INTEGER NOT NULL REFERENCES ideas(id),
    source_type          TEXT NOT NULL,   -- reddit|google_trends|meta_ads|serp|
                                           -- marketplace|worldbank|llm_estimate
    metric               TEXT NOT NULL,
    value                TEXT NOT NULL,   -- stored as text; parse per metric
    confidence           REAL NOT NULL,
    source_url           TEXT,
    raw_data             TEXT,            -- JSON blob of the raw API response
    retrieved_at         TEXT NOT NULL,
    cost_usd             REAL NOT NULL DEFAULT 0
);

CREATE INDEX idx_ideas_sweep ON ideas(sweep_run_id);
CREATE INDEX idx_scores_idea ON scores(idea_id);
CREATE INDEX idx_evidence_idea ON evidence(idea_id);
