# Project Meezan — Functional Requirements Document (v1.0)
### Business Idea Feasibility, Competition & Margin Ranking Engine

**Status:** Ready to build.
**Owner:** Personal, single-user tool.
**Build tool:** Claude Code ($20/mo plan — see Section 14 for how to work with it task-by-task).

---

## 0. How to use this document

This file is the **single source of truth**. It replaces the earlier FRD, the
Implementation Guide, and the various AI review comments — it already absorbs
the good parts of all of them and drops the parts that don't fit a
single-user, $9-budget, one-person-building-it-with-Claude-Code project.

**Practical workflow:**

1. Save this file into your repo as `docs/FRD.md`. Keep it there — don't
   re-paste it into every Claude Code session. Point Claude Code at the file
   instead (`Read docs/FRD.md for context`).
2. Never ask Claude Code to "build the app." Give it **one task at a time**
   from **Section 14**. Each task is scoped to a small, independently
   verifiable slice of work — roughly what fits in one focused Claude Code
   session without it wandering into other files.
3. Section 14 includes a copy-paste prompt template. Use it every time.
4. Sections 1–13 are the spec Claude Code should treat as ground truth when a
   task references "per the FRD."

If you (the human) change your mind about something mid-build, edit this
file first, then give Claude Code the next task referencing the updated
section. Don't argue scope changes in chat — argue them here.

---

## 1. Product Vision

Project Meezan is a personal decision-support tool that takes a **large
batch of candidate businesses** — ecommerce niches, digital products, SaaS
ideas, content niches, affiliate niches, service businesses — and ranks them
against each other so you know which handful are worth real manual research
next (customer interviews, landing-page tests, presells).

**It does not issue a yes/no verdict.** No scoring model — not this one, not
a VC's, not a consulting firm's — reliably predicts which single idea
succeeds. What structured scoring *is* good at is removing noise and
inconsistency from the **ranking** decision across dozens of candidates at
once. Every part of this tool is built to be honest about that, not to
oversell certainty it doesn't have.

**Why this exists:** a pattern of missing early opportunities (Bitcoin,
Reddit-to-video) because manual evaluation was too slow, combined with a
constant flood of idea candidates from business subreddits that manual
triage can't keep up with.

---

## 2. Architecture Principles

These are the rules Claude Code should follow on every task, not just the
ones that mention them explicitly.

1. **The LLM is an analyst, not the ranking engine.** DeepSeek produces
   per-dimension *judgments* (it's the only thing that can turn "213 Meta
   advertisers + 4,800 Etsy listings" into a 1–10 competitive-intensity
   score with a rationale). But the **composite score is computed by
   deterministic code**, not asked for from the model. This keeps the final
   ranking auditable and reproducible from the stored sub-scores — you can
   re-run the math without re-calling the LLM.
2. **Every number has provenance.** A score is never just `72`. It's `72`,
   traced to specific evidence rows, with a confidence value. If no real
   evidence was found, the dimension is marked `insufficient` and the range
   widens — the LLM is never allowed to quietly invent a number.
3. **Cheap first, deep later.** Every niche in a sweep gets a cheap,
   batched pass. Only the top slice (configurable, default top ~20%) gets
   the expensive deep pass. This is a hard budget rule, not a nice-to-have.
4. **A category is a config, not a code branch.** Ecommerce vs. digital
   product vs. SaaS differ in seed lists and prompt wording, not in
   pipeline logic. `weights.yaml` and seed-list files are how you add a new
   category — not new Python classes. (A full plugin-class system per
   category, as some reviewers suggested, is overkill for one person
   evaluating ideas — see Section 13.)
5. **Snapshots are immutable.** A sweep run's ideas/scores/evidence, once
   written, are never overwritten by a later sweep. This is nearly free to
   build now (just don't add UPDATE-in-place logic) and gives you idea
   history for free later.
6. **No verdicts.** Output is a research-priority tier, never "build this"
   or "skip this."

---

## 3. Constraints (do not relitigate these mid-build)

- **Personal, single-user.** No auth, no multi-tenant, not for resale.
- **DeepSeek budget: ~$9 total for prototyping.** The cheap pass must be
  cheap enough that a 50-niche sweep costs cents, not dollars. See Section
  9 for the batching strategy that makes this true, and check DeepSeek's
  current pricing page before writing the cost-guard constants — rates
  change and this document won't stay current on that.
- **Claude Code budget: $20/mo plan.** This is *why* Section 14 exists —
  work in small, single-purpose tasks so you don't burn a whole session's
  usage re-reading and re-reasoning about the entire codebase for a
  one-file change.
- **Free-tier data sources only.** Never scrape a source whose ToS
  prohibits it. Prefer an official API; if none exists and scraping is
  disallowed, skip the source and fall back to a flagged LLM estimate.
- **Never let the LLM invent a statistic.** Every number traces to
  retrieved evidence or is explicitly flagged `insufficient_evidence`.

---

## 4. System Architecture

```
                          Web UI (Streamlit)
        sweep trigger · ranked cards/table · deep-dive · spend sidebar
                                   │
                        Sweep Controller / Orchestrator
                    (SweepEngine, DiscoveryEngine — v1.1)
                                   │
                    ┌──────────────┴──────────────┐
                    │                              │
            Data Collection                 DeepSeek Provider
       (adapters → Evidence objects)      (cheap pass / deep pass /
                    │                       niche expansion)
                    │                              │
                    └──────────────┬───────────────┘
                                   │
                      Deterministic Scoring Engine
             (weighted composite score, confidence propagation,
                       research-priority tiering)
                                   │
                          SQLite Persistence
              (immutable sweep_runs / ideas / evidence / scores)
```

Adapters never talk to the LLM. The LLM never talks to adapters directly
(no raw tool-calling access for DeepSeek — it receives a pre-compiled text
payload of evidence and returns structured JSON). This keeps token usage
predictable and avoids the tool-calling loops that eat budget unpredictably.

---

## 5. Data Model (SQLite)

Denormalized "fast columns" live on `ideas` for sorting/filtering in the UI
(per-dimension detail still lives in `scores`/`evidence` for drill-down).

```sql
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
```

Nothing here is ever `UPDATE`d after a sweep completes except
`sweep_runs.status` and `sweep_runs.total_cost_usd` while it's running, and
`ideas.has_deep_pass` / the deep-pass columns when a Deep Dive is run on a
specific idea (that's an intentional, additive update — see Section 7.2).

---

## 6. Scoring Methodology

| Dimension | Weight | Notes |
|---|---|---|
| Market demand / feasibility | 20% | Real search/community signal, not vibes |
| Competitive intensity | 15% | Ad saturation, SERP competitor count, marketplace listings |
| Margin / profitability | 20% | Ecommerce: gross margin %. Digital: achievable price given saturation |
| Cost to start | 15% | $ to reach a sellable v1 |
| Complexity to build | 10% | 1–5, shown standalone too |
| Regulatory / operational friction | 10% | Payment rails, customs, COD — relevant for MENA sweeps |
| Trend momentum | 10% | Rising/flat/falling, not just current level |

`config/weights.yaml`:

```yaml
weights:
  market_demand: 0.20
  competitive_intensity: 0.15
  margin: 0.20
  cost_to_start: 0.15
  complexity: 0.10
  regulatory_friction: 0.10
  trend_momentum: 0.10

research_priority_thresholds:
  research_this_week_percentile: 0.85   # top 15%
  revisit_later_percentile: 0.30        # next 55%
  # bottom 30% = deprioritize
```

**Composite score (deterministic, computed in code — never asked from the
LLM):**

```
composite = Σ (dimension_value_1to10 * weight)          # 0-10 scale
composite_confidence = Σ (dimension_confidence * weight) # 0-1 scale
```

**Research priority tiering** (computed after a sweep completes, over all
ideas that reached at least the cheap pass): rank by `composite`, assign
`research_this_week` / `revisit_later` / `deprioritize` using the
percentile thresholds above. Never surface "proceed"/"don't proceed"
anywhere in the UI or prompts.

Every sub-score carries: `evidence_level` (`sufficient`/`insufficient`), a
one-line `rationale`, and — for margin specifically — the
`comparable_products` it was grounded in. If no comparable data was found,
widen the range and mark `insufficient` rather than inventing precision.

---

## 7. DeepSeek Prompt Contracts

All calls request **strict JSON only**, validated against a Pydantic model
before it touches the database. If validation fails, retry once with the
validation error appended to the prompt; if it fails twice, store the idea
with `evidence_level = insufficient` and move on — never crash the batch.

### 7.1 Niche expansion (once per sweep)

```python
class NicheExpansion(BaseModel):
    suggested_niches: list[str]
```

```
Given the parent category "{category}" and seed sub-niches {seed_list},
propose 10-20 additional specific sub-niches worth evaluating in the
{market} market. Return ONLY JSON matching this shape:
{"suggested_niches": ["...", "..."]}
```

### 7.2 Cheap pass — BATCHED, this is the main cost lever

Do not call the LLM once per niche. Batch **8 niches per call**. A 50-niche
sweep is ~7 calls instead of 50 — this is the single biggest cost saving
over the original one-call-per-niche design, so don't skip it.

```python
class CheapNicheScore(BaseModel):
    niche: str
    market_demand: int          # 1-10
    competitive_intensity: int  # 1-10
    margin_signal: int          # 1-10, rough — real $ ranges come from deep pass
    trend_momentum: int         # 1-10
    evidence_level: Literal["sufficient", "insufficient"]
    rationale: str              # one line, <= 200 chars

class CheapPassResponse(BaseModel):
    scores: list[CheapNicheScore]
```

```
You are an unbiased business-idea evaluator. For EACH niche below, using
only the evidence provided for that niche, rate market_demand,
competitive_intensity, margin_signal, and trend_momentum on a 1-10 scale.
If evidence for a niche is thin, say so in evidence_level rather than
guessing. Return ONLY JSON matching the schema — no prose, no markdown.

Market: {market}

Niches and evidence:
{niches_with_evidence_block}

Schema: {"scores": [{"niche": str, "market_demand": int, "competitive_intensity": int,
"margin_signal": int, "trend_momentum": int, "evidence_level": "sufficient"|"insufficient",
"rationale": str}, ...]}
```

### 7.3 Deep pass — reserved for top ~20% of a sweep, one niche per call

```python
class DeepScore(BaseModel):
    margin_min_pct: float
    margin_max_pct: float
    cost_to_v1_min_usd: int
    cost_to_v1_max_usd: int
    complexity: int                 # 1-5
    regulatory_friction: int        # 1-5
    comparables_used: list[str]     # e.g. "Etsy digital planners at $12"
    key_risks: list[str]            # short, plain-English — not a full risk taxonomy
    evidence_level: Literal["sufficient", "insufficient"]
    rationale: str
```

Deep pass gets richer evidence (SERP results, marketplace listing counts,
competitor pricing) than the cheap pass. `key_risks` is deliberately a
short free-text list, not a scored 11-dimension risk model — see Section
13 on why that's deferred.

---

## 8. Data Source Adapters

Common interface — every adapter returns a list of `Evidence` rows or an
explicit unavailable marker, never raises past its own boundary:

```python
class Evidence(TypedDict):
    source_type: str
    metric: str
    value: str
    confidence: float
    source_url: str | None
    raw_data: dict
    retrieved_at: str
    cost_usd: float

class AdapterResult(TypedDict):
    status: Literal["ok", "unavailable"]
    evidence: list[Evidence]
```

All adapters wrap external calls in exponential-backoff retry (3 attempts).
On the 3rd failure, return `{"status": "unavailable", "evidence": []}` —
never let one dead API kill the batch. The cheap/deep prompts must be
written to handle a niche with partial or zero evidence for some sources
(they already are, via `evidence_level`).

| Adapter | Source | Notes |
|---|---|---|
| Reddit | PRAW, free script app | `subreddit.rising(limit=50)`, filter by score/age (velocity) for discovery mode; simple mention counts for sweep mode |
| Google Trends | pytrends (unofficial) | `interest_over_time()` per niche; `rising_queries()` for discovery |
| Meta Ad Library | Free Graph API | `ads_archive` endpoint; 200 req/hr free tier — cache aggressively |
| SERP fallback | Free-tier search API, used sparingly | Fall back to a DeepSeek-assisted estimate flagged `insufficient` when quota runs out — never silently skip the dimension |
| Marketplace | Etsy / Gumroad / Notion galleries / font marketplaces | Prefer official API; skip (don't scrape around) any source whose ToS prohibits it |
| World Bank / HCP | wbdata / manual CSV | Market-size context for Morocco/MENA sweeps only |

`config/market_profiles.yaml` (start with one market, add more as data —
not code):

```yaml
morocco:
  payment_rails: [cod, bank_transfer]
  vat_pct: 20
  logistics_notes: "COD dominant, delivery ~2-5 days major cities"
  import_complexity: medium
```

---

## 9. Budget & Cost Management

```python
class CostGuard:
    def __init__(self, daily_budget_usd: float, total_budget_usd: float):
        ...
    def check_before_call(self, estimated_tokens: int) -> bool:
        """Return False (block the call) if it would exceed budget."""
    def record(self, tokens_used: int, cost_usd: float): ...
```

Track, per sweep and cumulatively: total spend, spend per idea (avg), spend
per source-of-cost (cheap pass vs. deep pass vs. niche expansion), and
remaining budget. Surface this in the Streamlit sidebar (Section 10) — not
a full enterprise cost dashboard, just enough to see the number ticking and
stop before you hit $9.

The cheap-pass batching in 7.2 is the primary lever. Confirm current
DeepSeek per-token pricing before hardcoding rate constants — don't trust
a stale number from a prior conversation.

---

## 10. Web UI (Streamlit)

**Layout:**
- **Sidebar:** sweep trigger (category + market inputs), category/market/
  score-threshold filters, running spend total, remaining budget.
- **Main area:** ranked list as clean cards (default) with a toggle to a
  sortable `st.dataframe` table view.

**Card (default view), per idea:**
- Title, 2–3 sentence plain-English description.
- Badges via `st.metric`: composite score, research-priority tier, margin
  range, cost range, complexity.
- `st.expander("View evidence & reasoning")` — full per-dimension
  breakdown, evidence sources, confidence, DeepSeek rationale.
- "Deep Dive" button — triggers the deep pass for that one idea (Section
  7.3), updates its row, `st.rerun()`.

**Table view:** `st.dataframe` with `column_config` for the same fields,
sortable, with a CSV export button.

**Critical Streamlit gotcha to guard against:** Streamlit reruns the whole
script on every widget interaction. Without guards, clicking anything
during/after a sweep can re-trigger the sweep or re-bill DeepSeek. Required
pattern:

```python
if "sweep_running" not in st.session_state:
    st.session_state.sweep_running = False

if st.button("Run Sweep", disabled=st.session_state.sweep_running):
    st.session_state.sweep_running = True
    # run sweep...
    st.session_state.sweep_running = False

@st.cache_data
def load_ideas(sweep_run_id: int):
    ...  # cached so switching filters doesn't re-hit the DB/LLM
```

---

## 11. Testing Strategy

- **Unit tests per adapter:** mock the external API, assert correct
  `Evidence` shape and correct fallback on failure.
- **Integration test for the sweep engine:** tiny seed list (2–3 niches),
  a stubbed/test DeepSeek endpoint (or a recorded fixture response),
  asserts a full sweep completes and writes expected rows.
- **Cost-guard unit tests:** assert `check_before_call` blocks correctly at
  the budget edge.
- UI testing is manual — not worth automating for a single-user tool.

---

## 12. Deployment

Local only.

```
requirements.txt
.env.example      # DEEPSEEK_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
                   # META_ACCESS_TOKEN, SERP_API_KEY (optional)
run.sh             # sets env, launches `streamlit run app.py`
docs/FRD.md        # this file
```

---

## 13. Explicitly deferred to v2+ (and why)

The Deepseek review raised a lot of genuinely good ideas. Most are correct
for a team building a durable product. They're deferred here — not
rejected — because building them all before v1 works would blow both the
$9 DeepSeek budget and the time you have with a $20/mo Claude Code plan,
and v1 doesn't need them to be useful. Revisit after the sweep engine has
run a few real sweeps and you know which of these you actually miss:

- **Full category-plugin class architecture** (`collect()` /
  `extract_metrics()` / `calculate_margin()` per category as separate
  classes). v1 gets the same flexibility from seed-list configs + prompt
  templates, at a fraction of the code.
- **Full 11-dimension risk taxonomy** (platform risk, supplier risk,
  currency risk, etc., each independently scored). v1 has `key_risks` as
  short free text on the deep pass. Promote to scored dimensions only if
  you find yourself repeatedly wanting to sort/filter by a specific risk
  type.
- **Opportunity cost / time-to-MVP as a scored dimension.** Good idea,
  genuinely — a 2-week idea at 80% margin can beat an 18-month idea at
  95%. Add `time_to_mvp_weeks` to the deep-pass schema once v1 is stable;
  it's a small, additive change to Section 7.3, not a redesign.
- **Versioned scoring models** (`scoring_model_version` beyond a static
  string) — the column already exists in the schema (Section 5) so this
  costs nothing to add later; just isn't load-bearing yet.
- **Feature flags system** for toggling sources on/off. For one user,
  commenting out a line in a config file does the same job.
- **Idea history / longitudinal tracking across sweeps**, **learning loop**
  (tag succeeded/failed after 6 months, calibrate weights), **historical
  comparison engine** (Jan vs. March). All genuinely valuable, all
  meaningless until you have several months of sweep history to learn
  from. The immutable-snapshot design in Section 5 means none of this
  data is lost — you can build this layer later purely by querying old
  `sweep_runs` rows.
- **Full observability stack** (latency, source health dashboards). Useful
  at team scale; for a personal tool, adapter unit tests + the cost
  dashboard cover what you actually need to notice.

---

## 14. Task Breakdown for Claude Code

**Rule:** one task per Claude Code session. Do not let it jump ahead to the
next task "while it's in there." Use this template every time, swapping in
the task block:

```
You are working on Project Meezan. Read docs/FRD.md for full context, but
ONLY implement the task below. Do not modify any files outside its listed
scope. Do not implement future tasks even if you can see how they'd
connect. When done, stop and report exactly what you built, what's stubbed
out, and what the next task needs from you.

[paste one task block from below]
```

| # | Task | Scope (files) | Definition of done |
|---|---|---|---|
| 0 | **Repo scaffolding** — reorganize the existing prototype into the structure this FRD assumes (`adapters/`, `scoring/`, `llm/`, `db/`, `config/`, `app.py`) without changing logic yet. | New folder structure, moved files, `requirements.txt`, `.env.example` | Existing tests (if any) still pass after the move; no behavior changes |
| 1 | **DB schema migration** — create the tables in Section 5, replacing/migrating the prototype's SQLite schema. | `db/schema.sql`, `db/migrate.py` | Running the migration script produces exactly the 4 tables + indexes in Section 5 on a fresh DB |
| 2 | **Evidence adapter interface** — define `Evidence`, `AdapterResult` types and the retry/backoff wrapper from Section 8. | `adapters/base.py` | A dummy adapter using the wrapper demonstrably returns `unavailable` after 3 simulated failures, `ok` otherwise |
| 3 | **Reddit adapter** — refactor/build PRAW adapter to the interface from Task 2. | `adapters/reddit.py` + its unit test | Unit test mocks PRAW, asserts correct `Evidence` shape and fallback behavior |
| 4 | **Google Trends adapter** | `adapters/trends.py` + test | Same bar as Task 3, mocked pytrends |
| 5 | **Meta Ad Library adapter** | `adapters/meta_ads.py` + test | Same bar, respects the 200 req/hr note with caching |
| 6 | **SERP fallback adapter** | `adapters/serp.py` + test | On quota exhaustion, returns evidence with `confidence` explicitly lowered, not a crash |
| 7 | **Marketplace adapters** (Etsy, Gumroad, Notion galleries) | `adapters/marketplace.py` + test | One adapter per source or a shared class — Claude Code's choice — each ToS-checked per Section 8 table |
| 8 | **World Bank/HCP adapter** | `adapters/worldbank.py` + test | Returns market-size evidence for at least the `morocco` profile |
| 9 | **Cost guard + DeepSeek provider** — token tracking, `CostGuard` from Section 9, DeepSeek client wrapper. | `llm/deepseek_provider.py`, `llm/cost_guard.py` + tests | Unit test proves a call is blocked once budget is exceeded |
| 10 | **Cheap-pass prompt + batching + parser** | `llm/cheap_pass.py` (prompt template, Pydantic model, batching logic from 7.2) | Given 10 fake niches, produces 2 batched calls (batch size 8), all outputs validate against `CheapPassResponse` |
| 11 | **Deterministic scoring engine** — composite score + confidence propagation + priority tiering from Section 6. | `scoring/engine.py` + tests | Given fixed dimension scores/weights, output matches hand-computed expected values |
| 12 | **SweepEngine orchestration** — wires seed list + niche expansion (7.1) + adapters (2–8) + cheap pass (10) + scoring (11) + storage (1). | `sweep_engine.py` | Running a sweep on a 3-niche seed list end-to-end writes correct rows to all 4 tables |
| 13 | **Deep-pass prompt + parser + `deep_score()`** | `llm/deep_pass.py` | Given one niche with richer evidence, returns a validated `DeepScore` and updates the idea's denormalized columns |
| 14 | **Streamlit — sweep trigger + spend sidebar** | `app.py` (sidebar section only) | Triggering a sweep from the UI works once, is disabled while running (session_state guard from Section 10), spend total updates |
| 15 | **Streamlit — ranked card view + table toggle + filters** | `app.py` (main area) | Cards show description + badges; table view sortable; category/market/score filters work |
| 16 | **Streamlit — deep dive + evidence expander** | `app.py` (per-card expander + button) | Clicking "Deep Dive" runs Task 13's function on that idea only and updates its card |
| 17 | **CSV export** | `app.py` (export button) | Exports the currently filtered ranked list to CSV |
| 18 | **Discovery engine (Reddit + Trends rising)** — v1.1, only after 0–17 are working end-to-end. | `discovery_engine.py` | Produces a second, separately-labeled set of candidate ideas, reuses the cheap-pass pipeline |
| 19 | **Test pass + README** | `tests/`, `README.md` | All unit/integration tests from Section 11 pass; README covers setup + `.env` + `run.sh` |

Suggested order: 0 → 1 → 2 → (3–8 in any order, they don't depend on each
other) → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 19 → 18 (discovery
last, exactly as the original brief specified).
