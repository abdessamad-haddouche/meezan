"""Project Meezan — deep-pass prompt, parser, and deep_score() (docs/FRD.md, Section 7.3, Task 13).

The deep pass is reserved for a single idea at a time — called on-demand
(a "Deep Dive"), not batched like the cheap pass (Task 10). Given an
`idea_id` already written by a prior cheap pass (Task 12's `run_sweep`),
`deep_score()`:

1. Re-gathers evidence for that idea's niche via the same Section 8
   adapters and DISABLED_ADAPTERS handling Task 12's sweep uses. (This
   module intentionally duplicates that small adapter-loop + env-parsing
   logic from sweep_engine.py rather than importing it — llm/cheap_pass.py
   sets the precedent of an `llm/` module having no dependency on the
   orchestrator, and importing from sweep_engine.py here would invert that
   and risk a circular import the moment sweep_engine.py needs to call
   into a deep pass itself. `marketplace.py` has nothing beyond
   `collect_etsy` today — Gumroad/Notion are skipped, no official
   marketplace-search API for either — so there's nothing extra to add
   for comparables.)
2. Calls DeepSeek on `DEEP_PASS_MODEL` ("deepseek-v4-pro" — deliberately
   not the cheap pass's v4-flash) with the deep-pass prompt, requesting
   strict JSON matching `DeepScore` (Section 7.3).
3. Validates against `DeepScore`. On a validation/parse failure, retries
   once with the error appended to the prompt; on a second failure,
   degrades to a maximally-uncertain `DeepScore` (`evidence_level =
   "insufficient"`) rather than crashing — the same retry-once-then-
   degrade pattern as the cheap pass (Task 10).
4. Writes `evidence` rows for everything freshly gathered, `scores` rows
   (`pass_type="deep"`) for the 3 deep-only dimensions, and updates the
   idea's denormalized margin/cost/complexity columns + `has_deep_pass`.
5. Recomputes the idea's composite via Task 11's `compute_composite`, now
   that all 7 dimensions exist, flipping `status` from "preliminary" to
   "final" — but does NOT touch `research_priority`, since re-tiering is a
   whole-sweep operation (Task 11's `assign_research_priority`), out of
   scope for one idea's deep dive.

DeepScore's `complexity`/`regulatory_friction` are raw 1-5 "how much"
ratings (higher = worse), and `cost_to_v1_min/max_usd` is a dollar range —
neither is already on the "higher = more favorable" 1-10 `scores.value`
scale every other dimension uses (Task 10/11). `_invert_1_to_5` and
`_cost_to_score` are this module's own deterministic, documented
conversion for those three dimensions into that scale — the FRD doesn't
specify one, so these constants are a reasonable first cut to revisit
once real deep-pass data exists.
"""

import json
import os
import sqlite3
from typing import Callable, Literal, TypedDict

from pydantic import BaseModel, ValidationError

from adapters import marketplace, meta_ads, reddit, serp, trends, worldbank
from adapters.base import AdapterResult, Evidence
from llm.deepseek_provider import DeepSeekProvider
from scoring.engine import CompositeResult, ScoreValue, compute_composite, load_weights_config

DEEP_PASS_MODEL = "deepseek-v4-pro"
MAX_RATIONALE_CHARS = 200

# Mirrors sweep_engine.py's cheap-pass confidence mapping (Task 12): the
# deep pass likewise gives one evidence_level for the whole niche, not a
# numeric confidence per dimension.
SUFFICIENT_CONFIDENCE = 0.7
INSUFFICIENT_CONFIDENCE = 0.3

DEEP_PASS_DIMENSIONS = ("cost_to_start", "complexity", "regulatory_friction")

# complexity/regulatory_friction: DeepScore's raw scale is 1 (best) - 5
# (worst). Invert onto 1-10 where higher is still more favorable:
# 1->10, 2->8, 3->6, 4->4, 5->2.
def _invert_1_to_5(rating: int) -> float:
    return (6 - rating) * 2.0


# cost_to_start: linear map from a $ midpoint onto 1-10, cheaper = higher
# score. $0 -> 10, $10k+ -> 1. This ceiling is this project's own choice
# (not FRD-specified) for a single-user MVP; revisit if real sweeps
# regularly produce ideas above it.
COST_SCORE_FLOOR_USD = 0.0
COST_SCORE_CEILING_USD = 10_000.0


def _cost_to_score(cost_usd: float) -> float:
    if cost_usd <= COST_SCORE_FLOOR_USD:
        return 10.0
    if cost_usd >= COST_SCORE_CEILING_USD:
        return 1.0
    fraction = (cost_usd - COST_SCORE_FLOOR_USD) / (COST_SCORE_CEILING_USD - COST_SCORE_FLOOR_USD)
    return 10.0 - fraction * 9.0


class DeepScore(BaseModel):
    margin_min_pct: float
    margin_max_pct: float
    cost_to_v1_min_usd: int
    cost_to_v1_max_usd: int
    complexity: int  # 1-5
    regulatory_friction: int  # 1-5
    comparables_used: list[str]
    key_risks: list[str]
    evidence_level: Literal["sufficient", "insufficient"]
    rationale: str


class DeepScoreResult(TypedDict):
    deep_score: DeepScore
    composite: CompositeResult


def _log(message: str) -> None:
    print(f"[deep_pass] {message}", flush=True)


def load_disabled_adapters() -> frozenset[str]:
    """Parse the DISABLED_ADAPTERS env var (comma-separated adapter labels,
    e.g. "meta_ads,etsy") into a set. Empty/unset -> nothing disabled.
    Mirrors sweep_engine.load_disabled_adapters (Task 12) exactly, so a
    deep pass skips the same sources a sweep does."""
    raw = os.environ.get("DISABLED_ADAPTERS", "")
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


_EVIDENCE_ADAPTERS: tuple[tuple[str, Callable[[str, str], AdapterResult]], ...] = (
    ("reddit", lambda niche, market: reddit.collect(niche)),
    ("google_trends", lambda niche, market: trends.collect(niche)),
    ("meta_ads", lambda niche, market: meta_ads.collect(niche)),
    ("serp", lambda niche, market: serp.collect(niche)),
    ("etsy", lambda niche, market: marketplace.collect_etsy(niche)),
    ("worldbank", lambda niche, market: worldbank.collect(market)),
)


def _gather_evidence(niche: str, market: str, disabled_adapters: frozenset[str]) -> list[Evidence]:
    """Same adapter set + DISABLED_ADAPTERS handling as
    sweep_engine._gather_evidence_for_niche (Task 12), duplicated here
    rather than imported — see the module docstring."""
    evidence: list[Evidence] = []
    for label, call in _EVIDENCE_ADAPTERS:
        if label in disabled_adapters:
            _log(f"{niche!r}: skipping {label} (disabled via DISABLED_ADAPTERS)")
            continue
        _log(f"{niche!r}: collecting {label} evidence")
        result = call(niche, market)
        _log(f"{niche!r}: {label} -> {result['status']} ({len(result['evidence'])} evidence row(s))")
        evidence.extend(result["evidence"])
    return evidence


def _build_evidence_block(evidence: list[Evidence]) -> str:
    if not evidence:
        return "No evidence available."
    return "\n".join(
        f"- {ev['source_type']}: {ev['metric']} = {ev['value']} (confidence {ev['confidence']})"
        for ev in evidence
    )


def _build_deep_pass_prompt(
    niche: str, market: str, evidence: list[Evidence], prior_error: str | None
) -> str:
    prompt = f"""You are an unbiased business-idea evaluator performing a DEEP evaluation
of ONE niche, using richer evidence than a quick pass. Using only the
evidence provided, estimate:
- margin_min_pct / margin_max_pct: achievable gross margin % range
- cost_to_v1_min_usd / cost_to_v1_max_usd: $ range to reach a sellable v1
- complexity: 1 (very simple) to 5 (very complex) to build
- regulatory_friction: 1 (minimal) to 5 (severe) — payment rails, customs, COD, licensing
- comparables_used: specific comparable products/prices this is grounded in (e.g. "Etsy digital planners at $12")
- key_risks: short, plain-English list of the main risks (not a full taxonomy)
If evidence is thin, say so in evidence_level rather than guessing. Return
ONLY JSON matching the schema — no prose, no markdown.

Niche: {niche}
Market: {market}

Evidence:
{_build_evidence_block(evidence)}

Schema: {{"margin_min_pct": float, "margin_max_pct": float, "cost_to_v1_min_usd": int,
"cost_to_v1_max_usd": int, "complexity": int, "regulatory_friction": int,
"comparables_used": [str, ...], "key_risks": [str, ...],
"evidence_level": "sufficient"|"insufficient", "rationale": str}}"""

    if prior_error:
        prompt += (
            "\n\nYour previous response failed validation with this error:\n"
            f"{prior_error}\n"
            "Fix the response and return ONLY valid JSON matching the schema above."
        )
    return prompt


def _call_deep_pass(
    provider: DeepSeekProvider,
    niche: str,
    market: str,
    evidence: list[Evidence],
    prior_error: str | None = None,
) -> tuple[DeepScore | None, str | None]:
    prompt = _build_deep_pass_prompt(niche, market, evidence, prior_error)
    result = provider.complete(prompt)

    if result["status"] != "ok":
        return None, f"provider call failed: status={result['status']}, message={result['message']}"

    try:
        data = json.loads(result["content"] or "")
    except json.JSONDecodeError as e:
        return None, f"response was not valid JSON: {e}"

    try:
        parsed = DeepScore.model_validate(data)
    except ValidationError as e:
        return None, str(e)

    return parsed, None


def _degrade(error: str | None) -> DeepScore:
    rationale = f"Deep-pass parsing failed twice: {error}"[:MAX_RATIONALE_CHARS]
    return DeepScore(
        margin_min_pct=0.0,
        margin_max_pct=100.0,
        cost_to_v1_min_usd=0,
        cost_to_v1_max_usd=int(COST_SCORE_CEILING_USD * 10),
        complexity=3,
        regulatory_friction=3,
        comparables_used=[],
        key_risks=["deep-pass evaluation failed twice; treat this score as unreliable"],
        evidence_level="insufficient",
        rationale=rationale,
    )


def _score_deep(provider: DeepSeekProvider, niche: str, market: str, evidence: list[Evidence]) -> DeepScore:
    parsed, error = _call_deep_pass(provider, niche, market, evidence)
    if parsed is None:
        _log(f"{niche!r}: first deep-pass response failed validation ({error}); retrying once")
        parsed, error = _call_deep_pass(provider, niche, market, evidence, prior_error=error)
    if parsed is None:
        _log(f"{niche!r}: deep pass failed twice ({error}); degrading to insufficient")
        return _degrade(error)
    return parsed


def _deep_score_confidence(deep: DeepScore) -> float:
    return SUFFICIENT_CONFIDENCE if deep.evidence_level == "sufficient" else INSUFFICIENT_CONFIDENCE


def _deep_score_to_score_values(deep: DeepScore) -> dict[str, ScoreValue]:
    confidence = _deep_score_confidence(deep)
    midpoint_cost = (deep.cost_to_v1_min_usd + deep.cost_to_v1_max_usd) / 2
    return {
        "cost_to_start": {"value": _cost_to_score(midpoint_cost), "confidence": confidence},
        "complexity": {"value": _invert_1_to_5(deep.complexity), "confidence": confidence},
        "regulatory_friction": {"value": _invert_1_to_5(deep.regulatory_friction), "confidence": confidence},
    }


def _fetch_idea(conn: sqlite3.Connection, idea_id: int) -> dict:
    row = conn.execute("SELECT name, market, category FROM ideas WHERE id = ?", (idea_id,)).fetchone()
    if row is None:
        raise ValueError(f"no idea with id={idea_id!r}")
    name, market, category = row
    return {"name": name, "market": market, "category": category}


def _fetch_cheap_pass_scores(conn: sqlite3.Connection, idea_id: int) -> dict[str, ScoreValue]:
    rows = conn.execute(
        "SELECT dimension, value, confidence FROM scores WHERE idea_id = ? AND pass_type = 'cheap'",
        (idea_id,),
    ).fetchall()
    return {dimension: {"value": value, "confidence": confidence} for dimension, value, confidence in rows}


def _insert_deep_scores(conn: sqlite3.Connection, idea_id: int, deep: DeepScore) -> None:
    score_values = _deep_score_to_score_values(deep)
    comparable_products_json = json.dumps(deep.comparables_used)
    for dimension in DEEP_PASS_DIMENSIONS:
        sv = score_values[dimension]
        conn.execute(
            """
            INSERT INTO scores (
                idea_id, dimension, value, confidence, evidence_level,
                rationale, comparable_products, pass_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'deep')
            """,
            (idea_id, dimension, sv["value"], sv["confidence"], deep.evidence_level, deep.rationale, comparable_products_json),
        )


def _insert_evidence(conn: sqlite3.Connection, idea_id: int, evidence: list[Evidence]) -> None:
    for ev in evidence:
        conn.execute(
            """
            INSERT INTO evidence (
                idea_id, source_type, metric, value, confidence,
                source_url, raw_data, retrieved_at, cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idea_id,
                ev["source_type"],
                ev["metric"],
                ev["value"],
                ev["confidence"],
                ev["source_url"],
                json.dumps(ev["raw_data"]),
                ev["retrieved_at"],
                ev["cost_usd"],
            ),
        )


def _update_idea_deep_pass_columns(conn: sqlite3.Connection, idea_id: int, deep: DeepScore) -> None:
    conn.execute(
        """
        UPDATE ideas
        SET margin_min_pct = ?, margin_max_pct = ?, cost_min_usd = ?, cost_max_usd = ?,
            complexity = ?, has_deep_pass = 1
        WHERE id = ?
        """,
        (deep.margin_min_pct, deep.margin_max_pct, deep.cost_to_v1_min_usd, deep.cost_to_v1_max_usd, deep.complexity, idea_id),
    )


def _update_idea_composite(conn: sqlite3.Connection, idea_id: int, composite: CompositeResult) -> None:
    conn.execute(
        "UPDATE ideas SET composite_score = ?, composite_confidence = ? WHERE id = ?",
        (composite["composite"], composite["confidence"], idea_id),
    )


def deep_score(
    idea_id: int,
    conn: sqlite3.Connection,
    provider: DeepSeekProvider | None = None,
    weights_config: dict | None = None,
    disabled_adapters: frozenset[str] | None = None,
) -> DeepScoreResult:
    """Run the deep pass (Section 7.3) for one idea and write its results.

    `idea_id` must already have cheap-pass `scores` rows (written by a
    prior `sweep_engine.run_sweep`) — its niche/market are read straight
    from the `ideas` row. `conn` must already point at a migrated
    database. `provider` defaults to a fresh DeepSeekProvider on
    DEEP_PASS_MODEL ("deepseek-v4-pro"); `weights_config` defaults to
    loading config/weights.yaml; `disabled_adapters` defaults to parsing
    DISABLED_ADAPTERS.

    Writes new `evidence` rows, 3 new `scores` rows (`pass_type="deep"`)
    for cost_to_start/complexity/regulatory_friction, and updates the
    idea's denormalized margin/cost/complexity/has_deep_pass columns.
    Recomputes and writes `composite_score`/`composite_confidence` (now
    "final" — all 7 dimensions exist) via Task 11's `compute_composite`,
    but deliberately does NOT touch `research_priority`: re-tiering needs
    every idea in the sweep, not just this one.

    Returns both the validated (or degraded) `DeepScore` and the
    recomputed `CompositeResult` so a caller/test can see the "final"
    status directly — the DB schema has no separate status column for it.
    """
    provider = provider or DeepSeekProvider(model=DEEP_PASS_MODEL)
    weights_config = weights_config or load_weights_config()
    disabled_adapters = disabled_adapters if disabled_adapters is not None else load_disabled_adapters()

    idea = _fetch_idea(conn, idea_id)
    niche, market = idea["name"], idea["market"]
    _log(f"idea_id={idea_id} ({niche!r}): starting deep pass")

    evidence = _gather_evidence(niche, market, disabled_adapters)

    _log(f"{niche!r}: calling DeepSeek ({DEEP_PASS_MODEL})")
    deep = _score_deep(provider, niche, market, evidence)

    _insert_evidence(conn, idea_id, evidence)
    _insert_deep_scores(conn, idea_id, deep)
    _update_idea_deep_pass_columns(conn, idea_id, deep)

    all_scores = _fetch_cheap_pass_scores(conn, idea_id)
    all_scores.update(_deep_score_to_score_values(deep))
    composite = compute_composite(all_scores, weights_config["weights"])
    _update_idea_composite(conn, idea_id, composite)

    conn.commit()
    _log(f"idea_id={idea_id} ({niche!r}): deep pass complete (status={composite['status']})")

    return {"deep_score": deep, "composite": composite}
