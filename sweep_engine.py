"""Project Meezan — SweepEngine orchestration (docs/FRD.md, Section 4, Task 12).

Wires together everything built in Tasks 1-11 into one real end-to-end
sweep:

    niche expansion (7.1) -> per-niche evidence collection (Section 8) ->
    batched cheap-pass scoring (Task 10) -> per-niche composite score +
    confidence (Task 11) -> research-priority tiering across the whole
    sweep (Task 11) -> one immutable sweep_runs/ideas/scores/evidence
    snapshot (Section 5).

An adapter returning "unavailable" for a niche never stops the sweep — it
just means less evidence feeds that niche's cheap-pass evidence_level, per
Section 8's own contract ("never let one dead API kill the batch").

Niche expansion (Section 7.1) wasn't built as its own task, so its prompt,
`NicheExpansion` schema, and retry-once-then-degrade handling live here.
On a second validation failure it degrades to the seed list alone rather
than crashing the sweep, matching Section 7's global rule for every
DeepSeek call in this pipeline.

Cheap-pass scores don't carry a numeric per-dimension confidence (Section
7.2's `CheapNicheScore` has one `evidence_level` for all 4 cheap-pass
dimensions of a niche), so `evidence_level` is mapped to a single numeric
confidence applied uniformly to that niche's 4 dimensions — see
`SUFFICIENT_CONFIDENCE` / `INSUFFICIENT_CONFIDENCE` below.

A real sweep can run for minutes (niche expansion + one evidence-adapter
call per niche + a batched cheap pass, all network/LLM calls), so `_log`
prints plain progress lines — which niche and which adapter/LLM call is
currently running — rather than leaving a terminal silent with no way to
tell a slow run from a hung one.

`DISABLED_ADAPTERS` (comma-separated env var, e.g. "meta_ads,etsy"), read
once per sweep in `run_sweep`, skips a named adapter entirely for every
niche in that sweep — never called, never retried, contributing no
evidence — for a source you don't have working credentials for yet.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, TypedDict

import yaml
from pydantic import BaseModel, ValidationError

from adapters import marketplace, meta_ads, reddit, serp, trends, worldbank
from adapters.base import AdapterResult, Evidence
from llm.cheap_pass import CHEAP_PASS_MODEL, CheapNicheScore, score_niches_cheap
from llm.deepseek_provider import DeepSeekProvider
from scoring.engine import CompositeResult, ScoreValue, assign_research_priority, compute_composite, load_weights_config

SEED_LISTS_DIR = Path(__file__).parent / "config" / "seed_lists"

SUFFICIENT_CONFIDENCE = 0.7
INSUFFICIENT_CONFIDENCE = 0.3

CHEAP_PASS_DIMENSIONS = ("market_demand", "competitive_intensity", "margin_signal", "trend_momentum")


def _log(message: str) -> None:
    print(f"[sweep] {message}", flush=True)


class NicheExpansion(BaseModel):
    suggested_niches: list[str]


class ExpandedNiche(TypedDict):
    niche: str
    origin: Literal["seed", "suggested"]


def load_seed_list(category: str, base_dir: Path | str = SEED_LISTS_DIR) -> list[str]:
    """Load config/seed_lists/<category>.yaml's `seed_niches` list."""
    path = Path(base_dir) / f"{category}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["seed_niches"]


def _build_expansion_prompt(
    category: str, seeds: list[str], market: str, prior_error: str | None
) -> str:
    seeds_joined = ", ".join(f'"{s}"' for s in seeds)
    prompt = f"""Given the parent category "{category}" and seed sub-niches [{seeds_joined}],
propose 10-20 additional specific sub-niches worth evaluating in the
{market} market. Return ONLY JSON matching this shape:
{{"suggested_niches": ["...", "..."]}}"""

    if prior_error:
        prompt += (
            "\n\nYour previous response failed validation with this error:\n"
            f"{prior_error}\n"
            "Fix the response and return ONLY valid JSON matching the schema above."
        )
    return prompt


def _call_expansion(
    provider: DeepSeekProvider,
    category: str,
    seeds: list[str],
    market: str,
    prior_error: str | None = None,
) -> tuple[list[str] | None, str | None]:
    prompt = _build_expansion_prompt(category, seeds, market, prior_error)
    result = provider.complete(prompt)

    if result["status"] != "ok":
        return None, f"provider call failed: status={result['status']}, message={result['message']}"

    try:
        data = json.loads(result["content"] or "")
    except json.JSONDecodeError as e:
        return None, f"response was not valid JSON: {e}"

    try:
        parsed = NicheExpansion.model_validate(data)
    except ValidationError as e:
        return None, str(e)

    return parsed.suggested_niches, None


def expand_niches(
    category: str, seeds: list[str], market: str, provider: DeepSeekProvider
) -> list[ExpandedNiche]:
    """Combine seed niches with DeepSeek-suggested ones (Section 7.1), deduplicated.

    On a validation/parsing failure, retries once with the error appended
    to the prompt; on a second failure, degrades to seeds-only rather than
    crashing the sweep (Section 7's global retry-once-then-degrade rule).
    Matching is case/whitespace-insensitive so a suggested niche that just
    repeats a seed is dropped, not double-counted.
    """
    _log(f"niche expansion: requesting suggestions for category={category!r} market={market!r} ({len(seeds)} seed niches)")
    suggested, error = _call_expansion(provider, category, seeds, market)
    if suggested is None:
        _log(f"niche expansion: first response failed validation ({error}); retrying once")
        suggested, error = _call_expansion(provider, category, seeds, market, prior_error=error)
    if suggested is None:
        _log(f"niche expansion: failed twice ({error}); falling back to seed niches only")
        suggested = []
    else:
        _log(f"niche expansion: got {len(suggested)} suggested niches")

    seed_keys = {s.strip().lower() for s in seeds}
    seen: set[str] = set()
    expanded: list[ExpandedNiche] = []
    for niche in seeds + suggested:
        key = niche.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        expanded.append({"niche": niche, "origin": "seed" if key in seed_keys else "suggested"})
    _log(f"niche expansion: {len(expanded)} niches total after dedup ({len(seeds)} seed + {len(expanded) - len(seeds)} suggested)")
    return expanded


_EVIDENCE_ADAPTERS: tuple[tuple[str, Callable[[str, str], AdapterResult]], ...] = (
    ("reddit", lambda niche, market: reddit.collect(niche)),
    ("google_trends", lambda niche, market: trends.collect(niche)),
    ("meta_ads", lambda niche, market: meta_ads.collect(niche)),
    ("serp", lambda niche, market: serp.collect(niche)),
    ("etsy", lambda niche, market: marketplace.collect_etsy(niche)),
    ("worldbank", lambda niche, market: worldbank.collect(market)),
)


def load_disabled_adapters() -> frozenset[str]:
    """Parse the DISABLED_ADAPTERS env var (comma-separated adapter labels,
    e.g. "meta_ads,etsy") into a set. Empty/unset -> nothing disabled."""
    raw = os.environ.get("DISABLED_ADAPTERS", "")
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


def _gather_evidence_for_niche(
    niche: str, market: str, disabled_adapters: frozenset[str] = frozenset()
) -> list[Evidence]:
    """Call every Section 8 adapter for `niche`, except ones named in
    `disabled_adapters` (skipped entirely — not called, not retried, as if
    each always returned "unavailable"). A genuinely "unavailable" result
    from a called adapter just contributes no evidence, per
    adapters/base.py's contract of never raising past its own boundary."""
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


def _cheap_score_confidence(cheap_score: CheapNicheScore) -> float:
    return SUFFICIENT_CONFIDENCE if cheap_score.evidence_level == "sufficient" else INSUFFICIENT_CONFIDENCE


def _cheap_score_to_score_values(cheap_score: CheapNicheScore) -> dict[str, ScoreValue]:
    confidence = _cheap_score_confidence(cheap_score)
    return {
        "market_demand": {"value": cheap_score.market_demand, "confidence": confidence},
        "competitive_intensity": {"value": cheap_score.competitive_intensity, "confidence": confidence},
        "margin_signal": {"value": cheap_score.margin_signal, "confidence": confidence},
        "trend_momentum": {"value": cheap_score.trend_momentum, "confidence": confidence},
    }


def _insert_sweep_run(conn: sqlite3.Connection, category: str, market: str, triggered_at: str) -> int:
    cursor = conn.execute(
        "INSERT INTO sweep_runs (category, market, triggered_at, status) VALUES (?, ?, ?, ?)",
        (category, market, triggered_at, "running"),
    )
    return cursor.lastrowid


def _update_sweep_status(conn: sqlite3.Connection, sweep_run_id: int, status: str) -> None:
    conn.execute("UPDATE sweep_runs SET status = ? WHERE id = ?", (status, sweep_run_id))


def _insert_idea(
    conn: sqlite3.Connection,
    sweep_run_id: int,
    expanded_niche: ExpandedNiche,
    category: str,
    market: str,
    composite_result: CompositeResult,
    research_priority: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO ideas (
            sweep_run_id, name, category, market, seed_or_suggested,
            composite_score, composite_confidence, research_priority,
            has_deep_pass, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            sweep_run_id,
            expanded_niche["niche"],
            category,
            market,
            expanded_niche["origin"],
            composite_result["composite"],
            composite_result["confidence"],
            research_priority,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    return cursor.lastrowid


def _insert_scores(conn: sqlite3.Connection, idea_id: int, cheap_score: CheapNicheScore) -> None:
    confidence = _cheap_score_confidence(cheap_score)
    for dimension in CHEAP_PASS_DIMENSIONS:
        conn.execute(
            """
            INSERT INTO scores (idea_id, dimension, value, confidence, evidence_level, rationale, pass_type)
            VALUES (?, ?, ?, ?, ?, ?, 'cheap')
            """,
            (
                idea_id,
                dimension,
                getattr(cheap_score, dimension),
                confidence,
                cheap_score.evidence_level,
                cheap_score.rationale,
            ),
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


def run_sweep(
    conn: sqlite3.Connection,
    category: str,
    market: str,
    seed_niches: list[str],
    provider: DeepSeekProvider | None = None,
    weights_config: dict | None = None,
) -> int:
    """Run one full sweep end-to-end and write its immutable snapshot (Section 5).

    `conn` must already point at a migrated database (see db/migrate.py) —
    this function only ever INSERTs new rows plus the one additive
    `sweep_runs.status` UPDATE, never touching prior sweeps. `provider`
    defaults to a fresh DeepSeekProvider on CHEAP_PASS_MODEL, shared for
    both niche expansion and the cheap pass; `weights_config` defaults to
    loading config/weights.yaml. Returns the new `sweep_runs.id`.
    """
    provider = provider or DeepSeekProvider(model=CHEAP_PASS_MODEL)
    weights_config = weights_config or load_weights_config()
    provider.start_sweep()
    disabled_adapters = load_disabled_adapters()

    sweep_run_id = _insert_sweep_run(conn, category, market, datetime.now(timezone.utc).isoformat())
    conn.commit()
    _log(f"sweep_run_id={sweep_run_id}: starting (category={category!r}, market={market!r}, {len(seed_niches)} seed niches)")
    if disabled_adapters:
        _log(f"sweep_run_id={sweep_run_id}: adapters disabled via DISABLED_ADAPTERS: {sorted(disabled_adapters)}")

    try:
        expanded = expand_niches(category, seed_niches, market, provider)

        evidence_by_niche = {
            en["niche"]: _gather_evidence_for_niche(en["niche"], market, disabled_adapters) for en in expanded
        }

        niches_with_evidence = [
            {"niche": en["niche"], "evidence": evidence_by_niche[en["niche"]]} for en in expanded
        ]
        _log(f"cheap pass: scoring {len(niches_with_evidence)} niches")
        cheap_scores = score_niches_cheap(niches_with_evidence, market=market, provider=provider)
        _log(f"cheap pass: {len(cheap_scores)} niches scored")
        cheap_scores_by_niche = {s.niche: s for s in cheap_scores}

        composites_by_niche = {
            en["niche"]: compute_composite(
                _cheap_score_to_score_values(cheap_scores_by_niche[en["niche"]]), weights_config["weights"]
            )
            for en in expanded
        }

        priority_input = [
            {"niche": niche, "composite": result["composite"]} for niche, result in composites_by_niche.items()
        ]
        prioritized = assign_research_priority(priority_input, weights_config["research_priority_thresholds"])
        priority_by_niche = {p["niche"]: p["research_priority"] for p in prioritized}

        for en in expanded:
            niche = en["niche"]
            composite_result = composites_by_niche[niche]
            _log(
                f"{niche!r}: writing rows (composite={composite_result['composite']:.2f}, "
                f"status={composite_result['status']}, priority={priority_by_niche[niche]})"
            )
            idea_id = _insert_idea(conn, sweep_run_id, en, category, market, composite_result, priority_by_niche[niche])
            _insert_scores(conn, idea_id, cheap_scores_by_niche[niche])
            _insert_evidence(conn, idea_id, evidence_by_niche[niche])

        _update_sweep_status(conn, sweep_run_id, "completed")
        conn.commit()
        _log(f"sweep_run_id={sweep_run_id}: completed ({len(expanded)} ideas)")
    except Exception as e:
        _update_sweep_status(conn, sweep_run_id, "failed")
        conn.commit()
        _log(f"sweep_run_id={sweep_run_id}: FAILED ({type(e).__name__}: {e})")
        raise

    return sweep_run_id
