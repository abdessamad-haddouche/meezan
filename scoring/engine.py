"""Project Meezan — deterministic scoring engine (docs/FRD.md, Section 6, Task 11).

Composite score, confidence propagation, and research-priority tiering —
all deterministic code, never asked from the LLM (Section 2, rule 1). The
LLM only produces per-dimension judgments; turning those into a ranking is
this module's job, so a sweep's ranking is always reproducible from its
stored sub-scores without re-calling DeepSeek.

The 7 scoring dimensions (Section 6) split into two passes: `market_demand`,
`competitive_intensity`, `margin_signal`, and `trend_momentum` come out of
the cheap pass (Task 10, run on every idea in a sweep). `cost_to_start`,
`complexity`, and `regulatory_friction` only exist after a deep pass (a
later task, run on a small top slice). `compute_composite` works with
whichever of the 7 are present in `scores`, renormalizing weights over just
those, so a cheap-pass-only idea still gets a usable composite — marked
"preliminary" rather than "final" until all 7 dimensions are in.
"""

from pathlib import Path
from typing import Literal, TypedDict

import yaml

DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "config" / "weights.yaml"

ResearchPriority = Literal["research_this_week", "revisit_later", "deprioritize"]


class ScoreValue(TypedDict):
    value: float  # 1-10
    confidence: float  # 0-1


class CompositeResult(TypedDict):
    composite: float  # weighted, same 1-10 scale as the input values
    confidence: float  # 0-1
    status: Literal["preliminary", "final"]


def load_weights_config(path: Path | str = DEFAULT_WEIGHTS_PATH) -> dict:
    """Load config/weights.yaml, returning its `weights` and
    `research_priority_thresholds` top-level dicts."""
    with open(path) as f:
        return yaml.safe_load(f)


def compute_composite(scores: dict[str, ScoreValue], weights: dict[str, float]) -> CompositeResult:
    """Weighted composite score + confidence, renormalized over present dimensions.

    Only dimensions present in both `scores` and `weights` count toward the
    result; their weights are renormalized to sum to 1 so a partial
    (cheap-pass-only) set of dimensions still produces a valid composite on
    the same 1-10 scale, instead of silently under-weighting toward zero.

    `status` is "final" only when every dimension in `weights` has a score
    (i.e. the deep pass has run) and "preliminary" otherwise (Section 6) —
    this is never a claim about score quality, only about pass completeness.

    Raises `ValueError` if `scores` contains a dimension not in `weights`
    (almost always a dimension-name typo/mismatch — e.g. "margin" vs.
    "margin_signal" — since every real dimension is listed in weights.yaml)
    or if no dimension in `scores` matches any in `weights` at all.
    """
    unknown = set(scores) - set(weights)
    if unknown:
        raise ValueError(f"scores contains dimension(s) not in weights: {sorted(unknown)}")

    present = {dim: w for dim, w in weights.items() if dim in scores}
    if not present:
        raise ValueError(
            "compute_composite requires at least one dimension present in both scores and weights"
        )

    weight_total = sum(present.values())
    composite = sum(scores[dim]["value"] * w for dim, w in present.items()) / weight_total
    confidence = sum(scores[dim]["confidence"] * w for dim, w in present.items()) / weight_total
    status: Literal["preliminary", "final"] = (
        "final" if present.keys() == weights.keys() else "preliminary"
    )

    return {"composite": composite, "confidence": confidence, "status": status}


def assign_research_priority(ideas: list[dict], thresholds: dict[str, float]) -> list[dict]:
    """Label each idea's research priority by its composite score's percentile rank.

    Never a verdict (Section 2, rule 6) — only these three labels, assigned
    by where an idea's `composite` score (each idea dict must carry a
    `composite` key, e.g. the output of `compute_composite`) falls relative
    to the *other* ideas in this same list. Thresholds are
    `research_this_week_percentile` / `revisit_later_percentile` from
    config/weights.yaml's `research_priority_thresholds`.

    An idea's percentile rank is the fraction of ideas in the list
    (itself included) scoring at or below its own composite, so ideas
    tied on composite always receive the same label regardless of input
    order. Returns a new list of dicts — input dicts are not mutated —
    each a shallow copy of the input idea with `research_priority` added.
    """
    if not ideas:
        return []

    composites = [idea["composite"] for idea in ideas]
    n = len(composites)
    research_this_week_pct = thresholds["research_this_week_percentile"]
    revisit_later_pct = thresholds["revisit_later_percentile"]

    def _label(composite: float) -> ResearchPriority:
        percentile = sum(1 for c in composites if c <= composite) / n
        if percentile >= research_this_week_pct:
            return "research_this_week"
        if percentile >= revisit_later_pct:
            return "revisit_later"
        return "deprioritize"

    return [{**idea, "research_priority": _label(idea["composite"])} for idea in ideas]
