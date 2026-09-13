"""Project Meezan — cheap-pass prompt, batching, and parser (docs/FRD.md, Section 7.2, Task 10).

Batches niches (already paired with their gathered evidence — evidence
collection itself is Task 12's job, not this module's) into groups of 8
and asks DeepSeek to rate market_demand, competitive_intensity,
margin_signal, and trend_momentum for every niche in one call. This
batching is the primary cost lever from Section 9: a 50-niche sweep costs
~7 calls instead of 50.

Every response is validated against `CheapPassResponse` before it's
trusted. A parse/validation failure (including the model returning the
wrong set of niches) gets one retry with the error appended to the
prompt; a second failure degrades that whole batch to
`evidence_level="insufficient"` scores instead of raising and killing the
rest of the sweep, per Section 7's global retry-once-then-degrade rule.
"""

import json
from typing import Literal, TypedDict

from pydantic import BaseModel, ValidationError

from adapters.base import Evidence
from llm.deepseek_provider import DeepSeekProvider

CHEAP_PASS_MODEL = "deepseek-v4-flash"
BATCH_SIZE = 8
MAX_RATIONALE_CHARS = 200


class CheapNicheScore(BaseModel):
    niche: str
    market_demand: int
    competitive_intensity: int
    margin_signal: int
    trend_momentum: int
    evidence_level: Literal["sufficient", "insufficient"]
    rationale: str


class CheapPassResponse(BaseModel):
    scores: list[CheapNicheScore]


class NicheWithEvidence(TypedDict):
    niche: str
    evidence: list[Evidence]


def score_niches_cheap(
    niches: list[NicheWithEvidence],
    market: str,
    provider: DeepSeekProvider | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[CheapNicheScore]:
    """Score every niche's cheap-pass dimensions, `batch_size` niches per call.

    `provider` defaults to a fresh `DeepSeekProvider`, always constructed
    with `model=CHEAP_PASS_MODEL` explicitly (never the provider's own
    default) so a change to that default elsewhere doesn't silently
    change which model the cheap pass runs on. Pass an existing provider
    to reuse its token-usage tracking across calls/batches/sweeps.
    """
    provider = provider or DeepSeekProvider(model=CHEAP_PASS_MODEL)
    scores: list[CheapNicheScore] = []
    for start in range(0, len(niches), batch_size):
        batch = niches[start : start + batch_size]
        scores.extend(_score_batch(provider, market, batch))
    return scores


def _score_batch(
    provider: DeepSeekProvider, market: str, batch: list[NicheWithEvidence]
) -> list[CheapNicheScore]:
    parsed, error = _call_and_validate(provider, market, batch)
    if parsed is None:
        parsed, error = _call_and_validate(provider, market, batch, prior_error=error)
    if parsed is None:
        return _degrade(batch, error)
    return parsed.scores


def _call_and_validate(
    provider: DeepSeekProvider,
    market: str,
    batch: list[NicheWithEvidence],
    prior_error: str | None = None,
) -> tuple[CheapPassResponse | None, str | None]:
    prompt = _build_prompt(market, batch, prior_error)
    result = provider.complete(prompt)

    if result["status"] != "ok":
        return None, f"provider call failed: status={result['status']}, message={result['message']}"

    try:
        data = json.loads(result["content"] or "")
    except json.JSONDecodeError as e:
        return None, f"response was not valid JSON: {e}"

    try:
        parsed = CheapPassResponse.model_validate(data)
    except ValidationError as e:
        return None, str(e)

    expected_niches = [n["niche"] for n in batch]
    got_niches = [s.niche for s in parsed.scores]
    if len(got_niches) != len(expected_niches) or set(got_niches) != set(expected_niches):
        return None, (
            f"response covered niches {got_niches!r}, expected exactly {expected_niches!r}"
        )

    return parsed, None


def _degrade(batch: list[NicheWithEvidence], error: str | None) -> list[CheapNicheScore]:
    rationale = f"Cheap-pass parsing failed twice: {error}"[:MAX_RATIONALE_CHARS]
    return [
        CheapNicheScore(
            niche=n["niche"],
            market_demand=1,
            competitive_intensity=1,
            margin_signal=1,
            trend_momentum=1,
            evidence_level="insufficient",
            rationale=rationale,
        )
        for n in batch
    ]


def _build_prompt(market: str, batch: list[NicheWithEvidence], prior_error: str | None) -> str:
    niches_block = _build_niches_block(batch)
    prompt = f"""You are an unbiased business-idea evaluator. For EACH niche below, using
only the evidence provided for that niche, rate market_demand,
competitive_intensity, margin_signal, and trend_momentum on a 1-10 scale.
If evidence for a niche is thin, say so in evidence_level rather than
guessing. Return ONLY JSON matching the schema — no prose, no markdown.

Market: {market}

Niches and evidence:
{niches_block}

Schema: {{"scores": [{{"niche": str, "market_demand": int, "competitive_intensity": int,
"margin_signal": int, "trend_momentum": int, "evidence_level": "sufficient"|"insufficient",
"rationale": str}}, ...]}}"""

    if prior_error:
        prompt += (
            "\n\nYour previous response failed validation with this error:\n"
            f"{prior_error}\n"
            "Fix the response and return ONLY valid JSON matching the schema above, "
            "covering exactly the niches listed."
        )
    return prompt


def _build_niches_block(batch: list[NicheWithEvidence]) -> str:
    lines = []
    for entry in batch:
        lines.append(f"- {entry['niche']}:")
        if not entry["evidence"]:
            lines.append("  - No evidence available.")
            continue
        for ev in entry["evidence"]:
            lines.append(
                f"  - {ev['source_type']}: {ev['metric']} = {ev['value']} "
                f"(confidence {ev['confidence']})"
            )
    return "\n".join(lines)
