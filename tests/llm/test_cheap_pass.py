"""Unit tests for llm/cheap_pass.py (docs/FRD.md, Section 7.2 / Task 10)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from llm.cheap_pass import CheapPassResponse, score_niches_cheap
from llm.deepseek_provider import DeepSeekProvider


def _fake_response(content: str, total_tokens: int = 10):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def _fake_evidence(source_type: str = "reddit", metric: str = "mentions", value: str = "12"):
    return {
        "source_type": source_type,
        "metric": metric,
        "value": value,
        "confidence": 0.5,
        "source_url": None,
        "raw_data": {},
        "retrieved_at": "2026-09-13T00:00:00Z",
        "cost_usd": 0.0,
    }


def _fake_niches(n: int):
    return [{"niche": f"niche-{i}", "evidence": [_fake_evidence()]} for i in range(n)]


def _valid_payload(niche_names: list[str]) -> str:
    return json.dumps(
        {
            "scores": [
                {
                    "niche": name,
                    "market_demand": 7,
                    "competitive_intensity": 4,
                    "margin_signal": 6,
                    "trend_momentum": 5,
                    "evidence_level": "sufficient",
                    "rationale": "Solid signal across sources.",
                }
                for name in niche_names
            ]
        }
    )


def test_10_niches_produce_exactly_2_batched_calls_all_validating():
    niches = _fake_niches(10)
    batch1_names = [n["niche"] for n in niches[:8]]
    batch2_names = [n["niche"] for n in niches[8:]]
    create_fn = MagicMock(
        side_effect=[
            _fake_response(_valid_payload(batch1_names)),
            _fake_response(_valid_payload(batch2_names)),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    scores = score_niches_cheap(niches, market="morocco", provider=provider)

    assert create_fn.call_count == 2
    assert [call.kwargs["model"] for call in create_fn.call_args_list] == [
        provider.model,
        provider.model,
    ]
    assert len(scores) == 10
    # Every returned score must validate against CheapPassResponse.
    CheapPassResponse(scores=scores)
    assert [s.niche for s in scores] == [n["niche"] for n in niches]
    assert all(s.evidence_level == "sufficient" for s in scores)


def test_default_provider_is_built_with_deepseek_v4_flash_explicitly(monkeypatch):
    import llm.cheap_pass as cheap_pass_module

    captured_kwargs = {}
    create_fn = MagicMock(return_value=_fake_response(_valid_payload(["niche-0"])))

    class _SpyProvider(DeepSeekProvider):
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            kwargs["create_fn"] = create_fn
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cheap_pass_module, "DeepSeekProvider", _SpyProvider)

    cheap_pass_module.score_niches_cheap(_fake_niches(1), market="morocco")

    assert captured_kwargs["model"] == "deepseek-v4-flash"


def test_retries_once_on_malformed_first_response_then_succeeds():
    niches = _fake_niches(3)
    names = [n["niche"] for n in niches]
    malformed = "this is not json"
    create_fn = MagicMock(
        side_effect=[
            _fake_response(malformed),
            _fake_response(_valid_payload(names)),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    scores = score_niches_cheap(niches, market="morocco", provider=provider)

    assert create_fn.call_count == 2
    second_call_messages = create_fn.call_args_list[1].kwargs["messages"]
    second_prompt = second_call_messages[0]["content"]
    assert "failed validation" in second_prompt or "not valid JSON" in second_prompt
    assert len(scores) == 3
    CheapPassResponse(scores=scores)
    assert all(s.evidence_level == "sufficient" for s in scores)


def test_degrades_to_insufficient_after_second_malformed_response():
    niches = _fake_niches(4)
    names = [n["niche"] for n in niches]
    create_fn = MagicMock(
        side_effect=[
            _fake_response("not json at all"),
            _fake_response("still not json"),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    scores = score_niches_cheap(niches, market="morocco", provider=provider)

    assert create_fn.call_count == 2
    assert len(scores) == 4
    CheapPassResponse(scores=scores)
    assert [s.niche for s in scores] == names
    assert all(s.evidence_level == "insufficient" for s in scores)
    assert all("parsing failed twice" in s.rationale for s in scores)


def test_degrades_when_second_response_is_missing_a_niche():
    niches = _fake_niches(2)
    names = [n["niche"] for n in niches]
    create_fn = MagicMock(
        side_effect=[
            _fake_response("garbage"),
            _fake_response(_valid_payload(names[:1])),  # drops one niche
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    scores = score_niches_cheap(niches, market="morocco", provider=provider)

    assert create_fn.call_count == 2
    assert len(scores) == 2
    assert [s.niche for s in scores] == names
    assert all(s.evidence_level == "insufficient" for s in scores)


def test_batches_dont_crash_the_whole_sweep_when_one_batch_degrades():
    niches = _fake_niches(9)  # batch of 8 + batch of 1
    batch1_names = [n["niche"] for n in niches[:8]]
    batch2_names = [n["niche"] for n in niches[8:]]
    create_fn = MagicMock(
        side_effect=[
            _fake_response("broken"),
            _fake_response("still broken"),
            _fake_response(_valid_payload(batch2_names)),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    scores = score_niches_cheap(niches, market="morocco", provider=provider)

    assert create_fn.call_count == 3
    assert len(scores) == 9
    first_batch_scores = [s for s in scores if s.niche in batch1_names]
    second_batch_scores = [s for s in scores if s.niche in batch2_names]
    assert all(s.evidence_level == "insufficient" for s in first_batch_scores)
    assert all(s.evidence_level == "sufficient" for s in second_batch_scores)
