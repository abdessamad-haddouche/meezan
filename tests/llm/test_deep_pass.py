"""Unit tests for llm/deep_pass.py (docs/FRD.md, Section 7.3 / Task 13)."""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from adapters import marketplace, meta_ads, reddit, serp, trends, worldbank
from db.migrate import migrate
from llm.deep_pass import DEEP_PASS_MODEL, deep_score
from llm.deepseek_provider import DeepSeekProvider
from scoring.engine import load_weights_config

ALL_SOURCE_TYPES = {"reddit", "google_trends", "meta_ads", "serp", "etsy", "worldbank"}
CHEAP_SCORES = {
    "market_demand": 8,
    "competitive_intensity": 4,
    "margin_signal": 6,
    "trend_momentum": 7,
}
CHEAP_CONFIDENCE = 0.7


def _fake_response(content: str, total_tokens: int = 10):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def _deep_score_payload(**overrides) -> str:
    payload = {
        "margin_min_pct": 15.0,
        "margin_max_pct": 35.0,
        "cost_to_v1_min_usd": 200,
        "cost_to_v1_max_usd": 800,
        "complexity": 2,
        "regulatory_friction": 4,
        "comparables_used": ["Etsy digital planners at $12", "Instagram pet accessories shop at $25"],
        "key_risks": ["Seasonal demand dip in summer"],
        "evidence_level": "sufficient",
        "rationale": "Strong comparables found across marketplaces.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _ok_result(source_type: str, metric: str, value: str) -> dict:
    return {
        "status": "ok",
        "evidence": [
            {
                "source_type": source_type,
                "metric": metric,
                "value": value,
                "confidence": 0.6,
                "source_url": None,
                "raw_data": {},
                "retrieved_at": "2026-09-13T00:00:00Z",
                "cost_usd": 0.0,
            }
        ],
    }


def _patch_all_adapters_ok(monkeypatch):
    monkeypatch.setattr(reddit, "collect", lambda niche: _ok_result("reddit", "mention_count", "12"))
    monkeypatch.setattr(trends, "collect", lambda niche: _ok_result("google_trends", "interest_over_time", "45"))
    monkeypatch.setattr(meta_ads, "collect", lambda niche: _ok_result("meta_ads", "ad_count", "3"))
    monkeypatch.setattr(serp, "collect", lambda niche: _ok_result("serp", "competitor_count", "8"))
    monkeypatch.setattr(marketplace, "collect_etsy", lambda niche: _ok_result("etsy", "listing_count", "100"))
    monkeypatch.setattr(worldbank, "collect", lambda market: _ok_result("worldbank", "gdp_per_capita_usd", "3200"))


@pytest.fixture
def db_conn(tmp_path):
    db_path = tmp_path / "meezan_test.db"
    migrate(db_path)
    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


def _seed_cheap_pass_idea(
    conn: sqlite3.Connection, niche="pet products", market="morocco", category="ecommerce"
) -> int:
    """Simulate what Task 12's run_sweep already wrote for this idea: one
    sweep_runs row, one ideas row (preliminary composite, some priority),
    and its 4 cheap-pass scores rows."""
    now = "2026-09-13T00:00:00Z"
    cursor = conn.execute(
        "INSERT INTO sweep_runs (category, market, triggered_at, status) VALUES (?, ?, ?, 'completed')",
        (category, market, now),
    )
    sweep_run_id = cursor.lastrowid
    cursor = conn.execute(
        """
        INSERT INTO ideas (
            sweep_run_id, name, category, market, seed_or_suggested,
            composite_score, composite_confidence, research_priority, has_deep_pass, created_at
        ) VALUES (?, ?, ?, ?, 'seed', 6.31, 0.72, 'revisit_later', 0, ?)
        """,
        (sweep_run_id, niche, category, market, now),
    )
    idea_id = cursor.lastrowid
    for dimension, value in CHEAP_SCORES.items():
        conn.execute(
            """
            INSERT INTO scores (idea_id, dimension, value, confidence, evidence_level, rationale, pass_type)
            VALUES (?, ?, ?, ?, 'sufficient', 'cheap-pass rationale', 'cheap')
            """,
            (idea_id, dimension, value, CHEAP_CONFIDENCE),
        )
    conn.commit()
    return idea_id


def test_default_provider_is_built_with_deepseek_v4_pro_explicitly(db_conn, monkeypatch):
    _patch_all_adapters_ok(monkeypatch)
    idea_id = _seed_cheap_pass_idea(db_conn)

    import llm.deep_pass as deep_pass_module

    captured_kwargs = {}
    create_fn = MagicMock(return_value=_fake_response(_deep_score_payload()))

    class _SpyProvider(DeepSeekProvider):
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            kwargs["create_fn"] = create_fn
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(deep_pass_module, "DeepSeekProvider", _SpyProvider)

    deep_pass_module.deep_score(idea_id, db_conn, weights_config=load_weights_config())

    assert captured_kwargs["model"] == "deepseek-v4-pro"


def test_deep_score_writes_rows_updates_idea_and_flips_status_to_final(db_conn, monkeypatch):
    _patch_all_adapters_ok(monkeypatch)
    idea_id = _seed_cheap_pass_idea(db_conn)

    create_fn = MagicMock(return_value=_fake_response(_deep_score_payload()))
    provider = DeepSeekProvider(create_fn=create_fn, model=DEEP_PASS_MODEL)

    result = deep_score(idea_id, db_conn, provider=provider, weights_config=load_weights_config())

    # DeepSeekProvider was called on the deep-pass model, not the cheap pass's.
    assert create_fn.call_count == 1
    assert create_fn.call_args.kwargs["model"] == DEEP_PASS_MODEL == "deepseek-v4-pro"

    deep = result["deep_score"]
    assert deep.evidence_level == "sufficient"
    composite = result["composite"]
    assert composite["status"] == "final"
    # market_demand .20*8 + competitive_intensity .15*4 + margin_signal .20*6
    # + cost_to_start .15*9.55 + complexity .10*8 + regulatory_friction .10*4
    # + trend_momentum .10*7, weights already summing to 1.0
    assert composite["composite"] == pytest.approx(6.7325)
    # every dimension's confidence is 0.7 (cheap + deep), so the weighted
    # average confidence is just 0.7 regardless of the weights.
    assert composite["confidence"] == pytest.approx(0.7)

    idea_row = db_conn.execute(
        "SELECT margin_min_pct, margin_max_pct, cost_min_usd, cost_max_usd, complexity, "
        "has_deep_pass, composite_score, composite_confidence, research_priority "
        "FROM ideas WHERE id = ?",
        (idea_id,),
    ).fetchone()
    (
        margin_min_pct,
        margin_max_pct,
        cost_min_usd,
        cost_max_usd,
        complexity,
        has_deep_pass,
        composite_score,
        composite_confidence,
        research_priority,
    ) = idea_row
    assert margin_min_pct == pytest.approx(15.0)
    assert margin_max_pct == pytest.approx(35.0)
    assert cost_min_usd == 200
    assert cost_max_usd == 800
    assert complexity == 2  # raw 1-5 rating, denormalized as-is (not inverted)
    assert has_deep_pass == 1
    assert composite_score == pytest.approx(6.7325)
    assert composite_confidence == pytest.approx(0.7)
    # Deep dive never recomputes research_priority — that's a whole-sweep operation.
    assert research_priority == "revisit_later"

    deep_rows = db_conn.execute(
        "SELECT dimension, value, confidence, evidence_level, pass_type, comparable_products "
        "FROM scores WHERE idea_id = ? AND pass_type = 'deep' ORDER BY dimension",
        (idea_id,),
    ).fetchall()
    by_dimension = {row[0]: row for row in deep_rows}
    assert set(by_dimension) == {"cost_to_start", "complexity", "regulatory_friction"}

    cost_row = by_dimension["cost_to_start"]
    assert cost_row[1] == pytest.approx(9.55)  # midpoint $500 -> 10 - (500/10000)*9
    assert cost_row[2] == pytest.approx(0.7)
    assert cost_row[3] == "sufficient"
    assert cost_row[4] == "deep"
    assert json.loads(cost_row[5]) == [
        "Etsy digital planners at $12",
        "Instagram pet accessories shop at $25",
    ]

    complexity_row = by_dimension["complexity"]
    assert complexity_row[1] == pytest.approx(8.0)  # (6-2)*2

    friction_row = by_dimension["regulatory_friction"]
    assert friction_row[1] == pytest.approx(4.0)  # (6-4)*2

    # Cheap-pass scores rows are untouched.
    cheap_rows = db_conn.execute(
        "SELECT dimension FROM scores WHERE idea_id = ? AND pass_type = 'cheap'", (idea_id,)
    ).fetchall()
    assert {r[0] for r in cheap_rows} == set(CHEAP_SCORES)

    evidence_sources = {
        r[0] for r in db_conn.execute("SELECT source_type FROM evidence WHERE idea_id = ?", (idea_id,))
    }
    assert evidence_sources == ALL_SOURCE_TYPES


def test_deep_score_degrades_to_insufficient_after_two_malformed_responses(db_conn, monkeypatch):
    _patch_all_adapters_ok(monkeypatch)
    idea_id = _seed_cheap_pass_idea(db_conn)

    create_fn = MagicMock(
        side_effect=[
            _fake_response("this is not json"),
            _fake_response("still not json"),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    result = deep_score(idea_id, db_conn, provider=provider, weights_config=load_weights_config())

    assert create_fn.call_count == 2  # first attempt + one retry, then degrade — no third call

    deep = result["deep_score"]
    assert deep.evidence_level == "insufficient"
    assert "parsing failed twice" in deep.rationale
    # Degraded to the widest possible uncertainty, not an invented number.
    assert deep.margin_min_pct == 0.0
    assert deep.margin_max_pct == 100.0
    assert deep.complexity == 3
    assert deep.regulatory_friction == 3

    # All 7 dimensions still exist (even if degraded), so status is still "final".
    assert result["composite"]["status"] == "final"

    deep_rows = db_conn.execute(
        "SELECT dimension, evidence_level FROM scores WHERE idea_id = ? AND pass_type = 'deep'",
        (idea_id,),
    ).fetchall()
    assert len(deep_rows) == 3
    assert all(evidence_level == "insufficient" for _dimension, evidence_level in deep_rows)

    has_deep_pass = db_conn.execute(
        "SELECT has_deep_pass FROM ideas WHERE id = ?", (idea_id,)
    ).fetchone()[0]
    assert has_deep_pass == 1
