"""Integration tests for sweep_engine.py (docs/FRD.md, Section 4 / Task 12)."""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from adapters import marketplace, meta_ads, reddit, serp, trends, worldbank
from db.migrate import migrate
from llm.deepseek_provider import DeepSeekProvider
from sweep_engine import run_sweep

ALL_SOURCE_TYPES = {"reddit", "google_trends", "meta_ads", "serp", "etsy", "worldbank"}
ALL_DIMENSIONS = {"market_demand", "competitive_intensity", "margin_signal", "trend_momentum"}


def _fake_response(content: str, total_tokens: int = 10):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def _expansion_payload(suggested: list[str]) -> str:
    return json.dumps({"suggested_niches": suggested})


def _cheap_pass_payload(niches: list[str]) -> str:
    return json.dumps(
        {
            "scores": [
                {
                    "niche": n,
                    "market_demand": 7,
                    "competitive_intensity": 4,
                    "margin_signal": 6,
                    "trend_momentum": 5,
                    "evidence_level": "sufficient",
                    "rationale": "Solid signal across sources.",
                }
                for n in niches
            ]
        }
    )


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


def test_full_sweep_writes_expected_rows_for_seed_and_expanded_niches(db_conn, monkeypatch):
    _patch_all_adapters_ok(monkeypatch)
    seeds = ["pet products", "modest wear"]
    create_fn = MagicMock(
        side_effect=[
            _fake_response(_expansion_payload(["cosmetics"])),
            _fake_response(_cheap_pass_payload(seeds + ["cosmetics"])),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    sweep_run_id = run_sweep(
        db_conn, category="ecommerce", market="morocco", seed_niches=seeds, provider=provider
    )

    # Exactly one sweep_runs row, completed.
    sweep_rows = db_conn.execute("SELECT id, status FROM sweep_runs").fetchall()
    assert sweep_rows == [(sweep_run_id, "completed")]

    idea_rows = db_conn.execute(
        "SELECT id, name, seed_or_suggested, composite_score, composite_confidence, research_priority "
        "FROM ideas WHERE sweep_run_id = ?",
        (sweep_run_id,),
    ).fetchall()
    assert len(idea_rows) == 3
    origins = {name: origin for _id, name, origin, *_ in idea_rows}
    assert origins == {"pet products": "seed", "modest wear": "seed", "cosmetics": "suggested"}

    for _id, _name, _origin, composite_score, composite_confidence, research_priority in idea_rows:
        assert composite_score is not None
        assert composite_confidence is not None
        assert research_priority in {"research_this_week", "revisit_later", "deprioritize"}

    for idea_id, name, *_ in idea_rows:
        dims = {
            r[0] for r in db_conn.execute("SELECT dimension FROM scores WHERE idea_id = ?", (idea_id,))
        }
        assert dims == ALL_DIMENSIONS, f"idea {name!r} missing score dimensions"

        sources = {
            r[0] for r in db_conn.execute("SELECT source_type FROM evidence WHERE idea_id = ?", (idea_id,))
        }
        assert sources == ALL_SOURCE_TYPES, f"idea {name!r} missing evidence sources"


def test_one_unavailable_adapter_does_not_crash_sweep_or_skip_niches(db_conn, monkeypatch):
    _patch_all_adapters_ok(monkeypatch)
    monkeypatch.setattr(meta_ads, "collect", lambda niche: {"status": "unavailable", "evidence": []})

    seeds = ["pet products", "modest wear"]
    create_fn = MagicMock(
        side_effect=[
            _fake_response(_expansion_payload([])),
            _fake_response(_cheap_pass_payload(seeds)),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    sweep_run_id = run_sweep(
        db_conn, category="ecommerce", market="morocco", seed_niches=seeds, provider=provider
    )

    status = db_conn.execute("SELECT status FROM sweep_runs WHERE id = ?", (sweep_run_id,)).fetchone()[0]
    assert status == "completed"

    idea_rows = db_conn.execute(
        "SELECT id, name FROM ideas WHERE sweep_run_id = ?", (sweep_run_id,)
    ).fetchall()
    assert {name for _id, name in idea_rows} == set(seeds)  # neither niche skipped

    for idea_id, name in idea_rows:
        sources = {
            r[0] for r in db_conn.execute("SELECT source_type FROM evidence WHERE idea_id = ?", (idea_id,))
        }
        assert "meta_ads" not in sources, f"idea {name!r} unexpectedly has meta_ads evidence"
        assert sources == ALL_SOURCE_TYPES - {"meta_ads"}

        dims = {
            r[0] for r in db_conn.execute("SELECT dimension FROM scores WHERE idea_id = ?", (idea_id,))
        }
        assert dims == ALL_DIMENSIONS


def test_disabled_adapters_are_never_called(db_conn, monkeypatch):
    monkeypatch.setenv("DISABLED_ADAPTERS", "meta_ads, etsy")

    reddit_mock = MagicMock(side_effect=lambda niche: _ok_result("reddit", "mention_count", "12"))
    trends_mock = MagicMock(side_effect=lambda niche: _ok_result("google_trends", "interest_over_time", "45"))
    meta_ads_mock = MagicMock(side_effect=lambda niche: _ok_result("meta_ads", "ad_count", "3"))
    serp_mock = MagicMock(side_effect=lambda niche: _ok_result("serp", "competitor_count", "8"))
    etsy_mock = MagicMock(side_effect=lambda niche: _ok_result("etsy", "listing_count", "100"))
    worldbank_mock = MagicMock(side_effect=lambda market: _ok_result("worldbank", "gdp_per_capita_usd", "3200"))

    monkeypatch.setattr(reddit, "collect", reddit_mock)
    monkeypatch.setattr(trends, "collect", trends_mock)
    monkeypatch.setattr(meta_ads, "collect", meta_ads_mock)
    monkeypatch.setattr(serp, "collect", serp_mock)
    monkeypatch.setattr(marketplace, "collect_etsy", etsy_mock)
    monkeypatch.setattr(worldbank, "collect", worldbank_mock)

    seeds = ["pet products", "modest wear"]
    create_fn = MagicMock(
        side_effect=[
            _fake_response(_expansion_payload([])),
            _fake_response(_cheap_pass_payload(seeds)),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    sweep_run_id = run_sweep(
        db_conn, category="ecommerce", market="morocco", seed_niches=seeds, provider=provider
    )

    # Disabled adapters: skipped entirely, never called.
    assert meta_ads_mock.call_count == 0
    assert etsy_mock.call_count == 0
    # Every other adapter still runs once per niche, as normal.
    assert reddit_mock.call_count == len(seeds)
    assert trends_mock.call_count == len(seeds)
    assert serp_mock.call_count == len(seeds)
    assert worldbank_mock.call_count == len(seeds)

    idea_ids = [r[0] for r in db_conn.execute("SELECT id FROM ideas WHERE sweep_run_id = ?", (sweep_run_id,))]
    assert len(idea_ids) == len(seeds)
    for idea_id in idea_ids:
        sources = {
            r[0] for r in db_conn.execute("SELECT source_type FROM evidence WHERE idea_id = ?", (idea_id,))
        }
        assert sources == ALL_SOURCE_TYPES - {"meta_ads", "etsy"}


def test_niche_expansion_degrades_to_seeds_only_after_two_malformed_responses(db_conn, monkeypatch):
    _patch_all_adapters_ok(monkeypatch)
    seeds = ["pet products", "modest wear"]
    create_fn = MagicMock(
        side_effect=[
            _fake_response("not json at all"),
            _fake_response("still not json"),
            _fake_response(_cheap_pass_payload(seeds)),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    sweep_run_id = run_sweep(
        db_conn, category="ecommerce", market="morocco", seed_niches=seeds, provider=provider
    )

    assert create_fn.call_count == 3  # 2 failed expansion attempts + 1 cheap-pass batch
    status = db_conn.execute("SELECT status FROM sweep_runs WHERE id = ?", (sweep_run_id,)).fetchone()[0]
    assert status == "completed"

    idea_names = {
        r[0] for r in db_conn.execute("SELECT name FROM ideas WHERE sweep_run_id = ?", (sweep_run_id,))
    }
    assert idea_names == set(seeds)  # no suggested niches, but sweep still completes
