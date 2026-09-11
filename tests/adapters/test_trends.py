"""Unit tests for adapters/trends.py (docs/FRD.md, Section 8 / Task 4)."""

from unittest.mock import MagicMock, patch

import pandas as pd

from adapters.trends import DEFAULT_TIMEFRAME, collect


def _make_pytrends_mock(df=None, side_effect=None):
    client = MagicMock()
    if side_effect is not None:
        client.interest_over_time.side_effect = side_effect
    else:
        client.interest_over_time.return_value = df
    return client


def test_successful_fetch_returns_correctly_shaped_evidence():
    niche = "solar panel cleaning kits"
    df = pd.DataFrame(
        {niche: [10, 20, 30], "isPartial": [False, False, False]},
        index=pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15"]),
    )
    pytrends = _make_pytrends_mock(df=df)

    result = collect(niche, pytrends=pytrends)

    assert result["status"] == "ok"
    assert len(result["evidence"]) == 1

    ev = result["evidence"][0]
    assert ev["source_type"] == "google_trends"
    assert ev["metric"] == "interest_over_time"
    assert ev["value"] == "20.0"
    assert 0.0 <= ev["confidence"] <= 1.0
    assert ev["source_url"].startswith("https://trends.google.com/trends/explore")
    assert ev["raw_data"]["timeframe"] == DEFAULT_TIMEFRAME
    assert ev["raw_data"]["series"] == [
        {"date": "2026-01-01", "value": 10},
        {"date": "2026-01-08", "value": 20},
        {"date": "2026-01-15", "value": 30},
    ]
    assert isinstance(ev["retrieved_at"], str) and ev["retrieved_at"]
    assert ev["cost_usd"] == 0.0

    pytrends.build_payload.assert_called_once_with(kw_list=[niche], timeframe=DEFAULT_TIMEFRAME)
    pytrends.interest_over_time.assert_called_once()


@patch("adapters.base.time.sleep", return_value=None)
def test_failure_after_three_retries_returns_unavailable(_mock_sleep):
    pytrends = _make_pytrends_mock(side_effect=RuntimeError("pytrends request failed"))

    result = collect("solar panel cleaning kits", pytrends=pytrends)

    assert result == {"status": "unavailable", "evidence": []}
    assert pytrends.interest_over_time.call_count == 3
