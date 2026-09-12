"""Unit tests for adapters/worldbank.py (docs/FRD.md, Section 8 / Task 8)."""

from unittest.mock import MagicMock, patch

from adapters.worldbank import (
    GDP_PER_CAPITA_USD,
    INTERNET_PENETRATION_PCT,
    collect,
)


def _make_fetch_mock(by_indicator=None, side_effect=None):
    fetch_fn = MagicMock()
    if side_effect is not None:
        fetch_fn.side_effect = side_effect
    else:
        fetch_fn.side_effect = lambda indicator, country: by_indicator[indicator]
    return fetch_fn


def test_successful_fetch_returns_correctly_shaped_evidence():
    by_indicator = {
        GDP_PER_CAPITA_USD: [
            {"indicator": {"id": GDP_PER_CAPITA_USD}, "country": {"id": "MA"}, "date": "2023", "value": 3559.7},
            {"indicator": {"id": GDP_PER_CAPITA_USD}, "country": {"id": "MA"}, "date": "2022", "value": 3350.1},
        ],
        INTERNET_PENETRATION_PCT: [
            {"indicator": {"id": INTERNET_PENETRATION_PCT}, "country": {"id": "MA"}, "date": "2023", "value": 90.2},
        ],
    }
    fetch_fn = _make_fetch_mock(by_indicator=by_indicator)

    result = collect("morocco", fetch_fn=fetch_fn)

    assert result["status"] == "ok"
    assert len(result["evidence"]) == 2

    by_metric = {ev["metric"]: ev for ev in result["evidence"]}

    gdp = by_metric["gdp_per_capita_usd"]
    assert gdp["source_type"] == "worldbank"
    assert gdp["value"] == "3559.7"
    assert 0.0 <= gdp["confidence"] <= 1.0
    assert gdp["source_url"].startswith("https://data.worldbank.org/indicator/")
    assert gdp["raw_data"] == {
        "indicator": GDP_PER_CAPITA_USD,
        "country": "MA",
        "date": "2023",
    }
    assert isinstance(gdp["retrieved_at"], str) and gdp["retrieved_at"]
    assert gdp["cost_usd"] == 0.0

    net = by_metric["internet_penetration_pct"]
    assert net["value"] == "90.2"
    assert net["raw_data"]["country"] == "MA"

    assert fetch_fn.call_count == 2
    fetch_fn.assert_any_call(GDP_PER_CAPITA_USD, country="MA")
    fetch_fn.assert_any_call(INTERNET_PENETRATION_PCT, country="MA")


@patch("adapters.base.time.sleep", return_value=None)
def test_failure_after_three_retries_returns_unavailable(_mock_sleep):
    fetch_fn = _make_fetch_mock(side_effect=RuntimeError("World Bank API request failed"))

    result = collect("morocco", fetch_fn=fetch_fn)

    assert result == {"status": "unavailable", "evidence": []}
    assert fetch_fn.call_count == 3


def test_unconfigured_market_returns_unavailable_without_calling_api():
    fetch_fn = MagicMock()

    result = collect("wakanda", fetch_fn=fetch_fn)

    assert result == {"status": "unavailable", "evidence": []}
    fetch_fn.assert_not_called()
