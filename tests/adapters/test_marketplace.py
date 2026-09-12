"""Unit tests for adapters/marketplace.py (docs/FRD.md, Section 8 / Task 7).

Only Etsy is exercised here — Gumroad and Notion template galleries have
no adapter to test; see the module docstring in adapters/marketplace.py
for why they were skipped rather than implemented via scraping.
"""

from unittest.mock import MagicMock, patch

from adapters.marketplace import collect_etsy


def _make_session_mock(count=None, results=None, raise_exc=None):
    session = MagicMock()
    if raise_exc is not None:
        session.get.side_effect = raise_exc
        return session

    response = MagicMock()
    response.raise_for_status.return_value = None
    data = {}
    if count is not None:
        data["count"] = count
    if results is not None:
        data["results"] = results
    response.json.return_value = data
    session.get.return_value = response
    return session


def test_etsy_successful_fetch_returns_correctly_shaped_evidence():
    session = _make_session_mock(count=42, results=[{"listing_id": i} for i in range(25)])

    result = collect_etsy("digital planner stickers", session=session, api_key="test-key")

    assert result["status"] == "ok"
    assert len(result["evidence"]) == 1

    ev = result["evidence"][0]
    assert ev["source_type"] == "etsy"
    assert ev["metric"] == "listing_count"
    assert ev["value"] == "42"
    assert 0.0 <= ev["confidence"] <= 1.0
    assert ev["source_url"].startswith("https://www.etsy.com/search")
    assert ev["raw_data"]["listing_count"] == 42
    assert isinstance(ev["retrieved_at"], str) and ev["retrieved_at"]
    assert ev["cost_usd"] == 0.0

    session.get.assert_called_once()
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["x-api-key"] == "test-key"
    assert kwargs["params"]["keywords"] == "digital planner stickers"


def test_etsy_falls_back_to_results_length_when_count_missing():
    session = _make_session_mock(results=[{"listing_id": i} for i in range(7)])

    result = collect_etsy("digital planner stickers", session=session, api_key="test-key")

    assert result["status"] == "ok"
    assert result["evidence"][0]["value"] == "7"


@patch("adapters.base.time.sleep", return_value=None)
def test_etsy_failure_after_three_retries_returns_unavailable(_mock_sleep):
    session = _make_session_mock(raise_exc=RuntimeError("Etsy API request failed"))

    result = collect_etsy("digital planner stickers", session=session, api_key="test-key")

    assert result == {"status": "unavailable", "evidence": []}
    assert session.get.call_count == 3
