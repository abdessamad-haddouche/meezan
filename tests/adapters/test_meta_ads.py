"""Unit tests for adapters/meta_ads.py (docs/FRD.md, Section 8 / Task 5)."""

from unittest.mock import MagicMock, patch

import pytest

from adapters import meta_ads
from adapters.meta_ads import collect


@pytest.fixture(autouse=True)
def _clear_cache():
    meta_ads.clear_cache()
    yield
    meta_ads.clear_cache()


def _make_session_mock(ad_count=None, raise_exc=None):
    session = MagicMock()
    if raise_exc is not None:
        session.get.side_effect = raise_exc
        return session

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{"id": str(i)} for i in range(ad_count)]}
    session.get.return_value = response
    return session


def test_successful_fetch_returns_correctly_shaped_evidence():
    session = _make_session_mock(ad_count=3)

    result = collect("solar panel cleaning kits", session=session, access_token="test-token")

    assert result["status"] == "ok"
    assert len(result["evidence"]) == 1

    ev = result["evidence"][0]
    assert ev["source_type"] == "meta_ads"
    assert ev["metric"] == "ad_count"
    assert ev["value"] == "3"
    assert 0.0 <= ev["confidence"] <= 1.0
    assert ev["source_url"].startswith("https://www.facebook.com/ads/library")
    assert ev["raw_data"] == {"country": "US", "ad_count": 3}
    assert isinstance(ev["retrieved_at"], str) and ev["retrieved_at"]
    assert ev["cost_usd"] == 0.0

    session.get.assert_called_once()
    _, kwargs = session.get.call_args
    assert kwargs["params"]["search_terms"] == "solar panel cleaning kits"
    assert kwargs["params"]["access_token"] == "test-token"


@patch("adapters.base.time.sleep", return_value=None)
def test_failure_after_three_retries_returns_unavailable(_mock_sleep):
    session = _make_session_mock(raise_exc=RuntimeError("Graph API request failed"))

    result = collect("solar panel cleaning kits", session=session, access_token="test-token")

    assert result == {"status": "unavailable", "evidence": []}
    assert session.get.call_count == 3


def test_second_call_within_cache_window_does_not_hit_api_again():
    session = _make_session_mock(ad_count=5)

    first = collect("solar panel cleaning kits", session=session, access_token="test-token")
    second = collect("solar panel cleaning kits", session=session, access_token="test-token")

    assert first == second
    session.get.assert_called_once()


def test_call_after_cache_expiry_hits_api_again():
    session = _make_session_mock(ad_count=5)

    collect(
        "solar panel cleaning kits",
        session=session,
        access_token="test-token",
        cache_ttl_seconds=0.0,
    )
    collect(
        "solar panel cleaning kits",
        session=session,
        access_token="test-token",
        cache_ttl_seconds=0.0,
    )

    assert session.get.call_count == 2


def test_different_niche_is_not_served_from_other_niche_cache():
    session = _make_session_mock(ad_count=5)

    collect("solar panel cleaning kits", session=session, access_token="test-token")
    collect("pet grooming subscription boxes", session=session, access_token="test-token")

    assert session.get.call_count == 2
