"""Unit tests for adapters/serp.py (docs/FRD.md, Section 8 / Task 6)."""

from unittest.mock import MagicMock, patch

import pytest

from adapters.serp import collect


def _make_response(status_code=200, data=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = data or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        response.raise_for_status.return_value = None
    return response


def _serper_session(status_code=200, data=None, raise_exc=None):
    session = MagicMock()
    if raise_exc is not None:
        session.post.side_effect = raise_exc
    else:
        session.post.return_value = _make_response(status_code, data)
    return session


def _serpapi_session(status_code=200, data=None, raise_exc=None):
    session = MagicMock()
    if raise_exc is not None:
        session.get.side_effect = raise_exc
    else:
        session.get.return_value = _make_response(status_code, data)
    return session


# ---- Serper.dev ----------------------------------------------------------


@patch.dict("os.environ", {"SERP_PROVIDER": "serper"})
def test_serper_successful_fetch_returns_correctly_shaped_evidence():
    session = _serper_session(data={"organic": [{"link": f"http://x/{i}"} for i in range(4)]})

    result = collect("solar panel cleaning kits", session=session, api_key="test-key")

    assert result["status"] == "ok"
    ev = result["evidence"][0]
    assert ev["source_type"] == "serp"
    assert ev["metric"] == "competitor_count"
    assert ev["value"] == "4"
    assert 0.0 <= ev["confidence"] <= 1.0
    assert ev["raw_data"] == {"provider": "serper", "competitor_count": 4}
    assert ev["cost_usd"] == 0.0

    session.post.assert_called_once()
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["X-API-KEY"] == "test-key"
    assert kwargs["json"] == {"q": "solar panel cleaning kits"}


@patch.dict("os.environ", {"SERP_PROVIDER": "serper"})
def test_serper_quota_exhausted_returns_lowered_confidence_not_unavailable():
    session = _serper_session(status_code=429, data={"message": "Too Many Requests"})

    result = collect("solar panel cleaning kits", session=session, api_key="test-key")

    assert result["status"] == "ok"
    ev = result["evidence"][0]
    assert ev["confidence"] == pytest.approx(0.1)
    assert ev["confidence"] < 0.2
    assert ev["raw_data"] == {"provider": "serper", "quota_exhausted": True}
    session.post.assert_called_once()  # not retried


@patch("adapters.base.time.sleep", return_value=None)
@patch.dict("os.environ", {"SERP_PROVIDER": "serper"})
def test_serper_genuine_failure_after_three_retries_returns_unavailable(_mock_sleep):
    session = _serper_session(raise_exc=RuntimeError("serper request failed"))

    result = collect("solar panel cleaning kits", session=session, api_key="test-key")

    assert result == {"status": "unavailable", "evidence": []}
    assert session.post.call_count == 3


# ---- SerpAPI --------------------------------------------------------------


@patch.dict("os.environ", {"SERP_PROVIDER": "serpapi"})
def test_serpapi_successful_fetch_returns_correctly_shaped_evidence():
    session = _serpapi_session(data={"organic_results": [{"link": f"http://x/{i}"} for i in range(5)]})

    result = collect("solar panel cleaning kits", session=session, api_key="test-key")

    assert result["status"] == "ok"
    ev = result["evidence"][0]
    assert ev["source_type"] == "serp"
    assert ev["metric"] == "competitor_count"
    assert ev["value"] == "5"
    assert 0.0 <= ev["confidence"] <= 1.0
    assert ev["raw_data"] == {"provider": "serpapi", "competitor_count": 5}
    assert ev["cost_usd"] == 0.0

    session.get.assert_called_once()
    _, kwargs = session.get.call_args
    assert kwargs["params"]["q"] == "solar panel cleaning kits"
    assert kwargs["params"]["api_key"] == "test-key"


@patch.dict("os.environ", {"SERP_PROVIDER": "serpapi"})
def test_serpapi_quota_exhausted_returns_lowered_confidence_not_unavailable():
    session = _serpapi_session(status_code=429, data={"error": "You have exceeded your monthly quota"})

    result = collect("solar panel cleaning kits", session=session, api_key="test-key")

    assert result["status"] == "ok"
    ev = result["evidence"][0]
    assert ev["confidence"] == pytest.approx(0.1)
    assert ev["confidence"] < 0.2
    assert ev["raw_data"] == {"provider": "serpapi", "quota_exhausted": True}
    session.get.assert_called_once()  # not retried


@patch.dict("os.environ", {"SERP_PROVIDER": "serpapi"})
def test_serpapi_quota_exhausted_via_error_message_without_429_status():
    session = _serpapi_session(status_code=200, data={"error": "Rate limit reached, run out of searches"})

    result = collect("solar panel cleaning kits", session=session, api_key="test-key")

    ev = result["evidence"][0]
    assert ev["confidence"] == pytest.approx(0.1)
    session.get.assert_called_once()


@patch("adapters.base.time.sleep", return_value=None)
@patch.dict("os.environ", {"SERP_PROVIDER": "serpapi"})
def test_serpapi_genuine_failure_after_three_retries_returns_unavailable(_mock_sleep):
    session = _serpapi_session(raise_exc=RuntimeError("serpapi request failed"))

    result = collect("solar panel cleaning kits", session=session, api_key="test-key")

    assert result == {"status": "unavailable", "evidence": []}
    assert session.get.call_count == 3


# ---- Provider selection ----------------------------------------------------


@patch.dict("os.environ", {}, clear=False)
def test_default_provider_is_serper_when_env_var_unset():
    import os as _os

    _os.environ.pop("SERP_PROVIDER", None)
    session = _serper_session(data={"organic": []})

    result = collect("niche", session=session, api_key="test-key")

    assert result["status"] == "ok"
    assert result["evidence"][0]["raw_data"]["provider"] == "serper"
    session.post.assert_called_once()
