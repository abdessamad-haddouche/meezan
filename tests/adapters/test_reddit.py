"""Unit tests for adapters/reddit.py (docs/FRD.md, Section 8 / Task 3)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from adapters.reddit import collect


def _make_reddit_mock(search_return=None, search_side_effect=None):
    reddit = MagicMock()
    search_mock = reddit.subreddit.return_value.search
    if search_side_effect is not None:
        search_mock.side_effect = search_side_effect
    else:
        search_mock.return_value = search_return or []
    return reddit


def test_successful_fetch_returns_correctly_shaped_evidence():
    submissions = [SimpleNamespace(score=10), SimpleNamespace(score=5), SimpleNamespace(score=0)]
    reddit = _make_reddit_mock(search_return=submissions)

    result = collect("solar panel cleaning kits", reddit=reddit)

    assert result["status"] == "ok"
    assert len(result["evidence"]) == 1

    ev = result["evidence"][0]
    assert ev["source_type"] == "reddit"
    assert ev["metric"] == "mention_count"
    assert ev["value"] == "3"
    assert 0.0 <= ev["confidence"] <= 1.0
    assert ev["source_url"].startswith("https://www.reddit.com/search")
    assert ev["raw_data"]["total_score"] == 15
    assert isinstance(ev["retrieved_at"], str) and ev["retrieved_at"]
    assert ev["cost_usd"] == 0.0

    reddit.subreddit.assert_called_with("all")
    reddit.subreddit.return_value.search.assert_called_once_with(
        "solar panel cleaning kits", limit=100
    )


@patch("adapters.base.time.sleep", return_value=None)
def test_failure_after_three_retries_returns_unavailable(_mock_sleep):
    reddit = _make_reddit_mock(search_side_effect=RuntimeError("PRAW request failed"))

    result = collect("solar panel cleaning kits", reddit=reddit)

    assert result == {"status": "unavailable", "evidence": []}
    assert reddit.subreddit.return_value.search.call_count == 3
