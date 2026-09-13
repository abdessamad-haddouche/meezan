"""Project Meezan — Reddit adapter (docs/FRD.md, Section 8).

Sweep mode: a simple mention-count signal for a niche, searched across all
of Reddit via PRAW. Returned as Evidence through the with_retry wrapper
from adapters/base.py. Discovery mode (subreddit.rising + velocity
filtering) is Task 18 — not built here.
"""

import os
from datetime import datetime, timezone
from urllib.parse import quote

import praw

from adapters.base import Evidence, with_retry

DEFAULT_LIMIT = 100
# PRAW/prawcore already default to this same value (praw.ini) if omitted;
# spelled out explicitly so a slow/unresponsive Reddit API can't hang a
# sweep indefinitely (it fails into with_retry's existing retry logic
# instead).
DEFAULT_TIMEOUT_SECONDS = 16


def _build_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "meezan/0.1"),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


@with_retry(max_attempts=3, base_delay=1.0)
def collect(
    niche: str, reddit: praw.Reddit | None = None, limit: int = DEFAULT_LIMIT
) -> list[Evidence]:
    """Raw collection: mention-count signal for `niche` across all of Reddit.

    Wrapped by `with_retry`, so calling this returns an `AdapterResult`
    (`{"status": ..., "evidence": [...]}`), not the raw `list[Evidence]`
    this function itself produces.

    `reddit` accepts an injected PRAW client, mainly so tests can pass a
    mock instead of hitting real credentials/network.
    """
    client = reddit or _build_client()
    submissions = list(client.subreddit("all").search(niche, limit=limit))
    mention_count = len(submissions)
    total_score = sum(getattr(s, "score", 0) for s in submissions)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    return [
        Evidence(
            source_type="reddit",
            metric="mention_count",
            value=str(mention_count),
            confidence=0.6 if mention_count > 0 else 0.2,
            source_url=f"https://www.reddit.com/search/?q={quote(niche)}",
            raw_data={"total_score": total_score, "limit": limit},
            retrieved_at=retrieved_at,
            cost_usd=0.0,
        )
    ]
