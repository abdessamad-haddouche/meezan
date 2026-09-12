"""Project Meezan — Meta Ad Library adapter (docs/FRD.md, Section 8).

Ad-saturation signal for a niche via the Meta Graph API's `ads_archive`
endpoint (free tier, 200 req/hr). Returned as Evidence through the
with_retry wrapper from adapters/base.py.

The free tier can't absorb repeat calls for the same niche within a sweep
(cheap pass + a later deep pass, retries elsewhere in the pipeline, etc.),
so successful results are cached in-memory per (niche, country) for
`DEFAULT_CACHE_TTL_SECONDS`.
"""

import os
import time
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from adapters.base import Evidence, with_retry

GRAPH_API_VERSION = "v19.0"
GRAPH_API_BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/ads_archive"
DEFAULT_COUNTRY = "US"
DEFAULT_CACHE_TTL_SECONDS = 3600.0  # matches the 200 req/hr free-tier window

_cache: dict[tuple[str, str], tuple[float, list[Evidence]]] = {}


def clear_cache() -> None:
    """Reset the in-memory cache. Mainly useful for tests."""
    _cache.clear()


def _get_cached(key: tuple[str, str], ttl_seconds: float) -> list[Evidence] | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    cached_at, evidence = entry
    if time.monotonic() - cached_at > ttl_seconds:
        return None
    return evidence


def _fetch_ad_count(niche: str, country: str, access_token: str, session) -> int:
    http = session or requests
    response = http.get(
        GRAPH_API_BASE_URL,
        params={
            "search_terms": niche,
            "ad_reached_countries": f'["{country}"]',
            "access_token": access_token,
            "limit": 100,
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return len(data.get("data", []))


@with_retry(max_attempts=3, base_delay=1.0)
def collect(
    niche: str,
    country: str = DEFAULT_COUNTRY,
    session=None,
    access_token: str | None = None,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
) -> list[Evidence]:
    """Raw collection: ad-saturation signal for `niche` via Meta Ad Library.

    Wrapped by `with_retry`, so calling this returns an `AdapterResult`
    (`{"status": ..., "evidence": [...]}`), not the raw `list[Evidence]`
    this function itself produces.

    A cache hit short-circuits before any request is made, so repeat calls
    for the same (niche, country) within `cache_ttl_seconds` cost zero
    quota. Only successful fetches are cached — a failed attempt never
    poisons the cache with an empty/partial result.

    `session` accepts an injected requests-like client so tests can mock
    the Graph API instead of hitting it for real.
    """
    key = (niche, country)
    cached = _get_cached(key, cache_ttl_seconds)
    if cached is not None:
        return cached

    token = access_token or os.environ["META_ACCESS_TOKEN"]
    ad_count = _fetch_ad_count(niche, country, token, session)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    evidence = [
        Evidence(
            source_type="meta_ads",
            metric="ad_count",
            value=str(ad_count),
            confidence=0.6 if ad_count > 0 else 0.2,
            source_url=f"https://www.facebook.com/ads/library/?q={quote(niche)}",
            raw_data={"country": country, "ad_count": ad_count},
            retrieved_at=retrieved_at,
            cost_usd=0.0,
        )
    ]

    _cache[key] = (time.monotonic(), evidence)
    return evidence
