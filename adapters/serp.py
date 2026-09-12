"""Project Meezan — SERP fallback adapter (docs/FRD.md, Section 8 / Task 6).

Competitor-count signal for the competitive_intensity dimension via a
free-tier search API, used sparingly. Returned as Evidence through the
with_retry wrapper from adapters/base.py.

Two providers are supported — Serper.dev and SerpAPI — selected at runtime
via the SERP_PROVIDER env var ("serper" or "serpapi", default "serper").
Each provider has its own internal fetch function responsible only for
making the request and reporting back (competitor_count, quota_exhausted);
`collect()` picks the right one and builds Evidence identically either
way, so nothing outside this module needs to know which provider is
active. `source_type` is always "serp"; the active provider is recorded in
`raw_data` for traceability.

The free tier is quota-limited, and a quota-exhausted response is NOT the
same as a genuine failure: retrying it wastes the (already-exhausted)
budget for no benefit, and treating it as "unavailable" would make the
downstream cheap-pass prompt silently skip the dimension. So quota
exhaustion is detected inside the per-provider fetch functions — before
`with_retry` ever sees an exception — and turned into Evidence with
confidence explicitly lowered to QUOTA_EXHAUSTED_CONFIDENCE, telling the
cheap-pass prompt to fall back to a wider, lower-confidence estimate
instead of skipping the dimension. Per FRD Section 4, adapters never call
DeepSeek directly, so that fallback estimate happens downstream, not here.
A genuine failure (bad response, network error, etc.) still propagates
normally and is retried up to 3 times by `with_retry` before giving up
with "unavailable".

An HTTP 429 is treated as the universal quota-exhausted signal for both
providers — that alone is enough to not misclassify a real failure as
quota exhaustion or vice versa. Provider-specific error-message keywords
are layered on top for cases where the API returns 200/403 with a quota
message instead of a 429.
"""

import os
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from adapters.base import Evidence, with_retry

SERPER_API_BASE_URL = "https://google.serper.dev/search"
SERPAPI_BASE_URL = "https://serpapi.com/search"
DEFAULT_PROVIDER = "serper"
QUOTA_EXHAUSTED_CONFIDENCE = 0.1

_QUOTA_KEYWORDS = ("quota", "run out", "rate limit", "too many requests")


def _is_quota_exhausted(status_code: int | None, error_text: str) -> bool:
    if status_code == 429:
        return True
    error = error_text.lower()
    return any(phrase in error for phrase in _QUOTA_KEYWORDS)


def _fetch_via_serper(niche: str, api_key: str, session) -> tuple[int, bool]:
    """Serper.dev: POST with an X-API-KEY header; results under "organic"."""
    http = session or requests
    response = http.post(
        SERPER_API_BASE_URL,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": niche},
        timeout=10,
    )
    data = response.json()
    error_text = str(data.get("message") or data.get("error") or "")

    if _is_quota_exhausted(getattr(response, "status_code", None), error_text):
        return 0, True

    response.raise_for_status()
    return len(data.get("organic", [])), False


def _fetch_via_serpapi(niche: str, api_key: str, session) -> tuple[int, bool]:
    """SerpAPI: GET with api_key as a query param; results under "organic_results"."""
    http = session or requests
    response = http.get(
        SERPAPI_BASE_URL,
        params={"q": niche, "api_key": api_key, "engine": "google"},
        timeout=10,
    )
    data = response.json()
    error_text = str(data.get("error") or "")

    if _is_quota_exhausted(getattr(response, "status_code", None), error_text):
        return 0, True

    response.raise_for_status()
    return len(data.get("organic_results", [])), False


_PROVIDERS = {
    "serper": _fetch_via_serper,
    "serpapi": _fetch_via_serpapi,
}


@with_retry(max_attempts=3, base_delay=1.0)
def collect(niche: str, session=None, api_key: str | None = None) -> list[Evidence]:
    """Raw collection: competitor-count signal for `niche` via a free-tier SERP API.

    Wrapped by `with_retry`, so calling this returns an `AdapterResult`
    (`{"status": ..., "evidence": [...]}`), not the raw `list[Evidence]`
    this function itself produces.

    The active provider is chosen from the SERP_PROVIDER env var ("serper"
    or "serpapi", default "serper"). `session` accepts an injected
    requests-like client so tests can mock either provider's API instead
    of hitting it for real.
    """
    provider = os.environ.get("SERP_PROVIDER", DEFAULT_PROVIDER).lower()
    fetch_fn = _PROVIDERS.get(provider)
    if fetch_fn is None:
        raise ValueError(f"Unknown SERP_PROVIDER: {provider!r}")

    key = api_key or os.environ["SERP_API_KEY"]
    retrieved_at = datetime.now(timezone.utc).isoformat()
    source_url = f"https://www.google.com/search?q={quote(niche)}"

    competitor_count, quota_exhausted = fetch_fn(niche, key, session)

    if quota_exhausted:
        return [
            Evidence(
                source_type="serp",
                metric="competitor_count",
                value="unknown",
                confidence=QUOTA_EXHAUSTED_CONFIDENCE,
                source_url=source_url,
                raw_data={"provider": provider, "quota_exhausted": True},
                retrieved_at=retrieved_at,
                cost_usd=0.0,
            )
        ]

    return [
        Evidence(
            source_type="serp",
            metric="competitor_count",
            value=str(competitor_count),
            confidence=0.6 if competitor_count > 0 else 0.2,
            source_url=source_url,
            raw_data={"provider": provider, "competitor_count": competitor_count},
            retrieved_at=retrieved_at,
            cost_usd=0.0,
        )
    ]
