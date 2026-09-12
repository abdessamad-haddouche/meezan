"""Project Meezan — Marketplace adapters (docs/FRD.md, Section 8 / Task 7).

Listing-count signal per digital-product marketplace, usable as a
margin/competition proxy for digital-product niches. Returned as Evidence
through the with_retry wrapper from adapters/base.py, one function per
source (matching the pattern of reddit.py / trends.py / meta_ads.py /
serp.py).

Per FRD Section 8 ("Prefer official API; skip (don't scrape around) any
source whose ToS prohibits it"), only Etsy is implemented here:

- Etsy: `collect_etsy()` — official Open API v3, `findAllListingsActive`
  (`GET /v3/application/listings/active`), authenticated with just an API
  key (`x-api-key` header, no OAuth needed for this public, read-only
  endpoint). Supports keyword search and returns a total `count`, which is
  exactly the listing-count signal this task needs.

- Gumroad: SKIPPED, not implemented. Two independent reasons: (1) Gumroad's
  official API (api.gumroad.com/v2) only exposes products belonging to the
  OAuth-authenticated seller's own account — there is no endpoint to search
  or count listings across the marketplace/Discover catalog by category or
  keyword, so it cannot answer the question this adapter needs regardless
  of ToS; (2) Gumroad's Terms of Service (Section 14) explicitly prohibit
  "spiders, robots, scrapers, crawlers, ... or the like" scraping any page
  of the Service, which rules out getting the signal by scraping Discover
  as a workaround.

- Notion template galleries: SKIPPED, not implemented. Notion's official
  API (developers.notion.com) grants an integration access only to
  workspace content (pages/databases) it has explicitly been shared with —
  it has no visibility into, or search surface for, the public Template
  Gallery/Marketplace catalog. There is no official way to get a listing
  or category count from it, and scraping the Marketplace website instead
  would be the same kind of ToS-skirting workaround this task says not to
  do, so it's skipped rather than attempted.

If either of those sources later grows an official marketplace-search API,
add a `collect_gumroad()` / `collect_notion_templates()` here following the
same shape as `collect_etsy()`.
"""

import os
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from adapters.base import Evidence, with_retry

ETSY_API_BASE_URL = "https://api.etsy.com/v3/application/listings/active"
DEFAULT_LIMIT = 100


def _fetch_etsy_listing_count(niche: str, api_key: str, session, limit: int) -> int:
    http = session or requests
    response = http.get(
        ETSY_API_BASE_URL,
        headers={"x-api-key": api_key},
        params={"keywords": niche, "limit": limit},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return int(data.get("count", len(data.get("results", []))))


@with_retry(max_attempts=3, base_delay=1.0)
def collect_etsy(
    niche: str, session=None, api_key: str | None = None, limit: int = DEFAULT_LIMIT
) -> list[Evidence]:
    """Raw collection: active-listing-count signal for `niche` via Etsy Open API v3.

    Wrapped by `with_retry`, so calling this returns an `AdapterResult`
    (`{"status": ..., "evidence": [...]}`), not the raw `list[Evidence]`
    this function itself produces.

    `session` accepts an injected requests-like client so tests can mock
    the Etsy API instead of hitting it for real.
    """
    key = api_key or os.environ["ETSY_API_KEY"]
    listing_count = _fetch_etsy_listing_count(niche, key, session, limit)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    return [
        Evidence(
            source_type="etsy",
            metric="listing_count",
            value=str(listing_count),
            confidence=0.6 if listing_count > 0 else 0.2,
            source_url=f"https://www.etsy.com/search?q={quote(niche)}",
            raw_data={"listing_count": listing_count, "limit": limit},
            retrieved_at=retrieved_at,
            cost_usd=0.0,
        )
    ]
