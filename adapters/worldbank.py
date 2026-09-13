"""Project Meezan — World Bank / HCP adapter (docs/FRD.md, Section 8 / Task 8).

Market-size context signal (GDP per capita, internet penetration, ...) via
the World Bank open data API (through the `wbdata` client), used to
ground Morocco/MENA sweeps per Section 8's market-profile table. Returned
as Evidence through the with_retry wrapper from adapters/base.py.

Only markets listed in MARKET_INDICATORS have a World Bank country code +
indicator set configured — currently just "morocco", matching the one
market defined in `config/market_profiles.yaml`. A market with no
configured indicators has nothing to fetch or retry (the gap is missing
configuration, not a transient failure), so `collect()` returns the
"unavailable" AdapterResult directly for it, without ever calling the API.

wbdata's own default HTTP session sets no request timeout at all, and
(in wbdata 1.1.0) `Client(session=...)` is accepted but never actually
wired into its internal `Fetcher` — so passing a session through wbdata's
own public API doesn't fix this. DEFAULT_FETCH_FN works around it by
building a client the normal way, then swapping in a timeout-enforcing
session directly on its `.fetcher` afterward, so a slow/unresponsive World
Bank endpoint fails into with_retry's existing retry logic instead of
hanging indefinitely.
"""

from datetime import datetime, timezone
from typing import Callable

import requests
import wbdata

from adapters.base import UNAVAILABLE, AdapterResult, Evidence, with_retry

DEFAULT_TIMEOUT_SECONDS = 10

# World Bank indicator codes: https://data.worldbank.org/indicator
GDP_PER_CAPITA_USD = "NY.GDP.PCAP.CD"
INTERNET_PENETRATION_PCT = "IT.NET.USER.ZS"

# market name (lowercase, matches config/market_profiles.yaml keys) ->
# (World Bank country code, {indicator code: metric name})
MARKET_INDICATORS: dict[str, tuple[str, dict[str, str]]] = {
    "morocco": (
        "MA",
        {
            GDP_PER_CAPITA_USD: "gdp_per_capita_usd",
            INTERNET_PENETRATION_PCT: "internet_penetration_pct",
        },
    ),
}

WorldBankFetchFn = Callable[..., list[dict]]


class _TimeoutSession(requests.Session):
    """A requests.Session that always applies a timeout, since wbdata's own
    session never sets one and gives no other way to configure it."""

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT_SECONDS)
        return super().request(method, url, **kwargs)


def _build_default_fetch_fn() -> WorldBankFetchFn:
    client = wbdata.Client()
    client.fetcher.session = _TimeoutSession()
    return client.get_data


DEFAULT_FETCH_FN: WorldBankFetchFn = _build_default_fetch_fn()


def _most_recent_value(observations: list[dict]) -> dict | None:
    """wbdata returns observations most-recent-first; skip gap years (value=None)."""
    for obs in observations:
        if obs.get("value") is not None:
            return obs
    return None


@with_retry(max_attempts=3, base_delay=1.0)
def _collect_indicators(
    country_code: str, indicators: dict[str, str], fetch_fn: WorldBankFetchFn
) -> list[Evidence]:
    """Raw collection: pull each configured indicator for `country_code`.

    Wrapped by `with_retry`, so calling this returns an `AdapterResult`
    (`{"status": ..., "evidence": [...]}`), not the raw `list[Evidence]`
    this function itself produces.
    """
    retrieved_at = datetime.now(timezone.utc).isoformat()
    evidence: list[Evidence] = []

    for indicator_code, metric_name in indicators.items():
        observations = fetch_fn(indicator_code, country=country_code)
        latest = _most_recent_value(observations or [])
        source_url = (
            f"https://data.worldbank.org/indicator/{indicator_code}"
            f"?locations={country_code}"
        )

        if latest is None:
            evidence.append(
                Evidence(
                    source_type="worldbank",
                    metric=metric_name,
                    value="unknown",
                    confidence=0.1,
                    source_url=source_url,
                    raw_data={"indicator": indicator_code, "country": country_code},
                    retrieved_at=retrieved_at,
                    cost_usd=0.0,
                )
            )
            continue

        evidence.append(
            Evidence(
                source_type="worldbank",
                metric=metric_name,
                value=str(latest["value"]),
                confidence=0.8,
                source_url=source_url,
                raw_data={
                    "indicator": indicator_code,
                    "country": country_code,
                    "date": latest.get("date"),
                },
                retrieved_at=retrieved_at,
                cost_usd=0.0,
            )
        )

    return evidence


def collect(market: str, fetch_fn: WorldBankFetchFn | None = None) -> AdapterResult:
    """Market-size context evidence (GDP per capita, internet penetration, ...) for `market`.

    Looks up the World Bank country code + indicators configured for
    `market` (case-insensitive) in MARKET_INDICATORS. If nothing is
    configured for it — true for every market except "morocco" today —
    this returns the "unavailable" AdapterResult immediately, since
    there's no indicator to fetch or retry.

    `fetch_fn` accepts an injected wbdata-like callable
    (`(indicator_code, country=...) -> list[dict]`, matching
    `wbdata.get_data`'s signature), mainly so tests can pass a mock
    instead of hitting the real World Bank API.
    """
    config = MARKET_INDICATORS.get(market.lower())
    if config is None:
        return dict(UNAVAILABLE)

    country_code, indicators = config
    fn = fetch_fn or DEFAULT_FETCH_FN
    return _collect_indicators(country_code, indicators, fn)
