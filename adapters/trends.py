"""Project Meezan — Google Trends adapter (docs/FRD.md, Section 8).

Sweep mode: an interest-over-time signal for a niche via pytrends
(unofficial API). Returned as Evidence through the with_retry wrapper from
adapters/base.py. Discovery mode (`rising_queries()`) is Task 18 — not
built here.
"""

from datetime import datetime, timezone
from urllib.parse import quote

from pytrends.request import TrendReq

from adapters.base import Evidence, with_retry

DEFAULT_TIMEFRAME = "today 12-m"
# pytrends already defaults to this same (connect, read) timeout if
# omitted; spelled out explicitly so a slow/unresponsive endpoint can't
# hang a sweep indefinitely (it fails into with_retry's existing retry
# logic instead).
DEFAULT_TIMEOUT_SECONDS = (2, 5)


def _build_client() -> TrendReq:
    return TrendReq(hl="en-US", tz=360, timeout=DEFAULT_TIMEOUT_SECONDS)


@with_retry(max_attempts=3, base_delay=1.0)
def collect(
    niche: str, pytrends: TrendReq | None = None, timeframe: str = DEFAULT_TIMEFRAME
) -> list[Evidence]:
    """Raw collection: interest-over-time signal for `niche` via Google Trends.

    Wrapped by `with_retry`, so calling this returns an `AdapterResult`
    (`{"status": ..., "evidence": [...]}`), not the raw `list[Evidence]`
    this function itself produces.

    `pytrends` accepts an injected `TrendReq` client, mainly so tests can
    pass a mock instead of hitting the real (unofficial) API.
    """
    client = pytrends or _build_client()
    client.build_payload(kw_list=[niche], timeframe=timeframe)
    df = client.interest_over_time()

    retrieved_at = datetime.now(timezone.utc).isoformat()

    if df is None or df.empty or niche not in df.columns:
        series: list[dict] = []
        avg_interest = 0
    else:
        series = [
            {
                "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                "value": int(row[niche]),
            }
            for idx, row in df.iterrows()
        ]
        avg_interest = round(float(df[niche].mean()), 2)

    return [
        Evidence(
            source_type="google_trends",
            metric="interest_over_time",
            value=str(avg_interest),
            confidence=0.6 if series else 0.2,
            source_url=f"https://trends.google.com/trends/explore?q={quote(niche)}",
            raw_data={"timeframe": timeframe, "series": series},
            retrieved_at=retrieved_at,
            cost_usd=0.0,
        )
    ]
