"""Project Meezan — adapter interface (docs/FRD.md, Section 8).

Every data-source adapter returns evidence through this common shape.
Adapters never raise past their own boundary: `with_retry` wraps the raw
collection call in exponential-backoff retry (3 attempts) and, if every
attempt fails, returns an explicit "unavailable" result instead of letting
one dead API kill the batch.
"""

import functools
import time
from typing import Callable, Literal, TypedDict, TypeVar


class Evidence(TypedDict):
    source_type: str
    metric: str
    value: str
    confidence: float
    source_url: str | None
    raw_data: dict
    retrieved_at: str
    cost_usd: float


class AdapterResult(TypedDict):
    status: Literal["ok", "unavailable"]
    evidence: list[Evidence]


UNAVAILABLE: AdapterResult = {"status": "unavailable", "evidence": []}

_CollectFn = TypeVar("_CollectFn", bound=Callable[..., list[Evidence]])


def with_retry(
    max_attempts: int = 3, base_delay: float = 1.0
) -> Callable[[_CollectFn], Callable[..., AdapterResult]]:
    """Wrap a raw collection function in exponential-backoff retry.

    The wrapped function should return a `list[Evidence]` on success or
    raise on failure. The decorated adapter instead always returns an
    `AdapterResult`: `{"status": "ok", "evidence": [...]}` on the first
    successful attempt, or `{"status": "unavailable", "evidence": []}` once
    `max_attempts` consecutive attempts have raised.
    """

    def decorator(fn: _CollectFn) -> Callable[..., AdapterResult]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> AdapterResult:
            for attempt in range(max_attempts):
                try:
                    evidence = fn(*args, **kwargs)
                    return {"status": "ok", "evidence": evidence}
                except Exception:
                    if attempt == max_attempts - 1:
                        return dict(UNAVAILABLE)
                    time.sleep(base_delay * (2**attempt))
            return dict(UNAVAILABLE)  # unreachable; keeps type checkers happy

        return wrapper

    return decorator
