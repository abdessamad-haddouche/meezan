"""Shared pytest fixtures (docs/FRD.md, Section 11).

Env isolation: the app reads several env vars directly (DISABLED_ADAPTERS,
per-adapter credentials, DEEPSEEK_API_KEY). If a developer's shell has the
real .env sourced (e.g. after a manual smoke test), those real values would
otherwise leak into every test — making a test that assumes a mocked
adapter is always called (or a specific DISABLED_ADAPTERS state) pass or
fail nondeterministically depending on shell state, not on the code. The
autouse fixture below clears every one of them before each test; a test
that wants a specific value sets it itself via `monkeypatch.setenv(...)`.
"""

import pytest

# Every env var read directly by application code (grep os.environ across
# adapters/, llm/, sweep_engine.py) — keep in sync if a new one is added.
ENV_VARS_READ_BY_APP_CODE = (
    "DISABLED_ADAPTERS",
    "DEEPSEEK_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "META_ACCESS_TOKEN",
    "SERP_PROVIDER",
    "SERP_API_KEY",
    "ETSY_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_VARS_READ_BY_APP_CODE:
        monkeypatch.delenv(name, raising=False)
