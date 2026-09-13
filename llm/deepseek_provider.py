"""Project Meezan — DeepSeek provider (docs/FRD.md, Section 7/9, Task 9).

Thin wrapper around DeepSeek's OpenAI-compatible chat completions API
(https://api.deepseek.com). This is only the transport + bookkeeping
layer that the cheap pass, deep pass, and niche expansion (Section 7)
will later call into — no prompt content lives here.

Two responsibilities live here because DeepSeek's own behavior demands it:

1. Cumulative token-usage tracking, split into "today" (resets when the
   wall-clock date rolls over) and "this sweep" (resets on `start_sweep()`),
   since both figures feed the spend sidebar in Section 10.
2. Turning DeepSeek's 402 "Insufficient Balance" response into a clean,
   expected stop condition instead of a raised exception or a retry loop —
   an exhausted balance won't recover on its own, so retrying is pointless.
   `complete()` never retries; callers get back a result dict either way.
"""

import os
from datetime import date
from typing import Callable, Literal, TypedDict

from openai import APIStatusError, OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
INSUFFICIENT_BALANCE_STATUS = 402

BALANCE_EXHAUSTED_MESSAGE = "DeepSeek balance exhausted — top up your account to continue"

CreateFn = Callable[..., object]


class DeepSeekResult(TypedDict):
    status: Literal["ok", "balance_exhausted"]
    content: str | None
    message: str | None
    tokens_used: int


class TokenUsage:
    """Cumulative token counts, tracked "today" and "this sweep"."""

    def __init__(self) -> None:
        self._today: date = date.today()
        self.today_tokens = 0
        self.sweep_tokens = 0
        self.total_tokens = 0

    def record(self, tokens: int) -> None:
        self._roll_day_if_needed()
        self.today_tokens += tokens
        self.sweep_tokens += tokens
        self.total_tokens += tokens

    def start_sweep(self) -> None:
        self.sweep_tokens = 0

    def _roll_day_if_needed(self) -> None:
        today = date.today()
        if today != self._today:
            self._today = today
            self.today_tokens = 0


def _build_create_fn(api_key: str | None) -> CreateFn:
    client = OpenAI(api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"), base_url=DEEPSEEK_BASE_URL)
    return client.chat.completions.create


class DeepSeekProvider:
    """Wraps DeepSeek chat completions with usage tracking + a clean 402 stop.

    `create_fn` accepts an injected callable matching
    `openai.OpenAI().chat.completions.create`'s signature, mainly so tests
    can pass a mock instead of hitting real credentials/network.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        create_fn: CreateFn | None = None,
    ) -> None:
        self.model = model
        self.usage = TokenUsage()
        self._create_fn = create_fn or _build_create_fn(api_key)

    def start_sweep(self) -> None:
        self.usage.start_sweep()

    def complete(self, messages: list[dict] | str, **kwargs) -> DeepSeekResult:
        """Run one chat completion. No retries — a 402 is a stop, not a glitch.

        `messages` accepts either a chat-format message list or a plain
        string prompt, which gets wrapped as a single user message before
        being sent — the DeepSeek API rejects a bare string.

        Thinking mode is disabled on every call: DeepSeek's V4 models run
        in thinking mode by default, which (a) can spend the whole
        `max_tokens` budget on internal `reasoning_content` and return
        empty final content, and (b) silently ignores `temperature` — both
        of which every FRD prompt spec relies on not happening. A caller
        can still override this by passing its own `extra_body["thinking"]`.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        extra_body = {"thinking": {"type": "disabled"}}
        extra_body.update(kwargs.pop("extra_body", None) or {})

        try:
            response = self._create_fn(
                model=self.model, messages=messages, extra_body=extra_body, **kwargs
            )
        except APIStatusError as e:
            if e.status_code == INSUFFICIENT_BALANCE_STATUS:
                return {
                    "status": "balance_exhausted",
                    "content": None,
                    "message": BALANCE_EXHAUSTED_MESSAGE,
                    "tokens_used": 0,
                }
            raise

        tokens_used = response.usage.total_tokens if response.usage else 0
        self.usage.record(tokens_used)

        return {
            "status": "ok",
            "content": response.choices[0].message.content,
            "message": None,
            "tokens_used": tokens_used,
        }
