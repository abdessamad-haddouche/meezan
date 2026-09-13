"""Unit tests for llm/deepseek_provider.py (docs/FRD.md, Section 7/9 / Task 9)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx2
import openai
import pytest

from llm.deepseek_provider import BALANCE_EXHAUSTED_MESSAGE, DeepSeekProvider


def _fake_response(content: str, total_tokens: int):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def _insufficient_balance_error() -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx2.Response(
        402,
        request=request,
        json={"error": {"message": "Insufficient Balance", "type": "insufficient_balance"}},
    )
    return openai.APIStatusError("Error code: 402", response=response, body=response.json())


def test_402_returns_clean_stop_message_without_raising_or_retrying():
    create_fn = MagicMock(side_effect=_insufficient_balance_error())
    provider = DeepSeekProvider(create_fn=create_fn)

    result = provider.complete(messages=[{"role": "user", "content": "hi"}])

    assert result == {
        "status": "balance_exhausted",
        "content": None,
        "message": BALANCE_EXHAUSTED_MESSAGE,
        "tokens_used": 0,
    }
    assert create_fn.call_count == 1


def test_non_402_api_status_errors_are_not_swallowed():
    request = httpx2.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx2.Response(500, request=request, json={"error": {"message": "boom"}})
    error = openai.APIStatusError("Error code: 500", response=response, body=response.json())
    create_fn = MagicMock(side_effect=error)
    provider = DeepSeekProvider(create_fn=create_fn)

    with pytest.raises(openai.APIStatusError):
        provider.complete(messages=[{"role": "user", "content": "hi"}])


def test_token_usage_accumulates_across_multiple_calls():
    create_fn = MagicMock(
        side_effect=[
            _fake_response("first", total_tokens=100),
            _fake_response("second", total_tokens=50),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    first = provider.complete(messages=[{"role": "user", "content": "a"}])
    second = provider.complete(messages=[{"role": "user", "content": "b"}])

    assert first == {"status": "ok", "content": "first", "message": None, "tokens_used": 100}
    assert second == {"status": "ok", "content": "second", "message": None, "tokens_used": 50}
    assert provider.usage.today_tokens == 150
    assert provider.usage.sweep_tokens == 150
    assert provider.usage.total_tokens == 150


def test_start_sweep_resets_sweep_tokens_but_not_today_tokens():
    create_fn = MagicMock(
        side_effect=[
            _fake_response("first", total_tokens=100),
            _fake_response("second", total_tokens=50),
        ]
    )
    provider = DeepSeekProvider(create_fn=create_fn)

    provider.complete(messages=[{"role": "user", "content": "a"}])
    provider.start_sweep()
    provider.complete(messages=[{"role": "user", "content": "b"}])

    assert provider.usage.sweep_tokens == 50
    assert provider.usage.today_tokens == 150
    assert provider.usage.total_tokens == 150


def test_string_prompt_is_wrapped_into_chat_message_list():
    create_fn = MagicMock(return_value=_fake_response("hi there", total_tokens=10))
    provider = DeepSeekProvider(create_fn=create_fn)

    provider.complete("Say hello in exactly 3 words.", max_tokens=20, temperature=0.5)

    create_fn.assert_called_once_with(
        model=provider.model,
        messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=20,
        temperature=0.5,
    )


def test_thinking_mode_is_disabled_on_every_call():
    create_fn = MagicMock(return_value=_fake_response("hi there", total_tokens=10))
    provider = DeepSeekProvider(create_fn=create_fn)

    provider.complete(messages=[{"role": "user", "content": "hi"}])

    _, call_kwargs = create_fn.call_args
    assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_caller_supplied_extra_body_is_merged_not_overwritten():
    create_fn = MagicMock(return_value=_fake_response("hi there", total_tokens=10))
    provider = DeepSeekProvider(create_fn=create_fn)

    provider.complete(messages=[{"role": "user", "content": "hi"}], extra_body={"foo": "bar"})

    _, call_kwargs = create_fn.call_args
    assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}, "foo": "bar"}


def test_balance_exhausted_call_does_not_affect_token_usage():
    create_fn = MagicMock(side_effect=_insufficient_balance_error())
    provider = DeepSeekProvider(create_fn=create_fn)

    provider.complete(messages=[{"role": "user", "content": "hi"}])

    assert provider.usage.today_tokens == 0
    assert provider.usage.sweep_tokens == 0
    assert provider.usage.total_tokens == 0
