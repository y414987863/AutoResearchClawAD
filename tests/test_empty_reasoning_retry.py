"""Tests for the empty-response retry in ``_chat_with_prompt``.

A reasoning model counts its hidden reasoning against ``max_tokens``. On a long
prompt with a tight budget it can exhaust the ceiling before emitting a single
visible token; the provider then answers HTTP 200 with an empty body and
``finish_reason="length"``. That is not an exception, so no retry fires and the
stage falls through to a much weaker path — Stage 9 was observed collapsing a
6767-token design prompt to a 122-token generic one, losing the hypotheses, the
literature and the real compute budget before trimming 19 conditions to 8.

The retry re-asks the *same* prompt with reasoning off and double the budget,
which keeps the full context — the part that actually determines plan quality.
"""

from __future__ import annotations

from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline._helpers import _chat_with_prompt


class _ScriptedLLM:
    """Returns each queued response in turn, recording the kwargs it saw."""

    def __init__(self, *responses: LLMResponse) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append(kwargs)
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def _empty_length() -> LLMResponse:
    return LLMResponse(
        content="", model="glm-5.2", completion_tokens=8192,
        finish_reason="length", truncated=True,
    )


def _ok(content: str = "baselines:\n  - a\n") -> LLMResponse:
    return LLMResponse(content=content, model="glm-5.2", finish_reason="stop")


def test_empty_length_response_is_retried_with_reasoning_off() -> None:
    llm = _ScriptedLLM(_empty_length(), _ok())
    resp = _chat_with_prompt(llm, "sys", "user", max_tokens=8192)

    assert resp.content == "baselines:\n  - a\n"
    assert len(llm.calls) == 2
    assert llm.calls[1]["reasoning"] is False
    assert llm.calls[1]["max_tokens"] == 16384


def test_retry_resends_the_original_prompt() -> None:
    """The whole point: the fallback path this replaces threw the context away."""
    llm = _ScriptedLLM(_empty_length(), _ok())
    long_user = "Topic: X\n" + ("context line\n" * 500)
    _chat_with_prompt(llm, "system prompt", long_user, max_tokens=4096)

    assert llm.calls[1]["system"] == "system prompt"


def test_a_normal_response_does_not_trigger_a_second_call() -> None:
    llm = _ScriptedLLM(_ok("fine"))
    assert _chat_with_prompt(llm, "s", "u", max_tokens=4096).content == "fine"
    assert len(llm.calls) == 1


def test_empty_response_with_other_finish_reasons_is_left_alone() -> None:
    """Only ``length`` means "budget consumed". An empty ``stop`` is the model
    genuinely answering with nothing, and re-asking would not change that."""
    llm = _ScriptedLLM(
        LLMResponse(content="", model="m", finish_reason="stop"), _ok()
    )
    assert _chat_with_prompt(llm, "s", "u", max_tokens=4096).content == ""
    assert len(llm.calls) == 1


def test_truncated_but_non_empty_response_is_kept() -> None:
    """A partial answer still carries content the caller's parser may salvage;
    discarding it to re-ask would be a regression."""
    partial = LLMResponse(
        content="baselines:", model="m", finish_reason="length", truncated=True
    )
    llm = _ScriptedLLM(partial, _ok())
    assert _chat_with_prompt(llm, "s", "u", max_tokens=4096).content == "baselines:"
    assert len(llm.calls) == 1


def test_a_failing_retry_returns_the_original_instead_of_raising() -> None:
    """Best-effort salvage: the caller still has its own fallbacks, and turning
    this into a raise would make a recoverable stage fail outright."""

    class _Boom(_ScriptedLLM):
        def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return _empty_length()
            raise RuntimeError("all models failed")

    llm = _Boom()
    resp = _chat_with_prompt(llm, "s", "u", max_tokens=4096)
    assert resp.content == ""
    assert resp.finish_reason == "length"


def test_retry_that_is_also_empty_falls_back_to_the_original() -> None:
    llm = _ScriptedLLM(_empty_length(), _empty_length())
    resp = _chat_with_prompt(llm, "s", "u", max_tokens=4096)
    assert resp.content == ""
    assert len(llm.calls) == 2


def test_json_mode_is_preserved_on_the_retry() -> None:
    llm = _ScriptedLLM(_empty_length(), _ok('{"a": 1}'))
    _chat_with_prompt(llm, "s", "u", json_mode=True, max_tokens=8192)
    assert llm.calls[1]["json_mode"] is True


class _AcpShapedLLM:
    """Exactly ``ACPClient.chat``'s signature — no ``**kwargs`` to absorb typos.

    ``create_llm_client`` returns an ACPClient under ``provider: acp`` and
    ``executor.py`` hands it to every stage, so anything ``_chat_with_prompt``
    passes must be a parameter ACPClient actually declares. There is no ACP
    integration test, so a new keyword would otherwise only surface as a
    TypeError in a live run.
    """

    def chat(
        self,
        messages,  # noqa: ANN001
        *,
        model=None,  # noqa: ANN001
        max_tokens=None,  # noqa: ANN001
        temperature=None,  # noqa: ANN001
        json_mode=False,  # noqa: ANN001
        system=None,  # noqa: ANN001
        strip_thinking=True,  # noqa: ANN001
        reasoning=None,  # noqa: ANN001
    ):
        return LLMResponse(content="acp says hi", model="acp:x", finish_reason="stop")


def test_chat_with_prompt_stays_callable_against_the_acp_signature() -> None:
    resp = _chat_with_prompt(
        _AcpShapedLLM(), "sys", "user", json_mode=True, max_tokens=8192,
        reasoning=False,
    )
    assert resp.content == "acp says hi"


def test_the_real_acp_client_accepts_every_keyword_chat_with_prompt_sends() -> None:
    """Pins the two signatures together rather than trusting the fake above."""
    import inspect

    from researchclaw.llm.acp_client import ACPClient
    from researchclaw.llm.client import LLMClient

    acp = set(inspect.signature(ACPClient.chat).parameters)
    sent = {"messages", "system", "json_mode", "max_tokens", "strip_thinking",
            "reasoning"}
    assert sent <= acp, f"ACPClient.chat is missing {sorted(sent - acp)}"
    assert sent <= set(inspect.signature(LLMClient.chat).parameters)
