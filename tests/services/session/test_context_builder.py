"""Tests for the ContextBuilder and its helper functions."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deeptutor.services.prompt import get_prompt_manager
from deeptutor.services.prompt.language import language_label
from deeptutor.services.session.context_builder import (
    MAX_HISTORY_PLAN_TOKENS,
    MAX_RAW_REBUILD_TOKENS,
    MAX_SUMMARY_OUTPUT_TOKENS,
    ContextBuilder,
    ContextBuildResult,
    _ContextSummaryAgent,
    build_history_text,
    count_tokens,
    extract_ask_user_clarifications,
    format_messages_as_transcript,
    trim_incomplete_tail,
)
from deeptutor.services.session.sqlite_store import SQLiteSessionStore

# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_none_string(self) -> None:
        assert count_tokens(None) == 0  # type: ignore[arg-type]

    def test_nonempty_returns_positive(self) -> None:
        result = count_tokens("Hello, world!")
        assert result > 0

    def test_longer_text_more_tokens(self) -> None:
        short = count_tokens("Hi")
        long = count_tokens("Hello, this is a longer sentence with many words in it.")
        assert long > short


# ---------------------------------------------------------------------------
# format_messages_as_transcript
# ---------------------------------------------------------------------------


class TestFormatMessagesAsTranscript:
    def test_basic_transcript(self) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = format_messages_as_transcript(messages)
        assert "User: Hi" in result
        assert "Assistant: Hello!" in result

    def test_system_role_mapped(self) -> None:
        messages = [{"role": "system", "content": "You are helpful."}]
        result = format_messages_as_transcript(messages)
        assert "System: You are helpful." in result

    def test_empty_content_skipped(self) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Again"},
        ]
        result = format_messages_as_transcript(messages)
        assert "User: Hi" in result
        assert "User: Again" in result
        assert "Assistant:" not in result

    def test_empty_list(self) -> None:
        assert format_messages_as_transcript([]) == ""

    def test_unknown_role_defaults_to_user(self) -> None:
        messages = [{"role": "tool", "content": "data"}]
        result = format_messages_as_transcript(messages)
        assert "User: data" in result


# ---------------------------------------------------------------------------
# build_history_text
# ---------------------------------------------------------------------------


class TestBuildHistoryText:
    def test_system_becomes_summary(self) -> None:
        messages = [{"role": "system", "content": "Summary here"}]
        result = build_history_text(messages)
        assert "Conversation summary:" in result
        assert "Summary here" in result

    def test_user_and_assistant_roles(self) -> None:
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        result = build_history_text(messages)
        assert "User: Q" in result
        assert "Assistant: A" in result

    def test_empty_content_skipped(self) -> None:
        messages = [{"role": "user", "content": ""}]
        assert build_history_text(messages) == ""


# ---------------------------------------------------------------------------
# ContextBuildResult
# ---------------------------------------------------------------------------


class TestContextBuildResult:
    def test_dataclass_fields(self) -> None:
        r = ContextBuildResult(
            conversation_history=[],
            conversation_summary="",
            context_text="",
            events=[],
            token_count=0,
            budget=1024,
        )
        assert r.budget == 1024
        assert r.token_count == 0


# ---------------------------------------------------------------------------
# ContextBuilder budget helpers
# ---------------------------------------------------------------------------


class TestContextBuilderBudgets:
    def _make_llm_config(
        self,
        max_tokens: int,
        *,
        model: str = "test-model",
        context_window: int | None = None,
    ) -> MagicMock:
        cfg = MagicMock()
        cfg.max_tokens = max_tokens
        cfg.model = model
        cfg.context_window = context_window
        return cfg

    def test_history_budget_uses_explicit_context_window(self) -> None:
        builder = ContextBuilder(store=MagicMock(), history_budget_ratio=0.35)
        budget = builder._history_budget(self._make_llm_config(4096, context_window=128000))
        assert budget == int(128000 * 0.35)

    def test_history_budget_uses_large_context_model_heuristic(self) -> None:
        builder = ContextBuilder(store=MagicMock(), history_budget_ratio=0.35)
        budget = builder._history_budget(self._make_llm_config(4096, model="gpt-4o-mini"))
        assert budget == int(128000 * 0.35)

    def test_history_budget_has_absolute_cap(self) -> None:
        builder = ContextBuilder(store=MagicMock(), history_budget_ratio=0.35)
        budget = builder._history_budget(self._make_llm_config(4096, context_window=983_616))
        assert budget == MAX_HISTORY_PLAN_TOKENS

    def test_history_budget_minimum(self) -> None:
        builder = ContextBuilder(store=MagicMock(), history_budget_ratio=0.01)
        budget = builder._history_budget(self._make_llm_config(100, model="unknown-local-model"))
        assert budget >= 256

    def test_summary_budget(self) -> None:
        builder = ContextBuilder(store=MagicMock(), summary_target_ratio=0.40)
        assert builder._summary_budget(1000) == 400

    def test_summary_budget_minimum(self) -> None:
        builder = ContextBuilder(store=MagicMock(), summary_target_ratio=0.01)
        assert builder._summary_budget(100) >= 96

    def test_summary_budget_has_absolute_cap(self) -> None:
        builder = ContextBuilder(store=MagicMock(), summary_target_ratio=0.40)
        assert builder._summary_budget(MAX_HISTORY_PLAN_TOKENS) == (MAX_SUMMARY_OUTPUT_TOKENS)

    def test_summary_budget_respects_configured_output_limit(self) -> None:
        builder = ContextBuilder(store=MagicMock(), summary_target_ratio=0.40)
        cfg = self._make_llm_config(4096, context_window=983_616)
        assert builder._summary_budget(MAX_HISTORY_PLAN_TOKENS, cfg) == 4096

    def test_summary_budget_can_follow_small_generation_limit(self) -> None:
        builder = ContextBuilder(store=MagicMock(), summary_target_ratio=0.40)
        cfg = self._make_llm_config(50, context_window=983_616)
        assert builder._summary_budget(MAX_HISTORY_PLAN_TOKENS, cfg) == 50

    def test_recent_budget(self) -> None:
        builder = ContextBuilder(store=MagicMock(), summary_target_ratio=0.40)
        recent = builder._recent_budget(1000)
        assert recent == 1000 - 400

    def test_recent_budget_minimum(self) -> None:
        builder = ContextBuilder(store=MagicMock(), summary_target_ratio=0.99)
        assert builder._recent_budget(200) >= 128

    def test_recent_budget_preserves_headroom_when_summary_is_capped(self) -> None:
        builder = ContextBuilder(store=MagicMock(), summary_target_ratio=0.40)
        cfg = self._make_llm_config(4096, context_window=983_616)
        recent_budget = builder._recent_budget(MAX_HISTORY_PLAN_TOKENS)
        summary_budget = builder._summary_budget(MAX_HISTORY_PLAN_TOKENS, cfg)

        assert recent_budget == int(MAX_HISTORY_PLAN_TOKENS * 0.60)
        assert recent_budget + summary_budget < MAX_HISTORY_PLAN_TOKENS

    def test_rebuild_source_budget_has_absolute_cap(self) -> None:
        builder = ContextBuilder(store=MagicMock())
        cfg = self._make_llm_config(4096, context_window=983_616)
        assert builder._rebuild_source_budget(cfg) == MAX_RAW_REBUILD_TOKENS


# ---------------------------------------------------------------------------
# ContextBuilder._build_history
# ---------------------------------------------------------------------------


class TestBuildHistory:
    def test_with_summary(self) -> None:
        builder = ContextBuilder(store=MagicMock())
        history = builder._build_history(
            "A summary",
            [{"role": "user", "content": "Q"}],
        )
        assert history[0] == {"role": "system", "content": "A summary"}
        assert history[1] == {"role": "user", "content": "Q"}

    def test_empty_summary_skipped(self) -> None:
        builder = ContextBuilder(store=MagicMock())
        history = builder._build_history(
            "",
            [{"role": "user", "content": "Q"}],
        )
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_system_messages_filtered(self) -> None:
        builder = ContextBuilder(store=MagicMock())
        history = builder._build_history(
            "",
            [
                {"role": "system", "content": "Instructions"},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_empty_content_filtered(self) -> None:
        builder = ContextBuilder(store=MagicMock())
        history = builder._build_history(
            "",
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "answer"},
            ],
        )
        assert len(history) == 1

    def test_resolved_ask_user_answers_are_rehydrated_as_user_context(self) -> None:
        before = "I can help, but one detail matters."
        after = "I adapted the explanation."
        message = {
            "role": "assistant",
            "content": before + after,
            "metadata": {"provider_response_state": {"reasoning_content": "private reasoning"}},
            "events": [
                {
                    "type": "tool_result",
                    "metadata": {
                        "tool_metadata": {
                            "ask_user": {
                                "questions": [
                                    {"id": "level", "prompt": "What have you studied?"},
                                    {"id": "goal", "prompt": "What is your goal?"},
                                ]
                            }
                        }
                    },
                },
                {
                    "type": "progress",
                    "metadata": {
                        "ask_user_resolved": True,
                        "assistant_content_offset": len(before),
                        "answers": [
                            {"questionId": "level", "text": "High-school calculus"},
                            {"questionId": "goal", "text": "Understand the intuition"},
                        ],
                    },
                },
            ],
        }

        clarification = extract_ask_user_clarifications(message)
        assert "What have you studied?" in clarification
        assert "High-school calculus" in clarification

        history = ContextBuilder(store=MagicMock())._build_history("", [message])
        assert history == [
            {"role": "assistant", "content": before},
            {"role": "user", "content": clarification},
            {
                "role": "assistant",
                "content": after,
                "_provider_response_state": {"reasoning_content": "private reasoning"},
            },
        ]
        transcript = format_messages_as_transcript([message])
        assert transcript.index(before) < transcript.index(clarification) < transcript.index(after)


# ---------------------------------------------------------------------------
# ContextBuilder._select_recent_messages
# ---------------------------------------------------------------------------


class TestSelectRecentMessages:
    def test_selects_from_end(self) -> None:
        builder = ContextBuilder(store=MagicMock())
        messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "new"},
        ]
        older, recent = builder._select_recent_messages(messages, recent_budget=9999)
        assert len(recent) == 3
        assert len(older) == 0

    def test_budget_limits_selection(self) -> None:
        builder = ContextBuilder(store=MagicMock())
        messages = [
            {"role": "user", "content": "A" * 200},
            {"role": "user", "content": "B" * 200},
            {"role": "user", "content": "C" * 200},
        ]
        older, recent = builder._select_recent_messages(messages, recent_budget=10)
        assert len(recent) >= 1
        assert len(older) + len(recent) == len(messages)

    def test_budget_includes_private_provider_response_state(self) -> None:
        builder = ContextBuilder(store=MagicMock())
        messages = [
            {
                "role": "assistant",
                "content": "short answer",
                "metadata": {"provider_response_state": {"reasoning_content": "x" * 1000}},
            },
            {"role": "user", "content": "new question"},
        ]

        older, recent = builder._select_recent_messages(messages, recent_budget=100)

        assert older == messages[:1]
        assert recent == messages[1:]


# ---------------------------------------------------------------------------
# ContextBuilder.build — with mocked store
# ---------------------------------------------------------------------------


class TestContextBuilderBuild:
    @pytest.mark.asyncio
    async def test_missing_session_returns_empty(self) -> None:
        store = AsyncMock()
        store.get_session = AsyncMock(return_value=None)
        store.get_messages_for_context = AsyncMock(return_value=[])

        builder = ContextBuilder(store=store)
        cfg = MagicMock()
        cfg.max_tokens = 4096

        result = await builder.build(session_id="missing", llm_config=cfg)
        assert result.conversation_history == []
        assert result.context_text == ""

    @pytest.mark.asyncio
    async def test_within_budget_no_summarize(self) -> None:
        store = AsyncMock()
        store.get_session = AsyncMock(
            return_value={
                "id": "s1",
                "compressed_summary": "",
                "summary_up_to_msg_id": 0,
            }
        )
        store.get_messages_for_context = AsyncMock(
            return_value=[
                {"id": 1, "role": "user", "content": "Hi"},
                {"id": 2, "role": "assistant", "content": "Hello!"},
            ]
        )

        builder = ContextBuilder(store=store)
        cfg = MagicMock()
        cfg.max_tokens = 4096

        result = await builder.build(session_id="s1", llm_config=cfg)
        assert len(result.conversation_history) == 2
        assert result.events == []
        assert result.token_count > 0

    @pytest.mark.asyncio
    async def test_large_context_model_avoids_premature_summarize(self) -> None:
        long_user = "user says " + ("alpha " * 1400)
        long_assistant = "assistant says " + ("beta " * 1400)
        store = AsyncMock()
        store.get_session = AsyncMock(
            return_value={
                "id": "s1",
                "compressed_summary": "",
                "summary_up_to_msg_id": 0,
            }
        )
        store.get_messages_for_context = AsyncMock(
            return_value=[
                {"id": 1, "role": "user", "content": long_user},
                {"id": 2, "role": "assistant", "content": long_assistant},
            ]
        )

        builder = ContextBuilder(store=store)
        cfg = MagicMock()
        cfg.max_tokens = 4096
        cfg.model = "gpt-4o-mini"
        cfg.context_window = None

        result = await builder.build(session_id="s1", llm_config=cfg)
        assert len(result.conversation_history) == 2
        assert result.events == []
        store.update_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_context_build_uses_capped_plan_with_headroom(self) -> None:
        store = AsyncMock()
        store.get_session = AsyncMock(
            return_value={
                "id": "s1",
                "compressed_summary": "",
                "summary_up_to_msg_id": 0,
            }
        )
        store.get_messages_for_context = AsyncMock(
            return_value=[
                {"id": 1, "role": "user", "content": "old turn"},
                {"id": 2, "role": "assistant", "content": "recent turn"},
            ]
        )

        builder = ContextBuilder(store=store)
        builder._summarize = AsyncMock(return_value=("SUMMARY", []))
        select_recent = MagicMock(wraps=builder._select_recent_messages)
        builder._select_recent_messages = select_recent
        cfg = MagicMock()
        cfg.max_tokens = 4096
        cfg.model = "qwen3.8-max"
        cfg.context_window = 983_616

        def _fake_count_tokens(text: str) -> int:
            if text in {"old turn", "recent turn"}:
                return 70_000
            if "User: old turn" in text and "Assistant: recent turn" in text:
                return 140_000
            return 100

        with patch(
            "deeptutor.services.session.context_builder.count_tokens",
            side_effect=_fake_count_tokens,
        ):
            result = await builder.build(session_id="s1", llm_config=cfg)

        assert result.budget == MAX_HISTORY_PLAN_TOKENS
        assert builder._summarize.call_args.kwargs["summary_budget"] == 4096
        assert select_recent.call_args.args[1] == int(MAX_HISTORY_PLAN_TOKENS * 0.60)
        store.update_summary.assert_awaited_once_with("s1", "SUMMARY", 1)


# ---------------------------------------------------------------------------
# trim_incomplete_tail
# ---------------------------------------------------------------------------


class TestTrimIncompleteTail:
    def test_drops_trailing_partial_line(self) -> None:
        assert trim_incomplete_tail("- done line\n- cut mid sent") == "- done line"

    def test_single_line_kept(self) -> None:
        assert trim_incomplete_tail("only one line, keep it") == "only one line, keep it"


# ---------------------------------------------------------------------------
# ContextBuilder._summarize
# ---------------------------------------------------------------------------


class TestContextBuilderSummarize:
    @pytest.mark.asyncio
    async def test_target_stays_below_small_output_cap(self) -> None:
        captured: dict[str, Any] = {}

        async def _stream_llm(**kwargs: Any):
            captured.update(kwargs)
            yield "short summary"

        agent = MagicMock()
        agent.stream_llm = _stream_llm
        builder = ContextBuilder(store=MagicMock())

        with patch(
            "deeptutor.services.session.context_builder._ContextSummaryAgent",
            return_value=agent,
        ):
            summary, _events = await builder._summarize(
                session_id="s1",
                language="en",
                source_text="User: hello",
                summary_budget=50,
            )

        assert summary == "short summary"
        assert captured["max_tokens"] == 50
        assert "under 40 tokens" in captured["user_prompt"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("language", "expected"),
        [
            ("ja", "Write the summary itself in 日本語."),
            ("zh", "摘要正文本身请使用中文（简体）撰写。"),
        ],
    )
    async def test_summary_instruction_names_the_session_language(
        self, language: str, expected: str
    ) -> None:
        captured: dict[str, Any] = {}

        async def _stream_llm(**kwargs: Any):
            captured.update(kwargs)
            yield "short summary"

        agent = MagicMock()
        agent.stream_llm = _stream_llm
        builder = ContextBuilder(store=MagicMock())

        with patch(
            "deeptutor.services.session.context_builder._ContextSummaryAgent",
            return_value=agent,
        ):
            await builder._summarize(
                session_id="s1",
                language=language,
                source_text="User: hello",
                summary_budget=50,
            )

        # The summary is replayed as a system row on every later turn, so an
        # instruction that never says which language to write in drags the
        # answer back to the prompt's own language (#1511).
        assert expected in captured["system_prompt"]


# ---------------------------------------------------------------------------
# ContextBuilder.build — summarize paths (rebuild / fold-in / failure / branch)
# ---------------------------------------------------------------------------


def _make_store(
    messages: list[dict[str, Any]],
    *,
    summary: str = "",
    up_to: int = 0,
) -> AsyncMock:
    store = AsyncMock()
    store.get_session = AsyncMock(
        return_value={
            "id": "s1",
            "compressed_summary": summary,
            "summary_up_to_msg_id": up_to,
        }
    )
    store.get_messages_for_context = AsyncMock(return_value=messages)
    return store


def _small_window_cfg() -> MagicMock:
    # context_window=1000 -> history budget 350, summary budget 140,
    # recent budget 210, raw-rebuild budget floor 1024.
    cfg = MagicMock()
    cfg.max_tokens = 512
    cfg.model = "test-model"
    cfg.context_window = 1000
    return cfg


class TestContextBuilderSummarizePaths:
    @pytest.mark.asyncio
    async def test_rebuilds_from_raw_when_prefix_fits(self) -> None:
        messages = [
            {"id": 1, "role": "user", "content": "RAW_OLDEST_MARKER " + "alpha " * 90},
            {"id": 2, "role": "assistant", "content": "beta " * 90},
            {"id": 3, "role": "user", "content": "gamma " * 200},
            {"id": 4, "role": "assistant", "content": "delta " * 400},
            {"id": 5, "role": "user", "content": "recent question"},
            {"id": 6, "role": "assistant", "content": "recent answer"},
        ]
        store = _make_store(messages, summary="OLD SUMMARY", up_to=2)
        builder = ContextBuilder(store=store)
        builder._summarize = AsyncMock(return_value=("NEW SUMMARY", []))

        result = await builder.build(session_id="s1", llm_config=_small_window_cfg())

        source = builder._summarize.call_args.kwargs["source_text"]
        # Raw prefix fits the rebuild budget: source is the original
        # transcript (anti-drift), not summary-of-summary fold-in.
        assert "Conversation history to summarize:" in source
        assert "RAW_OLDEST_MARKER" in source
        assert "Existing summary:" not in source
        store.update_summary.assert_awaited_once_with("s1", "NEW SUMMARY", 4)
        assert result.conversation_summary == "NEW SUMMARY"
        assert result.conversation_history[0] == {
            "role": "system",
            "content": "NEW SUMMARY",
        }

    @pytest.mark.asyncio
    async def test_folds_in_when_prefix_exceeds_rebuild_budget(self) -> None:
        messages = [
            {"id": 1, "role": "user", "content": "RAW_OLDEST_MARKER " + "w1 " * 600},
            {"id": 2, "role": "assistant", "content": "w2 " * 600},
            {"id": 3, "role": "user", "content": "FOLD_MARKER " + "w3 " * 400},
            {"id": 4, "role": "assistant", "content": "recent answer"},
        ]
        store = _make_store(messages, summary="OLD SUMMARY", up_to=2)
        builder = ContextBuilder(store=store)
        builder._summarize = AsyncMock(return_value=("NEW SUMMARY", []))

        await builder.build(session_id="s1", llm_config=_small_window_cfg())

        source = builder._summarize.call_args.kwargs["source_text"]
        # Prefix transcript (>1024 tokens) exceeds the rebuild budget:
        # degrade to folding the stored summary plus older turns only.
        assert "Existing summary:\nOLD SUMMARY" in source
        assert "FOLD_MARKER" in source
        assert "RAW_OLDEST_MARKER" not in source
        store.update_summary.assert_awaited_once_with("s1", "NEW SUMMARY", 3)

    @pytest.mark.asyncio
    async def test_failure_advances_fallback_watermark_and_keeps_recent_turn(self) -> None:
        messages = [
            {"id": 1, "role": "user", "content": "old " * 100},
            {"id": 2, "role": "assistant", "content": "older reply " * 100},
            {"id": 3, "role": "user", "content": "mid " * 400},
            {"id": 4, "role": "assistant", "content": "recent answer"},
        ]
        store = _make_store(messages, summary="OLD SUMMARY", up_to=2)
        builder = ContextBuilder(store=store)
        builder._summarize = AsyncMock(side_effect=RuntimeError("llm down"))

        result = await builder.build(session_id="s1", llm_config=_small_window_cfg())

        # The failed turn is not described as summarized. The watermark moves
        # past the dropped prefix so the next build does not resend it, while
        # the recent turn and the previous summary stay available.
        store.update_summary.assert_awaited_once_with("s1", "OLD SUMMARY", 3)
        assert result.token_count <= result.budget
        assert result.conversation_summary == "OLD SUMMARY"
        assert result.conversation_history[0] == {
            "role": "system",
            "content": "OLD SUMMARY",
        }
        contents = [m["content"] for m in result.conversation_history]
        assert "recent answer" in contents
        assert "mid " * 400 not in contents

    @pytest.mark.asyncio
    async def test_branch_switch_discards_sibling_summary(self) -> None:
        # Watermark 99 is not on this branch's ancestor chain: the stored
        # summary was folded from a sibling branch and must not leak in.
        messages = [
            {"id": 1, "role": "user", "content": "Hi"},
            {"id": 2, "role": "assistant", "content": "Hello!"},
        ]
        store = _make_store(messages, summary="SIBLING SUMMARY", up_to=99)
        builder = ContextBuilder(store=store)
        builder._summarize = AsyncMock(return_value=("", []))

        result = await builder.build(session_id="s1", llm_config=_small_window_cfg())

        assert result.conversation_summary == ""
        assert all(m["role"] != "system" for m in result.conversation_history)
        store.update_summary.assert_not_called()


async def _seed_compaction_session(
    store: SQLiteSessionStore,
) -> tuple[str, int, list[dict[str, Any]]]:
    session = await store.create_session()
    session_id = session["id"]
    await store.add_message(session_id, "user", "Original question [doc:7]")
    watermark = await store.add_message(session_id, "assistant", "Original answer")
    await store.update_summary(session_id, "PRIOR SUMMARY [doc:7]", watermark)
    await store.add_message(session_id, "user", "RETRY_MARKER [doc:9] " + "material " * 500)
    await store.add_message(
        session_id,
        "assistant",
        "Recent answer",
        metadata={"provider_response_state": {"reasoning_content": "private replay state"}},
    )
    return session_id, watermark, await store.get_messages_for_context(session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh", "ja", "zh-tw"])
async def test_compaction_uses_discovered_prompt_and_preserves_sources_and_private_state(
    sqlite_store: SQLiteSessionStore,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_id, _watermark, original = await _seed_compaction_session(sqlite_store)
    captured: dict[str, Any] = {}
    summary = "# Goals\nRetain the explanation [doc:7] [doc:9]."
    monkeypatch.setattr("deeptutor.agents.base_agent.get_agent_params", lambda _module: {})

    async def stream(_self: _ContextSummaryAgent, **kwargs: Any) -> AsyncIterator[str]:
        captured.update(kwargs)
        yield f"<think>private drafting notes</think>\n{summary}"

    monkeypatch.setattr(_ContextSummaryAgent, "stream_llm", stream)
    manager = get_prompt_manager()
    manager.clear_cache()
    result = await ContextBuilder(sqlite_store).build(
        session_id=session_id, llm_config=_small_window_cfg(), language=language
    )

    prompt = manager.load_prompts("chat", "context_summary_agent", language)["system"]
    assert (
        captured["system_prompt"]
        == prompt.format(summary_language=language_label(language)).strip()
    )
    assert language_label(language) in captured["system_prompt"]
    assert "[doc:7]" in captured["user_prompt"]
    assert "[doc:9]" in captured["user_prompt"]
    assert "private replay state" not in captured["user_prompt"]
    assert result.conversation_summary == summary
    assert result.conversation_history[-1]["_provider_response_state"] == {
        "reasoning_content": "private replay state"
    }
    session = await sqlite_store.get_session(session_id)
    assert session["compressed_summary"] == summary
    assert session["summary_up_to_msg_id"] == original[-2]["id"]
    assert await sqlite_store.get_messages_for_context(session_id) == original
    assert "No prompt file found" not in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["missing_prompt", "empty", "reasoning_only", "exception", "partial_error"]
)
async def test_failed_compaction_preserves_durable_history_and_bounds_prefix(
    sqlite_store: SQLiteSessionStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    session_id, watermark, original = await _seed_compaction_session(sqlite_store)
    monkeypatch.setattr("deeptutor.agents.base_agent.get_agent_params", lambda _module: {})
    manager = get_prompt_manager()
    manager.clear_cache()

    async def stream(_self: _ContextSummaryAgent, **kwargs: Any) -> AsyncIterator[str]:
        if failure == "missing_prompt":
            pytest.fail("A missing summary prompt must prevent the request")
        if failure == "exception":
            yield "partial summary"
            raise RuntimeError("stream interrupted")
        if failure == "partial_error":
            yield "partial summary"
            kwargs["outcome"].finish_reason = "error"
        elif failure == "reasoning_only":
            yield "<think>private drafting notes</think>"
        else:
            yield " \n "

    with monkeypatch.context() as missing:
        missing.setattr(_ContextSummaryAgent, "stream_llm", stream)
        if failure == "missing_prompt":
            missing.setattr(manager, "_candidate_prompt_dirs", lambda _module: [tmp_path])
        result = await ContextBuilder(sqlite_store).build(
            session_id=session_id, llm_config=_small_window_cfg()
        )

    session = await sqlite_store.get_session(session_id)
    assert session["compressed_summary"] == "PRIOR SUMMARY [doc:7]"
    # The oversized unsummarized user turn is behind a hard-truncation
    # watermark. Its text stays in the store, but the next build will not
    # send it to the summarizer again.
    assert session["summary_up_to_msg_id"] == original[-2]["id"]
    assert session["summary_up_to_msg_id"] != watermark
    assert result.token_count <= result.budget
    assert result.conversation_summary == "PRIOR SUMMARY [doc:7]"
    assert result.conversation_history[-1]["content"] == "Recent answer"
    assert result.conversation_history[-1]["_provider_response_state"] == {
        "reasoning_content": "private replay state"
    }
    assert "RETRY_MARKER" not in result.context_text
    assert await sqlite_store.get_messages_for_context(session_id) == original

    calls: list[str] = []

    async def retry(_self: _ContextSummaryAgent, **kwargs: Any) -> AsyncIterator[str]:
        calls.append(str(kwargs.get("user_prompt") or ""))
        yield "Recovered summary [doc:7] [doc:9]"

    monkeypatch.setattr(_ContextSummaryAgent, "stream_llm", retry)
    retried = await ContextBuilder(sqlite_store).build(
        session_id=session_id, llm_config=_small_window_cfg()
    )
    assert calls == []
    assert retried.token_count <= retried.budget
    assert "RETRY_MARKER" not in retried.context_text
    assert retried.conversation_history[-1]["content"] == "Recent answer"
    session = await sqlite_store.get_session(session_id)
    assert session["compressed_summary"] == "PRIOR SUMMARY [doc:7]"
    assert session["summary_up_to_msg_id"] == original[-2]["id"]
    assert await sqlite_store.get_messages_for_context(session_id) == original


def _assert_tool_pairs(messages: list[dict[str, Any]]) -> None:
    pending: list[str] = []
    for message in messages:
        if pending and message.get("role") != "tool":
            raise AssertionError(f"tool calls closed out of order: {pending}")
        for call in message.get("tool_calls") or []:
            pending.append(call["id"])
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            assert call_id in pending
            pending.remove(call_id)
    assert pending == []


def _large_window_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model="synthetic-model",
        binding="openai",
        context_window=1_048_576,
        max_tokens=4096,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("succeeds", [False, True])
async def test_single_oversized_group_stays_within_history_budget(
    sqlite_store: SQLiteSessionStore,
    succeeds: bool,
) -> None:
    """A retained group larger than the plan cannot survive, even after a summary."""
    session = await sqlite_store.create_session()
    session_id = session["id"]
    huge = "alpha " * 150_000
    for role, text in (
        ("user", "Earlier question"),
        ("assistant", "Earlier answer"),
        ("user", "Latest question"),
        ("assistant", huge),
    ):
        await sqlite_store.add_message(session_id, role, text)
    original = await sqlite_store.get_messages_for_context(session_id)
    builder = ContextBuilder(sqlite_store)
    summary = (
        AsyncMock(return_value=("Small successful summary", []))
        if succeeds
        else AsyncMock(side_effect=ValueError("fixture"))
    )
    with patch.object(builder, "_summarize", summary):
        result = await builder.build(session_id=session_id, llm_config=_large_window_cfg())

    assert result.budget == MAX_HISTORY_PLAN_TOKENS
    assert result.token_count <= result.budget
    assert huge not in result.context_text
    assert huge not in result.conversation_summary
    assert huge not in json.dumps(result.model_history)
    if not succeeds:
        assert result.conversation_summary.startswith("[Earlier conversation was truncated")
    assert await sqlite_store.get_messages_for_context(session_id) == original
    stored = await sqlite_store.get_session(session_id)
    assert stored["summary_up_to_msg_id"] not in (0, None, "")

    again = (
        AsyncMock(return_value=("Small successful summary", []))
        if succeeds
        else AsyncMock(side_effect=ValueError("fixture"))
    )
    follow_up = ContextBuilder(sqlite_store)
    with patch.object(follow_up, "_summarize", again):
        second = await follow_up.build(session_id=session_id, llm_config=_large_window_cfg())
    assert second.token_count <= second.budget
    assert huge not in second.context_text
    assert huge not in second.conversation_summary
    assert huge not in json.dumps(second.model_history)
    assert await sqlite_store.get_messages_for_context(session_id) == original
    if not succeeds:
        again.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("succeeds", [False, True])
async def test_oversized_model_turn_stays_within_budget_and_keeps_tool_pairs(
    sqlite_store: SQLiteSessionStore,
    succeeds: bool,
) -> None:
    from tests.services.session.test_model_history import turn_record

    session = await sqlite_store.create_session()
    session_id = session["id"]
    huge = "evidence " * 160_000
    await sqlite_store.add_message(session_id, "user", "Earlier question")
    await sqlite_store.add_message(session_id, "assistant", "Earlier answer")
    await sqlite_store.add_message(session_id, "user", "Latest question")
    await sqlite_store.add_message(
        session_id,
        "assistant",
        "Displayed answer",
        metadata={"model_turn": turn_record(huge)},
    )
    original = await sqlite_store.get_messages_for_context(session_id)
    builder = ContextBuilder(sqlite_store)
    summary = (
        AsyncMock(return_value=("Small successful summary", []))
        if succeeds
        else AsyncMock(side_effect=ValueError("fixture"))
    )
    with patch.object(builder, "_summarize", summary):
        result = await builder.build(session_id=session_id, llm_config=_large_window_cfg())

    assert result.budget == MAX_HISTORY_PLAN_TOKENS
    assert result.token_count <= result.budget
    _assert_tool_pairs(result.model_history or [])
    assert huge not in json.dumps(result.model_history)
    assert huge not in result.conversation_summary
    assert await sqlite_store.get_messages_for_context(session_id) == original

    again = (
        AsyncMock(return_value=("Small successful summary", []))
        if succeeds
        else AsyncMock(side_effect=ValueError("fixture"))
    )
    follow_up = ContextBuilder(sqlite_store)
    with patch.object(follow_up, "_summarize", again):
        second = await follow_up.build(session_id=session_id, llm_config=_large_window_cfg())
    assert second.token_count <= second.budget
    _assert_tool_pairs(second.model_history or [])
    assert huge not in json.dumps(second.model_history)
    assert await sqlite_store.get_messages_for_context(session_id) == original
    if not succeeds:
        again.assert_not_awaited()
