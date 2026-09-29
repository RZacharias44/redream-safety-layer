import asyncio
import sys
import types

import pytest

pytest.importorskip("streamlit", reason="optional UI extra not installed")

from streamlit.testing.v1 import AppTest

from AI.models import ChatResponse, Conversation
from AI.streamlit_app import normalize_language, parse_bool_param, process_message, run_async


def test_normalize_language_defaults_to_english_for_unknown_values():
    assert normalize_language("de") == "de"
    assert normalize_language("en") == "en"
    assert normalize_language("fr") == "en"
    assert normalize_language(None) == "en"


def test_parse_bool_param_accepts_common_truthy_values():
    assert parse_bool_param("true") is True
    assert parse_bool_param("1") is True
    assert parse_bool_param("yes") is True
    assert parse_bool_param("false") is False
    assert parse_bool_param(None) is False


def test_run_async_works_without_existing_event_loop():
    async def sample():
        return "ok"

    assert run_async(sample()) == "ok"


def test_run_async_works_inside_existing_event_loop():
    async def sample():
        return "ok"

    async def caller():
        return run_async(sample())

    assert asyncio.run(caller()) == "ok"


def test_process_message_delegates_to_local_chatbot(monkeypatch):
    fake_irt_app = types.ModuleType("AI.irt_app")

    captured = {}

    async def fake_process_chat_message(chat_input, conversation, safety_enabled=True):
        captured["safety_enabled"] = safety_enabled
        conversation.add_message(chat_input.message, "user", language=chat_input.language_override)
        conversation.add_message("assistant response", "assistant", "recording", chat_input.language_override)
        return ChatResponse(
            session_id=chat_input.session_id,
            stage="recording",
            response="assistant response",
            stages=["recording"],
            language=chat_input.language_override,
            usage={"total": 3},
        )

    fake_irt_app.process_chat_message = fake_process_chat_message
    monkeypatch.setitem(sys.modules, "AI.irt_app", fake_irt_app)

    conversation = Conversation(session_id="session-1")
    response = asyncio.run(
        process_message(
            "hello",
            conversation,
            "en",
            "user-1",
            safety_enabled=False,
        )
    )

    assert response.response == "assistant response"
    assert response.language == "en"
    assert captured["safety_enabled"] is False
    assert conversation.messages[0].content == "hello"
    assert conversation.messages[1].stage == "recording"


def test_streamlit_app_initial_page_renders_with_iframe_params():
    app = AppTest.from_file("AI/streamlit_app.py")
    app.query_params["lang"] = "de"
    app.query_params["embed"] = "true"
    app.query_params["theme"] = "dark"

    app.run(timeout=10)

    assert not app.exception
