"""Streamlit wrapper for the local IRT chatbot.

The production website embeds this app in an iframe and passes URL parameters
such as ``lang``, ``theme``, and ``embed``. Keep this module presentation-only:
conversation behavior lives in ``AI.irt_app``.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any, Awaitable, Optional

from dotenv import find_dotenv, load_dotenv

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - allows helper tests pre-install
    st = None

from AI.models import ChatInput, ChatResponse, Conversation


SUPPORTED_LANGUAGES = {"en", "de"}
DEFAULT_LANGUAGE = "en"

INTRO_MESSAGES = {
    "en": (
        "I'm here to help you work through your nightmare and reshape it into a "
        "more empowering story. Take your time and describe the nightmare in as "
        "much detail as feels okay."
    ),
    "de": (
        "Ich bin hier, um dir zu helfen, deinen Albtraum zu bearbeiten und ihn "
        "in eine staerkendere Geschichte umzuwandeln. Nimm dir Zeit und "
        "beschreibe den Albtraum so detailliert, wie es sich fuer dich okay anfuehlt."
    ),
}

UI_TEXT = {
    "en": {
        "title": "reDreamAI IRT Chatbot",
        "caption": "Local thesis prototype with the neuro-symbolic safety layer.",
        "input": "Write your message...",
        "reset": "New session",
        "language": "Language",
        "stage": "Stage",
        "safety_toggle": "Enable safety layer",
        "safety": "Safety check",
        "safety_disabled": "Safety layer is off for this Streamlit session.",
        "safety_inactive": "Safety layer activates during rewriting.",
        "no_safety": "No safety constraint returned on the last rewriting turn.",
        "processing": "Thinking...",
        "error": "The chatbot could not process this message.",
        "disclaimer": (
            "This prototype supports Imagery Rehearsal Therapy exercises. It is "
            "not medical advice and does not replace professional care."
        ),
    },
    "de": {
        "title": "reDreamAI IRT Chatbot",
        "caption": "Lokaler Thesis-Prototyp mit neuro-symbolischer Safety Layer.",
        "input": "Schreibe deine Nachricht...",
        "reset": "Neue Sitzung",
        "language": "Sprache",
        "stage": "Phase",
        "safety_toggle": "Safety Layer aktivieren",
        "safety": "Safety Check",
        "safety_disabled": "Safety Layer ist fuer diese Streamlit-Sitzung ausgeschaltet.",
        "safety_inactive": "Die Safety Layer wird in der Umschreibphase aktiv.",
        "no_safety": "Keine Safety-Constraint in der letzten Umschreibrunde zurueckgegeben.",
        "processing": "Ich denke nach...",
        "error": "Der Chatbot konnte diese Nachricht nicht verarbeiten.",
        "disclaimer": (
            "Dieser Prototyp unterstuetzt Imagery-Rehearsal-Therapy-Uebungen. "
            "Er bietet keine medizinische Beratung und ersetzt keine professionelle Hilfe."
        ),
    },
}


def normalize_language(value: Optional[str]) -> str:
    """Return a supported two-letter language code."""
    if value in SUPPORTED_LANGUAGES:
        return value
    return DEFAULT_LANGUAGE


def parse_bool_param(value: Optional[str]) -> bool:
    """Parse common truthy query parameter values."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def run_async(coro: Awaitable[Any]) -> Any:
    """Run an async chatbot call from Streamlit's synchronous execution model."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive bridge
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")


async def process_message(
    message: str,
    conversation: Conversation,
    language: str,
    user_id: Optional[str] = None,
    safety_enabled: bool = True,
) -> ChatResponse:
    """Delegate one chat turn to the existing local chatbot pipeline."""
    from AI.irt_app import process_chat_message

    chat_input = ChatInput(
        session_id=conversation.session_id,
        user_id=user_id,
        message=message,
        language_override=language,
    )
    return await process_chat_message(
        chat_input,
        conversation,
        safety_enabled=safety_enabled,
    )


def _get_query_param(name: str, default: Optional[str] = None) -> Optional[str]:
    if not hasattr(st.query_params, "get_all"):
        value = st.query_params.get(name, default)
        if isinstance(value, list):
            return value[-1] if value else default
        return value

    values = st.query_params.get_all(name)
    if not values:
        return default
    return values[-1]


def _configure_page(embed: bool) -> None:
    st.set_page_config(
        page_title="reDreamAI Chatbot",
        page_icon="R",
        layout="wide" if embed else "centered",
        initial_sidebar_state="collapsed" if embed else "expanded",
    )


def _inject_styles(embed: bool, theme: str) -> None:
    dark = theme == "dark"
    palette = {
        "bg": "#0b1220" if dark else "#f7f9fb",
        "panel": "#111a2b" if dark else "#ffffff",
        "panel_alt": "#162033" if dark else "#eef3f7",
        "text": "#f9fafb" if dark else "#111827",
        "muted": "#c8d1df" if dark else "#4b5563",
        "subtle": "#93a4b8" if dark else "#6b7280",
        "border": "#334155" if dark else "#dde5ee",
        "accent": "#41c7b8",
        "input_bg": "#f3f6fa" if dark else "#ffffff",
        "input_text": "#111827",
    }
    max_width = "100%" if embed else "860px"
    padding_top = "0.75rem" if embed else "1.5rem"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {palette["bg"]};
            color: {palette["text"]};
        }}
        .stApp, .stApp p, .stApp label, .stApp span, .stApp div {{
            color: {palette["text"]};
        }}
        .block-container {{
            max-width: {max_width};
            padding-top: {padding_top};
            padding-bottom: 1rem;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stSidebar"] {{
            background: {palette["panel"]};
            border-right: 1px solid {palette["border"]};
        }}
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: {palette["muted"]};
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {{
            color: {palette["text"]};
        }}
        [data-testid="stSidebar"] button {{
            color: #111827;
            background: #f9fafb;
            border: 1px solid #d7dde6;
        }}
        [data-testid="stChatMessage"] {{
            background: {palette["panel_alt"]};
            border: 1px solid {palette["border"]};
            border-radius: 8px;
            padding: 0.75rem;
        }}
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div {{
            color: {palette["text"]};
        }}
        .rd-caption, .rd-disclaimer, .rd-meta {{
            color: {palette["muted"]};
        }}
        .rd-caption, .rd-disclaimer {{
            font-size: 1rem;
            line-height: 1.65;
        }}
        .rd-brand {{
            color: {palette["accent"]};
            font-weight: 700;
        }}
        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottom"] {{
            background: {palette["bg"]} !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }}
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] > div {{
            background: {palette["bg"]} !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }}
        [data-testid="stBottomBlockContainer"] {{
            max-width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }}
        [data-testid="stChatInput"] {{
            max-width: 860px;
            margin-left: auto;
            margin-right: auto;
        }}
        [data-testid="stChatInput"] {{
            background: {palette["input_bg"]};
            border: 1px solid {palette["border"]};
            border-radius: 8px;
        }}
        [data-testid="stChatInput"] textarea {{
            color: {palette["input_text"]};
            caret-color: {palette["input_text"]};
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: #6b7280;
            opacity: 1;
        }}
        [data-testid="stChatInput"] button {{
            background: #dde5ee;
            color: #111827;
        }}
        code {{
            color: #166534;
            background: #ecfdf5;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _reset_session(language: str) -> None:
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    st.session_state.conversation = Conversation(
        session_id=session_id,
        user_id=st.session_state.user_id,
        language=language,
    )
    st.session_state.last_response = None


def _ensure_state(language: str, sid: Optional[str]) -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"streamlit-{uuid.uuid4()}"

    if "session_id" not in st.session_state:
        st.session_state.session_id = sid or str(uuid.uuid4())

    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation(
            session_id=st.session_state.session_id,
            user_id=st.session_state.user_id,
            language=language,
        )
    else:
        st.session_state.conversation.language = language

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    if "safety_enabled" not in st.session_state:
        st.session_state.safety_enabled = False


def _render_sidebar(language: str, text: dict[str, str]) -> str:
    with st.sidebar:
        selected = st.radio(
            text["language"],
            options=["en", "de"],
            format_func=lambda code: "English" if code == "en" else "Deutsch",
            index=0 if language == "en" else 1,
            horizontal=True,
        )
        if selected != language:
            st.query_params["lang"] = selected
            st.session_state.conversation.language = selected
            st.rerun()

        if st.button(text["reset"], use_container_width=True):
            _reset_session(selected)
            st.rerun()

        st.toggle(
            text["safety_toggle"],
            key="safety_enabled",
            help=text["safety_inactive"],
        )

        conversation = st.session_state.conversation
        current_stage = conversation.stages[-1] if conversation.stages else "recording"
        st.markdown(f"**{text['stage']}**: `{current_stage}`")

        last_response = st.session_state.last_response
        if not st.session_state.safety_enabled:
            st.caption(text["safety_disabled"])
        elif last_response and last_response.safety:
            safety = last_response.safety
            st.markdown(f"**{text['safety']}**")
            st.json(safety, expanded=False)
        elif current_stage == "rewriting" and last_response:
            st.caption(text["no_safety"])
        else:
            st.caption(text["safety_inactive"])

    return selected


def _render_history(conversation: Conversation, language: str) -> None:
    if not conversation.messages:
        with st.chat_message("assistant"):
            st.markdown(INTRO_MESSAGES[language])
        return

    for message in conversation.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)


def main() -> None:
    """Render the Streamlit chatbot app."""
    if st is None:  # pragma: no cover - only hit without dependency installed
        raise RuntimeError("Streamlit is not installed. Run `uv sync` first.")

    load_dotenv(find_dotenv())

    language = normalize_language(_get_query_param("lang"))
    embed = parse_bool_param(_get_query_param("embed"))
    theme = (_get_query_param("theme", "dark") or "dark").lower()
    sid = _get_query_param("sid")

    _configure_page(embed)
    _inject_styles(embed=embed, theme=theme)
    _ensure_state(language=language, sid=sid)

    text = UI_TEXT[language]
    selected_language = _render_sidebar(language, text)
    text = UI_TEXT[selected_language]

    st.markdown(f"<h1><span class='rd-brand'>re</span>DreamAI</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='rd-caption'>{text['caption']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='rd-disclaimer'>{text['disclaimer']}</p>", unsafe_allow_html=True)

    conversation: Conversation = st.session_state.conversation
    _render_history(conversation, selected_language)

    user_message = st.chat_input(text["input"])
    if not user_message:
        return

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner(text["processing"]):
            try:
                response = run_async(
                    process_message(
                        user_message,
                        conversation,
                        selected_language,
                        st.session_state.user_id,
                        safety_enabled=st.session_state.safety_enabled,
                    )
                )
                st.session_state.last_response = response
                st.markdown(response.response)
            except Exception as exc:
                st.error(f"{text['error']} {exc}")


if __name__ == "__main__":
    main()
