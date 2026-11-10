"""Minimal Streamlit presentation for locally grounded PDF chat."""

import os
import re
from html import escape

import requests
import streamlit as st

from src.llm.base import LLMError
from src.ui_service import MAX_UPLOAD_MB, StudySession


def _session() -> StudySession:
    """Reuse the local vector index for the current browser session."""
    if "study_session" not in st.session_state:
        st.session_state.study_session = StudySession()
    return st.session_state.study_session


def _styles() -> None:
    """Apply a consistent, accessible dark reading-focused design."""
    st.markdown(
        """
        <style>
        :root {
          --sr-bg: #0b1020;
          --sr-panel: #121b2e;
          --sr-border: rgba(162, 178, 218, .17);
          --sr-muted: #a4b1ca;
          --sr-ink: #f4f7ff;
          --sr-accent: #9a88ff;
        }
        .stApp {
          background:
            radial-gradient(ellipse 65% 33% at 52% -12%, rgba(114, 97, 221, .19), transparent 75%),
            #0b1020;
          color: var(--sr-ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container {
          max-width: 1020px;
          padding: 2.15rem 2.5rem 7rem;
        }
        section[data-testid="stSidebar"] {
          background: #101829;
          border-right: 1px solid var(--sr-border);
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
          gap: .75rem;
        }
        .sr-brand {
          display: flex;
          align-items: center;
          gap: .75rem;
          padding-bottom: 1rem;
          margin-bottom: .15rem;
          border-bottom: 1px solid var(--sr-border);
        }
        .sr-logo {
          display: inline-flex;
          width: 42px;
          height: 42px;
          border-radius: 14px;
          align-items: center;
          justify-content: center;
          font-size: 23px;
          background: linear-gradient(145deg, #7461e8, #423d9c);
          box-shadow: 0 10px 24px #070b1b88;
        }
        .sr-brand strong { color: #fafaff; font-size: 1.22rem; letter-spacing: -.03em; }
        .sr-brand small { display: block; color: var(--sr-muted); font-size: .73rem; }
        .sr-label {
          font-weight: 650;
          color: #dce5fc;
          font-size: .8rem;
          letter-spacing: .025em;
          margin: .7rem 0 .2rem;
        }
        .sr-status {
          display: inline-flex;
          gap: .5rem;
          align-items: center;
          color: #bec9dc;
          font-size: .81rem;
          border-radius: 12px;
          padding: .62rem .75rem;
          border: 1px solid var(--sr-border);
          background: rgba(255,255,255,.025);
          margin: .3rem 0;
          width: 100%;
        }
        .sr-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
        .sr-online { background: #4ad6ae; box-shadow: 0 0 0 3px #4ad6ae1b; }
        .sr-offline { background: #f5ad6a; box-shadow: 0 0 0 3px #f5ad6a1b; }
        .sr-stat {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: .76rem .82rem;
          background: rgba(255,255,255,.026);
          border: 1px solid var(--sr-border);
          border-radius: 12px;
          font-size: .86rem;
          color: #b7c5dc;
        }
        .sr-stat strong { font-size: .97rem; color: #f8f8ff; }
        .sr-heading { margin: .8rem 0 .25rem; }
        .sr-heading h1 {
          font-size: clamp(2rem, 4vw, 2.9rem);
          color: #f6f7ff;
          letter-spacing: -.045em;
          line-height: 1.1;
          font-weight: 760;
          margin: 0 0 .7rem;
        }
        .sr-heading p {
          color: #aebbd1;
          font-size: .98rem;
          line-height: 1.7;
          margin: 0 0 1.45rem;
        }
        .sr-eyebrow {
          font-size: .73rem;
          font-weight: 720;
          letter-spacing: .12em;
          color: #b9aaff;
          margin-bottom: .8rem;
        }
        .sr-welcome {
          padding: 3.4rem 2.4rem 3rem;
          border: 1px solid rgba(178, 164, 255, .2);
          border-radius: 24px;
          background:
            radial-gradient(ellipse at 93% -20%, rgba(134, 115, 255, .23), transparent 54%),
            linear-gradient(135deg, #18213b, #121a2e 70%);
          box-shadow: 0 16px 60px rgba(0, 0, 0, .16);
          margin: 1.2rem 0;
        }
        .sr-welcome h2 {
          font-size: clamp(1.5rem, 3vw, 2.1rem);
          letter-spacing: -.035em;
          margin: .1rem 0 .55rem;
          color: #f7f7ff;
        }
        .sr-welcome p { color: #bac6dc; line-height: 1.7; max-width: 560px; }
        .sr-small {
          font-size: .79rem;
          color: #8799b7;
          margin-top: 1.1rem;
          line-height: 1.6;
        }
        [data-testid="stChatMessage"] {
          border: 1px solid var(--sr-border);
          border-radius: 18px;
          padding: 1.12rem 1.3rem;
          margin: .7rem 0;
          background: #121a2c;
          box-shadow: 0 7px 20px #060b1850;
        }
        [data-testid="stChatMessage"] p { line-height: 1.85; font-size: 1rem; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
          background: #1a2036;
          border-color: rgba(162, 149, 237, .18);
        }
        /* Keep the composer compact and horizontally centred, not full-width. */
        [data-testid="stBottomBlockContainer"] {
          width: min(100%, 640px);
          max-width: 640px;
          margin-inline: auto;
        }
        [data-testid="stBottom"] [data-testid="stChatInput"] {
          width: 100%;
          margin-inline: auto;
          border: 1px solid rgba(160, 151, 247, .46);
          border-radius: 20px;
          background: linear-gradient(145deg, #19243b, #141d30);
          box-shadow: 0 10px 34px rgba(4, 8, 23, .38);
          transition: border-color .18s ease, box-shadow .18s ease;
        }
        [data-testid="stBottom"] [data-testid="stChatInput"]:focus-within {
          border-color: #aa9cff;
          box-shadow: 0 10px 36px rgba(6, 8, 27, .42),
                      0 0 0 3px rgba(154, 136, 255, .13);
        }
        /* Streamlit's textarea grows naturally as the user types more lines. */
        [data-testid="stChatInput"] textarea {
          min-height: 24px;
          max-height: 180px;
          line-height: 1.55;
          resize: none;
        }
        div.stButton > button {
          border-radius: 11px;
          min-height: 42px;
          font-weight: 600;
          transition: border-color .2s, background .2s;
        }
        div.stButton > button[kind="primary"] {
          background: linear-gradient(112deg, #8070ee, #675cce);
          color: white;
          border: 1px solid #998dfa;
          box-shadow: 0 8px 26px #7062e322;
        }
        [data-testid="stFileUploader"] {
          border-radius: 12px;
          overflow: hidden;
        }
        [data-testid="stExpander"] {
          border-radius: 12px;
          border: 1px solid var(--sr-border);
        }
        @media (max-width: 700px) {
          .block-container { padding: 1.4rem 1.1rem 6.7rem; }
          .sr-welcome { padding: 2.1rem 1.35rem; }
          [data-testid="stChatMessage"] { padding: .85rem 1rem; }
          .sr-heading h1 { font-size: 2rem; }
          [data-testid="stBottomBlockContainer"] {
            width: 100%;
            max-width: 100%;
          }
          [data-testid="stBottom"] [data-testid="stChatInput"] {
            border-radius: 16px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=15, show_spinner=False)
def _ollama_online() -> bool:
    """Check the local generation service without blocking the page."""
    try:
        return requests.get(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            + "/api/tags",
            timeout=0.6,
        ).ok
    except requests.RequestException:
        return False


def _sidebar(session: StudySession) -> int:
    """Keep document actions visible and hide optional retrieval tuning."""
    with st.sidebar:
        st.markdown(
            '<div class="sr-brand"><span class="sr-logo">📚</span>'
            '<div><strong>StudyRAG</strong><small>Your private study desk</small></div></div>',
            unsafe_allow_html=True,
        )
        model = escape(os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
        online = _ollama_online()
        state_class = "sr-online" if online else "sr-offline"
        state_text = "Ready to answer" if online else "Ollama is offline"
        st.markdown(
            f'<div class="sr-status"><span class="sr-dot {state_class}"></span>'
            f'<span>{state_text} · {model}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sr-label">YOUR DOCUMENTS</div>', unsafe_allow_html=True)
        uploads = st.file_uploader(
            "Choose notes",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help=f"PDF, TXT or Markdown, up to {MAX_UPLOAD_MB} MB each.",
        )
        if st.button(
            "✦  Add to study library",
            type="primary",
            disabled=not uploads,
            use_container_width=True,
        ):
            try:
                with st.spinner("Extracting text and preparing your notes…"):
                    counts = session.ingest(uploads or [])
            except (ValueError, OSError, RuntimeError, MemoryError) as exc:
                st.error(f"Could not index notes: {exc}")
            else:
                st.session_state.messages = []
                st.toast(
                    f"Ready: {len(counts)} file(s) indexed.",
                    icon="✅",
                )
                st.rerun()

        chunks = session.store.count()
        st.markdown(
            '<div class="sr-stat"><span>Study library</span>'
            f'<strong>{len(session.files)} files · {chunks} chunks</strong></div>',
            unsafe_allow_html=True,
        )
        if session.files:
            with st.expander("View indexed files"):
                for filename in session.files:
                    st.text(filename)

        st.markdown('<div class="sr-label">CHAT</div>', unsafe_allow_html=True)
        if st.button(
            "＋  New conversation",
            use_container_width=True,
            disabled=not bool(st.session_state.get("messages")),
        ):
            st.session_state.messages = []
            st.rerun()
        if st.button(
            "⌫  Clear library & chat",
            use_container_width=True,
            disabled=not bool(session.files),
        ):
            st.session_state.confirm_clear = True
        if st.session_state.get("confirm_clear"):
            st.warning("Remove all indexed notes and chat messages from this session?")
            col_yes, col_no = st.columns(2)
            if col_yes.button("Clear all", type="primary", use_container_width=True):
                session.close()
                st.session_state.pop("study_session", None)
                st.session_state.pop("messages", None)
                st.session_state.confirm_clear = False
                st.rerun()
            if col_no.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

        with st.expander("⚙  Advanced", expanded=False):
            top_k = st.slider(
                "Passages to retrieve",
                min_value=1,
                max_value=4,
                value=2,
                help="More passages give broader context; fewer can answer faster.",
            )
            st.caption(
                f"Up to {MAX_UPLOAD_MB} MB per file. Scanned PDFs use Bengali/English OCR."
            )
        if not online:
            st.caption("Launch Ollama with the StudyRAG start script to enable answers.")
    return top_k


def _history() -> list[tuple[str, str]]:
    """Use recent, non-error messages as follow-up context."""
    return [
        (message["role"], message["content"])
        for message in st.session_state.messages[-6:]
        if not message.get("error")
    ]


def _render_welcome(has_files: bool) -> None:
    """Show an inviting empty state without unnecessary UI controls."""
    if has_files:
        headline = "Your notes are ready."
        description = (
            "Ask a question about your documents in বাংলা or English. "
            "You can ask follow-up questions in the same conversation."
        )
        hint = "Try: এই অধ্যায়ের মূল বিষয় কী?"
    else:
        headline = "Your notes. Clear answers."
        description = (
            "Upload a PDF, TXT or Markdown document using the sidebar. "
            "StudyRAG will prepare it for questions, including scanned Bengali PDFs."
        )
        hint = "1 · Upload notes &nbsp;&nbsp; 2 · Add to library &nbsp;&nbsp; 3 · Start chatting"
    st.markdown(
        '<div class="sr-welcome"><div class="sr-eyebrow">YOUR STUDY SPACE</div>'
        f"<h2>{headline}</h2><p>{description}</p>"
        f'<div class="sr-small">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the upload workflow and grounded chat without business logic."""
    st.set_page_config(
        page_title="StudyRAG | Study with your notes",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _styles()
    session = _session()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    top_k = _sidebar(session)

    st.markdown(
        '<div class="sr-heading"><div class="sr-eyebrow">FOCUS MODE</div>'
        "<h1>Study smarter with your notes.</h1>"
        "<p>Ask in Bangla or English. Get focused answers from your documents.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    if not st.session_state.messages:
        _render_welcome(bool(session.files))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask anything about your notes…",
        disabled=not bool(session.files),
    )
    if not question:
        return

    history = _history()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding the answer in your notes…"):
                result = session.ask(
                    question,
                    "ollama",
                    history=history,
                    top_k=top_k,
                )
            answer = result.answer
            failed = False
        except (LLMError, ValueError, OSError) as exc:
            answer = f"Could not answer: {exc}"
            failed = True
        display_answer = re.sub(r"\s*\[\d+\]", "", answer)
        st.markdown(display_answer)
        if not failed and answer.startswith("I couldn't find") and session.last_debug:
            with st.expander("Why couldn't I find an answer?"):
                st.write(session.last_debug.get("failure_reason") or "No supporting passage.")
                st.caption("Try asking a more specific question from the uploaded notes.")
    st.session_state.messages.append(
        {"role": "assistant", "content": display_answer, "error": failed}
    )


if __name__ == "__main__":
    main()
