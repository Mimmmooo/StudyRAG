"""StudyRAG chat interface; document and RAG logic lives in src/."""

import os
import re

import requests
import streamlit as st

from src.llm.base import LLMError
from src.ui_service import MAX_UPLOAD_MB, StudySession


def _session() -> StudySession:
    """Reuse an index for the current Streamlit browser session."""
    if "study_session" not in st.session_state:
        st.session_state.study_session = StudySession()
    return st.session_state.study_session


def _chat_styles() -> None:
    """Use a roomy, readable study-chat layout."""
    st.markdown(
        """
        <style>
          .block-container {max-width: 960px; padding-top: 2.2rem;}
          [data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: .65rem;
          }
          [data-testid="stChatMessage"] p {
            line-height: 1.75;
            font-size: 1.02rem;
          }
          [data-testid="stChatInput"] {border-radius: 18px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60, show_spinner=False)
def _ollama_online() -> bool:
    """Check whether a local Ollama service is reachable from this server."""
    try:
        return requests.get(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            + "/api/tags",
            timeout=0.6,
        ).ok
    except requests.RequestException:
        return False


def _sidebar(session: StudySession) -> tuple[str, str | None, int]:
    """Configure model, retrieval, uploads and a fresh chat."""
    with st.sidebar:
        st.header("Study settings")
        st.caption(f"Upload limit: {MAX_UPLOAD_MB} MB per file · OCR-enabled build")
        provider = "ollama"
        key = None
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        st.caption(f"Local AI: Ollama ({model})")
        if not _ollama_online():
            st.warning("Start Ollama using the StudyRAG launcher before asking questions.")
        top_k = st.slider(
            "Passages to retrieve",
            min_value=1,
            max_value=4,
            value=2,
            help="Use 2 by default for quicker answers; raise for broader questions.",
        )
        uploads = st.file_uploader(
            "Upload your notes",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help=f"PDF, TXT and MD. Maximum {MAX_UPLOAD_MB} MB per file. Large scanned PDFs may require extra RAM.",
        )
        if uploads:
            st.caption(
                f"Ready to index {len(uploads)} file(s). Large books may take longer."
            )
        if st.button(
            "Index selected files",
            disabled=not bool(uploads),
            use_container_width=True,
            type="primary",
        ):
            try:
                with st.spinner("Reading and indexing notes..."):
                    counts = session.ingest(uploads or [])
            except (ValueError, OSError, RuntimeError, MemoryError) as exc:
                st.error(f"Could not index documents: {exc}")
                st.info(
                    "For very large PDFs, check your computer\u0027s free RAM and disk. "
                    "Scanned pages also need Bengali/English Tesseract OCR."
                )
            else:
                st.success(
                    "Indexed: " + ", ".join(
                        f"{name} ({count} chunks)" for name, count in counts.items()
                    )
                )
                st.session_state.messages = []
        st.caption(f"{len(session.files)} file(s) · {session.store.count()} chunks")
        if session.files:
            with st.expander("Indexed documents"):
                for filename in session.files:
                    st.text(filename)
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("Clear notes and chat", use_container_width=True):
            session.close()
            st.session_state.pop("study_session", None)
            st.session_state.pop("messages", None)
            st.rerun()
        st.caption("Indexed text stays in this computer\u0027s temporary session storage.")
    return provider, key, top_k


def _history() -> list[tuple[str, str]]:
    """Build limited conversation context without leaking UI metadata."""
    return [
        (message["role"], message["content"])
        for message in st.session_state.messages[-6:]
        if not message.get("error")
    ]


def main() -> None:
    """Render local notes upload and simple grounded chat."""
    st.set_page_config(
        page_title="StudyRAG · Chat with notes",
        page_icon="📚",
        layout="wide",
    )
    _chat_styles()
    st.title("📚 StudyRAG")
    st.caption("Upload your notes and ask questions in Bangla or English.")
    session = _session()
    provider, key, top_k = _sidebar(session)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not session.files:
        st.info("Upload a PDF, TXT or Markdown file, then click **Index selected files**.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask about your notes or follow up on an answer",
        disabled=not session.files,
    )
    if not question:
        return

    history = _history()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving passages and generating an answer..."):
                result = session.ask(
                    question,
                    provider,
                    api_key=key,
                    history=history,
                    top_k=top_k,
                )
            answer = result.answer
            failed = False
        except (LLMError, ValueError, OSError) as exc:
            answer = f"Could not answer: {exc}"
            failed = True
        # Older local models sometimes append numeric source markers.
        display_answer = re.sub(r"\s*\[\d+\]", "", answer)
        st.markdown(display_answer)
        if not failed and answer.startswith("I couldn't find") and session.last_debug:
            with st.expander("Why was no answer found?"):
                st.write(f"Reason: {session.last_debug.get('failure_reason') or 'Unknown'}")
                st.write(f"Indexed chunks: {session.last_debug['indexed_chunks']}")
                st.write(f"Retrieval cutoff: {session.last_debug['threshold']}")
                st.write(f"Nearest passages: {session.last_debug['matches']}")
                st.write(f"Model response: {session.last_debug['model_answer']!r}")
    st.session_state.messages.append(
        {"role": "assistant", "content": display_answer, "error": failed}
    )


if __name__ == "__main__":
    main()
