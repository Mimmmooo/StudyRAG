"""StudyRAG chat interface; document and RAG logic lives in src/."""

import os

import requests
import streamlit as st

from src.llm.base import LLMError
from src.provider_selection import select_default_provider
from src.rag.models import Citation
from src.ui_service import StudySession


def _session() -> StudySession:
    """Reuse an index for the current Streamlit browser session."""
    if "study_session" not in st.session_state:
        st.session_state.study_session = StudySession()
    return st.session_state.study_session


def _render_sources(sources: tuple[Citation, ...]) -> None:
    """Show matching passage text together with numbered file/page labels."""
    for index, source in enumerate(sources, start=1):
        with st.expander(f"[{index}] {source.source} · page {source.page}"):
            st.text(source.text)


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


def _hosted_key(name: str) -> str | None:
    """Read server-managed keys without displaying them in the UI."""
    try:
        key = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        key = None
    return str(key) if key else os.getenv(name) or None


def _sidebar(session: StudySession) -> tuple[str, str | None, int]:
    """Configure model, retrieval, uploads and a fresh chat."""
    with st.sidebar:
        st.header("Study settings")
        server_keys = {
            "groq": _hosted_key("GROQ_API_KEY"),
            "gemini": _hosted_key("GEMINI_API_KEY"),
        }
        available_ollama = _ollama_online()
        default_provider = select_default_provider(
            ollama_available=available_ollama,
            groq_key=server_keys["groq"],
            gemini_key=server_keys["gemini"],
        )
        choices = ["ollama", "groq", "gemini", "extractive"]
        # A versioned widget key avoids retained values from older deployments.
        provider = st.selectbox(
            "Answer mode",
            choices,
            index=choices.index(default_provider),
            key="answer_mode_v2",
            format_func=lambda value: {
                "ollama": "Ollama (local AI)",
                "groq": "Groq (AI answer)",
                "gemini": "Gemini (AI answer)",
                "extractive": "Source passages (no AI)",
            }[value],
        )
        key = server_keys.get(provider)
        if provider in {"gemini", "groq"} and not key:
            key = st.text_input(
                f"{provider.title()} API key",
                type="password",
                help="Your key is used for this session and not stored in the repository.",
            )
        elif provider in {"gemini", "groq"}:
            st.caption("AI answers are enabled by the app owner.")
        elif provider == "ollama":
            if available_ollama:
                st.caption("Local Ollama is available and will generate answers.")
            else:
                st.warning(
                    "Ollama is unavailable on this server. For cloud AI answers, "
                    "the app owner must configure a Gemini or Groq key."
                )
        else:
            st.caption("No key required. Shows relevant source text, not a generated AI explanation.")
        top_k = st.slider(
            "Passages to retrieve",
            min_value=1,
            max_value=5,
            value=3,
            help="More passages can improve coverage but use more context.",
        )
        uploads = st.file_uploader(
            "Upload your notes",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Text PDFs, TXT and MD. Maximum 200 MB per file. Large PDFs may exceed free hosting memory.",
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
                    "If a large PDF fails, try a smaller text-based PDF first. "
                    "Cloud memory limits may prevent very large textbooks from indexing."
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
        st.caption("Indexed text is stored temporarily on this app server.")
    return provider, key, top_k


def _history() -> list[tuple[str, str]]:
    """Build limited conversation context without leaking UI metadata."""
    return [
        (message["role"], message["content"])
        for message in st.session_state.messages[-6:]
        if not message.get("error")
    ]


def main() -> None:
    """Render notes upload, follow-up chat and page-aware citations."""
    st.set_page_config(
        page_title="StudyRAG · Chat with notes",
        page_icon="📚",
        layout="centered",
    )
    st.title("📚 StudyRAG")
    st.caption("Chat with your notes and inspect file/page citations.")
    session = _session()
    provider, key, top_k = _sidebar(session)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not session.files:
        st.info("Upload a PDF, TXT or Markdown file, then click **Index selected files**.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_sources(message.get("sources", ()))

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
            sources = result.citations
            failed = False
        except (LLMError, ValueError, OSError) as exc:
            answer = f"Could not answer: {exc}"
            sources = ()
            failed = True
        st.markdown(answer)
        if not sources and not failed and session.last_debug:
            with st.expander("Why was no answer found?"):
                st.write(f"Indexed chunks: {session.last_debug['indexed_chunks']}")
                st.write(f"Retrieval cutoff: {session.last_debug['threshold']}")
                st.write(f"Nearest passages: {session.last_debug['matches']}")
                st.write(f"Model response: {session.last_debug['model_answer']!r}")
        _render_sources(sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "error": failed}
    )


if __name__ == "__main__":
    main()
