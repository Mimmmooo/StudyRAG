"""StudyRAG chat interface; document and RAG logic lives in src/."""

import streamlit as st

from src.llm.base import LLMError
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


def _sidebar(session: StudySession) -> tuple[str, str | None, int]:
    """Configure model, retrieval, uploads and a fresh chat."""
    with st.sidebar:
        st.header("Study settings")
        provider = st.selectbox("LLM provider", ["extractive", "ollama", "gemini", "groq"])
        key = None
        if provider in {"gemini", "groq"}:
            key = st.text_input(
                f"{provider.title()} API key",
                type="password",
                help="Not saved to files or version control.",
            )
        elif provider == "ollama":
            st.caption("Local Ollama requires qwen2.5:3b.")
        else:
            st.caption("No API key needed. Shows matching note passages, not an AI-written answer.")
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
            help="Text PDFs, TXT and MD. Maximum 15 MB per file.",
        )
        if st.button("Index selected files", disabled=not uploads, use_container_width=True):
            try:
                with st.spinner("Reading and indexing notes..."):
                    counts = session.ingest(uploads or [])
            except (ValueError, OSError) as exc:
                st.error(str(exc))
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
        st.caption("Your notes are indexed in a temporary local session.")
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
    st.caption("Study your own notes with local retrieval and page citations.")
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
