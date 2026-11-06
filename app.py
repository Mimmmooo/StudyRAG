"""StudyRAG Streamlit interface; business logic lives in src/ui_service.py."""

import streamlit as st

from src.llm.base import LLMError
from src.ui_service import StudySession


def _session() -> StudySession:
    """Reuse a private study session across Streamlit reruns."""
    if "study_session" not in st.session_state:
        st.session_state.study_session = StudySession()
    return st.session_state.study_session


def _sidebar(session: StudySession) -> tuple[str, str | None]:
    """Display model selection, upload controls, and session reset."""
    with st.sidebar:
        st.header("Study settings")
        provider = st.selectbox("LLM provider", ["ollama", "gemini", "groq"])
        key = None
        if provider != "ollama":
            key = st.text_input(
                f"{provider.title()} API key",
                type="password",
                help="Your key stays in this browser session and is not written to disk.",
            )
        else:
            st.caption("Start Ollama locally and pull qwen2.5:3b first.")
        uploads = st.file_uploader(
            "Upload notes", type=["pdf", "txt", "md"], accept_multiple_files=True
        )
        if st.button("Index selected files", disabled=not uploads):
            try:
                with st.spinner("Reading, chunking and embedding notes..."):
                    counts = session.ingest(uploads or [])
                st.success(
                    "Indexed: " + ", ".join(
                        f"{name} ({count} chunks)" for name, count in counts.items()
                    )
                )
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        st.caption(
            f"Indexed files: {', '.join(session.files) if session.files else 'none'}"
        )
        if st.button("Clear notes and chat"):
            session.close()
            st.session_state.pop("study_session", None)
            st.session_state.pop("messages", None)
            st.rerun()
    return provider, key


def main() -> None:
    """Render upload, chat history, answers and source citations."""
    st.set_page_config(page_title="StudyRAG", page_icon="📚")
    st.title("📚 StudyRAG")
    st.caption("Chat with your study notes. Answers include file and page references.")
    session = _session()
    provider, key = _sidebar(session)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                with st.expander(
                    f"Source: {source.source}, page {source.page}"
                ):
                    st.text(source.text)
    question = st.chat_input(
        "Ask a question about your notes", disabled=not session.files
    )
    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching notes and generating answer..."):
                    result = session.ask(question, provider, api_key=key)
                answer = result.answer
                sources = result.citations
                debug = session.last_debug
            except (LLMError, ValueError, OSError) as exc:
                answer = f"Could not answer: {exc}"
                sources = ()
                debug = {}
            st.markdown(answer)
            if not sources and debug:
                with st.expander("Why was no answer found? (diagnostics)"):
                    st.write(f"Indexed chunks: {debug['indexed_chunks']}")
                    st.write(f"Retrieval cutoff: {debug['threshold']}")
                    st.write(f"Retrieved matches: {debug['matches']}")
                    st.write(f"Model response: {debug['model_answer']!r}")
            for source in sources:
                with st.expander(
                    f"Source: {source.source}, page {source.page}"
                ):
                    st.text(source.text)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
