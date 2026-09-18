"""Streamlit UI (see specs/design.md §11) — the walking-skeleton v1 for Phase 2.

Two panels: upload-and-index a document, then ask a question about it.
Citations and a confidence badge (Phase 8), multi-file upload (Phase 9) and
conversation history (Phase 10) come later — this version answers one
question at a time from whatever is currently indexed, with no chat history
kept between turns.
"""

import streamlit as st

from src.index.vector_store import IndexProviderMismatch, VectorStore
from src.ingest.chunker import chunk_document
from src.ingest.loader import load_bytes
from src.query.pipeline import answer_question

st.set_page_config(page_title="Lunorsoft RAG Assistant", page_icon="📚")


@st.cache_resource(show_spinner=False)
def get_store() -> VectorStore:
    """Open the vector store once per session and reuse it across reruns.

    Streamlit re-executes this whole script on every interaction, and
    `VectorStore()` is not cheap to construct — it opens a Chroma client and,
    on the local provider, loads the MiniLM model. Without caching, typing a
    single character into the question box would reload all of that.
    """
    return VectorStore()


def ingest_uploaded_file(uploaded_file, store: VectorStore) -> tuple[int, str]:
    """Load, chunk and index one uploaded file. Returns (chunk count, doc_id)."""
    document = load_bytes(uploaded_file.getvalue(), uploaded_file.name)
    chunks = chunk_document(document)
    n_chunks = store.add_chunks(chunks)
    return n_chunks, document.doc_id


def render_upload(store: VectorStore) -> None:
    st.subheader("1. Add a document")
    uploaded = st.file_uploader(
        "Upload a PDF, TXT or Markdown file",
        type=["pdf", "txt", "md", "markdown"],
    )
    # Indexing runs on an explicit button press rather than automatically on
    # upload: the file uploader's value survives every rerun of this script,
    # so an automatic trigger would re-embed the same file on every unrelated
    # interaction (e.g. typing a question) for as long as it stayed uploaded.
    if uploaded is not None and st.button("Index document"):
        with st.spinner(f"Indexing {uploaded.name}…"):
            try:
                n_chunks, _doc_id = ingest_uploaded_file(uploaded, store)
            except (ValueError, RuntimeError) as exc:
                # ValueError: unsupported type, empty/unreadable document
                # (src/ingest/loader.py, src/ingest/extract.py). RuntimeError:
                # embedding failure (src/index/embeddings.py). Per
                # specs/design.md §14, a bad document is surfaced, not silent.
                st.error(f"Could not index {uploaded.name}: {exc}")
            else:
                st.success(f"Indexed {n_chunks} chunk(s) from {uploaded.name}.")

    count = store.count()
    if count:
        docs = store.indexed_documents()
        st.caption(
            f"{count} chunk(s) indexed from {len(docs)} document(s): "
            + ", ".join(sorted(docs.values()))
        )
    else:
        st.info("No documents indexed yet — upload one above to get started.")


def render_query(store: VectorStore) -> None:
    st.subheader("2. Ask a question")

    question = st.text_input("Your question", placeholder="What is …?")
    ask = st.button("Ask")

    if ask and not question.strip():
        st.warning("Type a question first.")
        return

    if ask:
        with st.spinner("Thinking…"):
            try:
                envelope = answer_question(question, store=store)
            except (ValueError, RuntimeError) as exc:
                st.error(f"Could not answer that question: {exc}")
                return
        st.markdown(envelope.answer)
        st.caption(
            f"Retrieved {envelope.retrieved_k} chunk(s), "
            f"used {envelope.final_k} in the answer's context."
        )


def main() -> None:
    st.title("📚 Lunorsoft RAG Assistant")
    st.caption(
        "Ask questions grounded in your uploaded documents. Answers are "
        "generated only from retrieved context, and the assistant says so "
        "when it can't find an answer there."
    )

    try:
        store = get_store()
    except IndexProviderMismatch as exc:
        st.error(str(exc))
        st.stop()

    render_upload(store)
    st.divider()
    render_query(store)


if __name__ == "__main__":
    main()
