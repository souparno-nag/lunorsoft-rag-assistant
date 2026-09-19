"""Streamlit UI (see specs/design.md §11) — the walking-skeleton v1 for Phase 2.

Two panels: upload-and-index a document, then ask a question about it.
Citations and a confidence badge (Phase 8), multi-file upload (Phase 9) and
conversation history (Phase 10) come later — this version answers one
question at a time from whatever is currently indexed, with no chat history
kept between turns.

Indexing and retrieval both go through `Indexer`, which owns the vector store
and the keyword index together, so the UI never has to remember that a
document becomes two index entries.
"""

import streamlit as st

from src.models.schemas import AnswerEnvelope, Citation
from src.index.indexer import Indexer
from src.index.vector_store import IndexProviderMismatch
from src.ingest.loader import load_bytes
from src.query.pipeline import answer_question

st.set_page_config(page_title="Lunorsoft RAG Assistant", page_icon="📚")

# An excerpt the answer actually cited is shown whole — it is the evidence, and
# truncating it would defeat the point of offering it. An excerpt that was
# retrieved but not cited is shown as a preview: it is there so a reader can
# see what the model had available, which a first line or two answers.
UNCITED_PREVIEW_CHARS = 280


@st.cache_resource(show_spinner=False)
def get_indexer() -> Indexer:
    """Open both indexes once per session and reuse them across reruns.

    Streamlit re-executes this whole script on every interaction, and an
    `Indexer` is not cheap to construct — it opens a Chroma client, loads the
    MiniLM model on the local provider, and reads the keyword corpus from
    disk. Without caching, typing a single character into the question box
    would reload all of that.
    """
    return Indexer()


def ingest_uploaded_file(uploaded_file, indexer: Indexer) -> tuple[int, str]:
    """Load, chunk and index one uploaded file. Returns (chunk count, doc_id)."""
    document = load_bytes(uploaded_file.getvalue(), uploaded_file.name)
    n_chunks = indexer.add_document(document)
    return n_chunks, document.doc_id


def render_upload(indexer: Indexer) -> None:
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
                n_chunks, _doc_id = ingest_uploaded_file(uploaded, indexer)
            except (ValueError, RuntimeError) as exc:
                # ValueError: unsupported type, empty/unreadable document
                # (src/ingest/loader.py, src/ingest/extract.py). RuntimeError:
                # embedding failure (src/index/embeddings.py). Per
                # specs/design.md §14, a bad document is surfaced, not silent.
                st.error(f"Could not index {uploaded.name}: {exc}")
            else:
                st.success(f"Indexed {n_chunks} chunk(s) from {uploaded.name}.")

    count = indexer.count()
    if count:
        docs = indexer.indexed_documents()
        st.caption(
            f"{count} chunk(s) indexed from {len(docs)} document(s): "
            + ", ".join(sorted(docs.values()))
        )
        # An index built before the keyword half existed has vectors but no
        # BM25 corpus, so hybrid search would quietly fall back to dense-only
        # rather than fail. Say so instead of letting it look like a retrieval
        # quality problem.
        if indexer.needs_rebuild:
            st.warning(
                "This index predates keyword search, so only vector retrieval "
                "is active. Re-index the documents to enable hybrid search."
            )
    else:
        st.info("No documents indexed yet — upload one above to get started.")


def render_citations(envelope: AnswerEnvelope) -> None:
    """Show the excerpts behind an answer, cited ones first.

    Collapsed by default: the answer is what was asked for, and the evidence
    is one click away for a reader who wants to check it rather than something
    they have to scroll past to reach the next question.
    """
    if not envelope.citations:
        return

    cited = [c for c in envelope.citations if c.cited]
    total = len(envelope.citations)
    if cited:
        label = f"Sources — {len(cited)} of {total} excerpts cited"
    else:
        # Either a refusal, or an answer that neglected to cite. Both are worth
        # being able to inspect: seeing what retrieval found is how a reader
        # tells "the documents do not say" from "the search missed it".
        label = f"Sources — {total} excerpts retrieved, none cited"

    with st.expander(label):
        # Cited first, then the rest in the order the model saw them, so the
        # evidence is at the top rather than interleaved with what went unused.
        ordered = cited + [c for c in envelope.citations if not c.cited]
        for citation in ordered:
            st.markdown(_citation_heading(citation))
            st.markdown(_citation_body(citation))


def _citation_heading(citation: Citation) -> str:
    """`[2] paper.pdf · p. 8 · 5.4 Regularization` — the trail to the source."""
    parts = [f"**[{citation.marker}]** {citation.source_file}"]
    if citation.page is not None:
        parts.append(f"p. {citation.page}")
    if citation.section:
        parts.append(citation.section)
    trail = " · ".join(parts)
    return f"{trail} — cited" if citation.cited else f"{trail} — not cited"


def _citation_body(citation: Citation) -> str:
    """The excerpt text as a blockquote.

    Whitespace is collapsed to one line before rendering. Chunk text carries
    the line breaks of the page it came from, which mean nothing here, and
    flattening them also stops a line that begins with a `#` or a `-` in the
    source document from being rendered as a heading or a list inside the
    quote.
    """
    text = " ".join(citation.snippet.split())
    if not citation.cited and len(text) > UNCITED_PREVIEW_CHARS:
        text = text[:UNCITED_PREVIEW_CHARS].rstrip() + "…"
    return f"> {text}"


def render_query(indexer: Indexer) -> None:
    st.subheader("2. Ask a question")
    has_documents = indexer.count() > 0

    # T2.4: querying is blocked, not just discouraged, while the index is
    # empty — the input and button are disabled so there is nothing to submit,
    # and the prompt below says plainly what to do instead.
    question = st.text_input(
        "Your question",
        disabled=not has_documents,
        placeholder="Upload and index a document first" if not has_documents else "What is …?",
    )
    ask = st.button("Ask", disabled=not has_documents)

    if not has_documents:
        st.info("Upload and index a document above before asking a question.")
        return

    if ask and not question.strip():
        st.warning("Type a question first.")
        return

    if ask:
        with st.spinner("Thinking…"):
            try:
                envelope = answer_question(question, indexer=indexer)
            except (ValueError, RuntimeError) as exc:
                st.error(f"Could not answer that question: {exc}")
                return
        st.markdown(envelope.answer)
        render_citations(envelope)
        # Transparency touch from specs/design.md §11: say what the pipeline
        # actually did, so a demo can show the difference a transform makes
        # rather than assert it.
        transform = envelope.used_query_transform or "none"
        st.caption(
            f"Query transform: {transform} · retrieved {envelope.retrieved_k} "
            f"chunk(s) · used {envelope.final_k} in the answer's context."
        )


def main() -> None:
    st.title("📚 Lunorsoft RAG Assistant")
    st.caption(
        "Ask questions grounded in your uploaded documents. Answers are "
        "generated only from retrieved context, and the assistant says so "
        "when it can't find an answer there."
    )

    try:
        indexer = get_indexer()
    except IndexProviderMismatch as exc:
        st.error(str(exc))
        st.stop()

    render_upload(indexer)
    st.divider()
    render_query(indexer)


if __name__ == "__main__":
    main()
