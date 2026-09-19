"""Streamlit UI (see specs/design.md §11) — the walking-skeleton v1 for Phase 2.

Two panels: upload and index documents, then hold a conversation about them.
Each answer carries a confidence badge and the excerpts behind it, and the
transcript is kept for the session so follow-up questions can refer back.

Indexing and retrieval both go through `Indexer`, which owns the vector store
and the keyword index together, so the UI never has to remember that a
document becomes two index entries.
"""

import streamlit as st

from config import settings
from src.index.indexer import Indexer
from src.index.vector_store import IndexProviderMismatch
from src.ingest.loader import SUPPORTED_SUFFIXES, load_bytes, load_documents
from src.models.schemas import AnswerEnvelope, Citation, Turn
from src.query.pipeline import answer_question

st.set_page_config(page_title="Lunorsoft RAG Assistant", page_icon="📚")

# An excerpt the answer actually cited is shown whole — it is the evidence, and
# truncating it would defeat the point of offering it. An excerpt that was
# retrieved but not cited is shown as a preview: it is there so a reader can
# see what the model had available, which a first line or two answers.
UNCITED_PREVIEW_CHARS = 280

# The confidence bands of specs/design.md §5.6, in the colours §5.7 asks for.
# The band is computed in src/generate/grounding.py from the configured
# cutoffs; this only decides how it looks.
CONFIDENCE_COLOURS = {"high": "green", "medium": "orange", "low": "red"}


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


def bootstrap_sample_documents(indexer: Indexer) -> None:
    """Index whatever is in data/ the first time a session finds nothing indexed.

    storage/ is not committed, so a freshly deployed container has an empty
    index while data/ still holds the sample corpus. Without this, the public
    link opens on "no documents indexed yet" and a reviewer has to find a PDF
    before the app does anything — which is a poor showing for the one link
    that gets looked at.

    Guarded per session rather than per index, so clearing the index does not
    have it immediately repopulate underneath the user. A later session on the
    same container will bootstrap again, which is what makes a restarted
    deployment come back up working.
    """
    if st.session_state.get("bootstrap_done") or not settings.BOOTSTRAP_SAMPLE_DOCUMENTS:
        return
    st.session_state["bootstrap_done"] = True

    if indexer.count():
        return
    paths = sorted(
        path
        for path in settings.DATA_DIR.glob("*")
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        return

    with st.spinner(f"Preparing {len(paths)} sample document(s)…"):
        documents, failures = load_documents(paths)
        if documents:
            indexer.add_documents(documents)
    for name, reason in failures:
        st.warning(f"Could not load the sample document {name}: {reason}")
    if documents:
        st.info(
            "Loaded the sample document"
            + ("s" if len(documents) > 1 else "")
            + ": "
            + ", ".join(document.source_file for document in documents)
            + ". Upload your own above, or ask a question below."
        )


def ingest_uploaded_file(uploaded_file, indexer: Indexer) -> tuple[int, str]:
    """Load, chunk and index one uploaded file. Returns (chunk count, doc_id)."""
    document = load_bytes(uploaded_file.getvalue(), uploaded_file.name)
    n_chunks = indexer.add_document(document)
    return n_chunks, document.doc_id


def index_uploaded_files(uploaded_files, indexer: Indexer) -> None:
    """Index a batch of uploads, reporting each file's outcome separately.

    One unreadable file does not abandon the batch (specs/design.md §14): a
    scanned PDF with no text layer, or a file whose extension lies about its
    contents, is reported and skipped while the rest are indexed. Uploading
    ten documents and losing all of them to the fourth would be the worst
    possible behaviour here.
    """
    indexed: list[str] = []
    failed: list[tuple[str, str]] = []

    progress = st.progress(0.0)
    for position, uploaded in enumerate(uploaded_files, start=1):
        progress.progress(
            (position - 1) / len(uploaded_files), text=f"Indexing {uploaded.name}…"
        )
        try:
            n_chunks, _doc_id = ingest_uploaded_file(uploaded, indexer)
        except (ValueError, RuntimeError) as exc:
            # ValueError: unsupported type, empty or unreadable document.
            # RuntimeError: embedding failure. Both are the document's
            # problem, not the batch's.
            failed.append((uploaded.name, str(exc)))
        else:
            indexed.append(f"{uploaded.name} ({n_chunks} chunks)")
    progress.empty()

    if indexed:
        st.success("Indexed " + ", ".join(indexed) + ".")
    for name, reason in failed:
        st.error(f"Skipped {name}: {reason}")


def render_index_status(indexer: Indexer) -> None:
    """List what is currently indexed, and offer to clear it."""
    if not indexer.count():
        st.info("No documents indexed yet — upload one above to get started.")
        return

    documents = indexer.indexed_documents()
    counts = indexer.chunk_counts()
    st.caption(f"{indexer.count()} chunk(s) indexed from {len(documents)} document(s):")
    for doc_id, source_file in sorted(documents.items(), key=lambda pair: pair[1]):
        chunks = counts.get(doc_id)
        suffix = f" — {chunks} chunk(s)" if chunks else ""
        st.markdown(f"- `{source_file}`{suffix}")

    # An index built before the keyword half existed has vectors but no BM25
    # corpus, so hybrid search would quietly fall back to dense-only rather
    # than fail. Say so instead of letting it look like a retrieval problem.
    if indexer.needs_rebuild:
        st.warning(
            "This index predates keyword search, so only vector retrieval is "
            "active. Clear it and re-index to enable hybrid search."
        )

    render_clear_index(indexer)


def render_clear_index(indexer: Indexer) -> None:
    """The 'rebuild index' action of specs/design.md §11, as a two-step.

    Named for what it does. The design calls this "rebuild index", but nothing
    here could rebuild one: uploads are streamed into the index and never kept,
    so once the index is gone the documents have to be uploaded again. Offering
    a button called "rebuild" that silently destroys the only copy of the
    corpus would be a lie in the one place it costs most.

    Two steps for the same reason — it cannot be undone from inside the app.
    """
    if st.session_state.get("confirm_clear"):
        st.warning(
            f"Delete all {indexer.count()} indexed chunk(s)? The documents "
            "themselves are not stored, so they will need uploading again."
        )
        confirm, cancel = st.columns(2)
        if confirm.button("Yes, clear the index", type="primary"):
            indexer.reset()
            st.session_state["confirm_clear"] = False
            st.success("Index cleared.")
            st.rerun()
        if cancel.button("Cancel"):
            st.session_state["confirm_clear"] = False
            st.rerun()
        return

    if st.button("Clear index"):
        st.session_state["confirm_clear"] = True
        st.rerun()


def render_upload(indexer: Indexer) -> None:
    st.subheader("1. Add documents")
    uploaded = st.file_uploader(
        "Upload one or more PDF, TXT or Markdown files",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )
    # Indexing runs on an explicit button press rather than automatically on
    # upload: the file uploader's value survives every rerun of this script, so
    # an automatic trigger would re-embed the same files on every unrelated
    # interaction (e.g. typing a question) for as long as they stayed uploaded.
    if uploaded and st.button(f"Index {len(uploaded)} file(s)"):
        index_uploaded_files(uploaded, indexer)

    render_index_status(indexer)


def render_confidence(envelope: AnswerEnvelope) -> None:
    """Show how far the answer was verified against its sources.

    Nothing is rendered when there is no score. That is not a missing value to
    paper over: a refusal asserts nothing about the documents, so there is no
    claim to have checked, and a badge beside "I could not find this" would be
    telling the reader something that was never measured.

    The tooltip names the method as well as the score, because the two
    measures do not mean the same thing — a judge read the claims, an overlap
    count only compared vocabulary — and a reader deciding whether to trust an
    answer deserves to know which one produced the number.
    """
    band = envelope.confidence
    if band is None or envelope.faithfulness_score is None:
        return

    if envelope.grounding_method == "judge":
        how = (
            f"{envelope.faithfulness_score:.0%} of the answer's claims were "
            "judged supported by the retrieved excerpts."
        )
    else:
        how = (
            f"{envelope.faithfulness_score:.0%} of the answer's wording appears "
            "in the retrieved excerpts. The claim-by-claim check was "
            "unavailable, so this is a weaker word-overlap estimate and is "
            "capped at medium confidence."
        )

    st.badge(
        f"{band} confidence",
        color=CONFIDENCE_COLOURS.get(band, "gray"),
        help=how,
    )


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
        ambiguous = _ambiguous_names(envelope.citations)
        for citation in ordered:
            st.markdown(_citation_heading(citation, ambiguous))
            st.markdown(_citation_body(citation))


def _ambiguous_names(citations: list[Citation]) -> set[str]:
    """Filenames that belong to more than one document in this answer.

    Nothing stops two uploads being called "report.pdf" while holding
    different documents — `doc_id` is derived from the bytes, so the index
    keeps them apart perfectly well. It is the citation list that would show
    them as the same source, which is precisely where a reader is trying to
    tell sources apart.
    """
    seen: dict[str, set[str]] = {}
    for citation in citations:
        seen.setdefault(citation.source_file, set()).add(citation.doc_id)
    return {name for name, ids in seen.items() if len(ids) > 1}


def _citation_heading(citation: Citation, ambiguous: set[str] = frozenset()) -> str:
    """`[2] paper.pdf · p. 8 · 5.4 Regularization` — the trail to the source."""
    name = citation.source_file
    if name in ambiguous and citation.doc_id:
        # The doc_id already ends in a short hash of the file's contents,
        # which is exactly the distinguishing part.
        name = f"{name} ({citation.doc_id.rsplit('-', 1)[-1]})"
    parts = [f"**[{citation.marker}]** {name}"]
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


def render_pipeline_caption(envelope: AnswerEnvelope) -> None:
    """Say what the pipeline did to produce this answer.

    The transparency touch of specs/design.md §11, so a demo can show the
    difference a stage makes rather than assert it.
    """
    parts = [f"Query transform: {envelope.used_query_transform or 'none'}"]
    if envelope.standalone_question:
        parts.append(f"read as “{envelope.standalone_question}”")
    parts.append(f"retrieved {envelope.retrieved_k} chunk(s)")
    parts.append(f"used {envelope.final_k} in the answer's context")
    st.caption(" · ".join(parts))


def render_turn(turn: Turn) -> None:
    """Redraw one exchange, with the evidence it was served with."""
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        st.markdown(turn.envelope.answer)
        render_confidence(turn.envelope)
        render_citations(turn.envelope)
        render_pipeline_caption(turn.envelope)


def render_chat(indexer: Indexer) -> None:
    st.subheader("2. Ask a question")
    has_documents = indexer.count() > 0
    history: list[Turn] = st.session_state.setdefault("history", [])

    for turn in history:
        render_turn(turn)

    # T2.4: querying is blocked, not merely discouraged, while the index is
    # empty — the input is disabled so there is nothing to submit, and the
    # prompt below says plainly what to do instead.
    if not has_documents:
        st.info("Upload and index a document above before asking a question.")
    elif history and st.button("Clear conversation"):
        st.session_state["history"] = []
        st.rerun()

    question = st.chat_input(
        "Ask about your documents…" if has_documents else "Index a document first",
        disabled=not has_documents,
    )
    if not question or not question.strip():
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.spinner("Thinking…"):
        try:
            envelope = answer_question(question, indexer=indexer, history=history)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Could not answer that question: {exc}")
            return

    history.append(Turn(question=question, envelope=envelope))
    # Redraw from history, so the turn just added is rendered by exactly the
    # same code as every earlier one and cannot drift from them.
    st.rerun()


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

    bootstrap_sample_documents(indexer)
    render_upload(indexer)
    st.divider()
    render_chat(indexer)


if __name__ == "__main__":
    main()
