# Lunorsoft RAG Assistant

A mini AI knowledge assistant that answers questions grounded in a set of supplied
documents, going beyond naive RAG with structure-aware chunking, hybrid retrieval,
cross-encoder re-ranking, query transformation, and a grounding/faithfulness check.

See [`specs/design.md`](specs/design.md) for the architecture and
[`specs/requirements.md`](specs/requirements.md) for the scope.

## Setup

**Requires Python 3.12** (pinned in `.python-version`). Newer versions are not
yet verified against the pinned dependency set.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Run

```bash
streamlit run app.py
```
