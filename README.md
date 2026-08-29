# Fruity Fun 🍓

Fruity Fun is a kid-friendly chatbot that answers fruit questions from a PDF library and creates a safe, original picture for each request. It uses code-first hybrid RAG: Pinecone vector search, local BM25 keyword search, reciprocal-rank fusion (RRF), reranking, confidence gating, and a LangGraph workflow. It deliberately does **not** use GraphRAG or Neo4j.

## What it does

- Extracts fruit PDFs into page-aware JSON chunks.
- Stores dense OpenAI embeddings and chunk metadata in Pinecone.
- Searches Pinecone semantically and the local JSON corpus with BM25.
- Fuses both ranked lists with RRF, reranks the candidates, and calculates confidence.
- Answers with citations only when the retrieved evidence is strong enough.
- Generates a bright, kid-safe fruit illustration even for imaginative multi-fruit prompts.
- Degrades cleanly when credentials, PDFs, Pinecone, or image generation are unavailable.

## Architecture

```text
PDF files -> PyMuPDF -> JSON chunks -----+----> local BM25
                     |                   |
                     +-> embeddings -> Pinecone dense index
                                         |
question -> safety -> dense + BM25 -> RRF fusion -> rerank -> confidence
                                                            |        |
                                                grounded answer   safe image
                                                            |        |
                                                            +-> Streamlit
```

The factual and visual paths are intentionally separated. Text claims require corpus evidence; artwork can use safe imagination and does not pretend to be a factual source.

See the [detailed architecture diagram](docs/ARCHITECTURE.md) for the ingestion pipeline, LangGraph state flow, external services, and failure paths.

## Setup

Prerequisites: Python 3.11–3.13, an OpenAI API key, and a Pinecone account.

```bash
cp .env.example .env
uv sync --extra dev
```

Fill in `.env`. The two required secrets are:

```dotenv
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

Defaults create/use a serverless Pinecone index named `fruity-fun-dense` in AWS `us-east-1`. Change `PINECONE_CLOUD` and `PINECONE_REGION` if your Pinecone project uses another supported location. No Neo4j key or sparse Pinecone index is needed: BM25 runs over the checked-in JSON format locally.

## Add and ingest PDFs

1. Put one or more `.pdf` files in `data/pdfs/`.
2. Run:

```bash
uv run python scripts/ingest.py
```

The command writes `data/processed/fruit_chunks.json`, creates the Pinecone index if needed, and upserts vectors into the configured namespace. Re-running it is idempotent because chunk IDs are deterministic.

## Run the chatbot

```bash
uv run streamlit run app.py
```

Then open the local address printed by Streamlit. Try:

- `Why are strawberries unusual?`
- `Tell me three fun facts about mangoes.`
- `Draw mango, kiwi, grapes, and a tiny watermelon having a picnic.`

## Quality checks

```bash
uv run ruff check .
uv run pytest --cov=fruity_fun
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_LLM_MODEL` | Answering and reranking model | `gpt-5.6-terra` |
| `OPENAI_EMBEDDING_MODEL` | Dense embedding model | `text-embedding-3-small` |
| `OPENAI_IMAGE_MODEL` | Picture model | `gpt-image-2` |
| `PINECONE_DENSE_INDEX` | Dense vector index | `fruity-fun-dense` |
| `PINECONE_NAMESPACE` | Corpus version boundary | `fruit-corpus-v1` |
| `TOP_K_DENSE` / `TOP_K_SPARSE` | Candidate counts | `12` / `12` |
| `TOP_K_FINAL` | Final evidence count | `8` |
| `RRF_K` | RRF smoothing constant | `60` |
| `CONFIDENCE_THRESHOLD` | Grounding gate | `0.65` |

For a different embedding model, use a new Pinecone index because dimensions may differ.

## Repository map

```text
app.py                         Streamlit chat interface
scripts/ingest.py              PDF-to-JSON and Pinecone ingestion entry point
src/fruity_fun/agent.py        LangGraph safety/retrieval/answer/image flow
src/fruity_fun/corpus.py       PDF parsing, cleanup, and deterministic chunking
src/fruity_fun/retrieval.py    BM25, Pinecone search, RRF, and confidence
src/fruity_fun/ingest.py       Embedding and vector upsert pipeline
docs/RAG_FRAMEWORK.md          Filled-out RAG design framework
docs/EVALUATION_REPORT.md      Evaluation plan and failure analysis
docs/ARCHITECTURE.md           End-to-end architecture and failure paths
```

## Safety and privacy

Prompts are length-limited and screened before image generation. The picture prompt bans frightening or harmful content, visible text, brands, and photorealistic children. PDF passages and user questions are sent to configured OpenAI services; embeddings and chunk text are stored in your Pinecone project. Do not ingest confidential documents without the appropriate data controls.
