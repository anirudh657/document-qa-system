# Document QA System

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions
over a private document corpus and cites its sources.

The goal of this project is not to demonstrate that RAG works — that takes an afternoon.
The goal is to build the parts that separate a demo from a system you would actually
run: measured retrieval quality, regression testing, cost and latency budgets, and
observability into why any given answer was produced.

---

## Status

| | |
|---|---|
| **Phase** | Phase 1 — Basic RAG |
| **Progress** | 0% |
| **Live demo** | _not deployed yet_ |
| **Docs** | [PRD](docs/PRD.md) · [Architecture](docs/ARCHITECTURE.md) · [Evaluation](docs/EVALUATION.md) · [Roadmap](docs/ROADMAP.md) |
| **Task board** | [TASKS.md](TASKS.md) |

---

## What it does

1. **Ingest** — loads PDFs, Markdown, HTML and plain text; splits them into chunks that
   preserve document structure; embeds each chunk and writes it to a vector store
   alongside its metadata.
2. **Retrieve** — for a user question, runs hybrid search (sparse BM25 + dense vectors),
   reranks the candidates with a cross-encoder, and returns the top passages.
3. **Generate** — builds a grounded prompt from those passages and streams an answer
   with inline citations back to the caller.
4. **Measure** — every change is scored against a golden question set for both retrieval
   quality and answer faithfulness, so improvements are provable rather than vibes.

---

## Quickstart

**Requirements:** Python 3.11+, Docker, and an API key for your chosen LLM provider.

```bash
git clone https://github.com/anirudh657/document-qa-system.git
cd document-qa-system

cp .env.example .env        # then fill in your keys
make setup                  # create venv + install dependencies
make services               # start Qdrant, Postgres and Redis via docker compose
make ingest DOCS=./data/sample
make dev                    # API on http://localhost:8000
```

Ask a question:

```bash
curl -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the refund window?", "top_k": 5}'
```

---

## Repository layout

```
document-qa-system/
├── src/rag/
│   ├── api/             FastAPI app, routes, request/response schemas
│   ├── ingestion/       loaders, chunking strategies, indexing pipeline
│   ├── retrieval/       dense, sparse, hybrid fusion, reranking, query rewriting
│   ├── generation/      prompt templates, answer synthesis, citation handling
│   ├── evaluation/      golden datasets, metrics, eval runner
│   └── observability/   structured logging, tracing, cost + token accounting
├── docs/                PRD, architecture, evaluation methodology, ADRs
├── eval/datasets/       golden question sets (JSONL)
├── tests/               unit and integration tests
├── scripts/             one-off operational scripts
└── notebooks/           exploration only — nothing here is imported by src/
```

---

## Make targets

| Command | What it does |
|---|---|
| `make setup` | Create the virtualenv and install dependencies |
| `make services` | Start Qdrant, Postgres, Redis via docker compose |
| `make dev` | Run the API with hot reload |
| `make ingest DOCS=<path>` | Index a directory of documents |
| `make test` | Run the unit and integration test suites |
| `make eval` | Score the current build against the golden set |
| `make lint` | Run ruff and mypy |
| `make fmt` | Format with ruff |

---

## Documentation

- **[PRD](docs/PRD.md)** — the problem, users, scope, requirements, and success metrics.
- **[Architecture](docs/ARCHITECTURE.md)** — components, data flow, the request path, and
  the reasoning behind each technology choice.
- **[Evaluation](docs/EVALUATION.md)** — how retrieval and generation quality are measured
  and what the current numbers are.
- **[Roadmap](docs/ROADMAP.md)** — the five delivery phases.
- **[ADRs](docs/adr/)** — dated records of decisions that were hard to reverse.

---

## License

MIT — see [LICENSE](LICENSE).
