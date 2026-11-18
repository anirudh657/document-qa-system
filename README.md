# Document QA

A Retrieval-Augmented Generation (RAG) system that answers questions from a private
document corpus, cites the page behind every claim, and refuses when the documents do
not support an answer.

The interesting part of this project is not that RAG works — that takes an afternoon.
It is the measurement harness around it: a page-keyed golden dataset, retrieval metrics,
and a statistical test that says whether a change actually helped or whether the numbers
just moved.

---

## Result

Retrieval accuracy on the 34-question golden set (32 of them answerable):

| | 2026-09-24 | 2026-11-18 |
|---|---|---|
| hit rate@5 | 0.625 | **0.938** |
| MRR | — | 0.678 |
| NDCG@10 | — | 0.729 |
| recall@5 | — | 0.906 |

**10 questions fixed, 0 broken. McNemar's exact test, p = 0.0020.**

That improvement did not come from adding a retrieval technique. It came from finding a
page-attribution bug: chunks were labelled with the page they *started* on, but this
corpus averages ~132 tokens of text per page against a chunk size of 160, so most chunks
spanned a page boundary and ten pages had no chunk starting in them at all. Seven of the
eleven apparent "retrieval failures" were unreachable by construction — the right answer
was in the index, under the wrong page number.

It was only visible because the system was being measured. That is the whole argument of
the project.

---

## What is built

- **Ingestion** — PDF, Word, Markdown, HTML and plain text, split into overlapping
  passages that record the full page span they cover.
- **Retrieval** — dense vector search over Qdrant using `BAAI/bge-base-en-v1.5` (768d).
  BM25 sparse search, Reciprocal Rank Fusion, cross-encoder reranking and LLM query
  expansion are all implemented and switchable.
- **Generation** — Google Gemini, constrained to the retrieved passages. Every citation
  is validated against what was actually retrieved, so an invented citation is a fault
  rather than something the reader has to catch.
- **Web UI** — Next.js 16 / React 19 / TypeScript. Three screens: ask a question, inspect
  raw retrieval, and read the evaluation history. Documents can be uploaded from the
  browser, and retrieval can be scoped to a single document.
- **Evaluation** — 34 hand-written questions keyed on *pages*, not chunk IDs, so the same
  dataset stays valid after re-chunking. Scored without a language model, so a run is
  free and deterministic.
- **CI** — 330 tests, plus `ruff` and `mypy --strict`, on every push.

## What was measured and turned off

Recorded here because a negative result nobody writes down gets re-discovered:

- **Hybrid retrieval (BM25 + dense)** — no measurable gain on this corpus. Disabled.
- **Cross-encoder reranking** — measurably *harmful*. Disabled.
- **Query rewriting** — improves ranking (MRR +0.063, 6 questions improved against 2
  worsened) but at 32 questions the Wilcoxon signed-rank test gives p = 0.0921, which is
  not enough to call. Roughly 64 questions would settle it.

Both of the first two were measured against a baseline where seven questions were
unscoreable, so neither got a fair test and both are worth re-measuring. See
[docs/EVALUATION.md](docs/EVALUATION.md).

## Known failures

Two of the 32 questions still miss, and both are documented in the golden set:

- **q-022** — retrieval brackets the target page, returning 25–29 and 41 without ever
  landing on page 30, where the summary table lives.
- **q-026** — a vocabulary gap: the question says "count only the rows matching a
  condition" and never uses the word `COUNT`.

---

## Running it

**Requirements:** Python 3.12, Docker, Node 20+, and a Google Gemini API key.

```powershell
git clone https://github.com/anirudh657/document-qa-system.git
cd document-qa-system

copy .env.example .env          # then add your GEMINI_API_KEY
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,eval]"
```

Then three terminals:

```powershell
# 1 — vector store (plus Postgres and Redis)
docker compose up -d

# 2 — API on http://localhost:8000
uvicorn rag.api.main:app --reload --reload-dir src

# 3 — web UI on http://localhost:3000
cd frontend
npm install
npm run dev
```

`--reload-dir src` matters: without it the file watcher crawls `frontend/node_modules`
and the server restarts in a loop.

Index some documents and score the result:

```powershell
rag ingest data\ --recreate   # build the index
rag status                    # what is indexed, and is everything reachable
rag eval                      # score retrieval, writes to eval/runs/
```

On Linux and macOS the `Makefile` wraps the same commands (`make services`, `make dev`,
`make eval`). It assumes a POSIX virtualenv layout, so it does not work on Windows — use
the commands above.

### Corpus

`data/` is gitignored, so a fresh clone starts with an empty index. Put your own PDFs
there and run `rag ingest data\ --recreate`, or upload them from the web UI. The golden
dataset in `eval/datasets/golden.jsonl` is written against one specific corpus, so the
evaluation numbers only reproduce with those same documents.

---

## Repository layout

```
document-qa-system/
├── src/rag/
│   ├── api/             FastAPI app, routes, request/response schemas
│   ├── ingestion/       loaders, chunking, indexing pipeline
│   ├── retrieval/       dense, sparse, hybrid fusion, reranking, query rewriting
│   ├── generation/      prompt templates, answer synthesis, citation validation
│   ├── evaluation/      golden datasets, metrics, eval runner
│   └── observability/   structured logging, cost and token accounting
├── frontend/            Next.js web UI
├── docs/                PRD, architecture, evaluation methodology, ADRs
├── eval/datasets/       golden question sets (JSONL)
├── eval/runs/           every scored run, kept in version control as the record
├── tests/               unit and integration tests
└── scripts/             diagnostic and one-off scripts
```

---

## Documentation

- **[Evaluation](docs/EVALUATION.md)** — how quality is measured, every run, and what is
  still unresolved. Start here.
- **[PRD](docs/PRD.md)** — problem, users, scope, success metrics.
- **[Architecture](docs/ARCHITECTURE.md)** — components, data flow, and why each
  technology was chosen.
- **[Roadmap](docs/ROADMAP.md)** — delivery phases.
- **[ADRs](docs/adr/)** — dated records of decisions that were hard to reverse.

---

## License

MIT — see [LICENSE](LICENSE).
