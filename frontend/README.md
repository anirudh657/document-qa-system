# Frontend

Next.js 15 (App Router) + TypeScript. Three screens over the FastAPI service.

## Running

The API must be up first, from the repo root:

```bash
uvicorn rag.api.main:app --reload
```

Then, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

http://localhost:3000

## Why there is no CORS configuration

There is nothing in the browser that talks to `:8000`. Every request goes to this app's
own origin at `/api/*`, and `next.config.mjs` rewrites it to the API server-side. One
origin means no preflight, no allow-list to keep in step with deploy URLs, and no
backend change.

Point `API_URL` at a deployed API to move it; no application code knows the address.

## Layout

```
app/
  page.tsx             Ask — answer, citations, and the passages behind them
  search/page.tsx      Passages only, with scores. No model involved.
  evaluation/page.tsx  Reads eval/runs/*.json from disk and compares them
lib/
  types.ts             Mirrors src/rag/api/schemas.py — the wire contract
  api.ts               The only module that calls the API
  runs.ts              Reads evaluation runs; McNemar's exact test
components/            Nav, Chevron, PassageRows
```

## The evaluation page reads files, not an endpoint

The runs are already JSON files versioned with the repo, and this app runs on the same
machine, so `lib/runs.ts` reads `../eval/runs/` directly. Adding `GET /v1/evaluation`
just to render a table would be work with no reader. The moment the frontend is deployed
away from the repo, that endpoint becomes the right answer.

Its McNemar implementation is verified against `scipy.stats.binomtest` across twelve
fixed/broken splits, including the 9/0 that gives p = 0.0039.

## Known gaps

- **`SearchHit` has no `page_end`.** The backend records which pages a passage spans, but
  the search schema does not expose it, so this cannot show "covers 2 pages" the way the
  design does. One field in `schemas.py` and one line in `routes.py`.
- **No Docker service.** The API is not containerised yet, so a frontend container would
  have nothing to proxy to inside compose. Both go in together or neither does.
- **No streaming.** `QueryRequest.stream` exists in the schema and is ignored here;
  answers appear whole.
