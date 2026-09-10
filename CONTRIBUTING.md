# Contributing

Solo project, but the rules exist so future-me can review past-me.

## Workflow

1. Pick a task from [TASKS.md](TASKS.md), mark it `[~]`.
2. Branch: `phase-2/hybrid-retrieval`.
3. Commit in small pieces with real messages.
4. `make lint && make test && make eval` before opening a PR.
5. Mark the task `[x]` in the same PR that completes it.

## Commit messages

```
<type>(<scope>): <what changed>

feat(retrieval): add reciprocal rank fusion over dense and sparse
fix(ingestion): keep page numbers when a PDF has no text layer
eval(phase-2): record reranker ablation results
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `eval`, `chore`.

## Pull request rules

- Anything touching retrieval or generation must include eval numbers before and after.
- A PR that moves a metric down needs an explanation in the description.
- New behaviour needs a test. Bug fixes need the test that would have caught it.
