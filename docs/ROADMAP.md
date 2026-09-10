# Roadmap

Five phases. Each ends with something demonstrable, and nothing in a later phase starts
before the previous phase's exit criterion is met.

| Phase | Theme | Exit criterion |
|---|---|---|
| **1** | Basic RAG | An end-to-end answer with citations, and a recorded baseline score |
| **2** | Advanced Retrieval | Retrieval metrics beat the Phase 1 baseline, with an ablation table showing which change did what |
| **3** | Evaluation System | Eval runs in CI and blocks a regression |
| **4** | Production Features | Deployed with tracing, caching, auth and cost tracking |
| **5** | Final Polish | Load-tested, documented, demoed, with a before/after benchmark table |

The ordering matters and is not arbitrary. Phase 3 comes after Phase 2 rather than before
because you need something worth measuring first — but the *baseline* is recorded at the
end of Phase 1, because a before/after comparison with no "before" is worthless.

Task-level detail lives in [TASKS.md](../TASKS.md).
