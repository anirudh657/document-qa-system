# ADR 0001 — No orchestration framework

**Status:** Accepted
**Date:** 2026

## Context

LangChain and LlamaIndex both provide RAG pipelines out of the box: loaders, chunkers,
retrievers, rerankers and chains, wired together. Using one would make Phase 1 a day's
work instead of a week's.

## Decision

Call the vector store, embedding model, reranker and LLM directly. No orchestration
framework.

## Consequences

**Accepted cost:** more code to write and maintain in Phase 1. Some wheel reinvention.

**Reason:** the retrieval internals are the point of this project. Hybrid fusion,
reranking behaviour, chunking strategy and the exact contents of the prompt are the things
being evaluated and written about, and a framework puts an abstraction between the author
and all of them. Debugging a retrieval quality problem through three layers of framework
indirection is worse than writing the two hundred lines it replaces.

**Revisit if:** the project grows to need agentic tool use or multi-step planning, where
the framework's value is real rather than cosmetic.
