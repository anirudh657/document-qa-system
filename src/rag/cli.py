"""Command line interface.

    rag ingest ./data              index documents
    rag search "de morgan"         retrieval only, no LLM call
    rag ask "what are ..."         full pipeline, cited answer
    rag status                     configuration and index state

`search` exists separately from `ask` on purpose: retrieval is the half of a RAG
system that determines answer quality, and being able to inspect it without spending
an API request makes debugging retrieval much cheaper.

TODO(phase-3): eval
TODO(phase-4): eval-gate
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from rag.config import Settings, get_settings
from rag.generation.generator import Generator
from rag.generation.llm import LLMError, build_llm
from rag.ingestion.chunking import FixedSizeChunker
from rag.ingestion.embedding import build_embedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.store import IndexFingerprintMismatchError, VectorStore

app = typer.Typer(help="Document QA System", no_args_is_help=True)


def _store(settings: Settings) -> VectorStore:
    return VectorStore.connect(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.embedding_dim,
        settings.index_fingerprint(),
    )


def _embedder(settings: Settings):  # type: ignore[no-untyped-def]
    return build_embedder(
        settings.embedding_model,
        settings.embedding_dim,
        settings.embedding_batch_size,
    )


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to index"),
    recursive: bool = typer.Option(True, help="Recurse into subdirectories"),
    recreate: bool = typer.Option(False, help="Drop and rebuild the collection first"),
) -> None:
    """Index documents into the vector store."""
    settings = get_settings()
    source = Path(path)

    store = _store(settings)
    try:
        store.ensure_collection(recreate=recreate)
    except IndexFingerprintMismatchError as error:
        _fail(f"{error}\n\nRe-run with --recreate to rebuild the index.")

    pipeline = IngestionPipeline(
        FixedSizeChunker(settings.chunk_size, settings.chunk_overlap),
        _embedder(settings),
        store,
    )

    files = pipeline.discover(source, recursive)
    if not files:
        _fail(f"No supported documents found in {source}")

    typer.echo(f"Indexing {len(files)} file(s) from {source}\n")
    try:
        report = pipeline.run(source, recursive)
    except FileNotFoundError as error:
        _fail(str(error))

    for result in report.results:
        if result.ok:
            typer.secho(f"  ok    {result.path.name} — {result.chunks} chunks", fg="green")
        else:
            typer.secho(f"  skip  {result.path.name} — {result.skipped}", fg="yellow")

    typer.echo(
        f"\n{len(report.indexed)} indexed, {len(report.skipped)} skipped, "
        f"{report.total_chunks} chunks written. Collection holds {store.count()}."
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for"),
    top_k: int = typer.Option(5, help="Number of chunks to return"),
) -> None:
    """Retrieve chunks without calling the LLM. Free, and fast."""
    settings = get_settings()
    store = _store(settings)

    if not store.exists():
        _fail("No collection yet. Run 'rag ingest ./data' first.")

    hits = store.search(_embedder(settings).embed_query(query), top_k=top_k)
    if not hits:
        typer.echo("No matches.")
        return

    for rank, hit in enumerate(hits, start=1):
        page = f"page {hit.chunk.page_number}" if hit.chunk.page_number else "—"
        typer.secho(f"\n{rank}. {hit.final_score:.3f}  {page}", fg="cyan")
        typer.echo(f"   {hit.chunk.contextual_header}")
        typer.echo(f"   {hit.chunk.text[:220].strip()}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the corpus"),
    top_k: int = typer.Option(5, help="Chunks to retrieve before answering"),
) -> None:
    """Answer a question from the indexed documents, with citations."""
    settings = get_settings()
    store = _store(settings)

    if not store.exists():
        _fail("No collection yet. Run 'rag ingest ./data' first.")

    try:
        llm = build_llm(settings.llm_provider, settings.llm_model, settings.llm_key())
    except LLMError as error:
        _fail(str(error))

    passages = store.search(_embedder(settings).embed_query(question), top_k=top_k)

    try:
        answer = asyncio.run(Generator(llm).generate(question, passages))
    except LLMError as error:
        _fail(str(error))

    typer.echo(f"\n{answer.text}\n")

    if answer.citations:
        typer.secho("Sources", fg="cyan")
        for citation in answer.citations:
            page = f"page {citation.page_number}" if citation.page_number else "—"
            typer.echo(f"  [{citation.chunk_id[:8]}] {citation.source_path} — {page}")

    invented = Generator.validate_citations(answer, passages)
    if invented:
        typer.secho(
            f"\nWarning: the model cited {len(invented)} passage(s) that were never "
            f"retrieved: {', '.join(invented)}",
            fg="red",
        )

    typer.secho(
        f"\n{answer.input_tokens} in / {answer.output_tokens} out · "
        f"{answer.latency_ms} ms · {settings.llm_model}",
        dim=True,
    )


@app.command()
def status() -> None:
    """Show the current configuration and whether the index is reachable."""
    settings = get_settings()

    typer.secho("Configuration", fg="cyan")
    typer.echo(f"  embedding   {settings.embedding_model} ({settings.embedding_dim}d)")
    typer.echo(f"  chunking    size={settings.chunk_size} overlap={settings.chunk_overlap}")
    typer.echo(f"  llm         {settings.llm_provider} / {settings.llm_model}")
    typer.echo(f"  api key     {'set' if settings.llm_key() else 'MISSING'}")
    typer.echo(f"  fingerprint {settings.index_fingerprint()}")

    typer.secho("\nIndex", fg="cyan")
    try:
        store = _store(settings)
        if store.exists():
            typer.echo(f"  {settings.qdrant_url} — {store.count()} chunks indexed")
        else:
            typer.echo(f"  {settings.qdrant_url} — reachable, no collection yet")
    except Exception as error:
        typer.secho(f"  {settings.qdrant_url} — unreachable ({type(error).__name__})", fg="red")
        typer.echo("  Is Qdrant running? Try: docker compose up -d")


@app.command()
def eval(
    dataset: str = "eval/datasets/golden.jsonl",
    tags: str | None = None,
    compare: str | None = None,
) -> None:
    """Score the current build against the golden set."""
    raise NotImplementedError("TODO(phase-3)")


@app.command("eval-gate")
def eval_gate(max_drop: float = 0.02) -> None:
    """Exit non-zero when a metric regressed past the threshold."""
    raise NotImplementedError("TODO(phase-4)")


if __name__ == "__main__":
    app()
