"""Command line interface.

    rag ingest ./data
    rag eval --tags multi-hop
    rag eval-gate --max-drop 0.02

TODO(phase-1): ingest
TODO(phase-3): eval
TODO(phase-4): eval-gate
"""

import typer

app = typer.Typer(help="Document QA System")


@app.command()
def ingest(path: str, recursive: bool = True) -> None:
    """Index a file or directory."""
    raise NotImplementedError("TODO(phase-1)")


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
