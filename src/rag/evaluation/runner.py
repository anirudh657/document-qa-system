"""Eval harness.

Every run is stored with the git SHA and a config snapshot, so any number in the final
report can be traced back to the exact code and settings that produced it.

TODO(phase-3): run(), persistence, comparison against a previous run
TODO(phase-4): CI gate that fails on a >2% drop
"""

from pathlib import Path


class EvalRunner:
    def run(self, dataset: Path, tags: list[str] | None = None) -> dict:
        raise NotImplementedError("TODO(phase-3)")

    def compare(self, run_a: str, run_b: str) -> dict:
        raise NotImplementedError("TODO(phase-3)")

    def gate(self, max_drop: float = 0.02) -> bool:
        """Return False when any metric fell more than max_drop below baseline."""
        raise NotImplementedError("TODO(phase-4)")
