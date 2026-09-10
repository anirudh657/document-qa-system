"""Structured logging with trace correlation.

Secrets must never reach the log stream (NFR-7). Redaction is a processor in the
chain, not something remembered at each call site.

TODO(phase-4): configure(), redaction processor
"""


def configure(level: str = "INFO") -> None:
    raise NotImplementedError("TODO(phase-4)")
