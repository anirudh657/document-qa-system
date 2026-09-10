"""OpenTelemetry tracing across the request path.

Spans record what was retrieved, how reranking changed the ordering, the exact prompt
sent, and token counts. This is what turns "the answer was bad" into a diagnosable
event rather than a mystery.

TODO(phase-4): setup(), span helpers, cost accounting
"""


def setup(service_name: str, endpoint: str | None = None) -> None:
    raise NotImplementedError("TODO(phase-4)")
