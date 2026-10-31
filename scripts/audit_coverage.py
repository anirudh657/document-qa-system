"""Check the golden set against the index, before retrieval is involved at all.

**Why this comes before any more retrieval tuning.** Every retrieval metric in this
project asks "did search rank the right page highly?" None of them asks the prior
question: "is the right page in the index?" A page lost during ingestion — dropped as
boilerplate, rejected as a scan, or extracted with no text layer — is unreachable by
construction. No retriever, reranker or rewriter can surface it, and every experiment
run on top of it is measuring an ingestion bug while appearing to measure retrieval.

The suspicion this was written to test: all ten retrieval failures in the 34-question
golden set are questions about one document, and they pair up on pages — two different
questions expecting page 30 both fail, two expecting page 63 both fail. Failures that
cluster by page look like missing content, not like bad ranking.

Reads the index only. No model calls, no API quota, no network beyond Qdrant.

Run from the repo root:  python scripts/audit_coverage.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from rag.config import get_settings
from rag.evaluation.datasets import load_dataset, page_key
from rag.retrieval.store import VectorStore


def main() -> None:
    settings = get_settings()
    store = VectorStore.connect(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.embedding_dim,
        settings.index_fingerprint(),
    )

    chunks = store.iter_chunks()
    if not chunks:
        sys.exit("The index is empty. Run `rag ingest data/` first.")

    # Every page a chunk COVERS, which is what evaluation scores against. An earlier
    # version of this script read `page_number` alone — the page a chunk starts on —
    # and so reported the page-span fix as having changed nothing. A diagnostic that
    # predates the thing it is checking will confidently confirm the old behaviour.
    indexed: set[str] = set()
    chunks_per_page: Counter[str] = Counter()
    pages_per_source: defaultdict[str, set[int]] = defaultdict(set)
    empty_text = 0
    spanning = 0

    for chunk in chunks:
        covered: Sequence[int | None] = chunk.pages or [None]
        if len(covered) > 1:
            spanning += 1
        for page in covered:
            key = page_key(chunk.source, page)
            indexed.add(key)
            chunks_per_page[key] += 1
            if page is not None:
                pages_per_source[Path(chunk.source).stem].add(page)
        if not chunk.text.strip():
            empty_text += 1

    print(f"Index: {len(chunks)} chunk(s) covering {len(indexed)} distinct page(s).")
    print(f"  {spanning} chunk(s) carry a multi-page span (page_end set).")
    if spanning == 0:
        print("  WARNING: no chunk spans more than one page. Either every page is longer")
        print("  than a chunk, or the index predates page spans — re-run")
        print("  `rag ingest data/ --recreate`, since the fingerprint does not change when")
        print("  only the payload schema does and a stale index loads without complaint.")
    if empty_text:
        print(f"  WARNING: {empty_text} chunk(s) have no text — they can never match.")

    print("\nPage coverage per document")
    for source in sorted(pages_per_source):
        pages = pages_per_source[source]
        lo, hi = min(pages), max(pages)
        span = hi - lo + 1
        gaps = sorted(set(range(lo, hi + 1)) - pages)
        print(f"  {source}")
        print(f"    pages present: {len(pages)} of the {span}-page span {lo}-{hi}")
        if gaps:
            shown = ", ".join(str(g) for g in gaps[:25])
            more = f" … and {len(gaps) - 25} more" if len(gaps) > 25 else ""
            print(f"    MISSING pages: {shown}{more}")
            print("    A gap here is content that never reached the index. Questions")
            print("    pointing at these pages cannot be answered by any retriever.")
        else:
            print("    no gaps")

    # The question this script exists to answer.
    try:
        questions = load_dataset(Path("eval/datasets/golden.jsonl"))
    except Exception as error:
        sys.exit(f"\nCould not read the golden set: {error}")

    answerable = [q for q in questions if q.answerable]

    # Checked before per-page coverage, because the two failure modes look identical in
    # the metrics and have completely different fixes. A source name that matches no
    # indexed document fails every one of its questions at once — and this project has
    # already lost a whole evaluation to it, when questions named a file ("Handbook.pdf")
    # while chunks carried a title ("Handbook"). One rename fixes a dozen questions; a
    # missing page does not.
    indexed_stems = {key.rsplit("#p", 1)[0] for key in indexed}
    golden_stems = {Path(q.source).stem for q in answerable}
    orphan_stems = sorted(golden_stems - indexed_stems)

    print(
        f"\nGolden set: {len(answerable)} answerable question(s) over "
        f"{len(golden_stems)} source(s)."
    )

    if orphan_stems:
        print("\nSOURCE NAME MISMATCH — these names appear in no indexed document:")
        for stem in orphan_stems:
            count = sum(1 for q in answerable if Path(q.source).stem == stem)
            print(f"  {stem!r}  ({count} question(s) fail on this alone)")
        print("\n  Indexed documents are named:")
        for stem in sorted(indexed_stems):
            print(f"    {stem!r}")
        print("\n  This is a naming bug, not a retrieval result. Fix the `source` field")
        print("  in the dataset to match an indexed name, then re-run before reading any")
        print("  metric — these questions are currently unscoreable, not unanswerable.")

    unreachable = []
    partial = []
    for question in answerable:
        if Path(question.source).stem in orphan_stems:
            continue  # already reported above; not a page-level problem
        wanted = set(question.relevant_keys())
        found = wanted & indexed
        if not found:
            unreachable.append(question)
        elif found != wanted:
            partial.append((question, sorted(wanted - found)))

    print(
        f"\n  {len(unreachable)} question(s) name an indexed document but a page "
        "that is NOT in the index."
    )
    print(f"  {len(partial)} name several pages, some of which are missing.")

    if unreachable:
        print("\nUNREACHABLE PAGES — scored as retrieval failures, but cannot be found:")
        for question in unreachable:
            print(f"  {question.id}  wants {question.relevant_keys()}")
            print(f"        {question.question[:76]}")
        print("\n  Each of these is either an ingestion bug or a wrong page number in the")
        print("  dataset. Both are worth more than any retrieval technique: they cap the")
        print("  achievable score and add noise to every before/after comparison.")

    if partial:
        print("\nPARTIALLY REACHABLE — some expected pages are missing:")
        for question, missing in partial:
            print(f"  {question.id}  missing {missing}")

    if not unreachable and not partial:
        print("\n  Every expected page is present in the index. The failures are genuine")
        print("  ranking failures, and retrieval work is the right place to spend effort.")

    thin = [key for key, count in chunks_per_page.items() if count == 1]
    print(f"\n{len(thin)} page(s) are represented by a single chunk.")
    print("  Relevant when reading precision@5: that metric divides by the number of")
    print("  DISTINCT pages in the top-k, so pages split into many chunks shrink the")
    print("  divisor and raise the score without retrieval having improved.")


if __name__ == "__main__":
    main()
