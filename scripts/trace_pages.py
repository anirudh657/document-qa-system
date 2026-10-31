"""Find out whether a 'missing' page's text is in the index under the wrong page label.

Two earlier scripts disagreed, and the disagreement is the point:

  audit_coverage.py   10 pages have no chunk in the index
  diagnose_pages.py    3 pages were dropped by the loader

The other 7 load fine and still have no chunk. They are lost between loading and
indexing, and there is a mechanism that would do it. Pages are tracked by character
offset into one concatenated document, and a chunk is attributed to the page it starts
in. This corpus averages ~132 tokens of text per page against a chunk_size of 160, so
a chunk routinely spans two or three pages — and a page whose text falls entirely
inside a chunk that began on the previous page gets no chunk of its own.

If that is what happened, the text is indexed and merely carries a neighbour's page
number. That distinction decides everything downstream:

  mislabelled   Retrieval may already be finding the right text and being scored as
                wrong. The defect is in page attribution, the fix is chunking or
                attribution, and the recorded chunk-size ablation was measuring
                attribution noise rather than retrieval quality.

  truly absent  The content is gone and ingestion has to be fixed.

Compares raw PDF text against the chunks actually in the index. No model, no quota.

Run from the repo root:  python scripts/trace_pages.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader

from rag.config import get_settings
from rag.evaluation.datasets import load_dataset, page_key
from rag.ingestion.loaders import _normalise, _strip_boilerplate, find_boilerplate
from rag.retrieval.store import VectorStore

DATA = Path("data")


RUN_WORDS = 6


def fingerprints(text: str, count: int = 8) -> list[str]:
    """Distinctive word runs from a page, for locating that page inside a chunk.

    The input must be the page text AFTER boilerplate stripping. A first version
    fingerprinted the raw page and found nothing anywhere: the index holds filtered
    text, so every phrase spanning a stripped running header failed to match and the
    script confidently reported present content as absent.

    Runs overlap rather than tile. A sparse page yields few words, and non-overlapping
    runs threw away the remainder — leaving some pages with a single phrase to match on.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text)]
    if len(words) < RUN_WORDS:
        return [" ".join(words)] if words else []
    runs = [
        " ".join(words[start : start + RUN_WORDS])
        for start in range(0, len(words) - RUN_WORDS + 1, 2)
    ]
    return runs[:count]


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

    # Normalised chunk text, so a phrase match is not defeated by whitespace.
    haystack = [
        (page_key(c.source, c.page_number), re.sub(r"\s+", " ", c.text).lower()) for c in chunks
    ]
    indexed_keys = {key for key, _ in haystack}

    # Filtered exactly as the loader filters, so a fingerprint is drawn from the text
    # that should be in the index rather than from text the loader already removed.
    raw_by_key: dict[str, str] = {}
    for path in sorted(DATA.rglob("*.pdf")):
        reader = PdfReader(str(path))
        pages = [_normalise(page.extract_text() or "") for page in reader.pages]
        boilerplate = find_boilerplate(pages)
        for number, raw in enumerate(pages, start=1):
            kept = _strip_boilerplate(raw, boilerplate) if boilerplate else raw
            raw_by_key[page_key(path.name, number)] = kept

    try:
        questions = load_dataset(Path("eval/datasets/golden.jsonl"))
    except Exception as error:
        sys.exit(f"Could not read the golden set: {error}")

    wanted: list[str] = []
    for question in questions:
        if question.answerable:
            wanted.extend(k for k in question.relevant_keys() if k not in indexed_keys)
    wanted = sorted(set(wanted))

    if not wanted:
        print("Every page the golden set points at has a chunk in the index.")
        return

    print(f"{len(wanted)} page(s) wanted by the golden set have no chunk of their own.\n")

    mislabelled = 0
    absent = 0
    unreadable = 0

    for key in wanted:
        raw = raw_by_key.get(key, "")
        if len(raw.strip()) < 40:
            unreadable += 1
            print(f"{key}: the PDF page itself yields {len(raw.strip())} chars — nothing to find.")
            continue

        phrases = fingerprints(raw)
        # Distinct phrases matched per page label, not total matches: one phrase can
        # appear in several overlapping chunks that share a label, which made an early
        # version print counts like "6/3 phrases".
        matched: dict[str, set[str]] = {}
        for phrase in phrases:
            for chunk_key, text in haystack:
                if phrase in text:
                    matched.setdefault(chunk_key, set()).add(phrase)
        hits = {key: len(found) for key, found in matched.items()}

        if hits:
            mislabelled += 1
            found = sorted(hits.items(), key=lambda kv: -kv[1])
            where = ", ".join(f"{k} ({n}/{len(phrases)} phrases)" for k, n in found[:3])
            print(f"{key}: TEXT IS INDEXED, labelled -> {where}")
        else:
            absent += 1
            print(f"{key}: not found in any chunk — the text really is absent.")

    print(f"\n{'=' * 70}")
    print(f"  {mislabelled:>3}  page(s) MISLABELLED — text is in the index under another page.")
    print(f"  {absent:>3}  page(s) genuinely ABSENT from the index.")
    print(f"  {unreadable:>3}  page(s) unreadable in the PDF (scan, blank or furniture only).")

    if mislabelled:
        print("\nFor the mislabelled pages retrieval may already be surfacing the correct")
        print("text and being scored as a failure, because evaluation matches on a page")
        print("number the chunk never carried. Those are attribution bugs, not retrieval")
        print("results — and the chunk-size ablation in EVALUATION.md was comparing")
        print("configurations whose page labels were differently wrong.")


if __name__ == "__main__":
    main()
