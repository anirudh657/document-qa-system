"""Explain, page by page, why a page never reached the index.

`audit_coverage.py` establishes WHICH pages are missing. This answers WHY, and the
answer decides the fix, because two very different problems look identical downstream:

  NO TEXT LAYER   `extract_text()` returns nothing. The page is a screenshot, a photo
                  or a scan. Our code is behaving correctly and the remedy is OCR, or
                  correcting a golden question that points at an unreadable page.

  FILTERED OUT    Extraction returned real text and our own pipeline then removed all
                  of it — boilerplate stripping also drops any line that is purely
                  digits, so a page of short repeated lines and numbers can be emptied
                  completely. This is our bug, and the content is recoverable.

The loader skips an emptied page with a bare `continue`: no log, no counter, no warning.
The document still loads, `rag ingest` still reports success, and the page is simply
gone. This script exists because nothing else in the pipeline would ever tell you.

Imports the real loader internals rather than reimplementing them. A reimplementation
that drifted from the loader would produce a confident, wrong diagnosis.

Reads PDFs only. No index, no model, no network.

Run from the repo root:
    python scripts/diagnose_pages.py
    python scripts/diagnose_pages.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader

from rag.ingestion.loaders import (
    _normalise,
    _strip_boilerplate,
    find_boilerplate,
)

DATA = Path("data")


def diagnose(path: Path, show_all: bool) -> tuple[int, int, int]:
    reader = PdfReader(str(path))
    raw_pages = [_normalise(page.extract_text() or "") for page in reader.pages]
    boilerplate = find_boilerplate(raw_pages)

    needed = max(3, int(len(raw_pages) * 0.6))

    print(f"\n{path.name}")
    print(
        f"  {len(raw_pages)} page(s); {len(boilerplate)} line(s) treated as boilerplate "
        f"(appearing on {needed}+ pages)"
    )
    for line in sorted(boilerplate)[:6]:
        print(f"    repeated: {line[:70]!r}")

    # Two earlier versions of this function tried to label the cause automatically and
    # both got it wrong: judging by whether raw text existed called every image-only
    # page a bug (the running header is still text), and judging by page size called a
    # page whose "Aim/Procedure/Result" template was stripped a scan. The cause is not
    # reliably recoverable from the text alone, so this reports the evidence and names
    # the category by what was actually removed, leaving the judgement visible.
    furniture = 0
    stripped_body = 0
    blank = 0

    for number, raw in enumerate(raw_pages, start=1):
        kept = _strip_boilerplate(raw, boilerplate) if boilerplate else raw
        if kept and not show_all:
            continue

        removed = [ln.strip() for ln in raw.splitlines() if ln.strip() and ln.strip() not in kept]
        substantial = [ln for ln in removed if len(ln) > 40 and not ln.isdigit()]

        if kept:
            verdict = "kept"
        elif not removed:
            blank += 1
            verdict = "BLANK — no text of any kind; a scan, a photo or an empty page"
        elif substantial:
            stripped_body += 1
            verdict = f"BODY TEXT REMOVED — {len(substantial)} substantial line(s), see below"
        else:
            furniture += 1
            verdict = "FURNITURE ONLY — header, footer and page number; no body text"

        print(f"  page {number:>3}  raw {len(raw):>5} chars -> kept {len(kept):>5}  {verdict}")

        if not kept:
            for line in removed[:8]:
                if line.isdigit():
                    why = "bare page number"
                elif line in boilerplate:
                    why = f"repeated on >={needed} pages"
                else:
                    why = "removed"
                print(f"           {why:<26} {line[:50]!r}")

    return blank, furniture, stripped_body


def main() -> None:
    show_all = "--all" in sys.argv
    if not DATA.is_dir():
        sys.exit(f"No {DATA}/ directory. Run from the repo root.")

    pdfs = sorted(DATA.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs under {DATA}/.")

    totals = [diagnose(path, show_all) for path in pdfs]
    blank = sum(t[0] for t in totals)
    furniture = sum(t[1] for t in totals)
    stripped = sum(t[2] for t in totals)

    print(f"\n{'=' * 70}")
    print(f"{blank + furniture + stripped} page(s) never reached the index.\n")

    print(f"  {stripped:>3}  BODY TEXT REMOVED — substantial lines stripped as repeated.")
    print("       Read the lines printed above. If they are a per-page template")
    print("       ('Aim:', 'Procedure:', 'Result:') the page has no unique content and")
    print("       dropping it is correct — the golden question pointing at it is wrong.")
    print("       If they are real content, boilerplate detection is too aggressive and")
    print("       the content is recoverable.\n")

    print(f"  {furniture:>3}  FURNITURE ONLY — header, footer and page number, no body.")
    print("       Nothing to retrieve. A golden question aimed here needs its page")
    print("       number corrected.\n")

    print(f"  {blank:>3}  BLANK — no text layer at all; a scan, photo or empty page.")
    print("       Needs OCR, or the question needs repointing.\n")

    print("In every case the loader dropped the page with a bare `continue`: no log, no")
    print("counter, no warning. That silence is a defect regardless of which category a")
    print("page falls into, and it is why these pages went unnoticed for the whole")
    print("project while `rag ingest` reported success.")


if __name__ == "__main__":
    main()
