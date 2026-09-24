"""Find which page of a PDF contains each marker string.

Golden questions are only as good as their expected pages. Estimating a page from
a block of extracted text is unreliable — this reads the PDF page by page and
reports where each answer actually lives.

    python find_pages.py "data/MySQL Handbook.pdf"
"""

import sys
from pathlib import Path

from pypdf import PdfReader

# Question id -> a distinctive string from its expected answer.
MARKERS = {
    "q-015": "MySQL Workbench is a visual tool",
    "q-016": "sudo apt install mysql-server",
    "q-017": "GRANT ALL PRIVILEGES",
    "q-018": "INSERT INTO users",
    "q-019": "SELECT * FROM users;",
    "q-020": "<>",
    "q-021": "IS NULL",
    "q-022": "Prevents duplicate values",
    "q-023": "AUTO_INCREMENT PRIMARY KEY",
    "q-024": "Must be NOT NULL and UNIQUE",
    "q-025": "AVG(salary)",
    "q-026": "COUNT(*) FROM users",
    "q-027": "is a foreign key",
    "q-028": "DROP FOREIGN KEY fk_user",
    "q-029": "ADD CONSTRAINT fk_user",
    "q-030": "ON DELETE CASCADE",
    "q-031": "DROP VIEW",
    "q-032": "not duplicated data",
    "q-033": "consume extra disk space",
    "q-034": "idx_gender_salary",
}


def main(path: Path) -> None:
    pages = [(n, (p.extract_text() or "")) for n, p in enumerate(PdfReader(str(path)).pages, 1)]

    for qid, marker in MARKERS.items():
        hits = [n for n, text in pages if marker.lower() in text.lower()]
        if not hits:
            print(f"{qid}: NOT FOUND — {marker!r}")
        elif len(hits) == 1:
            print(f"{qid}: page {hits[0]}")
        else:
            print(f"{qid}: pages {hits}   (marker appears {len(hits)}x)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/MySQL Handbook.pdf"))