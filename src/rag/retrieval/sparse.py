"""BM25 sparse retrieval.

Exists because dense retrieval has a specific, predictable blind spot: it matches on
meaning, so terms carrying little meaning are nearly invisible to it. Exact
identifiers, product codes, rare proper nouns, section labels — "Experiment 1",
"CREATE DATABASE", "idx_email" — are exactly the queries where embeddings do worst
and where a user is most certain what they are looking for.

BM25 has the opposite profile. It scores documents on literal term overlap, weighting
rare terms heavily, and has no notion of meaning at all. A term appearing once in a
corpus scores enormously on a query containing it — which is precisely the case dense
retrieval buries.

Neither is better. They fail in different places, and that complementarity is the
whole argument for running both (see hybrid.py).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag.models import Chunk, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "without",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "its",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "again",
        "there",
        "here",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "as",
    ]
)
"""Words too common to carry retrieval signal.

Stopwords are removed from QUERIES only, never from the indexed corpus. BM25 already
discounts frequent terms by document frequency when scoring a document, but it cannot
undo the damage a long query does: "What is Experiment 1 in the OOP lab file?"
tokenises to nine terms of which seven appear almost everywhere, and the two that
matter get swamped.

Measured on the project's own golden set: without this, that query returned five
near-identically-scored pages (0.033, 0.032, 0.031, 0.031, 0.030) — the two
retrievers agreeing on the same wrong answer, which is the failure mode fusion cannot
fix. BM25 was designed for keyword queries; a natural-language question needs the
filler stripped before it behaves.

Deliberately a short hand-written list rather than a linguistic stopword set. Words
like "not" and "no" are excluded from removal elsewhere in some systems because they
carry meaning; here they are frequent enough across technical prose to be noise, and
BM25 has no notion of negation to preserve anyway.
"""


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    Underscores are kept inside tokens so identifiers like `idx_email` survive as one
    term rather than splitting into two common words. No stemming: it would damage the
    exact-match behaviour this retriever exists to provide.

    `drop_stopwords` is applied to queries, not to the corpus — see STOPWORDS. If
    filtering would remove every token, the unfiltered tokens are returned instead, so
    a query made entirely of common words still searches for something.
    """
    tokens = _TOKEN.findall(text.lower())
    if not drop_stopwords:
        return tokens
    filtered = [token for token in tokens if token not in STOPWORDS]
    return filtered or tokens


class SparseRetriever:
    """BM25 over the chunk corpus.

    The index is built from the chunks themselves and held in memory. `Chunk.embedding_text()`
    is used rather than the bare text so the contextual header ("MySQL Handbook > Page 5")
    is searchable too — a query naming the document or page should match it.
    """

    def __init__(self, chunks: Sequence[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self._bm25 = None
        if self.chunks:
            corpus = [tokenize(chunk.embedding_text()) for chunk in self.chunks]
            # A chunk that tokenises to nothing would make BM25 divide by zero.
            corpus = [tokens or ["\x00"] for tokens in corpus]
            self._bm25 = BM25Okapi(corpus)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the highest-scoring chunks for a query, best first.

        Chunks scoring zero are dropped rather than padded in. A zero BM25 score means
        no query term appeared at all, so returning them to fill `top_k` would put
        known-irrelevant text into the fusion pool and, downstream, into the prompt.
        """
        if self._bm25 is None or not self.chunks:
            return []

        tokens = tokenize(query, drop_stopwords=True)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((score, index) for index, score in enumerate(scores) if score > 0),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]

        return [
            RetrievedChunk(chunk=self.chunks[index], sparse_score=float(score))
            for score, index in ranked
        ]
