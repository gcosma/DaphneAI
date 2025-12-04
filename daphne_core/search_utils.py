"""Lightweight search utilities shared between UI components and core logic."""

import re
from typing import List, Sequence, Set

STOP_WORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
    "this",
    "but",
    "they",
    "have",
    "had",
    "what",
    "said",
    "each",
    "which",
    "she",
    "do",
    "how",
    "their",
    "if",
    "up",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
    "her",
    "would",
    "make",
    "like",
    "into",
    "him",
    "time",
    "two",
    "more",
    "go",
    "no",
    "way",
    "could",
    "my",
    "than",
    "first",
    "been",
    "call",
    "who",
    "also",
    "any",
    "new",
    "where",
    "much",
    "just",
    "after",
    "very",
    "well",
    "here",
    "should",
    "still",
}


def filter_stop_words(words: Sequence[str]) -> List[str]:
    """Remove common stop words and short tokens."""
    return [w for w in words if len(w) > 2 and w.lower() not in STOP_WORDS]


def get_meaningful_words(text: str) -> List[str]:
    """Extract meaningful (non-stopword) tokens from text."""
    if not text:
        return []
    tokens = re.findall(r"\b[\w-]+\b", text)
    return filter_stop_words(tokens)


__all__ = ["STOP_WORDS", "get_meaningful_words", "filter_stop_words"]
