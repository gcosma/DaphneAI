"""
v2 alignment pipeline: cleaner, generalisable implementation.

This package is developed alongside the existing v1 code in `daphne_core`
and is intended to become the canonical path once it matches or improves
behaviour on the reference documents and new documents.

High-level responsibilities:
- `preprocess`: PDF/text → normalised text and structural spans.
- `recommendations`: extract structured recommendations from preprocessed text.
- `responses`: extract structured and scattered responses from preprocessed text.
- `alignment`: align recommendations with responses using configurable strategies.
"""

from .types import AlignmentResult, PreprocessedText, Recommendation, Response  # noqa: F401

__all__ = [
    "PreprocessedText",
    "Recommendation",
    "Response",
    "AlignmentResult",
]

