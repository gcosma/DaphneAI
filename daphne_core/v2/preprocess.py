"""
v2 preprocessing: PDF/text → normalised text and structural spans.

This module provides shared preprocessing used by both recommendation
and response extractors in the v2 pipeline.
"""

from pathlib import Path
from typing import Optional

from .types import PreprocessedText


def extract_text(
    pdf_path: Path,
    profile: Optional[str] = None,
) -> PreprocessedText:
    """
    Extract and normalise text from a PDF into a `PreprocessedText` object.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file to process.
    profile:
        Optional string indicating a document profile, e.g. "gov_uk_report".
        Profiles can apply document‑specific cleaning such as footer removal.

    Returns
    -------
    PreprocessedText
        Normalised text with sentence and page spans ready for downstream
        recommendation/response extraction.
    """
    raise NotImplementedError("v2.extract_text is not implemented yet")


