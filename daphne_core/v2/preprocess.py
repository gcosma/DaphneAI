"""
v2 preprocessing: PDF → normalised text and structural spans.

This module provides shared preprocessing used by both recommendation
and response extractors in the v2 pipeline.

For now we implement a single canonical pipeline based on:
- `unstructured.partition_pdf` for PDF → elements (layout-aware), and
- `syntok` for sentence segmentation.

Future document-specific tweaks (e.g. GOV.UK-specific footer/header handling)
should be driven by the `profile` parameter, but we deliberately defer that
until we have exhausted what the centralised pipeline can do on its own.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .types import PreprocessedText


Span = Tuple[int, int]
logger = logging.getLogger(__name__)


def _require_unstructured() -> "callable":
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "unstructured is required for v2 preprocessing. "
            "Install it with `pip install unstructured[local-inference]`."
        ) from exc
    return partition_pdf


def _require_syntok():
    try:
        from syntok import segmenter  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "syntok is required for v2 sentence segmentation. "
            "Install it with `pip install syntok`."
        ) from exc
    return segmenter


def _build_text_and_pages(pdf_path: Path) -> Tuple[str, List[Span]]:
    """
    Use unstructured to convert a PDF into a single text string and page spans.

    We group elements by their page number and concatenate text per page,
    separating pages with a double newline. Page spans are offsets into the
    final combined text.
    """
    partition_pdf = _require_unstructured()
    elements = partition_pdf(filename=str(pdf_path))

    # Group text chunks by page number while preserving original order.
    pages: dict[int, List[str]] = {}
    for el in elements:
        text = getattr(el, "text", "") or ""
        if not text.strip():
            continue
        metadata = getattr(el, "metadata", None)
        page_number = getattr(metadata, "page_number", None)
        if page_number is None:
            # Treat elements without page numbers as part of page 1.
            page_number = 1
        pages.setdefault(page_number, []).append(text.strip())

    # Heuristic header/footer detection via repetition across pages.
    # We look at the top/bottom few elements on each page and identify
    # lines that repeat on many pages; these are treated as structural
    # artefacts (e.g. GOV.UK banners, URLs, timestamps).
    def norm_line(line: str) -> str:
        """Normalise a line for header/footer repetition checks."""
        stripped = " ".join(line.split())
        if not stripped:
            return ""
        # Normalise common dynamic parts so that, e.g., "16/84" and "17/84"
        # are treated as the same pattern.
        stripped = re.sub(r"\b\d+/\d+\b", "PAGE_NUM/TOTAL", stripped)
        # Normalise simple date/time patterns.
        stripped = re.sub(
            r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?",
            "DATE_TIME",
            stripped,
            flags=re.IGNORECASE,
        )
        return stripped

    total_pages = len(pages)
    header_footer_counts: dict[str, int] = {}
    top_k = 2
    bottom_k = 3

    for _, chunks in pages.items():
        if not chunks:
            continue
        # Top band
        for line in chunks[:top_k]:
            norm = norm_line(line)
            if len(norm) < 10:
                continue
            header_footer_counts[norm] = header_footer_counts.get(norm, 0) + 1
        # Bottom band
        for line in chunks[-bottom_k:]:
            norm = norm_line(line)
            if len(norm) < 10:
                continue
            header_footer_counts[norm] = header_footer_counts.get(norm, 0) + 1

    repeated_lines: set[str] = set()
    min_count = 0
    if total_pages > 0:
        min_count = max(2, int(0.3 * total_pages))
        for line, count in header_footer_counts.items():
            if count >= min_count:
                repeated_lines.add(line)

    if repeated_lines:
        logger.info(
            "v2 preprocess: identified %d repeated header/footer lines (min_count=%d)",
            len(repeated_lines),
            min_count,
        )

    page_spans: List[Span] = []
    current = ""

    for page_number in sorted(pages.keys()):
        filtered_chunks: List[str] = []
        for line in pages[page_number]:
            norm = norm_line(line)
            if norm in repeated_lines:
                continue
            filtered_chunks.append(line)

        # Join elements on a page with a *blank* line between them so that
        # downstream sentence/paragraph segmentation (via syntok) can more
        # easily treat headings, footers, and body text as separate units.
        # This mirrors the earlier bake-off behaviour that cleanly separated
        # "Recommendation 8", footers, and paragraph text into distinct lines.
        page_text = "\n\n".join(filtered_chunks).strip()
        if not page_text:
            continue
        if current:
            current += "\n\n"
        start = len(current)
        current += page_text
        end = len(current)
        page_spans.append((start, end))

    full_text = current
    return full_text, page_spans


def _build_sentence_spans(text: str) -> List[Span]:
    """
    Use syntok to segment `text` into sentences and return their spans.

    We reconstruct each sentence's surface form from syntok tokens and then
    locate it in the original text, scanning forward to keep complexity low.
    """
    segmenter = _require_syntok()
    spans: List[Span] = []
    offset = 0
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            sent_text = "".join(token.spacing + token.value for token in sentence).lstrip()
            if not sent_text:
                continue
            start = text.find(sent_text, offset)
            if start == -1:
                continue
            end = start + len(sent_text)
            spans.append((start, end))
            offset = end
    return spans


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
        Reserved for future document-specific tweak profiles. For now the
        pipeline is purely general and does not branch on this value.

    Returns
    -------
    PreprocessedText
        Normalised text with sentence and page spans ready for downstream
        recommendation/response extraction.
    """
    # Currently we ignore `profile` and run a single canonical pipeline based
    # on unstructured + syntok. Future profiles should be implemented as
    # small, explicit modifications on top of this core, not silent fallbacks.
    full_text, page_spans = _build_text_and_pages(pdf_path)
    sentence_spans = _build_sentence_spans(full_text)
    return PreprocessedText(text=full_text, sentence_spans=sentence_spans, page_spans=page_spans)
