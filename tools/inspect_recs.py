"""
Utility to inspect numbered recommendation blocks extracted from a PDF.

This is meant for debugging/analysis of how StrictRecommendationExtractor
handles "Recommendation N ..." sections in documents like Report1.pdf.

Example usage (from repo root):

    python -m tools.inspect_recs --pdf samplepdfs/Report1.pdf --max-chars 400

It will print each numbered block, whether the extractor keeps or drops it,
and how it is classified (garbage/meta/genuine, inferred number, etc.).
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from daphne_core.integration_helper import extract_text_from_file
from daphne_core.recommendation_extractor import StrictRecommendationExtractor

logger = logging.getLogger(__name__)


@dataclass
class UploadedFileShim:
    """Minimal shim to mimic Streamlit's uploaded file interface."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def load_text_from_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    uploaded = UploadedFileShim(path)
    return extract_text_from_file(uploaded) or ""


def inspect_numbered_recommendations(pdf_path: Path, min_confidence: float, max_chars: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Inspecting numbered recommendations in %s", pdf_path)
    text = load_text_from_pdf(pdf_path)
    logger.info("Loaded text of length %d", len(text))

    extractor = StrictRecommendationExtractor()

    # Use the same heading-based segmentation as StrictRecommendationExtractor,
    # including running over the CLEANED full text.
    cleaned_full_text = extractor.clean_text(text)
    heading_pattern = re.compile(
        r"(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b",
        re.IGNORECASE,
    )
    heading_matches = list(heading_pattern.finditer(cleaned_full_text))
    logger.info("Found %d numbered recommendation headings", len(heading_matches))

    # Run the normal extraction once so we know which positions are kept
    extracted = extractor.extract_recommendations(text, min_confidence=min_confidence)
    kept_positions = {rec.get("position") for rec in extracted if rec.get("in_section")}

    for idx, match in enumerate(heading_matches):
        start = match.start()
        end = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(cleaned_full_text)
        raw_block = cleaned_full_text[start:end]
        cleaned = extractor.clean_text(raw_block)
        is_garbage, garbage_reason = extractor.is_garbage(cleaned, is_numbered_rec=True)
        is_meta = extractor.is_meta_recommendation(cleaned)
        is_rec, conf, method, verb = extractor.is_genuine_recommendation(cleaned, is_numbered_rec=True)

        # Prefer the number inferred from the cleaned text (method),
        # fall back to the raw heading match if needed. Heading pattern already
        # supports spaced digits like "1 1".
        heading_num = match.group(1)
        extra_digit = match.group(2)
        if extra_digit and len(heading_num) == 1:
            heading_rec_num = f"{heading_num}{extra_digit}"
        else:
            heading_rec_num = heading_num

        rec_num = heading_rec_num
        if method.startswith("numbered_recommendation_"):
            try:
                rec_num = method.split("_")[-1]
            except Exception:
                rec_num = heading_rec_num

        kept = idx in kept_positions
        status_flags = []
        status_flags.append("KEPT" if kept else "DROPPED")
        if is_garbage:
            status_flags.append(f"GARBAGE:{garbage_reason}")
        if is_meta:
            status_flags.append("META")
        if is_rec:
            status_flags.append("GENUINE")

        snippet = cleaned[:max_chars].replace("\n", " ")
        raw_preview = raw_block[:max_chars].replace("\n", " ")
        print("=" * 80)
        print(f"Block #{idx} | rec_num={rec_num} | pos={idx} | {' | '.join(status_flags)}")
        print(f"  method={method}, verb={verb}, confidence={conf:.3f}")
        print(f"  CLEANED snippet: {snippet}")
        print(f"  RAW     snippet: {raw_preview!r}")

    print("\nDone inspecting numbered recommendations.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect numbered recommendation blocks extracted from a PDF.",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the PDF to inspect (e.g. samplepdfs/Report1.pdf)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.75,
        help="Minimum confidence used by the strict extractor (default: 0.75)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Maximum number of characters to show per block snippet (default: 400)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_numbered_recommendations(args.pdf, args.min_confidence, args.max_chars)


if __name__ == "__main__":
    main()
