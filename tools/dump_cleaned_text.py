"""
Dump the extractor-cleaned text for a PDF to a .txt file.

This applies StrictRecommendationExtractor.clean_text to the processed text
produced by extract_text_from_file, so we can verify behaviours like footer
removal in isolation (e.g. around Recommendation 8).

Example usage (from repo root):

    python -m tools.dump_cleaned_text \
        --pdf samplepdfs/Report1.pdf \
        --out output/Report1_footers_removed.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from daphne_core.integration_helper import extract_text_from_file
from daphne_core.recommendation_extractor import StrictRecommendationExtractor


@dataclass
class UploadedFileShim:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def dump_cleaned_text(pdf_path: Path, out_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    uploaded = UploadedFileShim(pdf_path)
    processed_text = extract_text_from_file(uploaded) or ""

    extractor = StrictRecommendationExtractor()
    cleaned_text = extractor.clean_text(processed_text)

    # Simple debug: count footer marker occurrences before/after cleaning.
    footer_marker = (
        "Rapid review into data on mental health inpatient settings: final report "
        "and recommendations - GOV.UK"
    )
    before = processed_text.lower().count(footer_marker.lower())
    after = cleaned_text.lower().count(footer_marker.lower())
    print(f"Footer marker occurrences: before_clean={before}, after_clean={after}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned_text, encoding="utf-8")

    print(f"Wrote extractor-cleaned text for {pdf_path} to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump StrictRecommendationExtractor.clean_text(processed_text) for a PDF.",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the PDF to dump (e.g. samplepdfs/Report1.pdf)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to the output .txt file (e.g. output/Report1_footers_removed.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_cleaned_text(args.pdf, args.out)


if __name__ == "__main__":
    main()

