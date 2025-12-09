"""
Dump the cleaned/processed text for a PDF as seen by the extractor.

This goes through the same stack as the app:
    integration_helper.extract_text_from_file -> document_processor.* -> clean_extracted_text

so we can inspect what `StrictRecommendationExtractor` actually receives,
separately from the raw PyPDF2 output.

Example usage (from repo root):

    python -m tools.dump_processed_text \
        --pdf samplepdfs/Report1.pdf \
        --out output/Report1_processed.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from daphne_core.integration_helper import extract_text_from_file


@dataclass
class UploadedFileShim:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def dump_processed_text(pdf_path: Path, out_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    uploaded = UploadedFileShim(pdf_path)
    text = extract_text_from_file(uploaded) or ""

    # Debug: count how many GOV.UK footer blobs are still present in the
    # processed text. This helps verify whether footer removal heuristics in
    # the extractor/cleaning stack are effective.
    footer_marker = (
        "Rapid review into data on mental health inpatient settings: final report "
        "and recommendations - GOV.UK"
    )
    footer_count = text.lower().count(footer_marker.lower())
    print(f"Footers present in processed text: {footer_count}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(f"Wrote processed text for {pdf_path} to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump processed (cleaned) text for a PDF as seen by the extractor.",
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
        help="Path to the output .txt file (e.g. output/Report1_processed.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_processed_text(args.pdf, args.out)


if __name__ == "__main__":
    main()
