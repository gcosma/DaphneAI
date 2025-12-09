"""
Utility to dump raw text extracted from a PDF (via PyPDF2) to a .txt file.

This bypasses the higher-level document_processor cleaning so we can inspect
exactly what PyPDF2 returns for debugging issues like misread recommendation
numbers (e.g. "11" vs "1").

Example usage (from repo root):

    python -m tools.dump_pdf_raw --pdf samplepdfs/Report1.pdf --out output/Report1_raw.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import PyPDF2  # type: ignore[import-not-found]


def dump_pdf_raw(pdf_path: Path, out_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

    raw_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(raw_text, encoding="utf-8")

    print(f"Wrote raw PDF text for {pdf_path} to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump raw PyPDF2 text from a PDF to a .txt file.",
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
        help="Path to the output .txt file (e.g. output/Report1_raw.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_pdf_raw(args.pdf, args.out)


if __name__ == "__main__":
    main()

