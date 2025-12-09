"""
Inspect response headers used for rec_number inference in extract_response_sentences.

This helper shows what the current implementation sees when it scans for
\"Government response to recommendation N\" headers, so we can verify whether
the header-based rec_num extraction is correct before looking at body
cross-references.

Example usage (from repo root):

    python -m tools.inspect_response_headers \
        --pdf samplepdfs/Report1Response.pdf \
        --max-chars 200
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from daphne_core.integration_helper import extract_text_from_file
from daphne_core.alignment_engine import clean_pdf_artifacts


@dataclass
class UploadedFileShim:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def load_cleaned_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    uploaded = UploadedFileShim(pdf_path)
    raw_text = extract_text_from_file(uploaded) or ""
    return clean_pdf_artifacts(raw_text)


def inspect_response_headers(pdf_path: Path, max_chars: int) -> None:
    text = load_cleaned_text(pdf_path)
    print(f"Loaded cleaned response text from {pdf_path} (length={len(text)})")

    pattern = re.compile(r"Government\s+response\s+to\s+recommendation\s+(\d+)", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    print(f"Found {len(matches)} government response headers")

    for idx, m in enumerate(matches, 1):
        rec_num = m.group(1)
        start = max(0, m.start() - max_chars // 2)
        end = min(len(text), m.end() + max_chars // 2)
        snippet = text[start:end].replace("\n", " ")
        print("=" * 80)
        print(f"Header #{idx} | rec_num={rec_num}")
        print(f"  snippet: {snippet}")

    print("\nDone inspecting response headers.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect 'Government response to recommendation N' headers and their rec_num.",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the responses PDF (e.g. samplepdfs/Report1Response.pdf)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200,
        help="Maximum number of characters to show around each header (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_response_headers(args.pdf, args.max_chars)


if __name__ == "__main__":
    main()

