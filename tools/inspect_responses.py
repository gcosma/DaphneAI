"""
Inspect responses extracted from a PDF using extract_response_sentences.

This mirrors the response extraction used by the Streamlit alignment tab
(`alignment_ui.render_simple_alignment_interface`) so we can see exactly what
responses are being detected, their inferred recommendation numbers, and
their types (structured section vs sentence-level).

Example usage (from repo root):

    python -m tools.inspect_responses \
        --pdf samplepdfs/Report1Response.pdf \
        --max-chars 200
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from daphne_core.integration_helper import extract_text_from_file
from daphne_core.alignment_engine import extract_response_sentences


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


def inspect_responses(pdf_path: Path, max_chars: int) -> None:
    text = load_text_from_pdf(pdf_path)
    print(f"Loaded response document {pdf_path} with length {len(text)} characters")

    responses: List[Dict[str, Any]] = extract_response_sentences(text)
    print(f"Total responses extracted: {len(responses)}")

    structured = [r for r in responses if r.get("response_type") == "structured"]
    sentence_level = [r for r in responses if r.get("response_type") == "sentence"]
    print(f"  Structured sections: {len(structured)}")
    print(f"  Sentence-level:      {len(sentence_level)}")

    print("\nSample responses:")
    for idx, resp in enumerate(responses, 1):
        snippet = (resp.get("text", "") or "").strip().replace("\n", " ")
        snippet = (snippet[:max_chars] + "...") if len(snippet) > max_chars else snippet
        rec_num = resp.get("rec_number")
        rtype = resp.get("response_type", "unknown")
        confidence = resp.get("confidence", 0.0)
        print("=" * 80)
        print(f"Response #{idx} | rec_num={rec_num} | type={rtype} | conf={confidence:.2f}")
        print(f"  snippet: {snippet}")

    print("\nDone inspecting responses.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect responses extracted from a PDF using extract_response_sentences.",
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
        help="Maximum number of characters to show per response snippet (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_responses(args.pdf, args.max_chars)


if __name__ == "__main__":
    main()

