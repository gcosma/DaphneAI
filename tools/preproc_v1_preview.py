"""
Preview helper for the v1-style preprocessing / sentence splitting.

This is intended to provide a point of comparison with the v2 pipeline:
it uses the same text extractor as the legacy app (integration_helper)
and a simple regex-based sentence splitter, and writes:
- A JSON file with the full text and sentence spans.
- A human-readable sentence-per-line dump.

Outputs are written into `output/` by default so they remain gitignored.

Usage (from repo root):

    python -m tools.preproc_v1_preview --pdf samplepdfs/Report1.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from daphne_core.integration_helper import extract_text_from_file

logger = logging.getLogger(__name__)


SentenceSpan = Tuple[int, int]


@dataclass
class UploadedFileShim:
    """Minimal shim to mimic Streamlit's uploaded file interface."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def load_text_from_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    uploaded = UploadedFileShim(pdf_path)
    text = extract_text_from_file(uploaded) or ""
    return text


def sentence_spans_regex(text: str) -> List[SentenceSpan]:
    """
    Simple regex-based sentence splitter.

    This mirrors the v1-style approach where text is split on punctuation
    boundaries, without layout awareness.
    """
    spans: List[SentenceSpan] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v1-style preprocessing on a PDF and dump sentence splits for inspection.",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the PDF to process (e.g. samplepdfs/Report1.pdf)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write preview files into (default: output/)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
    args = parse_args()

    pdf_path: Path = args.pdf
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running v1-style preprocessing on %s", pdf_path)
    text = load_text_from_pdf(pdf_path)

    spans = sentence_spans_regex(text)

    stem = pdf_path.stem
    json_path = out_dir / f"{stem}_v1_preproc.json"
    sentences_path = out_dir / f"{stem}_v1_sentences.txt"

    logger.info("Writing JSON preview to %s", json_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "text": text,
                "sentence_spans": spans,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Writing sentence-per-line dump to %s", sentences_path)
    with sentences_path.open("w", encoding="utf-8") as f:
        for idx, (start, end) in enumerate(spans, 1):
            snippet = text[start:end].replace("\n", " ").strip()
            f.write(f"[{idx:04d}] {snippet}\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()

