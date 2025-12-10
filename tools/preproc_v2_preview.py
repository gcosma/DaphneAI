"""
Preview helper for the v2 preprocessing pipeline.

Runs `daphne_core.v2.preprocess.extract_text` on a PDF and writes:
- A JSON file with the full text, sentence spans, and page spans.
- A human-readable sentence-per-line dump.

Outputs are written into `output/` by default so they remain gitignored.

Usage (from repo root):

    python -m tools.preproc_v2_preview --pdf samplepdfs/Report1.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from daphne_core.v2.preprocess import extract_text

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v2 preprocessing on a PDF and dump structured output for inspection.",
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

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Running v2 preprocessing on %s", pdf_path)
    preprocessed = extract_text(pdf_path)

    stem = pdf_path.stem
    json_path = out_dir / f"{stem}_v2_preproc.json"
    sentences_path = out_dir / f"{stem}_v2_sentences.txt"

    logger.info("Writing JSON preview to %s", json_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "text": preprocessed.text,
                "sentence_spans": preprocessed.sentence_spans,
                "page_spans": preprocessed.page_spans,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Writing sentence-per-line dump to %s", sentences_path)
    with sentences_path.open("w", encoding="utf-8") as f:
        text = preprocessed.text
        for idx, (start, end) in enumerate(preprocessed.sentence_spans, 1):
            snippet = text[start:end].replace("\n", " ").strip()
            f.write(f"[{idx:04d}] {snippet}\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()

