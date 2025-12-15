"""
Inspect responses extracted by the v2 pipeline from a PDF.

This helper runs:
- v2 preprocessing (`daphne_core.v2.preprocess.extract_text`), then
- v2 response extraction (`ResponseExtractorV2.extract`),
and prints a concise summary of structured and action-verb responses.

Usage (from repo root):

    python -m tools.inspect_responses_v2 --pdf samplepdfs/Report1Response.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from daphne_core.v2.preprocess import extract_text
from daphne_core.v2.responses import ResponseExtractorV2

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect responses extracted by the v2 pipeline from a PDF.",
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
        help="Maximum number of characters to show per response snippet (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
    args = parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Running v2 preprocessing on %s", pdf_path)
    preprocessed = extract_text(pdf_path)

    extractor = ResponseExtractorV2()
    logger.info("Extracting responses (v2) from preprocessed text")
    resps = extractor.extract(preprocessed, source_document=pdf_path.name)

    print("\n=== v2 Responses Preview ===")
    print(f"PDF: {pdf_path}")
    print(f"Total responses: {len(resps)}\n")

    for idx, resp in enumerate(resps, 1):
        start, end = resp.span
        span_info = f"{start}-{end}"
        snippet = resp.text.replace("\n", " ").strip()
        if len(snippet) > args.max_chars:
            snippet = snippet[: args.max_chars] + "..."
        print("=" * 80)
        print(
            f"Resp #{idx} | rec_number={resp.rec_number} | type={resp.response_type} "
            f"| span={span_info} | source={resp.source_document}"
        )
        print("-" * 80)
        print(snippet)
        print()

    print("=== End of v2 responses preview ===\n")


if __name__ == "__main__":
    main()
