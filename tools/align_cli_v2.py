"""
CLI helper for the v2 alignment pipeline.

Runs the v2 stack end-to-end starting from PDFs:
- v2 preprocessing on recommendations and responses,
- v2 recommendation extraction,
- v2 response extraction,
- v2 alignment (`AlignmentStrategyV2`),
and prints a concise alignment summary.

Usage (from repo root):

    python -m tools.align_cli_v2 \
        --recs-pdf samplepdfs/Report1.pdf \
        --resps-pdf samplepdfs/Report1Response.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from daphne_core.v2.preprocess import extract_text as extract_text_v2
from daphne_core.v2.recommendations import RecommendationExtractorV2
from daphne_core.v2.responses import ResponseExtractorV2
from daphne_core.v2.alignment import AlignmentStrategyV2

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v2 recommendation/response alignment from PDFs.",
    )
    parser.add_argument(
        "--recs-pdf",
        required=True,
        type=Path,
        help="Path to the recommendations PDF (e.g. samplepdfs/Report1.pdf)",
    )
    parser.add_argument(
        "--resps-pdf",
        required=True,
        type=Path,
        help="Path to the responses PDF (e.g. samplepdfs/Report1Response.pdf)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Maximum number of alignments to print (default: 5).",
    )
    parser.add_argument(
        "--enforce-one-to-one",
        action="store_true",
        help="Enforce 1:1 matching between recommendations and responses.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
    args = parse_args()

    recs_pdf: Path = args.recs_pdf
    resps_pdf: Path = args.resps_pdf

    if not recs_pdf.exists():
        raise FileNotFoundError(f"Recommendations PDF not found: {recs_pdf}")
    if not resps_pdf.exists():
        raise FileNotFoundError(f"Responses PDF not found: {resps_pdf}")

    logger.info("Running v2 preprocessing on %s", recs_pdf)
    recs_pre = extract_text_v2(recs_pdf)
    logger.info("Running v2 preprocessing on %s", resps_pdf)
    resps_pre = extract_text_v2(resps_pdf)

    rec_extractor = RecommendationExtractorV2()
    resp_extractor = ResponseExtractorV2()

    logger.info("Extracting v2 recommendations")
    recs = rec_extractor.extract(recs_pre, source_document=recs_pdf.name)
    logger.info("Extracting v2 responses")
    resps = resp_extractor.extract(resps_pre, source_document=resps_pdf.name)

    logger.info(
        "Aligning %d recommendations with %d responses (v2)",
        len(recs),
        len(resps),
    )
    strategy = AlignmentStrategyV2(enforce_one_to_one=args.enforce_one_to_one)
    alignments = strategy.align(recs, resps)

    print("\n=== v2 Alignment Summary ===")
    print(f"Recommendations PDF: {recs_pdf}")
    print(f"Responses PDF      : {resps_pdf}")
    print(f"Total recommendations: {len(recs)}")
    print(f"Total responses      : {len(resps)}")
    print(f"Total alignments     : {len(alignments)}\n")

    for idx, alignment in enumerate(alignments[: args.max_items], 1):
        rec = alignment.recommendation
        resp = alignment.response
        sim = alignment.similarity
        method = alignment.match_method

        rec_snippet = rec.text.replace("\n", " ").strip()
        if len(rec_snippet) > 140:
            rec_snippet = rec_snippet[:140] + "..."

        print("=" * 80)
        print(f"Alignment #{idx}")
        print(
            f"  Rec: id={rec.rec_id} | num={rec.rec_number} "
            f"| source={rec.source_document}"
        )
        print(f"  Rec snippet: {rec_snippet}")

        if resp is None:
            print("  → No response found")
        else:
            resp_snippet = resp.text.replace("\n", " ").strip()
            if len(resp_snippet) > 140:
                resp_snippet = resp_snippet[:140] + "..."
            print(
                f"  → Resp: id={resp.rec_id} | num={resp.rec_number} "
                f"| type={resp.response_type} | source={resp.source_document}"
            )
            print(f"    similarity={sim:.3f} | method={method}")
            print(f"    snippet: {resp_snippet}")

    print("\n=== End of v2 alignment summary ===\n")


if __name__ == "__main__":
    main()
