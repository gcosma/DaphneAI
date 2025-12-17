"""
Inspect recommendations extracted by the v2 pipeline from a PDF.

This helper runs:
- v2 preprocessing (`daphne_core.v2.preprocess.extract_text`), then
- v2 recommendation extraction (`RecommendationExtractorV2.extract`),
and prints a concise summary plus an optional snippet per recommendation.

Outputs are printed to stdout; you can also redirect them or add CSV export
later if needed.

Usage (from repo root):

    python -m tools.inspect_recs_v2 --pdf samplepdfs/Report1.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from daphne_core.v2.preprocess import extract_text
from daphne_core.v2.recommendations import (
    RecommendationExtractorV2,
    EXPLICIT_RECS_PROFILE,
    PFD_REPORT_PROFILE,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect recommendations extracted by the v2 pipeline from a PDF.",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the recommendations PDF (e.g. samplepdfs/Report1.pdf)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200,
        help="Maximum number of characters to show per recommendation snippet (default: 200).",
    )
    parser.add_argument(
        "--profile",
        choices=[EXPLICIT_RECS_PROFILE, PFD_REPORT_PROFILE],
        default=EXPLICIT_RECS_PROFILE,
        help="v2 document profile: explicit recommendations vs PFD (coroner) report.",
    )
    parser.add_argument(
        "--atomize-pfd-concerns",
        action="store_true",
        help=(
            "When using --profile pfd_report, split numbered MATTERS OF CONCERN blocks into "
            "sentence-level items for review."
        ),
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

    extractor = RecommendationExtractorV2(
        profile=args.profile,
        pfd_atomize_concerns=bool(args.atomize_pfd_concerns),
    )
    logger.info("Extracting recommendations (v2) from preprocessed text")
    recs = extractor.extract(preprocessed, source_document=pdf_path.name)

    numbered = [r for r in recs if getattr(r, "rec_type", None) == "numbered"]
    pfd_concerns = [r for r in recs if getattr(r, "rec_type", None) == "pfd_concern"]
    pfd_directives = [r for r in recs if getattr(r, "rec_type", None) == "pfd_directive"]
    action_verb = [r for r in recs if getattr(r, "rec_type", None) == "action_verb"]

    print("\n=== v2 Recommendations Preview ===")
    print(f"PDF: {pdf_path}")
    print(
        f"Total recommendations: {len(recs)} "
        f"(numbered={len(numbered)}, pfd_concerns={len(pfd_concerns)}, "
        f"pfd_directives={len(pfd_directives)}, action-verb={len(action_verb)})\n"
    )

    for label, subset in (
        ("Numbered recommendations", numbered),
        ("PFD concerns (MATTERS OF CONCERN)", pfd_concerns),
        ("PFD directive sentences", pfd_directives),
        ("Action-verb recommendations", action_verb),
    ):
        if not subset:
            continue
        print(f"--- {label} ---")
        for idx, rec in enumerate(subset, 1):
            start, end = rec.span
            span_info = f"{start}-{end}"
            snippet = rec.text.replace("\n", " ").strip()
            if len(snippet) > args.max_chars:
                snippet = snippet[: args.max_chars] + "..."
            print("=" * 80)
            print(
                f"Rec #{idx} | rec_id={rec.rec_id} | rec_number={rec.rec_number} "
                f"| type={getattr(rec, 'rec_type', None)} "
                f"| method={getattr(rec, 'detection_method', None)} "
                f"| conf={getattr(rec, 'confidence', None)} "
                f"| span={span_info} | source={rec.source_document}"
            )
            print("-" * 80)
            print(snippet)
            print()

    print("=== End of v2 recommendations preview ===\n")


if __name__ == "__main__":
    main()
