"""
Inspect PFD (Regulation 28) response-block segmentation and scoped alignment.

This tool is intended as a human-in-the-loop preview for PFD-style documents,
especially long-form narrative reports (e.g. report8) with separate responder
letters (e.g. report8response1).

Usage (from repo root):

    python -m tools.inspect_pfd_alignment \
        --report recsandresps/report8.pdf \
        --response recsandresps/report8response1.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from daphne_core.v2.preprocess import extract_text as extract_text_v2
from daphne_core.v2.recommendations import RecommendationExtractorV2, PFD_REPORT_PROFILE
from daphne_core.v2.pfd_alignment import (
    align_pfd_directives_to_response_blocks,
    infer_responder_aliases,
    segment_pfd_response_blocks,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect PFD response-block segmentation and scoped alignment.",
    )
    parser.add_argument("--report", required=True, type=Path, help="Path to the PFD report PDF (e.g. report8.pdf)")
    parser.add_argument(
        "--response",
        required=True,
        type=Path,
        help="Path to a response PDF (e.g. report8response1.pdf)",
    )
    parser.add_argument(
        "--max-directives",
        type=int,
        default=20,
        help="Max directives to print (default: 20).",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=10,
        help="Max response blocks to print (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
    args = parse_args()

    report_pdf: Path = args.report
    response_pdf: Path = args.response

    if not report_pdf.exists():
        raise FileNotFoundError(f"Report PDF not found: {report_pdf}")
    if not response_pdf.exists():
        raise FileNotFoundError(f"Response PDF not found: {response_pdf}")

    logger.info("Preprocessing PFD report: %s", report_pdf)
    report_pre = extract_text_v2(report_pdf)
    logger.info("Preprocessing response: %s", response_pdf)
    resp_pre = extract_text_v2(response_pdf)

    recs = RecommendationExtractorV2(profile=PFD_REPORT_PROFILE).extract(report_pre, source_document=report_pdf.name)
    directives = [r for r in recs if getattr(r, "rec_type", None) == "pfd_directive"]
    concerns = [r for r in recs if getattr(r, "rec_type", None) == "pfd_concern"]

    print("\n=== PFD Recommendations (report) ===")
    print(f"Report: {report_pdf}")
    print(f"Total recs: {len(recs)} (pfd_concerns={len(concerns)}, pfd_directives={len(directives)})\n")

    logger.info("Segmenting response into thematic blocks")
    blocks = segment_pfd_response_blocks(resp_pre, source_document=response_pdf.name)

    responder_aliases = infer_responder_aliases(resp_pre.text)
    print("=== Response Identity (inferred) ===")
    print(f"Response: {response_pdf}")
    print(f"Responder aliases: {sorted(responder_aliases)}\n")

    print("=== Response Blocks (preview) ===")
    for idx, block in enumerate(blocks[: args.max_blocks], 1):
        snippet = block.text.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        print("-" * 80)
        print(f"Block #{idx}: header={block.header!r} span={block.span[0]}-{block.span[1]}")
        print(snippet)
    if len(blocks) > args.max_blocks:
        print(f"... ({len(blocks) - args.max_blocks} more blocks)\n")
    else:
        print()

    logger.info("Aligning directives to response blocks with scope gating")
    alignments = align_pfd_directives_to_response_blocks(directives, blocks, responder_aliases=responder_aliases)

    print("=== Scoped Alignment (directives → blocks) ===")
    for idx, a in enumerate(alignments[: args.max_directives], 1):
        d = a.directive
        d_snip = d.text.replace("\n", " ").strip()
        if len(d_snip) > 180:
            d_snip = d_snip[:180] + "..."

        print("=" * 80)
        print(f"Directive #{idx}: {d.rec_id} status={a.status}")
        if a.addressees:
            print(f"  addressees={list(a.addressees)}")
        print(f"  text: {d_snip}")
        if a.response_block is None:
            print("  → no matched block")
        else:
            print(f"  → block={a.response_block.header!r} span={a.response_block.span[0]}-{a.response_block.span[1]}")
            if a.response_snippet:
                print(f"  snippet: {a.response_snippet}")

    if len(alignments) > args.max_directives:
        print(f"... ({len(alignments) - args.max_directives} more directives)\n")
    else:
        print()

    # Identify orphan blocks (response-to-findings candidates).
    used_block_starts = {a.response_block.span[0] for a in alignments if a.response_block is not None}
    orphans = [b for b in blocks if b.span[0] not in used_block_starts]
    print("=== Unmatched Response Blocks (candidate: response_to_findings) ===")
    if not orphans:
        print("(none)\n")
    else:
        for idx, block in enumerate(orphans[: args.max_blocks], 1):
            snippet = block.text.replace("\n", " ").strip()
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            print("-" * 80)
            print(f"Orphan #{idx}: header={block.header!r}")
            print(snippet)
        if len(orphans) > args.max_blocks:
            print(f"... ({len(orphans) - args.max_blocks} more orphans)\n")
        else:
            print()


if __name__ == "__main__":
    main()

