"""
CLI-style helper to mirror the Streamlit "Align recommendations and responses"
workflow on the command line, starting from PDF files.

Usage (from repo root):

    python -m tools.align_cli \
        --recs-pdf samplepdfs/Report1.pdf \
        --resps-pdf samplepdfs/Report1Response.pdf

This is an observational tool only: it prints extracted recommendations,
responses, and alignments so we can compare behaviour with the UI.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from daphne_core.integration_helper import extract_text_from_file
from daphne_core.recommendation_extractor import extract_recommendations
from daphne_core.alignment_engine import RecommendationResponseMatcher, extract_response_sentences

logger = logging.getLogger(__name__)


@dataclass
class UploadedFileShim:
    """Minimal shim to mimic Streamlit's uploaded file interface."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def load_document_from_pdf(path: Path) -> Dict[str, Any]:
    """Extract text from a PDF file using the same stack as the app."""
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    uploaded = UploadedFileShim(path)
    text = extract_text_from_file(uploaded) or ""

    doc: Dict[str, Any] = {
        "filename": uploaded.name,
        "text": text,
        "word_count": len(text.split()) if text else 0,
    }
    return doc


def run_alignment_pipeline(
    recs_pdf: Path,
    resps_pdf: Path,
    min_confidence: float = 0.75,
) -> Dict[str, Any]:
    """
    End-to-end alignment pipeline starting from two PDFs, mirroring the
    canonical Streamlit alignment tab (`alignment_ui.render_simple_alignment_interface`).

    Returns the structured results so tests or other callers can introspect.
    """
    logger.info("Loading documents...")
    recs_doc = load_document_from_pdf(recs_pdf)
    resps_doc = load_document_from_pdf(resps_pdf)

    logger.info(
        "Extracting recommendations from %s (min_confidence=%.2f)",
        recs_doc["filename"],
        min_confidence,
    )
    rec_text = recs_doc.get("text", "") or ""
    recommendations = extract_recommendations(rec_text, min_confidence=min_confidence)
    for rec in recommendations:
        rec["document"] = recs_doc
        rec["sentence"] = rec.get("text", rec.get("sentence", ""))

    logger.info(
        "Extracting responses from %s using extract_response_sentences",
        resps_doc["filename"],
    )
    resp_text = resps_doc.get("text", "") or ""
    responses = extract_response_sentences(resp_text)

    logger.info(
        "Aligning %d recommendations with %d responses using RecommendationResponseMatcher",
        len(recommendations),
        len(responses),
    )
    matcher = RecommendationResponseMatcher()
    alignments = matcher.find_best_matches(recommendations, responses)

    return {
        "recs_doc": recs_doc,
        "resps_doc": resps_doc,
        "recommendations": recommendations,
        "responses": responses,
        "alignments": alignments,
    }


def print_summary(results: Dict[str, Any], max_items: int = 5) -> None:
    """Print a concise summary of recommendations, responses, and alignments."""
    recs = results["recommendations"]
    resps = results["responses"]
    alignments = results["alignments"]

    print("\n=== Documents ===")
    print(f"- Recommendations PDF: {results['recs_doc']['filename']} ({results['recs_doc']['word_count']} words)")
    print(f"- Responses PDF      : {results['resps_doc']['filename']} ({results['resps_doc']['word_count']} words)")

    print("\n=== Extracted Recommendations ===")
    print(f"Total: {len(recs)}")
    for idx, rec in enumerate(recs[:max_items], 1):
        text = rec.get("text", "").strip().replace("\n", " ")
        snippet = (text[:200] + "...") if len(text) > 200 else text
        verb = rec.get("verb", "unknown")
        method = rec.get("method", "unknown")
        conf = rec.get("confidence", 0.0)
        print(f"{idx:3d}. [{verb} | {method} | {conf:.2f}] {snippet}")
    if len(recs) > max_items:
        print(f"... ({len(recs) - max_items} more recommendations)")

    print("\n=== Extracted Responses ===")
    print(f"Total: {len(resps)}")
    for idx, resp in enumerate(resps[:max_items], 1):
        sentence = resp.get("sentence", "").strip().replace("\n", " ")
        snippet = (sentence[:200] + "...") if len(sentence) > 200 else sentence
        doc_name = resp.get("document", {}).get("filename", "unknown")
        pattern = resp.get("pattern", "")
        print(f"{idx:3d}. [{doc_name} | {pattern}] {snippet}")
    if len(resps) > max_items:
        print(f"... ({len(resps) - max_items} more responses)")

    print("\n=== Alignments (semantic matcher) ===")
    print(f"Total: {len(alignments)} (one row per recommendation)")
    for idx, alignment in enumerate(alignments[:max_items], 1):
        rec = alignment.get("recommendation", {})
        rec_text = rec.get("text", rec.get("sentence", "")).strip().replace("\n", " ")
        rec_snippet = (rec_text[:120] + "...") if len(rec_text) > 120 else rec_text
        has_response = alignment.get("has_response", False)
        resp = alignment.get("response")

        print(f"{idx:3d}. REC: {rec_snippet}")
        if has_response and resp:
            resp_text = resp.get("response_text", resp.get("text", "")).strip().replace("\n", " ")
            resp_snippet = (resp_text[:120] + "...") if len(resp_text) > 120 else resp_text
            score = resp.get("similarity", 0.0)
            status = resp.get("status", "Unclear")
            source = resp.get("source_document", "unknown")
            method = resp.get("match_method", "semantic")
            print(f"     → RESP: {resp_snippet}")
            print(f"       score={score:.2f} status={status} method={method} source={source}")
        else:
            print("     → No response match")

    print("\nDone.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end recommendation/response alignment from PDFs.",
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
        "--min-confidence",
        type=float,
        default=0.75,
        help="Minimum confidence for recommendation extraction (default: 0.75, matching alignment UI)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Maximum number of items to print per section (default: 5)",
    )
    parser.add_argument(
        "--export-alignments-csv",
        type=Path,
        default=None,
        help="Optional path to write full alignment details as CSV (for Excel/analysis).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    results = run_alignment_pipeline(
        recs_pdf=args.recs_pdf,
        resps_pdf=args.resps_pdf,
        min_confidence=args.min_confidence,
    )
    print_summary(results, max_items=args.max_items)

    if args.export_alignments_csv is not None:
        path: Path = args.export_alignments_csv
        logger.info("Writing alignments CSV to %s", path)

        alignments = results["alignments"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "AlignmentIndex",
                    "RecommendationText",
                    "RecommendationConfidence",
                    "RecommendationMethod",
                    "ResponseStatus",
                    "ResponseSimilarity",
                    "ResponseText",
                    "ResponseMatchMethod",
                    "ResponseSourceDocument",
                    "HasResponse",
                ]
            )

            for idx, alignment in enumerate(alignments, 1):
                rec = alignment.get("recommendation", {})
                rec_text = rec.get("text", rec.get("sentence", "")) or ""
                rec_conf = rec.get("confidence", 0.0)
                rec_method = rec.get("method", "unknown")

                has_response = alignment.get("has_response", False)
                resp = alignment.get("response") if has_response else None

                if resp:
                    status = resp.get("status", "Unclear")
                    similarity = resp.get("similarity", 0.0)
                    resp_text = resp.get("response_text", resp.get("text", "")) or ""
                    match_method = resp.get("match_method", "semantic")
                    source_doc = resp.get("source_document", "unknown")
                else:
                    status = "No Response"
                    similarity = 0.0
                    resp_text = ""
                    match_method = ""
                    source_doc = ""

                writer.writerow(
                    [
                        idx,
                        rec_text,
                        rec_conf,
                        rec_method,
                        status,
                        similarity,
                        resp_text,
                        match_method,
                        source_doc,
                        bool(has_response),
                    ]
                )



if __name__ == "__main__":
    main()
