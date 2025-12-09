"""
Preprocessing bake-off helper for v2.

Runs multiple PDF→text→sentences pipelines on the same PDF and writes
sentence-per-line dumps into `output/` for manual inspection.

Pipelines:
- A: baseline extractor + simple regex sentence split.
- B: baseline extractor + syntok sentence split (if syntok is installed).
- C: pdfplumber extractor + syntok sentence split (if pdfplumber + syntok are installed).
- D: unstructured.partition_pdf (heavy, optional) + syntok sentence split.

Usage (from repo root):

    python -m tools.preproc_bakeoff --pdf samplepdfs/Report1.pdf

Outputs (if pdf name is Report1.pdf):
- output/Report1_A.txt
- output/Report1_B.txt (if syntok available)
- output/Report1_C.txt (if pdfplumber + syntok available)
- output/Report1_D.txt (if unstructured + syntok available)
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional
    psutil = None  # type: ignore

import time
import os

logger = logging.getLogger(__name__)


Sentence = Tuple[int, int]


def _load_pdf_baseline(pdf_path: Path) -> str:
    """Use the same baseline text extractor as the main app."""
    from daphne_core.integration_helper import extract_text_from_file

    class UploadedFileShim:
        def __init__(self, path: Path):
            self._path = path

        @property
        def name(self) -> str:
            return self._path.name

        def getvalue(self) -> bytes:
            return self._path.read_bytes()

    uploaded = UploadedFileShim(pdf_path)
    text = extract_text_from_file(uploaded) or ""
    return text


def _load_pdf_pdfplumber(pdf_path: Path) -> Optional[str]:
    """Extract text with pdfplumber if available."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.warning("pdfplumber not installed; skipping pdfplumber pipeline")
        return None

    pages: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _load_pdf_unstructured(pdf_path: Path) -> Optional[str]:
    """Extract text using unstructured.partition_pdf if available."""
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
    except ImportError:
        logger.warning("unstructured not installed; skipping heavy pipeline")
        return None

    elements = partition_pdf(filename=str(pdf_path))
    # Join paragraph-like elements; ignore tables/images for this bake-off.
    chunks: List[str] = []
    for el in elements:
        text = getattr(el, "text", "") or ""
        if not text.strip():
            continue
        chunks.append(text.strip())
    return "\n\n".join(chunks)


def _sentences_regex(text: str) -> List[Sentence]:
    """Very simple regex-based sentence splitter over the whole text."""
    spans: List[Sentence] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _sentences_syntok(text: str) -> Optional[List[Sentence]]:
    """Sentence spans using syntok if available."""
    try:
        from syntok import segmenter  # type: ignore
    except ImportError:
        logger.warning("syntok not installed; skipping syntok-based pipelines")
        return None

    spans: List[Sentence] = []
    offset = 0
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            sent_text = "".join(token.spacing + token.value for token in sentence).lstrip()
            if not sent_text:
                continue
            start = text.find(sent_text, offset)
            if start == -1:
                continue
            end = start + len(sent_text)
            spans.append((start, end))
            offset = end
    return spans


def _dump_sentences(text: str, spans: List[Sentence], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (start, end) in enumerate(spans, 1):
            snippet = text[start:end].replace("\n", " ").strip()
            f.write(f"[{idx:04d}] {snippet}\n")


def run_bakeoff(pdf_path: Path, output_dir: Path) -> None:
    stem = pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    def mem_rss_mb() -> Optional[float]:
        if psutil is None:
            return None
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)

    logger.info("Running pipeline A (baseline extractor + regex sentences)")
    mem_before = mem_rss_mb()
    t0 = time.perf_counter()
    text_a = _load_pdf_baseline(pdf_path)
    spans_a = _sentences_regex(text_a)
    t1 = time.perf_counter()
    mem_after = mem_rss_mb()
    _dump_sentences(text_a, spans_a, output_dir / f"{stem}_A.txt")
    logger.info(
        "Pipeline A: %d sentences | time=%.3fs%s",
        len(spans_a),
        t1 - t0,
        f" | ΔRSS={mem_after - mem_before:.1f}MB" if mem_before is not None and mem_after is not None else "",
    )

    logger.info("Running pipeline B (baseline extractor + syntok sentences)")
    mem_before = mem_rss_mb()
    t0 = time.perf_counter()
    spans_b = _sentences_syntok(text_a)
    if spans_b is not None:
        t1 = time.perf_counter()
        mem_after = mem_rss_mb()
        _dump_sentences(text_a, spans_b, output_dir / f"{stem}_B.txt")
        logger.info(
            "Pipeline B: %d sentences | time=%.3fs%s",
            len(spans_b),
            t1 - t0,
            f" | ΔRSS={mem_after - mem_before:.1f}MB" if mem_before is not None and mem_after is not None else "",
        )

    logger.info("Running pipeline C (pdfplumber + syntok sentences)")
    mem_before = mem_rss_mb()
    t0 = time.perf_counter()
    text_c = _load_pdf_pdfplumber(pdf_path)
    if text_c is not None:
        spans_c = _sentences_syntok(text_c)
        if spans_c is not None:
            t1 = time.perf_counter()
            mem_after = mem_rss_mb()
            _dump_sentences(text_c, spans_c, output_dir / f"{stem}_C.txt")
            logger.info(
                "Pipeline C: %d sentences | time=%.3fs%s",
                len(spans_c),
                t1 - t0,
                f" | ΔRSS={mem_after - mem_before:.1f}MB" if mem_before is not None and mem_after is not None else "",
            )

    logger.info("Running pipeline D (unstructured.partition_pdf + syntok sentences)")
    mem_before = mem_rss_mb()
    t0 = time.perf_counter()
    text_d = _load_pdf_unstructured(pdf_path)
    if text_d is not None:
        spans_d = _sentences_syntok(text_d)
        if spans_d is not None:
            t1 = time.perf_counter()
            mem_after = mem_rss_mb()
            _dump_sentences(text_d, spans_d, output_dir / f"{stem}_D.txt")
            logger.info(
                "Pipeline D: %d sentences | time=%.3fs%s",
                len(spans_d),
                t1 - t0,
                f" | ΔRSS={mem_after - mem_before:.1f}MB" if mem_before is not None and mem_after is not None else "",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple PDF→text→sentences pipelines and dump outputs for inspection.",
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
        help="Directory to write sentence dumps into (default: output/)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run_bakeoff(args.pdf, args.output_dir)


if __name__ == "__main__":
    main()
