"""
Inspect how `unstructured.partition_pdf` parses tables in a PDF.

This is a probe tool to answer:
  "How far off are we from solving tables based on the baseline performance
   of our library?"

It runs `partition_pdf` on the given PDF, extracts `Table` elements, and
writes:
- A JSON dump with raw `Table.to_dict()` output (where available).
- A human-readable text file with table texts and page numbers.

Outputs go into `output/` by default so they remain gitignored.

Usage (from repo root):

    python -m tools.inspect_tables_v2 --pdf samplepdfs/Report1.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect table elements produced by unstructured.partition_pdf.",
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
        help="Directory to write inspection files into (default: output/)",
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

    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
        from unstructured.documents.elements import Table  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "unstructured is required for table inspection. "
            "Install it with `pip install unstructured[local-inference]`."
        ) from exc

    logger.info("Running unstructured.partition_pdf on %s", pdf_path)
    elements = partition_pdf(filename=str(pdf_path))

    tables: List[Any] = [el for el in elements if isinstance(el, Table)]
    logger.info("Found %d table elements", len(tables))

    stem = pdf_path.stem
    json_path = out_dir / f"{stem}_v2_tables.json"
    txt_path = out_dir / f"{stem}_v2_tables.txt"

    # JSON dump with as much structure as unstructured exposes.
    json_payload = []
    for idx, table in enumerate(tables, 1):
        meta = getattr(table, "metadata", None)
        page = getattr(meta, "page_number", None) if meta is not None else None
        if hasattr(table, "to_dict"):
            raw = table.to_dict()  # type: ignore[assignment]
        else:
            raw = {"text": getattr(table, "text", ""), "repr": repr(table)}
        json_payload.append(
            {
                "index": idx,
                "page_number": page,
                "raw": raw,
            }
        )

    logger.info("Writing table JSON dump to %s", json_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    # Human-readable text dump for quick eyeballing.
    logger.info("Writing human-readable table dump to %s", txt_path)
    with txt_path.open("w", encoding="utf-8") as f:
        for idx, table in enumerate(tables, 1):
            meta = getattr(table, "metadata", None)
            page = getattr(meta, "page_number", None) if meta is not None else "unknown"
            text = getattr(table, "text", "") or ""
            f.write("=" * 80 + "\n")
            f.write(f"Table #{idx} (page {page})\n")
            f.write("=" * 80 + "\n")
            f.write(text.strip() + "\n\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()

