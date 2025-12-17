"""
Streamlit meeting UI: side-by-side PFD extractor comparison.

This app is intentionally minimal for screenshare:
- Single shared PDF uploader
- One-button run
- 3 parallel columns:
  1) v1 strict extractor (run over v2-preprocessed text)
  2) v2 PFD concerns as blocks
  3) v2 PFD concerns atomised into sentences

Run:
    streamlit run pfd_compare_app.py
"""

from __future__ import annotations

import hashlib
import re
import traceback
from pathlib import Path

import streamlit as st

from daphne_core.recommendation_extractor import extract_recommendations
from daphne_core.v2.preprocess import extract_text as extract_text_v2
from daphne_core.v2.recommendations import PFD_REPORT_PROFILE, RecommendationExtractorV2


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem) or "uploaded"


def _save_uploaded_pdf(uploaded) -> Path:
    pdf_bytes = uploaded.getvalue()
    digest = hashlib.sha256(pdf_bytes).hexdigest()[:12]

    out_dir = Path("output/uploads/pfd_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{_safe_stem(uploaded.name)}_{digest}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(pdf_bytes)
    return pdf_path


def _render_v1_column(v1_recs: list[dict]) -> None:
    st.subheader("Action Verbs")
    st.caption("v1 strict extractor (legacy action-verb channel), running over v2-preprocessed text")
    st.metric("Count", len(v1_recs))

    if not v1_recs:
        st.info("No v1 recommendations found.")
        return

    v1_sorted = sorted(v1_recs, key=lambda r: r.get("confidence", 0), reverse=True)
    for idx, rec in enumerate(v1_sorted, 1):
        text = (rec.get("text") or "").strip()
        conf = float(rec.get("confidence") or 0.0)
        method = rec.get("method", "unknown")
        title = f"{idx}. {method} ({conf:.0%})"
        with st.expander(title, expanded=(idx <= 3)):
            st.write(text)


def _render_v2_concerns_column(recs_v2, title: str, caption: str) -> None:
    st.subheader(title)
    st.caption(caption)
    pfd_concerns = [r for r in recs_v2 if getattr(r, "rec_type", None) == "pfd_concern"]
    st.metric("Count", len(pfd_concerns))

    if not pfd_concerns:
        st.info("No v2 PFD concerns found.")
        return

    pfd_concerns_sorted = sorted(pfd_concerns, key=lambda r: (r.rec_number or 9999, r.span[0]))
    for idx, rec in enumerate(pfd_concerns_sorted, 1):
        label = rec.rec_number or rec.rec_id or ""
        exp_title = f"{idx}. Concern {label}".strip()
        with st.expander(exp_title, expanded=(idx <= 5)):
            st.write(rec.text.strip())


def _atomised_trigger_title(text: str, detection_method: str | None) -> str | None:
    if detection_method:
        suffix = detection_method.split(":")[-1]
        v1_map = {
            "entity_should": "should",
            "we_recommend": "we recommend",
            "should_be_passive": "should be",
            "modal_verb": "should/must/shall",
            "imperative": "imperative",
        }
        if suffix in v1_map:
            return v1_map[suffix]
        if suffix and suffix not in {"pfd_matters_of_concern_atomized", "pfd_concerns_fallback_atomized"}:
            # For pfd trigger labels we store human-readable phrases in the suffix.
            return suffix

    cleaned = text.strip()
    cleaned = re.sub(r"^(?:[-–•\s]+)", "", cleaned)
    cleaned = re.sub(r"^(?:Part\s+\d+\s*)?(?:\(\d+\)|\d+\.)\s*", "", cleaned, flags=re.IGNORECASE)
    words = cleaned.split()
    return " ".join(words[:4]) if words else None


def _render_v2_atomised_column(recs_v2) -> None:
    st.subheader("Extended Action Verbs")
    st.caption("v2 PFD concerns atomised into sentence-level items.")
    pfd_concerns = [r for r in recs_v2 if getattr(r, "rec_type", None) == "pfd_concern"]
    st.metric("Count", len(pfd_concerns))

    if not pfd_concerns:
        st.info("No v2 PFD concerns found.")
        return

    pfd_concerns_sorted = sorted(pfd_concerns, key=lambda r: (r.rec_number or 9999, r.span[0]))
    for idx, rec in enumerate(pfd_concerns_sorted, 1):
        trigger = _atomised_trigger_title(rec.text, getattr(rec, "detection_method", None))
        if not trigger:
            continue
        exp_title = f"{idx}. {trigger} (Concern {rec.rec_number or rec.rec_id or ''})".strip()
        with st.expander(exp_title, expanded=(idx <= 5)):
            st.write(rec.text.strip())


def main() -> None:
    st.set_page_config(page_title="DaphneAI – PFD Compare", layout="wide")
    st.title("🧪 PFD Extractor Comparison")
    st.caption("For meeting screenshare: Action Verbs vs Full concerns vs Extended Action Verbs on the same preprocessing.")

    uploaded = st.file_uploader("Upload a PFD (coroner) PDF", type=["pdf"])
    if not uploaded:
        st.info("Upload a PDF to begin.")
        return

    min_conf = st.slider(
        "v1 min_confidence (applied to v1 action-verb inference)",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05,
    )

    pdf_path = _save_uploaded_pdf(uploaded)
    st.caption(f"Saved to: `{pdf_path}`")

    if st.button("🔍 Run comparison", type="primary"):
        try:
            with st.spinner("Running v2 preprocessing…"):
                preprocessed = extract_text_v2(pdf_path)

            with st.spinner("Extracting v1 recommendations over v2-preprocessed text…"):
                v1_recs = extract_recommendations(preprocessed.text, min_confidence=min_conf)

            with st.spinner("Extracting v2 PFD concerns (blocks)…"):
                v2_blocks = RecommendationExtractorV2(
                    profile=PFD_REPORT_PROFILE,
                    pfd_atomize_concerns=False,
                ).extract(preprocessed, source_document=pdf_path.name)

            with st.spinner("Extracting v2 PFD concerns (atomised)…"):
                v2_atom = RecommendationExtractorV2(
                    profile=PFD_REPORT_PROFILE,
                    pfd_atomize_concerns=True,
                ).extract(preprocessed, source_document=pdf_path.name)

            st.session_state["pfd_compare"] = {
                "pdf_path": str(pdf_path),
                "min_conf": float(min_conf),
                "v1_recs": v1_recs,
                "v2_blocks": v2_blocks,
                "v2_atom": v2_atom,
            }
        except Exception as e:
            st.error(f"❌ Comparison run failed: {e}")
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            return

    results = st.session_state.get("pfd_compare")
    if not results or results.get("pdf_path") != str(pdf_path) or results.get("min_conf") != float(min_conf):
        st.info("Click “Run comparison” to generate side-by-side results.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        _render_v1_column(results["v1_recs"])
    with col2:
        _render_v2_concerns_column(
            results["v2_blocks"],
            title="Full concerns",
            caption="v2 PFD concerns as blocks (one item per numbered concern).",
        )
    with col3:
        _render_v2_atomised_column(results["v2_atom"])


if __name__ == "__main__":
    main()
