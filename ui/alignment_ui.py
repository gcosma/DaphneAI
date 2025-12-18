"""
🔗 Recommendation-Response Alignment Interface
Uses the core alignment engine and shared extractors; UI concerns only.
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path

import pandas as pd
import streamlit as st

from daphne_core.alignment_engine import RecommendationResponseMatcher, extract_response_sentences
from daphne_core.v2.preprocess import extract_text as extract_text_v2
from daphne_core.v2.recommendations import RecommendationExtractorV2
from daphne_core.v2.responses import ResponseExtractorV2
from daphne_core.v2.alignment import AlignmentStrategyV2
from daphne_core.v2.pfd_alignment import (
    PfdScopedAlignment,
    align_pfd_directives_to_response_blocks,
    infer_responder_aliases,
    segment_pfd_response_blocks,
)

logger = logging.getLogger(__name__)


@st.cache_resource
def get_matcher():
    """Get or create cached matcher instance."""
    return RecommendationResponseMatcher()


def render_simple_alignment_interface(documents: List[Dict[str, Any]]):
    """Render the alignment interface with semantic matching."""
    st.markdown("### 🔗 Match Recommendations to Responses")

    matcher = get_matcher()
    if matcher.use_transformer:
        st.success("🧠 Using semantic matching (sentence-transformers)")
    else:
        st.info("📊 Using keyword matching (install sentence-transformers for better results)")

    if not documents:
        st.warning("📁 Please upload documents first in the Upload tab.")
        return

    doc_names = [doc["filename"] for doc in documents]

    st.markdown("---")
    st.markdown("#### 📄 Step 1: Select Documents")

    doc_mode = st.radio(
        "Are recommendations and responses in the same document?",
        ["Same document", "Different documents"],
        horizontal=True,
        help="Select 'Same document' if both recommendations and responses are in one file",
    )

    if doc_mode == "Same document":
        selected_doc = st.selectbox(
            "Select document containing both recommendations and responses:",
            doc_names,
            key="single_doc_select",
        )
        rec_docs = [selected_doc]
        resp_docs = [selected_doc]
    else:
        col1, col2 = st.columns(2)
        with col1:
            rec_doc = st.selectbox(
                "📋 Recommendation document:",
                doc_names,
                key="rec_doc_select",
            )
            rec_docs = [rec_doc]
        with col2:
            suggested = [n for n in doc_names if any(t in n.lower() for t in ["response", "reply", "answer"])]
            default_resp = suggested[0] if suggested else doc_names[0]
            resp_doc = st.selectbox(
                "📢 Response document:",
                doc_names,
                index=doc_names.index(default_resp) if default_resp in doc_names else 0,
                key="resp_doc_select",
            )
            resp_docs = [resp_doc]

    st.markdown("---")
    st.markdown("#### 🔍 Step 2: Extract & Match")

    st.caption("Canonical alignment: v2 preprocessing + structure-first extraction + alignment.")
    engine = "v2 (experimental)"

    v2_doc_type = st.selectbox(
        "Document type",
        ["Recommendation report", "PFD (coroner) report"],
        key="v2_alignment_doc_type",
        help="Choose how the canonical pipeline interprets document structure and how alignment is performed.",
    )
    single_paragraph = True
    st.caption(
        "Display: single paragraph (display-only). Extraction and matching still use the underlying sentence/layout structures."
    )

    if st.button("🚀 Extract Recommendations & Find Responses", type="primary"):
        rec_text = ""
        for doc_name in rec_docs:
            doc = next((d for d in documents if d["filename"] == doc_name), None)
            if doc and "text" in doc:
                rec_text += doc["text"] + "\n\n"

        resp_text = ""
        for doc_name in resp_docs:
            doc = next((d for d in documents if d["filename"] == doc_name), None)
            if doc and "text" in doc:
                resp_text += doc["text"] + "\n\n"

        if engine == "v1 (current)":
            if not rec_text:
                st.error("Could not read recommendation document")
                return
            if not resp_text:
                st.error("Could not read response document")
                return

            progress = st.progress(0, text="Extracting recommendations (v1)...")

            try:
                from daphne_core.recommendation_extractor import extract_recommendations

                recommendations = extract_recommendations(rec_text, min_confidence=0.75)
                if not recommendations:
                    st.warning("⚠️ No recommendations found in the document.")
                    progress.empty()
                    return

                progress.progress(33, text=f"Found {len(recommendations)} recommendations. Extracting responses...")
                responses = extract_response_sentences(resp_text)
                if not responses:
                    st.warning("⚠️ No response patterns found in the document.")
                    progress.empty()
                    return

                progress.progress(66, text=f"Found {len(responses)} responses. Matching...")
                alignments = matcher.find_best_matches(recommendations, responses)

                progress.progress(100, text="Complete!")
                progress.empty()

                st.session_state.alignment_results = alignments
                st.session_state.extracted_recommendations = recommendations

                st.success(f"✅ Matched {len(recommendations)} recommendations with {len(responses)} responses (v1)")
            except Exception as e:  # pragma: no cover - UI path
                progress.empty()
                st.error(f"Error: {e}")
                import traceback

                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                return
        else:
            # v2 path: require original PDF paths to be available.
            rec_doc_name = rec_docs[0]
            resp_doc_name = resp_docs[0]
            rec_doc = next((d for d in documents if d["filename"] == rec_doc_name), None)
            resp_doc = next((d for d in documents if d["filename"] == resp_doc_name), None)

            rec_pdf_path = rec_doc.get("pdf_path") if rec_doc else None
            resp_pdf_path = resp_doc.get("pdf_path") if resp_doc else None

            if not rec_pdf_path or not resp_pdf_path:
                st.error(
                    "v2 engine requires original PDF files. "
                    "Please ensure you uploaded PDFs (not only text) in this session."
                )
                return

            progress = st.progress(0, text="Extracting recommendations (canonical)...")

            try:
                recs_pre = extract_text_v2(Path(rec_pdf_path))
                resps_pre = extract_text_v2(Path(resp_pdf_path))

                if v2_doc_type == "PFD (coroner) report":
                    rec_extractor = RecommendationExtractorV2(profile="pfd_report", enable_pfd_directives=True)
                    recommendations_v2 = rec_extractor.extract(recs_pre, source_document=rec_doc_name)
                    if not recommendations_v2:
                        st.warning("⚠️ No recommendations found in the PDF (PFD mode).")
                        progress.empty()
                        return

                    directives = [r for r in recommendations_v2 if getattr(r, "rec_type", None) == "pfd_directive"]
                    concerns = [r for r in recommendations_v2 if getattr(r, "rec_type", None) == "pfd_concern"]
                    items_for_alignment = directives if directives else concerns

                    progress.progress(
                        33,
                        text=(
                            f"Found {len(recommendations_v2)} PFD recommendations "
                            f"(directives={len(directives)}, concerns={len(concerns)}). "
                            "Segmenting response blocks..."
                        ),
                    )

                    responder_aliases = infer_responder_aliases(resps_pre.text)
                    blocks = segment_pfd_response_blocks(resps_pre, source_document=resp_doc_name)

                    progress.progress(
                        66,
                        text=(
                            f"Segmented {len(blocks)} response blocks; "
                            f"running scoped alignment (aliases={len(responder_aliases)})..."
                        ),
                    )

                    pfd_alignments = align_pfd_directives_to_response_blocks(
                        items_for_alignment,
                        blocks,
                        responder_aliases=responder_aliases,
                    )

                    progress.progress(100, text="Complete!")
                    progress.empty()

                    st.session_state.v2_alignment_mode = "pfd"
                    st.session_state.pfd_alignment_results = pfd_alignments
                    st.session_state.pfd_recommendations = recommendations_v2
                    st.session_state.pfd_directives = items_for_alignment
                    st.session_state.pfd_concerns = concerns
                    st.session_state.pfd_response_blocks = blocks
                    st.session_state.pfd_responder_aliases = sorted(responder_aliases)

                    st.success(
                        f"✅ PFD mode: aligned {len(items_for_alignment)} extracted items to "
                        f"{len(blocks)} response blocks (canonical)"
                    )
                else:
                    rec_extractor = RecommendationExtractorV2(profile="explicit_recs")
                    resp_extractor = ResponseExtractorV2()

                    recommendations_v2 = rec_extractor.extract(recs_pre, source_document=rec_doc_name)
                    if not recommendations_v2:
                        st.warning("⚠️ No recommendations found in the PDF.")
                        progress.empty()
                        return

                    progress.progress(
                        33,
                        text=f"Found {len(recommendations_v2)} recommendations (v2). Extracting responses...",
                    )

                    responses_v2 = resp_extractor.extract(resps_pre, source_document=resp_doc_name)
                    if not responses_v2:
                        st.warning(
                            "⚠️ No structured responses found for this response document in v2. "
                            "Showing extracted recommendations only."
                        )
                        progress.progress(100, text="Complete!")
                        progress.empty()
                        st.session_state.v2_alignment_mode = "explicit"
                        st.session_state.v2_alignment_results = []
                        st.session_state.v2_recommendations = recommendations_v2
                        st.session_state.v2_responses = []
                        return

                    progress.progress(
                        66,
                        text=f"Found {len(responses_v2)} responses. Matching...",
                    )

                    strategy = AlignmentStrategyV2(
                        enforce_one_to_one=False,
                        use_embeddings=bool(matcher.use_transformer),
                        semantic_matcher=matcher,
                    )
                    alignments_v2 = strategy.align(recommendations_v2, responses_v2)

                    progress.progress(100, text="Complete!")
                    progress.empty()

                    st.session_state.v2_alignment_mode = "explicit"
                    st.session_state.v2_alignment_results = alignments_v2
                    st.session_state.v2_recommendations = recommendations_v2
                    st.session_state.v2_responses = responses_v2

                    st.success(
                        f"✅ Matched {len(recommendations_v2)} recommendations with "
                        f"{len(responses_v2)} responses (canonical)"
                    )
            except Exception as e:  # pragma: no cover - UI path
                progress.empty()
                st.error(f"Error in v2 pipeline: {e}")
                import traceback

                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                return

    if st.session_state.get("v2_alignment_mode") == "pfd" and "pfd_alignment_results" in st.session_state:
        display_pfd_alignment_results(st.session_state.pfd_alignment_results)
    elif "v2_alignment_results" in st.session_state:
        display_v2_alignment_results(st.session_state.v2_alignment_results)


def display_alignment_results(alignments: List[Dict[str, Any]]):
    """Display alignment results produced by the matcher."""
    st.markdown("---")
    st.markdown("### 📊 Alignment Results")

    total = len(alignments)
    with_response = sum(1 for a in alignments if a.get("has_response"))

    status_counts = Counter()
    for a in alignments:
        if a.get("has_response") and a.get("response"):
            status_counts[a["response"].get("status", "Unclear")] += 1
        else:
            status_counts["No Response"] += 1

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total)
    col2.metric("✅ Accepted", status_counts.get("Accepted", 0))
    col3.metric("⚠️ Partial", status_counts.get("Partial", 0))
    col4.metric("❌ Rejected", status_counts.get("Rejected", 0))
    col5.metric("❓ No Response", status_counts.get("No Response", 0) + status_counts.get("Unclear", 0))

    st.markdown("---")

    sort_option = st.selectbox(
        "Sort by:",
        ["Original Order", "Status (Accepted first)", "Status (No Response first)", "Match Confidence"],
    )

    sorted_alignments = alignments.copy()
    if sort_option == "Status (Accepted first)":
        status_order = {"Accepted": 0, "Partial": 1, "Noted": 2, "Rejected": 3, "Unclear": 4, "No Response": 5}
        sorted_alignments.sort(
            key=lambda x: status_order.get(
                x.get("response", {}).get("status") if x.get("has_response") else "No Response",
                5,
            )
        )
    elif sort_option == "Status (No Response first)":
        sorted_alignments.sort(key=lambda x: 0 if not x.get("has_response") else 1)
    elif sort_option == "Match Confidence":
        sorted_alignments.sort(
            key=lambda x: x.get("response", {}).get("similarity") if x.get("has_response") else 0,
            reverse=True,
        )

    st.markdown("#### 📋 Detailed Results")
    for idx, alignment in enumerate(sorted_alignments, 1):
        rec = alignment.get("recommendation", {})
        resp = alignment.get("response")

        if not resp or not alignment.get("has_response"):
            status_icon = "❓"
            status_text = "No Response Found"
        else:
            status = resp.get("status", "Unclear")
            status_icon = {"Accepted": "✅", "Partial": "⚠️", "Rejected": "❌", "Noted": "📝"}.get(status, "❓")
            status_text = status

        rec_preview = rec.get("text", "")
        rec_preview = rec_preview[:80] + "..." if len(rec_preview) > 80 else rec_preview

        with st.expander(f"{status_icon} **{idx}.** {rec_preview}", expanded=(idx <= 3)):
            st.markdown("**📝 Recommendation:**")
            st.info(rec.get("text", ""))

            col1, col2 = st.columns(2)
            col1.caption(f"Confidence: {rec.get('confidence', 0):.0%}")
            col2.caption(f"Method: {rec.get('method', 'unknown')}")

            st.markdown("---")
            st.markdown(f"**📢 Government Response:** {status_icon} **{status_text}**")

            if resp and alignment.get("has_response"):
                st.success(resp.get("response_text", resp.get("text", "No response text available")))

                col1, col2, col3 = st.columns(3)
                col1.caption(f"Match: {resp.get('similarity', alignment.get('score', 0)):.0%}")
                col2.caption(f"Method: {resp.get('match_method', 'unknown')}")
                col3.caption(f"Source: {resp.get('source_document', 'unknown')}")
            else:
                st.warning("No matching response found.")

    st.markdown("---")
    st.markdown("#### 💾 Export Results")

    col1, col2 = st.columns(2)
    with col1:
        export_data: List[Dict[str, Any]] = []
        for idx, a in enumerate(alignments, 1):
            rec = a.get("recommendation", {})
            resp = a.get("response") if a.get("has_response") else None
            export_data.append(
                {
                    "Number": idx,
                    "Recommendation": rec.get("text", ""),
                    "Recommendation_Confidence": rec.get("confidence", 0),
                    "Response_Status": resp.get("status", "No Response") if resp else "No Response",
                    "Response_Text": resp.get("response_text", "") if resp else "",
                    "Match_Confidence": resp.get("similarity", 0) if resp else 0,
                    "Match_Method": resp.get("match_method", "") if resp else "",
                }
            )

        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

        with col2:
            report = f"""# Recommendation-Response Alignment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Recommendations: {total}
- Responses Found: {with_response}
- Accepted: {status_counts.get('Accepted', 0)}
- Partial: {status_counts.get('Partial', 0)}
- Rejected: {status_counts.get('Rejected', 0)}
- No Response: {status_counts.get('No Response', 0)}

## Details
"""
        for idx, a in enumerate(alignments, 1):
            rec = a.get("recommendation", {})
            resp = a.get("response") if a.get("has_response") else None
            status = resp.get("status", "No Response") if resp else "No Response"
            report += f"""
### Recommendation {idx}
**Status:** {status}

**Recommendation:**
> {rec.get('text', '')}

**Response:**
> {resp.get('response_text', 'No response found') if resp else 'No response found'}

---
"""

        st.download_button(
            "📥 Download Report",
            report,
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            "text/markdown",
        )


def display_v2_alignment_results(alignments: List[Any]):
    """Display v2 alignment results with a v1-style UI."""
    st.markdown("---")
    st.markdown("### 📊 Alignment Results")

    from daphne_core.text_utils import format_display_markdown

    single_paragraph = True

    def fmt(text: str) -> str:
        return format_display_markdown(text, single_paragraph=single_paragraph)

    total = len(alignments)
    with_response = sum(1 for a in alignments if a.response is not None)

    def classify_status(text: str) -> str:
        matcher = get_matcher()
        try:
            status, _conf = matcher._classify_response_status(text)  # type: ignore[attr-defined]
            return status
        except Exception:  # pragma: no cover - defensive UI path
            return "Unclear"

    status_counts = Counter()
    per_alignment_status: list[str] = []
    for a in alignments:
        if a.response is None:
            status = "No Response"
        else:
            status = classify_status(a.response.text or "")
        per_alignment_status.append(status)
        status_counts[status] += 1

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total)
    col2.metric("✅ Accepted", status_counts.get("Accepted", 0))
    col3.metric("⚠️ Partial", status_counts.get("Partial", 0))
    col4.metric("❌ Rejected", status_counts.get("Rejected", 0))
    col5.metric("❓ No Response", status_counts.get("No Response", 0) + status_counts.get("Unclear", 0))

    st.markdown("---")

    sort_option = st.selectbox(
        "Sort by:",
        ["Original Order", "Status (Accepted first)", "Status (No Response first)", "Match Confidence"],
        key="v2_alignment_sort",
    )

    indexed = list(enumerate(alignments))
    if sort_option == "Status (Accepted first)":
        status_order = {"Accepted": 0, "Partial": 1, "Noted": 2, "Rejected": 3, "Unclear": 4, "No Response": 5}

        def key_fn(pair: tuple[int, Any]) -> int:
            idx0, _a = pair
            return status_order.get(per_alignment_status[idx0], 5)

        indexed.sort(key=key_fn)
    elif sort_option == "Status (No Response first)":
        indexed.sort(key=lambda p: 0 if p[1].response is None else 1)
    elif sort_option == "Match Confidence":
        indexed.sort(key=lambda p: float(p[1].similarity or 0.0), reverse=True)

    st.markdown("#### 📋 Detailed Results")
    for display_idx, (orig_idx, alignment) in enumerate(indexed, 1):
        rec = alignment.recommendation
        resp = alignment.response
        sim = float(alignment.similarity or 0.0)
        method = alignment.match_method

        status = per_alignment_status[orig_idx]
        status_icon = {"Accepted": "✅", "Partial": "⚠️", "Rejected": "❌", "Noted": "📝"}.get(status, "❓")

        rec_preview = rec.text.strip().replace("\n", " ")
        if len(rec_preview) > 80:
            rec_preview = rec_preview[:80] + "..."

        with st.expander(f"{status_icon} **{display_idx}.** {rec_preview}", expanded=(display_idx <= 3)):
            st.markdown("**📝 Recommendation:**")
            st.info(fmt(rec.text))

            col_a, col_b, col_c = st.columns(3)
            col_a.caption(f"ID: {rec.rec_id!r}")
            col_b.caption(f"Num: {rec.rec_number}")
            col_c.caption(f"Source: {rec.source_document}")

            st.markdown("---")
            st.markdown(f"**📢 Government Response:** {status_icon} **{status}**")

            if resp is None:
                st.warning("No matching response found.")
            else:
                st.success(fmt(resp.text))
                col1, col2, col3 = st.columns(3)
                col1.caption(f"Match: {sim:.0%}")
                col2.caption(f"Method: {method}")
                col3.caption(f"Source: {resp.source_document}")

    st.markdown("---")
    st.markdown("#### 💾 Export Results")

    export_rows: List[Dict[str, Any]] = []
    for idx, alignment in enumerate(alignments, 1):
        rec = alignment.recommendation
        resp = alignment.response
        status = per_alignment_status[idx - 1]
        export_rows.append(
            {
                "Number": idx,
                "Recommendation_ID": rec.rec_id,
                "Recommendation_Number": rec.rec_number,
                "Recommendation_Text": rec.text,
                "Response_ID": resp.rec_id if resp else None,
                "Response_Number": resp.rec_number if resp else None,
                "Response_Text": resp.text if resp else "",
                "Response_Status": status,
                "Match_Confidence": float(alignment.similarity or 0.0),
                "Match_Method": alignment.match_method,
            }
        )

    df = pd.DataFrame(export_rows)
    st.download_button(
        "📥 Download CSV",
        df.to_csv(index=False),
        f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
    )


def display_pfd_alignment_results(alignments: List[PfdScopedAlignment]):
    """Display PFD scoped alignment results."""
    st.markdown("---")
    st.markdown("### 📊 Alignment Results (v2 PFD mode)")

    from daphne_core.text_utils import format_display_markdown

    single_paragraph = True

    def fmt(text: str) -> str:
        return format_display_markdown(text, single_paragraph=single_paragraph)

    directives = st.session_state.get("pfd_directives", [])
    blocks = st.session_state.get("pfd_response_blocks", [])
    aliases = st.session_state.get("pfd_responder_aliases", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Directives", len(directives))
    col2.metric("Response blocks", len(blocks))
    col3.metric("Alignments", len(alignments))

    if aliases:
        with st.expander("Responder identity (inferred)", expanded=False):
            st.write(aliases)

    status_counts = Counter(a.status for a in alignments)
    with st.expander("Status breakdown", expanded=False):
        st.json(dict(status_counts))

    st.markdown("#### 📋 Directive → Block matches")
    for idx, a in enumerate(alignments, 1):
        directive = a.directive
        title = directive.text.strip().replace("\n", " ")
        if len(title) > 80:
            title = title[:80] + "..."
        with st.expander(f"🧾 {idx}. [{a.status}] {title}", expanded=(idx <= 3)):
            st.markdown("**Directive:**")
            st.info(fmt(directive.text))
            if a.addressees:
                st.caption(f"Addressees: {', '.join(a.addressees)}")
            st.markdown("---")
            st.markdown("**Matched response block:**")
            if a.response_block is None:
                st.warning("No matched response block.")
            else:
                st.success(f"Block: {a.response_block.header}")
                if a.response_snippet:
                    st.write(fmt(a.response_snippet))

    # Orphan blocks: response_to_findings candidates.
    if blocks:
        used_starts = {a.response_block.span[0] for a in alignments if a.response_block is not None}
        orphans = [b for b in blocks if b.span[0] not in used_starts]
        st.markdown("---")
        st.markdown("#### 🧩 Unmatched Response Blocks (response to findings)")
        if not orphans:
            st.caption("No unmatched blocks.")
        else:
            for idx, b in enumerate(orphans[:10], 1):
                snippet = b.text.replace("\n", " ").strip()
                if len(snippet) > 220:
                    snippet = snippet[:220] + "..."
                with st.expander(f"📄 Block {idx}: {b.header}", expanded=False):
                    st.write(fmt(snippet))


__all__ = ["render_simple_alignment_interface"]
