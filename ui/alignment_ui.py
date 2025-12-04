"""
🔗 Recommendation-Response Alignment Interface
Uses the core alignment engine and shared extractors; UI concerns only.
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from daphne_core.alignment_engine import RecommendationResponseMatcher, extract_response_sentences

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

        if not rec_text:
            st.error("Could not read recommendation document")
            return
        if not resp_text:
            st.error("Could not read response document")
            return

        progress = st.progress(0, text="Extracting recommendations...")

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

            st.success(f"✅ Matched {len(recommendations)} recommendations with {len(responses)} responses")
        except Exception as e:  # pragma: no cover - UI path
            progress.empty()
            st.error(f"Error: {e}")
            import traceback

            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            return

    if "alignment_results" in st.session_state and st.session_state.alignment_results:
        display_alignment_results(st.session_state.alignment_results)


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


__all__ = ["render_simple_alignment_interface"]
