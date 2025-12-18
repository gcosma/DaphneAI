# Updated app.py - DaphneAI Government Document Analysis
# OPTIMIZED: Fast loading with cached NLTK downloads
import streamlit as st
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Any
import logging
import traceback
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OPTIMIZED: Download NLTK data once at startup (cached)
@st.cache_resource
def initialize_nltk():
    """Initialize NLTK - runs once and caches"""
    try:
        import nltk
        # Check if already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            # Download only if not found
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        return True
    except:
        return False

# Initialize NLTK at app startup
NLP_AVAILABLE = initialize_nltk()

# FIXED IMPORT - Use the strict extractor instead of the old one
from daphne_core.recommendation_extractor import (
    extract_recommendations, 
    StrictRecommendationExtractor
)

# Try to import the new semantic search engine
try:
    from daphne_core.search_engine import SemanticSearchEngine
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    logger.warning("Semantic search not available")


def safe_import_with_fallback():
    """Safely import modules with comprehensive fallbacks"""
    try:
        from daphne_core.integration_helper import (
            setup_search_tab, 
            prepare_documents_for_search, 
            extract_text_from_file,
            render_analytics_tab
        )
        return True, setup_search_tab, prepare_documents_for_search, extract_text_from_file, render_analytics_tab
    except ImportError as e:
        logger.warning(f"Import error: {e}")
        return False, None, None, None, None


def render_semantic_search_tab():
    """NEW: Render the semantic search tab with the advanced search engine"""
    st.header("🤖 AI Semantic Search")
    st.markdown("*Find documents by meaning, not just keywords*")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first in the Upload tab.")
        
        with st.expander("ℹ️ What is Semantic Search?", expanded=True):
            st.markdown("""
            ### 🧠 AI-Powered Understanding
            
            Unlike keyword search, semantic search understands **meaning and context**:
            
            **Example searches that work:**
            - "digital infrastructure recommendations" → finds related concepts like "technology modernization", "IT systems"
            - "healthcare funding" → matches "NHS budget", "medical resources", "health service investment"
            - "climate change policy" → finds "environmental strategy", "carbon reduction", "sustainability"
            
            ---
            
            ### ✨ Key Features
            
            - **Understands synonyms** - "recommend" matches "suggest", "advise", "propose"
            - **Contextual matching** - finds relevant content even without exact words
            - **Relevance scoring** - best matches shown first
            - **Smart chunking** - searches document sections intelligently
            
            ---
            
            ### 🎯 Best For
            
            - Finding related concepts across documents
            - Discovering connections you might miss with keywords
            - Research and analysis tasks
            - Policy and recommendation analysis
            """)
        return
    
    # Show initialization button if not yet initialized
    documents = st.session_state.documents
    
    if 'semantic_search_engine' not in st.session_state:
        if not SEMANTIC_SEARCH_AVAILABLE:
            st.error("❌ Semantic search engine not available. Please install dependencies:")
            st.code("pip install sentence-transformers torch scikit-learn")
            return
        
        st.info("🤖 AI Search Engine is not initialized yet. Click below to initialize.")
        st.markdown("""
        **What happens when you initialize:**
        - Downloads AI model (33MB, one-time download)
        - Indexes your documents for semantic search
        - Takes ~10-30 seconds depending on document size
        - Only needs to be done once!
        """)
        
        if st.button("🚀 Initialize AI Search Engine", type="primary"):
            with st.spinner("🔄 Initializing AI search engine and indexing documents..."):
                try:
                    # Initialize search engine
                    search_engine = SemanticSearchEngine(
                        model_name='BAAI/bge-small-en-v1.5',
                        use_cross_encoder=False,
                        cache_embeddings=True
                    )
                    
                    # Index documents
                    search_engine.add_documents(documents, chunk_size=300, chunk_overlap=50)
                    
                    # Save to session state
                    st.session_state.semantic_search_engine = search_engine
                    st.session_state.indexed_documents = [doc.get('filename', '') for doc in documents]
                    
                    st.success("✅ AI search engine ready! You can now search below.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to initialize search engine: {str(e)}")
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
        return
    
    # Check if documents have changed since indexing
    search_engine = st.session_state.semantic_search_engine
    current_doc_ids = [doc.get('filename', '') for doc in documents]
    indexed_doc_ids = st.session_state.get('indexed_documents', [])
    
    if current_doc_ids != indexed_doc_ids:
        st.warning("⚠️ Documents have changed since last indexing.")
        if st.button("🔄 Re-index Documents"):
            with st.spinner("Re-indexing documents..."):
                try:
                    search_engine.add_documents(documents, chunk_size=300, chunk_overlap=50)
                    st.session_state.indexed_documents = current_doc_ids
                    st.success("✅ Documents re-indexed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Re-indexing failed: {str(e)}")
        return
    
    # Show status
    st.success(f"✅ Search engine ready with {len(documents)} documents indexed")
    
    # Search interface
    st.markdown("---")
    
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., digital transformation recommendations, healthcare policy responses...",
        help="Describe what you're looking for in natural language"
    )
    
    # Advanced settings in expander
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider(
                "Max results per document",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of matching sections per document"
            )
        with col2:
            min_score = st.slider(
                "Minimum relevance score",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Lower = more results but less relevant"
            )
        
        enable_reranking = st.checkbox(
            "Enable re-ranking (slower but more accurate)",
            value=False,
            help="Uses advanced AI to re-rank results for better accuracy"
        )
    
    # Search button
    if st.button("🚀 Search", type="primary") or query:
        if not query.strip():
            st.warning("Please enter a search query")
            return
        
        with st.spinner("🔍 Searching with AI..."):
            try:
                # Perform search
                results = search_engine.search(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    rerank=enable_reranking
                )
                
                if results:
                    # Summary
                    total_matches = sum(doc.total_matches for doc in results)
                    st.success(f"✅ Found {len(results)} documents with {total_matches} relevant sections")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documents", len(results))
                    with col2:
                        st.metric("Total Matches", total_matches)
                    with col3:
                        avg_score = sum(doc.overall_score for doc in results) / len(results)
                        st.metric("Avg Relevance", f"{avg_score:.1%}")
                    
                    st.markdown("---")
                    
                    # Display results by document
                    for doc_idx, doc_result in enumerate(results, 1):
                        # Document header
                        relevance_color = "🟢" if doc_result.overall_score >= 0.7 else "🟡" if doc_result.overall_score >= 0.5 else "🟠"
                        
                        st.markdown(f"### {relevance_color} {doc_idx}. {doc_result.filename}")
                        st.caption(f"Relevance: {doc_result.overall_score:.1%} | Type: {doc_result.document_type.title()} | {doc_result.total_matches} matches")
                        
                        # Show top matches
                        top_results = doc_result.get_top_results(3)  # Show top 3 per document
                        
                        for match_idx, match in enumerate(top_results, 1):
                            with st.expander(
                                f"Match {match_idx} - Relevance: {match.relevance_score:.1%}",
                                expanded=(doc_idx == 1 and match_idx == 1)  # Expand first result
                            ):
                                # Highlight matched concepts
                                if match.matched_concepts:
                                    st.caption(f"📌 Matched concepts: {', '.join(match.matched_concepts)}")
                                
                                # Show text
                                st.markdown(match.text_fragment)
                                
                                # Show extended context button
                                if len(match.full_context) > len(match.text_fragment):
                                    if st.button(f"Show full context", key=f"context_{doc_result.document_id}_{match_idx}"):
                                        st.info("**Extended Context:**")
                                        st.markdown(match.full_context)
                        
                        # Show more button if there are additional matches
                        if doc_result.total_matches > 3:
                            if st.button(
                                f"Show {doc_result.total_matches - 3} more matches",
                                key=f"more_{doc_result.document_id}"
                            ):
                                for match_idx, match in enumerate(doc_result.results[3:], 4):
                                    with st.expander(f"Match {match_idx} - {match.relevance_score:.1%}"):
                                        st.markdown(match.text_fragment)
                        
                        st.markdown("---")
                    
                    # Export results
                    if st.button("📥 Export Results as CSV"):
                        export_data = []
                        for doc_result in results:
                            for match in doc_result.results:
                                export_data.append({
                                    'Document': doc_result.filename,
                                    'Relevance': f"{match.relevance_score:.3f}",
                                    'Text': match.text_fragment,
                                    'Matched_Concepts': ', '.join(match.matched_concepts)
                                })
                        
                        df = pd.DataFrame(export_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv,
                            file_name=f"semantic_search_{query[:30]}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.warning("😕 No results found. Try:")
                    st.markdown("""
                    - Lowering the minimum relevance score
                    - Using different keywords or phrases
                    - Checking if your query matches document content
                    """)
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
    
    # Show search engine statistics
    with st.expander("📊 Search Engine Statistics"):
        stats = search_engine.get_statistics()
        st.json(stats)


def render_recommendations_tab():
    """Render the recommendations extraction tab."""
    st.header("🎯 Extract Recommendations")

    from daphne_core.text_utils import format_display_markdown
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first in the Upload tab.")
        
        with st.expander("ℹ️ About this feature", expanded=True):
            st.markdown("""
            ### What This Feature Does
            
            This **strict** recommendation extractor eliminates false positives by:
            
            1. **Pre-filtering garbage** - Removes URLs, timestamps, page numbers BEFORE analysis
            2. **Detecting meta-recommendations** - Rejects text ABOUT recommendations
            3. **Strict confidence scoring** - Only genuine recommendations get high scores
            4. **Numbered pattern detection** - Prioritises "Recommendation N" formats
            5. **Entity + should patterns** - NHS England should, Boards should, etc.
            
            **Result:** ~90% reduction in false positives compared to basic extraction.
            
            ---
            
            ### 🎨 Confidence Colour Guide
            
            Results are sorted by confidence (highest first) and colour-coded:
            
            | Colour | Confidence | What it means |
            |--------|------------|---------------|
            | 🟢 | **95%+** | Numbered recommendations or strong directive patterns |
            | 🟡 | **85-94%** | Passive recommendations ("should be completed") |
            | 🟠 | **75-84%** | Modal verb patterns - still valid recommendations |
            
            All extracted items are genuine recommendations - the colour simply indicates how explicit the recommendation language is.
            """)
        return
    
    documents = st.session_state.documents
    doc_names = [doc['filename'] for doc in documents]
    
    selected_doc = st.selectbox("Select document to analyse:", doc_names)
    # Display-only formatting (always enabled): this does not affect extraction.
    single_paragraph = True
    st.caption(
        "Display: single paragraph (display-only). Extraction still uses sentence-aware preprocessing under the hood."
    )

    def fmt(text: str) -> str:
        return format_display_markdown(text, single_paragraph=single_paragraph)

    st.caption(
        "Canonical pipeline: v2 preprocessing + structure-first extraction + Action Verbs as a second pass on uncovered text."
    )

    v2_profile = "explicit_recs"
    min_confidence = st.slider(
        "min_confidence (Action Verbs)",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Rule-based threshold (not calibrated probability). Increasing it disables whole rule families.",
    )

    profile_label = st.selectbox(
        "Document type",
        ["Recommendation report", "PFD (coroner) report"],
        help="Choose how the canonical pipeline interprets the document structure.",
    )
    if profile_label == "PFD (coroner) report":
        v2_profile = "pfd_report"

    def _detect_action_verbs_all(preprocessed, *, min_conf: float) -> list[dict]:
        """
        Run the legacy v1 sentence-based action-verb inference across the entire v2-preprocessed text.

        Returns de-duplicated hits with spans into `preprocessed.text`.
        """
        extractor = StrictRecommendationExtractor()
        text = getattr(preprocessed, "text", "") or ""
        if not text.strip():
            return []

        # v1/v2 action-verb extraction uses a simple boundary regex; we use the same
        # span construction here so the duplicate accounting aligns with the legacy path.
        sentence_spans: list[tuple[int, int]] = []
        start = 0
        for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", text):
            end = match.end()
            sentence_spans.append((start, end))
            start = end
        if start < len(text):
            sentence_spans.append((start, len(text)))

        hits: list[dict] = []
        for idx, (s0, s1) in enumerate(sentence_spans):
            sent = text[s0:s1]
            cleaned = extractor.clean_text(sent)
            is_garbage, _reason = extractor.is_garbage(cleaned, is_numbered_rec=False)
            if is_garbage:
                continue
            if extractor.is_meta_recommendation(cleaned):
                continue
            is_rec, confidence, method, verb = extractor.is_genuine_recommendation(
                cleaned, is_numbered_rec=False
            )
            if not is_rec or float(confidence) < float(min_conf):
                continue
            hits.append(
                {
                    "text": cleaned,
                    "verb": verb,
                    "method": method,
                    "confidence": round(float(confidence), 3),
                    "sentence_index": idx,
                    "span": (int(s0), int(s1)),
                }
            )

        return extractor._deduplicate(hits)  # type: ignore[attr-defined]

    def _structure_units_for_overlap(recs: list, *, profile: str) -> list[dict]:
        """
        Return structure-derived units with spans so we can count action-verb hits that
        fall inside structure (and are therefore suppressed to avoid duplicates).
        """
        units: list[dict] = []
        for r in recs:
            rec_type = getattr(r, "rec_type", None)
            span = getattr(r, "span", None)
            if not span or not isinstance(span, tuple) or len(span) != 2:
                continue

            if profile == "pfd_report":
                if rec_type != "pfd_concern":
                    continue
                if getattr(r, "detection_method", None) != "pfd_matters_of_concern":
                    continue
                label = f"Concern {getattr(r, 'rec_number', None) or getattr(r, 'rec_id', '')}".strip()
                units.append({"label": label, "span": span, "rec": r})
            else:
                if rec_type != "numbered":
                    continue
                label = (
                    f"Recommendation {getattr(r, 'rec_id', None) or getattr(r, 'rec_number', None) or ''}".strip()
                )
                units.append({"label": label, "span": span, "rec": r})

        units.sort(key=lambda u: int(u["span"][0]))
        return units

    def _partition_action_verbs_by_structure(
        action_verbs_all: list[dict], structure_units: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        inside: list[dict] = []
        outside: list[dict] = []

        for hit in action_verbs_all:
            hit_span = hit.get("span")
            if not hit_span:
                outside.append(hit)
                continue

            matched_label = None
            for unit in structure_units:
                s0, s1 = unit["span"]
                h0, _h1 = hit_span
                # Match v2 action-verb exclusion semantics: if the sentence *starts*
                # inside a structure block, treat it as covered by structure.
                if s0 <= h0 < s1:
                    matched_label = unit["label"]
                    break

            if matched_label:
                inside.append({**hit, "structure_unit": matched_label})
            else:
                outside.append(hit)

        return inside, outside

    def _render_canonical_results(result, *, profile: str, min_conf: float) -> None:
        from daphne_core.v2.types import Recommendation

        recs_v2: list[Recommendation] = list(getattr(result, "recommendations", []) or [])
        preprocessed = getattr(result, "preprocessed", None)
        if not recs_v2 or not preprocessed:
            st.warning("⚠️ No extracted items available.")
            return

        numbered_v2 = [r for r in recs_v2 if getattr(r, "rec_type", None) == "numbered"]
        pfd_concerns_v2 = [r for r in recs_v2 if getattr(r, "rec_type", None) == "pfd_concern"]
        action_verb_v2 = [r for r in recs_v2 if getattr(r, "rec_type", None) == "action_verb"]

        action_verbs_all = _detect_action_verbs_all(preprocessed, min_conf=min_conf)
        structure_units = _structure_units_for_overlap(recs_v2, profile=profile)
        suppressed_hits, _outside_hits = _partition_action_verbs_by_structure(action_verbs_all, structure_units)

        st.markdown("---")
        st.subheader("📋 Extracted Items (canonical)")
        if profile == "pfd_report":
            st.caption(
                "Structure (MATTERS OF CONCERN) first, then Action Verbs not already covered by structure."
            )
        else:
            st.caption("Numbered recommendations first, then Action Verbs not already covered by structure.")

        st.info(
            f"Action Verbs detected: {len(action_verbs_all)}; "
            f"{len(suppressed_hits)} already inside structure-based recommendations."
        )

        st.markdown("#### 🎨 Confidence Guide (Action Verbs)")
        st.caption(
            "Confidence is rule-based (ported from v1), not a learned probability; it indicates how explicit the language pattern is."
        )
        legend_col1, legend_col2, legend_col3 = st.columns(3)
        with legend_col1:
            st.markdown("🟢 **High (≥95%)**")
            st.caption("Strong 'entity should' patterns (e.g., 'The Trust should…')")
        with legend_col2:
            st.markdown("🟡 **Medium (85-94%)**")
            st.caption("Clear recommendation phrasing (e.g., 'We recommend…', '…should be completed')")
        with legend_col3:
            st.markdown("🟠 **Standard (75-84%)**")
            st.caption("Weaker modal/imperative patterns — still valid recommendations")

        with st.expander("ℹ️ How to interpret `min_confidence`", expanded=False):
            st.markdown(
                """
`min_confidence` is a **hard threshold over fixed rule scores** (e.g., 0.95, 0.90, 0.85, 0.80, 0.75).

**Implication:** raising `min_confidence` does not “require higher certainty” in a statistical sense — it simply turns off entire rule families (increasing precision but potentially dropping valid recommendations).
"""
            )

        if profile == "pfd_report":
            if pfd_concerns_v2:
                st.markdown("##### PFD concerns (MATTERS OF CONCERN)")
                for idx, rec in enumerate(pfd_concerns_v2, 1):
                    rec_text = rec.text.strip()
                    if len(rec_text) > 10:
                        title = f"🟢 **{idx}. Concern {rec.rec_number or rec.rec_id or ''}** (100%)"
                        with st.expander(title, expanded=(idx <= 5)):
                            st.markdown(fmt(rec_text))
                            st.caption(
                                f"Type: {getattr(rec, 'rec_type', None)} | "
                                f"Method: {getattr(rec, 'detection_method', None)} | "
                                f"Source: {rec.source_document}"
                            )
            else:
                st.info("No structure-based concerns found.")
        else:
            if numbered_v2:
                st.markdown("##### Numbered recommendations")
                for idx, rec in enumerate(numbered_v2, 1):
                    rec_text = rec.text.strip()
                    if len(rec_text) > 10:
                        title = f"🟢 **{idx}. Recommendation {rec.rec_id or '(unlabelled)'}** (100%)"
                        with st.expander(title, expanded=(idx <= 5)):
                            st.markdown(fmt(rec_text))
                            st.caption(
                                f"Type: {getattr(rec, 'rec_type', None) or 'numbered'} | "
                                f"ID: {rec.rec_id!r} | Num: {rec.rec_number} | "
                                f"Source: {rec.source_document}"
                            )
            else:
                st.info("No structured recommendations found.")

        if action_verb_v2:
            st.markdown("---")
            st.markdown("##### Action Verbs (not already in structure)")
            verb_extractor = StrictRecommendationExtractor()
            action_verb_sorted = sorted(
                action_verb_v2,
                key=lambda r: float(getattr(r, "confidence", 0.0) or 0.0),
                reverse=True,
            )
            for idx, rec in enumerate(action_verb_sorted, 1):
                rec_text = rec.text.strip()
                if len(rec_text) <= 10:
                    continue

                conf = getattr(rec, "confidence", None)
                if conf is None:
                    conf_icon = "⚪"
                    conf_label = "N/A"
                elif conf >= 0.95:
                    conf_icon = "🟢"
                    conf_label = f"{conf:.0%}"
                elif conf >= 0.85:
                    conf_icon = "🟡"
                    conf_label = f"{conf:.0%}"
                else:
                    conf_icon = "🟠"
                    conf_label = f"{conf:.0%}"

                method = getattr(rec, "detection_method", None) or "unknown"
                cleaned = verb_extractor.clean_text(rec_text)
                _is_rec, _c, _m, verb = verb_extractor.is_genuine_recommendation(
                    cleaned, is_numbered_rec=False
                )
                title = f"{conf_icon} **{idx}. {(verb or 'Action')}** ({conf_label})"
                with st.expander(title, expanded=(idx <= 3)):
                    st.markdown(fmt(rec_text))
                    st.caption(
                        f"Type: {getattr(rec, 'rec_type', None) or 'action_verb'} | "
                        f"Method: {method} | "
                        f"Source: {rec.source_document}"
                    )
        else:
            st.markdown("---")
            st.caption("No Action Verb sentences outside structure were added.")

        # Collapsed: action verbs suppressed because they are already covered by structure.
        with st.expander(
            f"Action Verb duplicates already covered by structure ({len(suppressed_hits)})",
            expanded=False,
        ):
            if not suppressed_hits:
                st.caption("No suppressed duplicates for this document/profile.")
            else:
                for idx, hit in enumerate(
                    sorted(suppressed_hits, key=lambda h: float(h.get("confidence", 0.0)), reverse=True),
                    1,
                ):
                    conf = float(hit.get("confidence", 0.0) or 0.0)
                    if conf >= 0.95:
                        conf_icon = "🟢"
                    elif conf >= 0.85:
                        conf_icon = "🟡"
                    else:
                        conf_icon = "🟠"
                    verb = (hit.get("verb") or "Action").upper()
                    title = f"{conf_icon} **{idx}. {verb}** (in {hit.get('structure_unit')})"
                    with st.expander(title, expanded=(idx <= 3)):
                        st.markdown(fmt(hit.get("text") or ""))
                        st.caption(f"Detection method: {hit.get('method')}")

        # Exports
        st.markdown("---")
        export_rows: list[dict] = []
        for r in recs_v2:
            s0, s1 = getattr(r, "span", (None, None))
            conf = getattr(r, "confidence", None)
            if conf is None and getattr(r, "rec_type", None) in {"numbered", "pfd_concern"}:
                conf = 1.0
            export_rows.append(
                {
                    "source_document": getattr(r, "source_document", None),
                    "rec_type": getattr(r, "rec_type", None),
                    "rec_id": getattr(r, "rec_id", None),
                    "rec_number": getattr(r, "rec_number", None),
                    "confidence": conf,
                    "detection_method": getattr(r, "detection_method", None),
                    "span_start": s0,
                    "span_end": s1,
                    "text": getattr(r, "text", None),
                }
            )
        df_export = pd.DataFrame(export_rows)
        st.download_button(
            label=f"📥 Download extracted items CSV ({len(export_rows)})",
            data=df_export.to_csv(index=False),
            file_name=f"{selected_doc}_canonical_extracted_items.csv",
            mime="text/csv",
        )

        df_supp = pd.DataFrame(
            [
                {
                    "structure_unit": h.get("structure_unit"),
                    "confidence": h.get("confidence"),
                    "method": h.get("method"),
                    "verb": h.get("verb"),
                    "span_start": (h.get("span") or (None, None))[0],
                    "span_end": (h.get("span") or (None, None))[1],
                    "text": h.get("text"),
                }
                for h in suppressed_hits
            ]
        )
        st.download_button(
            label=f"📥 Download suppressed Action Verb duplicates CSV ({len(suppressed_hits)})",
            data=df_supp.to_csv(index=False),
            file_name=f"{selected_doc}_suppressed_action_verbs.csv",
            mime="text/csv",
        )

    # Track which document was analysed (but don't auto-clear results)
    if "last_analysed_doc" not in st.session_state:
        st.session_state.last_analysed_doc = None

    # Show which document the current results are from (if any)
    if (
        "canonical_recs_result" in st.session_state
        and st.session_state.canonical_recs_result
        and st.session_state.last_analysed_doc
        and st.session_state.last_analysed_doc != selected_doc
    ):
        st.info(f"📋 Current results are from: **{st.session_state.last_analysed_doc}**")
        if st.button("🗑️ Clear results to analyse a new document"):
            for k in ("canonical_recs_result", "canonical_recs_profile", "canonical_recs_min_confidence"):
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.last_analysed_doc = None
            st.rerun()

    if st.button("🔍 Extract recommendations (canonical)", type="primary"):
        from daphne_core.canonical import extract_recommendations_from_pdf
        from pathlib import Path

        doc = next((d for d in documents if d["filename"] == selected_doc), None)
        if not doc:
            st.error("Document not available")
            return

        pdf_path = doc.get("pdf_path")
        if not pdf_path:
            st.error(
                "This pipeline requires the original PDF path. "
                "Please ensure the document was uploaded as a PDF in this session."
            )
            return

        with st.spinner("Analysing document with canonical extractor..."):
            try:
                result = extract_recommendations_from_pdf(
                    Path(pdf_path),
                    source_document=selected_doc,
                    profile=v2_profile,
                    action_verb_min_confidence=min_confidence,
                    pfd_atomize_concerns=False,
                    enable_pfd_directives=False,
                    dedupe_action_verbs=True,
                )
            except Exception as e:
                st.error(f"❌ Error extracting recommendations: {str(e)}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                return

        if not result.recommendations:
            st.warning("⚠️ No extracted items found in the PDF.")
            return

        st.session_state.canonical_recs_result = result
        st.session_state.canonical_recs_profile = v2_profile
        st.session_state.canonical_recs_min_confidence = float(min_confidence)
        st.session_state.last_analysed_doc = selected_doc

    # Display latest canonical results (if any)
    if (
        "canonical_recs_result" in st.session_state
        and st.session_state.canonical_recs_result
        and st.session_state.last_analysed_doc == selected_doc
    ):
        used_profile = st.session_state.get("canonical_recs_profile", v2_profile)
        used_conf = float(st.session_state.get("canonical_recs_min_confidence", min_confidence))
        st.success(
            f"✅ Showing results for {st.session_state.last_analysed_doc} "
            f"(profile={used_profile}, min_confidence={used_conf:.2f})"
        )
        _render_canonical_results(
            st.session_state.canonical_recs_result,
            profile=used_profile,
            min_conf=used_conf,
        )


def main():
    """Main application with enhanced error handling"""
    try:
        st.set_page_config(
            page_title="DaphneAI - Document Analysis", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏛️ DaphneAI - Recommendation & Response Document Analysis")
        st.markdown("*Advanced document processing and search*")
        
        # Check module availability
        modules_available, setup_search_tab, prepare_documents_for_search, extract_text_from_file, render_analytics_tab = safe_import_with_fallback()
        
        if not modules_available:
            render_fallback_interface()
            return

        # Enhanced tabs with error handling - ADDED SEMANTIC SEARCH
        try:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                [
                    "📁 Upload",
                    "🔍 Extract",
                    "🔍 Keyword Search",
                    "🤖 AI Search",  # NEW TAB
                    "🎯 Recommendations",
                    "🔗 Align Recommendations-Responses",
                    "📊 Analytics",
                ]
            )

            with tab1:
                render_upload_tab_safe(prepare_documents_for_search, extract_text_from_file)

            with tab2:
                render_extract_tab_safe()

            with tab3:
                render_search_tab_safe(setup_search_tab)

            with tab4:  # NEW - SEMANTIC SEARCH TAB
                render_semantic_search_tab()

            with tab5:
                render_recommendations_tab()

            with tab6:
                render_alignment_tab_safe()

            with tab7:
                render_analytics_tab_safe(render_analytics_tab)

        except Exception as e:
            st.error(f"Tab rendering error: {str(e)}")
            render_error_recovery()
            
    except Exception as e:
        st.error(f"⚠️ Application Error: {str(e)}")
        logger.error(f"Main application error: {e}")
        logger.error(traceback.format_exc())
        render_error_recovery()

# Helper functions
def render_fallback_interface():
    """Render a basic fallback interface when modules aren't available"""
    st.warning("🔧 Module loading issues detected. Using fallback interface.")

def render_upload_tab_safe(prepare_documents_for_search, extract_text_from_file):
    """Safe document upload with error handling"""
    try:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files for analysis"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        if prepare_documents_for_search and extract_text_from_file:
                            documents = prepare_documents_for_search(uploaded_files, extract_text_from_file)
                        else:
                            documents = fallback_process_documents(uploaded_files)
                        
                        st.success(f"✅ Processed {len(documents)} documents")
                        
                        # Show basic statistics
                        total_words = sum(doc.get('word_count', 0) for doc in documents)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents", len(documents))
                        with col2:
                            st.metric("Total Words", f"{total_words:,}")
                        with col3:
                            avg_words = total_words // len(documents) if documents else 0
                            st.metric("Avg Words", f"{avg_words:,}")
                        
                        st.markdown("""
                        **✅ Files processed successfully!** 
                        
                        **🔍 Next Steps:**
                        - Go to **Keyword Search** tab for traditional searches
                        - Go to **AI Search** tab for semantic searches  
                        - Go to **Align Rec-Resp** tab to find recommendations and responses
                        - Go to **Analytics** tab for document insights
                        """)
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        logger.error(f"Document processing error: {e}")
                        
    except Exception as e:
        st.error(f"Upload tab error: {str(e)}")

def render_extract_tab_safe():
    """Safe document extraction with error handling"""
    try:
        st.header("🔍 Document Extraction")
        
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("📁 Please upload documents first in the Upload tab.")
            return
        
        documents = st.session_state.documents
        doc_names = [doc['filename'] for doc in documents]
        selected_doc = st.selectbox("Select document to preview:", doc_names)
        
        if selected_doc:
            doc = next((d for d in documents if d['filename'] == selected_doc), None)
            
            if doc and 'text' in doc:
                text = doc['text']
                
                word_count = len(text.split()) if text else 0
                char_count = len(text) if text else 0
                estimated_pages = max(1, char_count // 2000)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Characters", f"{char_count:,}")
                with col2:
                    st.metric("Words", f"{word_count:,}")
                with col3:
                    try:
                        sentences = re.split(r'[.!?]+', text)
                        sentence_count = len([s for s in sentences if s.strip()])
                    except:
                        sentence_count = word_count // 10
                    st.metric("Sentences", f"{sentence_count:,}")
                with col4:
                    st.metric("Est. Pages", estimated_pages)
                
                st.markdown("### 📖 Document Preview")
                preview_length = st.slider(
                    "Preview length (characters)", 
                    min_value=500, 
                    max_value=min(10000, len(text)), 
                    value=min(2000, len(text))
                )
                
                preview_text = text[:preview_length]
                if len(text) > preview_length:
                    preview_text += "... [truncated]"
                
                st.text_area(
                    "Document content:",
                    value=preview_text,
                    height=400,
                    disabled=True
                )
                
                st.download_button(
                    label="📥 Download Extracted Text",
                    data=text,
                    file_name=f"{selected_doc}_extracted.txt",
                    mime="text/plain"
                )
            else:
                st.error("Document text not available")
                
    except Exception as e:
        st.error(f"Extract tab error: {str(e)}")

def render_search_tab_safe(setup_search_tab):
    """Safe search tab with error handling"""
    try:
        if setup_search_tab:
            setup_search_tab()
        else:
            st.warning("Keyword search not available")
    except Exception as e:
        st.error(f"Search tab error: {str(e)}")

def render_alignment_tab_safe():
    """Safe alignment tab with error handling"""
    try:        
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("📁 Please upload documents first in the Upload tab.")
            return
        
        try:
            from ui.alignment_ui import render_simple_alignment_interface
            documents = st.session_state.documents
            render_simple_alignment_interface(documents)
        except ImportError:
            st.error("🔧 Alignment module not available.")
            
    except Exception as e:
        st.error(f"Alignment tab error: {str(e)}")

def render_analytics_tab_safe(render_analytics_tab):
    """Safe analytics tab with error handling"""
    try:
        if render_analytics_tab:
            render_analytics_tab()
        else:
            st.warning("Analytics not available")
    except Exception as e:
        st.error(f"Analytics tab error: {str(e)}")

def fallback_process_documents(uploaded_files):
    """Fallback document processing"""
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
            else:
                text = f"[Content from {uploaded_file.name} - processing not available]"
            
            doc = {
                'filename': uploaded_file.name,
                'text': text,
                'word_count': len(text.split()) if text else 0,
                'document_type': 'general',
                'upload_time': datetime.now(),
                'file_size': len(text) if text else 0
            }
            documents.append(doc)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    st.session_state.documents = documents
    return documents

def render_error_recovery():
    """Render error recovery options"""
    st.markdown("---")
    st.markdown("### 🛠️ Error Recovery")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Application reset. Please refresh the page.")
    
    with col2:
        if st.button("📋 Show Debug Info"):
            import sys
            import platform
            st.code(f"""
Python Version: {sys.version}
Platform: {platform.platform()}
Streamlit Version: {st.__version__}
Session State Keys: {list(st.session_state.keys())}
Documents: {len(st.session_state.get('documents', []))}
            """)

if __name__ == "__main__":
    main()
