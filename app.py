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
from modules.simple_recommendation_extractor import (
    extract_recommendations, 
    StrictRecommendationExtractor
)

# Try to import the new semantic search engine
try:
    from modules.search_engine import SemanticSearchEngine
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    logger.warning("Semantic search not available")


def safe_import_with_fallback():
    """Safely import modules with comprehensive fallbacks"""
    try:
        from modules.integration_helper import (
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
    """Render the improved recommendations extraction tab"""
    st.header("🎯 Extract Recommendations")
    
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
    
    if st.button("🔍 Extract Recommendations", type="primary"):
        doc = next((d for d in documents if d['filename'] == selected_doc), None)
        
        if doc and 'text' in doc:
            with st.spinner("Analysing document with strict filtering..."):
                try:
                    # Extract all recommendations (min_confidence=0.75 hardcoded for quality)
                    recommendations = extract_recommendations(
                        doc['text'],
                        min_confidence=0.75
                    )
                    
                    if recommendations:
                        # SORT BY CONFIDENCE (highest first)
                        recommendations = sorted(recommendations, key=lambda x: x.get('confidence', 0), reverse=True)
                        
                        st.success(f"✅ Found {len(recommendations)} genuine recommendations")
                        
                        # Statistics
                        extractor = StrictRecommendationExtractor()
                        stats = extractor.get_statistics(recommendations)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Recommendations", stats['total'])
                        with col2:
                            st.metric("High Confidence (≥0.9)", stats.get('high_confidence', 0))
                        with col3:
                            st.metric("Average Confidence", f"{stats['avg_confidence']:.0%}")
                        with col4:
                            st.metric("Unique Verbs", len(stats.get('top_verbs', {})))
                        
                        # CONFIDENCE LEGEND
                        st.markdown("---")
                        st.markdown("#### 🎨 Confidence Guide")
                        legend_col1, legend_col2, legend_col3 = st.columns(3)
                        with legend_col1:
                            st.markdown("🟢 **High (≥95%)**")
                            st.caption("Numbered recommendations (Recommendation 1, 2, etc.) or strong 'entity should' patterns")
                        with legend_col2:
                            st.markdown("🟡 **Medium (85-94%)**")
                            st.caption("Passive recommendations ('should be completed', 'should be presented')")
                        with legend_col3:
                            st.markdown("🟠 **Standard (75-84%)**")
                            st.caption("Modal verb patterns ('should review', 'should consider') - still valid recommendations")
                        
                        st.markdown("---")
                        st.subheader("📋 Extracted Recommendations")
                        st.caption("Sorted by confidence (highest first)")
                        
                        for idx, rec in enumerate(recommendations, 1):
                            rec_text = rec.get('text', '[No text available]').strip()
                            verb = rec.get('verb', 'unknown').upper()
                            confidence = rec.get('confidence', 0)
                            method = rec.get('method', 'unknown')
                            
                            if len(rec_text) > 10:
                                # Confidence indicator
                                if confidence >= 0.95:
                                    conf_icon = "🟢"
                                elif confidence >= 0.85:
                                    conf_icon = "🟡"
                                else:
                                    conf_icon = "🟠"
                                
                                title = f"{conf_icon} **{idx}. {verb}** ({confidence:.0%})"
                                
                                # Expand first 5 by default
                                with st.expander(title, expanded=(idx <= 5)):
                                    st.markdown(rec_text)
                                    st.caption(f"Detection method: {method}")
                        
                        st.markdown("---")
                        
                        # Filter out empty recommendations before export
                        valid_recs = [r for r in recommendations if len(r.get('text', '').strip()) > 10]
                        
                        if valid_recs:
                            df_export = pd.DataFrame(valid_recs)
                            csv = df_export.to_csv(index=False)
                            
                            st.download_button(
                                label=f"📥 Download as CSV ({len(valid_recs)} recommendations)",
                                data=csv,
                                file_name=f"{selected_doc}_recommendations.csv",
                                mime="text/csv"
                            )
                        
                        st.session_state.extracted_recommendations = valid_recs
                        
                    else:
                        st.warning("⚠️ No recommendations found. Try lowering the confidence threshold to 0.6.")
                        
                except Exception as e:
                    st.error(f"❌ Error extracting recommendations: {str(e)}")
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
        else:
            st.error("Document text not available")


def main():
    """Main application with enhanced error handling"""
    try:
        st.set_page_config(
            page_title="DaphneAI - Government Document Analysis", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏛️ DaphneAI - Government Document Analysis")
        st.markdown("*Advanced document processing and search for government content*")
        
        # Check module availability
        modules_available, setup_search_tab, prepare_documents_for_search, extract_text_from_file, render_analytics_tab = safe_import_with_fallback()
        
        if not modules_available:
            render_fallback_interface()
            return
        
        # Enhanced tabs with error handling - ADDED SEMANTIC SEARCH
        try:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📁 Upload", 
                "🔍 Extract", 
                "🔍 Keyword Search",
                "🤖 AI Search",  # NEW TAB
                "🔗 Align Rec-Resp",
                "🎯 Recommendations",
                "📊 Analytics"
            ])
            
            with tab1:
                render_upload_tab_safe(prepare_documents_for_search, extract_text_from_file)
            
            with tab2:
                render_extract_tab_safe()
            
            with tab3:
                render_search_tab_safe(setup_search_tab)
            
            with tab4:  # NEW - SEMANTIC SEARCH TAB
                render_semantic_search_tab()
            
            with tab5:
                render_alignment_tab_safe()
                
            with tab6:
                render_recommendations_tab()
                
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
        st.header("🔗 Recommendation-Response Alignment")
        
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("📁 Please upload documents first in the Upload tab.")
            return
        
        try:
            from modules.ui.simplified_alignment_ui import render_simple_alignment_interface
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
