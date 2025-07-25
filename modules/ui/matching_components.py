# ===============================================
# FILE: modules/ui/matching_components.py (UPDATED VERSION)
# ===============================================

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import required modules with error handling
try:
    import sys
    sys.path.append('modules')
    from vector_store import VectorStoreManager
    from rag_engine import RAGQueryEngine
    from recommendation_matcher import RecommendationResponseMatcher
    from bert_annotator import BERTConceptAnnotator
    from .shared_components import add_error_message, show_progress_indicator
except ImportError as e:
    logging.error(f"Import error in matching_components: {e}")
    # Create mock classes for development
    class VectorStoreManager:
        def add_documents(self, docs): return True
        def get_collection_stats(self): return {'total_documents': 0}
    class RAGQueryEngine:
        def __init__(self, vs): pass
    class RecommendationResponseMatcher:
        def __init__(self, rag, bert): pass
        def match_recommendation_to_responses(self, rec): return []
    class BERTConceptAnnotator:
        def __init__(self): pass

def render_matching_tab():
    """Render the response matching tab"""
    st.header("🔗 Find Responses to Recommendations")
    
    if not st.session_state.extracted_recommendations:
        st.warning("⚠️ Please extract recommendations first in the Extract Content tab.")
        return
    
    st.markdown("""
    Find responses to recommendations using AI-powered semantic search combined with 
    concept matching. This uses both RAG (Retrieval-Augmented Generation) and BERT 
    concept validation for high-accuracy matching.
    """)
    
    # Initialize systems
    initialize_matching_systems()
    
    # Document indexing
    render_indexing_interface()
    
    # Matching interface
    render_matching_interface()
    
    # Display results
    display_matching_results()

def initialize_matching_systems():
    """Initialize vector store, RAG engine, and matcher"""
    # Initialize vector store manager
    if not st.session_state.vector_store_manager:
        with st.spinner("🚀 Initializing vector store..."):
            try:
                st.session_state.vector_store_manager = VectorStoreManager()
                st.success("✅ Vector store initialized!")
            except Exception as e:
                st.error(f"❌ Vector store initialization failed: {str(e)}")
                add_error_message(f"Vector store init failed: {str(e)}")
                st.session_state.vector_store_manager = VectorStoreManager()
    
    # Initialize RAG engine
    if not st.session_state.rag_engine:
        with st.spinner("🧠 Initializing RAG engine..."):
            try:
                st.session_state.rag_engine = RAGQueryEngine(st.session_state.vector_store_manager)
                st.success("✅ RAG engine initialized!")
            except Exception as e:
                st.error(f"❌ RAG engine initialization failed: {str(e)}")
                add_error_message(f"RAG engine init failed: {str(e)}")
    
    # Initialize BERT annotator
    if not st.session_state.bert_annotator:
        with st.spinner("🤖 Initializing BERT annotator..."):
            try:
                st.session_state.bert_annotator = BERTConceptAnnotator()
                st.success("✅ BERT annotator initialized!")
            except Exception as e:
                st.error(f"❌ BERT annotator initialization failed: {str(e)}")
                add_error_message(f"BERT init failed: {str(e)}")
    
    # Initialize matcher
    if not st.session_state.recommendation_matcher:
        if st.session_state.rag_engine and st.session_state.bert_annotator:
            with st.spinner("🔗 Initializing recommendation matcher..."):
                try:
                    st.session_state.recommendation_matcher = RecommendationResponseMatcher(
                        st.session_state.rag_engine, 
                        st.session_state.bert_annotator
                    )
                    st.success("✅ Recommendation matcher initialized!")
                except Exception as e:
                    st.error(f"❌ Matcher initialization failed: {str(e)}")
                    add_error_message(f"Matcher init failed: {str(e)}")

def render_indexing_interface():
    """Render document indexing interface with section awareness"""
    st.subheader("📚 Document Indexing")
    
    # ✅ UPDATED: Enhanced indexing status with section information
    vector_store = st.session_state.vector_store_manager
    if vector_store:
        stats = vector_store.get_collection_stats()
        indexed_docs = stats.get('total_documents', 0)
        
        # Get document statistics
        docs = st.session_state.uploaded_documents
        total_docs = len(docs)
        sections_docs = len([d for d in docs if d.get('sections')])
        total_sections = sum(len(d.get('sections', [])) for d in docs)
        
        # Display enhanced metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents Available", total_docs)
        
        with col2:
            st.metric("With Sections", sections_docs)
        
        with col3:
            st.metric("Total Sections", total_sections)
        
        with col4:
            st.metric("Chunks Indexed", indexed_docs)
        
        # ✅ NEW: Show section breakdown
        if total_sections > 0:
            with st.expander("📋 Section Breakdown", expanded=False):
                section_types = {}
                for doc in docs:
                    for section in doc.get('sections', []):
                        sec_type = section['type']
                        section_types[sec_type] = section_types.get(sec_type, 0) + 1
                
                for sec_type, count in section_types.items():
                    st.write(f"• {sec_type.title()}: {count} sections")
        
        # Indexing controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📇 Index All Documents", type="primary", use_container_width=True):
                index_documents_for_search()
        
        with col2:
            if st.button("🔄 Re-index Documents", use_container_width=True):
                reindex_documents()
        
        # ✅ NEW: Indexing options
        with st.expander("⚙️ Indexing Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                index_mode = st.radio(
                    "What to index:",
                    ["All Content", "Sections Only", "Full Documents Only"],
                    help="Choose what content to include in the search index"
                )
            
            with col2:
                chunk_strategy = st.radio(
                    "Chunking strategy:",
                    ["Smart Chunking", "Fixed Size", "Section-based"],
                    help="How to split documents for indexing"
                )
            
            # Store options in session state
            st.session_state.index_mode = index_mode
            st.session_state.chunk_strategy = chunk_strategy
        
        # Show content distribution if available
        content_dist = stats.get('content_distribution', {})
        if content_dist:
            st.subheader("📊 Indexed Content Distribution")
            dist_df = pd.DataFrame(list(content_dist.items()), columns=['Content Type', 'Count'])
            st.bar_chart(dist_df.set_index('Content Type'))
    
    else:
        st.error("Vector store not available. Please refresh the page.")

def index_documents_for_search():
    """✅ UPDATED: Index uploaded documents with section awareness"""
    if not st.session_state.uploaded_documents:
        st.warning("No documents available to index.")
        return
    
    vector_store = st.session_state.vector_store_manager
    if not vector_store:
        st.error("Vector store not initialized.")
        return
    
    # Get indexing preferences
    index_mode = st.session_state.get('index_mode', 'All Content')
    chunk_strategy = st.session_state.get('chunk_strategy', 'Smart Chunking')
    
    # Set processing status
    st.session_state.processing_status = "indexing"
    
    with st.spinner("📇 Indexing documents for semantic search..."):
        try:
            # ✅ UPDATED: Prepare documents with section awareness
            documents_for_indexing = prepare_documents_for_indexing(
                st.session_state.uploaded_documents, 
                index_mode, 
                chunk_strategy
            )
            
            if not documents_for_indexing:
                st.warning("No content prepared for indexing.")
                return
            
            # Index documents
            success = vector_store.add_documents(documents_for_indexing)
            
            if success:
                st.success(f"✅ Successfully indexed {len(documents_for_indexing)} document chunks!")
                
                # ✅ UPDATED: Enhanced success message with section info
                sections_indexed = sum(1 for doc in documents_for_indexing 
                                     if doc.get('metadata', {}).get('content_source') == 'section')
                full_docs_indexed = len(documents_for_indexing) - sections_indexed
                
                if sections_indexed > 0:
                    st.info(f"📋 Indexed {sections_indexed} sections and {full_docs_indexed} full document chunks")
                
                # Update stats
                stats = vector_store.get_collection_stats()
                st.info(f"📊 Vector store now contains {stats.get('total_documents', 0)} total chunks")
            else:
                st.error("❌ Failed to index documents. Check logs for details.")
                
        except Exception as e:
            error_msg = f"Error indexing documents: {str(e)}"
            st.error(f"❌ {error_msg}")
            add_error_message(error_msg)
            logging.error(f"Indexing error: {e}", exc_info=True)
        
        finally:
            st.session_state.processing_status = "idle"

def prepare_documents_for_indexing(docs: List[Dict], index_mode: str, chunk_strategy: str) -> List[Dict]:
    """
    ✅ NEW: Prepare documents for indexing with section awareness
    """
    documents_for_indexing = []
    
    for doc in docs:
        base_metadata = {
            'source': doc['filename'],
            'document_type': doc['document_type'],
            'upload_time': doc.get('upload_time', ''),
            'file_size': doc.get('file_size', 0),
            'extraction_type': doc.get('extraction_type', 'unknown')
        }
        
        # Merge original metadata
        if 'metadata' in doc:
            base_metadata.update(doc['metadata'])
        
        sections = doc.get('sections', [])
        
        if index_mode == "Sections Only" and sections:
            # Index only sections
            for section in sections:
                section_doc = {
                    'content': section['content'],
                    'source': doc['filename'],
                    'document_type': f"{doc['document_type']} - {section['type'].title()} Section",
                    'metadata': {
                        **base_metadata,
                        'content_source': 'section',
                        'section_type': section['type'],
                        'section_title': section['title'],
                        'page_start': section['page_start'],
                        'page_end': section['page_end'],
                        'section_word_count': section.get('content_stats', {}).get('word_count', 0)
                    }
                }
                documents_for_indexing.append(section_doc)
        
        elif index_mode == "Full Documents Only" or not sections:
            # Index full document content
            full_doc = {
                'content': doc['content'],
                'source': doc['filename'],
                'document_type': doc['document_type'],
                'metadata': {
                    **base_metadata,
                    'content_source': 'full_document',
                    'total_sections': len(sections)
                }
            }
            documents_for_indexing.append(full_doc)
        
        else:  # "All Content" - index both sections and full content
            # Add sections if available
            if sections:
                for section in sections:
                    section_doc = {
                        'content': section['content'],
                        'source': doc['filename'],
                        'document_type': f"{doc['document_type']} - {section['type'].title()} Section",
                        'metadata': {
                            **base_metadata,
                            'content_source': 'section',
                            'section_type': section['type'],
                            'section_title': section['title'],
                            'page_start': section['page_start'],
                            'page_end': section['page_end']
                        }
                    }
                    documents_for_indexing.append(section_doc)
            
            # Also add full document for broader context
            full_doc = {
                'content': doc['content'],
                'source': doc['filename'],
                'document_type': doc['document_type'],
                'metadata': {
                    **base_metadata,
                    'content_source': 'full_document',
                    'total_sections': len(sections)
                }
            }
            documents_for_indexing.append(full_doc)
    
    logging.info(f"Prepared {len(documents_for_indexing)} document chunks for indexing")
    return documents_for_indexing

def reindex_documents():
    """Re-index all documents (clear and re-add)"""
    st.warning("⚠️ This will clear existing index and re-index all documents.")
    
    if st.button("⚠️ Confirm Re-indexing", type="secondary"):
        # Clear vector store first (if method exists)
        vector_store = st.session_state.vector_store_manager
        if vector_store and hasattr(vector_store, 'clear_collection'):
            try:
                vector_store.clear_collection()
                st.info("🗑️ Cleared existing index")
            except Exception as e:
                st.warning(f"Could not clear existing index: {e}")
        
        # Re-index documents
        index_documents_for_search()

def render_matching_interface():
    """Render the main matching interface"""
    st.subheader("🎯 Response Matching")
    
    recommendations = st.session_state.extracted_recommendations
    
    if not recommendations:
        st.info("No recommendations available for matching.")
        return
    
    # Check if documents are indexed
    vector_store = st.session_state.vector_store_manager
    if vector_store:
        stats = vector_store.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            st.warning("⚠️ No documents indexed yet. Please index documents first using the button above.")
            return
    
    # Matching configuration
    render_matching_configuration()
    
    # Recommendation selection
    render_recommendation_selection()
    
    # Batch matching options
    render_batch_matching_options()

def render_matching_configuration():
    """Render matching configuration options"""
    with st.expander("⚙️ Matching Configuration"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            similarity_threshold = st.slider(
                "Similarity Threshold",
                0.3, 0.9, 0.7, 0.05,
                help="Minimum similarity for response matching",
                key="matching_similarity_threshold"
            )
        
        with col2:
            max_responses = st.number_input(
                "Max Responses per Recommendation",
                1, 20, 10,
                help="Maximum number of responses to find",
                key="max_responses_per_rec"
            )
        
        with col3:
            concept_weight = st.slider(
                "Concept Weight",
                0.0, 1.0, 0.3, 0.05,
                help="Weight of concept matching vs semantic similarity",
                key="concept_weight"
            )
        
        # ✅ NEW: Section-specific matching options
        col1, col2 = st.columns(2)
        
        with col1:
            prefer_sections = st.checkbox(
                "Prefer Section Matches",
                value=True,
                help="Prioritize matches from extracted sections over full document content"
            )
            st.session_state.prefer_sections = prefer_sections
        
        with col2:
            use_concept_validation = st.checkbox(
                "Use Concept Validation",
                value=True,
                help="Validate matches using BERT concept analysis"
            )
            st.session_state.use_concept_validation = use_concept_validation

def render_recommendation_selection():
    """Render recommendation selection for matching"""
    st.subheader("📋 Select Recommendations to Match")
    
    recommendations = st.session_state.extracted_recommendations
    
    # Create recommendation display options
    rec_options = []
    for i, rec in enumerate(recommendations):
        rec_text = rec.get('text', '')[:100]
        rec_id = rec.get('id', f'rec_{i+1}')
        doc_source = rec.get('document_source', 'Unknown')
        rec_options.append(f"{rec_id}: {rec_text}... (from {doc_source})")
    
    # Selection interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_recs = st.multiselect(
            "Choose recommendations to find responses for:",
            rec_options,
            default=rec_options[:5] if len(rec_options) > 5 else rec_options,
            key="selected_recommendations_for_matching"
        )
    
    with col2:
        if st.button("📋 Select All", use_container_width=True):
            st.session_state.selected_recommendations_for_matching = rec_options
            st.rerun()
        
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.selected_recommendations_for_matching = []
            st.rerun()
    
    # Store selected indices
    selected_indices = [rec_options.index(sel) for sel in selected_recs if sel in rec_options]
    st.session_state.selected_rec_indices = selected_indices
    
    if selected_recs:
        st.success(f"✅ Selected {len(selected_recs)} recommendations for matching")

def render_batch_matching_options():
    """Render batch matching execution options"""
    if not st.session_state.get('selected_rec_indices'):
        return
    
    st.subheader("🚀 Execute Matching")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Find Responses", type="primary", use_container_width=True):
            execute_batch_matching()
    
    with col2:
        if st.button("📊 Quick Analysis", use_container_width=True):
            execute_quick_analysis()
    
    with col3:
        if st.button("📋 Export Matches", use_container_width=True):
            export_matching_results()

def execute_batch_matching():
    """✅ UPDATED: Execute batch matching with section awareness"""
    selected_indices = st.session_state.get('selected_rec_indices', [])
    recommendations = st.session_state.extracted_recommendations
    
    if not selected_indices:
        st.warning("No recommendations selected for matching.")
        return
    
    # Get matcher
    matcher = st.session_state.recommendation_matcher
    if not matcher:
        st.error("Recommendation matcher not initialized.")
        return
    
    st.subheader("🔗 Finding Responses...")
    
    # Progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_matches = {}
        processing_stats = {
            'total_processed': 0,
            'total_matches_found': 0,
            'section_matches': 0,
            'full_doc_matches': 0,
            'high_confidence_matches': 0
        }
        
        for i, rec_idx in enumerate(selected_indices):
            if rec_idx >= len(recommendations):
                continue
                
            rec = recommendations[rec_idx]
            
            # Update progress
            progress = (i + 1) / len(selected_indices)
            progress_bar.progress(progress)
            status_text.text(f"Matching: {rec.get('id', f'Rec {rec_idx+1}')}")
            
            try:
                # Execute matching
                matches = matcher.match_recommendation_to_responses(rec)
                
                # ✅ UPDATED: Enhanced match processing with section info
                enhanced_matches = []
                for match in matches:
                    match_info = match.copy()
                    
                    # Check if match is from a section
                    metadata = match.get('metadata', {})
                    content_source = metadata.get('content_source', 'unknown')
                    
                    if content_source == 'section':
                        match_info['match_source'] = 'section'
                        match_info['section_type'] = metadata.get('section_type', 'unknown')
                        match_info['section_pages'] = f"{metadata.get('page_start', '')}-{metadata.get('page_end', '')}"
                        processing_stats['section_matches'] += 1
                    else:
                        match_info['match_source'] = 'full_document'
                        processing_stats['full_doc_matches'] += 1
                    
                    # Track high confidence matches
                    if match.get('confidence', 0) > 0.8:
                        processing_stats['high_confidence_matches'] += 1
                    
                    enhanced_matches.append(match_info)
                
                all_matches[rec.get('id', f'rec_{rec_idx}')] = enhanced_matches
                processing_stats['total_matches_found'] += len(enhanced_matches)
                processing_stats['total_processed'] += 1
                
            except Exception as e:
                st.error(f"Error matching recommendation {rec.get('id', rec_idx)}: {e}")
                continue
    
    # Clear progress indicators
    progress_container.empty()
    
    # Store results
    st.session_state.matching_results = {
        'matches': all_matches,
        'stats': processing_stats,
        'timestamp': datetime.now(),
        'config': {
            'similarity_threshold': st.session_state.get('matching_similarity_threshold', 0.7),
            'max_responses': st.session_state.get('max_responses_per_rec', 10),
            'prefer_sections': st.session_state.get('prefer_sections', True),
            'use_concept_validation': st.session_state.get('use_concept_validation', True)
        }
    }
    
    # Display results
    with results_container:
        display_batch_matching_results(all_matches, processing_stats)

def display_batch_matching_results(matches: Dict, stats: Dict):
    """✅ UPDATED: Display batch matching results with section information"""
    total_recs = stats.get('total_processed', 0)
    total_matches = stats.get('total_matches_found', 0)
    section_matches = stats.get('section_matches', 0)
    full_doc_matches = stats.get('full_doc_matches', 0)
    high_conf_matches = stats.get('high_confidence_matches', 0)
    
    # Summary
    st.success(f"🎉 **Matching completed!** Found **{total_matches}** potential responses for **{total_recs}** recommendations.")
    
    # ✅ NEW: Enhanced statistics with section breakdown
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", total_matches)
    
    with col2:
        st.metric("Section Matches", section_matches)
    
    with col3:
        st.metric("Full Doc Matches", full_doc_matches)
    
    with col4:
        st.metric("High Confidence", high_conf_matches)
    
    if total_matches > 0:
        # Display individual matches
        for rec_id, rec_matches in matches.items():
            if rec_matches:
                with st.expander(f"📋 {rec_id} ({len(rec_matches)} matches found)", expanded=False):
                    for i, match in enumerate(rec_matches, 1):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Match {i}:**")
                            st.write(match.get('text', 'No text available')[:300] + "...")
                        
                        with col2:
                            st.write(f"**Confidence:** {match.get('confidence', 0):.2f}")
                            st.write(f"**Source:** {match.get('source', 'Unknown')}")
                            
                            # ✅ NEW: Show section information
                            match_source = match.get('match_source', 'unknown')
                            if match_source == 'section':
                                st.write(f"**Section:** {match.get('section_type', 'Unknown').title()}")
                                section_pages = match.get('section_pages', '')
                                if section_pages and section_pages != '-':
                                    st.write(f"**Pages:** {section_pages}")
                            else:
                                st.write(f"**Type:** Full document")
                        
                        st.write("---")

def execute_quick_analysis():
    """Execute quick analysis of selected recommendations"""
    st.info("🚧 Quick analysis feature coming soon!")

def export_matching_results():
    """Export matching results with section information"""
    if not st.session_state.get('matching_results'):
        st.warning("No matching results to export.")
        return
    
    matches = st.session_state.matching_results.get('matches', {})
    
    # ✅ UPDATED: Enhanced export with section information
    export_data = []
    for rec_id, rec_matches in matches.items():
        for i, match in enumerate(rec_matches, 1):
            export_row = {
                'recommendation_id': rec_id,
                'match_number': i,
                'confidence': match.get('confidence', 0),
                'match_text': match.get('text', ''),
                'source_document': match.get('source', ''),
                'match_source_type': match.get('match_source', 'unknown'),
                'section_type': match.get('section_type', ''),
                'section_pages': match.get('section_pages', ''),
                'timestamp': datetime.now().isoformat()
            }
            export_data.append(export_row)
    
    if export_data:
        df = pd.DataFrame(export_data)
        csv_data = df.to_csv(index=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"matching_results_{timestamp}.csv"
        
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
        
        st.success(f"✅ Export ready! {len(export_data)} matches included.")

def display_matching_results():
    """Display previously generated matching results"""
    if not st.session_state.get('matching_results'):
        return
    
    results = st.session_state.matching_results
    matches = results.get('matches', {})
    stats = results.get('stats', {})
    timestamp = results.get('timestamp')
    
    st.subheader("📊 Previous Matching Results")
    
    if timestamp:
        st.write(f"**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Recommendations Processed", stats.get('total_processed', 0))
    
    with col2:
        st.metric("Total Matches", stats.get('total_matches_found', 0))
    
    with col3:
        st.metric("High Confidence", stats.get('high_confidence_matches', 0))
    
    # ✅ NEW: Section breakdown in results
    section_matches = stats.get('section_matches', 0)
    full_doc_matches = stats.get('full_doc_matches', 0)
    
    if section_matches > 0 or full_doc_matches > 0:
        st.write("**Match Source Breakdown:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• Section matches: {section_matches}")
        with col2:
            st.write(f"• Full document matches: {full_doc_matches}")
    
    # Show detailed results
    if matches:
        display_option = st.radio(
            "Display format:",
            ["Summary View", "Detailed View", "Export View"],
            horizontal=True,
            key="previous_matching_display"
        )
        
        if display_option == "Summary View":
            show_matching_summary(matches)
        elif display_option == "Detailed View":
            show_matching_detailed(matches)
        elif display_option == "Export View":
            export_matching_results()

def show_matching_summary(matches: Dict):
    """Show matching results in summary format"""
    for rec_id, rec_matches in matches.items():
        if rec_matches:
            with st.expander(f"📋 {rec_id} - {len(rec_matches)} matches", expanded=False):
                for i, match in enumerate(rec_matches[:3], 1):  # Show top 3
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Match {i}:** {match.get('text', '')[:150]}...")
                    
                    with col2:
                        st.write(f"**Confidence:** {match.get('confidence', 0):.2f}")
                        match_source = match.get('match_source', 'unknown')
                        if match_source == 'section':
                            st.write(f"**Section:** {match.get('section_type', 'Unknown')}")
                        else:
                            st.write(f"**Source:** Full doc")
                
                if len(rec_matches) > 3:
                    st.write(f"... and {len(rec_matches) - 3} more matches")

def show_matching_detailed(matches: Dict):
    """Show matching results in detailed format"""
    for rec_id, rec_matches in matches.items():
        if rec_matches:
            st.markdown(f"### 📋 {rec_id}")
            
            for i, match in enumerate(rec_matches, 1):
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Match {i}:**")
                        st.write(match.get('text', 'No text available'))
                    
                    with col2:
                        st.write(f"**Confidence:** {match.get('confidence', 0):.2f}")
                        st.write(f"**Source:** {match.get('source', 'Unknown')}")
                        
                        # Enhanced section information
                        match_source = match.get('match_source', 'unknown')
                        if match_source == 'section':
                            st.write(f"**Section Type:** {match.get('section_type', 'Unknown').title()}")
                            section_pages = match.get('section_pages', '')
                            if section_pages and section_pages != '-':
                                st.write(f"**Pages:** {section_pages}")
                        else:
                            st.write(f"**Type:** Full document match")
                        
                        # Additional metadata if available
                        metadata = match.get('metadata', {})
                        if metadata.get('section_title'):
                            st.write(f"**Section Title:** {metadata['section_title']}")
                    
                    st.markdown("---")

# ✅ NEW: Section-aware utility functions
def get_matching_statistics():
    """Get comprehensive matching statistics"""
    if not st.session_state.get('matching_results'):
        return {}
    
    results = st.session_state.matching_results
    matches = results.get('matches', {})
    stats = results.get('stats', {})
    
    # Calculate additional statistics
    total_recommendations = len(matches)
    recommendations_with_matches = len([rec for rec, rec_matches in matches.items() if rec_matches])
    
    avg_matches_per_rec = (stats.get('total_matches_found', 0) / total_recommendations 
                          if total_recommendations > 0 else 0)
    
    section_match_rate = (stats.get('section_matches', 0) / stats.get('total_matches_found', 1) * 100
                         if stats.get('total_matches_found', 0) > 0 else 0)
    
    return {
        'total_recommendations': total_recommendations,
        'recommendations_with_matches': recommendations_with_matches,
        'match_rate': recommendations_with_matches / total_recommendations * 100 if total_recommendations > 0 else 0,
        'avg_matches_per_recommendation': avg_matches_per_rec,
        'section_match_percentage': section_match_rate,
        'high_confidence_rate': (stats.get('high_confidence_matches', 0) / stats.get('total_matches_found', 1) * 100
                               if stats.get('total_matches_found', 0) > 0 else 0)
    }

def validate_matching_setup():
    """Validate that matching system is properly set up"""
    issues = []
    
    # Check documents
    if not st.session_state.uploaded_documents:
        issues.append("No documents uploaded")
    
    # Check recommendations
    if not st.session_state.extracted_recommendations:
        issues.append("No recommendations extracted")
    
    # Check vector store
    vector_store = st.session_state.vector_store_manager
    if not vector_store:
        issues.append("Vector store not initialized")
    else:
        stats = vector_store.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            issues.append("No documents indexed in vector store")
    
    # Check RAG engine
    if not st.session_state.rag_engine:
        issues.append("RAG engine not initialized")
    
    # Check matcher
    if not st.session_state.recommendation_matcher:
        issues.append("Recommendation matcher not initialized")
    
    return len(issues) == 0, issues

def show_matching_help():
    """Show help information for matching functionality"""
    with st.expander("❓ Matching Help", expanded=False):
        st.markdown("""
        ### How Response Matching Works:
        
        **1. Document Indexing:**
        - Documents are split into searchable chunks
        - Sections are indexed separately for precise matching
        - Vector embeddings are created for semantic search
        
        **2. Similarity Search:**
        - Uses AI embeddings to find semantically similar content
        - Searches both section content and full document chunks
        - Returns ranked results by similarity score
        
        **3. Concept Validation:**
        - BERT analyzes conceptual themes in recommendations and responses
        - Validates matches based on shared concepts
        - Improves accuracy by filtering semantically similar but topically different content
        
        **4. Section-Aware Matching:**
        - Prioritizes matches from relevant sections (recommendations/responses)
        - Tracks page numbers for precise source location
        - Distinguishes between section matches and full document matches
        
        ### Tips for Better Results:
        - **Index all content** for comprehensive coverage
        - **Use section extraction** when uploading documents for focused matching
        - **Adjust similarity threshold** based on your precision/recall needs
        - **Enable concept validation** for higher accuracy
        - **Check section matches first** - they're often more relevant
        """)

# Initialize matching-related session state
def initialize_matching_state():
    """Initialize matching-specific session state variables"""
    if 'matching_results' not in st.session_state:
        st.session_state.matching_results = {}
    
    if 'selected_rec_indices' not in st.session_state:
        st.session_state.selected_rec_indices = []
    
    if 'prefer_sections' not in st.session_state:
        st.session_state.prefer_sections = True
    
    if 'use_concept_validation' not in st.session_state:
        st.session_state.use_concept_validation = True
    
    if 'index_mode' not in st.session_state:
        st.session_state.index_mode = 'All Content'
    
    if 'chunk_strategy' not in st.session_state:
        st.session_state.chunk_strategy = 'Smart Chunking'

# Call initialization
initialize_matching_state()

# Error handling wrapper for matching operations
def safe_matching_operation(func, *args, **kwargs):
    """Wrapper for safe execution of matching operations"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Matching operation failed: {e}", exc_info=True)
        st.error(f"Matching operation failed: {str(e)}")
        return None

# Export key functions for use by other components
__all__ = [
    'render_matching_tab',
    'index_documents_for_search',
    'get_matching_statistics',
    'validate_matching_setup'
]
