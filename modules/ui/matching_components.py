# ===============================================
# FILE: modules/ui/matching_components.py
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
                st.session_state.rag_engine = RAGQueryEngine(st.session_state.vector_store_manager)
    
    # Initialize BERT annotator if not already done
    if not st.session_state.bert_annotator:
        with st.spinner("🏷️ Initializing BERT annotator..."):
            try:
                st.session_state.bert_annotator = BERTConceptAnnotator()
                st.success("✅ BERT annotator initialized!")
            except Exception as e:
                st.error(f"❌ BERT annotator initialization failed: {str(e)}")
                add_error_message(f"BERT annotator init failed: {str(e)}")
                st.session_state.bert_annotator = BERTConceptAnnotator()
    
    # Initialize recommendation matcher
    if not st.session_state.recommendation_matcher:
        if st.session_state.rag_engine and st.session_state.bert_annotator:
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
    """Render document indexing interface"""
    st.subheader("📚 Document Indexing")
    
    # Check indexing status
    vector_store = st.session_state.vector_store_manager
    if vector_store:
        stats = vector_store.get_collection_stats()
        indexed_docs = stats.get('total_documents', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents Indexed", indexed_docs)
        
        with col2:
            st.metric("Available Documents", len(st.session_state.uploaded_documents))
        
        with col3:
            embedding_dim = stats.get('embedding_dimension', 'Unknown')
            st.metric("Embedding Dimension", embedding_dim)
        
        # Indexing controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📇 Index All Documents", type="primary", use_container_width=True):
                index_documents_for_search()
        
        with col2:
            if st.button("🔄 Re-index Documents", use_container_width=True):
                reindex_documents()
        
        # Show content distribution if available
        content_dist = stats.get('content_distribution', {})
        if content_dist:
            st.subheader("📊 Indexed Content Distribution")
            dist_df = pd.DataFrame(list(content_dist.items()), columns=['Content Type', 'Count'])
            st.bar_chart(dist_df.set_index('Content Type'))
    
    else:
        st.error("Vector store not available. Please refresh the page.")

def index_documents_for_search():
    """Index uploaded documents in the vector store"""
    if not st.session_state.uploaded_documents:
        st.warning("No documents available to index.")
        return
    
    vector_store = st.session_state.vector_store_manager
    if not vector_store:
        st.error("Vector store not initialized.")
        return
    
    # Set processing status
    st.session_state.processing_status = "indexing"
    
    with st.spinner("📇 Indexing documents for semantic search..."):
        try:
            # Prepare documents for indexing
            documents_for_indexing = []
            
            for doc in st.session_state.uploaded_documents:
                doc_for_index = {
                    'content': doc['content'],
                    'source': doc['filename'],
                    'document_type': doc['document_type'],
                    'metadata': doc.get('metadata', {}),
                    'upload_time': doc.get('upload_time', ''),
                    'file_size': doc.get('file_size', 0)
                }
                documents_for_indexing.append(doc_for_index)
            
            # Index documents
            success = vector_store.add_documents(documents_for_indexing)
            
            if success:
                st.success(f"✅ Successfully indexed {len(documents_for_indexing)} documents!")
                
                # Update stats
                stats = vector_store.get_collection_stats()
                st.info(f"📊 Vector store now contains {stats.get('total_documents', 0)} document chunks")
            else:
                st.error("❌ Failed to index documents. Check logs for details.")
                
        except Exception as e:
            error_msg = f"Error indexing documents: {str(e)}"
            st.error(f"❌ {error_msg}")
            add_error_message(error_msg)
            logging.error(f"Indexing error: {e}", exc_info=True)
        
        finally:
            st.session_state.processing_status = "idle"

def reindex_documents():
    """Re-index all documents (clear and re-add)"""
    st.warning("⚠️ This will clear existing index and re-index all documents.")
    
    if st.button("⚠️ Confirm Re-indexing", type="secondary"):
        # For now, just call index_documents_for_search
        # TODO: Implement actual clearing of vector store
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
        
        # Advanced options
        col1, col2 = st.columns(2)
        
        with col1:
            use_concept_validation = st.checkbox(
                "Use Concept Validation",
                value=True,
                help="Validate matches using BERT concept overlap",
                key="use_concept_validation"
            )
            
            include_low_confidence = st.checkbox(
                "Include Low Confidence Matches",
                value=False,
                help="Include matches below similarity threshold",
                key="include_low_confidence"
            )
        
        with col2:
            filter_response_type = st.checkbox(
                "Filter Response Documents",
                value=True,
                help="Only search in documents identified as responses",
                key="filter_response_type"
            )
            
            enable_query_expansion = st.checkbox(
                "Enable Query Expansion",
                value=False,
                help="Expand search queries with related terms",
                key="enable_query_expansion"
            )

def render_recommendation_selection():
    """Render recommendation selection interface"""
    st.subheader("📋 Select Recommendations to Match")
    
    recommendations = st.session_state.extracted_recommendations
    
    # Selection mode
    selection_mode = st.radio(
        "Selection Mode:",
        ["Single Recommendation", "Multiple Recommendations", "All Recommendations"],
        key="matching_selection_mode"
    )
    
    if selection_mode == "Single Recommendation":
        # Single recommendation selection
        rec_options = [f"{rec.id}: {rec.text[:60]}..." for rec in recommendations]
        selected_rec_index = st.selectbox(
            "Select recommendation:",
            range(len(rec_options)),
            format_func=lambda x: rec_options[x],
            key="single_rec_selection"
        )
        
        if st.button("🔍 Find Responses", type="primary", use_container_width=True):
            find_responses_for_recommendation(selected_rec_index)
    
    elif selection_mode == "Multiple Recommendations":
        # Multiple recommendation selection
        rec_options = [f"{rec.id}: {rec.text[:60]}..." for rec in recommendations]
        selected_indices = st.multiselect(
            "Select recommendations:",
            range(len(rec_options)),
            format_func=lambda x: rec_options[x],
            key="multi_rec_selection"
        )
        
        if selected_indices:
            if st.button("🔍 Find Responses for Selected", type="primary", use_container_width=True):
                find_responses_for_multiple_recommendations(selected_indices)
        else:
            st.info("Please select recommendations to match.")
    
    elif selection_mode == "All Recommendations":
        # All recommendations
        st.info(f"Ready to match all {len(recommendations)} recommendations")
        
        if st.button("🔍 Find Responses for All", type="primary", use_container_width=True):
            find_responses_for_all_recommendations()

def render_batch_matching_options():
    """Render batch matching options"""
    with st.expander("🔄 Batch Processing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input(
                "Batch Size",
                1, 20, 5,
                help="Number of recommendations to process at once",
                key="batch_matching_size"
            )
            
            parallel_processing = st.checkbox(
                "Parallel Processing",
                value=False,
                help="Process multiple recommendations simultaneously",
                key="parallel_matching"
            )
        
        with col2:
            save_intermediate = st.checkbox(
                "Save Intermediate Results",
                value=True,
                help="Save results after each batch",
                key="save_intermediate_results"
            )
            
            stop_on_error = st.checkbox(
                "Stop on Error",
                value=False,
                help="Stop processing if an error occurs",
                key="stop_on_error"
            )

def find_responses_for_recommendation(rec_index: int):
    """Find responses for a single recommendation"""
    recommendations = st.session_state.extracted_recommendations
    
    if rec_index >= len(recommendations):
        st.error("Invalid recommendation selected.")
        return
    
    recommendation = recommendations[rec_index]
    matcher = st.session_state.recommendation_matcher
    
    if not matcher:
        st.error("Recommendation matcher not initialized.")
        return
    
    with st.spinner(f"🔍 Finding responses for {recommendation.id}..."):
        try:
            responses = matcher.match_recommendation_to_responses(recommendation)
            
            # Store results
            if rec_index not in st.session_state.matching_results:
                st.session_state.matching_results[rec_index] = {}
            
            st.session_state.matching_results[rec_index] = {
                'recommendation': recommendation,
                'responses': responses,
                'search_time': datetime.now().isoformat(),
                'configuration': {
                    'similarity_threshold': st.session_state.get('matching_similarity_threshold', 0.7),
                    'max_responses': st.session_state.get('max_responses_per_rec', 10),
                    'concept_weight': st.session_state.get('concept_weight', 0.3)
                }
            }
            
            if responses:
                st.success(f"✅ Found {len(responses)} potential responses for {recommendation.id}!")
            else:
                st.warning(f"⚠️ No matching responses found for {recommendation.id}")
            
            # Show quick preview
            if responses:
                show_quick_response_preview(recommendation, responses[:3])
                
        except Exception as e:
            error_msg = f"Error finding responses for {recommendation.id}: {str(e)}"
            st.error(f"❌ {error_msg}")
            add_error_message(error_msg)
            logging.error(f"Response matching error: {e}", exc_info=True)

def find_responses_for_multiple_recommendations(selected_indices: List[int]):
    """Find responses for multiple selected recommendations"""
    recommendations = st.session_state.extracted_recommendations
    matcher = st.session_state.recommendation_matcher
    
    if not matcher:
        st.error("Recommendation matcher not initialized.")
        return
    
    # Filter valid indices
    valid_indices = [i for i in selected_indices if i < len(recommendations)]
    
    if not valid_indices:
        st.error("No valid recommendations selected.")
        return
    
    # Progress tracking
    total_recs = len(valid_indices)
    progress_container = st.container()
    status_container = st.container()
    
    successful_matches = 0
    failed_matches = []
    
    # Set processing status
    st.session_state.processing_status = "matching"
    
    try:
        for i, rec_index in enumerate(valid_indices):
            recommendation = recommendations[rec_index]
            current_step = i + 1
            
            # Update progress
            with progress_container:
                show_progress_indicator(current_step, total_recs, f"Matching {recommendation.id}")
            
            with status_container:
                status_text = st.empty()
                status_text.info(f"🔍 Processing: {recommendation.id}")
            
            try:
                responses = matcher.match_recommendation_to_responses(recommendation)
                
                # Store results
                st.session_state.matching_results[rec_index] = {
                    'recommendation': recommendation,
                    'responses': responses,
                    'search_time': datetime.now().isoformat(),
                    'configuration': {
                        'similarity_threshold': st.session_state.get('matching_similarity_threshold', 0.7),
                        'max_responses': st.session_state.get('max_responses_per_rec', 10),
                        'concept_weight': st.session_state.get('concept_weight', 0.3)
                    }
                }
                
                successful_matches += 1
                status_text.success(f"✅ Found {len(responses)} responses for {recommendation.id}")
                
            except Exception as e:
                error_msg = f"Error matching {recommendation.id}: {str(e)}"
                failed_matches.append(error_msg)
                add_error_message(error_msg)
                status_text.error(f"❌ Failed: {recommendation.id}")
                logging.error(f"Matching error: {e}", exc_info=True)
        
        # Clear progress displays
        progress_container.empty()
        status_container.empty()
        
        # Show results summary
        show_batch_matching_summary(successful_matches, failed_matches, total_recs)
        
    finally:
        st.session_state.processing_status = "idle"

def find_responses_for_all_recommendations():
    """Find responses for all recommendations"""
    recommendations = st.session_state.extracted_recommendations
    all_indices = list(range(len(recommendations)))
    find_responses_for_multiple_recommendations(all_indices)

def show_quick_response_preview(recommendation, responses: List[Dict]):
    """Show a quick preview of found responses"""
    st.subheader("🔍 Response Preview")
    
    with st.expander(f"Preview for {recommendation.id}", expanded=True):
        st.markdown(f"**Recommendation:** {recommendation.text[:200]}...")
        
        st.markdown(f"**Top {len(responses)} Responses:**")
        
        for i, response in enumerate(responses, 1):
            confidence = response.get('combined_confidence', response.get('similarity_score', 0))
            match_type = response.get('match_type', 'UNKNOWN')
            
            # Color coding
            if confidence >= 0.8:
                color = "🟢"
            elif confidence >= 0.6:
                color = "🟡"
            else:
                color = "🔴"
            
            st.markdown(f"**{i}. {color} {match_type} (confidence: {confidence:.2f})**")
            st.markdown(f"*Source:* {response.get('source', 'Unknown')}")
            response_text = response.get('text', '')
            preview_text = response_text[:150] + "..." if len(response_text) > 150 else response_text
            st.markdown(f"*Text:* {preview_text}")
            
            # Show concept overlap if available
            concept_overlap = response.get('concept_overlap', {})
            shared_themes = concept_overlap.get('shared_themes', [])
            if shared_themes:
                st.markdown(f"*Shared Themes:* {', '.join(shared_themes[:3])}")
            
            st.markdown("---")

def show_batch_matching_summary(successful: int, failed: List[str], total: int):
    """Show summary of batch matching results"""
    if successful > 0:
        st.success(f"🎉 Successfully matched {successful} of {total} recommendations!")
        
        # Quick stats
        total_responses_found = 0
        for result in st.session_state.matching_results.values():
            total_responses_found += len(result.get('responses', []))
        
        st.info(f"📊 Total responses found: {total_responses_found}")
    
    if failed:
        st.error(f"❌ Failed to match {len(failed)} recommendations:")
        for error in failed[:5]:  # Show first 5 errors
            st.write(f"• {error}")
        
        if len(failed) > 5:
            st.write(f"... and {len(failed) - 5} more errors")

def display_matching_results():
    """Display matching results with detailed analysis"""
    results = st.session_state.get('matching_results', {})
    
    if not results:
        st.info("💡 No matching results yet. Select recommendations above and click 'Find Responses' to begin.")
        return
    
    st.subheader("🎯 Matching Results")
    
    # Results overview
    display_matching_overview(results)
    
    # Detailed results
    display_detailed_matching_results(results)
    
    # Export options
    render_matching_export_options(results)

def display_matching_overview(results: Dict):
    """Display overview of matching results"""
    # Calculate statistics
    total_recommendations = len(results)
    total_responses = sum(len(result.get('responses', [])) for result in results.values())
    
    # Confidence distribution
    all_confidences = []
    match_type_counts = {}
    
    for result in results.values():
        for response in result.get('responses', []):
            confidence = response.get('combined_confidence', response.get('similarity_score', 0))
            all_confidences.append(confidence)
            
            match_type = response.get('match_type', 'UNKNOWN')
            match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recommendations Matched", total_recommendations)
    
    with col2:
        st.metric("Total Responses Found", total_responses)
    
    with col3:
        avg_responses = total_responses / total_recommendations if total_recommendations > 0 else 0
        st.metric("Avg Responses per Rec", f"{avg_responses:.1f}")
    
    with col4:
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Charts
    if all_confidences:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Confidence Distribution")
            confidence_df = pd.DataFrame({'Confidence': all_confidences})
            st.histogram_chart(confidence_df, x='Confidence')
        
        with col2:
            st.subheader("🏷️ Match Types")
            if match_type_counts:
                match_df = pd.DataFrame(list(match_type_counts.items()), columns=['Match Type', 'Count'])
                st.bar_chart(match_df.set_index('Match Type'))

def display_detailed_matching_results(results: Dict):
    """Display detailed matching results"""
    st.subheader("📋 Detailed Results")
    
    # Filter and sort options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_filter = st.slider(
            "Min Confidence:",
            0.0, 1.0, 0.0, 0.05,
            key="results_confidence_filter"
        )
    
    with col2:
        match_type_filter = st.selectbox(
            "Filter by Match Type:",
            options=['All'] + list(set(
                response.get('match_type', 'UNKNOWN')
                for result in results.values()
                for response in result.get('responses', [])
            )),
            key="results_match_type_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["Confidence (High to Low)", "Recommendation ID", "Response Count"],
            key="results_sort_by"
        )
    
    # Process and filter results
    display_items = []
    
    for rec_index, result in results.items():
        recommendation = result['recommendation']
        responses = result.get('responses', [])
        
        # Filter responses
        filtered_responses = []
        for response in responses:
            confidence = response.get('combined_confidence', response.get('similarity_score', 0))
            match_type = response.get('match_type', 'UNKNOWN')
            
            if confidence >= confidence_filter:
                if match_type_filter == 'All' or match_type == match_type_filter:
                    filtered_responses.append(response)
        
        if filtered_responses:  # Only include if has matching responses
            display_items.append({
                'rec_index': rec_index,
                'recommendation': recommendation,
                'responses': filtered_responses,
                'result': result
            })
    
    # Apply sorting
    if sort_by == "Confidence (High to Low)":
        display_items.sort(key=lambda x: max(
            (r.get('combined_confidence', r.get('similarity_score', 0)) for r in x['responses']), 
            default=0
        ), reverse=True)
    elif sort_by == "Recommendation ID":
        display_items.sort(key=lambda x: x['recommendation'].id)
    elif sort_by == "Response Count":
        display_items.sort(key=lambda x: len(x['responses']), reverse=True)
    
    # Display results
    if not display_items:
        st.warning("No results match the current filters.")
        return
    
    st.write(f"Showing {len(display_items)} recommendations with matching responses")
    
    # Paginated display
    items_per_page = 5
    total_pages = (len(display_items) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.selectbox(
            "Page:",
            range(1, total_pages + 1),
            key="results_page"
        )
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_items = display_items[start_idx:end_idx]
    else:
        page_items = display_items
    
    # Display items
    for item in page_items:
        display_single_matching_result(item)

def display_single_matching_result(item: Dict):
    """Display a single matching result"""
    recommendation = item['recommendation']
    responses = item['responses']
    
    # Summary line
    best_confidence = max(
        r.get('combined_confidence', r.get('similarity_score', 0)) 
        for r in responses
    ) if responses else 0
    
    confidence_color = "🟢" if best_confidence >= 0.8 else "🟡" if best_confidence >= 0.6 else "🔴"
    
    with st.expander(f"{confidence_color} {recommendation.id} - {len(responses)} responses (best: {best_confidence:.2f})"):
        # Recommendation details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Original Recommendation:**")
            st.write(recommendation.text)
        
        with col2:
            st.markdown("**Recommendation Details:**")
            st.write(f"**ID:** {recommendation.id}")
            st.write(f"**Source:** {recommendation.document_source}")
            st.write(f"**Section:** {recommendation.section_title}")
            st.write(f"**Confidence:** {recommendation.confidence_score:.2f}")
        
        # Responses
        st.markdown("**Found Responses:**")
        
        for i, response in enumerate(responses, 1):
            confidence = response.get('combined_confidence', response.get('similarity_score', 0))
            match_type = response.get('match_type', 'UNKNOWN')
            
            # Color coding
            if confidence >= 0.8:
                resp_color = "🟢"
            elif confidence >= 0.6:
                resp_color = "🟡"
            else:
                resp_color = "🔴"
            
            with st.container():
                st.markdown(f"**{i}. {resp_color} Response - {match_type} (confidence: {confidence:.2f})**")
                
                resp_col1, resp_col2 = st.columns([2, 1])
                
                with resp_col1:
                    st.write(f"**Source:** {response.get('source', 'Unknown')}")
                    response_text = response.get('text', 'No text available')
                    st.write(f"**Text:** {response_text}")
                
                with resp_col2:
                    st.write(f"**Similarity:** {response.get('similarity_score', 0):.3f}")
                    st.write(f"**Combined:** {confidence:.3f}")
                    
                    # Concept overlap info
                    concept_overlap = response.get('concept_overlap', {})
                    if concept_overlap:
                        overlap_score = concept_overlap.get('overlap_score', 0)
                        shared_themes = concept_overlap.get('shared_themes', [])
                        
                        st.write(f"**Concept Overlap:** {overlap_score:.3f}")
                        if shared_themes:
                            st.write(f"**Shared Themes:** {len(shared_themes)}")
                            for theme in shared_themes[:3]:  # Show first 3
                                st.write(f"  • {theme}")
                
                st.markdown("---")

def render_matching_export_options(results: Dict):
    """Render export options for matching results"""
    st.subheader("📥 Export Matching Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Summary CSV", use_container_width=True):
            export_matching_summary(results)
    
    with col2:
        if st.button("📋 Export Detailed CSV", use_container_width=True):
            export_detailed_matching_results(results)
    
    with col3:
        if st.button("📄 Export JSON", use_container_width=True):
            export_matching_json(results)

def export_matching_summary(results: Dict):
    """Export matching summary as CSV"""
    if not results:
        st.warning("No matching results to export.")
        return
    
    # Prepare summary data
    summary_data = []
    
    for rec_index, result in results.items():
        recommendation = result['recommendation']
        responses = result.get('responses', [])
        
        # Calculate statistics
        response_count = len(responses)
        avg_confidence = sum(
            r.get('combined_confidence', r.get('similarity_score', 0)) 
            for r in responses
        ) / response_count if response_count > 0 else 0
        
        best_confidence = max(
            r.get('combined_confidence', r.get('similarity_score', 0)) 
            for r in responses
        ) if response_count > 0 else 0
        
        # Match type distribution
        match_types = [r.get('match_type', 'UNKNOWN') for r in responses]
        high_conf_count = len([r for r in responses if r.get('combined_confidence', r.get('similarity_score', 0)) >= 0.8])
        
        summary_data.append({
            'Recommendation_ID': recommendation.id,
            'Recommendation_Text': recommendation.text[:100] + "..." if len(recommendation.text) > 100 else recommendation.text,
            'Source_Document': recommendation.document_source,
            'Response_Count': response_count,
            'Average_Confidence': avg_confidence,
            'Best_Confidence': best_confidence,
            'High_Confidence_Count': high_conf_count,
            'Search_Time': result.get('search_time', ''),
            'Match_Types': ', '.join(set(match_types)) if match_types else 'None'
        })
    
    df = pd.DataFrame(summary_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Summary CSV",
        data=csv,
        file_name=f"matching_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_detailed_matching_results(results: Dict):
    """Export detailed matching results as CSV"""
    if not results:
        st.warning("No matching results to export.")
        return
    
    # Prepare detailed data
    detailed_data = []
    
    for rec_index, result in results.items():
        recommendation = result['recommendation']
        responses = result.get('responses', [])
        
        for response in responses:
            concept_overlap = response.get('concept_overlap', {})
            
            detailed_data.append({
                'Recommendation_ID': recommendation.id,
                'Recommendation_Text': recommendation.text,
                'Recommendation_Source': recommendation.document_source,
                'Response_Source': response.get('source', 'Unknown'),
                'Response_Text': response.get('text', ''),
                'Similarity_Score': response.get('similarity_score', 0),
                'Combined_Confidence': response.get('combined_confidence', response.get('similarity_score', 0)),
                'Match_Type': response.get('match_type', 'UNKNOWN'),
                'Concept_Overlap_Score': concept_overlap.get('overlap_score', 0),
                'Shared_Themes_Count': len(concept_overlap.get('shared_themes', [])),
                'Shared_Themes': ', '.join(concept_overlap.get('shared_themes', [])),
                'Search_Time': result.get('search_time', '')
            })
    
    df = pd.DataFrame(detailed_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Detailed CSV",
        data=csv,
        file_name=f"detailed_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_matching_json(results: Dict):
    """Export matching results as JSON"""
    if not results:
        st.warning("No matching results to export.")
        return
    
    # Prepare JSON data (make it serializable)
    export_data = {}
    
    for rec_index, result in results.items():
        recommendation = result['recommendation']
        
        export_data[str(rec_index)] = {
            'recommendation': {
                'id': recommendation.id,
                'text': recommendation.text,
                'document_source': recommendation.document_source,
                'section_title': recommendation.section_title,
                'confidence_score': recommendation.confidence_score
            },
            'responses': result.get('responses', []),
            'search_time': result.get('search_time', ''),
            'configuration': result.get('configuration', {}),
            'response_count': len(result.get('responses', []))
        }
    
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
    
    st.download_button(
        label="📥 Download JSON",
        data=json_str.encode('utf-8'),
        file_name=f"matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
