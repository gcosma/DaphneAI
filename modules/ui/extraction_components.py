# ===============================================
# FILE: modules/ui/extraction_components.py (ENHANCED VERSION)
# ===============================================
from pathlib import Path
import streamlit as st
import pandas as pd
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import required modules with error handling
try:
    import sys
    sys.path.append('modules')
    from llm_extractor import LLMRecommendationExtractor
    from core_utils import Recommendation, extract_concern_text, extract_metadata
    from .shared_components import add_error_message, show_progress_indicator
except ImportError as e:
    logging.error(f"Import error in extraction_components: {e}")
    # Create mock classes for development
    class LLMRecommendationExtractor:
        def extract_recommendations_and_concerns(self, text, source): 
            return {'recommendations': [], 'concerns': []}
    class Recommendation:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def extract_concern_text(content): return ""
    def extract_metadata(content): return {}

def render_extraction_tab():
    """Render the content extraction tab with enhanced concern extraction"""
    st.header("🔍 Enhanced Content Extraction")
    
    if not st.session_state.uploaded_documents:
        st.warning("⚠️ Please upload documents first in the Upload tab.")
        return
    
    st.markdown("""
    Extract recommendations and **coroner concerns** from uploaded documents using enhanced AI-powered analysis 
    or pattern-based methods.
    
    **🆕 Enhanced Features:**
    - 🎯 **Improved Concern Extraction**: Better pattern matching for coroner documents
    - 📄 **Enhanced PDF Processing**: Handles OCR issues and various document formats
    - 📊 **Confidence Scoring**: Advanced scoring for extraction quality
    - 🔍 **Metadata Extraction**: Automatically extract case refs, dates, and names
    """)
    
    # Extraction configuration
    render_extraction_configuration()
    
    # Document selection and extraction
    render_extraction_interface()
    
    # Enhanced concern extraction section
    render_enhanced_concern_extraction()
    
    # Display results
    display_extraction_results()

def render_extraction_configuration():
    """Render extraction configuration options"""
    st.subheader("⚙️ Extraction Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        extraction_method = st.selectbox(
            "Extraction Method",
            ["AI-Powered (GPT)", "Pattern-Based", "Hybrid (Recommended)", "Enhanced Pattern (Concerns)"],
            index=2,  # Default to Hybrid
            help="Choose how to extract recommendations and concerns",
            key="extraction_method"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.6, 0.05,
            help="Minimum confidence for including extracted items",
            key="confidence_threshold"
        )
    
    with col3:
        max_extractions = st.number_input(
            "Max Extractions per Document",
            1, 100, 50,
            help="Maximum number of items to extract per document",
            key="max_extractions"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_context = st.checkbox(
                "Include Context",
                value=True,
                help="Include surrounding text for better understanding",
                key="include_context"
            )
            
            extract_concerns = st.checkbox(
                "Extract Concerns",
                value=True,
                help="Also extract concerns and issues from documents",
                key="extract_concerns"
            )
            
            use_enhanced_extraction = st.checkbox(
                "Use Enhanced Concern Patterns",
                value=True,
                help="Use improved pattern matching for coroner concerns",
                key="use_enhanced_extraction"
            )
        
        with col2:
            min_text_length = st.number_input(
                "Minimum Text Length",
                10, 500, 50,
                help="Minimum character length for extracted items",
                key="min_text_length"
            )
            
            use_smart_filtering = st.checkbox(
                "Smart Filtering",
                value=True,
                help="Use AI to filter out irrelevant extractions",
                key="use_smart_filtering"
            )
            
            extract_metadata = st.checkbox(
                "Extract Metadata",
                value=True,
                help="Extract case refs, dates, names from documents",
                key="extract_metadata"
            )

def render_extraction_interface():
    """Render the main extraction interface"""
    st.subheader("📄 Document Selection & Extraction")
    
    # Document selection
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_docs = st.multiselect(
            "Select documents to extract from:",
            options=doc_options,
            default=doc_options,  # Select all by default
            help="Choose which documents to process for extraction",
            key="selected_docs_extraction"
        )
    
    with col2:
        st.markdown("**Extraction Info:**")
        st.write(f"• Documents selected: {len(selected_docs)}")
        st.write(f"• Method: {st.session_state.get('extraction_method', 'Hybrid')}")
        st.write(f"• Confidence: {st.session_state.get('confidence_threshold', 0.6):.1f}")
        
        if st.session_state.get('extract_concerns', True):
            st.write("• ✅ Extract concerns")
        else:
            st.write("• ❌ Skip concerns")
        
        if st.session_state.get('use_enhanced_extraction', True):
            st.write("• 🎯 Enhanced patterns")
        
        if st.session_state.get('extract_metadata', True):
            st.write("• 📊 Extract metadata")
    
    # Extraction button
    if selected_docs:
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True):
            extract_content_from_documents(selected_docs)
    else:
        st.info("Please select documents to extract from.")

def render_enhanced_concern_extraction():
    """Render enhanced concern extraction interface"""
    st.markdown("---")
    st.subheader("⚠️ Enhanced Coroner Concern Extraction")
    
    st.markdown("""
    **Specialized extraction for coroner documents** with improved pattern matching and error handling.
    This addresses the PDF extraction failures you've been experiencing.
    """)
    
    # Enhanced extraction options
    with st.expander("🎯 Enhanced Concern Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            enhanced_min_confidence = st.slider(
                "Enhanced Confidence Threshold:", 
                0.0, 1.0, 0.6,
                help="Minimum confidence for enhanced concern extraction",
                key="enhanced_confidence"
            )
            
            extract_enhanced_metadata = st.checkbox(
                "Extract Enhanced Metadata", 
                value=True,
                help="Extract case refs, coroner names, areas, dates",
                key="enhanced_metadata"
            )
        
        with col2:
            show_debug_info = st.checkbox(
                "Show Debug Information",
                value=False,
                help="Show detailed extraction process information",
                key="show_debug"
            )
            
            only_concerns = st.checkbox(
                "Extract Only Concerns",
                value=False,
                help="Skip recommendations, extract only concerns",
                key="only_concerns"
            )
    
    # Enhanced extraction button
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    if uploaded_docs:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎯 Run Enhanced Concern Extraction", type="secondary", use_container_width=True):
                run_enhanced_concern_extraction(uploaded_docs)
        
        with col2:
            st.info(f"📄 {len(uploaded_docs)} documents ready")
    else:
        st.info("Upload documents first to use enhanced extraction")

def run_enhanced_concern_extraction(documents: List[Dict]):
    """Run enhanced concern extraction using improved patterns"""
    
    # Get settings
    min_confidence = st.session_state.get('enhanced_confidence', 0.6)
    extract_metadata_flag = st.session_state.get('enhanced_metadata', True)
    show_debug = st.session_state.get('show_debug', False)
    only_concerns = st.session_state.get('only_concerns', False)
    
    # Initialize results
    extracted_concerns = []
    extraction_debug = []
    
    # Progress tracking
    progress_container = st.container()
    status_container = st.container()
    
    total_docs = len(documents)
    
    try:
        for i, doc in enumerate(documents):
            current_step = i + 1
            doc_name = doc['filename']
            
            # Update progress
            with progress_container:
                show_progress_indicator(current_step, total_docs, f"Enhanced extraction: {doc_name}")
            
            with status_container:
                status_text = st.empty()
                status_text.info(f"🎯 Processing: {doc_name}")
            
            try:
                content = doc.get('content', '')
                if not content:
                    status_text.warning(f"⚠️ No content in {doc_name}")
                    continue
                
                # Use enhanced concern extraction
                concern_text = extract_concern_text(content)
                
                debug_info = {
                    'document': doc_name,
                    'content_length': len(content),
                    'concern_found': bool(concern_text),
                    'concern_length': len(concern_text) if concern_text else 0
                }
                
                if concern_text and len(concern_text) > 20:  # Minimum meaningful length
                    # Calculate confidence (simplified version)
                    confidence = calculate_enhanced_confidence(concern_text)
                    
                    if confidence >= min_confidence:
                        concern_data = {
                            'id': f"enhanced_concern_{current_step}",
                            'text': concern_text,
                            'document_source': doc_name,
                            'confidence_score': confidence,
                            'extraction_method': 'enhanced_pattern',
                            'timestamp': datetime.now().isoformat(),
                            'type': 'concern'
                        }
                        
                        # Add metadata if requested
                        if extract_metadata_flag:
                            metadata = extract_metadata(content)
                            concern_data['metadata'] = metadata
                            debug_info['metadata_extracted'] = len(metadata)
                        
                        extracted_concerns.append(concern_data)
                        debug_info['extracted'] = True
                        status_text.success(f"✅ Extracted concern from {doc_name} (Confidence: {confidence:.2f})")
                    else:
                        debug_info['extracted'] = False
                        debug_info['reason'] = f"Low confidence: {confidence:.2f}"
                        status_text.warning(f"⚠️ Low confidence extraction from {doc_name}")
                else:
                    debug_info['extracted'] = False
                    debug_info['reason'] = "No concern text found or too short"
                    status_text.warning(f"⚠️ No concerns found in {doc_name}")
                
                extraction_debug.append(debug_info)
                
            except Exception as e:
                error_msg = f"Enhanced extraction error for {doc_name}: {str(e)}"
                add_error_message(error_msg)
                status_text.error(f"❌ Error processing {doc_name}")
                logging.error(f"Enhanced extraction error: {e}", exc_info=True)
                
                extraction_debug.append({
                    'document': doc_name,
                    'extracted': False,
                    'error': str(e)
                })
        
        # Update session state
        if only_concerns:
            st.session_state.extracted_concerns = extracted_concerns
        else:
            # Merge with existing concerns
            existing_concerns = st.session_state.get('extracted_concerns', [])
            st.session_state.extracted_concerns = existing_concerns + extracted_concerns
        
        st.session_state.enhanced_extraction_debug = extraction_debug
        
        # Clear progress displays
        progress_container.empty()
        status_container.empty()
        
        # Show results
        show_enhanced_extraction_results(extracted_concerns, extraction_debug, show_debug)
        
    except Exception as e:
        st.error(f"Enhanced extraction failed: {str(e)}")
        logging.error(f"Enhanced extraction error: {e}", exc_info=True)

def calculate_enhanced_confidence(text: str) -> float:
    """Calculate confidence score for enhanced extraction"""
    base_confidence = 0.7
    
    # Length indicators
    if len(text) > 100:
        base_confidence += 0.1
    if len(text) > 300:
        base_confidence += 0.1
    
    # Structure indicators
    if re.search(r'\d+\.', text):  # Numbered points
        base_confidence += 0.05
    if re.search(r'[A-Z][a-z]+:', text):  # Section headers
        base_confidence += 0.05
    if re.search(r'(?:should|must|ought to)', text, re.IGNORECASE):
        base_confidence += 0.05
    if re.search(r'(?:concern|issue|problem|risk)', text, re.IGNORECASE):
        base_confidence += 0.05
    
    return min(base_confidence, 1.0)

def show_enhanced_extraction_results(concerns: List[Dict], debug_info: List[Dict], show_debug: bool):
    """Show results of enhanced extraction"""
    
    if concerns:
        st.success(f"🎯 Enhanced extraction completed! Found {len(concerns)} concerns")
        
        # Quick metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Concerns Extracted", len(concerns))
        
        with col2:
            avg_confidence = sum(c['confidence_score'] for c in concerns) / len(concerns)
            st.metric("Average Confidence", f"{avg_confidence:.2f}")
        
        with col3:
            with_metadata = len([c for c in concerns if c.get('metadata')])
            st.metric("With Metadata", f"{with_metadata}/{len(concerns)}")
        
        # Preview concerns
        with st.expander("📋 Preview Extracted Concerns", expanded=True):
            for i, concern in enumerate(concerns[:3]):  # Show first 3
                st.write(f"**Concern {i+1}** ({concern['document_source']}):")
                preview_text = concern['text'][:200] + "..." if len(concern['text']) > 200 else concern['text']
                st.write(preview_text)
                st.write(f"*Confidence: {concern['confidence_score']:.2f}*")
                
                if concern.get('metadata'):
                    metadata_preview = {k: v for k, v in list(concern['metadata'].items())[:3]}
                    st.write(f"*Metadata: {metadata_preview}*")
                
                st.write("---")
            
            if len(concerns) > 3:
                st.info(f"... and {len(concerns) - 3} more concerns")
    else:
        st.warning("⚠️ No concerns extracted. Try adjusting the confidence threshold or check document content.")
    
    # Show debug information if requested
    if show_debug and debug_info:
        with st.expander("🔍 Debug Information"):
            debug_df = pd.DataFrame(debug_info)
            st.dataframe(debug_df, use_container_width=True)
            
            # Summary stats
            total_docs = len(debug_info)
            successful = len([d for d in debug_info if d.get('extracted')])
            
            st.write(f"**Summary:** {successful}/{total_docs} documents successfully processed")
            
            # Show failed extractions
            failed = [d for d in debug_info if not d.get('extracted')]
            if failed:
                st.write("**Failed extractions:**")
                for fail in failed:
                    reason = fail.get('reason', fail.get('error', 'Unknown error'))
                    st.write(f"- {fail['document']}: {reason}")

def extract_content_from_documents(selected_docs: List[str]):
    """Extract recommendations and concerns from selected documents (ENHANCED VERSION)"""
    if not selected_docs:
        st.warning("No documents selected for extraction.")
        return
    
    # Get configuration
    extraction_method = st.session_state.get('extraction_method', 'Hybrid (Recommended)')
    confidence_threshold = st.session_state.get('confidence_threshold', 0.6)
    max_extractions = st.session_state.get('max_extractions', 50)
    extract_concerns = st.session_state.get('extract_concerns', True)
    min_text_length = st.session_state.get('min_text_length', 50)
    use_enhanced_extraction = st.session_state.get('use_enhanced_extraction', True)
    
    # Initialize extractor
    extractor = LLMRecommendationExtractor()
    
    # Progress tracking
    total_docs = len(selected_docs)
    progress_container = st.container()
    status_container = st.container()
    results_container = st.container()
    
    all_recommendations = []
    all_concerns = []
    processing_results = []
    
    # Set processing status
    st.session_state.processing_status = "extracting"
    
    try:
        for i, doc_name in enumerate(selected_docs):
            current_step = i + 1
            
            # Update progress
            with progress_container:
                show_progress_indicator(current_step, total_docs, f"Extracting from {doc_name}")
            
            with status_container:
                status_text = st.empty()
                status_text.info(f"🔍 Processing: {doc_name}")
            
            try:
                # Find document
                doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
                if not doc:
                    status_text.error(f"❌ Document not found: {doc_name}")
                    continue
                
                # Enhanced extraction for concerns if enabled
                enhanced_concerns = []
                if extract_concerns and use_enhanced_extraction:
                    try:
                        concern_text = extract_concern_text(doc['content'])
                        if concern_text and len(concern_text) >= min_text_length:
                            confidence = calculate_enhanced_confidence(concern_text)
                            if confidence >= confidence_threshold:
                                enhanced_concern = {
                                    'id': f"enhanced_{doc_name}_{i}",
                                    'text': concern_text,
                                    'document_source': doc_name,
                                    'confidence_score': confidence,
                                    'extraction_method': 'enhanced_pattern',
                                    'category': 'coroner_concern',
                                    'type': 'concern'
                                }
                                
                                # Add metadata if extraction is enabled
                                if st.session_state.get('extract_metadata', True):
                                    metadata = extract_metadata(doc['content'])
                                    enhanced_concern['metadata'] = metadata
                                
                                enhanced_concerns.append(enhanced_concern)
                    except Exception as e:
                        logging.warning(f"Enhanced extraction failed for {doc_name}: {e}")
                
                # Standard extraction
                extraction_result = extractor.extract_recommendations_and_concerns(
                    doc['content'], 
                    doc['filename']
                )
                
                recommendations = extraction_result.get('recommendations', [])
                standard_concerns = extraction_result.get('concerns', [])
                
                # Apply filtering to recommendations
                filtered_recommendations = []
                for rec in recommendations:
                    if (rec.confidence_score >= confidence_threshold and 
                        len(rec.text) >= min_text_length):
                        filtered_recommendations.append(rec)
                
                # Apply filtering to standard concerns
                filtered_concerns = []
                if extract_concerns:
                    for concern in standard_concerns:
                        if (concern.get('confidence_score', 0) >= confidence_threshold and 
                            len(concern.get('text', '')) >= min_text_length):
                            filtered_concerns.append(concern)
                
                # Combine enhanced and standard concerns
                all_doc_concerns = enhanced_concerns + filtered_concerns
                
                # Limit extractions
                filtered_recommendations = filtered_recommendations[:max_extractions]
                all_doc_concerns = all_doc_concerns[:max_extractions]
                
                # Store results
                all_recommendations.extend(filtered_recommendations)
                all_concerns.extend(all_doc_concerns)
                
                processing_results.append({
                    'document': doc_name,
                    'recommendations_found': len(filtered_recommendations),
                    'concerns_found': len(all_doc_concerns),
                    'enhanced_concerns': len(enhanced_concerns),
                    'standard_concerns': len(filtered_concerns),
                    'status': 'success'
                })
                
                concern_summary = f"{len(all_doc_concerns)} concerns ({len(enhanced_concerns)} enhanced + {len(filtered_concerns)} standard)"
                status_text.success(f"✅ Extracted {len(filtered_recommendations)} recommendations, {concern_summary} from {doc_name}")
                
            except Exception as e:
                error_msg = f"Error extracting from {doc_name}: {str(e)}"
                add_error_message(error_msg)
                processing_results.append({
                    'document': doc_name,
                    'recommendations_found': 0,
                    'concerns_found': 0,
                    'enhanced_concerns': 0,
                    'standard_concerns': 0,
                    'status': 'error',
                    'error': str(e)
                })
                status_text.error(f"❌ Failed: {doc_name}")
                logging.error(f"Extraction error: {e}", exc_info=True)
        
        # Update session state
        st.session_state.extracted_recommendations = all_recommendations
        st.session_state.extracted_concerns = all_concerns
        st.session_state.last_extraction_results = processing_results
        st.session_state.last_processing_time = datetime.now().isoformat()
        
        # Clear progress displays
        progress_container.empty()
        status_container.empty()
        
        # Show final results
        with results_container:
            show_extraction_summary(processing_results, all_recommendations, all_concerns)
    
    finally:
        st.session_state.processing_status = "idle"

def show_extraction_summary(processing_results: List[Dict], recommendations: List, concerns: List):
    """Show summary of extraction results with enhanced metrics"""
    st.success("🎉 Enhanced Extraction Complete!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        successful_docs = len([r for r in processing_results if r['status'] == 'success'])
        st.metric("Documents Processed", f"{successful_docs}/{len(processing_results)}")
    
    with col2:
        total_recommendations = len(recommendations)
        st.metric("Recommendations Found", total_recommendations)
    
    with col3:
        total_concerns = len(concerns)
        enhanced_concerns = sum(r.get('enhanced_concerns', 0) for r in processing_results)
        st.metric("Concerns Found", f"{total_concerns} ({enhanced_concerns} enhanced)")
    
    with col4:
        avg_per_doc = (total_recommendations + total_concerns) / len(processing_results) if processing_results else 0
        st.metric("Avg per Document", f"{avg_per_doc:.1f}")
    
    # Enhanced results table
    if processing_results:
        st.subheader("📊 Enhanced Processing Details")
        
        results_df = pd.DataFrame(processing_results)
        
        # Rename columns for display
        display_columns = {
            'document': 'Document',
            'recommendations_found': 'Recommendations',
            'concerns_found': 'Total Concerns',
            'enhanced_concerns': 'Enhanced Concerns',
            'standard_concerns': 'Standard Concerns',
            'status': 'Status'
        }
        
        display_df = results_df.rename(columns=display_columns)
        
        # Add status icons
        if 'Status' in display_df.columns:
            display_df['Status'] = display_df['Status'].map({
                'success': '✅ Success',
                'error': '❌ Error'
            })
        
        # Select columns to display
        cols_to_show = ['Document', 'Recommendations', 'Total Concerns', 'Enhanced Concerns', 'Status']
        available_cols = [col for col in cols_to_show if col in display_df.columns]
        
        st.dataframe(display_df[available_cols], use_container_width=True)
    
    # Show any errors
    failed_docs = [r for r in processing_results if r['status'] == 'error']
    if failed_docs:
        with st.expander("⚠️ Processing Errors"):
            for result in failed_docs:
                st.error(f"**{result['document']}:** {result.get('error', 'Unknown error')}")

# Keep all existing functions unchanged
def display_extraction_results():
    """Display extracted recommendations and concerns"""
    if not st.session_state.extracted_recommendations and not st.session_state.extracted_concerns:
        st.info("💡 No extractions yet. Configure settings above and click 'Start Extraction' to begin.")
        return
    
    # Results tabs
    rec_tab, concern_tab, combined_tab = st.tabs(["📋 Recommendations", "⚠️ Concerns", "📊 Combined View"])
    
    with rec_tab:
        display_extracted_recommendations()
    
    with concern_tab:
        display_extracted_concerns()
    
    with combined_tab:
        display_combined_results()

def display_extracted_recommendations():
    """Display extracted recommendations with interactive features"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    
    if not recommendations:
        st.info("No recommendations extracted yet.")
        return
    
    st.subheader(f"📋 Extracted Recommendations ({len(recommendations)})")
    
    # Filtering and sorting
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_filter = st.selectbox(
            "Filter by Source:",
            options=['All'] + sorted(list(set(rec.document_source for rec in recommendations))),
            key="rec_source_filter"
        )
    
    with col2:
        confidence_filter = st.slider(
            "Min Confidence:",
            0.0, 1.0, 0.0, 0.05,
            key="rec_confidence_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["Confidence (High to Low)", "Confidence (Low to High)", "Document Source", "Text Length"],
            key="rec_sort_by"
        )
    
    # Apply filters
    filtered_recs = recommendations
    
    if source_filter != 'All':
        filtered_recs = [rec for rec in filtered_recs if rec.document_source == source_filter]
    
    filtered_recs = [rec for rec in filtered_recs if rec.confidence_score >= confidence_filter]
    
    # Apply sorting
    if sort_by == "Confidence (High to Low)":
        filtered_recs.sort(key=lambda x: x.confidence_score, reverse=True)
    elif sort_by == "Confidence (Low to High)":
        filtered_recs.sort(key=lambda x: x.confidence_score)
    elif sort_by == "Document Source":
        filtered_recs.sort(key=lambda x: x.document_source)
    elif sort_by == "Text Length":
        filtered_recs.sort(key=lambda x: len(x.text), reverse=True)
    
    if not filtered_recs:
        st.warning("No recommendations match the current filters.")
        return
    
    # Display recommendations
    st.write(f"Showing {len(filtered_recs)} of {len(recommendations)} recommendations")
    
    # Create summary table
    rec_data = []
    for i, rec in enumerate(filtered_recs):
        rec_data.append({
            "Index": i,
            "ID": rec.id,
            "Preview": rec.text[:100] + "..." if len(rec.text) > 100 else rec.text,
            "Source": rec.document_source,
            "Section": rec.section_title,
            "Confidence": f"{rec.confidence_score:.2f}",
            "Length": len(rec.text)
        })
    
    df = pd.DataFrame(rec_data)
    
    # Display with selection capability
    selected_indices = st.dataframe(
        df.drop('Index', axis=1),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row"
    )
    
    # Show detailed view for selected recommendations
    if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
        st.subheader("📖 Detailed View")
        
        for idx in selected_indices.selection.rows:
            if idx < len(filtered_recs):
                rec = filtered_recs[idx]
                display_recommendation_detail(rec)

def display_recommendation_detail(rec):
    """Display detailed view of a single recommendation"""
    with st.expander(f"📋 Recommendation {rec.id}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Full Text:**")
            st.write(rec.text)
        
        with col2:
            st.markdown("**Details:**")
            st.write(f"**ID:** {rec.id}")
            st.write(f"**Source:** {rec.document_source}")
            st.write(f"**Section:** {rec.section_title}")
            st.write(f"**Page:** {rec.page_number or 'N/A'}")
            st.write(f"**Confidence:** {rec.confidence_score:.2f}")
            st.write(f"**Length:** {len(rec.text)} characters")
            
            # Show metadata if available
            if hasattr(rec, 'metadata') and rec.metadata:
                st.markdown("**Metadata:**")
                for key, value in rec.metadata.items():
                    st.write(f"• **{key}:** {value}")

def display_extracted_concerns():
    """Display extracted concerns with enhanced features"""
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not concerns:
        st.info("No concerns extracted yet.")
        return
    
    st.subheader(f"⚠️ Extracted Concerns ({len(concerns)})")
    
    # Enhanced filtering options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by extraction method
        extraction_methods = list(set(concern.get('extraction_method', 'Unknown') for concern in concerns))
        method_filter = st.selectbox(
            "Filter by Method:",
            options=['All'] + extraction_methods,
            key="concern_method_filter"
        )
    
    with col2:
        confidence_filter = st.slider(
            "Min Confidence:",
            0.0, 1.0, 0.0, 0.05,
            key="concern_confidence_filter"
        )
    
    with col3:
        # Filter by document source
        sources = list(set(concern.get('document_source', 'Unknown') for concern in concerns))
        source_filter = st.selectbox(
            "Filter by Source:",
            options=['All'] + sorted(sources),
            key="concern_source_filter"
        )
    
    # Apply filters
    filtered_concerns = concerns
    
    if method_filter != 'All':
        filtered_concerns = [c for c in filtered_concerns if c.get('extraction_method', 'Unknown') == method_filter]
    
    if source_filter != 'All':
        filtered_concerns = [c for c in filtered_concerns if c.get('document_source', 'Unknown') == source_filter]
    
    filtered_concerns = [c for c in filtered_concerns if c.get('confidence_score', 0) >= confidence_filter]
    
    if not filtered_concerns:
        st.warning("No concerns match the current filters.")
        return
    
    st.write(f"Showing {len(filtered_concerns)} of {len(concerns)} concerns")
    
    # Enhanced display options
    display_mode = st.radio(
        "Display Mode:",
        ["Table View", "Detailed Cards", "Comparison Mode"],
        horizontal=True,
        key="concern_display_mode"
    )
    
    if display_mode == "Table View":
        display_concerns_table(filtered_concerns)
    elif display_mode == "Detailed Cards":
        display_concerns_cards(filtered_concerns)
    elif display_mode == "Comparison Mode":
        display_concerns_comparison(filtered_concerns)

def display_concerns_table(concerns: List[Dict]):
    """Display concerns in table format"""
    # Convert concerns to DataFrame for display
    concern_data = []
    for i, concern in enumerate(concerns):
        concern_data.append({
            "Index": i + 1,
            "ID": concern.get('id', f"CONCERN-{i+1}"),
            "Preview": concern.get('text', '')[:100] + "..." if len(concern.get('text', '')) > 100 else concern.get('text', ''),
            "Source": concern.get('document_source', 'Unknown'),
            "Method": concern.get('extraction_method', 'Unknown'),
            "Confidence": f"{concern.get('confidence_score', 0):.2f}",
            "Length": len(concern.get('text', '')),
            "Has Metadata": "✅" if concern.get('metadata') else "❌"
        })
    
    df = pd.DataFrame(concern_data)
    
    # Display table with selection
    selected_rows = st.dataframe(
        df.drop('Index', axis=1),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row"
    )
    
    # Show detailed view for selected concerns
    if hasattr(selected_rows, 'selection') and selected_rows.selection.rows:
        st.subheader("📖 Selected Concerns Details")
        
        for idx in selected_rows.selection.rows:
            if idx < len(concerns):
                concern = concerns[idx]
                display_concern_detail(concern)

def display_concerns_cards(concerns: List[Dict]):
    """Display concerns as detailed cards"""
    # Pagination for large numbers of concerns
    items_per_page = 5
    total_pages = (len(concerns) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.selectbox(f"Page (showing {items_per_page} per page):", range(1, total_pages + 1)) - 1
        start_idx = page * items_per_page
        end_idx = min(start_idx + items_per_page, len(concerns))
        page_concerns = concerns[start_idx:end_idx]
    else:
        page_concerns = concerns
    
    for concern in page_concerns:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {concern.get('id', 'Unknown ID')}")
                st.write(f"**Source:** {concern.get('document_source', 'Unknown')}")
                st.write(f"**Method:** {concern.get('extraction_method', 'Unknown')}")
                
                # Show full text with expandable option
                text = concern.get('text', '')
                if len(text) > 300:
                    with st.expander("📄 Full Text"):
                        st.write(text)
                    st.write(text[:300] + "...")
                else:
                    st.write(text)
            
            with col2:
                st.metric("Confidence", f"{concern.get('confidence_score', 0):.2f}")
                st.metric("Length", f"{len(concern.get('text', ''))} chars")
                
                if concern.get('metadata'):
                    with st.expander("📊 Metadata"):
                        for key, value in concern['metadata'].items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            st.markdown("---")

def display_concerns_comparison(concerns: List[Dict]):
    """Display concerns in comparison mode"""
    if len(concerns) < 2:
        st.info("Need at least 2 concerns for comparison mode.")
        return
    
    st.write("**Select concerns to compare side by side:**")
    
    col1, col2 = st.columns(2)
    
    # Create concern options for selection
    concern_options = {}
    for i, concern in enumerate(concerns):
        preview = concern.get('text', '')[:80] + "..." if len(concern.get('text', '')) > 80 else concern.get('text', '')
        source = concern.get('document_source', 'Unknown')
        concern_options[f"[{source}] {preview}"] = i
    
    with col1:
        concern1_label = st.selectbox("Select first concern:", list(concern_options.keys()), key="comp_concern1")
        if concern1_label:
            concern1 = concerns[concern_options[concern1_label]]
            display_single_concern_comparison(concern1, "Concern 1")
    
    with col2:
        concern2_label = st.selectbox("Select second concern:", list(concern_options.keys()), key="comp_concern2")
        if concern2_label:
            concern2 = concerns[concern_options[concern2_label]]
            display_single_concern_comparison(concern2, "Concern 2")
    
    # Show similarity analysis if both selected
    if concern1_label and concern2_label and concern1_label != concern2_label:
        st.markdown("---")
        st.subheader("🔍 Similarity Analysis")
        
        similarity = calculate_text_similarity(
            concern1.get('text', ''), 
            concern2.get('text', '')
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Text Similarity", f"{similarity:.3f}")
        with col2:
            same_source = concern1.get('document_source') == concern2.get('document_source')
            st.metric("Same Source", "✅" if same_source else "❌")
        with col3:
            same_method = concern1.get('extraction_method') == concern2.get('extraction_method')
            st.metric("Same Method", "✅" if same_method else "❌")

def display_single_concern_comparison(concern: Dict, title: str):
    """Display a single concern in comparison format"""
    st.subheader(title)
    st.write(f"**Source:** {concern.get('document_source', 'Unknown')}")
    st.write(f"**Method:** {concern.get('extraction_method', 'Unknown')}")
    st.write(f"**Confidence:** {concern.get('confidence_score', 0):.2f}")
    
    text = concern.get('text', '')
    if len(text) > 200:
        st.text_area("", value=text, height=150, disabled=True, key=f"{title}_text")
    else:
        st.write(text)
    
    if concern.get('metadata'):
        with st.expander("📊 Metadata"):
            for key, value in concern['metadata'].items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using token overlap"""
    if not text1 or not text2:
        return 0.0
    
    # Simple tokenization
    tokens1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower()))
    tokens2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower()))
    
    # Jaccard similarity
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0

def display_concern_detail(concern):
    """Display detailed view of a single concern with enhanced information"""
    with st.expander(f"⚠️ Concern {concern.get('id')}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Full Text:**")
            st.write(concern.get('text', 'No text available'))
            
            # Show extraction details
            st.markdown("**Extraction Details:**")
            st.write(f"• **Method:** {concern.get('extraction_method', 'Unknown')}")
            st.write(f"• **Timestamp:** {concern.get('timestamp', 'Unknown')}")
            
            if concern.get('type'):
                st.write(f"• **Type:** {concern.get('type', 'Unknown')}")
        
        with col2:
            st.markdown("**Metrics:**")
            st.write(f"**ID:** {concern.get('id', 'N/A')}")
            st.write(f"**Source:** {concern.get('document_source', 'Unknown')}")
            st.write(f"**Section:** {concern.get('section', 'Unknown')}")
            st.write(f"**Page:** {concern.get('page_number', 'N/A')}")
            st.write(f"**Confidence:** {concern.get('confidence_score', 0):.2f}")
            st.write(f"**Category:** {concern.get('category', 'General')}")
            st.write(f"**Length:** {len(concern.get('text', ''))} characters")
            
            # Enhanced metadata display
            if concern.get('metadata'):
                st.markdown("**📊 Extracted Metadata:**")
                metadata = concern['metadata']
                for key, value in metadata.items():
                    formatted_key = key.replace('_', ' ').title()
                    st.write(f"• **{formatted_key}:** {value}")

def display_combined_results():
    """Display combined view of recommendations and concerns with enhanced analytics"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not recommendations and not concerns:
        st.info("No extractions available for combined view.")
        return
    
    st.subheader("📊 Enhanced Combined Results Analysis")
    
    # Enhanced summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recommendations", len(recommendations))
    
    with col2:
        total_concerns = len(concerns)
        enhanced_concerns = len([c for c in concerns if c.get('extraction_method') == 'enhanced_pattern'])
        st.metric("Total Concerns", f"{total_concerns} ({enhanced_concerns} enhanced)")
    
    with col3:
        total_items = len(recommendations) + len(concerns)
        st.metric("Total Items", total_items)
    
    with col4:
        if recommendations:
            avg_confidence = sum(rec.confidence_score for rec in recommendations) / len(recommendations)
        elif concerns:
            avg_confidence = sum(c.get('confidence_score', 0) for c in concerns) / len(concerns)
        else:
            avg_confidence = 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Enhanced analytics tabs
    analytics_tabs = st.tabs(["📈 Distribution", "🔍 Quality Analysis", "📊 Method Comparison", "📥 Export"])
    
    with analytics_tabs[0]:
        display_distribution_analysis(recommendations, concerns)
    
    with analytics_tabs[1]:
        display_quality_analysis(recommendations, concerns)
    
    with analytics_tabs[2]:
        display_method_comparison(concerns)
    
    with analytics_tabs[3]:
        display_enhanced_export_options(recommendations, concerns)

def display_distribution_analysis(recommendations: List, concerns: List[Dict]):
    """Display distribution analysis"""
    st.subheader("📈 Distribution Analysis")
    
    # Source distribution
    source_data = {}
    
    for rec in recommendations:
        source = rec.document_source
        if source not in source_data:
            source_data[source] = {'recommendations': 0, 'concerns': 0}
        source_data[source]['recommendations'] += 1
    
    for concern in concerns:
        source = concern.get('document_source', 'Unknown')
        if source not in source_data:
            source_data[source] = {'recommendations': 0, 'concerns': 0}
        source_data[source]['concerns'] += 1
    
    if source_data:
        st.write("**Distribution by Source Document:**")
        chart_data = []
        for source, counts in source_data.items():
            chart_data.append({
                'Source': source,
                'Recommendations': counts['recommendations'],
                'Concerns': counts['concerns']
            })
        
        chart_df = pd.DataFrame(chart_data)
        st.bar_chart(chart_df.set_index('Source'))
        
        # Summary table
        st.dataframe(chart_df, use_container_width=True)

def display_quality_analysis(recommendations: List, concerns: List[Dict]):
    """Display quality analysis"""
    st.subheader("🔍 Quality Analysis")
    
    # Confidence distribution
    if recommendations or concerns:
        st.write("**Confidence Score Distribution:**")
        
        all_confidences = []
        all_confidences.extend([rec.confidence_score for rec in recommendations])
        all_confidences.extend([c.get('confidence_score', 0) for c in concerns])
        
        if all_confidences:
            confidence_df = pd.DataFrame({'Confidence': all_confidences})
            st.histogram_chart(confidence_df['Confidence'])
            
            # Quality metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_quality = len([c for c in all_confidences if c >= 0.8])
                st.metric("High Quality (≥0.8)", f"{high_quality}/{len(all_confidences)}")
            
            with col2:
                medium_quality = len([c for c in all_confidences if 0.6 <= c < 0.8])
                st.metric("Medium Quality (0.6-0.8)", f"{medium_quality}/{len(all_confidences)}")
            
            with col3:
                low_quality = len([c for c in all_confidences if c < 0.6])
                st.metric("Low Quality (<0.6)", f"{low_quality}/{len(all_confidences)}")

def display_method_comparison(concerns: List[Dict]):
    """Display comparison of extraction methods"""
    st.subheader("📊 Extraction Method Comparison")
    
    if not concerns:
        st.info("No concerns available for method comparison.")
        return
    
    # Group by extraction method
    method_stats = {}
    for concern in concerns:
        method = concern.get('extraction_method', 'Unknown')
        if method not in method_stats:
            method_stats[method] = {
                'count': 0,
                'total_confidence': 0,
                'total_length': 0,
                'with_metadata': 0
            }
        
        stats = method_stats[method]
        stats['count'] += 1
        stats['total_confidence'] += concern.get('confidence_score', 0)
        stats['total_length'] += len(concern.get('text', ''))
        if concern.get('metadata'):
            stats['with_metadata'] += 1
    
    # Create comparison table
    comparison_data = []
    for method, stats in method_stats.items():
        comparison_data.append({
            'Method': method,
            'Count': stats['count'],
            'Avg Confidence': f"{stats['total_confidence'] / stats['count']:.2f}",
            'Avg Length': f"{stats['total_length'] / stats['count']:.0f}",
            'With Metadata %': f"{(stats['with_metadata'] / stats['count']) * 100:.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Show method performance
    if len(method_stats) > 1:
        st.write("**Method Performance Analysis:**")
        
        # Find best performing method
        best_method = max(method_stats.items(), key=lambda x: x[1]['total_confidence'] / x[1]['count'])
        st.success(f"🏆 **Best performing method:** {best_method[0]} (Avg confidence: {best_method[1]['total_confidence'] / best_method[1]['count']:.2f})")
        
        # Show enhanced vs standard comparison
        enhanced_stats = method_stats.get('enhanced_pattern')
        standard_stats = method_stats.get('llm')
        
        if enhanced_stats and standard_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Enhanced Pattern Method:**")
                st.write(f"• Count: {enhanced_stats['count']}")
                st.write(f"• Avg Confidence: {enhanced_stats['total_confidence'] / enhanced_stats['count']:.2f}")
                st.write(f"• Metadata Rate: {(enhanced_stats['with_metadata'] / enhanced_stats['count']) * 100:.1f}%")
            
            with col2:
                st.write("**Standard LLM Method:**")
                st.write(f"• Count: {standard_stats['count']}")
                st.write(f"• Avg Confidence: {standard_stats['total_confidence'] / standard_stats['count']:.2f}")
                st.write(f"• Metadata Rate: {(standard_stats['with_metadata'] / standard_stats['count']) * 100:.1f}%")

def display_enhanced_export_options(recommendations: List, concerns: List[Dict]):
    """Display enhanced export options"""
    st.subheader("📥 Enhanced Export Options")
    
    # Export sections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📋 Recommendations**")
        if recommendations:
            if st.button("📄 Export Recommendations", use_container_width=True):
                export_recommendations_csv()
        else:
            st.write("No recommendations to export")
    
    with col2:
        st.write("**⚠️ Concerns**")
        if concerns:
            if st.button("⚠️ Export All Concerns", use_container_width=True):
                export_concerns_csv()
            
            enhanced_concerns = [c for c in concerns if c.get('extraction_method') == 'enhanced_pattern']
            if enhanced_concerns:
                if st.button("🎯 Export Enhanced Only", use_container_width=True):
                    export_enhanced_concerns_csv(enhanced_concerns)
        else:
            st.write("No concerns to export")
    
    with col3:
        st.write("**📊 Combined Data**")
        if recommendations or concerns:
            if st.button("📊 Export Combined", use_container_width=True):
                export_combined_csv()
            
            if st.button("📈 Export Analytics Report", use_container_width=True):
                export_analytics_report(recommendations, concerns)
        else:
            st.write("No data to export")

def export_enhanced_concerns_csv(enhanced_concerns: List[Dict]):
    """Export only enhanced concerns to CSV"""
    if not enhanced_concerns:
        st.warning("No enhanced concerns to export.")
        return
    
    # Prepare data for export
    export_data = []
    for concern in enhanced_concerns:
        row_data = {
            'ID': concern.get('id', ''),
            'Text': concern.get('text', ''),
            'Source_Document': concern.get('document_source', ''),
            'Confidence_Score': concern.get('confidence_score', 0),
            'Extraction_Method': concern.get('extraction_method', 'enhanced_pattern'),
            'Text_Length': len(concern.get('text', '')),
            'Timestamp': concern.get('timestamp', '')
        }
        
        # Add metadata fields
        if concern.get('metadata'):
            for key, value in concern['metadata'].items():
                row_data[f'metadata_{key}'] = value
        
        export_data.append(row_data)
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Enhanced Concerns CSV",
        data=csv,
        file_name=f"enhanced_concerns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_analytics_report(recommendations: List, concerns: List[Dict]):
    """Export comprehensive analytics report"""
    
    # Create analytics summary
    analytics_data = {
        'summary': {
            'total_recommendations': len(recommendations),
            'total_concerns': len(concerns),
            'enhanced_concerns': len([c for c in concerns if c.get('extraction_method') == 'enhanced_pattern']),
            'export_timestamp': datetime.now().isoformat()
        },
        'method_comparison': {},
        'source_distribution': {},
        'quality_metrics': {}
    }
    
    # Method comparison
    method_stats = {}
    for concern in concerns:
        method = concern.get('extraction_method', 'Unknown')
        if method not in method_stats:
            method_stats[method] = {'count': 0, 'avg_confidence': 0, 'confidences': []}
        method_stats[method]['count'] += 1
        method_stats[method]['confidences'].append(concern.get('confidence_score', 0))
    
    for method, stats in method_stats.items():
        if stats['confidences']:
            stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
    
    analytics_data['method_comparison'] = method_stats
    
    # Export as JSON
    json_data = json.dumps(analytics_data, indent=2, default=str)
    
    st.download_button(
        label="📈 Download Analytics Report (JSON)",
        data=json_data,
        file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

# Keep existing export functions for backward compatibility
def export_recommendations_csv():
    """Export recommendations to CSV"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    
    if not recommendations:
        st.warning("No recommendations to export.")
        return
    
    # Prepare data for export
    export_data = []
    for rec in recommendations:
        export_data.append({
            'ID': rec.id,
            'Text': rec.text,
            'Source_Document': rec.document_source,
            'Section': rec.section_title,
            'Page_Number': rec.page_number,
            'Confidence_Score': rec.confidence_score,
            'Text_Length': len(rec.text),
            'Extraction_Method': getattr(rec, 'metadata', {}).get('extraction_method', 'Unknown')
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Recommendations CSV",
        data=csv,
        file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_concerns_csv():
    """Export concerns to CSV"""
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not concerns:
        st.warning("No concerns to export.")
        return
    
    # Prepare data for export
    export_data = []
    for concern in concerns:
        row_data = {
            'ID': concern.get('id', ''),
            'Text': concern.get('text', ''),
            'Source_Document': concern.get('document_source', ''),
            'Section': concern.get('section', ''),
            'Page_Number': concern.get('page_number', ''),
            'Confidence_Score': concern.get('confidence_score', 0),
            'Category': concern.get('category', ''),
            'Text_Length': len(concern.get('text', '')),
            'Extraction_Method': concern.get('extraction_method', 'Unknown'),
            'Timestamp': concern.get('timestamp', '')
        }
        
        # Add metadata as separate columns
        if concern.get('metadata'):
            for key, value in concern['metadata'].items():
                row_data[f'metadata_{key}'] = value
        
        export_data.append(row_data)
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Concerns CSV",
        data=csv,
        file_name=f"concerns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_combined_csv():
    """Export combined recommendations and concerns to CSV"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not recommendations and not concerns:
        st.warning("No data to export.")
        return
    
    # Prepare combined data
    export_data = []
    
    for rec in recommendations:
        export_data.append({
            'Type': 'Recommendation',
            'ID': rec.id,
            'Text': rec.text,
            'Source_Document': rec.document_source,
            'Section': rec.section_title,
            'Page_Number': rec.page_number,
            'Confidence_Score': rec.confidence_score,
            'Category': getattr(rec, 'metadata', {}).get('category', 'General'),
            'Text_Length': len(rec.text),
            'Extraction_Method': getattr(rec, 'metadata', {}).get('extraction_method', 'llm')
        })
    
    for concern in concerns:
        row_data = {
            'Type': 'Concern',
            'ID': concern.get('id', ''),
            'Text': concern.get('text', ''),
            'Source_Document': concern.get('document_source', ''),
            'Section': concern.get('section', ''),
            'Page_Number': concern.get('page_number', ''),
            'Confidence_Score': concern.get('confidence_score', 0),
            'Category': concern.get('category', 'General'),
            'Text_Length': len(concern.get('text', '')),
            'Extraction_Method': concern.get('extraction_method', 'Unknown')
        }
        
        export_data.append(row_data)
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Combined CSV",
        data=csv,
        file_name=f"combined_extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
