# ===============================================
# FILE: modules/ui/extraction_components.py (COMPLETE VERSION)
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import json
import re
from pathlib import Path

# Import required modules with robust error handling (FIXED VARIABLE NAMES)
try:
    import sys
    sys.path.append('modules')
    from enhanced_section_extractor import EnhancedSectionExtractor
    ENHANCED_EXTRACTOR_AVAILABLE = True
    logging.info("✅ EnhancedSectionExtractor imported successfully")
except ImportError as import_error:
    ENHANCED_EXTRACTOR_AVAILABLE = False
    logging.warning(f"EnhancedSectionExtractor not available: {import_error}")
    
    class EnhancedSectionExtractor:
        def extract_individual_recommendations(self, text):
            return []
        def extract_individual_responses(self, text):
            return []
        def compare_inquiry_and_response_documents(self, inquiry_text, response_text):
            return {'matched_pairs': [], 'unmatched_recommendations': [], 'unmatched_responses': []}

try:
    from llm_extractor import LLMRecommendationExtractor
    LLM_EXTRACTOR_AVAILABLE = True
    logging.info("✅ LLMRecommendationExtractor imported successfully")
except ImportError as import_error:
    LLM_EXTRACTOR_AVAILABLE = False
    logging.warning(f"LLMRecommendationExtractor not available: {import_error}")
    
    class LLMRecommendationExtractor:
        def extract_recommendations(self, text, source):
            return {'recommendations': [], 'extraction_info': {'method': 'mock', 'error': 'LLM not available'}}
        def get_extraction_stats(self, results):
            return {'total_recommendations': 0, 'total_concerns': 0}
        def validate_extraction(self, results):
            return {'quality_score': 0.0, 'issues': ['LLM extractor not available']}

try:
    from core_utils import SecurityValidator, extract_government_document_metadata, detect_inquiry_document_structure
    CORE_UTILS_AVAILABLE = True
    logging.info("✅ Core utilities imported successfully")
except ImportError as import_error:
    CORE_UTILS_AVAILABLE = False
    logging.warning(f"Core utilities not available: {import_error}")
    
    class SecurityValidator:
        @staticmethod
        def validate_text_input(text, max_length=10000):
            return str(text)[:max_length] if text else ""
    
    def extract_government_document_metadata(content):
        return {'document_type': 'unknown'}
    
    def detect_inquiry_document_structure(content):
        return {'document_structure': 'unknown', 'has_recommendations': False, 'has_responses': False}

# Configure logging
logging.basicConfig(level=logging.INFO)

# ===============================================
# MAIN EXTRACTION TAB FUNCTION
# ===============================================

def render_extraction_tab():
    """Render the main extraction tab for government inquiry documents"""
    st.header("🔍 Recommendation & Response Extraction")
    
    st.markdown("""
    Extract **recommendations** from inquiry reports and **government responses** from response documents. 
    The system automatically detects document type and uses appropriate extraction patterns for any UK government inquiry.
    """)
    
    # Show component availability status
    if not ENHANCED_EXTRACTOR_AVAILABLE:
        st.warning("⚠️ Enhanced section extractor not available - using fallback mode")
    if not LLM_EXTRACTOR_AVAILABLE:
        st.info("ℹ️ AI-powered extraction not available - using pattern-based extraction only")
    
    # Check if documents are uploaded
    if not st.session_state.get('uploaded_documents', []):
        st.info("📁 Please upload documents first in the Upload tab.")
        return
    
    # Validate documents for extraction
    is_valid, validation_message = validate_documents_for_extraction()
    
    if not is_valid:
        st.error(f"❌ {validation_message}")
        return
    
    st.success(f"✅ {validation_message}")
    
    # Main extraction interface
    render_extraction_interface()
    
    # Show previous extraction results if available
    if st.session_state.get('extraction_results'):
        render_extraction_results()
    
    # Document analysis section
    render_document_analysis()

# ===============================================
# EXTRACTION INTERFACE
# ===============================================

def render_extraction_interface():
    """Render the extraction interface with options"""
    st.subheader("🚀 Extract Recommendations & Responses")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Extraction method selection
        extraction_method = st.selectbox(
            "Extraction Method:",
            options=["Enhanced Pattern-Based", "AI-Powered (OpenAI)", "Combined (Pattern + AI)"],
            index=0,
            help="Choose extraction method based on available components"
        )
        
        # Document selection
        docs = st.session_state.get('uploaded_documents', [])
        doc_options = [f"{doc.get('filename', 'Unknown')} ({detect_document_type(doc)})" for doc in docs]
        
        selected_docs = st.multiselect(
            "Select documents to process:",
            options=doc_options,
            default=doc_options,  # Select all by default
            help="Choose which documents to extract from"
        )
    
    with col2:
        st.markdown("**Extraction Options**")
        
        extract_recommendations = st.checkbox("Extract Recommendations", value=True)
        extract_responses = st.checkbox("Extract Government Responses", value=True)
        extract_individual_items = st.checkbox("Extract Individual Numbered Items", value=True)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            confidence_threshold = st.slider(
                "Confidence Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Minimum confidence score for extracted items"
            )
            
            max_items_per_doc = st.number_input(
                "Max items per document:",
                min_value=1,
                max_value=500,
                value=50,
                help="Maximum number of items to extract per document"
            )
            
            include_context = st.checkbox(
                "Include context information",
                value=True,
                help="Include surrounding text for context"
            )
    
    # Process button
    if st.button("🔍 Start Extraction", type="primary", disabled=not selected_docs):
        if not extract_recommendations and not extract_responses:
            st.error("Please select at least one extraction type")
        else:
            # Get selected document indices
            selected_indices = [i for i, opt in enumerate(doc_options) if opt in selected_docs]
            selected_documents = [docs[i] for i in selected_indices]
            
            process_document_extraction(
                selected_documents,
                extraction_method,
                extract_recommendations,
                extract_responses,
                extract_individual_items,
                confidence_threshold,
                max_items_per_doc,
                include_context
            )

# ===============================================
# EXTRACTION PROCESSING
# ===============================================

def process_document_extraction(
    documents: List[Dict[str, Any]],
    method: str,
    extract_recommendations: bool,
    extract_responses: bool,
    extract_individual_items: bool,
    confidence_threshold: float,
    max_items_per_doc: int,
    include_context: bool
):
    """Process document extraction with the selected method"""
    
    st.subheader("🔄 Processing Documents...")
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_recommendations = []
    all_responses = []
    all_individual_items = []
    doc_results = []
    
    # Initialize extractors
    enhanced_extractor = EnhancedSectionExtractor() if ENHANCED_EXTRACTOR_AVAILABLE else None
    llm_extractor = LLMRecommendationExtractor() if LLM_EXTRACTOR_AVAILABLE else None
    
    # Process each document
    for i, doc in enumerate(documents):
        doc_name = doc.get('filename', f'Document {i+1}')
        status_text.text(f"Processing {doc_name}...")
        progress_bar.progress((i + 1) / len(documents))
        
        try:
            # Get document content
            content = get_document_content_for_extraction(doc)
            
            if not content or len(content.strip()) < 100:
                doc_results.append({
                    'document': doc_name,
                    'status': '⚠️ No suitable content found',
                    'recommendations_found': 0,
                    'responses_found': 0,
                    'individual_items_found': 0,
                    'document_type': detect_document_type(doc)
                })
                continue
            
            # Analyze document structure
            doc_structure = detect_inquiry_document_structure(content)
            doc_metadata = extract_government_document_metadata(content)
            
            recommendations = []
            responses = []
            individual_items = []
            
            # Extract based on selected method
            if method == "Enhanced Pattern-Based" and enhanced_extractor:
                if extract_recommendations:
                    recommendations = enhanced_extractor.extract_individual_recommendations(content)
                if extract_responses:
                    responses = enhanced_extractor.extract_individual_responses(content)
                    
            elif method == "AI-Powered (OpenAI)" and llm_extractor:
                if extract_recommendations or extract_responses:
                    llm_result = llm_extractor.extract_recommendations(content, doc_name)
                    extracted_items = llm_result.get('recommendations', [])
                    
                    # Separate recommendations and responses
                    if extract_recommendations:
                        recommendations = [item for item in extracted_items if item.get('type') == 'recommendation']
                    if extract_responses:
                        responses = [item for item in extracted_items if item.get('type') == 'response']
                        
            elif method == "Combined (Pattern + AI)":
                # Use both methods and merge results
                if enhanced_extractor:
                    if extract_recommendations:
                        pattern_recs = enhanced_extractor.extract_individual_recommendations(content)
                        recommendations.extend(pattern_recs)
                    if extract_responses:
                        pattern_resps = enhanced_extractor.extract_individual_responses(content)
                        responses.extend(pattern_resps)
                
                if llm_extractor:
                    llm_result = llm_extractor.extract_recommendations(content, doc_name)
                    extracted_items = llm_result.get('recommendations', [])
                    
                    if extract_recommendations:
                        ai_recs = [item for item in extracted_items if item.get('type') == 'recommendation']
                        recommendations.extend(ai_recs)
                    if extract_responses:
                        ai_resps = [item for item in extracted_items if item.get('type') == 'response']
                        responses.extend(ai_resps)
            
            # Filter by confidence threshold
            recommendations = [r for r in recommendations if r.get('confidence_score', 0) >= confidence_threshold]
            responses = [r for r in responses if r.get('confidence_score', 0) >= confidence_threshold]
            
            # Limit number of items
            recommendations = recommendations[:max_items_per_doc]
            responses = responses[:max_items_per_doc]
            
            # Add document context if requested
            if include_context:
                for item in recommendations + responses:
                    item['document_context'] = {
                        'filename': doc_name,
                        'document_type': doc_structure.get('document_structure', 'unknown'),
                        'inquiry_type': doc_metadata.get('inquiry_type', 'unknown'),
                        'extraction_method': method
                    }
            
            # Store results
            all_recommendations.extend(recommendations)
            all_responses.extend(responses)
            all_individual_items.extend(recommendations + responses)
            
            doc_results.append({
                'document': doc_name,
                'status': '✅ Success',
                'recommendations_found': len(recommendations),
                'responses_found': len(responses),
                'individual_items_found': len(recommendations) + len(responses),
                'document_type': doc_structure.get('document_structure', 'unknown'),
                'inquiry_type': doc_metadata.get('inquiry_type', 'unknown'),
                'confidence_avg': sum(item.get('confidence_score', 0) for item in recommendations + responses) / max(1, len(recommendations) + len(responses)),
                'extraction_method': method,
                'sections_processed': len(doc.get('sections', [])),
                'content_length': len(content)
            })
            
        except Exception as extraction_error:
            logging.error(f"Error extracting from {doc_name}: {extraction_error}")
            doc_results.append({
                'document': doc_name,
                'status': f'❌ Error: {str(extraction_error)[:50]}...',
                'recommendations_found': 0,
                'responses_found': 0,
                'individual_items_found': 0,
                'error': str(extraction_error),
                'document_type': 'error'
            })
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Extraction completed!")
    
    # Store results in session state
    st.session_state.extraction_results = {
        'recommendations': all_recommendations,
        'responses': all_responses,
        'individual_items': all_individual_items,
        'document_results': doc_results,
        'extraction_method': method,
        'extraction_date': datetime.now().isoformat(),
        'total_documents': len(documents),
        'successful_extractions': len([r for r in doc_results if '✅' in r['status']]),
        'settings': {
            'confidence_threshold': confidence_threshold,
            'max_items_per_doc': max_items_per_doc,
            'include_context': include_context,
            'extract_recommendations': extract_recommendations,
            'extract_responses': extract_responses
        }
    }
    
    # Show results summary
    show_extraction_summary(all_recommendations, all_responses, doc_results, method)

# ===============================================
# RESULTS DISPLAY
# ===============================================

def show_extraction_summary(recommendations: List[Dict], responses: List[Dict], doc_results: List[Dict], method: str):
    """Show summary of extraction results"""
    st.success(f"🎉 Extraction completed using {method}!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Processed", len(doc_results))
    with col2:
        successful = len([r for r in doc_results if '✅' in r['status']])
        st.metric("Successful", successful)
    with col3:
        total_recommendations = len(recommendations)
        st.metric("Recommendations", total_recommendations)
    with col4:
        total_responses = len(responses)
        st.metric("Responses", total_responses)
    
    # Document results table
    if doc_results:
        st.subheader("📊 Document Processing Results")
        
        results_df = pd.DataFrame(doc_results)
        
        # Format the results table
        display_df = results_df[[
            'document', 'status', 'document_type', 'recommendations_found', 
            'responses_found', 'extraction_method'
        ]].copy()
        
        display_df.columns = [
            'Document', 'Status', 'Document Type', 'Recommendations', 
            'Responses', 'Method'
        ]
        
        st.dataframe(display_df, use_container_width=True)
    
    # Quick preview of extracted items
    if recommendations or responses:
        st.subheader("🔍 Extracted Items Preview")
        
        preview_tab1, preview_tab2 = st.tabs(["📝 Recommendations", "📋 Responses"])
        
        with preview_tab1:
            if recommendations:
                show_items_preview(recommendations, "recommendation")
            else:
                st.info("No recommendations extracted with current settings")
        
        with preview_tab2:
            if responses:
                show_items_preview(responses, "response")
            else:
                st.info("No responses extracted with current settings")

def render_extraction_results():
    """Render detailed extraction results if available"""
    results = st.session_state.get('extraction_results', {})
    if not results:
        return
    
    st.subheader("📋 Detailed Extraction Results")
    
    # Results overview
    with st.expander("📊 Extraction Overview", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Extraction Summary:**")
            st.write(f"- **Method:** {results.get('extraction_method', 'Unknown')}")
            st.write(f"- **Date:** {results.get('extraction_date', 'Unknown')}")
            st.write(f"- **Documents:** {results.get('total_documents', 0)}")
            st.write(f"- **Success Rate:** {results.get('successful_extractions', 0)}/{results.get('total_documents', 0)}")
        
        with col2:
            st.write("**Items Extracted:**")
            st.write(f"- **Recommendations:** {len(results.get('recommendations', []))}")
            st.write(f"- **Responses:** {len(results.get('responses', []))}")
            st.write(f"- **Total Items:** {len(results.get('individual_items', []))}")
    
    # Detailed item view
    tab1, tab2, tab3 = st.tabs(["📝 All Recommendations", "📋 All Responses", "⚙️ Settings"])
    
    with tab1:
        recommendations = results.get('recommendations', [])
        if recommendations:
            show_detailed_items(recommendations, "recommendation")
        else:
            st.info("No recommendations found")
    
    with tab2:
        responses = results.get('responses', [])
        if responses:
            show_detailed_items(responses, "response")
        else:
            st.info("No responses found")
    
    with tab3:
        settings = results.get('settings', {})
        st.json(settings)
    
    # Export options
    if results.get('individual_items'):
        render_extraction_export_options(results)

# ===============================================
# HELPER FUNCTIONS
# ===============================================

def get_document_content_for_extraction(doc: Dict[str, Any]) -> str:
    """Get the best content from document for extraction"""
    
    # Priority 1: Use sections if available and relevant
    sections = doc.get('sections', [])
    if sections:
        relevant_content = []
        for section in sections:
            section_type = section.get('type', '').lower()
            # Include recommendation and response sections
            if any(keyword in section_type for keyword in ['recommendation', 'response', 'finding', 'conclusion']):
                relevant_content.append(f"=== {section.get('title', 'Section')} ===\n")
                relevant_content.append(section.get('content', ''))
        
        if relevant_content:
            return '\n\n'.join(relevant_content)
    
    # Priority 2: Use full text content
    content = doc.get('text', '') or doc.get('content', '')
    if content and len(content.strip()) > 100:
        return content
    
    # Priority 3: Fallback to any available content
    return doc.get('full_text', '') or ""

def detect_document_type(doc: Dict[str, Any]) -> str:
    """Detect the type of government document"""
    try:
        filename = doc.get('filename', '').lower()
        content = doc.get('text', '') or doc.get('content', '')
        
        if not content:
            return 'unknown'
        
        content_lower = content.lower()
        
        # Check filename patterns
        if 'response' in filename:
            return 'government_response'
        elif any(word in filename for word in ['inquiry', 'report', 'investigation']):
            return 'inquiry_report'
        
        # Check content patterns
        if re.search(r'(?i)government\s+response', content):
            return 'government_response'
        elif re.search(r'(?i)(?:inquiry\s+)?report', content) and re.search(r'(?i)recommendations?', content):
            return 'inquiry_report'
        elif re.search(r'(?i)cabinet\s+office', content):
            return 'cabinet_office_document'
        
        return 'government_document'
        
    except Exception as type_error:
        logging.error(f"Error detecting document type: {type_error}")
        return 'unknown'

def validate_documents_for_extraction() -> tuple[bool, str]:
    """Validate that documents are ready for extraction"""
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        return False, "No documents uploaded"
    
    # Check content availability
    content_available = 0
    for doc in docs:
        content = doc.get('text', '') or doc.get('content', '')
        if content and len(content.strip()) > 50:
            content_available += 1
    
    if content_available == 0:
        return False, "No documents have readable content"
    
    # Analyze document types
    doc_types = {}
    for doc in docs:
        doc_type = detect_document_type(doc)
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    # Create validation message
    type_summary = ", ".join([f"{count} {doc_type.replace('_', ' ')}" for doc_type, count in doc_types.items()])
    
    return True, f"Ready: {content_available} documents with content ({type_summary})"

def show_items_preview(items: List[Dict], item_type: str, max_preview: int = 3):
    """Show a preview of extracted items"""
    if not items:
        st.info(f"No {item_type}s extracted")
        return
    
    st.write(f"**Showing {min(len(items), max_preview)} of {len(items)} {item_type}s:**")
    
    for i, item in enumerate(items[:max_preview]):
        with st.expander(f"{item_type.title()} {i+1}: {item.get('text', '')[:100]}...", expanded=i == 0):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Text:**")
                st.write(item.get('text', 'No text available'))
            
            with col2:
                st.write("**Metadata:**")
                st.write(f"- **Number:** {item.get('number', 'N/A')}")
                st.write(f"- **Confidence:** {item.get('confidence_score', 0):.2f}")
                
                if item_type == 'response':
                    response_type = item.get('response_type', 'unclear')
                    st.write(f"- **Type:** {response_type.replace('_', ' ').title()}")
                
                if item.get('document_context'):
                    context = item['document_context']
                    st.write(f"- **Source:** {context.get('filename', 'Unknown')}")
    
    if len(items) > max_preview:
        st.info(f"... and {len(items) - max_preview} more {item_type}s. View all in the detailed results below.")

def show_detailed_items(items: List[Dict], item_type: str):
    """Show detailed view of all extracted items"""
    if not items:
        st.info(f"No {item_type}s found")
        return
    
    st.write(f"**{len(items)} {item_type}s extracted:**")
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(f"Search {item_type}s:", key=f"search_{item_type}")
    with col2:
        min_confidence = st.slider(
            "Min confidence:",
            0.0, 1.0, 0.0, 0.1,
            key=f"confidence_{item_type}"
        )
    
    # Filter items
    filtered_items = items
    if search_term:
        filtered_items = [item for item in filtered_items if search_term.lower() in item.get('text', '').lower()]
    if min_confidence > 0:
        filtered_items = [item for item in filtered_items if item.get('confidence_score', 0) >= min_confidence]
    
    st.write(f"Showing {len(filtered_items)} of {len(items)} {item_type}s")
    
    # Display items
    for i, item in enumerate(filtered_items):
        with st.expander(f"{item.get('number', i+1)}. {item.get('text', '')[:80]}...", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Full Text:**")
                st.write(item.get('text', 'No text available'))
                
                if item.get('context'):
                    st.write("**Context:**")
                    st.caption(item.get('context', ''))
            
            with col2:
                st.write("**Details:**")
                st.write(f"- **ID:** {item.get('number', 'N/A')}")
                st.write(f"- **Confidence:** {item.get('confidence_score', 0):.3f}")
                st.write(f"- **Length:** {item.get('length', len(item.get('text', '')))}")
                st.write(f"- **Words:** {item.get('word_count', len(item.get('text', '').split()))}")
                
                if item_type == 'response':
                    response_type = item.get('response_type', 'unclear')
                    emoji = {'accepted': '✅', 'rejected': '❌', 'partially_accepted': '⚡', 'under_consideration': '🤔'}.get(response_type, '❓')
                    st.write(f"- **Response Type:** {emoji} {response_type.replace('_', ' ').title()}")
                
                if item.get('document_context'):
                    context = item['document_context']
                    st.write(f"- **Document:** {context.get('filename', 'Unknown')}")
                    st.write(f"- **Doc Type:** {context.get('document_type', 'Unknown')}")

def render_document_analysis():
    """Render document structure analysis section"""
    if not st.session_state.get('uploaded_documents'):
        return
    
    st.subheader("📈 Document Analysis")
    
    with st.expander("🔍 Document Structure Analysis", expanded=False):
        docs = st.session_state.get('uploaded_documents', [])
        
        analysis_results = []
        for doc in docs:
            content = get_document_content_for_extraction(doc)
            if content:
                structure = detect_inquiry_document_structure(content)
                metadata = extract_government_document_metadata(content)
                
                analysis_results.append({
                    'filename': doc.get('filename', 'Unknown'),
                    'document_type': structure.get('document_structure', 'unknown'),
                    'inquiry_type': metadata.get('inquiry_type', 'unknown'),
                    'has_recommendations': structure.get('has_recommendations', False),
                    'has_responses': structure.get('has_responses', False),
                    'estimated_recommendations': structure.get('estimated_recommendations', 0),
                    'estimated_responses': structure.get('estimated_responses', 0),
                    'has_table_of_contents': structure.get('has_table_of_contents', False),
                    'content_length': len(content)
                })
        
        if analysis_results:
            analysis_df = pd.DataFrame(analysis_results)
            st.dataframe(analysis_df, use_container_width=True)
        else:
            st.info("No documents available for analysis")

def render_extraction_export_options(results: Dict[str, Any]):
    """Render export options for extraction results"""
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download as CSV"):
            export_extraction_csv(results)
    
    with col2:
        if st.button("📋 Download as JSON"):
            export_extraction_json(results)
    
    with col3:
        if st.button("📄 Download as Text"):
            export_extraction_text(results)

def export_extraction_csv(results: Dict[str, Any]):
    """Export extraction results as CSV"""
    try:
        all_items = []
        
        for item in results.get('recommendations', []):
            all_items.append({
                'type': 'recommendation',
                'number': item.get('number', ''),
                'text': item.get('text', ''),
                'confidence': item.get('confidence_score', 0),
                'document': item.get('document_context', {}).get('filename', 'Unknown'),
                'length': item.get('length', 0),
                'word_count': item.get('word_count', 0)
            })
        
        for item in results.get('responses', []):
            all_items.append({
                'type': 'response',
                'number': item.get('number', ''),
                'text': item.get('text', ''),
                'confidence': item.get('confidence_score', 0),
                'response_type': item.get('response_type', 'unclear'),
                'document': item.get('document_context', {}).get('filename', 'Unknown'),
                'length': item.get('length', 0),
                'word_count': item.get('word_count', 0)
            })
        
        if all_items:
            df = pd.DataFrame(all_items)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                "📥 Download CSV",
                csv_data,
                f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.error("No data to export")
            
    except Exception as export_error:
        st.error(f"Error exporting CSV: {export_error}")

def export_extraction_json(results: Dict[str, Any]):
    """Export extraction results as JSON"""
    try:
        export_data = {
            'extraction_metadata': {
                'method': results.get('extraction_method', 'Unknown'),
                'date': results.get('extraction_date', datetime.now().isoformat()),
                'total_documents': results.get('total_documents', 0),
                'settings': results.get('settings', {})
            },
            'recommendations': results.get('recommendations', []),
            'responses': results.get('responses', []),
            'document_results': results.get('document_results', [])
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            "📥 Download JSON",
            json_data,
            f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
        
    except Exception as export_error:
        st.error(f"Error exporting JSON: {export_error}")

def export_extraction_text(results: Dict[str, Any]):
    """Export extraction results as formatted text"""
    try:
        text_lines = [
            "# Extraction Results Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Method: {results.get('extraction_method', 'Unknown')}",
            f"Documents Processed: {results.get('total_documents', 0)}",
            "",
            "## Recommendations",
            ""
        ]
        
        recommendations = results.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                text_lines.extend([
                    f"### Recommendation {rec.get('number', i)}",
                    f"**Confidence:** {rec.get('confidence_score', 0):.3f}",
                    f"**Source:** {rec.get('document_context', {}).get('filename', 'Unknown')}",
                    "",
                    rec.get('text', 'No text available'),
                    "",
                    "---",
                    ""
                ])
        else:
            text_lines.append("No recommendations found.")
        
        text_lines.extend([
            "",
            "## Government Responses",
            ""
        ])
        
        responses = results.get('responses', [])
        if responses:
            for i, resp in enumerate(responses, 1):
                response_type = resp.get('response_type', 'unclear')
                emoji = {'accepted': '✅', 'rejected': '❌', 'partially_accepted': '⚡', 'under_consideration': '🤔'}.get(response_type, '❓')
                
                text_lines.extend([
                    f"### Response {resp.get('number', i)} {emoji}",
                    f"**Type:** {response_type.replace('_', ' ').title()}",
                    f"**Confidence:** {resp.get('confidence_score', 0):.3f}",
                    f"**Source:** {resp.get('document_context', {}).get('filename', 'Unknown')}",
                    "",
                    resp.get('text', 'No text available'),
                    "",
                    "---",
                    ""
                ])
        else:
            text_lines.append("No responses found.")
        
        # Add document results summary
        text_lines.extend([
            "",
            "## Document Processing Summary",
            ""
        ])
        
        for doc_result in results.get('document_results', []):
            text_lines.extend([
                f"**{doc_result['document']}**",
                f"- Status: {doc_result['status']}",
                f"- Type: {doc_result.get('document_type', 'Unknown')}",
                f"- Recommendations: {doc_result.get('recommendations_found', 0)}",
                f"- Responses: {doc_result.get('responses_found', 0)}",
                ""
            ])
        
        text_content = "\n".join(text_lines)
        
        st.download_button(
            "📥 Download Text Report",
            text_content,
            f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )
        
    except Exception as export_error:
        st.error(f"Error exporting text: {export_error}")

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def log_extraction_action(action: str, details: str = ""):
    """Log extraction-related user actions"""
    timestamp = datetime.now().isoformat()
    logging.info(f"[{timestamp}] Extraction Action: {action} - {details}")
    
    # Store in session state for debugging
    if 'extraction_history' not in st.session_state:
        st.session_state.extraction_history = []
    
    st.session_state.extraction_history.append({
        'timestamp': timestamp,
        'action': action,
        'details': details
    })
    
    # Keep only last 50 actions
    if len(st.session_state.extraction_history) > 50:
        st.session_state.extraction_history = st.session_state.extraction_history[-50:]

def get_extraction_statistics():
    """Get extraction statistics from session state"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        return {
            'total_extractions': 0,
            'total_recommendations': 0,
            'total_responses': 0,
            'last_extraction': None
        }
    
    return {
        'total_extractions': 1,  # Could track multiple extraction sessions
        'total_recommendations': len(results.get('recommendations', [])),
        'total_responses': len(results.get('responses', [])),
        'last_extraction': results.get('extraction_date'),
        'last_method': results.get('extraction_method'),
        'success_rate': results.get('successful_extractions', 0) / max(1, results.get('total_documents', 1))
    }

def clear_extraction_results():
    """Clear extraction results from session state"""
    if 'extraction_results' in st.session_state:
        del st.session_state.extraction_results
    log_extraction_action("clear_results", "Extraction results cleared")

def validate_extraction_settings(settings: Dict[str, Any]) -> tuple[bool, str]:
    """Validate extraction settings"""
    try:
        confidence_threshold = settings.get('confidence_threshold', 0.3)
        max_items = settings.get('max_items_per_doc', 50)
        
        if not (0.0 <= confidence_threshold <= 1.0):
            return False, "Confidence threshold must be between 0.0 and 1.0"
        
        if not (1 <= max_items <= 1000):
            return False, "Max items per document must be between 1 and 1000"
        
        return True, "Settings are valid"
        
    except Exception as validation_error:
        return False, f"Settings validation error: {validation_error}"

# ===============================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# ===============================================

def show_recommendations_compact(recommendations: List[Dict]):
    """Show recommendations in compact format - updated for government documents"""
    if not recommendations:
        st.info("No recommendations to display")
        return
    
    for i, rec in enumerate(recommendations, 1):
        # Create descriptive title
        title = f"**{rec.get('number', i)}** - {rec.get('type', 'recommendation').replace('_', ' ').title()}"
        preview = rec.get('text', '')[:100]
        if len(rec.get('text', '')) > 100:
            preview += "..."
        
        with st.expander(f"{title}: {preview}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Full Text:**")
                st.write(rec.get('text', 'No text available'))
            
            with col2:
                st.write("**Details:**")
                st.write(f"- **Confidence:** {rec.get('confidence_score', 0):.3f}")
                st.write(f"- **Source:** {rec.get('document_context', {}).get('filename', 'Unknown')}")
                
                if rec.get('length'):
                    st.write(f"- **Length:** {rec.get('length')} chars")

def get_recommendations_for_annotation():
    """Get extracted recommendations for annotation - updated interface"""
    results = st.session_state.get('extraction_results', {})
    recommendations = results.get('recommendations', [])
    
    if not recommendations:
        return []
    
    # Convert to format expected by annotation components
    formatted_recs = []
    for rec in recommendations:
        formatted_rec = {
            'id': rec.get('number', len(formatted_recs) + 1),
            'text': rec.get('text', ''),
            'source': rec.get('document_context', {}).get('filename', 'Unknown'),
            'confidence': rec.get('confidence_score', 0.0),
            'metadata': {
                'extraction_method': results.get('extraction_method', 'unknown'),
                'document_type': rec.get('document_context', {}).get('document_type', 'unknown'),
                'length': rec.get('length', len(rec.get('text', ''))),
                'word_count': rec.get('word_count', len(rec.get('text', '').split()))
            }
        }
        formatted_recs.append(formatted_rec)
    
    return formatted_recs

# ===============================================
# MODULE EXPORTS
# ===============================================

__all__ = [
    # Main functions
    'render_extraction_tab',
    'render_extraction_interface',
    'render_extraction_results',
    'process_document_extraction',
    
    # Display functions
    'show_extraction_summary',
    'show_detailed_items',
    'show_items_preview',
    'render_document_analysis',
    
    # Export functions
    'render_extraction_export_options',
    'export_extraction_csv',
    'export_extraction_json',
    'export_extraction_text',
    
    # Utility functions
    'get_document_content_for_extraction',
    'detect_document_type',
    'validate_documents_for_extraction',
    'log_extraction_action',
    'get_extraction_statistics',
    'clear_extraction_results',
    'validate_extraction_settings',
    
    # Backwards compatibility
    'show_recommendations_compact',
    'get_recommendations_for_annotation',
    
    # Availability flags
    'ENHANCED_EXTRACTOR_AVAILABLE',
    'LLM_EXTRACTOR_AVAILABLE',
    'CORE_UTILS_AVAILABLE'
]

# ===============================================
# MODULE INITIALIZATION
# ===============================================

# Log module initialization
if ENHANCED_EXTRACTOR_AVAILABLE and LLM_EXTRACTOR_AVAILABLE:
    logging.info("✅ Extraction components loaded with full functionality (Enhanced + AI)")
elif ENHANCED_EXTRACTOR_AVAILABLE:
    logging.info("✅ Extraction components loaded with enhanced pattern-based extraction")
elif LLM_EXTRACTOR_AVAILABLE:
    logging.info("✅ Extraction components loaded with AI-powered extraction only")
else:
    logging.warning("⚠️ Extraction components loaded with basic functionality only")

logging.info("🎉 Extraction components module is COMPLETE and ready for universal inquiry document processing!")
