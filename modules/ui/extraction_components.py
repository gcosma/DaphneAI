# ===============================================
# EXTRACTION COMPONENTS MODULE - FIXED DATA COMPATIBILITY
# Fixes data structure mismatch between Upload and Extract tabs
# 
# This module provides extraction functionality that properly reads
# documents processed by the Upload tab, handling the nested data
# structure where text content is stored in 'extraction_result'.
#
# Key Fixes:
# - Updated get_document_content_for_extraction() to handle nested data
# - Fixed validate_documents_for_extraction() to properly check content
# - Enhanced detect_document_type() to work with upload tab data
# - Added debugging information for data structure issues
#
# Author: Recommendation-Response Tracker Team  
# Version: 2.1 - Data compatibility fix
# Last Updated: 2025
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import enhanced extractor with fallback
ENHANCED_EXTRACTOR_AVAILABLE = False
LLM_EXTRACTOR_AVAILABLE = False
CORE_UTILS_AVAILABLE = False

try:
    from modules.enhanced_section_extractor import EnhancedSectionExtractor
    ENHANCED_EXTRACTOR_AVAILABLE = True
    logging.info("✅ Enhanced section extractor imported successfully")
except ImportError as import_error:
    logging.warning(f"⚠️ Enhanced section extractor not available: {import_error}")
    # Create fallback extractor
    class EnhancedSectionExtractor:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
        
        def extract_individual_recommendations(self, text: str) -> List[Dict[str, Any]]:
            """Fallback recommendation extraction"""
            recommendations = []
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                # Simple pattern matching for recommendations
                if re.search(r'(?i)recommendation\s+(\d+)', line_stripped):
                    match = re.search(r'(?i)recommendation\s+(\d+)[\s:\-\.]*(.+)', line_stripped)
                    if match:
                        recommendations.append({
                            'number': match.group(1),
                            'text': match.group(2).strip(),
                            'line_number': i,
                            'confidence': 0.7,
                            'extraction_method': 'fallback_pattern'
                        })
                elif re.search(r'(?i)^\s*(\d+)[\.\)]\s+(.{20,})', line_stripped):
                    match = re.search(r'(?i)^\s*(\d+)[\.\)]\s+(.+)', line_stripped)
                    if match:
                        recommendations.append({
                            'number': match.group(1),
                            'text': match.group(2).strip(),
                            'line_number': i,
                            'confidence': 0.6,
                            'extraction_method': 'fallback_numbered'
                        })
            
            return recommendations[:50]  # Limit to 50 items
        
        def extract_individual_responses(self, text: str) -> List[Dict[str, Any]]:
            """Fallback response extraction"""
            responses = []
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                # Simple pattern matching for responses
                if re.search(r'(?i)response\s+to\s+recommendation\s+(\d+)', line_stripped):
                    match = re.search(r'(?i)response\s+to\s+recommendation\s+(\d+)[\s:\-\.]*(.+)', line_stripped)
                    if match:
                        responses.append({
                            'number': match.group(1),
                            'text': match.group(2).strip(),
                            'line_number': i,
                            'confidence': 0.7,
                            'extraction_method': 'fallback_response_pattern'
                        })
                elif re.search(r'(?i)(accepted|rejected|not\s+accepted)', line_stripped):
                    responses.append({
                        'number': str(len(responses) + 1),
                        'text': line_stripped,
                        'line_number': i,
                        'confidence': 0.5,
                        'extraction_method': 'fallback_status_pattern'
                    })
            
            return responses[:50]  # Limit to 50 items

try:
    from modules.llm_extractor import LLMRecommendationExtractor
    LLM_EXTRACTOR_AVAILABLE = True
    logging.info("✅ LLM extractor imported successfully")
except ImportError as import_error:
    logging.warning(f"⚠️ LLM extractor not available: {import_error}")
    # Create fallback LLM extractor
    class LLMRecommendationExtractor:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
        
        def extract_recommendations(self, text: str, document_name: str = "") -> Dict[str, Any]:
            """Fallback LLM extraction"""
            return {
                'recommendations': [],
                'error': 'LLM extractor not available',
                'extraction_method': 'fallback_llm'
            }

try:
    from modules.core_utils import (
        SecurityValidator, 
        log_user_action, 
        extract_government_document_metadata,
        detect_inquiry_document_structure
    )
    CORE_UTILS_AVAILABLE = True
    logging.info("✅ Core utilities imported successfully")
except ImportError as import_error:
    CORE_UTILS_AVAILABLE = False
    logging.warning(f"Core utilities not available: {import_error}")
    
    class SecurityValidator:
        @staticmethod
        def validate_text_input(text, max_length=10000):
            return str(text)[:max_length] if text else ""
    
    def log_user_action(action, details):
        logging.info(f"User action: {action} - {details}")
    
    def extract_government_document_metadata(content):
        return {'document_type': 'unknown'}
    
    def detect_inquiry_document_structure(content):
        return {'document_structure': 'unknown', 'has_recommendations': False, 'has_responses': False}

# ===============================================
# FIXED DATA ACCESS FUNCTIONS
# ===============================================

def get_document_content_for_extraction(doc: Dict[str, Any]) -> str:
    """
    FIXED: Get document content from uploaded document data structure
    
    This function now properly handles the nested data structure where
    text content is stored in doc['extraction_result']['text']
    """
    try:
        # Method 1: Check if text is directly available (legacy format)
        if doc.get('text'):
            return doc['text']
        
        if doc.get('content'):
            return doc['content']
        
        # Method 2: Check extraction_result (new upload format)
        extraction_result = doc.get('extraction_result', {})
        if extraction_result:
            # Try text field first
            if extraction_result.get('text'):
                return extraction_result['text']
            
            # Try content field as fallback
            if extraction_result.get('content'):
                return extraction_result['content']
        
        # Method 3: Debug - show what fields are available
        available_fields = list(doc.keys())
        logging.warning(f"Could not find text content. Available fields: {available_fields}")
        
        # Try to extract from any field that might contain text
        for field in ['raw_text', 'extracted_text', 'document_text']:
            if doc.get(field):
                return doc[field]
        
        return ""
        
    except Exception as e:
        logging.error(f"Error extracting document content: {e}")
        return ""

def detect_document_type(doc: Dict[str, Any]) -> str:
    """
    FIXED: Detect document type from uploaded document data
    
    Now properly handles nested data structure and provides fallbacks
    """
    try:
        # Check if metadata already contains document type
        metadata = doc.get('metadata', {})
        if metadata.get('document_type'):
            return metadata['document_type']
        
        # Check extraction result metadata
        extraction_result = doc.get('extraction_result', {})
        if extraction_result.get('metadata', {}).get('document_type'):
            return extraction_result['metadata']['document_type']
        
        # Get content for analysis
        content = get_document_content_for_extraction(doc)
        
        if not content:
            return 'no_content'
        
        content_lower = content.lower()
        
        # Analyze content to determine type
        if re.search(r'(?i)government\s+response', content):
            return 'government_response'
        elif re.search(r'(?i)(?:inquiry\s+)?report', content) and re.search(r'(?i)recommendations?', content):
            return 'inquiry_report'
        elif re.search(r'(?i)cabinet\s+office', content):
            return 'cabinet_office_document'
        elif 'recommendation' in content_lower:
            return 'document_with_recommendations'
        elif 'response' in content_lower:
            return 'document_with_responses'
        
        return 'government_document'
        
    except Exception as type_error:
        logging.error(f"Error detecting document type: {type_error}")
        return 'unknown'

def validate_documents_for_extraction() -> Tuple[bool, str]:
    """
    FIXED: Validate that documents are ready for extraction
    
    Now properly checks nested data structure from upload tab
    """
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        return False, "No documents uploaded"
    
    # Debug: Show document structure
    if docs:
        sample_doc = docs[0]
        logging.info(f"Sample document structure: {list(sample_doc.keys())}")
        if sample_doc.get('extraction_result'):
            logging.info(f"Extraction result keys: {list(sample_doc['extraction_result'].keys())}")
    
    # Check content availability
    content_available = 0
    inquiry_docs = 0
    response_docs = 0
    total_text_length = 0
    
    for doc in docs:
        content = get_document_content_for_extraction(doc)
        if content and len(content.strip()) > 50:
            content_available += 1
            total_text_length += len(content)
            
            # Check document type
            doc_type = detect_document_type(doc)
            if doc_type == 'inquiry_report':
                inquiry_docs += 1
            elif doc_type == 'government_response':
                response_docs += 1
    
    if content_available == 0:
        return False, "No documents have readable content"
    
    # Enhanced validation message
    validation_parts = []
    if inquiry_docs > 0:
        validation_parts.append(f"{inquiry_docs} inquiry report(s)")
    if response_docs > 0:
        validation_parts.append(f"{response_docs} government response(s)")
    
    type_summary = ", ".join(validation_parts) if validation_parts else f"{content_available} documents"
    
    avg_length = total_text_length // content_available if content_available > 0 else 0
    
    return True, f"Ready: {content_available} documents with content ({type_summary}), avg {avg_length:,} chars"

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
    col1, col2 = st.columns(2)
    with col1:
        if ENHANCED_EXTRACTOR_AVAILABLE:
            st.success("✅ Enhanced extractor available")
        else:
            st.warning("⚠️ Enhanced extractor using fallback mode")
    
    with col2:
        if LLM_EXTRACTOR_AVAILABLE:
            st.success("✅ AI-powered extraction available")
        else:
            st.info("ℹ️ AI-powered extraction not available - using pattern-based extraction only")
    
    # Check if documents are uploaded
    if not st.session_state.get('uploaded_documents', []):
        st.info("📁 Please upload documents first in the Upload tab.")
        return
    
    # Debug information
    with st.expander("🐛 Debug Information"):
        docs = st.session_state.get('uploaded_documents', [])
        st.write(f"**Found {len(docs)} uploaded documents:**")
        
        for i, doc in enumerate(docs):
            st.write(f"**Document {i+1}: {doc.get('filename', 'Unknown')}**")
            content = get_document_content_for_extraction(doc)
            st.write(f"- Content length: {len(content):,} characters")
            st.write(f"- Document type: {detect_document_type(doc)}")
            st.write(f"- Available fields: {list(doc.keys())}")
            if doc.get('extraction_result'):
                st.write(f"- Extraction result fields: {list(doc['extraction_result'].keys())}")
            st.write("---")
    
    # Validate documents for extraction
    is_valid, validation_message = validate_documents_for_extraction()
    
    if not is_valid:
        st.error(f"❌ {validation_message}")
        
        # Show troubleshooting info
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common issues:**
            - **No readable content**: The PDF text extraction may have failed
            - **Nested data structure**: Text might be in extraction_result
            - **Processing incomplete**: Re-upload the document in the Upload tab
            
            **What to try:**
            1. Go back to Upload tab and re-process the document
            2. Check if the PDF is text-based (not scanned image)
            3. Try a different PDF file
            """)
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
        available_methods = ["Enhanced Pattern-Based"]
        if LLM_EXTRACTOR_AVAILABLE:
            available_methods.extend(["AI-Powered (OpenAI)", "Combined (Pattern + AI)"])
        
        extraction_method = st.selectbox(
            "Extraction Method:",
            options=available_methods,
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
                value=0.0,
                step=0.1,
                help="Minimum confidence score for extracted items"
            )
            
            max_items_per_doc = st.number_input(
                "Max items per document:",
                min_value=1,
                max_value=500,
                value=100,
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
# EXTRACTION PROCESSING - FIXED
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
    enhanced_extractor = EnhancedSectionExtractor() if ENHANCED_EXTRACTOR_AVAILABLE else EnhancedSectionExtractor()
    llm_extractor = LLMRecommendationExtractor() if LLM_EXTRACTOR_AVAILABLE else None
    
    # Process each document
    for i, doc in enumerate(documents):
        doc_name = doc.get('filename', f'Document {i+1}')
        status_text.text(f"Processing {doc_name}...")
        progress_bar.progress((i + 1) / len(documents))
        
        try:
            # Get document content using fixed function
            content = get_document_content_for_extraction(doc)
            
            if not content or len(content.strip()) < 100:
                doc_results.append({
                    'document': doc_name,
                    'status': '⚠️ No suitable content found',
                    'recommendations_found': 0,
                    'responses_found': 0,
                    'individual_items_found': 0,
                    'document_type': detect_document_type(doc),
                    'content_length': len(content)
                })
                continue
            
            # Analyze document structure
            doc_structure = detect_inquiry_document_structure(content) if CORE_UTILS_AVAILABLE else {}
            doc_metadata = extract_government_document_metadata(content) if CORE_UTILS_AVAILABLE else {}
            
            recommendations = []
            responses = []
            individual_items = []
            
            # Extract based on selected method
            if method == "Enhanced Pattern-Based":
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
            recommendations = [r for r in recommendations if r.get('confidence', 0) >= confidence_threshold]
            responses = [r for r in responses if r.get('confidence', 0) >= confidence_threshold]
            
            # Limit number of items
            recommendations = recommendations[:max_items_per_doc]
            responses = responses[:max_items_per_doc]
            
            # Add document context
            for item in recommendations + responses:
                item['document_context'] = {
                    'filename': doc_name,
                    'document_type': detect_document_type(doc),
                    'document_structure': doc_structure,
                    'document_metadata': doc_metadata,
                    'content_length': len(content)
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
                'individual_items_found': len(recommendations + responses),
                'document_type': detect_document_type(doc),
                'content_length': len(content),
                'extraction_method': method
            })
            
        except Exception as e:
            logging.error(f"Error processing document {doc_name}: {e}")
            doc_results.append({
                'document': doc_name,
                'status': f'❌ Error: {str(e)}',
                'recommendations_found': 0,
                'responses_found': 0,
                'individual_items_found': 0,
                'document_type': 'error',
                'content_length': 0,
                'extraction_method': method
            })
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("Extraction complete!")
    
    # Store results in session state
    st.session_state.extraction_results = {
        'recommendations': all_recommendations,
        'responses': all_responses,
        'individual_items': all_individual_items,
        'document_results': doc_results,
        'extraction_timestamp': datetime.now().isoformat(),
        'extraction_method': method,
        'total_documents': len(documents),
        'successful_documents': len([r for r in doc_results if '✅' in r['status']]),
        'settings': {
            'confidence_threshold': confidence_threshold,
            'max_items_per_doc': max_items_per_doc,
            'include_context': include_context
        }
    }
    
    # Display results
    render_extraction_results()

# ===============================================
# RESULTS DISPLAY
# ===============================================

def render_extraction_results():
    """Render extraction results"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        return
    
    st.subheader("📊 Extraction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Processed", results.get('total_documents', 0))
    with col2:
        st.metric("Recommendations Found", len(results.get('recommendations', [])))
    with col3:
        st.metric("Responses Found", len(results.get('responses', [])))
    with col4:
        st.metric("Success Rate", f"{(results.get('successful_documents', 0) / max(results.get('total_documents', 1), 1) * 100):.1f}%")
    
    # Document results table
    if results.get('document_results'):
        st.write("**Document Processing Details:**")
        doc_results_df = pd.DataFrame(results['document_results'])
        st.dataframe(doc_results_df, use_container_width=True)
    
    # Show extracted items
    tab1, tab2 = st.tabs(["📝 Recommendations", "📋 Responses"])
    
    with tab1:
        recommendations = results.get('recommendations', [])
        if recommendations:
            show_detailed_items(recommendations, 'recommendation')
        else:
            st.info("No recommendations extracted")
    
    with tab2:
        responses = results.get('responses', [])
        if responses:
            show_detailed_items(responses, 'response')
        else:
            st.info("No responses extracted")

def show_detailed_items(items: List[Dict], item_type: str):
    """Show detailed view of all extracted items"""
    if not items:
        st.info(f"No {item_type}s to display")
        return
    
    st.write(f"**{len(items)} {item_type}s found:**")
    
    for i, item in enumerate(items):
        with st.expander(f"{item_type.title()} {i+1}: {item.get('text', '')[:80]}...", expanded=False):
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
                st.write(f"- **Confidence:** {item.get('confidence', 0):.3f}")
                st.write(f"- **Method:** {item.get('extraction_method', 'unknown')}")
                st.write(f"- **Length:** {len(item.get('text', ''))}")
                st.write(f"- **Words:** {len(item.get('text', '').split())}")
                
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
                structure = detect_inquiry_document_structure(content) if CORE_UTILS_AVAILABLE else {}
                metadata = extract_government_document_metadata(content) if CORE_UTILS_AVAILABLE else {}
                
                analysis_results.append({
                    'filename': doc.get('filename', 'Unknown'),
                    'document_type': detect_document_type(doc),
                    'content_length': len(content),
                    'estimated_recommendations': content.lower().count('recommendation'),
                    'estimated_responses': content.lower().count('response'),
                    'has_table_of_contents': 'contents' in content.lower(),
                    'processor_status': doc.get('processor_status', 'unknown')
                })
        
        if analysis_results:
            analysis_df = pd.DataFrame(analysis_results)
            st.dataframe(analysis_df, use_container_width=True)
        else:
            st.info("No documents available for analysis")

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def clear_extraction_results():
    """Clear extraction results from session state"""
    if 'extraction_results' in st.session_state:
        del st.session_state.extraction_results

def get_extraction_statistics() -> Dict[str, Any]:
    """Get extraction statistics"""
    results = st.session_state.get('extraction_results', {})
    
    return {
        'total_recommendations': len(results.get('recommendations', [])),
        'total_responses': len(results.get('responses', [])),
        'total_documents_processed': results.get('total_documents', 0),
        'successful_documents': results.get('successful_documents', 0),
        'extraction_method': results.get('extraction_method', 'unknown'),
        'extraction_timestamp': results.get('extraction_timestamp', ''),
        'settings': results.get('settings', {})
    }

def log_extraction_action(action: str, details: str = ""):
    """Log extraction-related user actions"""
    timestamp = datetime.now().isoformat()
    logging.info(f"[{timestamp}] Extraction Action: {action} - {details}")

def validate_extraction_settings(
    confidence_threshold: float,
    max_items_per_doc: int,
    extract_recommendations: bool,
    extract_responses: bool
) -> Tuple[bool, str]:
    """Validate extraction settings"""
    
    if not extract_recommendations and not extract_responses:
        return False, "Must select at least one extraction type"
    
    if confidence_threshold < 0 or confidence_threshold > 1:
        return False, "Confidence threshold must be between 0 and 1"
    
    if max_items_per_doc < 1 or max_items_per_doc > 1000:
        return False, "Max items per document must be between 1 and 1000"
    
    return True, "Settings are valid"

def debug_document_analysis(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Debug document structure for troubleshooting"""
    
    debug_info = {
        'filename': doc.get('filename', 'Unknown'),
        'available_fields': list(doc.keys()),
        'has_extraction_result': bool(doc.get('extraction_result')),
        'extraction_result_fields': [],
        'content_sources': [],
        'content_lengths': {},
        'document_type': detect_document_type(doc)
    }
    
    # Check extraction result structure
    if doc.get('extraction_result'):
        extraction_result = doc['extraction_result']
        debug_info['extraction_result_fields'] = list(extraction_result.keys())
        
        # Check for text in various fields
        for field in ['text', 'content', 'raw_text']:
            if extraction_result.get(field):
                debug_info['content_sources'].append(f"extraction_result.{field}")
                debug_info['content_lengths'][f"extraction_result.{field}"] = len(extraction_result[field])
    
    # Check for direct text fields
    for field in ['text', 'content', 'raw_text', 'document_text']:
        if doc.get(field):
            debug_info['content_sources'].append(field)
            debug_info['content_lengths'][field] = len(doc[field])
    
    # Get final content
    final_content = get_document_content_for_extraction(doc)
    debug_info['final_content_length'] = len(final_content)
    debug_info['final_content_preview'] = final_content[:200] + "..." if len(final_content) > 200 else final_content
    
    return debug_info

# ===============================================
# EXPORT FUNCTIONS
# ===============================================

def export_extraction_results_csv() -> str:
    """Export extraction results as CSV"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        return ""
    
    # Combine recommendations and responses
    all_items = []
    
    for rec in results.get('recommendations', []):
        all_items.append({
            'type': 'recommendation',
            'number': rec.get('number', ''),
            'text': rec.get('text', ''),
            'confidence': rec.get('confidence', 0),
            'extraction_method': rec.get('extraction_method', ''),
            'document': rec.get('document_context', {}).get('filename', ''),
            'document_type': rec.get('document_context', {}).get('document_type', ''),
            'line_number': rec.get('line_number', ''),
            'word_count': len(rec.get('text', '').split())
        })
    
    for resp in results.get('responses', []):
        all_items.append({
            'type': 'response',
            'number': resp.get('number', ''),
            'text': resp.get('text', ''),
            'confidence': resp.get('confidence', 0),
            'extraction_method': resp.get('extraction_method', ''),
            'document': resp.get('document_context', {}).get('filename', ''),
            'document_type': resp.get('document_context', {}).get('document_type', ''),
            'line_number': resp.get('line_number', ''),
            'word_count': len(resp.get('text', '').split())
        })
    
    if all_items:
        df = pd.DataFrame(all_items)
        return df.to_csv(index=False)
    
    return ""

def export_extraction_results_json() -> str:
    """Export extraction results as JSON"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        return "{}"
    
    # Create export structure
    export_data = {
        'extraction_metadata': {
            'timestamp': results.get('extraction_timestamp', ''),
            'method': results.get('extraction_method', ''),
            'total_documents': results.get('total_documents', 0),
            'successful_documents': results.get('successful_documents', 0),
            'settings': results.get('settings', {})
        },
        'recommendations': results.get('recommendations', []),
        'responses': results.get('responses', []),
        'document_results': results.get('document_results', [])
    }
    
    import json
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def render_extraction_export_options():
    """Render export options for extraction results"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        st.info("No extraction results to export")
        return
    
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download as CSV"):
            csv_data = export_extraction_results_csv()
            if csv_data:
                st.download_button(
                    label="💾 Download CSV File",
                    data=csv_data,
                    file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📋 Download as JSON"):
            json_data = export_extraction_results_json()
            if json_data:
                st.download_button(
                    label="💾 Download JSON File",
                    data=json_data,
                    file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col3:
        if st.button("📄 View Summary Report"):
            render_extraction_summary_report()

def render_extraction_summary_report():
    """Render a summary report of extraction results"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        return
    
    st.subheader("📈 Extraction Summary Report")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Recommendations", len(results.get('recommendations', [])))
        st.metric("Total Responses", len(results.get('responses', [])))
    
    with col2:
        st.metric("Documents Processed", results.get('total_documents', 0))
        st.metric("Success Rate", f"{(results.get('successful_documents', 0) / max(results.get('total_documents', 1), 1) * 100):.1f}%")
    
    with col3:
        avg_confidence_recs = 0
        recs = results.get('recommendations', [])
        if recs:
            avg_confidence_recs = sum(r.get('confidence', 0) for r in recs) / len(recs)
        
        avg_confidence_resps = 0
        resps = results.get('responses', [])
        if resps:
            avg_confidence_resps = sum(r.get('confidence', 0) for r in resps) / len(resps)
        
        st.metric("Avg Confidence (Recs)", f"{avg_confidence_recs:.3f}")
        st.metric("Avg Confidence (Resps)", f"{avg_confidence_resps:.3f}")
    
    # Document breakdown
    if results.get('document_results'):
        st.write("**Document Processing Details:**")
        doc_df = pd.DataFrame(results['document_results'])
        st.dataframe(doc_df, use_container_width=True)
    
    # Settings used
    st.write("**Extraction Settings:**")
    settings = results.get('settings', {})
    for key, value in settings.items():
        st.write(f"- **{key.replace('_', ' ').title()}:** {value}")

# ===============================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# ===============================================

def show_recommendations_compact(recommendations: List[Dict]):
    """Backwards compatibility function for showing recommendations"""
    return show_detailed_items(recommendations, 'recommendation')

def get_recommendations_for_annotation() -> List[Dict]:
    """Get extracted recommendations for annotation tab"""
    results = st.session_state.get('extraction_results', {})
    return results.get('recommendations', [])

# ===============================================
# MODULE EXPORTS
# ===============================================

__all__ = [
    # Main functions
    'render_extraction_tab',
    'render_extraction_interface',
    'render_extraction_results',
    'process_document_extraction',
    
    # Data access functions (FIXED)
    'get_document_content_for_extraction',
    'detect_document_type',
    'validate_documents_for_extraction',
    
    # Display functions
    'show_detailed_items',
    'render_document_analysis',
    'render_extraction_summary_report',
    
    # Export functions
    'render_extraction_export_options',
    'export_extraction_results_csv',
    'export_extraction_results_json',
    
    # Utility functions
    'log_extraction_action',
    'get_extraction_statistics',
    'clear_extraction_results',
    'validate_extraction_settings',
    'debug_document_analysis',
    
    # Backwards compatibility
    'show_recommendations_compact',
    'get_recommendations_for_annotation',
    
    # Availability flags
    'ENHANCED_EXTRACTOR_AVAILABLE',
    'LLM_EXTRACTOR_AVAILABLE',
    'CORE_UTILS_AVAILABLE'
]

# ===============================================
# MODULE INITIALIZATION LOG
# ===============================================

if ENHANCED_EXTRACTOR_AVAILABLE and LLM_EXTRACTOR_AVAILABLE:
    logging.info("✅ Extraction components loaded with full functionality (Enhanced + AI)")
elif ENHANCED_EXTRACTOR_AVAILABLE:
    logging.info("✅ Extraction components loaded with enhanced pattern-based extraction")
elif LLM_EXTRACTOR_AVAILABLE:
    logging.info("✅ Extraction components loaded with AI-powered extraction only")
else:
    logging.info("✅ Extraction components loaded with fallback functionality")

logging.info("🎉 FIXED Extraction components module ready - data compatibility issues resolved!")
