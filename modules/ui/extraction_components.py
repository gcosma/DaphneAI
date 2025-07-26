# ===============================================
# EXTRACTION COMPONENTS MODULE - ENHANCED WITH SMART EXTRACTION
# Complete replacement that includes both standard and smart extraction
# 
# This module replaces the existing extraction_components.py with:
# - Fixed data compatibility (reads from upload tab properly)
# - Smart complete extraction (captures full recommendations)
# - Advanced download capabilities
# - Fallback support for missing dependencies
#
# Key Features:
# - Multi-line content extraction
# - Context-aware boundary detection  
# - Smart paragraph reconstruction
# - Enhanced download capabilities with multiple formats
# - Quality scoring and analysis
# - Backwards compatibility with existing code
#
# Author: Recommendation-Response Tracker Team
# Version: 3.0 - Enhanced with Smart Extraction
# Last Updated: 2025
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from pathlib import Path
import io

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import dependencies with fallbacks
ENHANCED_EXTRACTOR_AVAILABLE = False
LLM_EXTRACTOR_AVAILABLE = False
CORE_UTILS_AVAILABLE = False

try:
    from modules.enhanced_section_extractor import EnhancedSectionExtractor
    ENHANCED_EXTRACTOR_AVAILABLE = True
    logging.info("✅ Enhanced section extractor imported successfully")
except ImportError as import_error:
    logging.warning(f"⚠️ Enhanced section extractor not available: {import_error}")

try:
    from modules.llm_extractor import LLMRecommendationExtractor
    LLM_EXTRACTOR_AVAILABLE = True
    logging.info("✅ LLM extractor imported successfully")
except ImportError as import_error:
    logging.warning(f"⚠️ LLM extractor not available: {import_error}")

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
    
    # Fallback implementations
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
# SMART EXTRACTOR CLASS (INTEGRATED)
# ===============================================

class SmartExtractor:
    """
    Intelligent extractor that captures complete recommendations and responses
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced patterns for capturing complete content
        self.recommendation_start_patterns = [
            r'(?i)^\s*recommendation\s+(\d+(?:\.\d+)*)\s*[:\.\-]?\s*(.*)$',
            r'(?i)^\s*(\d+(?:\.\d+)*)\.\s+(.*)$',
            r'(?i)^\s*(\d+(?:\.\d+)*)\)\s+(.*)$',
            r'(?i)^\s*that\s+(.+)$',
            r'(?i)^\s*we\s+recommend\s+that\s+(.+)$',
            r'(?i)^\s*it\s+is\s+recommended\s+that\s+(.+)$',
            r'(?i)^\s*the\s+(?:inquiry|committee|panel)\s+recommends?\s+that\s+(.+)$'
        ]
        
        self.response_start_patterns = [
            r'(?i)^\s*response\s+to\s+recommendation\s+(\d+(?:\.\d+)*)\s*[:\.\-]?\s*(.*)$',
            r'(?i)^\s*recommendation\s+(\d+(?:\.\d+)*)\s*[:\.\-]?\s*response\s*[:\.\-]?\s*(.*)$',
            r'(?i)^\s*government\s+response\s*[:\.\-]?\s*(.*)$',
            r'(?i)^\s*(accepted|not\s+accepted|partially\s+accepted|rejected)\s*[:\.\-]?\s*(.*)$',
            r'(?i)^\s*the\s+government\s+(accepts?|rejects?|acknowledges?)\s+(.+)$'
        ]
        
        # Patterns that indicate content continuation
        self.continuation_patterns = [
            r'^\s*[a-z]',
            r'^\s*(?:and|or|but|however|furthermore|additionally|moreover)',
            r'^\s*(?:this|that|these|those|such|it)',
            r'^\s*(?:the|a|an)\s+\w+\s+(?:will|shall|should|must|may)',
            r'^\s*(?:in|on|at|by|for|with|through|during|following)'
        ]
        
        # Patterns that indicate end of content
        self.end_patterns = [
            r'(?i)^\s*recommendation\s+\d+',
            r'(?i)^\s*response\s+to\s+recommendation',
            r'(?i)^\s*\d+\.\s+',
            r'(?i)^\s*(?:appendix|annex|bibliography|references)',
            r'(?i)^\s*(?:signed|dated|chair\s+of)',
            r'(?i)^\s*(?:minister|secretary\s+of\s+state)',
            r'(?i)^\s*(?:next\s+steps?|conclusion|summary)',
            r'^\s*$'
        ]

    def extract_complete_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete recommendations with full context"""
        recommendations = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            rec_match = self._find_recommendation_start(line)
            
            if rec_match:
                rec_number, initial_content = rec_match
                complete_content, end_line = self._collect_complete_content(
                    lines, i, initial_content, 'recommendation'
                )
                
                if complete_content and len(complete_content.strip()) > 20:
                    confidence = self._calculate_content_confidence(complete_content, 'recommendation')
                    
                    recommendations.append({
                        'number': rec_number,
                        'text': complete_content.strip(),
                        'start_line': i,
                        'end_line': end_line,
                        'confidence': confidence,
                        'word_count': len(complete_content.split()),
                        'char_count': len(complete_content),
                        'extraction_method': 'smart_complete',
                        'content_type': 'recommendation',
                        'quality_score': self._assess_content_quality(complete_content),
                        'context_lines': min(5, end_line - i),
                        'extracted_at': datetime.now().isoformat()
                    })
                
                i = end_line
            else:
                i += 1
        
        return recommendations

    def extract_complete_responses(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete responses with full context"""
        responses = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            resp_match = self._find_response_start(line)
            
            if resp_match:
                resp_number, initial_content, response_type = resp_match
                complete_content, end_line = self._collect_complete_content(
                    lines, i, initial_content, 'response'
                )
                
                if complete_content and len(complete_content.strip()) > 20:
                    confidence = self._calculate_content_confidence(complete_content, 'response')
                    
                    responses.append({
                        'number': resp_number,
                        'text': complete_content.strip(),
                        'start_line': i,
                        'end_line': end_line,
                        'confidence': confidence,
                        'word_count': len(complete_content.split()),
                        'char_count': len(complete_content),
                        'extraction_method': 'smart_complete',
                        'content_type': 'response',
                        'response_type': response_type,
                        'quality_score': self._assess_content_quality(complete_content),
                        'context_lines': min(5, end_line - i),
                        'extracted_at': datetime.now().isoformat()
                    })
                
                i = end_line
            else:
                i += 1
        
        return responses

    def _find_recommendation_start(self, line: str) -> Optional[Tuple[str, str]]:
        """Find if a line starts a recommendation"""
        for pattern in self.recommendation_start_patterns:
            match = re.match(pattern, line)
            if match:
                if len(match.groups()) >= 2:
                    return match.group(1), match.group(2)
                else:
                    return "auto", match.group(1)
        return None

    def _find_response_start(self, line: str) -> Optional[Tuple[str, str, str]]:
        """Find if a line starts a response"""
        for pattern in self.response_start_patterns:
            match = re.match(pattern, line)
            if match:
                if 'accepted' in line.lower():
                    response_type = 'accepted'
                elif 'rejected' in line.lower() or 'not accepted' in line.lower():
                    response_type = 'rejected'
                elif 'partially' in line.lower():
                    response_type = 'partially_accepted'
                else:
                    response_type = 'government_response'
                
                if len(match.groups()) >= 2:
                    return match.group(1), match.group(2), response_type
                else:
                    return "auto", match.group(1), response_type
        return None

    def _collect_complete_content(self, lines: List[str], start_idx: int, initial_content: str, content_type: str) -> Tuple[str, int]:
        """Collect complete multi-line content"""
        content_parts = [initial_content] if initial_content.strip() else []
        current_idx = start_idx + 1
        consecutive_empty_lines = 0
        max_empty_lines = 2 if content_type == 'recommendation' else 1
        
        while current_idx < len(lines):
            line = lines[current_idx].strip()
            
            if not line:
                consecutive_empty_lines += 1
                if consecutive_empty_lines >= max_empty_lines:
                    break
                current_idx += 1
                continue
            else:
                consecutive_empty_lines = 0
            
            if self._is_content_end(line, content_type):
                break
            
            if self._is_content_continuation(line, content_parts):
                content_parts.append(line)
                current_idx += 1
                continue
            
            if self._looks_like_new_item(line):
                break
            
            if len(' '.join(content_parts)) > 100 and not self._clearly_continues(line):
                break
            
            content_parts.append(line)
            current_idx += 1
        
        return ' '.join(content_parts), current_idx

    def _is_content_end(self, line: str, content_type: str) -> bool:
        """Check if line indicates end of content"""
        for pattern in self.end_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _is_content_continuation(self, line: str, existing_parts: List[str]) -> bool:
        """Check if line continues existing content"""
        if not existing_parts:
            return True
        
        for pattern in self.continuation_patterns:
            if re.match(pattern, line):
                return True
        
        if existing_parts and line and line[0].islower():
            last_part = existing_parts[-1].strip()
            if last_part and not last_part.endswith('.'):
                return True
        
        return False

    def _looks_like_new_item(self, line: str) -> bool:
        """Check if line looks like start of new item"""
        new_item_patterns = [
            r'(?i)^\s*recommendation\s+\d+',
            r'(?i)^\s*response\s+to\s+recommendation',
            r'(?i)^\s*\d+\.\s+[A-Z]',
            r'(?i)^\s*that\s+[A-Z]',
        ]
        
        for pattern in new_item_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _clearly_continues(self, line: str) -> bool:
        """Check if line clearly continues previous content"""
        continuation_indicators = [
            r'^\s*(?:and|or|but|however|therefore|thus|consequently)',
            r'^\s*(?:furthermore|additionally|moreover|also)',
            r'^\s*(?:this|that|these|those|such)',
            r'^\s*(?:in particular|for example|specifically)',
            r'^\s*[a-z]'
        ]
        
        for pattern in continuation_indicators:
            if re.match(pattern, line):
                return True
        return False

    def _calculate_content_confidence(self, content: str, content_type: str) -> float:
        """Calculate confidence score"""
        confidence = 0.5
        word_count = len(content.split())
        
        if word_count >= 10:
            confidence += 0.2
        if word_count >= 20:
            confidence += 0.1
        if word_count >= 50:
            confidence += 0.1
        
        if content_type == 'recommendation':
            if re.search(r'(?i)\b(?:should|must|shall|recommend|propose)\b', content):
                confidence += 0.2
        elif content_type == 'response':
            if re.search(r'(?i)\b(?:accept|reject|implement|consider|agree)\b', content):
                confidence += 0.2
        
        if content.count('.') >= 2:
            confidence += 0.1
        
        return min(1.0, confidence)

    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality"""
        quality = 0.5
        
        if content.strip().endswith('.'):
            quality += 0.2
        
        sentences = content.split('.')
        if len(sentences) >= 2:
            quality += 0.1
        
        words = content.split()
        if len(words) >= 15:
            quality += 0.1
        
        if content and content[0].isupper():
            quality += 0.1
        
        return min(1.0, quality)

# ===============================================
# FIXED DATA ACCESS FUNCTIONS (from previous fix)
# ===============================================

def get_document_content_for_extraction(doc: Dict[str, Any]) -> str:
    """Get document content from uploaded document data structure"""
    try:
        # Method 1: Check direct fields
        if doc.get('text'):
            return doc['text']
        if doc.get('content'):
            return doc['content']
        
        # Method 2: Check extraction_result (upload tab format)
        extraction_result = doc.get('extraction_result', {})
        if extraction_result:
            if extraction_result.get('text'):
                return extraction_result['text']
            if extraction_result.get('content'):
                return extraction_result['content']
        
        return ""
        
    except Exception as e:
        logging.error(f"Error extracting document content: {e}")
        return ""

def detect_document_type(doc: Dict[str, Any]) -> str:
    """Detect document type from uploaded document data"""
    try:
        # Check metadata
        metadata = doc.get('metadata', {})
        if metadata.get('document_type'):
            return metadata['document_type']
        
        extraction_result = doc.get('extraction_result', {})
        if extraction_result.get('metadata', {}).get('document_type'):
            return extraction_result['metadata']['document_type']
        
        # Analyze content
        content = get_document_content_for_extraction(doc)
        if not content:
            return 'no_content'
        
        content_lower = content.lower()
        
        if re.search(r'(?i)government\s+response', content):
            return 'government_response'
        elif re.search(r'(?i)(?:inquiry\s+)?report', content) and re.search(r'(?i)recommendations?', content):
            return 'inquiry_report'
        elif 'recommendation' in content_lower:
            return 'document_with_recommendations'
        elif 'response' in content_lower:
            return 'document_with_responses'
        
        return 'government_document'
        
    except Exception as e:
        logging.error(f"Error detecting document type: {e}")
        return 'unknown'

def validate_documents_for_extraction() -> Tuple[bool, str]:
    """Validate that documents are ready for extraction"""
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        return False, "No documents uploaded"
    
    content_available = 0
    total_text_length = 0
    
    for doc in docs:
        content = get_document_content_for_extraction(doc)
        if content and len(content.strip()) > 50:
            content_available += 1
            total_text_length += len(content)
    
    if content_available == 0:
        return False, "No documents have readable content"
    
    avg_length = total_text_length // content_available if content_available > 0 else 0
    return True, f"Ready: {content_available} documents with content, avg {avg_length:,} chars"

# ===============================================
# MAIN EXTRACTION TAB FUNCTION
# ===============================================

def render_extraction_tab():
    """Main extraction tab with both standard and smart extraction"""
    st.header("🔍 Recommendation & Response Extraction")
    
    st.markdown("""
    Extract **recommendations** from inquiry reports and **government responses** from response documents.
    Choose between standard pattern-based extraction or smart complete extraction.
    """)
    
    # Check if documents are uploaded
    if not st.session_state.get('uploaded_documents', []):
        st.info("📁 Please upload documents first in the Upload tab.")
        return
    
    # Validate documents
    is_valid, validation_message = validate_documents_for_extraction()
    
    if not is_valid:
        st.error(f"❌ {validation_message}")
        return
    
    st.success(f"✅ {validation_message}")
    
    # Extraction method selection
    extraction_type = st.radio(
        "Choose extraction method:",
        [
            "🧠 Smart Complete Extraction (Recommended)", 
            "⚡ Standard Pattern Extraction",
            "🤖 AI-Powered Extraction" + (" (Available)" if LLM_EXTRACTOR_AVAILABLE else " (Unavailable)")
        ],
        index=0,
        help="Smart extraction captures complete recommendations and responses, not fragments"
    )
    
    if extraction_type.startswith("🧠 Smart"):
        render_smart_extraction_interface()
    elif extraction_type.startswith("⚡ Standard"):
        render_standard_extraction_interface()
    elif extraction_type.startswith("🤖 AI-Powered"):
        if LLM_EXTRACTOR_AVAILABLE:
            render_ai_extraction_interface()
        else:
            st.error("AI-powered extraction requires OpenAI API key and LLM extractor module")

# ===============================================
# SMART EXTRACTION INTERFACE
# ===============================================

def render_smart_extraction_interface():
    """Render smart extraction interface"""
    st.subheader("🧠 Smart Complete Extraction")
    
    st.markdown("""
    **Captures complete recommendations and responses, not just fragments.**
    
    ✅ Multi-line content extraction  
    ✅ Context-aware boundary detection  
    ✅ Complete paragraph reconstruction  
    ✅ Enhanced download formats  
    """)
    
    # Document selection
    docs = st.session_state.get('uploaded_documents', [])
    doc_options = [f"{doc.get('filename', 'Unknown')}" for doc in docs]
    
    selected_docs = st.multiselect(
        "Select documents for smart extraction:",
        options=doc_options,
        default=doc_options,
        help="Choose documents to process with smart extraction"
    )
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        extract_recommendations = st.checkbox("Extract Complete Recommendations", value=True)
        extract_responses = st.checkbox("Extract Complete Responses", value=True)
    
    with col2:
        min_word_count = st.slider("Minimum words per item:", 5, 50, 15)
        quality_threshold = st.slider("Quality threshold:", 0.0, 1.0, 0.5, 0.1)
    
    # Process button
    if st.button("🚀 Start Smart Extraction", type="primary", disabled=not selected_docs):
        if not extract_recommendations and not extract_responses:
            st.error("Please select at least one extraction type")
        else:
            process_smart_extraction(
                selected_docs, docs, extract_recommendations, 
                extract_responses, min_word_count, quality_threshold
            )

def process_smart_extraction(
    selected_docs: List[str], 
    all_docs: List[Dict], 
    extract_recommendations: bool,
    extract_responses: bool, 
    min_word_count: int, 
    quality_threshold: float
):
    """Process documents with smart extraction"""
    
    extractor = SmartExtractor()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_recommendations = []
    all_responses = []
    processing_results = []
    
    # Get selected document objects
    selected_doc_objects = [doc for doc in all_docs if doc.get('filename') in selected_docs]
    
    for i, doc in enumerate(selected_doc_objects):
        filename = doc.get('filename', f'Document {i+1}')
        status_text.text(f"Smart extracting from {filename}...")
        progress_bar.progress((i + 1) / len(selected_doc_objects))
        
        try:
            content = get_document_content_for_extraction(doc)
            
            if not content or len(content.strip()) < 100:
                processing_results.append({
                    'filename': filename,
                    'status': '⚠️ Insufficient content',
                    'recommendations': 0,
                    'responses': 0
                })
                continue
            
            # Smart extraction
            doc_recommendations = []
            if extract_recommendations:
                doc_recommendations = extractor.extract_complete_recommendations(content)
                doc_recommendations = [
                    rec for rec in doc_recommendations 
                    if rec.get('word_count', 0) >= min_word_count 
                    and rec.get('quality_score', 0) >= quality_threshold
                ]
                for rec in doc_recommendations:
                    rec['document_context'] = {'filename': filename}
            
            doc_responses = []
            if extract_responses:
                doc_responses = extractor.extract_complete_responses(content)
                doc_responses = [
                    resp for resp in doc_responses 
                    if resp.get('word_count', 0) >= min_word_count 
                    and resp.get('quality_score', 0) >= quality_threshold
                ]
                for resp in doc_responses:
                    resp['document_context'] = {'filename': filename}
            
            all_recommendations.extend(doc_recommendations)
            all_responses.extend(doc_responses)
            
            total_words = sum(item.get('word_count', 0) for item in doc_recommendations + doc_responses)
            avg_quality = sum(item.get('quality_score', 0) for item in doc_recommendations + doc_responses) / max(len(doc_recommendations + doc_responses), 1)
            
            processing_results.append({
                'filename': filename,
                'status': '✅ Success',
                'recommendations': len(doc_recommendations),
                'responses': len(doc_responses),
                'total_words': total_words,
                'avg_quality': f"{avg_quality:.3f}"
            })
            
        except Exception as e:
            processing_results.append({
                'filename': filename,
                'status': f'❌ Error: {str(e)}',
                'recommendations': 0,
                'responses': 0,
                'total_words': 0,
                'avg_quality': '0.000'
            })
    
    # Store results
    st.session_state.extraction_results = {
        'recommendations': all_recommendations,
        'responses': all_responses,
        'processing_results': processing_results,
        'extraction_method': 'smart_complete',
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'min_word_count': min_word_count,
            'quality_threshold': quality_threshold
        }
    }
    
    # Display results
    render_extraction_results()

# ===============================================
# STANDARD EXTRACTION INTERFACE (simplified)
# ===============================================

def render_standard_extraction_interface():
    """Render standard pattern-based extraction interface"""
    st.subheader("⚡ Standard Pattern Extraction")
    st.info("💡 For better results, try Smart Complete Extraction above")
    
    # Simple interface for standard extraction
    docs = st.session_state.get('uploaded_documents', [])
    
    if st.button("🔍 Run Standard Extraction"):
        # Use existing enhanced extractor if available
        if ENHANCED_EXTRACTOR_AVAILABLE:
            standard_extractor = EnhancedSectionExtractor()
        else:
            st.error("Enhanced extractor not available")
            return
        
        all_recommendations = []
        all_responses = []
        
        for doc in docs:
            content = get_document_content_for_extraction(doc)
            if content:
                recommendations = standard_extractor.extract_individual_recommendations(content)
                responses = standard_extractor.extract_individual_responses(content)
                
                for rec in recommendations:
                    rec['document_context'] = {'filename': doc.get('filename', 'Unknown')}
                for resp in responses:
                    resp['document_context'] = {'filename': doc.get('filename', 'Unknown')}
                
                all_recommendations.extend(recommendations)
                all_responses.extend(responses)
        
        st.session_state.extraction_results = {
            'recommendations': all_recommendations,
            'responses': all_responses,
            'extraction_method': 'standard_pattern',
            'timestamp': datetime.now().isoformat()
        }
        
        render_extraction_results()

def render_ai_extraction_interface():
    """Render AI-powered extraction interface"""
    st.subheader("🤖 AI-Powered Extraction")
    st.warning("⚠️ Requires OpenAI API key and may incur costs")
    
    if st.button("🤖 Run AI Extraction"):
        st.info("AI extraction would be implemented here with LLM extractor")

# ===============================================
# RESULTS DISPLAY WITH DOWNLOADS
# ===============================================

def render_extraction_results():
    """Render extraction results with enhanced download options"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        return
    
    st.subheader("📊 Extraction Results")
    
    recommendations = results.get('recommendations', [])
    responses = results.get('responses', [])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recommendations", len(recommendations))
    with col2:
        st.metric("Responses", len(responses))
    with col3:
        total_words = sum(item.get('word_count', 0) for item in recommendations + responses)
        st.metric("Total Words", f"{total_words:,}")
    with col4:
        avg_quality = sum(item.get('quality_score', 0) for item in recommendations + responses) / max(len(recommendations + responses), 1)
        st.metric("Avg Quality", f"{avg_quality:.3f}")
    
    # Processing results if available
    if results.get('processing_results'):
        st.write("**Document Processing Summary:**")
        results_df = pd.DataFrame(results['processing_results'])
        st.dataframe(results_df, use_container_width=True)
    
    # Enhanced Download Section
    if recommendations or responses:
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download CSV"):
                csv_data = create_comprehensive_csv(recommendations, responses)
                if csv_data:
                    st.download_button(
                        label="💾 Download CSV File",
                        data=csv_data,
                        file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📋 Download JSON"):
                json_data = create_detailed_json(recommendations, responses, results)
                if json_data:
                    st.download_button(
                        label="💾 Download JSON File",
                        data=json_data,
                        file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col3:
            if st.button("📄 Download Report"):
                report_data = create_formatted_report(recommendations, responses)
                if report_data:
                    st.download_button(
                        label="💾 Download Report",
                        data=report_data,
                        file_name=f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        # Preview results
        st.subheader("🔍 Content Preview")
        
        preview_tabs = st.tabs(["📝 Recommendations", "📋 Responses", "📊 Quality Analysis"])
        
        with preview_tabs[0]:
            if recommendations:
                render_items_preview(recommendations, "Recommendation")
            else:
                st.info("No recommendations extracted")
        
        with preview_tabs[1]:
            if responses:
                render_items_preview(responses, "Response")
            else:
                st.info("No responses extracted")
        
        with preview_tabs[2]:
            render_quality_analysis(recommendations + responses)

def render_items_preview(items: List[Dict], item_type: str):
    """Render preview of extracted items"""
    
    # Sort by quality score if available
    if items and items[0].get('quality_score') is not None:
        sorted_items = sorted(items, key=lambda x: x.get('quality_score', 0), reverse=True)
        st.write(f"**{len(items)} {item_type}s Found (sorted by quality, showing top 10)**")
    else:
        sorted_items = items
        st.write(f"**{len(items)} {item_type}s Found (showing first 10)**")
    
    for i, item in enumerate(sorted_items[:10]):
        # Quality indicator
        if item.get('quality_score') is not None:
            quality = item.get('quality_score', 0)
            quality_color = "🟢" if quality > 0.8 else "🟡" if quality > 0.6 else "🔴"
            quality_text = f" (Q: {quality:.3f})"
        else:
            quality_color = "📄"
            quality_text = ""
        
        # Preview text
        preview_text = item.get('text', '')[:80] + "..." if len(item.get('text', '')) > 80 else item.get('text', '')
        
        with st.expander(f"{quality_color} {item_type} {item.get('number', i+1)}: {preview_text}{quality_text}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Complete Text:**")
                st.write(item.get('text', 'No content'))
                
                if item.get('response_type'):
                    response_emoji = {
                        'accepted': '✅ Accepted',
                        'rejected': '❌ Rejected', 
                        'partially_accepted': '⚡ Partially Accepted',
                        'government_response': '📋 Government Response'
                    }.get(item.get('response_type'), '❓ Unknown')
                    st.write(f"**Response Type:** {response_emoji}")
            
            with col2:
                st.write("**Metrics:**")
                if item.get('quality_score') is not None:
                    st.write(f"📊 Quality: {item.get('quality_score', 0):.3f}")
                if item.get('confidence') is not None:
                    st.write(f"🎯 Confidence: {item.get('confidence', 0):.3f}")
                if item.get('word_count') is not None:
                    st.write(f"📝 Words: {item.get('word_count', 0)}")
                if item.get('extraction_method'):
                    st.write(f"🔧 Method: {item.get('extraction_method', 'unknown')}")
                
                if item.get('document_context'):
                    st.write(f"📄 Source: {item['document_context'].get('filename', 'Unknown')}")

def render_quality_analysis(items: List[Dict]):
    """Render quality analysis of extracted items"""
    
    if not items:
        st.info("No items to analyze")
        return
    
    # Check if quality scores are available
    has_quality_scores = any(item.get('quality_score') is not None for item in items)
    has_word_counts = any(item.get('word_count') is not None for item in items)
    
    if has_quality_scores or has_word_counts:
        col1, col2 = st.columns(2)
        
        with col1:
            if has_quality_scores:
                st.write("**Quality Distribution:**")
                qualities = [item.get('quality_score', 0) for item in items if item.get('quality_score') is not None]
                if qualities:
                    high_quality = len([q for q in qualities if q >= 0.8])
                    medium_quality = len([q for q in qualities if 0.6 <= q < 0.8])
                    low_quality = len([q for q in qualities if q < 0.6])
                    
                    st.write(f"🟢 High Quality (≥0.8): {high_quality}")
                    st.write(f"🟡 Medium Quality (0.6-0.8): {medium_quality}")
                    st.write(f"🔴 Low Quality (<0.6): {low_quality}")
            
            if has_word_counts:
                st.write("**Content Length Distribution:**")
                word_counts = [item.get('word_count', 0) for item in items if item.get('word_count') is not None]
                if word_counts:
                    short_items = len([w for w in word_counts if w < 20])
                    medium_items = len([w for w in word_counts if 20 <= w < 50])
                    long_items = len([w for w in word_counts if w >= 50])
                    
                    st.write(f"📝 Short (<20 words): {short_items}")
                    st.write(f"📄 Medium (20-50 words): {medium_items}")
                    st.write(f"📚 Long (≥50 words): {long_items}")
        
        with col2:
            st.write("**Summary Statistics:**")
            if has_quality_scores:
                qualities = [item.get('quality_score', 0) for item in items if item.get('quality_score') is not None]
                if qualities:
                    st.write(f"Average Quality: {sum(qualities)/len(qualities):.3f}")
            
            if has_word_counts:
                word_counts = [item.get('word_count', 0) for item in items if item.get('word_count') is not None]
                if word_counts:
                    st.write(f"Average Word Count: {sum(word_counts)/len(word_counts):.1f}")
                    st.write(f"Total Words: {sum(word_counts):,}")
            
            # Show best items
            if has_quality_scores:
                st.write("**Highest Quality Items:**")
                best_items = sorted([item for item in items if item.get('quality_score') is not None], 
                                  key=lambda x: x.get('quality_score', 0), reverse=True)[:3]
                for i, item in enumerate(best_items, 1):
                    preview = item.get('text', '')[:40] + "..." if len(item.get('text', '')) > 40 else item.get('text', '')
                    st.write(f"{i}. {preview} (Q: {item.get('quality_score', 0):.3f})")
    else:
        st.write("**Basic Statistics:**")
        st.write(f"Total Items: {len(items)}")
        st.write(f"Items with Text: {len([item for item in items if item.get('text')])}")

# ===============================================
# DOWNLOAD FUNCTIONS
# ===============================================

def create_comprehensive_csv(recommendations: List[Dict], responses: List[Dict]) -> str:
    """Create comprehensive CSV with all extracted content"""
    all_items = []
    
    # Process recommendations
    for i, rec in enumerate(recommendations):
        all_items.append({
            'Item_ID': f"REC_{i+1:03d}",
            'Type': 'Recommendation',
            'Number': rec.get('number', f'Auto_{i+1}'),
            'Full_Text': rec.get('text', ''),
            'Word_Count': rec.get('word_count', len(rec.get('text', '').split())),
            'Character_Count': rec.get('char_count', len(rec.get('text', ''))),
            'Confidence_Score': rec.get('confidence', 0),
            'Quality_Score': rec.get('quality_score', 0),
            'Extraction_Method': rec.get('extraction_method', 'unknown'),
            'Context_Lines': rec.get('context_lines', 0),
            'Start_Line': rec.get('start_line', 0),
            'End_Line': rec.get('end_line', 0),
            'Document_Source': rec.get('document_context', {}).get('filename', 'Unknown'),
            'Extracted_At': rec.get('extracted_at', datetime.now().isoformat()),
            'Content_Preview': rec.get('text', '')[:100] + '...' if len(rec.get('text', '')) > 100 else rec.get('text', '')
        })
    
    # Process responses
    for i, resp in enumerate(responses):
        all_items.append({
            'Item_ID': f"RESP_{i+1:03d}",
            'Type': 'Response',
            'Number': resp.get('number', f'Auto_{i+1}'),
            'Full_Text': resp.get('text', ''),
            'Word_Count': resp.get('word_count', len(resp.get('text', '').split())),
            'Character_Count': resp.get('char_count', len(resp.get('text', ''))),
            'Confidence_Score': resp.get('confidence', 0),
            'Quality_Score': resp.get('quality_score', 0),
            'Extraction_Method': resp.get('extraction_method', 'unknown'),
            'Context_Lines': resp.get('context_lines', 0),
            'Start_Line': resp.get('start_line', 0),
            'End_Line': resp.get('end_line', 0),
            'Document_Source': resp.get('document_context', {}).get('filename', 'Unknown'),
            'Response_Type': resp.get('response_type', 'unknown'),
            'Extracted_At': resp.get('extracted_at', datetime.now().isoformat()),
            'Content_Preview': resp.get('text', '')[:100] + '...' if len(resp.get('text', '')) > 100 else resp.get('text', '')
        })
    
    if all_items:
        df = pd.DataFrame(all_items)
        return df.to_csv(index=False)
    return ""

def create_detailed_json(recommendations: List[Dict], responses: List[Dict], metadata: Dict) -> str:
    """Create detailed JSON export"""
    export_data = {
        'extraction_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'total_responses': len(responses),
            'extraction_method': metadata.get('extraction_method', 'unknown'),
            **metadata
        },
        'recommendations': recommendations,
        'responses': responses,
        'summary': {
            'total_items': len(recommendations) + len(responses),
            'total_words': sum(item.get('word_count', 0) for item in recommendations + responses),
            'average_quality': sum(item.get('quality_score', 0) for item in recommendations + responses) / max(len(recommendations + responses), 1),
            'extraction_timestamp': datetime.now().isoformat()
        }
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def create_formatted_report(recommendations: List[Dict], responses: List[Dict]) -> str:
    """Create formatted text report"""
    report_lines = [
        "=" * 80,
        "RECOMMENDATION-RESPONSE EXTRACTION REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Recommendations: {len(recommendations)}",
        f"Total Responses: {len(responses)}",
        "",
        "RECOMMENDATIONS",
        "=" * 40,
        ""
    ]
    
    # Add recommendations
    for i, rec in enumerate(recommendations, 1):
        report_lines.extend([
            f"RECOMMENDATION {rec.get('number', i)}",
            "-" * 30,
            f"Quality: {rec.get('quality_score', 0):.3f} | Confidence: {rec.get('confidence', 0):.3f}",
            f"Words: {rec.get('word_count', 0)} | Method: {rec.get('extraction_method', 'unknown')}",
            "",
            rec.get('text', 'No content'),
            "",
            ""
        ])
    
    # Add responses
    if responses:
        report_lines.extend([
            "",
            "RESPONSES",
            "=" * 40,
            ""
        ])
        
        for i, resp in enumerate(responses, 1):
            report_lines.extend([
                f"RESPONSE {resp.get('number', i)} ({resp.get('response_type', 'Unknown').upper()})",
                "-" * 40,
                f"Quality: {resp.get('quality_score', 0):.3f} | Confidence: {resp.get('confidence', 0):.3f}",
                f"Words: {resp.get('word_count', 0)} | Method: {resp.get('extraction_method', 'unknown')}",
                "",
                resp.get('text', 'No content'),
                "",
                ""
            ])
    
    return "\n".join(report_lines)

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
        'extraction_method': results.get('extraction_method', 'unknown'),
        'extraction_timestamp': results.get('timestamp', ''),
        'settings': results.get('settings', {})
    }

def get_recommendations_for_annotation() -> List[Dict]:
    """Get extracted recommendations for annotation tab (backwards compatibility)"""
    results = st.session_state.get('extraction_results', {})
    return results.get('recommendations', [])

# ===============================================
# MODULE EXPORTS
# ===============================================

__all__ = [
    # Main functions
    'render_extraction_tab',
    'render_smart_extraction_interface',
    'render_standard_extraction_interface',
    'render_ai_extraction_interface',
    'render_extraction_results',
    
    # Processing functions
    'process_smart_extraction',
    
    # Data access functions (FIXED)
    'get_document_content_for_extraction',
    'detect_document_type',
    'validate_documents_for_extraction',
    
    # Display functions
    'render_items_preview',
    'render_quality_analysis',
    
    # Download functions
    'create_comprehensive_csv',
    'create_detailed_json',
    'create_formatted_report',
    
    # Utility functions
    'clear_extraction_results',
    'get_extraction_statistics',
    'get_recommendations_for_annotation',
    
    # Classes
    'SmartExtractor',
    
    # Availability flags
    'ENHANCED_EXTRACTOR_AVAILABLE',
    'LLM_EXTRACTOR_AVAILABLE',
    'CORE_UTILS_AVAILABLE'
]

# ===============================================
# MODULE INITIALIZATION
# ===============================================

logging.info("🎉 Enhanced Extraction Components loaded with Smart Extraction capabilities!")
logging.info(f"Available features: Enhanced={ENHANCED_EXTRACTOR_AVAILABLE}, LLM={LLM_EXTRACTOR_AVAILABLE}, Core={CORE_UTILS_AVAILABLE}")
