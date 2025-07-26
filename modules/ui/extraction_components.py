# modules/ui/extraction_components.py
# COMPLETE ENHANCED FILE - Enhanced patterns for Government Response documents

import streamlit as st
import pandas as pd
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# Fix the Document import issue that's causing errors
try:
    from langchain.schema import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        # Create a simple Document class fallback
        class Document:
            def __init__(self, page_content: str, metadata: Dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

# Import RAG components with error handling
try:
    from modules.ui.rag_components import render_rag_extraction_interface
    RAG_COMPONENTS_AVAILABLE = True
    logging.info("✅ RAG components imported successfully")
except ImportError as e:
    RAG_COMPONENTS_AVAILABLE = False
    logging.error(f"❌ RAG components import failed: {e}")

try:
    from modules.rag_extractor import IntelligentRAGExtractor
    RAG_EXTRACTOR_AVAILABLE = True
    logging.info("✅ RAG extractor imported successfully")
except ImportError as e:
    RAG_EXTRACTOR_AVAILABLE = False
    logging.error(f"❌ RAG extractor import failed: {e}")

def get_enhanced_extraction_patterns():
    """
    Comprehensive patterns based on Government Response documents
    These patterns will capture every variation of recommendations and responses
    """
    
    # RECOMMENDATION PATTERNS - From inquiry/original documents
    recommendation_patterns = [
        # Direct numbered recommendations (most common in your docs)
        r'(?i)Recommendation\s+(\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s*[:\.)]\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
        
        # Sub-recommendations with letters/roman numerals  
        r'(?i)Recommendation\s+\d+[a-z]*\s*\)\s*([iv]+)\s*\)\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
        
        # We recommend/suggest patterns
        r'(?i)(?:We|I|The\s+(?:committee|panel|inquiry|report))\s+(?:recommend|suggest|propose)\s+(?:that\s+)?([^.!?]{20,500}[.!?])',
        
        # Should/must/ought patterns for requirements
        r'(?i)(?:should|must|ought\s+to|need\s+to)\s+be\s+([^.!?]{20,400}[.!?])',
        
        # Action required patterns
        r'(?i)(?:action\s+(?:required|needed|should\s+be\s+taken)|steps?\s+(?:should|must)\s+be\s+taken)\s*:?\s*([^.!?]{20,400}[.!?])',
        
        # Formal recommendation language
        r'(?i)(?:it\s+is\s+recommended|the\s+following\s+recommendation|this\s+recommendation)\s*:?\s*([^.!?]{20,400}[.!?])',
        
        # Framework/system establishment
        r'(?i)(?:framework|system|process|mechanism)\s+(?:should|must)\s+be\s+(?:established|created|implemented|developed)\s+(?:for\s+|to\s+)?([^.!?]{20,400}[.!?])',
        
        # Training/education recommendations
        r'(?i)(?:training|education|guidance)\s+(?:should|must)\s+be\s+(?:provided|given|delivered)\s+(?:to\s+)?([^.!?]{20,400}[.!?])',
        
        # Review/monitoring recommendations
        r'(?i)(?:review|monitor|assess|evaluate)\s+(?:should|must)\s+be\s+(?:conducted|carried\s+out|undertaken)\s*([^.!?]{20,400}[.!?])',
        
        # Progress recommendations
        r'(?i)(?:progress\s+(?:in\s+)?implementation|next\s+steps)\s+(?:should|must)\s+be\s+([^.!?]{20,400}[.!?])',
        
        # That clauses with recommendations
        r'(?i)(?:recommend|suggest|propose)\s+that\s+([^.!?]{20,500}[.!?])',
    ]
    
    # RESPONSE PATTERNS - From government response documents
    response_patterns = [
        # Core acceptance patterns (your most important ones)
        r'(?i)(?:This\s+recommendation|Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s+is\s+(?:accepted\s+in\s+full|accepted\s+in\s+principle|not\s+accepted|partially\s+accepted|rejected)\s*(?:by\s+(?:the\s+)?(?:UK\s+Government|Scottish\s+Government|Welsh\s+Government|Northern\s+Ireland\s+Executive|Government))?\.?\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
        
        # Acceptance with implementation details
        r'(?i)(?:accepted\s+in\s+(?:full|principle)|not\s+accepted|rejected|partially\s+accepted)\s*\.?\s*([^.!?]{20,500}[.!?])',
        
        # Government will/commits patterns
        r'(?i)(?:The\s+(?:UK\s+)?Government|We|The\s+(?:Department|Ministry))\s+(?:will|shall|commits?\s+to|agrees?\s+to|accepts?)\s+([^.!?]{20,500}[.!?])',
        
        # Implementation and progress patterns
        r'(?i)(?:Implementation|Progress|Work)\s+(?:will\s+begin|is\s+underway|has\s+begun|is\s+ongoing)\s*([^.!?]{20,500}[.!?])',
        
        # Establishment patterns
        r'(?i)(?:will\s+establish|is\s+establishing|has\s+established|establish)\s+(?:a\s+)?([^.!?]{20,500}[.!?])',
        
        # Review and monitoring responses
        r'(?i)(?:review|assessment|evaluation)\s+(?:will\s+be|is\s+being|has\s+been)\s+(?:conducted|undertaken|carried\s+out)\s*([^.!?]{20,500}[.!?])',
        
        # Funding and resource commitments
        r'(?i)(?:funding|resources?|investment)\s+(?:will\s+be|has\s+been|is\s+being)\s+(?:provided|allocated|committed)\s*([^.!?]{20,500}[.!?])',
        
        # Timeline and deadline responses
        r'(?i)(?:within\s+\d+\s+(?:months?|years?)|by\s+\d+|during\s+the\s+(?:first|next))\s*([^.!?]{20,500}[.!?])',
        
        # Action taken patterns
        r'(?i)(?:action\s+(?:has\s+been\s+taken|is\s+being\s+taken|will\s+be\s+taken)|measures?\s+(?:have\s+been|are\s+being|will\s+be)\s+(?:implemented|introduced|established))\s*([^.!?]{20,500}[.!?])',
        
        # Stakeholder involvement
        r'(?i)(?:working\s+with|in\s+collaboration\s+with|together\s+with)\s+(?:[^.!?]{5,100})\s+(?:to\s+)?([^.!?]{20,500}[.!?])',
        
        # Response with rationale
        r'(?i)(?:The\s+rationale|This\s+is\s+because|The\s+reason)\s+(?:for\s+this\s+(?:decision|approach|response))?\s*([^.!?]{20,500}[.!?])',
        
        # Cross-nation responses (specific to your documents)
        r'(?i)(?:UK\s+Government|Scottish\s+Government|Welsh\s+Government|Northern\s+Ireland\s+Executive)\s+(?:accepts?|agrees?|will)\s+([^.!?]{20,500}[.!?])',
        
        # Quality and safety management responses
        r'(?i)(?:quality\s+and\s+safety|patient\s+safety|safety\s+management)\s+(?:system|framework|approach)\s*([^.!?]{20,500}[.!?])',
        
        # Training and education responses
        r'(?i)(?:training|education|guidance)\s+(?:will\s+be|is\s+being|has\s+been)\s+(?:provided|delivered|enhanced|improved)\s*([^.!?]{20,500}[.!?])',
        
        # Accepting in principle (from your specific document)
        r'(?i)Accepting\s+in\s+principle\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
        
        # Response continuation patterns
        r'(?i)(?:These\s+recommendations|Parts\s+of\s+these\s+recommendations)\s+are\s+(?:accepted|being\s+taken\s+forward)\s*([^.!?]{20,500}[.!?])',
    ]
    
    return {
        'recommendation_patterns': recommendation_patterns,
        'response_patterns': response_patterns
    }

def get_document_content_for_extraction(doc: Dict) -> str:
    """
    Extract content from document for processing
    This function was missing and causing all the import errors!
    """
    if not isinstance(doc, dict):
        return ""
    
    # Try different content fields in order of preference
    content_fields = [
        'content',           # Primary content field
        'text',             # Alternative text field
        'extracted_text',   # OCR or extracted text
        'page_content',     # PDF page content
        'raw_text',         # Raw text content
        'body',             # Document body
        'full_text',        # Full text content
        'document_text'     # Document text field
    ]
    
    content = ""
    
    # Try each field until we find content
    for field in content_fields:
        if field in doc and doc[field]:
            potential_content = str(doc[field]).strip()
            if len(potential_content) > len(content):
                content = potential_content
    
    # If no direct content, try to extract from sections
    if not content and 'sections' in doc and doc['sections']:
        sections_text = []
        for section in doc['sections']:
            if isinstance(section, dict) and 'content' in section:
                section_content = str(section['content']).strip()
                if section_content:
                    sections_text.append(section_content)
            elif isinstance(section, str) and section.strip():
                sections_text.append(section.strip())
        
        if sections_text:
            content = '\n\n'.join(sections_text)
    
    # Final fallback - try to extract from any string field
    if not content:
        for key, value in doc.items():
            if isinstance(value, str) and len(value) > 100:
                content = value
                break
    
    return content.strip() if content else ""

def get_document_content(doc: Dict) -> str:
    """
    Alias for get_document_content_for_extraction to maintain compatibility
    """
    return get_document_content_for_extraction(doc)

def validate_document_structure(doc: Dict) -> Dict:
    """Validate and fix document structure"""
    
    # Ensure required fields exist
    if 'filename' not in doc:
        doc['filename'] = f"document_{id(doc)}"
    
    if 'document_type' not in doc:
        doc['document_type'] = 'unknown'
    
    # Ensure content is available
    content = get_document_content_for_extraction(doc)
    if content:
        doc['content'] = content
        doc['content_length'] = len(content)
        doc['has_content'] = True
    else:
        doc['has_content'] = False
        doc['content'] = ""
        doc['content_length'] = 0
    
    return doc

def enhanced_pattern_extract(content: str, patterns: List[str], max_results: int = 50, min_length: int = 25) -> List[Dict]:
    """Enhanced pattern extraction for government documents"""
    extractions = []
    seen_content = set()
    
    for pattern in patterns:
        try:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                # Extract the full match or the last capturing group
                if match.groups():
                    # If there are capturing groups, use the last one (the content)
                    text = match.group(-1).strip()
                else:
                    # If no capturing groups, use the full match
                    text = match.group().strip()
                
                # Filter by length
                if len(text) < min_length:
                    continue
                
                # Clean up text
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                
                # Check for duplicates
                text_normalized = text.lower().strip()
                if text_normalized in seen_content:
                    continue
                seen_content.add(text_normalized)
                
                # Skip if it's just a recommendation number or header
                if re.match(r'^(?:Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s*$', text, re.IGNORECASE):
                    continue
                
                extractions.append({
                    'content': text,
                    'confidence': 0.85,
                    'extraction_method': 'enhanced_pattern',
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'pattern_used': pattern[:50] + "..." if len(pattern) > 50 else pattern
                })
                
                if len(extractions) >= max_results:
                    break
            
            if len(extractions) >= max_results:
                break
                
        except re.error as e:
            logging.warning(f"Regex pattern error: {e}")
            continue
    
    return extractions

def basic_pattern_extract(content: str, patterns: List[str], max_results: int = 25, min_length: int = 20) -> List[Dict]:
    """Basic pattern extraction fallback"""
    extractions = []
    seen_content = set()
    
    for pattern in patterns:
        try:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                text = match.group().strip()
                
                # Filter by length
                if len(text) < min_length:
                    continue
                
                # Check for duplicates
                text_normalized = text.lower().strip()
                if text_normalized in seen_content:
                    continue
                seen_content.add(text_normalized)
                
                extractions.append({
                    'content': text,
                    'confidence': 0.7,
                    'extraction_method': 'pattern_basic',
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
                
                if len(extractions) >= max_results:
                    break
            
            if len(extractions) >= max_results:
                break
                
        except re.error as e:
            logging.warning(f"Regex pattern error: {e}")
            continue
    
    return extractions

def store_extraction_results(recommendations: List[Dict], responses: List[Dict], method: str):
    """Store extraction results in session state"""
    
    # Initialize if not exists
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = {}
    
    # Store results
    st.session_state.extraction_results = {
        'recommendations': recommendations,
        'responses': responses,
        'method': method,
        'timestamp': datetime.now(),
        'total_extracted': len(recommendations) + len(responses)
    }
    
    # Update session state flags
    st.session_state.extracted_recommendations = recommendations
    st.session_state.extracted_responses = responses

def display_extraction_results():
    """Display extraction results with enhanced formatting"""
    
    if 'extraction_results' not in st.session_state:
        st.info("No extraction results available.")
        return
    
    results = st.session_state.extraction_results
    recommendations = results.get('recommendations', [])
    responses = results.get('responses', [])
    
    st.success(f"✅ Extraction completed using {results.get('method', 'unknown')} method")
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recommendations", len(recommendations))
    with col2:
        st.metric("Responses", len(responses))
    with col3:
        st.metric("Total Extracted", len(recommendations) + len(responses))
    
    # Display results with confidence scores
    if recommendations:
        st.subheader("📋 Recommendations Found")
        for i, rec in enumerate(recommendations[:10], 1):
            confidence = rec.get('confidence', 0)
            method = rec.get('extraction_method', 'unknown')
            with st.expander(f"Recommendation {i} (Confidence: {confidence:.2f}, Method: {method})"):
                st.write(rec.get('content', ''))
                
        if len(recommendations) > 10:
            st.info(f"... and {len(recommendations) - 10} more recommendations")
    
    if responses:
        st.subheader("💬 Responses Found")
        for i, resp in enumerate(responses[:10], 1):
            confidence = resp.get('confidence', 0)
            method = resp.get('extraction_method', 'unknown')
            with st.expander(f"Response {i} (Confidence: {confidence:.2f}, Method: {method})"):
                st.write(resp.get('content', ''))
                
        if len(responses) > 10:
            st.info(f"... and {len(responses) - 10} more responses")

def render_extraction_tab():
    """Main extraction tab interface"""
    st.header("📄 Extract Content")
    
    # Check if documents are uploaded
    docs = st.session_state.get('uploaded_documents', [])
    if not docs:
        st.warning("⚠️ Please upload documents first in the Upload tab.")
        return
    
    st.info(f"📊 {len(docs)} documents available for extraction")
    
    # Extraction method selection
    method = st.selectbox(
        "Choose Extraction Method:",
        ["Enhanced Complete", "Smart Complete", "Pattern Basic", "RAG Intelligent"],
        help="Different methods for extracting recommendations and responses"
    )
    
    # Method-specific interfaces
    if method == "Enhanced Complete":
        render_enhanced_extraction_interface()
    elif method == "RAG Intelligent" and RAG_COMPONENTS_AVAILABLE:
        render_rag_extraction_interface()
    elif method == "Smart Complete":
        render_smart_extraction_interface()
    elif method == "Pattern Basic":
        render_basic_pattern_interface()
    else:
        st.error("Selected extraction method is not available")
        render_enhanced_extraction_interface()  # Fallback to enhanced

def render_enhanced_extraction_interface():
    """Enhanced extraction interface with government document patterns"""
    st.subheader("🚀 Enhanced Complete Extraction")
    st.info("Optimized for UK Government Response documents with comprehensive pattern matching")
    
    col1, col2 = st.columns(2)
    with col1:
        min_length = st.slider("Minimum Length", 20, 200, 25)
        max_results = st.slider("Max Results per Type", 10, 200, 50)
    
    with col2:
        context_window = st.slider("Context Window", 50, 500, 200)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7)
    
    # Document selection
    docs = st.session_state.get('uploaded_documents', [])
    doc_names = [doc.get('filename', f'Document {i+1}') for i, doc in enumerate(docs)]
    
    selected_docs = st.multiselect(
        "Select documents to process:",
        doc_names,
        default=doc_names,
        help="Choose which documents to extract from"
    )
    
    if st.button("🚀 Start Enhanced Extraction", type="primary"):
        if not selected_docs:
            st.error("Please select at least one document")
        else:
            process_enhanced_extraction(selected_docs, min_length, max_results, context_window, confidence_threshold)

def render_smart_extraction_interface():
    """Smart extraction interface"""
    st.subheader("🧠 Smart Complete Extraction")
    
    col1, col2 = st.columns(2)
    with col1:
        min_length = st.slider("Minimum Length", 20, 200, 50)
        max_results = st.slider("Max Results per Type", 10, 100, 25)
    
    with col2:
        context_window = st.slider("Context Window", 50, 500, 200)
    
    if st.button("🚀 Start Smart Extraction", type="primary"):
        process_smart_extraction(min_length, max_results, context_window)

def render_basic_pattern_interface():
    """Basic pattern extraction interface"""
    st.subheader("⚡ Basic Pattern Extraction")
    
    col1, col2 = st.columns(2)
    with col1:
        min_length = st.slider("Minimum Length", 10, 100, 20)
        max_results = st.slider("Max Results", 5, 50, 25)
    
    if st.button("⚡ Start Basic Extraction", type="primary"):
        process_basic_extraction(min_length, max_results)

def process_enhanced_extraction(selected_docs: List[str], min_length: int, max_results: int, 
                               context_window: int, confidence_threshold: float):
    """Process enhanced extraction with government document patterns"""
    docs = st.session_state.get('uploaded_documents', [])
    patterns = get_enhanced_extraction_patterns()
    
    all_recommendations = []
    all_responses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Filter selected documents
    selected_doc_objects = [doc for doc in docs if doc.get('filename') in selected_docs]
    
    for i, doc in enumerate(selected_doc_objects):
        filename = doc.get('filename', f'Document {i+1}')
        status_text.text(f"🚀 Processing {filename}...")
        progress_bar.progress((i + 1) / len(selected_doc_objects))
        
        content = get_document_content_for_extraction(doc)
        if not content:
            st.warning(f"No content found in {filename}")
            continue
        
        st.info(f"Processing {filename}: {len(content)} characters")
        
        # Extract recommendations using enhanced patterns
        doc_recommendations = enhanced_pattern_extract(
            content, patterns['recommendation_patterns'], max_results, min_length
        )
        
        # Extract responses using enhanced patterns
        doc_responses = enhanced_pattern_extract(
            content, patterns['response_patterns'], max_results, min_length
        )
        
        # Add metadata and filter by confidence
        for item in doc_recommendations:
            if item['confidence'] >= confidence_threshold:
                item['document_context'] = {'filename': filename}
                item['extraction_type'] = 'recommendation'
                all_recommendations.append(item)
        
        for item in doc_responses:
            if item['confidence'] >= confidence_threshold:
                item['document_context'] = {'filename': filename}
                item['extraction_type'] = 'response'
                all_responses.append(item)
    
    # Store results
    store_extraction_results(all_recommendations, all_responses, 'enhanced_complete')
    
    status_text.text("✅ Enhanced extraction completed!")
    progress_bar.progress(1.0)
    
    # Display results
    display_extraction_results()

def process_smart_extraction(min_length: int, max_results: int, context_window: int):
    """Process smart extraction"""
    docs = st.session_state.get('uploaded_documents', [])
    
    # Enhanced patterns for government documents
    recommendation_patterns = [
        r'(?:Recommendation|We recommend|I recommend|The committee recommends?)\s*:?\s*([^.!?]*[.!?])',
        r'(?:It is recommended|We suggest|We propose)\s+(?:that\s+)?([^.!?]*[.!?])',
        r'(?:The following recommendation|This recommendation)\s*:?\s*([^.!?]*[.!?])',
        r'(?:Recommendation\s+\d+|R\d+)\s*:?\s*([^.!?]*[.!?])',
        r'(?:Action\s+required|Action\s+needed)\s*:?\s*([^.!?]*[.!?])',
    ]
    
    response_patterns = [
        r'(?:We accept|We agree|The government accepts?|The department accepts?)\s*(?:that\s+)?([^.!?]*[.!?])',
        r'(?:In response|Response|Our response)\s*:?\s*([^.!?]*[.!?])',
        r'(?:This recommendation|The recommendation)\s+(?:is|has been|will be|was)\s+([^.!?]*[.!?])',
        r'(?:Accepted|Agreed|Implemented|Partially accepted|Rejected)\s*:?\s*([^.!?]*[.!?])',
        r'(?:Action taken|Implementation|Progress made)\s*:?\s*([^.!?]*[.!?])',
    ]
    
    all_recommendations = []
    all_responses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, doc in enumerate(docs):
        filename = doc.get('filename', f'Document {i+1}')
        status_text.text(f"🧠 Processing {filename}...")
        progress_bar.progress((i + 1) / len(docs))
        
        content = get_document_content_for_extraction(doc)
        if not content:
            continue
        
        # Extract recommendations
        doc_recommendations = extract_with_patterns(
            content, recommendation_patterns, 'recommendation', 
            min_length, max_results // 2, context_window, filename
        )
        
        # Extract responses  
        doc_responses = extract_with_patterns(
            content, response_patterns, 'response',
            min_length, max_results // 2, context_window, filename
        )
        
        all_recommendations.extend(doc_recommendations)
        all_responses.extend(doc_responses)
    
    # Store results
    store_extraction_results(all_recommendations, all_responses, 'smart_complete')
    
    status_text.text("✅ Smart extraction completed!")
    progress_bar.progress(1.0)
    
    # Display results
    display_extraction_results()

def process_basic_extraction(min_length: int, max_results: int):
    """Process basic pattern extraction"""
    docs = st.session_state.get('uploaded_documents', [])
    
    # Simple patterns
    rec_patterns = [
        r'(?:recommend|suggestion)[^.!?]*[.!?]',
        r'(?:We|I)\s+(?:recommend|suggest)[^.!?]*[.!?]'
    ]
    
    resp_patterns = [
        r'(?:accept|agreed|implement)[^.!?]*[.!?]',
        r'(?:response|reply)[^.!?]*[.!?]'
    ]
    
    all_recommendations = []
    all_responses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, doc in enumerate(docs):
        filename = doc.get('filename', f'Document {i+1}')
        status_text.text(f"⚡ Processing {filename}...")
        progress_bar.progress((i + 1) / len(docs))
        
        content = get_document_content_for_extraction(doc)
        if not content:
            continue
        
        # Basic extraction
        doc_recommendations = basic_pattern_extract(content, rec_patterns, max_results, min_length)
        doc_responses = basic_pattern_extract(content, resp_patterns, max_results, min_length)
        
        # Add metadata
        for item in doc_recommendations:
            item['document_context'] = {'filename': filename}
            item['extraction_method'] = 'pattern_basic'
            item['extraction_type'] = 'recommendation'
            
        for item in doc_responses:
            item['document_context'] = {'filename': filename}
            item['extraction_method'] = 'pattern_basic'
            item['extraction_type'] = 'response'
        
        all_recommendations.extend(doc_recommendations)
        all_responses.extend(doc_responses)
    
    # Store results
    store_extraction_results(all_recommendations, all_responses, 'pattern_basic')
    
    status_text.text("✅ Pattern extraction completed!")
    progress_bar.progress(1.0)
    
    # Display results
    display_extraction_results()

def extract_with_patterns(content: str, patterns: List[str], extraction_type: str,
                         min_length: int, max_results: int, context_window: int, filename: str) -> List[Dict]:
    """Extract content using improved pattern matching"""
    
    extractions = []
    seen_content = set()
    
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        for match in matches:
            text = match.group().strip()
            
            # Filter by length
            if len(text) < min_length:
                continue
            
            # Check for duplicates
            text_normalized = text.lower().strip()
            if text_normalized in seen_content:
                continue
            seen_content.add(text_normalized)
            
            # Get context
            start_pos = max(0, match.start() - context_window)
            end_pos = min(len(content), match.end() + context_window)
            context = content[start_pos:end_pos]
            
            extractions.append({
                'content': text,
                'confidence': 0.8,
                'extraction_method': 'smart_complete',
                'extraction_type': extraction_type,
                'context': context,
                'document_context': {'filename': filename},
                'position': match.start()
            })
            
            if len(extractions) >= max_results:
                break
        
        if len(extractions) >= max_results:
            break
    
    return extractions

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return re.sub(r'\s+', ' ', text.lower().strip())

def extract_numbered_recommendations(content: str) -> List[Dict]:
    """Extract specifically numbered recommendations from government documents"""
    extractions = []
    
    # Pattern for numbered recommendations like "Recommendation 1:", "Recommendation 6a) iii)", etc.
    pattern = r'(?i)(Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s*[:\.)]\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))'
    
    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    for match in matches:
        rec_number = match.group(1).strip()
        rec_content = match.group(2).strip()
        
        if len(rec_content) > 20:  # Filter out empty or very short content
            extractions.append({
                'content': f"{rec_number}: {rec_content}",
                'recommendation_number': rec_number,
                'confidence': 0.95,
                'extraction_method': 'numbered_recommendation',
                'extraction_type': 'recommendation'
            })
    
    return extractions

def extract_acceptance_responses(content: str) -> List[Dict]:
    """Extract specific acceptance/rejection responses from government documents"""
    extractions = []
    
    # Pattern for acceptance statements
    acceptance_pattern = r'(?i)(?:This\s+recommendation|Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s+is\s+(accepted\s+in\s+(?:full|principle)|not\s+accepted|partially\s+accepted|rejected)\s*(?:by\s+(?:the\s+)?(?:UK\s+Government|Scottish\s+Government|Welsh\s+Government|Northern\s+Ireland\s+Executive|Government))?\.?\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))'
    
    matches = re.finditer(acceptance_pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    for match in matches:
        acceptance_type = match.group(1).strip()
        additional_content = match.group(2).strip() if len(match.groups()) > 1 else ""
        
        full_content = match.group(0).strip()
        if additional_content:
            full_content += " " + additional_content
        
        extractions.append({
            'content': full_content,
            'acceptance_type': acceptance_type,
            'confidence': 0.95,
            'extraction_method': 'acceptance_response',
            'extraction_type': 'response'
        })
    
    return extractions

def process_document_with_all_methods(doc: Dict, filename: str) -> Dict:
    """Process a single document with all extraction methods for maximum coverage"""
    content = get_document_content_for_extraction(doc)
    if not content:
        return {'recommendations': [], 'responses': []}
    
    all_recommendations = []
    all_responses = []
    
    # Method 1: Enhanced patterns
    patterns = get_enhanced_extraction_patterns()
    enhanced_recs = enhanced_pattern_extract(content, patterns['recommendation_patterns'], 25, 25)
    enhanced_resps = enhanced_pattern_extract(content, patterns['response_patterns'], 25, 25)
    
    # Method 2: Numbered recommendations
    numbered_recs = extract_numbered_recommendations(content)
    
    # Method 3: Acceptance responses
    acceptance_resps = extract_acceptance_responses(content)
    
    # Combine and deduplicate
    for rec in enhanced_recs + numbered_recs:
        rec['document_context'] = {'filename': filename}
        rec['extraction_type'] = 'recommendation'
        all_recommendations.append(rec)
    
    for resp in enhanced_resps + acceptance_resps:
        resp['document_context'] = {'filename': filename}
        resp['extraction_type'] = 'response'
        all_responses.append(resp)
    
    # Deduplicate based on content similarity
    all_recommendations = deduplicate_extractions(all_recommendations)
    all_responses = deduplicate_extractions(all_responses)
    
    return {
        'recommendations': all_recommendations,
        'responses': all_responses
    }

def deduplicate_extractions(extractions: List[Dict]) -> List[Dict]:
    """Remove duplicate extractions based on content similarity"""
    if not extractions:
        return []
    
    unique_extractions = []
    seen_content = set()
    
    for extraction in extractions:
        content = extraction.get('content', '')
        normalized = normalize_text(content)
        
        # Check for exact duplicates
        if normalized in seen_content:
            continue
        
        # Check for substantial overlap (>80% similarity)
        is_duplicate = False
        for seen in seen_content:
            if len(seen) > 0 and len(normalized) > 0:
                # Simple similarity check
                shorter = min(len(seen), len(normalized))
                longer = max(len(seen), len(normalized))
                if shorter / longer > 0.8:
                    # Check if one is contained in the other
                    if normalized in seen or seen in normalized:
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            seen_content.add(normalized)
            unique_extractions.append(extraction)
    
    # Sort by confidence and length (prefer higher confidence and longer content)
    unique_extractions.sort(key=lambda x: (x.get('confidence', 0), len(x.get('content', ''))), reverse=True)
    
    return unique_extractions

def export_extraction_results(format_type: str = "csv") -> None:
    """Export extraction results to file"""
    if 'extraction_results' not in st.session_state:
        st.error("No extraction results to export")
        return
    
    results = st.session_state.extraction_results
    recommendations = results.get('recommendations', [])
    responses = results.get('responses', [])
    
    if format_type == "csv":
        # Create DataFrames
        rec_df = pd.DataFrame(recommendations) if recommendations else pd.DataFrame()
        resp_df = pd.DataFrame(responses) if responses else pd.DataFrame()
        
        # Add type column
        if not rec_df.empty:
            rec_df['type'] = 'recommendation'
        if not resp_df.empty:
            resp_df['type'] = 'response'
        
        # Combine
        combined_df = pd.concat([rec_df, resp_df], ignore_index=True) if not rec_df.empty or not resp_df.empty else pd.DataFrame()
        
        if not combined_df.empty:
            csv_data = combined_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.error("No data to export")
    
    elif format_type == "json":
        import json
        export_data = {
            'extraction_metadata': {
                'method': results.get('method', 'unknown'),
                'timestamp': results.get('timestamp', datetime.now()).isoformat(),
                'total_extracted': results.get('total_extracted', 0)
            },
            'recommendations': recommendations,
            'responses': responses
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def render_extraction_results_tab():
    """Render the extraction results tab with export options"""
    st.header("📊 Extraction Results")
    
    if 'extraction_results' not in st.session_state:
        st.info("No extraction results available. Please run an extraction first.")
        return
    
    # Display results
    display_extraction_results()
    
    # Export options
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as CSV"):
            export_extraction_results("csv")
    
    with col2:
        if st.button("Export as JSON"):
            export_extraction_results("json")
    
    # Advanced filtering
    st.subheader("🔍 Filter Results")
    
    results = st.session_state.extraction_results
    recommendations = results.get('recommendations', [])
    responses = results.get('responses', [])
    
    if recommendations or responses:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
        
        # Filter by confidence
        filtered_recs = [r for r in recommendations if r.get('confidence', 0) >= min_confidence]
        filtered_resps = [r for r in responses if r.get('confidence', 0) >= min_confidence]
        
        st.write(f"Filtered results: {len(filtered_recs)} recommendations, {len(filtered_resps)} responses")
        
        # Update session state with filtered results
        if st.button("Apply Filter"):
            st.session_state.extraction_results['recommendations'] = filtered_recs
            st.session_state.extraction_results['responses'] = filtered_resps
            st.rerun()

def test_enhanced_patterns():
    """Test function for the enhanced patterns"""
    sample_text = """
    Recommendation 6a) iii) (accepted in principle by NI Executive)
    
    This recommendation is accepted in principle by the UK Government, 
    the Scottish Government, the Welsh Government, and the Northern Ireland Executive.
    
    Implementation will begin immediately through existing NHS structures.
    
    Recommendation 7b) Progress in implementation of the Transfusion 2024 recommendations be 
    reviewed, and next steps be determined and promulgated.
    
    This recommendation is accepted in full by the Scottish Government.
    
    The Government will establish a review process within 12 months.
    
    We recommend that all healthcare providers should implement comprehensive safety protocols.
    
    Action should be taken to ensure patient safety across all departments.
    """
    
    patterns = get_enhanced_extraction_patterns()
    
    recommendations = enhanced_pattern_extract(sample_text, patterns['recommendation_patterns'], 10, 20)
    responses = enhanced_pattern_extract(sample_text, patterns['response_patterns'], 10, 20)
    
    print(f"Found {len(recommendations)} recommendations and {len(responses)} responses")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"Recommendation {i}: {rec['content'][:100]}...")
    
    for i, resp in enumerate(responses, 1):
        print(f"Response {i}: {resp['content'][:100]}...")

# Export all required functions
__all__ = [
    'get_document_content_for_extraction',
    'get_document_content', 
    'render_extraction_tab',
    'render_extraction_results_tab',
    'validate_document_structure',
    'store_extraction_results',
    'display_extraction_results',
    'get_enhanced_extraction_patterns',
    'enhanced_pattern_extract',
    'process_enhanced_extraction',
    'export_extraction_results',
    'test_enhanced_patterns',
    'Document'
]
