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
                
            def __str__(self):
                return self.page_content

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
    ]
    
    # RESPONSE PATTERNS - From government response documents
    response_patterns = [
        # Direct acceptance patterns
        r'(?i)(?:accepts?|accepting|agreed?|agreeing)\s+(?:this\s+recommendation|recommendation\s+\d+)\s*([^.!?]{20,500}[.!?])',
        
        # Government commitment patterns
        r'(?i)(?:The\s+Government|We|The\s+Department)\s+(?:will|shall|commits?\s+to|has\s+committed\s+to)\s+([^.!?]{20,500}[.!?])',
        
        # Implementation patterns
        r'(?i)(?:will\s+be\s+implemented|is\s+being\s+implemented|has\s+been\s+implemented|implementation\s+will)\s+([^.!?]{20,500}[.!?])',
        
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
    
    # If still no content, try metadata or filename info
    if not content and 'filename' in doc:
        content = f"Document: {doc['filename']}"
    
    return content.strip() if content else ""

def get_document_content(doc: Dict) -> str:
    """
    Alias for get_document_content_for_extraction to maintain compatibility
    """
    return get_document_content_for_extraction(doc)

def validate_documents_for_extraction(documents: List[Dict]) -> tuple:
    """
    Validate documents are ready for extraction
    This function was missing and causing import errors!
    """
    if not documents:
        return False, "No documents provided"
    
    valid_docs = []
    issues = []
    
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            issues.append(f"Document {i+1}: Not a valid document format")
            continue
            
        content = get_document_content_for_extraction(doc)
        if not content:
            issues.append(f"Document {i+1}: No content found")
            continue
            
        if len(content.strip()) < 100:
            issues.append(f"Document {i+1}: Content too short ({len(content)} chars)")
            continue
            
        valid_docs.append(doc)
    
    if not valid_docs:
        return False, f"No valid documents found. Issues: {'; '.join(issues)}"
    
    success_msg = f"{len(valid_docs)} documents ready for extraction"
    if issues:
        success_msg += f". Note: {len(issues)} documents skipped"
    
    return True, success_msg

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

def extract_with_patterns(content: str, patterns: List[str], extraction_type: str, 
                         min_length: int = 25, max_results: int = 50, 
                         context_window: int = 200, source_doc: str = "unknown") -> List[Dict]:
    """Enhanced pattern extraction for government documents"""
    extractions = []
    seen_content = set()
    
    for pattern_idx, pattern in enumerate(patterns):
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
                
                # Extract context
                start_pos = max(0, match.start() - context_window)
                end_pos = min(len(content), match.end() + context_window)
                context = content[start_pos:end_pos].strip()
                
                extractions.append({
                    'text': text,
                    'type': extraction_type,
                    'confidence': 0.8,
                    'source': 'pattern_extraction',
                    'position': match.start(),
                    'pattern_index': pattern_idx,
                    'extraction_method': 'regex_pattern',
                    'source_document': source_doc,
                    'context': context,
                    'length': len(text)
                })
                
                if len(extractions) >= max_results:
                    break
                    
        except re.error as e:
            logging.warning(f"Regex error in pattern {pattern_idx}: {e}")
            continue
        
        if len(extractions) >= max_results:
            break
    
    return extractions

class SmartExtractor:
    """
    Smart extraction class for government documents
    This class was missing and causing all the import errors!
    """
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        self.patterns = get_enhanced_extraction_patterns()
    
    def extract_recommendations_and_responses(self, documents, min_length=50, max_results=25, context_window=200):
        """Extract recommendations and responses using AI-enhanced patterns"""
        
        recommendations = []
        responses = []
        
        for doc_idx, doc in enumerate(documents):
            content = get_document_content_for_extraction(doc)
            if not content:
                continue
                
            # Extract using patterns
            doc_recommendations = self._extract_with_patterns(
                content, self.patterns['recommendation_patterns'], 'recommendation', doc_idx
            )
            doc_responses = self._extract_with_patterns(
                content, self.patterns['response_patterns'], 'response', doc_idx
            )
            
            recommendations.extend(doc_recommendations)
            responses.extend(doc_responses)
        
        # Filter by length and limit results
        recommendations = [r for r in recommendations if len(r['text']) >= min_length][:max_results]
        responses = [r for r in responses if len(r['text']) >= min_length][:max_results]
        
        return {
            'recommendations': recommendations,
            'responses': responses,
            'extraction_stats': {
                'total_documents': len(documents),
                'total_recommendations': len(recommendations),
                'total_responses': len(responses),
                'extraction_time': datetime.now().isoformat()
            }
        }
    
    def _extract_with_patterns(self, content, patterns, extraction_type, doc_idx):
        """Extract content using regex patterns"""
        results = []
        
        for pattern_idx, pattern in enumerate(patterns):
            try:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for match in matches:
                    # Get the captured group or full match
                    if match.groups():
                        extracted_text = match.group(1).strip()
                    else:
                        extracted_text = match.group(0).strip()
                    
                    if len(extracted_text.strip()) >= 20:  # Minimum length filter
                        results.append({
                            'text': extracted_text,
                            'type': extraction_type,
                            'confidence': 0.8,  # Pattern-based confidence
                            'source': 'pattern_extraction',
                            'position': match.start(),
                            'document_index': doc_idx,
                            'pattern_index': pattern_idx,
                            'extraction_method': 'regex_pattern'
                        })
            except re.error as e:
                if self.debug_mode:
                    self.logger.warning(f"Regex error in pattern {pattern_idx}: {e}")
                continue
        
        return results

def store_extraction_results(recommendations: List[Dict], responses: List[Dict], method: str):
    """Store extraction results in session state"""
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = {}
    
    st.session_state.extraction_results[method] = {
        'recommendations': recommendations,
        'responses': responses,
        'timestamp': datetime.now().isoformat(),
        'method': method
    }
    
    # Also store in main session state for compatibility
    st.session_state.extracted_recommendations = recommendations
    st.session_state.extracted_responses = responses

def process_smart_extraction(min_length: int, max_results: int, context_window: int):
    """Process smart complete extraction"""
    docs = st.session_state.get('uploaded_documents', [])
    
    # Enhanced patterns for complete extraction
    recommendation_patterns = [
        r'(?:Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s*[:\.)]\s*([^.!?]*[.!?])',
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
        
        # Extract with basic patterns
        doc_recommendations = extract_with_patterns(
            content, rec_patterns, 'recommendation', 
            min_length, max_results // 2, 100, filename
        )
        
        doc_responses = extract_with_patterns(
            content, resp_patterns, 'response',
            min_length, max_results // 2, 100, filename
        )
        
        all_recommendations.extend(doc_recommendations)
        all_responses.extend(doc_responses)
    
    # Store results
    store_extraction_results(all_recommendations, all_responses, 'basic_pattern')
    
    status_text.text("✅ Basic extraction completed!")
    progress_bar.progress(1.0)
    
    # Display results
    display_extraction_results()

def render_extraction_tab():
    """Main extraction tab rendering function"""
    st.header("🔍 Recommendation & Response Extraction")
    
    # Check if documents are uploaded
    if 'uploaded_documents' not in st.session_state or not st.session_state.uploaded_documents:
        st.warning("⚠️ Please upload documents first in the Upload tab.")
        return
    
    st.markdown("""
    Extract recommendations and government responses from uploaded documents using advanced pattern matching 
    and AI-enhanced analysis specifically designed for UK Government inquiry reports.
    """)
    
    # Document selection
    uploaded_docs = st.session_state.uploaded_documents
    doc_names = [doc.get('filename', f'Document {i+1}') for i, doc in enumerate(uploaded_docs)]
    
    selected_doc_names = st.multiselect(
        "📄 Select Documents to Process",
        doc_names,
        default=doc_names,
        help="Choose which documents to extract from"
    )
    
    if not selected_doc_names:
        st.warning("Please select at least one document to process.")
        return
    
    # Get selected documents
    selected_docs = [doc for doc in uploaded_docs 
                    if doc.get('filename') in selected_doc_names]
    
    # Extraction settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_length = st.slider("Minimum Text Length", 20, 200, 50,
                              help="Minimum character length for extracted items")
    
    with col2:
        max_results = st.slider("Max Results per Type", 10, 100, 25,
                               help="Maximum number of items to extract")
    
    with col3:
        extraction_method = st.selectbox(
            "Extraction Method",
            ["Smart Pattern Matching", "Basic Pattern Matching"],
            help="Choose extraction approach"
        )
    
    # Validate documents
    valid, validation_msg = validate_documents_for_extraction(selected_docs)
    
    if valid:
        st.success(f"✅ {validation_msg}")
    else:
        st.error(f"❌ {validation_msg}")
        return
    
    # Extraction button
    if st.button("🚀 Start Extraction", type="primary"):
        with st.spinner("🔍 Extracting recommendations and responses..."):
            try:
                if extraction_method == "Smart Pattern Matching":
                    # Initialize extractor
                    extractor = SmartExtractor(debug_mode=True)
                    
                    # Run extraction
                    results = extractor.extract_recommendations_and_responses(
                        selected_docs,
                        min_length=min_length,
                        max_results=max_results
                    )
                    
                    # Store results in session state
                    st.session_state.extracted_recommendations = results['recommendations']
                    st.session_state.extracted_responses = results['responses']
                    st.session_state.extraction_stats = results['extraction_stats']
                    
                    # Display results
                    display_extraction_results(results)
                
                else:  # Basic Pattern Matching
                    process_basic_extraction(min_length, max_results)
                
            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
                if st.checkbox("Show debug info"):
                    st.exception(e)

def display_extraction_results(results=None):
    """Display extraction results in a user-friendly format"""
    
    # Get results from parameter or session state
    if results:
        recommendations = results['recommendations']
        responses = results['responses']
        stats = results['extraction_stats']
    else:
        recommendations = st.session_state.get('extracted_recommendations', [])
        responses = st.session_state.get('extracted_responses', [])
        stats = st.session_state.get('extraction_stats', {})
    
    if not recommendations and not responses:
        st.info("No extraction results to display.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Recommendations", len(recommendations))
    with col2:
        st.metric("📝 Responses", len(responses))
    with col3:
        st.metric("📄 Documents", stats.get('total_documents', 0))
    with col4:
        st.metric("⚡ Status", "Complete", delta="Success")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📝 Responses", "📊 Summary"])
    
    with tab1:
        if recommendations:
            st.subheader(f"📋 Found {len(recommendations)} Recommendations")
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"Recommendation {i} (Confidence: {rec.get('confidence', 0):.1%})"):
                    st.write(rec['text'])
                    
                    # Metadata
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.caption(f"Source: {rec.get('source', 'Unknown')}")
                        st.caption(f"Method: {rec.get('extraction_method', 'Unknown')}")
                    with meta_col2:
                        st.caption(f"Document: {rec.get('document_index', 'Unknown')}")
                        st.caption(f"Position: {rec.get('position', 'Unknown')}")
        else:
            st.info("No recommendations found with current settings.")
    
    with tab2:
        if responses:
            st.subheader(f"📝 Found {len(responses)} Responses")
            for i, resp in enumerate(responses, 1):
                with st.expander(f"Response {i} (Confidence: {resp.get('confidence', 0):.1%})"):
                    st.write(resp['text'])
                    
                    # Metadata
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.caption(f"Source: {resp.get('source', 'Unknown')}")
                        st.caption(f"Method: {resp.get('extraction_method', 'Unknown')}")
                    with meta_col2:
                        st.caption(f"Document: {resp.get('document_index', 'Unknown')}")
                        st.caption(f"Position: {resp.get('position', 'Unknown')}")
        else:
            st.info("No responses found with current settings.")
    
    with tab3:
        st.subheader("📊 Extraction Summary")
        
        # Create summary DataFrame
        summary_data = []
        for item_type, items in [("Recommendations", recommendations), ("Responses", responses)]:
            for item in items:
                summary_data.append({
                    'Type': item_type,
                    'Text Preview': item['text'][:100] + "..." if len(item['text']) > 100 else item['text'],
                    'Confidence': f"{item.get('confidence', 0):.1%}",
                    'Source': item.get('source', 'Unknown'),
                    'Document': item.get('document_index', 'Unknown')
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.info("No data to display in summary.")

def render_enhanced_extraction_interface():
    """Enhanced extraction interface with advanced options"""
    st.subheader("🔬 Enhanced Government Document Extraction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_length = st.slider("Minimum Length", 20, 200, 50)
        max_results = st.slider("Max Results per Type", 10, 100, 25)
    
    with col2:
        context_window = st.slider("Context Window", 50, 500, 200)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.6)
    
    with col3:
        extraction_scope = st.selectbox(
            "Extraction Scope",
            ["Full Document", "Sections Only", "Recommendations Only", "Responses Only"]
        )
    
    # Document selection
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    doc_names = [doc.get('filename', f'Document {i+1}') for i, doc in enumerate(uploaded_docs)]
    
    selected_docs = st.multiselect(
        "📄 Select Documents",
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
    
    for i, doc in enumerate(docs):
        if doc.get('filename') not in selected_docs:
            continue
            
        filename = doc.get('filename', f'Document {i+1}')
        status_text.text(f"🔬 Processing {filename}...")
        progress_bar.progress((i + 1) / len([d for d in docs if d.get('filename') in selected_docs]))
        
        content = get_document_content_for_extraction(doc)
        if not content:
            continue
        
        # Extract recommendations
        doc_recommendations = extract_with_patterns(
            content, patterns['recommendation_patterns'], 'recommendation', 
            min_length, max_results, context_window, filename
        )
        
        # Extract responses
        doc_responses = extract_with_patterns(
            content, patterns['response_patterns'], 'response',
            min_length, max_results, context_window, filename
        )
        
        # Filter by confidence
        doc_recommendations = [r for r in doc_recommendations if r.get('confidence', 0) >= confidence_threshold]
        doc_responses = [r for r in doc_responses if r.get('confidence', 0) >= confidence_threshold]
        
        all_recommendations.extend(doc_recommendations)
        all_responses.extend(doc_responses)
    
    # Store results
    store_extraction_results(all_recommendations, all_responses, 'enhanced_government')
    
    status_text.text("✅ Enhanced extraction completed!")
    progress_bar.progress(1.0)
    
    # Display results
    display_extraction_results()

def render_rag_extraction_interface_fallback():
    """Fallback RAG interface when RAG components not available"""
    st.subheader("🤖 AI-Powered Extraction")
    st.info("🚧 Advanced RAG extraction not available. Please install required dependencies or use pattern-based extraction.")
    
    st.markdown("""
    **To enable RAG extraction:**
    ```bash
    pip install sentence-transformers scikit-learn transformers torch
    ```
    """)

def render_extraction_method_tabs():
    """Render different extraction method tabs"""
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Smart Complete", 
        "🔬 Enhanced Gov", 
        "⚡ Basic Pattern", 
        "🤖 AI-Powered"
    ])
    
    with tab1:
        render_smart_extraction_interface()
    
    with tab2:
        render_enhanced_extraction_interface()
    
    with tab3:
        render_basic_pattern_interface()
    
    with tab4:
        if RAG_COMPONENTS_AVAILABLE:
            render_rag_extraction_interface()
        else:
            render_rag_extraction_interface_fallback()

def get_extraction_statistics():
    """Get comprehensive extraction statistics"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        return None
    
    stats = {
        'total_methods': len(results),
        'total_recommendations': 0,
        'total_responses': 0,
        'methods_used': list(results.keys()),
        'latest_extraction': None
    }
    
    latest_time = None
    for method, data in results.items():
        stats['total_recommendations'] += len(data.get('recommendations', []))
        stats['total_responses'] += len(data.get('responses', []))
        
        extraction_time = data.get('timestamp')
        if extraction_time and (not latest_time or extraction_time > latest_time):
            latest_time = extraction_time
            stats['latest_extraction'] = method
    
    return stats

def export_all_extraction_results():
    """Export all extraction results"""
    results = st.session_state.get('extraction_results', {})
    
    if not results:
        st.warning("No extraction results to export.")
        return
    
    # Combine all results
    all_data = []
    for method, data in results.items():
        for item_type, items in [('recommendation', data.get('recommendations', [])), 
                                ('response', data.get('responses', []))]:
            for item in items:
                all_data.append({
                    'Method': method,
                    'Type': item_type,
                    'Text': item.get('text', ''),
                    'Confidence': item.get('confidence', 0),
                    'Source': item.get('source', ''),
                    'Document': item.get('source_document', ''),
                    'Position': item.get('position', ''),
                    'Length': len(item.get('text', '')),
                    'Timestamp': data.get('timestamp', '')
                })
    
    if all_data:
        df = pd.DataFrame(all_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            "📥 Download All Results (CSV)",
            csv,
            f"all_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
        
        # JSON export
        json_data = {
            'export_timestamp': datetime.now().isoformat(),
            'extraction_methods': list(results.keys()),
            'total_items': len(all_data),
            'results': results
        }
        
        import json
        json_str = json.dumps(json_data, indent=2, default=str)
        
        st.download_button(
            "📥 Download All Results (JSON)",
            json_str,
            f"all_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )

def render_extraction_dashboard():
    """Render extraction results dashboard"""
    st.subheader("📊 Extraction Dashboard")
    
    stats = get_extraction_statistics()
    
    if not stats:
        st.info("No extraction results available. Run an extraction first.")
        return
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Methods Used", stats['total_methods'])
    with col2:
        st.metric("Total Recommendations", stats['total_recommendations'])
    with col3:
        st.metric("Total Responses", stats['total_responses'])
    with col4:
        st.metric("Latest Method", stats['latest_extraction'] or "None")
    
    # Export options
    st.subheader("📥 Export Options")
    export_all_extraction_results()
    
    # Method comparison
    if stats['total_methods'] > 1:
        st.subheader("🔄 Method Comparison")
        
        results = st.session_state.get('extraction_results', {})
        comparison_data = []
        
        for method, data in results.items():
            comparison_data.append({
                'Method': method,
                'Recommendations': len(data.get('recommendations', [])),
                'Responses': len(data.get('responses', [])),
                'Total Items': len(data.get('recommendations', [])) + len(data.get('responses', [])),
                'Timestamp': data.get('timestamp', '')
            })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

# Export the main functions for the module
__all__ = [
    'render_extraction_tab',
    'SmartExtractor', 
    'get_document_content_for_extraction',
    'validate_documents_for_extraction',
    'get_enhanced_extraction_patterns',
    'extract_with_patterns',
    'store_extraction_results',
    'display_extraction_results',
    'process_smart_extraction',
    'process_basic_extraction',
    'render_extraction_method_tabs',
    'render_extraction_dashboard',
    'export_all_extraction_results'
]
