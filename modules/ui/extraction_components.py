# ===============================================
# FILE: modules/ui/extraction_components.py (COMPLETE REVISED VERSION)
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

# Enhanced extraction utilities
class EnhancedConcernExtractor:
    """Enhanced concern extraction with multiple robust methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_concerns_robust(self, content: str, document_name: str = "") -> Dict:
        """Extract concerns using multiple robust methods"""
        if not content or len(content.strip()) < 10:
            return {
                'concerns': [],
                'debug_info': {
                    'error': 'No content or content too short',
                    'content_length': len(content) if content else 0
                }
            }
        
        # Normalize content
        normalized_content = self._normalize_content(content)
        
        # Try different extraction methods
        all_concerns = []
        debug_info = {'methods_tried': [], 'results': {}}
        
        # Method 1: Standard patterns
        try:
            standard_concerns = self._extract_standard_patterns(normalized_content)
            debug_info['methods_tried'].append('standard_patterns')
            debug_info['results']['standard_patterns'] = len(standard_concerns)
            all_concerns.extend(standard_concerns)
        except Exception as e:
            debug_info['results']['standard_patterns'] = f"Error: {e}"
        
        # Method 2: Flexible patterns
        try:
            flexible_concerns = self._extract_flexible_patterns(normalized_content)
            debug_info['methods_tried'].append('flexible_patterns')
            debug_info['results']['flexible_patterns'] = len(flexible_concerns)
            all_concerns.extend(flexible_concerns)
        except Exception as e:
            debug_info['results']['flexible_patterns'] = f"Error: {e}"
        
        # Method 3: Section detection
        try:
            section_concerns = self._extract_section_patterns(normalized_content)
            debug_info['methods_tried'].append('section_detection')
            debug_info['results']['section_detection'] = len(section_concerns)
            all_concerns.extend(section_concerns)
        except Exception as e:
            debug_info['results']['section_detection'] = f"Error: {e}"
        
        # Method 4: Keyword extraction
        try:
            keyword_concerns = self._extract_keyword_patterns(normalized_content)
            debug_info['methods_tried'].append('keyword_extraction')
            debug_info['results']['keyword_extraction'] = len(keyword_concerns)
            all_concerns.extend(keyword_concerns)
        except Exception as e:
            debug_info['results']['keyword_extraction'] = f"Error: {e}"
        
        # Remove duplicates and add metadata
        unique_concerns = self._process_concerns(all_concerns, document_name)
        
        debug_info['final_stats'] = {
            'total_raw': len(all_concerns),
            'unique': len(unique_concerns),
            'duplicates_removed': len(all_concerns) - len(unique_concerns)
        }
        
        return {
            'concerns': unique_concerns,
            'debug_info': debug_info
        }
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for better pattern matching"""
        if not content:
            return ""
        
        # Fix common OCR issues
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
        content = re.sub(r'(\w)(\d)', r'\1 \2', content)
        content = re.sub(r'(\d)(\w)', r'\1 \2', content)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def _extract_standard_patterns(self, content: str) -> List[Dict]:
        """Standard coroner concern patterns"""
        patterns = [
            r"CORONER'S\s+CONCERNS?:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|$)",
            r"MATTERS?\s+OF\s+CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|$)",
            r"The\s+MATTERS?\s+OF\s+CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|$)",
            r"CORONER'S\s+CONCERNS?\s+are:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|$)",
            r"MATTERS?\s+OF\s+CONCERN\s+are:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|$)",
            # Handle OCR issues
            r"(?:CORONER'S|CORONERS)\s*(?:CONCERNS?|CONCERN):?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|$)",
            r"(?:MATTERS?|MATTER)\s*OF\s*(?:CONCERNS?|CONCERN):?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|$)",
        ]
        
        concerns = []
        for pattern in patterns:
            try:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    concern_text = match.group(1).strip()
                    if concern_text and len(concern_text) > 20:
                        cleaned = self._clean_concern_text(concern_text)
                        if cleaned:
                            concerns.append({
                                'text': cleaned,
                                'type': 'coroner_concern',
                                'method': 'standard_pattern',
                                'pattern': pattern[:50] + "..."
                            })
            except re.error as e:
                self.logger.warning(f"Regex error: {e}")
                continue
        
        return concerns
    
    def _extract_flexible_patterns(self, content: str) -> List[Dict]:
        """Flexible patterns for different document formats"""
        sections = re.split(r'\n\s*\n', content)
        concerns = []
        
        concern_indicators = [
            r'concern.*?about', r'matter.*?of.*?concern', r'issue.*?identified',
            r'problem.*?with', r'deficiency.*?in', r'failure.*?to',
            r'inadequate.*?(?:provision|system|process)', r'insufficient.*?(?:resources|training)',
            r'lack.*?of.*?(?:oversight|training|resources)', r'poor.*?(?:communication|coordination)'
        ]
        
        for section in sections:
            for indicator in concern_indicators:
                if re.search(indicator, section, re.IGNORECASE):
                    sentences = re.split(r'[.!?]+', section)
                    for sentence in sentences:
                        if (re.search(indicator, sentence, re.IGNORECASE) and 
                            len(sentence.strip()) > 30):
                            concerns.append({
                                'text': sentence.strip(),
                                'type': 'identified_concern',
                                'method': 'flexible_pattern',
                                'indicator': indicator
                            })
        
        return concerns
    
    def _extract_section_patterns(self, content: str) -> List[Dict]:
        """Extract from structured sections (lists, etc.)"""
        concerns = []
        
        list_patterns = [
            r'(\d+[\.\)])\s*([^0-9]+?)(?=\d+[\.\)]|$)',
            r'([•\-\*])\s*([^•\-\*\n]+?)(?=[•\-\*]|$)',
            r'([a-z][\.\)])\s*([^a-z\.\)]+?)(?=[a-z][\.\)]|$)',
        ]
        
        for pattern in list_patterns:
            try:
                matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    item_text = match.group(2).strip()
                    if (len(item_text) > 30 and 
                        any(word in item_text.lower() for word in 
                            ['concern', 'issue', 'problem', 'risk', 'deficiency', 'failure'])):
                        concerns.append({
                            'text': item_text,
                            'type': 'structured_concern',
                            'method': 'section_detection',
                            'format': 'numbered' if match.group(1)[0].isdigit() else 'bulleted'
                        })
            except re.error:
                continue
        
        return concerns
    
    def _extract_keyword_patterns(self, content: str) -> List[Dict]:
        """Keyword-based extraction as fallback"""
        concern_keywords = [
            'inadequate', 'insufficient', 'failure', 'breach', 'deficient',
            'lacking', 'poor', 'substandard', 'unsafe', 'risk', 'problem',
            'issue', 'concern', 'deficiency', 'shortcoming', 'weakness'
        ]
        
        sentences = re.split(r'[.!?]+', content)
        concerns = []
        
        for sentence in sentences:
            keyword_count = sum(1 for keyword in concern_keywords 
                              if keyword in sentence.lower())
            
            if keyword_count >= 2 and len(sentence.strip()) > 40:
                concerns.append({
                    'text': sentence.strip(),
                    'type': 'keyword_concern',
                    'method': 'keyword_extraction',
                    'keyword_matches': keyword_count
                })
        
        return concerns
    
    def _clean_concern_text(self, text: str) -> str:
        """Clean extracted concern text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common footer text
        text = re.sub(r'YOU ARE UNDER A DUTY.*$', '', text, re.IGNORECASE)
        text = re.sub(r'COPIES.*$', '', text, re.IGNORECASE)
        text = re.sub(r'SIGNED:.*$', '', text, re.IGNORECASE)
        
        return text.strip()
    
    def _process_concerns(self, concerns: List[Dict], document_name: str) -> List[Dict]:
        """Process and deduplicate concerns"""
        if not concerns:
            return []
        
        # Add metadata
        for i, concern in enumerate(concerns):
            concern.update({
                'id': f"{document_name}_{i}",
                'document_source': document_name,
                'confidence_score': self._calculate_confidence(concern['text']),
                'text_length': len(concern['text']),
                'word_count': len(concern['text'].split()),
                'extracted_at': datetime.now().isoformat()
            })
        
        # Remove duplicates
        unique_concerns = []
        for concern in concerns:
            is_duplicate = False
            concern_words = set(concern['text'].lower().split())
            
            for existing in unique_concerns:
                existing_words = set(existing['text'].lower().split())
                similarity = len(concern_words & existing_words) / len(concern_words | existing_words)
                
                if similarity > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if concern['confidence_score'] > existing['confidence_score']:
                        unique_concerns.remove(existing)
                        unique_concerns.append(concern)
                    break
            
            if not is_duplicate:
                unique_concerns.append(concern)
        
        # Sort by confidence
        unique_concerns.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return unique_concerns
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted concern"""
        if not text:
            return 0.0
        
        base_score = 0.5
        
        # Length indicators
        if len(text) > 100: base_score += 0.1
        if len(text) > 300: base_score += 0.1
        
        # Content indicators
        concern_words = ['concern', 'issue', 'problem', 'risk', 'deficiency', 'failure']
        word_matches = sum(1 for word in concern_words if word in text.lower())
        base_score += word_matches * 0.05
        
        # Structure indicators
        if re.search(r'\d+\.', text): base_score += 0.05
        if re.search(r'[A-Z][a-z]+:', text): base_score += 0.05
        if re.search(r'(?:should|must|ought to)', text, re.IGNORECASE): base_score += 0.05
        
        return min(base_score, 1.0)

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
    - 🔧 **Debug Tools**: Comprehensive debugging for extraction issues
    """)
    
    # Document status check first
    render_document_status_check()
    
    # Extraction configuration
    render_extraction_configuration()
    
    # Document selection and extraction
    render_extraction_interface()
    
    # Enhanced concern extraction section
    render_enhanced_concern_extraction()
    
    # Debug extraction issues
    render_extraction_debug_tools()
    
    # Display results
    display_extraction_results()

def render_document_status_check():
    """Check and display document status for extraction"""
    st.subheader("📋 Document Status Check")
    
    docs_with_content = []
    docs_with_issues = []
    
    for doc in st.session_state.uploaded_documents:
        content = doc.get('content', '')
        content_length = len(content)
        
        if content_length == 0:
            docs_with_issues.append((doc['filename'], "No content", "❌"))
        elif content_length < 100:
            docs_with_issues.append((doc['filename'], f"Very short ({content_length} chars)", "⚠️"))
        else:
            docs_with_content.append((doc['filename'], f"{content_length:,} characters", "✅"))
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", len(st.session_state.uploaded_documents))
    
    with col2:
        st.metric("Ready for Extraction", len(docs_with_content))
    
    with col3:
        st.metric("Need Attention", len(docs_with_issues))
    
    # Status details
    if docs_with_issues:
        with st.expander("⚠️ Documents with Issues", expanded=True):
            for filename, issue, status in docs_with_issues:
                st.write(f"{status} **{filename}**: {issue}")
            
            if st.button("🔧 Try Enhanced PDF Re-processing"):
                st.info("Enhanced PDF re-processing would go here (requires original file access)")
    
    if docs_with_content:
        with st.expander("✅ Documents Ready for Extraction"):
            for filename, status, icon in docs_with_content:
                st.write(f"{icon} **{filename}**: {status}")

def render_extraction_debug_tools():
    """Render debugging tools for extraction issues"""
    st.markdown("---")
    st.subheader("🔧 Extraction Debug Tools")
    
    if not st.session_state.get('uploaded_documents'):
        st.info("Upload documents first to use debug tools.")
        return
    
    # Debug options
    with st.expander("🔍 Debug Document Content", expanded=False):
        doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
        selected_debug_doc = st.selectbox("Select document to debug:", doc_options, key="debug_doc")
        
        if selected_debug_doc and st.button("🔍 Analyze Document Content"):
            debug_document_content(selected_debug_doc)
    
    # Test extraction patterns
    with st.expander("🧪 Test Extraction Patterns", expanded=False):
        if st.button("🧪 Test Current Patterns on All Documents"):
            test_extraction_patterns()
    
    # Enhanced extraction test
    with st.expander("🚀 Enhanced Extraction Test", expanded=False):
        st.markdown("Test the enhanced extraction methods on your documents:")
        
        test_confidence = st.slider("Test Confidence Threshold", 0.0, 1.0, 0.3, 0.05, key="test_confidence")
        
        if st.button("🚀 Run Enhanced Extraction Test"):
            run_enhanced_extraction_test(test_confidence)

def debug_document_content(document_name: str):
    """Debug specific document content"""
    doc = next((d for d in st.session_state.uploaded_documents 
               if d['filename'] == document_name), None)
    
    if not doc:
        st.error(f"Document {document_name} not found")
        return
    
    st.write(f"**Debugging Document:** {document_name}")
    
    content = doc.get('content', '')
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Content Length", f"{len(content):,} chars")
    
    with col2:
        word_count = len(content.split()) if content else 0
        st.metric("Word Count", f"{word_count:,}")
    
    with col3:
        line_count = content.count('\n') if content else 0
        st.metric("Line Count", line_count)
    
    if not content:
        st.error("❌ **No content found in this document!**")
        st.markdown("""
        **Possible causes:**
        - PDF is a scanned image (needs OCR)
        - PDF is password protected
        - PDF extraction failed during upload
        - File is corrupted
        """)
        return
    
    # Content analysis
    st.subheader("📊 Content Analysis")
    
    # Look for concern-related keywords
    concern_keywords = ['concern', 'coroner', 'matter', 'issue', 'problem', 'recommendation', 'failure', 'inadequate']
    found_keywords = []
    
    for keyword in concern_keywords:
        count = content.lower().count(keyword)
        if count > 0:
            found_keywords.append(f"{keyword} ({count})")
    
    if found_keywords:
        st.success(f"✅ **Found relevant keywords:** {', '.join(found_keywords)}")
    else:
        st.warning("⚠️ **No relevant keywords found**")
    
    # Sample content around concerns
    concern_pos = content.lower().find('concern')
    if concern_pos > -1:
        st.subheader("📝 Sample Content Around 'Concern'")
        start = max(0, concern_pos - 200)
        end = min(len(content), concern_pos + 400)
        sample = content[start:end]
        st.text_area("", value=sample, height=150, disabled=True)
    else:
        st.subheader("📝 Document Preview (First 500 chars)")
        st.text_area("", value=content[:500], height=150, disabled=True)
    
    # Pattern test
    st.subheader("🔍 Pattern Matching Test")
    
    # Test standard extraction
    try:
        from core_utils import extract_concern_text
        standard_result = extract_concern_text(content)
        
        if standard_result:
            st.success(f"✅ **Standard extraction found:** {len(standard_result)} characters")
            with st.expander("View extracted text"):
                st.write(standard_result[:300] + "..." if len(standard_result) > 300 else standard_result)
        else:
            st.warning("⚠️ **Standard extraction found nothing**")
    except Exception as e:
        st.error(f"❌ **Standard extraction failed:** {e}")
    
    # Test enhanced extraction
    try:
        extractor = EnhancedConcernExtractor()
        result = extractor.extract_concerns_robust(content, document_name)
        concerns = result['concerns']
        debug_info = result['debug_info']
        
        if concerns:
            st.success(f"✅ **Enhanced extraction found:** {len(concerns)} concerns")
            
            for i, concern in enumerate(concerns[:3]):  # Show first 3
                with st.expander(f"Enhanced Concern {i+1} (Confidence: {concern['confidence_score']:.2f})"):
                    st.write(f"**Method:** {concern['method']}")
                    st.write(f"**Type:** {concern['type']}")
                    st.write(f"**Text:** {concern['text'][:200]}...")
        else:
            st.warning("⚠️ **Enhanced extraction found no concerns**")
        
        # Show debug info
        with st.expander("🔍 Enhanced Extraction Debug Info"):
            st.json(debug_info)
            
    except Exception as e:
        st.error(f"❌ **Enhanced extraction failed:** {e}")

def test_extraction_patterns():
    """Test extraction patterns on all documents"""
    st.write("**Testing extraction patterns on all documents...**")
    
    results = []
    
    for doc in st.session_state.uploaded_documents:
        filename = doc['filename']
        content = doc.get('content', '')
        
        result = {
            'document': filename,
            'content_length': len(content),
            'has_content': len(content) > 0,
            'standard_extraction': 'Not tested',
            'enhanced_extraction': 'Not tested',
            'keywords_found': []
        }
        
        if not content:
            result['standard_extraction'] = 'No content'
            result['enhanced_extraction'] = 'No content'
            results.append(result)
            continue
        
        # Test keywords
        keywords = ['concern', 'coroner', 'matter', 'issue', 'problem']
        for keyword in keywords:
            if keyword in content.lower():
                result['keywords_found'].append(keyword)
        
        # Test standard extraction
        try:
            from core_utils import extract_concern_text
            standard_result = extract_concern_text(content)
            result['standard_extraction'] = f"Found {len(standard_result)} chars" if standard_result else "No concerns found"
        except Exception as e:
            result['standard_extraction'] = f"Error: {str(e)}"
        
        # Test enhanced extraction
        try:
            extractor = EnhancedConcernExtractor()
            enhanced_result = extractor.extract_concerns_robust(content, filename)
            concerns_count = len(enhanced_result['concerns'])
            result['enhanced_extraction'] = f"Found {concerns_count} concerns" if concerns_count > 0 else "No concerns found"
        except Exception as e:
            result['enhanced_extraction'] = f"Error: {str(e)}"
        
        results.append(result)
    
    # Display results
    for result in results:
        with st.expander(f"📄 {result['document']} - {result['content_length']:,} chars"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Standard Extraction:** {result['standard_extraction']}")
                st.write(f"**Keywords Found:** {', '.join(result['keywords_found']) if result['keywords_found'] else 'None'}")
            
            with col2:
                st.write(f"**Enhanced Extraction:** {result['enhanced_extraction']}")
                st.write(f"**Has Content:** {'✅' if result['has_content'] else '❌'}")

def run_enhanced_extraction_test(confidence_threshold: float):
    """Run enhanced extraction test on all documents"""
    st.write("**Running enhanced extraction test...**")
    
    extractor = EnhancedConcernExtractor()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_concerns = []
    doc_results = []
    
    docs_with_content = [doc for doc in st.session_state.uploaded_documents if doc.get('content')]
    
    for i, doc in enumerate(docs_with_content):
        progress = (i + 1) / len(docs_with_content)
        progress_bar.progress(progress)
        status_text.text(f"Testing {doc['filename']}...")
        
        try:
            result = extractor.extract_concerns_robust(doc['content'], doc['filename'])
            concerns = result['concerns']
            
            # Filter by confidence
            filtered_concerns = [c for c in concerns if c['confidence_score'] >= confidence_threshold]
            
            all_concerns.extend(filtered_concerns)
            
            doc_results.append({
                'document': doc['filename'],
                'total_concerns': len(concerns),
                'filtered_concerns': len(filtered_concerns),
                'methods_tried': len(result['debug_info'].get('methods_tried', [])),
                'status': 'success'
            })
            
        except Exception as e:
            doc_results.append({
                'document': doc['filename'],
                'total_concerns': 0,
                'filtered_concerns': 0,
                'methods_tried': 0,
                'status': f'Error: {str(e)}'
            })
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    st.success(f"🎉 **Test Complete!** Found {len(all_concerns)} concerns (confidence ≥ {confidence_threshold})")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        successful_docs = len([r for r in doc_results if r['status'] == 'success'])
        st.metric("Documents Processed", f"{successful_docs}/{len(docs_with_content)}")
    
    with col2:
        st.metric("Total Concerns Found", len(all_concerns))
    
    with col3:
        avg_per_doc = len(all_concerns) / successful_docs if successful_docs > 0 else 0
        st.metric("Average per Document", f"{avg_per_doc:.1f}")
    
    # Results table
    results_df = pd.DataFrame(doc_results)
    st.dataframe(results_df, use_container_width=True)
    
    # Sample concerns
    if all_concerns:
        st.subheader("📋 Sample Extracted Concerns")
        for i, concern in enumerate(all_concerns[:3]):
            with st.expander(f"Sample {i+1} - {concern['document_source']} (Confidence: {concern['confidence_score']:.2f})"):
                st.write(f"**Method:** {concern['method']}")
                st.write(f"**Type:** {concern['type']}")
                st.write(f"**Text:** {concern['text'][:300]}...")
        
        if st.button("💾 Save These Results to Session"):
            st.session_state.extracted_concerns = all_concerns
            st.success("✅ Results saved to session state!")

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
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Standard Extraction", type="primary", use_container_width=True):
                extract_content_from_documents(selected_docs)
        
        with col2:
            if st.button("🎯 Enhanced Extraction Only", type="secondary", use_container_width=True):
                run_enhanced_extraction_only(selected_docs)
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

def run_enhanced_extraction_only(selected_docs: List[str]):
    """Run only enhanced extraction on selected documents"""
    if not selected_docs:
        st.warning("No documents selected for extraction.")
        return
    
    extractor = EnhancedConcernExtractor()
    
    # Get settings
    confidence_threshold = st.session_state.get('confidence_threshold', 0.6)
    max_extractions = st.session_state.get('max_extractions', 50)
    
    # Progress tracking
    progress_container = st.container()
    status_container = st.container()
    
    all_concerns = []
    processing_results = []
    
    for i, doc_name in enumerate(selected_docs):
        current_step = i + 1
        
        # Update progress
        with progress_container:
            show_progress_indicator(current_step, len(selected_docs), f"Enhanced extraction: {doc_name}")
        
        with status_container:
            status_text = st.empty()
            status_text.info(f"🎯 Processing: {doc_name}")
        
        try:
            # Find document
            doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
            if not doc:
                status_text.error(f"❌ Document not found: {doc_name}")
                continue
            
            content = doc.get('content', '')
            if not content:
                status_text.warning(f"⚠️ No content in {doc_name}")
                processing_results.append({
                    'document': doc_name,
                    'concerns_found': 0,
                    'status': 'no_content'
                })
                continue
            
            # Run enhanced extraction
            result = extractor.extract_concerns_robust(content, doc_name)
            concerns = result['concerns']
            
            # Filter by confidence and limit
            filtered_concerns = [c for c in concerns if c['confidence_score'] >= confidence_threshold]
            filtered_concerns = filtered_concerns[:max_extractions]
            
            all_concerns.extend(filtered_concerns)
            
            processing_results.append({
                'document': doc_name,
                'concerns_found': len(filtered_concerns),
                'total_found': len(concerns),
                'methods_tried': len(result['debug_info'].get('methods_tried', [])),
                'status': 'success'
            })
            
            status_text.success(f"✅ Found {len(filtered_concerns)} concerns in {doc_name}")
            
        except Exception as e:
            error_msg = f"Enhanced extraction error for {doc_name}: {str(e)}"
            add_error_message(error_msg)
            processing_results.append({
                'document': doc_name,
                'concerns_found': 0,
                'status': 'error',
                'error': str(e)
            })
            status_text.error(f"❌ Error processing {doc_name}")
            logging.error(f"Enhanced extraction error: {e}", exc_info=True)
    
    # Update session state
    st.session_state.extracted_concerns = all_concerns
    st.session_state.last_extraction_results = processing_results
    st.session_state.last_processing_time = datetime.now().isoformat()
    
    # Clear progress displays
    progress_container.empty()
    status_container.empty()
    
    # Show results
    show_enhanced_only_results(all_concerns, processing_results)

def show_enhanced_only_results(concerns: List[Dict], processing_results: List[Dict]):
    """Show results of enhanced-only extraction"""
    st.success("🎯 Enhanced Extraction Complete!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        successful_docs = len([r for r in processing_results if r['status'] == 'success'])
        st.metric("Documents Processed", f"{successful_docs}/{len(processing_results)}")
    
    with col2:
        st.metric("Concerns Found", len(concerns))
    
    with col3:
        if concerns:
            avg_confidence = sum(c['confidence_score'] for c in concerns) / len(concerns)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        else:
            st.metric("Avg Confidence", "0.00")
    
    with col4:
        docs_with_concerns = len([r for r in processing_results if r['concerns_found'] > 0])
        st.metric("Docs with Concerns", docs_with_concerns)
    
    # Processing details
    if processing_results:
        st.subheader("📊 Processing Details")
        results_df = pd.DataFrame(processing_results)
        
        # Add status icons
        status_map = {
            'success': '✅ Success',
            'no_content': '⚠️ No Content',
            'error': '❌ Error'
        }
        
        if 'status' in results_df.columns:
            results_df['Status'] = results_df['status'].map(status_map)
        
        st.dataframe(results_df, use_container_width=True)
    
    # Sample concerns
    if concerns:
        st.subheader("🎯 Sample Extracted Concerns")
        for i, concern in enumerate(concerns[:5]):  # Show first 5
            with st.expander(f"Concern {i+1} - {concern['document_source']} (Confidence: {concern['confidence_score']:.2f})"):
                st.write(f"**Method:** {concern['method']}")
                st.write(f"**Type:** {concern['type']}")
                st.write(f"**Text:** {concern['text'][:300]}...")
                
                if 'indicator' in concern:
                    st.write(f"**Indicator:** {concern['indicator']}")

def run_enhanced_concern_extraction(documents: List[Dict]):
    """Run enhanced concern extraction using improved patterns"""
    
    # Get settings
    min_confidence = st.session_state.get('enhanced_confidence', 0.6)
    extract_metadata_flag = st.session_state.get('enhanced_metadata', True)
    show_debug = st.session_state.get('show_debug', False)
    only_concerns = st.session_state.get('only_concerns', False)
    
    # Initialize enhanced extractor
    extractor = EnhancedConcernExtractor()
    
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
                    extraction_debug.append({
                        'document': doc_name,
                        'extracted': False,
                        'reason': 'No content'
                    })
                    continue
                
                # Use enhanced concern extraction
                result = extractor.extract_concerns_robust(content, doc_name)
                concerns = result['concerns']
                debug_info = result['debug_info']
                
                # Filter by confidence
                good_concerns = [c for c in concerns if c['confidence_score'] >= min_confidence]
                
                if good_concerns:
                    # Add metadata if requested
                    for concern in good_concerns:
                        if extract_metadata_flag:
                            try:
                                metadata = extract_metadata(content)
                                concern['metadata'] = metadata
                            except:
                                concern['metadata'] = {}
                    
                    extracted_concerns.extend(good_concerns)
                    
                    extraction_debug.append({
                        'document': doc_name,
                        'content_length': len(content),
                        'concerns_found': len(good_concerns),
                        'total_found': len(concerns),
                        'methods_tried': debug_info.get('methods_tried', []),
                        'extracted': True
                    })
                    
                    status_text.success(f"✅ Extracted {len(good_concerns)} concerns from {doc_name}")
                else:
                    extraction_debug.append({
                        'document': doc_name,
                        'content_length': len(content),
                        'concerns_found': 0,
                        'total_found': len(concerns),
                        'methods_tried': debug_info.get('methods_tried', []),
                        'extracted': False,
                        'reason': f"No concerns met confidence threshold ({min_confidence})"
                    })
                    status_text.warning(f"⚠️ No high-confidence concerns found in {doc_name}")
                
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
        
        # Method breakdown
        method_counts = {}
        for concern in concerns:
            method = concern.get('method', 'Unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            st.write("**Extraction Methods Used:**")
            method_cols = st.columns(len(method_counts))
            for i, (method, count) in enumerate(method_counts.items()):
                with method_cols[i]:
                    st.metric(method.replace('_', ' ').title(), count)
        
        # Preview concerns
        with st.expander("📋 Preview Extracted Concerns", expanded=True):
            for i, concern in enumerate(concerns[:3]):  # Show first 3
                st.write(f"**Concern {i+1}** ({concern['document_source']}):")
                preview_text = concern['text'][:200] + "..." if len(concern['text']) > 200 else concern['text']
                st.write(preview_text)
                st.write(f"*Method: {concern['method']}, Confidence: {concern['confidence_score']:.2f}*")
                
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
    
    # Initialize extractors
    llm_extractor = LLMRecommendationExtractor()
    enhanced_extractor = EnhancedConcernExtractor()
    
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
                
                content = doc.get('content', '')
                if not content:
                    status_text.warning(f"⚠️ No content in {doc_name}")
                    processing_results.append({
                        'document': doc_name,
                        'recommendations_found': 0,
                        'concerns_found': 0,
                        'enhanced_concerns': 0,
                        'standard_concerns': 0,
                        'status': 'no_content'
                    })
                    continue
                
                # Enhanced extraction for concerns if enabled
                enhanced_concerns = []
                if extract_concerns and use_enhanced_extraction:
                    try:
                        result = enhanced_extractor.extract_concerns_robust(content, doc_name)
                        raw_concerns = result['concerns']
                        
                        # Filter enhanced concerns
                        for concern in raw_concerns:
                            if (concern['confidence_score'] >= confidence_threshold and 
                                len(concern['text']) >= min_text_length):
                                
                                # Reformat for compatibility
                                enhanced_concern = {
                                    'id': concern['id'],
                                    'text': concern['text'],
                                    'document_source': doc_name,
                                    'confidence_score': concern['confidence_score'],
                                    'extraction_method': 'enhanced_pattern',
                                    'category': 'coroner_concern',
                                    'type': 'concern',
                                    'method': concern['method'],
                                    'timestamp': concern.get('extracted_at', datetime.now().isoformat())
                                }
                                
                                # Add metadata if extraction is enabled
                                if st.session_state.get('extract_metadata', True):
                                    try:
                                        metadata = extract_metadata(content)
                                        enhanced_concern['metadata'] = metadata
                                    except:
                                        enhanced_concern['metadata'] = {}
                                
                                enhanced_concerns.append(enhanced_concern)
                                
                    except Exception as e:
                        logging.warning(f"Enhanced extraction failed for {doc_name}: {e}")
                
                # Standard LLM extraction
                standard_recommendations = []
                standard_concerns = []
                
                try:
                    extraction_result = llm_extractor.extract_recommendations_and_concerns(
                        content, 
                        doc_name
                    )
                    
                    raw_recommendations = extraction_result.get('recommendations', [])
                    raw_concerns = extraction_result.get('concerns', [])
                    
                    # Apply filtering to recommendations
                    for rec in raw_recommendations:
                        if (rec.confidence_score >= confidence_threshold and 
                            len(rec.text) >= min_text_length):
                            standard_recommendations.append(rec)
                    
                    # Apply filtering to standard concerns
                    if extract_concerns:
                        for concern in raw_concerns:
                            if (concern.get('confidence_score', 0) >= confidence_threshold and 
                                len(concern.get('text', '')) >= min_text_length):
                                standard_concerns.append(concern)
                
                except Exception as e:
                    logging.warning(f"Standard extraction failed for {doc_name}: {e}")
                
                # Combine all results
                all_doc_concerns = enhanced_concerns + standard_concerns
                
                # Limit extractions
                standard_recommendations = standard_recommendations[:max_extractions]
                all_doc_concerns = all_doc_concerns[:max_extractions]
                
                # Store results
                all_recommendations.extend(standard_recommendations)
                all_concerns.extend(all_doc_concerns)
                
                processing_results.append({
                    'document': doc_name,
                    'recommendations_found': len(standard_recommendations),
                    'concerns_found': len(all_doc_concerns),
                    'enhanced_concerns': len(enhanced_concerns),
                    'standard_concerns': len(standard_concerns),
                    'status': 'success'
                })
                
                concern_summary = f"{len(all_doc_concerns)} concerns ({len(enhanced_concerns)} enhanced + {len(standard_concerns)} standard)"
                status_text.success(f"✅ Extracted {len(standard_recommendations)} recommendations, {concern_summary} from {doc_name}")
                
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
                'no_content': '⚠️ No Content',
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
    
    # Show documents with no content
    no_content_docs = [r for r in processing_results if r['status'] == 'no_content']
    if no_content_docs:
        with st.expander("📄 Documents with No Content"):
            st.warning("These documents had no extractable content:")
            for result in no_content_docs:
                st.write(f"• {result['document']}")
            st.info("💡 These documents may need re-processing with enhanced PDF extraction or OCR.")

# Keep all existing display functions unchanged
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
        ["Table View", "Detailed Cards", "Method Comparison"],
        horizontal=True,
        key="concern_display_mode"
    )
    
    if display_mode == "Table View":
        display_concerns_table(filtered_concerns)
    elif display_mode == "Detailed Cards":
        display_concerns_cards(filtered_concerns)
    elif display_mode == "Method Comparison":
        display_method_comparison_view(filtered_concerns)

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

def display_method_comparison_view(concerns: List[Dict]):
    """Display method comparison view"""
    st.subheader("📊 Extraction Method Comparison")
    
    if not concerns:
        st.info("No concerns available for method comparison.")
        return
    
    # Group by extraction method
    method_groups = {}
    for concern in concerns:
        method = concern.get('extraction_method', 'Unknown')
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append(concern)
    
    # Create comparison metrics
    comparison_data = []
    for method, method_concerns in method_groups.items():
        total_confidence = sum(c.get('confidence_score', 0) for c in method_concerns)
        avg_confidence = total_confidence / len(method_concerns) if method_concerns else 0
        avg_length = sum(len(c.get('text', '')) for c in method_concerns) / len(method_concerns) if method_concerns else 0
        with_metadata = len([c for c in method_concerns if c.get('metadata')])
        
        comparison_data.append({
            'Method': method,
            'Count': len(method_concerns),
            'Avg Confidence': f"{avg_confidence:.2f}",
            'Avg Length': f"{avg_length:.0f}",
            'With Metadata': f"{with_metadata}/{len(method_concerns)}",
            'Metadata %': f"{(with_metadata / len(method_concerns)) * 100:.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Show samples from each method
    st.subheader("🔍 Sample Concerns by Method")
    
    for method, method_concerns in method_groups.items():
        with st.expander(f"{method} - {len(method_concerns)} concerns"):
            # Show top 2 concerns by confidence
            top_concerns = sorted(method_concerns, key=lambda x: x.get('confidence_score', 0), reverse=True)[:2]
            
            for i, concern in enumerate(top_concerns):
                st.write(f"**Sample {i+1}** (Confidence: {concern.get('confidence_score', 0):.2f})")
                st.write(f"**Source:** {concern.get('document_source', 'Unknown')}")
                preview = concern.get('text', '')[:200] + "..." if len(concern.get('text', '')) > 200 else concern.get('text', '')
                st.write(f"**Text:** {preview}")
                st.write("---")

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
            
            if concern.get('method'):
                st.write(f"• **Sub-method:** {concern.get('method', 'Unknown')}")
        
        with col2:
            st.markdown("**Metrics:**")
            st.write(f"**ID:** {concern.get('id', 'N/A')}")
            st.write(f"**Source:** {concern.get('document_source', 'Unknown')}")
            st.write(f"**Confidence:** {concern.get('confidence_score', 0):.2f}")
            st.write(f"**Category:** {concern.get('category', 'General')}")
            st.write(f"**Length:** {len(concern.get('text', ''))} characters")
            st.write(f"**Word Count:** {concern.get('word_count', 'N/A')}")
            
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
            'Sub_Method': concern.get('method', ''),
            'Text_Length': len(concern.get('text', '')),
            'Word_Count': concern.get('word_count', ''),
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

# Emergency fix function for immediate use
def render_emergency_extraction_fix():
    """Emergency fix for extraction failures - can be called separately"""
    st.markdown("---")
    st.subheader("🚨 Emergency Extraction Fix")
    
    st.markdown("""
    **Having extraction problems?** This emergency tool can help diagnose and fix issues 
    with the failing extractions you're experiencing.
    """)
    
    if not st.session_state.get('uploaded_documents'):
        st.info("Upload documents first to use this emergency fix.")
        return
    
    # Quick diagnosis
    st.write("**📋 Quick Document Diagnosis:**")
    
    docs_with_issues = []
    docs_ok = []
    
    for doc in st.session_state.uploaded_documents:
        filename = doc['filename']
        content = doc.get('content', '')
        content_length = len(content)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"📄 {filename}")
        
        with col2:
            if content_length == 0:
                st.error("No content")
                docs_with_issues.append(doc)
            elif content_length < 100:
                st.warning(f"{content_length} chars")
                docs_with_issues.append(doc)
            else:
                st.success(f"{content_length:,} chars")
                docs_ok.append(doc)
        
        with col3:
            # Quick check for concern-related words
            if content and any(word in content.lower() for word in ['concern', 'coroner', 'matter', 'issue']):
                st.info("✓ Keywords")
            else:
                st.warning("No keywords")
    
    # Show summary and emergency extraction
    if docs_with_issues:
        st.error(f"🚨 **Found {len(docs_with_issues)} documents with extraction issues**")
        
        for doc in docs_with_issues:
            st.write(f"• {doc['filename']}: No readable content extracted")
        
        st.markdown("""
        **Possible causes:**
        - PDFs are scanned images (need OCR)
        - PDFs are password protected  
        - PDF extraction failed during upload
        - Files are corrupted
        """)
    
    if docs_ok:
        st.success(f"✅ **{len(docs_ok)} documents have extractable content**")
        
        # Emergency enhanced extraction
        st.markdown("### 🚀 Emergency Enhanced Extraction")
        
        emergency_confidence = st.slider("Emergency Extraction Confidence", 0.0, 1.0, 0.3, 0.05)
        
        if st.button("🚀 Run Emergency Enhanced Extraction", type="primary"):
            run_emergency_enhanced_extraction(docs_ok, emergency_confidence)

def run_emergency_enhanced_extraction(docs_with_content: List[Dict], confidence_threshold: float):
    """Run emergency enhanced extraction"""
    st.write("**🚨 Running emergency enhanced extraction...**")
    
    extractor = EnhancedConcernExtractor()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_concerns = []
    emergency_results = []
    
    for i, doc in enumerate(docs_with_content):
        progress = (i + 1) / len(docs_with_content)
        progress_bar.progress(progress)
        status_text.text(f"Emergency processing: {doc['filename']}...")
        
        try:
            content = doc.get('content', '')
            
            # Run enhanced extraction
            result = extractor.extract_concerns_robust(content, doc['filename'])
            concerns = result['concerns']
            debug_info = result['debug_info']
            
            # Filter by confidence
            good_concerns = [c for c in concerns if c['confidence_score'] >= confidence_threshold]
            
            all_concerns.extend(good_concerns)
            
            emergency_results.append({
                'document': doc['filename'],
                'total_found': len(concerns),
                'filtered_found': len(good_concerns),
                'methods_tried': len(debug_info.get('methods_tried', [])),
                'status': 'success' if good_concerns else 'no_concerns'
            })
            
            if good_concerns:
                status_text.success(f"✅ Emergency extraction found {len(good_concerns)} concerns in {doc['filename']}")
            else:
                status_text.warning(f"⚠️ No concerns found in {doc['filename']}")
                
        except Exception as e:
            emergency_results.append({
                'document': doc['filename'],
                'total_found': 0,
                'filtered_found': 0,
                'methods_tried': 0,
                'status': f'Error: {str(e)}'
            })
            status_text.error(f"❌ Emergency extraction failed for {doc['filename']}")
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Show emergency results
    if all_concerns:
        st.success(f"🎉 **Emergency Extraction Successful!** Found {len(all_concerns)} concerns")
        
        # Quick save option
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Emergency Results", type="primary"):
                st.session_state.extracted_concerns = all_concerns
                st.success("✅ Emergency results saved!")
        
        with col2:
            if st.button("📥 Download Emergency Results"):
                export_emergency_results(all_concerns)
        
        # Show sample results
        st.subheader("📋 Emergency Extraction Samples")
        for i, concern in enumerate(all_concerns[:3]):
            with st.expander(f"Emergency Result {i+1} - {concern['document_source']} (Confidence: {concern['confidence_score']:.2f})"):
                st.write(f"**Method:** {concern['method']}")
                st.write(f"**Text:** {concern['text'][:200]}...")
        
        # Results table
        results_df = pd.DataFrame(emergency_results)
        st.dataframe(results_df, use_container_width=True)
        
    else:
        st.error("😞 **Emergency extraction found no concerns**")
        st.markdown("""
        **Emergency troubleshooting:**
        1. Check if documents contain the expected content
        2. Try lowering the confidence threshold further
        3. Documents may need OCR processing for scanned images
        4. Consider manual content verification
        """)

def export_emergency_results(concerns: List[Dict]):
    """Export emergency extraction results"""
    export_data = []
    for concern in concerns:
        export_data.append({
            'Emergency_ID': concern.get('id', ''),
            'Text': concern.get('text', ''),
            'Source_Document': concern.get('document_source', ''),
            'Confidence_Score': concern.get('confidence_score', 0),
            'Method': concern.get('method', ''),
            'Type': concern.get('type', ''),
            'Text_Length': len(concern.get('text', '')),
            'Emergency_Timestamp': datetime.now().isoformat()
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Emergency Results",
        data=csv,
        file_name=f"emergency_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Add this to the end of your render_extraction_tab function
# Just add this line at the very end:
# render_emergency_extraction_fix()

# Additional utility function for quick testing
def quick_test_document_extraction():
    """Quick test function for document extraction"""
    st.markdown("---")
    st.subheader("🧪 Quick Extraction Test")
    
    if not st.session_state.get('uploaded_documents'):
        st.info("Upload documents first to test extraction.")
        return
    
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    test_doc = st.selectbox("Select document to test:", doc_options, key="quick_test_doc")
    
    if test_doc and st.button("🧪 Quick Test"):
        doc = next((d for d in st.session_state.uploaded_documents 
                   if d['filename'] == test_doc), None)
        
        if doc:
            content = doc.get('content', '')
            
            if not content:
                st.error("❌ No content in this document")
                return
            
            st.success(f"✅ Document has {len(content):,} characters")
            
            # Test different extraction methods
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Standard Extraction Test:**")
                try:
                    standard_result = extract_concern_text(content)
                    if standard_result:
                        st.success(f"✅ Found {len(standard_result)} chars")
                        with st.expander("Preview"):
                            st.write(standard_result[:200] + "...")
                    else:
                        st.warning("⚠️ No results")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            
            with col2:
                st.write("**Enhanced Extraction Test:**")
                try:
                    extractor = EnhancedConcernExtractor()
                    enhanced_result = extractor.extract_concerns_robust(content, test_doc)
                    concerns = enhanced_result['concerns']
                    
                    if concerns:
                        st.success(f"✅ Found {len(concerns)} concerns")
                        best_concern = max(concerns, key=lambda x: x['confidence_score'])
                        with st.expander("Best Result"):
                            st.write(f"**Confidence:** {best_concern['confidence_score']:.2f}")
                            st.write(f"**Method:** {best_concern['method']}")
                            st.write(f"**Text:** {best_concern['text'][:200]}...")
                    else:
                        st.warning("⚠️ No concerns found")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Final note: To use this revised file, simply replace your existing 
# modules/ui/extraction_components.py with this complete code.
# 
# The key improvements:
# 1. ✅ Enhanced document status checking
# 2. ✅ Robust multi-method concern extraction 
# 3. ✅ Comprehensive debugging tools
# 4. ✅ Emergency extraction fix for failed documents
# 5. ✅ Better error handling and user feedback
# 6. ✅ Enhanced export options
# 7. ✅ Backward compatibility with existing functions
#
# This should resolve your extraction failures and provide much better
# debugging capabilities to understand what's happening with your PDFs.

# END OF FILE - COMPLETE EXTRACTION COMPONENTSbutton(
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
            'Sub_Method': concern.get('method', ''),
            'Timestamp': concern.get('timestamp', '')
        }
        
        # Add metadata as separate columns
        if concern.get('metadata'):
            for key, value in concern['metadata'].items():
                row_data[f'metadata_{key}'] = value
        
        export_data.append(row_data)
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    # REPLACE the incomplete "st.download_" line in your v6 code with this:

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

# Emergency fix function for immediate use
def render_emergency_extraction_fix():
    """Emergency fix for extraction failures - can be called separately"""
    st.markdown("---")
    st.subheader("🚨 Emergency Extraction Fix")
    
    st.markdown("""
    **Having extraction problems?** This emergency tool can help diagnose and fix issues 
    with the failing extractions you're experiencing.
    """)
    
    if not st.session_state.get('uploaded_documents'):
        st.info("Upload documents first to use this emergency fix.")
        return
    
    # Quick diagnosis
    st.write("**📋 Quick Document Diagnosis:**")
    
    docs_with_issues = []
    docs_ok = []
    
    for doc in st.session_state.uploaded_documents:
        filename = doc['filename']
        content = doc.get('content', '')
        content_length = len(content)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"📄 {filename}")
        
        with col2:
            if content_length == 0:
                st.error("No content")
                docs_with_issues.append(doc)
            elif content_length < 100:
                st.warning(f"{content_length} chars")
                docs_with_issues.append(doc)
            else:
                st.success(f"{content_length:,} chars")
                docs_ok.append(doc)
        
        with col3:
            # Quick check for concern-related words
            if content and any(word in content.lower() for word in ['concern', 'coroner', 'matter', 'issue']):
                st.info("✓ Keywords")
            else:
                st.warning("No keywords")
    
    # Show summary and emergency extraction
    if docs_with_issues:
        st.error(f"🚨 **Found {len(docs_with_issues)} documents with extraction issues**")
        
        for doc in docs_with_issues:
            st.write(f"• {doc['filename']}: No readable content extracted")
        
        st.markdown("""
        **Possible causes:**
        - PDFs are scanned images (need OCR)
        - PDFs are password protected  
        - PDF extraction failed during upload
        - Files are corrupted
        """)
    
    if docs_ok:
        st.success(f"✅ **{len(docs_ok)} documents have extractable content**")
        
        # Emergency enhanced extraction
        st.markdown("### 🚀 Emergency Enhanced Extraction")
        
        emergency_confidence = st.slider("Emergency Extraction Confidence", 0.0, 1.0, 0.3, 0.05)
        
        if st.button("🚀 Run Emergency Enhanced Extraction", type="primary"):
            run_emergency_enhanced_extraction(docs_ok, emergency_confidence)

def run_emergency_enhanced_extraction(docs_with_content: List[Dict], confidence_threshold: float):
    """Run emergency enhanced extraction"""
    st.write("**🚨 Running emergency enhanced extraction...**")
    
    extractor = EnhancedConcernExtractor()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_concerns = []
    emergency_results = []
    
    for i, doc in enumerate(docs_with_content):
        progress = (i + 1) / len(docs_with_content)
        progress_bar.progress(progress)
        status_text.text(f"Emergency processing: {doc['filename']}...")
        
        try:
            content = doc.get('content', '')
            
            # Run enhanced extraction
            result = extractor.extract_concerns_robust(content, doc['filename'])
            concerns = result['concerns']
            debug_info = result['debug_info']
            
            # Filter by confidence
            good_concerns = [c for c in concerns if c['confidence_score'] >= confidence_threshold]
            
            all_concerns.extend(good_concerns)
            
            emergency_results.append({
                'document': doc['filename'],
                'total_found': len(concerns),
                'filtered_found': len(good_concerns),
                'methods_tried': len(debug_info.get('methods_tried', [])),
                'status': 'success' if good_concerns else 'no_concerns'
            })
            
            if good_concerns:
                status_text.success(f"✅ Emergency extraction found {len(good_concerns)} concerns in {doc['filename']}")
            else:
                status_text.warning(f"⚠️ No concerns found in {doc['filename']}")
                
        except Exception as e:
            emergency_results.append({
                'document': doc['filename'],
                'total_found': 0,
                'filtered_found': 0,
                'methods_tried': 0,
                'status': f'Error: {str(e)}'
            })
            status_text.error(f"❌ Emergency extraction failed for {doc['filename']}")
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Show emergency results
    if all_concerns:
        st.success(f"🎉 **Emergency Extraction Successful!** Found {len(all_concerns)} concerns")
        
        # Quick save option
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Emergency Results", type="primary"):
                st.session_state.extracted_concerns = all_concerns
                st.success("✅ Emergency results saved!")
        
        with col2:
            if st.button("📥 Download Emergency Results"):
                export_emergency_results(all_concerns)
        
        # Show sample results
        st.subheader("📋 Emergency Extraction Samples")
        for i, concern in enumerate(all_concerns[:3]):
            with st.expander(f"Emergency Result {i+1} - {concern['document_source']} (Confidence: {concern['confidence_score']:.2f})"):
                st.write(f"**Method:** {concern['method']}")
                st.write(f"**Text:** {concern['text'][:200]}...")
        
        # Results table
        results_df = pd.DataFrame(emergency_results)
        st.dataframe(results_df, use_container_width=True)
        
    else:
        st.error("😞 **Emergency extraction found no concerns**")
        st.markdown("""
        **Emergency troubleshooting:**
        1. Check if documents contain the expected content
        2. Try lowering the confidence threshold further
        3. Documents may need OCR processing for scanned images
        4. Consider manual content verification
        """)

def export_emergency_results(concerns: List[Dict]):
    """Export emergency extraction results"""
    export_data = []
    for concern in concerns:
        export_data.append({
            'Emergency_ID': concern.get('id', ''),
            'Text': concern.get('text', ''),
            'Source_Document': concern.get('document_source', ''),
            'Confidence_Score': concern.get('confidence_score', 0),
            'Method': concern.get('method', ''),
            'Type': concern.get('type', ''),
            'Text_Length': len(concern.get('text', '')),
            'Emergency_Timestamp': datetime.now().isoformat()
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Emergency Results",
        data=csv,
        file_name=f"emergency_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Additional utility function for quick testing
def quick_test_document_extraction():
    """Quick test function for document extraction"""
    st.markdown("---")
    st.subheader("🧪 Quick Extraction Test")
    
    if not st.session_state.get('uploaded_documents'):
        st.info("Upload documents first to test extraction.")
        return
    
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    test_doc = st.selectbox("Select document to test:", doc_options, key="quick_test_doc")
    
    if test_doc and st.button("🧪 Quick Test"):
        doc = next((d for d in st.session_state.uploaded_documents 
                   if d['filename'] == test_doc), None)
        
        if doc:
            content = doc.get('content', '')
            
            if not content:
                st.error("❌ No content in this document")
                return
            
            st.success(f"✅ Document has {len(content):,} characters")
            
            # Test different extraction methods
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Standard Extraction Test:**")
                try:
                    standard_result = extract_concern_text(content)
                    if standard_result:
                        st.success(f"✅ Found {len(standard_result)} chars")
                        with st.expander("Preview"):
                            st.write(standard_result[:200] + "...")
                    else:
                        st.warning("⚠️ No results")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            
            with col2:
                st.write("**Enhanced Extraction Test:**")
                try:
                    extractor = EnhancedConcernExtractor()
                    enhanced_result = extractor.extract_concerns_robust(content, test_doc)
                    concerns = enhanced_result['concerns']
                    
                    if concerns:
                        st.success(f"✅ Found {len(concerns)} concerns")
                        best_concern = max(concerns, key=lambda x: x['confidence_score'])
                        with st.expander("Best Result"):
                            st.write(f"**Confidence:** {best_concern['confidence_score']:.2f}")
                            st.write(f"**Method:** {best_concern['method']}")
                            st.write(f"**Text:** {best_concern['text'][:200]}...")
                    else:
                        st.warning("⚠️ No concerns found")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
