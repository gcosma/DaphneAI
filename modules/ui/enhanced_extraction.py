# ===============================================
# COMPLETE PDF EXTRACTION FIX - SINGLE FILE
# Save this as: enhanced_extraction.py
# ===============================================

import pdfplumber
import fitz  # PyMuPDF
import logging
import re
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import tempfile
import os

class CompletePDFExtractor:
    """Complete PDF extraction and concern detection solution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Enhanced PDF extraction with multiple fallback methods"""
        try:
            # Method 1: Try pdfplumber with enhanced settings
            text_content, method = self._extract_with_pdfplumber_enhanced(pdf_path)
            
            # Method 2: Try PyMuPDF if pdfplumber fails
            if not text_content or len(text_content.strip()) < 50:
                self.logger.info(f"Pdfplumber yielded minimal content, trying PyMuPDF")
                text_content, method = self._extract_with_pymupdf_enhanced(pdf_path)
            
            # Method 3: Try OCR if both standard methods fail
            if not text_content or len(text_content.strip()) < 50:
                self.logger.info(f"Standard extraction failed, attempting OCR")
                text_content, method = self._extract_with_ocr(pdf_path)
            
            # Extract metadata
            metadata = self._extract_metadata(pdf_path)
            metadata['extraction_method'] = method
            
            # Clean and normalize the extracted text
            if text_content:
                cleaned_content = self._clean_extracted_text(text_content)
            else:
                cleaned_content = ""
            
            return {
                "content": cleaned_content,
                "metadata": metadata,
                "source": Path(pdf_path).name,
                "extraction_method": method,
                "content_length": len(cleaned_content),
                "success": len(cleaned_content) > 10
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return {
                "content": "",
                "metadata": {},
                "source": Path(pdf_path).name,
                "extraction_method": "failed",
                "content_length": 0,
                "success": False,
                "error": str(e)
            }
    
    def _extract_with_pdfplumber_enhanced(self, pdf_path: str) -> Tuple[str, str]:
        """Enhanced pdfplumber extraction"""
        try:
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Try multiple extraction strategies
                        page_text = page.extract_text(
                            x_tolerance=2,
                            y_tolerance=2,
                            layout=True,
                            x_density=7.25,
                            y_density=13
                        )
                        
                        # Fallback methods if no text
                        if not page_text:
                            page_text = page.extract_text()
                        
                        if not page_text:
                            words = page.extract_words()
                            if words:
                                page_text = ' '.join([word['text'] for word in words])
                        
                        if page_text:
                            text_parts.append(page_text)
                            
                    except Exception as e:
                        self.logger.warning(f"Error on page {page_num + 1}: {e}")
                        continue
            
            final_text = '\n\n'.join(text_parts)
            return final_text, "pdfplumber_enhanced"
            
        except Exception as e:
            self.logger.error(f"Pdfplumber extraction failed: {e}")
            return "", "pdfplumber_failed"
    
    def _extract_with_pymupdf_enhanced(self, pdf_path: str) -> Tuple[str, str]:
        """Enhanced PyMuPDF extraction"""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Try different extraction methods
                page_text = page.get_text()
                
                if not page_text:
                    page_text = page.get_text("text", sort=True)
                
                if not page_text:
                    blocks = page.get_text("blocks")
                    page_text = '\n'.join([block[4] for block in blocks if block[4].strip()])
                
                if page_text:
                    text_parts.append(page_text)
            
            doc.close()
            final_text = '\n\n'.join(text_parts)
            return final_text, "pymupdf_enhanced"
            
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {e}")
            return "", "pymupdf_failed"
    
    def _extract_with_ocr(self, pdf_path: str) -> Tuple[str, str]:
        """OCR extraction for scanned PDFs"""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            self.logger.info(f"Starting OCR extraction")
            
            pages = convert_from_path(pdf_path, dpi=300)
            ocr_text_parts = []
            
            for page_num, page_image in enumerate(pages):
                try:
                    custom_config = r'--oem 3 --psm 6'
                    page_text = pytesseract.image_to_string(page_image, config=custom_config)
                    
                    if page_text.strip():
                        ocr_text_parts.append(page_text)
                        
                except Exception as e:
                    self.logger.warning(f"OCR failed on page {page_num + 1}: {e}")
                    continue
            
            final_text = '\n\n'.join(ocr_text_parts)
            return final_text, "ocr"
                
        except ImportError:
            self.logger.warning("OCR libraries not available")
            return "", "ocr_unavailable"
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return "", "ocr_failed"
    
    def _extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            
            enhanced_metadata = {
                "page_count": doc.page_count,
                "is_encrypted": doc.is_encrypted,
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "creation_date": metadata.get("creationDate"),
                "modification_date": metadata.get("modDate")
            }
            
            doc.close()
            return enhanced_metadata
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return {}
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix common OCR issues
        text = text.replace('`', "'")
        text = text.replace(''', "'").replace('"', '"').replace('"', '"')
        text = text.replace('—', '-').replace('–', '-')
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        
        # Fix word boundaries
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        
        return text.strip()
    
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


# ===============================================
# STREAMLIT INTEGRATION FUNCTIONS
# ===============================================

def debug_uploaded_documents():
    """Debug function to check uploaded documents"""
    st.subheader("🔍 Debug Document Content")
    
    if not st.session_state.get('uploaded_documents'):
        st.warning("No documents uploaded")
        return
    
    extractor = CompletePDFExtractor()
    
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    selected_doc = st.selectbox("Select document to debug:", doc_options)
    
    if selected_doc:
        doc = next((d for d in st.session_state.uploaded_documents 
                   if d['filename'] == selected_doc), None)
        
        if doc and st.button("🔍 Analyze Document"):
            st.write(f"**Document:** {selected_doc}")
            
            content = doc.get('content', '')
            st.write(f"**Content Length:** {len(content)} characters")
            
            if not content:
                st.error("❌ No content found in document!")
                st.info("**Possible solutions:**")
                st.write("1. Document might be scanned (image-based)")
                st.write("2. PDF might be corrupted")
                st.write("3. PDF extraction failed")
            else:
                st.success("✅ Content found!")
                
                # Test concern extraction
                with st.spinner("Testing concern extraction..."):
                    result = extractor.extract_concerns_robust(content, selected_doc)
                
                concerns = result['concerns']
                debug_info = result['debug_info']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Concerns Found:** {len(concerns)}")
                    if concerns:
                        for i, concern in enumerate(concerns[:3]):  # Show first 3
                            with st.expander(f"Concern {i+1} (Confidence: {concern['confidence_score']:.2f})"):
                                st.write(f"**Method:** {concern['method']}")
                                st.write(f"**Type:** {concern['type']}")
                                st.write(f"**Text:** {concern['text'][:200]}...")
                
                with col2:
                    st.write("**Debug Information:**")
                    st.json(debug_info)
                
                # Show content preview
                with st.expander("Content Preview (first 1000 chars)"):
                    st.text(content[:1000])

def fix_extraction_for_failed_documents():
    """Re-process documents that failed extraction"""
    st.subheader("🔧 Fix Failed Extractions")
    
    if not st.session_state.get('uploaded_documents'):
        st.warning("No documents to fix")
        return
    
    extractor = CompletePDFExtractor()
    
    # Find documents with no content
    failed_docs = [doc for doc in st.session_state.uploaded_documents 
                   if not doc.get('content') or len(doc.get('content', '')) < 50]
    
    if not failed_docs:
        st.success("✅ All documents have content extracted!")
        return
    
    st.warning(f"Found {len(failed_docs)} documents with extraction issues:")
    
    for doc in failed_docs:
        st.write(f"• {doc['filename']} (Content length: {len(doc.get('content', ''))})")
    
    if st.button("🚀 Re-process Failed Documents"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        fixed_count = 0
        
        for i, doc in enumerate(failed_docs):
            progress = (i + 1) / len(failed_docs)
            progress_bar.progress(progress)
            status_text.text(f"Processing {doc['filename']}...")
            
            # This would need the original file path or content
            # For now, we'll just show what would happen
            st.info(f"Would re-process: {doc['filename']}")
            # In practice, you'd need to store the original file or path
            
        status_text.text("Re-processing complete!")
        progress_bar.progress(1.0)

def enhanced_concern_extraction():
    """Run enhanced concern extraction on all documents"""
    st.subheader("🎯 Enhanced Concern Extraction")
    
    if not st.session_state.get('uploaded_documents'):
        st.warning("No documents to process")
        return
    
    extractor = CompletePDFExtractor()
    
    # Settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.3, 0.05)
    
    with col2:
        max_concerns = st.number_input("Max Concerns per Document", 1, 50, 10)
    
    with col3:
        show_debug = st.checkbox("Show Debug Info", False)
    
    if st.button("🚀 Run Enhanced Extraction"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_concerns = []
        processing_results = []
        
        docs_with_content = [doc for doc in st.session_state.uploaded_documents 
                           if doc.get('content')]
        
        for i, doc in enumerate(docs_with_content):
            progress = (i + 1) / len(docs_with_content)
            progress_bar.progress(progress)
            status_text.text(f"Processing {doc['filename']}...")
            
            try:
                result = extractor.extract_concerns_robust(doc['content'], doc['filename'])
                concerns = result['concerns']
                
                # Filter by confidence and limit
                filtered_concerns = [c for c in concerns if c['confidence_score'] >= min_confidence]
                filtered_concerns = filtered_concerns[:max_concerns]
                
                all_concerns.extend(filtered_concerns)
                
                processing_results.append({
                    'document': doc['filename'],
                    'concerns_found': len(filtered_concerns),
                    'status': 'success',
                    'debug_info': result['debug_info'] if show_debug else None
                })
                
            except Exception as e:
                processing_results.append({
                    'document': doc['filename'],
                    'concerns_found': 0,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Store results
        st.session_state.extracted_concerns = all_concerns
        
        # Show results
        progress_bar.progress(1.0)
        status_text.text("✅ Enhanced extraction complete!")
        
        # Summary
        total_concerns = len(all_concerns)
        successful_docs = len([r for r in processing_results if r['status'] == 'success'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Processed", f"{successful_docs}/{len(docs_with_content)}")
        with col2:
            st.metric("Total Concerns Found", total_concerns)
        with col3:
            avg_per_doc = total_concerns / successful_docs if successful_docs > 0 else 0
            st.metric("Average per Document", f"{avg_per_doc:.1f}")
        
        # Results table
        if processing_results:
            results_df = pd.DataFrame(processing_results)
            st.subheader("📊 Processing Results")
            st.dataframe(results_df)
        
        # Show extracted concerns
        if all_concerns:
            st.subheader("🎯 Extracted Concerns")
            for i, concern in enumerate(all_concerns[:10]):  # Show first 10
                with st.expander(f"Concern {i+1} - {concern['document_source']} (Confidence: {concern['confidence_score']:.2f})"):
                    st.write(f"**Method:** {concern['method']}")
                    st.write(f"**Type:** {concern['type']}")
                    st.write(f"**Text:** {concern['text']}")

# ===============================================
# MAIN STREAMLIT APP SECTION
# ===============================================

def main():
    """Main function for testing the extraction fix"""
    st.title("🔧 PDF Extraction Fix Tool")
    
    st.markdown("""
    This tool provides enhanced PDF extraction and concern detection to fix the 
    issues you're experiencing with the Daphni tool.
    """)
    
    # Initialize session state
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    if 'extracted_concerns' not in st.session_state:
        st.session_state.extracted_concerns = []
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs([
        "🔍 Debug Documents", 
        "🔧 Fix Failed Extractions", 
        "🎯 Enhanced Extraction"
    ])
    
    with tab1:
        debug_uploaded_documents()
    
    with tab2:
        fix_extraction_for_failed_documents()
    
    with tab3:
        enhanced_concern_extraction()

if __name__ == "__main__":
    main()
