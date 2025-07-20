# ===============================================
# COMPLETE PDF EXTRACTION FIX - REVISED WITH CONCERN SPLITTING FIXES
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
        """Standard patterns for coroner concerns - IMPROVED to capture complete sections"""
        
        # Enhanced patterns that capture more complete concern sections
        patterns = [
            # Main concern section patterns - capture everything until a clear section break
            r"(?:THE\s+)?(?:CORONER|CORONER'S)\s+(?:CONCERNS?|CONCERN)\s+(?:ARE|IS)?:?\s*(.*?)(?=(?:ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS?|YOUR\s+RESPONSE|COPIES|SIGNED|DATED\s+THIS|FURTHER\s+ACTION|PREVENTION\s+OF\s+FUTURE\s+DEATHS)\b|$)",
            r"(?:MATTERS?|MATTER)\s+OF\s+(?:CONCERNS?|CONCERN)\s+(?:ARE|IS)?:?\s*(.*?)(?=(?:ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS?|YOUR\s+RESPONSE|COPIES|SIGNED|DATED\s+THIS|FURTHER\s+ACTION|PREVENTION\s+OF\s+FUTURE\s+DEATHS)\b|$)",
            
            # Alternative patterns for different document formats
            r"(?:CONCERNS?|MATTERS?\s+OF\s+CONCERN)\s+(?:RAISED|IDENTIFIED|NOTED)?:?\s*(.*?)(?=(?:ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS?|YOUR\s+RESPONSE|RECOMMENDATIONS?|COPIES|SIGNED|DATED\s+THIS)\b|$)",
            
            # Pattern for documents with numbered concern sections
            r"CONCERN\s+(?:NO\.?\s*)?(\d+)[\.\:\s]+(.*?)(?=CONCERN\s+(?:NO\.?\s*)?\d+|ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS?|$)",
        ]
        
        concerns = []
        for pattern in patterns:
            try:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    if len(match.groups()) >= 1:
                        concern_text = match.group(-1).strip()  # Get the last capture group
                        
                        if concern_text and len(concern_text) > 50:  # Increased minimum length
                            cleaned = self._clean_concern_text(concern_text)
                            if cleaned and len(cleaned) > 30:
                                concerns.append({
                                    'text': cleaned,
                                    'type': 'coroner_concern_complete',
                                    'method': 'standard_pattern_enhanced',
                                    'pattern': pattern[:50] + "...",
                                    'captured_length': len(cleaned)
                                })
            except re.error as e:
                self.logger.warning(f"Regex error in standard patterns: {e}")
                continue
        
        return concerns
    
    def _extract_flexible_patterns(self, content: str) -> List[Dict]:
        """Flexible patterns - FIXED to keep concerns together"""
        sections = re.split(r'\n\s*\n', content)
        concerns = []
        
        concern_indicators = [
            r'concern.*?about', r'matter.*?of.*?concern', r'issue.*?identified',
            r'problem.*?with', r'deficiency.*?in', r'failure.*?to',
            r'inadequate.*?(?:provision|system|process)', r'insufficient.*?(?:resources|training)',
            r'lack.*?of.*?(?:oversight|training|resources)', r'poor.*?(?:communication|coordination)'
        ]
        
        for section in sections:
            # Check if this section contains concern indicators
            has_concern_indicators = False
            for indicator in concern_indicators:
                if re.search(indicator, section, re.IGNORECASE):
                    has_concern_indicators = True
                    break
            
            if has_concern_indicators and len(section.strip()) > 50:
                # Keep the ENTIRE section together instead of splitting by sentences
                cleaned_section = self._clean_concern_text(section.strip())
                if cleaned_section and len(cleaned_section) > 30:
                    concerns.append({
                        'text': cleaned_section,
                        'type': 'identified_concern_section',
                        'method': 'flexible_pattern_unified',
                        'indicators_found': [ind for ind in concern_indicators 
                                           if re.search(ind, section, re.IGNORECASE)]
                    })
        
        return concerns
    
    def _extract_section_patterns(self, content: str) -> List[Dict]:
        """Extract from structured sections - FIXED to group related items"""
        concerns = []
        
        # Look for concern sections with headers
        concern_section_patterns = [
            r'(?:CONCERNS?|MATTERS?\s+OF\s+CONCERN|ISSUES?\s+IDENTIFIED)[:\s]*\n((?:(?:\d+[\.\)]|\w+[\.\)]|[•\-\*]|\w+\s*:).*?\n?)+)',
            r'(?:THE\s+)?(?:CORONER|CORONER\'S)\s+(?:CONCERNS?|MATTERS?\s+OF\s+CONCERN)[:\s]*\n((?:(?:\d+[\.\)]|\w+[\.\)]|[•\-\*]).*?\n?)+)',
        ]
        
        for pattern in concern_section_patterns:
            try:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for match in matches:
                    section_content = match.group(1).strip()
                    
                    # Keep the entire section together instead of splitting into individual items
                    if len(section_content) > 50:
                        cleaned_section = self._clean_concern_text(section_content)
                        if cleaned_section:
                            concerns.append({
                                'text': cleaned_section,
                                'type': 'structured_concern_section',
                                'method': 'section_detection_unified',
                                'format': 'multi_item_section'
                            })
            except re.error:
                continue
        
        # If no section headers found, look for numbered/bulleted lists but group them
        if not concerns:
            concerns.extend(self._extract_grouped_list_items(content))
        
        return concerns
    
    def _extract_grouped_list_items(self, content: str) -> List[Dict]:
        """Extract list items but group related ones together"""
        concerns = []
        
        # Look for groups of numbered/bulleted items that might be concerns
        list_group_patterns = [
            # Numbered items (1. 2. 3. etc.) - capture entire groups
            r'((?:\d+[\.\)]\s*[^0-9]+?(?=\d+[\.\)]|$))+)',
            # Lettered items (a. b. c. etc.) - capture entire groups
            r'((?:[a-z][\.\)]\s*[^a-z\.\)]+?(?=[a-z][\.\)]|$))+)',
            # Bulleted items (• - * etc.) - capture entire groups
            r'((?:[•\-\*]\s*[^•\-\*\n]+?(?=[•\-\*]|$))+)',
        ]
        
        for pattern in list_group_patterns:
            try:
                matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    group_text = match.group(1).strip()
                    
                    # Check if this group contains concern-related content
                    concern_keywords = ['concern', 'issue', 'problem', 'risk', 'deficiency', 'failure', 
                                      'inadequate', 'insufficient', 'lacking', 'poor', 'unsafe']
                    
                    keyword_count = sum(1 for keyword in concern_keywords 
                                      if keyword in group_text.lower())
                    
                    # If the group has enough concern keywords and sufficient length, keep it together
                    if keyword_count >= 2 and len(group_text) > 100:
                        cleaned_group = self._clean_concern_text(group_text)
                        if cleaned_group:
                            concerns.append({
                                'text': cleaned_group,
                                'type': 'grouped_list_concern',
                                'method': 'section_detection_grouped',
                                'keyword_matches': keyword_count,
                                'format': 'grouped_list'
                            })
            except re.error:
                continue
        
        return concerns
    
    def _extract_keyword_patterns(self, content: str) -> List[Dict]:
        """Keyword-based extraction - FIXED to capture complete paragraphs"""
        concern_keywords = [
            'inadequate', 'insufficient', 'failure', 'breach', 'deficient',
            'lacking', 'poor', 'substandard', 'unsafe', 'risk', 'problem',
            'issue', 'concern', 'deficiency', 'shortcoming', 'weakness'
        ]
        
        # Split into paragraphs instead of sentences to keep context together
        paragraphs = re.split(r'\n\s*\n', content)
        concerns = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 100:  # Skip very short paragraphs
                continue
                
            # Count concern keywords in the paragraph
            keyword_count = sum(1 for keyword in concern_keywords 
                              if keyword.lower() in paragraph.lower())
            
            # If paragraph has multiple concern keywords, treat it as a complete concern
            if keyword_count >= 2:
                cleaned_paragraph = self._clean_concern_text(paragraph)
                if cleaned_paragraph and len(cleaned_paragraph) > 50:
                    # Find which keywords matched
                    matched_keywords = [keyword for keyword in concern_keywords 
                                      if keyword.lower() in paragraph.lower()]
                    
                    concerns.append({
                        'text': cleaned_paragraph,
                        'type': 'keyword_concern_paragraph',
                        'method': 'keyword_extraction_unified',
                        'keyword_matches': keyword_count,
                        'matched_keywords': matched_keywords
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
        """Process and deduplicate concerns - IMPROVED deduplication"""
        if not concerns:
            return []
        
        # Add metadata to all concerns
        for i, concern in enumerate(concerns):
            concern.update({
                'id': f"{document_name}_concern_{i+1}",
                'document_source': document_name,
                'confidence_score': self._calculate_confidence(concern['text']),
                'text_length': len(concern['text']),
                'word_count': len(concern['text'].split()),
                'paragraph_count': len([p for p in concern['text'].split('\n\n') if p.strip()]),
                'extracted_at': datetime.now().isoformat()
            })
        
        # Enhanced deduplication that considers semantic similarity, not just word overlap
        unique_concerns = []
        
        for concern in concerns:
            is_duplicate = False
            concern_words = set(concern['text'].lower().split())
            
            for existing in unique_concerns:
                existing_words = set(existing['text'].lower().split())
                
                # Calculate Jaccard similarity
                intersection = len(concern_words & existing_words)
                union = len(concern_words | existing_words)
                similarity = intersection / union if union > 0 else 0
                
                # Also check for substring containment (one concern contained in another)
                text1_clean = concern['text'].lower().strip()
                text2_clean = existing['text'].lower().strip()
                
                containment = (text1_clean in text2_clean) or (text2_clean in text1_clean)
                
                # Mark as duplicate if high similarity OR one contains the other
                if similarity > 0.6 or containment:
                    is_duplicate = True
                    # Keep the longer, more detailed version
                    if len(concern['text']) > len(existing['text']):
                        # Replace existing with this more detailed version
                        unique_concerns.remove(existing)
                        unique_concerns.append(concern)
                    break
            
            if not is_duplicate:
                unique_concerns.append(concern)
        
        # Sort by confidence and text length (prefer longer, more detailed concerns)
        unique_concerns.sort(key=lambda x: (x['confidence_score'], x['text_length']), reverse=True)
        
        return unique_concerns
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score - IMPROVED scoring"""
        if not text:
            return 0.0
        
        base_score = 0.5
        
        # Length indicators (longer text usually means more complete concern)
        text_length = len(text)
        if text_length > 200: base_score += 0.1
        if text_length > 500: base_score += 0.1
        if text_length > 1000: base_score += 0.1
        
        # Paragraph structure (multiple paragraphs suggest complete concern)
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        if paragraph_count > 1: base_score += 0.1
        if paragraph_count > 2: base_score += 0.1
        
        # Content indicators
        concern_words = ['concern', 'issue', 'problem', 'risk', 'deficiency', 'failure',
                        'inadequate', 'insufficient', 'lacking', 'poor', 'unsafe']
        word_matches = sum(1 for word in concern_words if word in text.lower())
        base_score += min(word_matches * 0.05, 0.2)  # Cap at 0.2 bonus
        
        # Structure indicators
        if re.search(r'\d+[\.\)]', text): base_score += 0.05  # Numbered points
        if re.search(r'[A-Z][a-z]+:', text): base_score += 0.05  # Section headers
        if re.search(r'(?:because|due to|as a result|therefore)', text, re.IGNORECASE): base_score += 0.05  # Causal language
        
        # Penalize very short text (likely fragments)
        if text_length < 100: base_score -= 0.2
        if text_length < 50: base_score -= 0.3
        
        return min(base_score, 1.0)  # Cap at 1.0


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
                                st.write(f"**Length:** {len(concern['text'])} chars")
                                st.write(f"**Text:** {concern['text'][:300]}...")
                
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
