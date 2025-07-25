# ===============================================
# FILE: modules/enhanced_section_extractor.py (COMPLETE VERSION)
# ===============================================

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import io
import hashlib
from datetime import datetime

# PDF processing imports with fallbacks
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available - using fallback PDF processing")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available - using fallback PDF processing")

# Configure logging
logging.basicConfig(level=logging.INFO)

class EnhancedSectionExtractor:
    """
    Enhanced section extractor for government documents and inquiry reports.
    
    Supports universal inquiry document processing, not limited to any specific inquiry.
    Handles table-of-contents based documents, numbered sections, and government response formats.
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the enhanced section extractor"""
        self.debug_mode = debug_mode
        self.extraction_stats = {
            'documents_processed': 0,
            'sections_extracted': 0,
            'recommendations_found': 0,
            'responses_found': 0,
            'errors_encountered': 0
        }
        
        # Government document patterns (universal for any UK inquiry)
        self.government_patterns = {
            'inquiry_types': [
                r'(?i)(\w+(?:\s+\w+)*)\s+inquiry',
                r'(?i)(\w+(?:\s+\w+)*)\s+commission', 
                r'(?i)(\w+(?:\s+\w+)*)\s+investigation',
                r'(?i)(\w+(?:\s+\w+)*)\s+review'
            ],
            'section_headers': [
                r'(?i)^(?:\d+\.?\d*\s+)?(?:government\s+)?responses?\s+to\s+(?:individual\s+)?recommendations?',
                r'(?i)^(?:\d+\.?\d*\s+)?recommendations?(?:\s+and\s+responses?)?',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:the\s+)?government\'?s?\s+response',
                r'(?i)^(?:\d+\.?\d*\s+)?cabinet\s+office\s+response',
                r'(?i)^(?:\d+\.?\d*\s+)?departmental\s+responses?',
                r'(?i)^(?:\d+\.?\d*\s+)?implementation\s+(?:plan|strategy)',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:key\s+)?findings?\s+and\s+recommendations?',
                r'(?i)^(?:\d+\.?\d*\s+)?executive\s+summary'
            ],
            'recommendation_patterns': [
                r'(?i)(?:^|\n)\s*(?:recommendation\s+)?(\d+)[\.\):\s]+(.+?)(?=(?:^|\n)\s*(?:recommendation\s+)?\d+[\.\):\s]|$)',
                r'(?i)(?:^|\n)\s*(\d+)[\.\)]\s*(.+?)(?=(?:^|\n)\s*\d+[\.\)]|$)',
                r'(?i)(?:^|\n)\s*•\s*(.+?)(?=(?:^|\n)\s*•|$)'
            ],
            'response_patterns': [
                r'(?i)(?:government\s+)?response(?:\s+to\s+recommendation\s+(\d+))?[:\s]*(.+?)(?=(?:^|\n)(?:government\s+)?response|$)',
                r'(?i)(?:the\s+government\s+)?(?:accepts?|rejects?|agrees?|disagrees?)\s*[:\s]*(.+?)(?=(?:^|\n)|$)',
                r'(?i)(?:accepted|rejected|partially\s+accepted)[:\s]*(.+?)(?=(?:^|\n)|$)'
            ]
        }
    
    def extract_sections_from_pdf(self, pdf_content: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """
        Main method to extract sections from PDF content
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Original filename for reference
            
        Returns:
            Dictionary containing extracted sections and metadata
        """
        try:
            self.extraction_stats['documents_processed'] += 1
            
            # Try pdfplumber first, then PyMuPDF as fallback
            if PDFPLUMBER_AVAILABLE:
                result = self._extract_pages_with_pdfplumber(pdf_content, filename)
            elif PYMUPDF_AVAILABLE:
                result = self._extract_pages_with_pymupdf(pdf_content, filename)
            else:
                # Fallback to basic extraction
                result = self._extract_with_fallback(pdf_content, filename)
            
            if result and result.get('full_text'):
                # Process the extracted text for sections
                sections = self._find_recommendation_response_sections(result['full_text'])
                result['sections'] = sections
                result['section_count'] = len(sections)
                
                # Extract individual items
                recommendations = self.extract_individual_recommendations(result['full_text'])
                responses = self.extract_individual_responses(result['full_text'])
                
                result['individual_recommendations'] = recommendations
                result['individual_responses'] = responses
                
                # Update statistics
                self.extraction_stats['sections_extracted'] += len(sections)
                self.extraction_stats['recommendations_found'] += len(recommendations)
                self.extraction_stats['responses_found'] += len(responses)
                
                # Add document analysis
                result['document_analysis'] = self._analyze_government_document_structure(result['full_text'])
                
                if self.debug_mode:
                    logging.info(f"Extracted {len(sections)} sections, {len(recommendations)} recommendations, {len(responses)} responses from {filename}")
            
            return result
            
        except Exception as extraction_error:
            self.extraction_stats['errors_encountered'] += 1
            logging.error(f"Error extracting sections from {filename}: {extraction_error}")
            return {
                'success': False,
                'error': str(extraction_error),
                'filename': filename,
                'sections': [],
                'full_text': "",
                'page_count': 0
            }
    
    def _extract_pages_with_pdfplumber(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text using pdfplumber (primary method)"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                pages_text = []
                pages_metadata = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text with layout preservation
                        text = page.extract_text(layout=True)
                        if text:
                            pages_text.append(text)
                            
                            # Extract page metadata
                            page_metadata = {
                                'page_number': page_num,
                                'bbox': page.bbox,
                                'width': page.width,
                                'height': page.height,
                                'rotation': getattr(page, 'rotation', 0),
                                'char_count': len(text),
                                'word_count': len(text.split())
                            }
                            pages_metadata.append(page_metadata)
                        
                    except Exception as page_error:
                        logging.warning(f"Error extracting page {page_num} from {filename}: {page_error}")
                        continue
                
                return {
                    'success': True,
                    'filename': filename,
                    'page_count': len(pdf.pages),
                    'extracted_pages': len(pages_text),
                    'pages_text': pages_text,
                    'pages_metadata': pages_metadata,
                    'full_text': '\n\n'.join(pages_text),
                    'extraction_method': 'pdfplumber',
                    'file_hash': hashlib.md5(pdf_content).hexdigest()
                }
                
        except Exception as pdfplumber_error:
            logging.error(f"pdfplumber extraction failed for {filename}: {pdfplumber_error}")
            # Fall back to PyMuPDF if available
            if PYMUPDF_AVAILABLE:
                return self._extract_pages_with_pymupdf(pdf_content, filename)
            else:
                raise pdfplumber_error
    
    def _extract_pages_with_pymupdf(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF (fallback method)"""
        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            pages_text = []
            pages_metadata = []
            
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    
                    # Extract text with layout hints
                    text = page.get_text(sort=True)
                    if text:
                        pages_text.append(text)
                        
                        # Extract page metadata
                        page_metadata = {
                            'page_number': page_num + 1,
                            'bbox': page.rect,
                            'width': page.rect.width,
                            'height': page.rect.height,
                            'rotation': page.rotation,
                            'char_count': len(text),
                            'word_count': len(text.split())
                        }
                        pages_metadata.append(page_metadata)
                    
                except Exception as page_error:
                    logging.warning(f"Error extracting page {page_num + 1} from {filename}: {page_error}")
                    continue
            
            pdf_document.close()
            
            return {
                'success': True,
                'filename': filename,
                'page_count': pdf_document.page_count,
                'extracted_pages': len(pages_text),
                'pages_text': pages_text,
                'pages_metadata': pages_metadata,
                'full_text': '\n\n'.join(pages_text),
                'extraction_method': 'pymupdf',
                'file_hash': hashlib.md5(pdf_content).hexdigest()
            }
            
        except Exception as pymupdf_error:
            logging.error(f"PyMuPDF extraction failed for {filename}: {pymupdf_error}")
            raise pymupdf_error
    
    def _extract_with_fallback(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Fallback extraction method when no PDF libraries available"""
        logging.warning(f"Using fallback extraction for {filename} - PDF libraries not available")
        
        # Try to extract some basic info
        try:
            # Look for PDF text markers
            content_str = pdf_content.decode('latin-1', errors='ignore')
            
            # Very basic text extraction from PDF streams
            text_matches = re.findall(r'(?<=\n)\s*([A-Za-z].{10,})\s*(?=\n)', content_str)
            extracted_text = '\n'.join(text_matches[:100])  # Limit to prevent noise
            
            return {
                'success': True,
                'filename': filename,
                'page_count': 1,
                'extracted_pages': 1,
                'pages_text': [extracted_text],
                'pages_metadata': [{'page_number': 1, 'char_count': len(extracted_text)}],
                'full_text': extracted_text,
                'extraction_method': 'fallback',
                'file_hash': hashlib.md5(pdf_content).hexdigest(),
                'warning': 'PDF libraries not available - basic extraction used'
            }
            
        except Exception as fallback_error:
            logging.error(f"Fallback extraction failed for {filename}: {fallback_error}")
            return {
                'success': False,
                'error': str(fallback_error),
                'filename': filename,
                'extraction_method': 'fallback_failed'
            }
    
    def _find_recommendation_response_sections(self, text: str) -> List[Dict[str, Any]]:
        """Find and extract recommendation/response sections from document text"""
        sections = []
        
        try:
            # First, try to find section headers using government patterns
            section_starts = self._find_section_starts(text)
            
            if section_starts:
                # Extract content between section headers
                for i, (header, start_pos) in enumerate(section_starts):
                    end_pos = section_starts[i + 1][1] if i + 1 < len(section_starts) else len(text)
                    section_content = text[start_pos:end_pos].strip()
                    
                    if section_content and len(section_content) > 50:  # Minimum content length
                        section = {
                            'header': header.strip(),
                            'content': section_content,
                            'start_position': start_pos,
                            'end_position': end_pos,
                            'length': len(section_content),
                            'word_count': len(section_content.split()),
                            'section_type': self._classify_section_type(header),
                            'confidence_score': self._calculate_section_confidence(header, section_content)
                        }
                        sections.append(section)
            
            # If no clear sections found, try to extract based on content patterns
            if not sections:
                sections = self._extract_sections_by_content_patterns(text)
            
            return sections
            
        except Exception as section_error:
            logging.error(f"Error finding sections: {section_error}")
            return sections
    
    def _find_section_starts(self, text: str) -> List[Tuple[str, int]]:
        """Find the start positions of major sections in the document"""
        section_starts = []
        
        try:
            # Use government document patterns
            for pattern in self.government_patterns['section_headers']:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    header = match.group(0)
                    position = match.start()
                    section_starts.append((header, position))
            
            # Also look for table of contents patterns
            toc_patterns = [
                r'(?i)(?:^|\n)\s*(\d+\.?\d*)\s+([A-Z][^.\n]{10,})\s*\.{2,}\s*\d+',  # TOC with dots
                r'(?i)(?:^|\n)\s*(\d+\.?\d*)\s+([A-Z][^.\n]{10,})\s*\d+',  # TOC without dots
                r'(?i)(?:^|\n)\s*((?:Chapter|Section)\s+\d+[:\s]+[A-Z][^.\n]{5,})'  # Chapter/Section headers
            ]
            
            for pattern in toc_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        header = f"{match.group(1)} {match.group(2)}"
                    else:
                        header = match.group(1)
                    position = match.start()
                    section_starts.append((header, position))
            
            # Sort by position and remove duplicates
            section_starts = sorted(list(set(section_starts)), key=lambda x: x[1])
            
            if self.debug_mode:
                logging.info(f"Found {len(section_starts)} section headers")
            
            return section_starts
            
        except Exception as start_error:
            logging.error(f"Error finding section starts: {start_error}")
            return []
    
    def _extract_sections_by_content_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections based on content patterns when headers aren't clear"""
        sections = []
        
        try:
            # Look for recommendation blocks
            rec_pattern = r'(?i)((?:recommendations?|we\s+recommend|it\s+is\s+recommended).*?)(?=(?:recommendations?|conclusions?|next\s+steps|$))'
            rec_matches = re.finditer(rec_pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in rec_matches:
                content = match.group(1).strip()
                if len(content) > 100:  # Minimum content length
                    section = {
                        'header': 'Recommendations (Content-based)',
                        'content': content,
                        'start_position': match.start(),
                        'end_position': match.end(),
                        'length': len(content),
                        'word_count': len(content.split()),
                        'section_type': 'recommendations',
                        'confidence_score': 0.7,  # Lower confidence for content-based extraction
                        'extraction_method': 'content_pattern'
                    }
                    sections.append(section)
            
            # Look for response blocks
            resp_pattern = r'(?i)((?:government\s+)?response.*?)(?=(?:recommendations?|conclusions?|next\s+steps|$))'
            resp_matches = re.finditer(resp_pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in resp_matches:
                content = match.group(1).strip()
                if len(content) > 100:
                    section = {
                        'header': 'Government Response (Content-based)',
                        'content': content,
                        'start_position': match.start(),
                        'end_position': match.end(),
                        'length': len(content),
                        'word_count': len(content.split()),
                        'section_type': 'response',
                        'confidence_score': 0.7,
                        'extraction_method': 'content_pattern'
                    }
                    sections.append(section)
            
            return sections
            
        except Exception as pattern_error:
            logging.error(f"Error extracting sections by content patterns: {pattern_error}")
            return []
    
    def _classify_section_type(self, header: str) -> str:
        """Classify the type of section based on header text"""
        header_lower = header.lower()
        
        if any(word in header_lower for word in ['recommendation', 'recommend']):
            return 'recommendations'
        elif any(word in header_lower for word in ['response', 'government', 'cabinet']):
            return 'response'
        elif any(word in header_lower for word in ['finding', 'conclusion']):
            return 'findings'
        elif any(word in header_lower for word in ['executive', 'summary']):
            return 'summary'
        elif any(word in header_lower for word in ['implementation', 'action']):
            return 'implementation'
        else:
            return 'other'
    
    def _calculate_section_confidence(self, header: str, content: str) -> float:
        """Calculate confidence score for section extraction"""
        confidence = 0.5  # Base confidence
        
        # Header quality indicators
        header_lower = header.lower()
        if any(word in header_lower for word in ['recommendation', 'response', 'government']):
            confidence += 0.2
        
        # Content quality indicators
        content_lower = content.lower()
        if any(word in content_lower for word in ['should', 'must', 'recommend', 'ensure']):
            confidence += 0.1
        
        # Length indicators
        if len(content) > 500:
            confidence += 0.1
        if len(content) > 1000:
            confidence += 0.1
        
        # Structure indicators
        if re.search(r'\d+\.', content):  # Numbered points
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _analyze_government_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of a government document"""
        analysis = {
            'document_type': 'unknown',
            'has_table_of_contents': False,
            'has_numbered_sections': False,
            'has_recommendations': False,
            'has_responses': False,
            'estimated_page_count': 1,
            'content_sections': [],
            'inquiry_type': None
        }
        
        try:
            text_lower = text.lower()
            
            # Detect document type
            if re.search(r'(?i)government\s+response', text):
                analysis['document_type'] = 'government_response'
            elif re.search(r'(?i)inquiry\s+report', text):
                analysis['document_type'] = 'inquiry_report'
            elif re.search(r'(?i)(?:interim|progress)\s+report', text):
                analysis['document_type'] = 'interim_report'
            
            # Detect inquiry type (universal patterns)
            for pattern in self.government_patterns['inquiry_types']:
                match = re.search(pattern, text)
                if match:
                    inquiry_name = match.group(1).strip()
                    # Clean up common prefixes
                    inquiry_name = re.sub(r'(?i)^(?:the\s+|public\s+|independent\s+)', '', inquiry_name)
                    analysis['inquiry_type'] = inquiry_name
                    break
            
            # Check for table of contents
            toc_indicators = [
                r'(?i)(?:table\s+of\s+)?contents',
                r'(?i)\d+\.\d+\s+[A-Z]',
                r'(?i)list\s+of\s+(?:chapters|sections)'
            ]
            
            for pattern in toc_indicators:
                if re.search(pattern, text):
                    analysis['has_table_of_contents'] = True
                    break
            
            # Check for numbered sections
            numbered_sections = len(re.findall(r'(?:^|\n)\s*\d+\.\d+\s+[A-Z]', text))
            if numbered_sections >= 3:
                analysis['has_numbered_sections'] = True
            
            # Check for recommendations
            rec_count = len(re.findall(r'(?i)\brecommendation\b', text))
            if rec_count >= 3:
                analysis['has_recommendations'] = True
            
            # Check for responses
            resp_count = len(re.findall(r'(?i)\bresponse\b', text))
            if resp_count >= 3:
                analysis['has_responses'] = True
            
            # Estimate page count (rough approximation)
            char_count = len(text)
            analysis['estimated_page_count'] = max(1, char_count // 2000)  # ~2000 chars per page
            
            return analysis
            
        except Exception as analysis_error:
            logging.error(f"Error analyzing document structure: {analysis_error}")
            return analysis
    
    def extract_individual_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual numbered recommendations from text"""
        recommendations = []
        
        try:
            # Try different recommendation patterns
            for pattern in self.government_patterns['recommendation_patterns']:
                matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        number = match.group(1)
                        content = match.group(2).strip()
                    else:
                        number = str(len(recommendations) + 1)
                        content = match.group(1).strip()
                    
                    if content and len(content) > 20:  # Minimum content length
                        recommendation = {
                            'number': number,
                            'text': content,
                            'start_position': match.start(),
                            'end_position': match.end(),
                            'length': len(content),
                            'word_count': len(content.split()),
                            'confidence_score': self._calculate_item_confidence(content, 'recommendation')
                        }
                        recommendations.append(recommendation)
            
            # Remove duplicates and sort by number
            recommendations = self._deduplicate_items(recommendations, 'number')
            recommendations.sort(key=lambda x: int(x['number']) if x['number'].isdigit() else 999)
            
            if self.debug_mode:
                logging.info(f"Extracted {len(recommendations)} individual recommendations")
            
            return recommendations
            
        except Exception as rec_error:
            logging.error(f"Error extracting individual recommendations: {rec_error}")
            return []
    
    def extract_individual_responses(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual government responses from text"""
        responses = []
        
        try:
            # Try different response patterns
            for pattern in self.government_patterns['response_patterns']:
                matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
                
                for match in matches:
                    if len(match.groups()) >= 2 and match.group(1):
                        number = match.group(1)
                        content = match.group(2).strip()
                    else:
                        number = str(len(responses) + 1)
                        content = match.group(1).strip() if match.group(1) else match.group(0).strip()
                    
                    if content and len(content) > 20:
                        response = {
                            'number': number,
                            'text': content,
                            'start_position': match.start(),
                            'end_position': match.end(),
                            'length': len(content),
                            'word_count': len(content.split()),
                            'response_type': self._classify_response_type(content),
                            'confidence_score': self._calculate_item_confidence(content, 'response')
                        }
                        responses.append(response)
            
            # Remove duplicates and sort by number
            responses = self._deduplicate_items(responses, 'number')
            responses.sort(key=lambda x: int(x['number']) if x['number'].isdigit() else 999)
            
            if self.debug_mode:
                logging.info(f"Extracted {len(responses)} individual responses")
            
            return responses
            
        except Exception as resp_error:
            logging.error(f"Error extracting individual responses: {resp_error}")
            return []
    
    def _classify_response_type(self, response_text: str) -> str:
        """Classify the type of government response"""
        text_lower = response_text.lower()
        
        if any(word in text_lower for word in ['accept', 'agree', 'implement', 'will']):
            return 'accepted'
        elif any(word in text_lower for word in ['reject', 'disagree', 'decline', 'cannot']):
            return 'rejected'
        elif any(word in text_lower for word in ['partially', 'in part', 'some aspects']):
            return 'partially_accepted'
        elif any(word in text_lower for word in ['consider', 'review', 'examine']):
            return 'under_consideration'
        else:
            return 'unclear'
    
    def _calculate_item_confidence(self, content: str, item_type: str) -> float:
        """Calculate confidence score for individual items"""
        confidence = 0.5  # Base confidence
        
        # Length bonus
        if len(content) > 100:
            confidence += 0.1
        if len(content) > 300:
            confidence += 0.1
        
        # Content quality for recommendations
        if item_type == 'recommendation':
            if any(word in content.lower() for word in ['should', 'must', 'recommend', 'ensure']):
                confidence += 0.2
        
        # Content quality for responses
        elif item_type == 'response':
            if any(word in content.lower() for word in ['accept', 'reject', 'implement', 'government']):
                confidence += 0.2
        
        # Structure indicators
        if re.search(r'[.!?]', content):  # Has proper punctuation
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_items(self, items: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
        """Remove duplicate items based on key field"""
        seen_keys = set()
        deduplicated = []
        
        for item in items:
            key = item.get(key_field, '')
            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated.append(item)
        
        return deduplicated
    
    def validate_sections(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate extracted sections and return quality metrics"""
        validation = {
            'is_valid': True,
            'section_count': len(sections),
            'total_content_length': 0,
            'average_confidence': 0.0,
            'issues': [],
            'warnings': []
        }
        
        try:
            if not sections:
                validation['is_valid'] = False
                validation['issues'].append("No sections extracted")
                return validation
            
            # Calculate metrics
            total_length = sum(section.get('length', 0) for section in sections)
            total_confidence = sum(section.get('confidence_score', 0) for section in sections)
            
            validation['total_content_length'] = total_length
            validation['average_confidence'] = total_confidence / len(sections) if sections else 0
            
            # Check for issues
            for i, section in enumerate(sections):
                if section.get('length', 0) < 50:
                    validation['warnings'].append(f"Section {i+1} is very short ({section.get('length', 0)} chars)")
                
                if section.get('confidence_score', 0) < 0.5:
                    validation['warnings'].append(f"Section {i+1} has low confidence ({section.get('confidence_score', 0):.2f})")
            
            # Overall quality check
            if validation['average_confidence'] < 0.6:
                validation['warnings'].append(f"Overall extraction confidence is low ({validation['average_confidence']:.2f})")
            
            return validation
            
        except Exception as validation_error:
            logging.error(f"Error validating sections: {validation_error}")
            validation['is_valid'] = False
            validation['issues'].append(f"Validation error: {validation_error}")
            return validation
    
    def compare_inquiry_and_response_documents(self, inquiry_text: str, response_text: str) -> Dict[str, Any]:
        """Compare inquiry and response documents to find matches"""
        comparison = {
            'inquiry_recommendations': [],
            'response_items': [],
            'matched_pairs': [],
            'unmatched_recommendations': [],
            'unmatched_responses': [],
            'match_confidence': 0.0
        }
        
        try:
            # Extract recommendations from inquiry
            inquiry_recs = self.extract_individual_recommendations(inquiry_text)
            response_items = self.extract_individual_responses(response_text)
            
            comparison['inquiry_recommendations'] = inquiry_recs
            comparison['response_items'] = response_items
            
            # Match recommendations to responses by number
            matched_pairs = []
            unmatched_recs = []
            unmatched_resps = list(response_items)  # Start with all responses
            
            for rec in inquiry_recs:
                rec_number = rec.get('number', '')
                matched = False
                
                # Look for corresponding response
                for resp in response_items:
                    resp_number = resp.get('number', '')
                    if rec_number == resp_number:
                        matched_pairs.append({
                            'recommendation': rec,
                            'response': resp,
                            'match_type': 'exact_number',
                            'confidence': 0.9
                        })
                        if resp in unmatched_resps:
                            unmatched_resps.remove(resp)
                        matched = True
                        break
                
                if not matched:
                    unmatched_recs.append(rec)
            
            comparison['matched_pairs'] = matched_pairs
            comparison['unmatched_recommendations'] = unmatched_recs
            comparison['unmatched_responses'] = unmatched_resps
            
            # Calculate overall match confidence
            if inquiry_recs:
                match_rate = len(matched_pairs) / len(inquiry_recs)
                comparison['match_confidence'] = match_rate
            
            if self.debug_mode:
                logging.info(f"Matched {len(matched_pairs)}/{len(inquiry_recs)} recommendations to responses")
            
            return comparison
            
        except Exception as comparison_error:
            logging.error(f"Error comparing documents: {comparison_error}")
            return comparison
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get current extraction statistics"""
        return {
            **self.extraction_stats,
            'success_rate': (self.extraction_stats['documents_processed'] - self.extraction_stats['errors_encountered']) / max(1, self.extraction_stats['documents_processed']),
            'average_sections_per_document': self.extraction_stats['sections_extracted'] / max(1, self.extraction_stats['documents_processed']),
            'average_recommendations_per_document': self.extraction_stats['recommendations_found'] / max(1, self.extraction_stats['documents_processed'])
        }

# ===============================================
# STANDALONE UTILITY FUNCTIONS
# ===============================================

def extract_sections_from_pdf(pdf_content: bytes, filename: str = "document.pdf", debug: bool = False) -> Dict[str, Any]:
    """
    Standalone function to extract sections from PDF content
    
    Args:
        pdf_content: PDF file content as bytes
        filename: Original filename for reference
        debug: Enable debug logging
        
    Returns:
        Dictionary containing extracted sections and metadata
    """
    extractor = EnhancedSectionExtractor(debug_mode=debug)
    return extractor.extract_sections_from_pdf(pdf_content, filename)

def get_section_summary(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get a summary of extracted sections"""
    if not sections:
        return {'total_sections': 0, 'total_length': 0, 'section_types': {}}
    
    summary = {
        'total_sections': len(sections),
        'total_length': sum(section.get('length', 0) for section in sections),
        'total_words': sum(section.get('word_count', 0) for section in sections),
        'average_confidence': sum(section.get('confidence_score', 0) for section in sections) / len(sections),
        'section_types': {}
    }
    
    # Count section types
    type_counts = {}
    for section in sections:
        section_type = section.get('section_type', 'unknown')
        type_counts[section_type] = type_counts.get(section_type, 0) + 1
    
    summary['section_types'] = type_counts
    return summary

def validate_section_extraction(sections: List[Dict[str, Any]]) -> bool:
    """Quick validation of section extraction results"""
    if not sections:
        return False
    
    # Check if we have meaningful content
    total_length = sum(section.get('length', 0) for section in sections)
    if total_length < 100:
        return False
    
    # Check average confidence
    avg_confidence = sum(section.get('confidence_score', 0) for section in sections) / len(sections)
    if avg_confidence < 0.3:
        return False
    
    return True

def extract_government_recommendations(text: str) -> List[Dict[str, Any]]:
    """Extract individual recommendations from government document text"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_individual_recommendations(text)

def extract_government_responses(text: str) -> List[Dict[str, Any]]:
    """Extract individual responses from government document text"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_individual_responses(text)

# ===============================================
# TESTING AND DEBUGGING FUNCTIONS
# ===============================================

def test_extraction(pdf_path: str = None, pdf_content: bytes = None) -> Dict[str, Any]:
    """Test the extraction functionality"""
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        filename = Path(pdf_path).name
    elif pdf_content:
        filename = "test_document.pdf"
    else:
        return {'error': 'No PDF provided for testing'}
    
    try:
        extractor = EnhancedSectionExtractor(debug_mode=True)
        result = extractor.extract_sections_from_pdf(pdf_content, filename)
        
        # Add test summary
        test_summary = {
            'test_passed': result.get('success', False),
            'sections_found': len(result.get('sections', [])),
            'recommendations_found': len(result.get('individual_recommendations', [])),
            'responses_found': len(result.get('individual_responses', [])),
            'extraction_method': result.get('extraction_method', 'unknown'),
            'document_analysis': result.get('document_analysis', {}),
            'statistics': extractor.get_extraction_statistics()
        }
        
        result['test_summary'] = test_summary
        return result
        
    except Exception as test_error:
        return {
            'error': str(test_error),
            'test_passed': False
        }

def debug_section_detection(text: str, max_length: int = 2000) -> Dict[str, Any]:
    """Debug section detection by showing step-by-step process"""
    debug_info = {
        'input_length': len(text),
        'input_preview': text[:max_length] + ('...' if len(text) > max_length else ''),
        'pattern_matches': {},
        'section_candidates': [],
        'final_sections': []
    }
    
    try:
        extractor = EnhancedSectionExtractor(debug_mode=True)
        
        # Test each pattern
        for pattern_name, patterns in extractor.government_patterns.items():
            debug_info['pattern_matches'][pattern_name] = []
            
            if isinstance(patterns, list):
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        debug_info['pattern_matches'][pattern_name].append({
                            'match_text': match.group(0),
                            'position': match.start(),
                            'groups': match.groups()
                        })
        
        # Find section starts
        section_starts = extractor._find_section_starts(text)
        debug_info['section_candidates'] = [{'header': header, 'position': pos} for header, pos in section_starts]
        
        # Extract final sections
        sections = extractor._find_recommendation_response_sections(text)
        debug_info['final_sections'] = [
            {
                'header': section.get('header', ''),
                'type': section.get('section_type', ''),
                'length': section.get('length', 0),
                'confidence': section.get('confidence_score', 0)
            }
            for section in sections
        ]
        
        return debug_info
        
    except Exception as debug_error:
        debug_info['error'] = str(debug_error)
        return debug_info

def test_individual_extraction(text: str) -> Dict[str, Any]:
    """Test individual recommendation and response extraction"""
    try:
        extractor = EnhancedSectionExtractor(debug_mode=True)
        
        recommendations = extractor.extract_individual_recommendations(text)
        responses = extractor.extract_individual_responses(text)
        
        return {
            'recommendations': recommendations,
            'responses': responses,
            'recommendation_count': len(recommendations),
            'response_count': len(responses),
            'statistics': extractor.get_extraction_statistics()
        }
        
    except Exception as test_error:
        return {
            'error': str(test_error),
            'recommendations': [],
            'responses': []
        }

# ===============================================
# MODULE EXPORTS
# ===============================================

__all__ = [
    # Main class
    'EnhancedSectionExtractor',
    
    # Standalone functions
    'extract_sections_from_pdf',
    'get_section_summary',
    'validate_section_extraction',
    'extract_government_recommendations',
    'extract_government_responses',
    
    # Testing and debugging
    'test_extraction',
    'debug_section_detection', 
    'test_individual_extraction',
    
    # Availability flags
    'PDFPLUMBER_AVAILABLE',
    'PYMUPDF_AVAILABLE'
]

# ===============================================
# MODULE INITIALIZATION
# ===============================================

# Log module initialization
if PDFPLUMBER_AVAILABLE and PYMUPDF_AVAILABLE:
    logging.info("✅ Enhanced Section Extractor loaded with full PDF support (pdfplumber + PyMuPDF)")
elif PDFPLUMBER_AVAILABLE:
    logging.info("✅ Enhanced Section Extractor loaded with pdfplumber support")
elif PYMUPDF_AVAILABLE:
    logging.info("✅ Enhanced Section Extractor loaded with PyMuPDF support")
else:
    logging.warning("⚠️ Enhanced Section Extractor loaded with fallback mode (no PDF libraries)")

logging.info("🎉 Enhanced Section Extractor module is COMPLETE and ready for universal inquiry document processing!")
