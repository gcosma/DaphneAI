# ===============================================
# FILE: modules/enhanced_section_extractor.py (COMPLETE REVISED VERSION)
# ===============================================

import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import io

# PDF processing libraries with fallback handling
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logging.info("✅ pdfplumber imported successfully")
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("⚠️ pdfplumber not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logging.info("✅ PyMuPDF imported successfully")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("⚠️ PyMuPDF not available")

class EnhancedSectionExtractor:
    """
    Enhanced section extractor with multi-strategy recommendation extraction
    Supports all UK government inquiry formats including Infected Blood Inquiry
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize with enhanced patterns for all inquiry types"""
        self.debug_mode = debug_mode
        self.extraction_stats = {
            'documents_processed': 0,
            'sections_extracted': 0,
            'recommendations_found': 0,
            'responses_found': 0,
            'last_extraction_time': None,
            'total_processing_time': 0
        }
        
        # Standard government document patterns
        self.government_patterns = {
            'section_headers': [
                # Standard government section headers
                r'(?i)^(?:\d+\.?\d*\s+)?(?:key\s+)?recommendations?\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:government\s+)?responses?\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:official\s+)?response\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:response\s+to\s+(?:the\s+)?(?:inquiry\s+)?recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:departmental\s+response)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:ministry\s+response)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:cabinet\s+office\s+response)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:implementation\s+(?:plan|strategy))\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:action\s+plan)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:government\s+action)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:next\s+steps?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:progress\s+(?:report|update))\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:response\s+to\s+recommendation\s+\d+)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:recommendation\s+\d+\s*[-:]?\s*response)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:responses?\s+to\s+individual\s+recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:detailed\s+responses?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:government\s+responses?\s+to\s+recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s+)?(?:recommendation\s+\d+)\s*$',
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
        
        # NEW: Infected Blood Inquiry specific patterns
        self.infected_blood_patterns = {
            # "I recommend:" block extraction
            'i_recommend_block': r'(?i)I\s+recommend:\s*\n(.*?)(?=\n\s*[A-Z][a-z]+\s+\d+|$)',
            
            # Section 1.5 extraction
            'section_1_5': r'(?i)1\.5\s+recommendations?\s*\n(.*?)(?=\n\s*(?:\d+\.\d+|appendix|references|$))',
            
            # Numbered recommendations: "8. Finding the undiagnosed"
            'numbered_main': r'(?i)(?:^|\n)\s*(\d+)\.\s+([^\n]+)(?:\n((?:(?!\d+\.).)*?))?',
            
            # Sub-recommendations: "(a) When doctors become aware..."
            'sub_lettered': r'(?i)\(([a-z])\)\s+([^\n]+(?:\n(?!\([a-z]\)|(?:\n\s*\d+\.)).)*)',
            
            # Sub-sub-recommendations: "(i) clinical audit should..."
            'sub_roman': r'(?i)\(([ivx]+)\)\s+([^\n]+(?:\n(?!\([ivx]+\)|(?:\n\s*\d+\.)).)*)',
        }
        
        # Section end markers
        self.section_end_markers = [
            r'(?i)^(?:conclusion|summary|appendix|annex|bibliography|references|index)\s*$',
            r'(?i)^(?:part|chapter|section)\s+[ivxlcdm]+\s*$',
            r'(?i)^(?:part|chapter|section)\s+\d+\s*$',
            r'(?i)^(?:next steps?|way forward|future work)\s*$',
            r'(?i)^(?:contact details?|further information)\s*$',
            r'(?i)^(?:glossary|abbreviations|acronyms)\s*$',
            r'(?i)^(?:published by|crown copyright|©)\s*',
            r'(?i)^(?:signed|dated this|chair of (?:the )?inquiry)\s*',
            r'(?i)^(?:minister for|secretary of state)\s*',
        ]

    def _preprocess_text(self, text: str) -> str:
        """Fix PDF extraction issues that break pattern matching"""
        if not text:
            return ""
        
        # Fix common PDF extraction issues
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix word boundaries
        text = re.sub(r'(\d+)\.([A-Z])', r'\1. \2', text)  # Fix numbered lists
        text = re.sub(r'([.!?])([A-Z])', r'\1\n\2', text)  # Add line breaks after sentences
        
        # Fix recommendation formatting
        text = re.sub(r'Recommendation(\d+)', r'Recommendation \1', text)
        text = re.sub(r'I recommend(\d+)', r'I recommend:\n\1', text)
        
        return text.strip()

    def _is_valid_recommendation(self, text: str) -> bool:
        """Updated validation rules - more permissive for Infected Blood Inquiry"""
        if not text or len(text.strip()) < 10:  # LOWERED from 50
            return False
        
        if len(text) > 5000:  # INCREASED limit
            return False
        
        text_lower = text.lower()
        
        # Check for exclude phrases
        exclude_phrases = ['table of contents', 'page ', 'copyright', 'printed', 'isbn']
        if any(phrase in text_lower for phrase in exclude_phrases):
            return False
        
        # EXPANDED indicator words for Infected Blood Inquiry
        indicators = [
            # Action words
            'should', 'must', 'ought', 'recommend', 'establish', 'implement', 
            'ensure', 'develop', 'create', 'review', 'consider', 'examine',
            
            # Inquiry-specific terms
            'patient', 'clinical', 'blood', 'transfusion', 'hepatitis', 
            'haemophilia', 'care', 'voice', 'enabled', 'empowered',
            'audit', 'satisfaction', 'concern', 'reported', 'board',
            'safety', 'training', 'monitoring', 'framework', 'funding'
        ]
        
        return any(indicator in text_lower for indicator in indicators)

    def extract_all_recommendations(self, text: str, document_source: str = "") -> List[Dict[str, Any]]:
        """Extract using multiple strategies for better coverage"""
        all_recommendations = []
        
        # Preprocess text first
        text = self._preprocess_text(text)
        
        # Strategy 1: Extract from "I recommend:" blocks
        strategy1_recs = self._extract_i_recommend_blocks(text, document_source)
        all_recommendations.extend(strategy1_recs)
        
        # Strategy 2: Extract from "1.5 Recommendations" section
        strategy2_recs = self._extract_section_1_5(text, document_source)
        all_recommendations.extend(strategy2_recs)
        
        # Strategy 3: Extract numbered recommendations
        strategy3_recs = self._extract_numbered_recommendations(text, document_source)
        all_recommendations.extend(strategy3_recs)
        
        # Strategy 4: Extract sub-recommendations
        strategy4_recs = self._extract_sub_recommendations(text, document_source)
        all_recommendations.extend(strategy4_recs)
        
        # Strategy 5: Final scan for missed items
        strategy5_recs = self._scan_for_missed_recommendations(text, document_source, all_recommendations)
        all_recommendations.extend(strategy5_recs)
        
        # Remove duplicates and validate
        final_recommendations = self._deduplicate_and_validate(all_recommendations)
        
        if self.debug_mode:
            logging.info(f"✅ Multi-strategy extraction found {len(final_recommendations)} unique recommendations")
        
        return final_recommendations

    def _extract_i_recommend_blocks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract from 'I recommend:' blocks"""
        recommendations = []
        
        pattern = self.infected_blood_patterns['i_recommend_block']
        matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            block_content = match.group(1).strip()
            if len(block_content) > 50:
                # Extract numbered items from this block
                block_recs = self._extract_numbered_from_block(block_content, source)
                recommendations.extend(block_recs)
        
        return recommendations

    def _extract_section_1_5(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract from section 1.5 Recommendations"""
        recommendations = []
        
        pattern = self.infected_blood_patterns['section_1_5']
        matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            section_content = match.group(1).strip()
            if len(section_content) > 100:
                section_recs = self._extract_numbered_from_block(section_content, source)
                recommendations.extend(section_recs)
        
        return recommendations

    def _extract_numbered_from_block(self, block_content: str, source: str) -> List[Dict[str, Any]]:
        """Extract numbered items from a text block"""
        recommendations = []
        
        # Pattern for numbered items: "8. Finding the undiagnosed"
        pattern = r'(?:^|\n)\s*(\d+)\.\s+([^\n]+(?:\n(?!\d+\.).)*?)'
        matches = re.finditer(pattern, block_content, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            rec_id = match.group(1)
            rec_text = match.group(2).strip()
            
            # Clean up the text
            rec_text = re.sub(r'\s+', ' ', rec_text)
            rec_text = rec_text.replace('\n', ' ')
            
            if self._is_valid_recommendation(rec_text):
                recommendations.append({
                    'id': rec_id,
                    'text': rec_text,
                    'type': 'numbered_recommendation',
                    'source': 'block_extraction',
                    'document_source': source,
                    'confidence': 0.95
                })
        
        return recommendations

    def _extract_numbered_recommendations(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract numbered recommendations from full text"""
        recommendations = []
        
        pattern = self.infected_blood_patterns['numbered_main']
        matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            rec_id = match.group(1)
            rec_title = match.group(2).strip()
            rec_content = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else ""
            
            # Combine title and content
            full_text = f"{rec_title}. {rec_content}".strip()
            
            if self._is_valid_recommendation(full_text):
                recommendations.append({
                    'id': rec_id,
                    'text': full_text,
                    'type': 'numbered_recommendation',
                    'source': 'full_text_extraction',
                    'document_source': source,
                    'confidence': 0.9
                })
        
        return recommendations

    def _extract_sub_recommendations(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract sub-recommendations (lettered and roman numeral)"""
        recommendations = []
        
        # Extract lettered sub-recommendations
        pattern = self.infected_blood_patterns['sub_lettered']
        matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            sub_id = match.group(1)
            sub_text = match.group(2).strip()
            
            if self._is_valid_recommendation(sub_text):
                recommendations.append({
                    'id': f"sub_{sub_id}",
                    'text': sub_text,
                    'type': 'sub_recommendation',
                    'source': 'lettered_extraction',
                    'document_source': source,
                    'confidence': 0.85
                })
        
        # Extract roman numeral sub-recommendations
        pattern = self.infected_blood_patterns['sub_roman']
        matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            sub_id = match.group(1)
            sub_text = match.group(2).strip()
            
            if self._is_valid_recommendation(sub_text):
                recommendations.append({
                    'id': f"sub_{sub_id}",
                    'text': sub_text,
                    'type': 'sub_sub_recommendation',
                    'source': 'roman_extraction',
                    'document_source': source,
                    'confidence': 0.8
                })
        
        return recommendations

    def _scan_for_missed_recommendations(self, text: str, source: str, existing_recs: List[Dict]) -> List[Dict[str, Any]]:
        """Final scan for any missed recommendations"""
        recommendations = []
        existing_texts = set(rec.get('text', '')[:50] for rec in existing_recs)
        
        # Look for any "recommend" statements that might have been missed
        pattern = r'(?i)(?:i\s+)?recommend[s]?\s+(?:that\s+)?([^.]+\.)'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            rec_text = match.group(1).strip()
            rec_preview = rec_text[:50]
            
            if rec_preview not in existing_texts and self._is_valid_recommendation(rec_text):
                recommendations.append({
                    'id': f"missed_{len(recommendations)+1}",
                    'text': rec_text,
                    'type': 'general_recommendation',
                    'source': 'final_scan',
                    'document_source': source,
                    'confidence': 0.7
                })
                existing_texts.add(rec_preview)
        
        return recommendations

    def _deduplicate_and_validate(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and validate final recommendations"""
        seen_texts = set()
        unique_recs = []
        
        for rec in recommendations:
            # Create a normalized version for comparison
            text_key = re.sub(r'\s+', ' ', rec.get('text', '').lower().strip())[:100]
            
            if text_key not in seen_texts and len(text_key) > 10:
                seen_texts.add(text_key)
                unique_recs.append(rec)
        
        # Sort by ID if numeric, otherwise by confidence
        def sort_key(rec):
            rec_id = rec.get('id', '')
            if rec_id.isdigit():
                return (0, int(rec_id))
            elif rec_id.startswith('sub_') and rec_id[4:].isdigit():
                return (1, int(rec_id[4:]))
            else:
                return (2, rec.get('confidence', 0))
        
        unique_recs.sort(key=sort_key)
        return unique_recs

    def extract_individual_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual numbered recommendations from text - ENHANCED VERSION"""
        try:
            # Use multi-strategy extraction for better coverage
            recommendations = self.extract_all_recommendations(text, "unknown_source")
            
            # Convert to the expected format
            formatted_recommendations = []
            for rec in recommendations:
                formatted_rec = {
                    'number': rec.get('id', ''),
                    'text': rec.get('text', ''),
                    'start_position': 0,  # Not tracked in new method
                    'end_position': 0,    # Not tracked in new method
                    'length': len(rec.get('text', '')),
                    'word_count': len(rec.get('text', '').split()),
                    'confidence_score': rec.get('confidence', 0.5),
                    'extraction_method': rec.get('source', 'multi_strategy'),
                    'type': rec.get('type', 'recommendation')
                }
                formatted_recommendations.append(formatted_rec)
            
            if self.debug_mode:
                logging.info(f"✅ Extracted {len(formatted_recommendations)} recommendations using multi-strategy approach")
            
            return formatted_recommendations
            
        except Exception as rec_error:
            logging.error(f"Error in enhanced recommendation extraction: {rec_error}")
            return []

    def extract_individual_responses(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual government responses from text"""
        responses = []
        
        try:
            # Try different response patterns
            for pattern in self.government_patterns['response_patterns']:
                matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        number = match.group(1) if match.group(1) else str(len(responses) + 1)
                        content = match.group(2).strip()
                    else:
                        number = str(len(responses) + 1)
                        content = match.group(1).strip()
                    
                    if content and len(content) > 20:  # Minimum content length
                        response = {
                            'number': number,
                            'text': content,
                            'start_position': match.start(),
                            'end_position': match.end(),
                            'length': len(content),
                            'word_count': len(content.split()),
                            'confidence_score': self._calculate_item_confidence(content, 'response'),
                            'response_type': self._classify_response_type(content)
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

    def _calculate_item_confidence(self, text: str, item_type: str) -> float:
        """Calculate confidence score for extracted item"""
        confidence = 0.5
        
        # Length-based confidence
        if len(text) > 100:
            confidence += 0.2
        if len(text) > 300:
            confidence += 0.1
        
        # Content-based confidence
        if item_type == 'recommendation':
            if any(word in text.lower() for word in ['should', 'must', 'recommend']):
                confidence += 0.2
        elif item_type == 'response':
            if any(word in text.lower() for word in ['accepted', 'implemented', 'government']):
                confidence += 0.2
        
        return min(confidence, 1.0)

    def _classify_response_type(self, text: str) -> str:
        """Classify the type of government response"""
        text_lower = text.lower()
        
        if 'accepted in full' in text_lower or 'fully accepted' in text_lower:
            return 'accepted'
        elif 'accepted in principle' in text_lower or 'partially accepted' in text_lower:
            return 'partially_accepted'
        elif 'not accepted' in text_lower or 'rejected' in text_lower:
            return 'rejected'
        elif 'under consideration' in text_lower or 'being reviewed' in text_lower:
            return 'under_consideration'
        else:
            return 'unclear'

    def _deduplicate_items(self, items: List[Dict], key_field: str) -> List[Dict]:
        """Remove duplicate items based on content similarity"""
        seen_content = set()
        unique_items = []
        
        for item in items:
            # Create a normalized content hash
            content = item.get('text', '')
            normalized_content = re.sub(r'\s+', ' ', content.lower().strip())[:100]
            
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                unique_items.append(item)
        
        return unique_items

    def debug_extraction(self, text: str):
        """Debug helper for troubleshooting extraction issues"""
        print(f"✅ Text length: {len(text):,} characters")
        print(f"✅ Contains 'I recommend': {'i recommend' in text.lower()}")
        print(f"✅ Contains '1.5 Recommendations': {'1.5 recommendations' in text.lower()}")
        print(f"✅ Contains numbered items: {len(re.findall(r'\\d+\\. ', text))}")
        
        # Test each pattern
        for name, pattern in self.infected_blood_patterns.items():
            try:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                print(f"✅ Pattern {name}: {matches} matches")
            except:
                print(f"❌ Pattern {name}: ERROR")

    def extract_sections_from_pdf(self, pdf_content: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """Main method to extract sections from PDF content"""
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
                
                result['extraction_method'] = 'enhanced_multi_strategy'
                result['success'] = True
                
                if self.debug_mode:
                    logging.info(f"✅ Successfully processed {filename}: {len(recommendations)} recommendations, {len(responses)} responses")
            
            return result
            
        except Exception as extraction_error:
            logging.error(f"Error extracting sections from {filename}: {extraction_error}")
            return {
                'success': False,
                'error': str(extraction_error),
                'filename': filename,
                'extraction_method': 'failed'
            }

    def _extract_pages_with_pdfplumber(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text using pdfplumber with enhanced error handling"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                pages_text = []
                pages_metadata = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text() or ""
                        pages_text.append(page_text)
                        pages_metadata.append({
                            'page_number': page_num,
                            'char_count': len(page_text),
                            'bbox': page.bbox if hasattr(page, 'bbox') else None
                        })
                    except Exception as page_error:
                        logging.warning(f"Error extracting page {page_num}: {page_error}")
                        pages_text.append("")
                        pages_metadata.append({'page_number': page_num, 'char_count': 0})
                
                full_text = '\n\n'.join(pages_text)
                
                return {
                    'success': True,
                    'filename': filename,
                    'page_count': len(pdf.pages),
                    'extracted_pages': len(pages_text),
                    'pages_text': pages_text,
                    'pages_metadata': pages_metadata,
                    'full_text': full_text,
                    'extraction_method': 'pdfplumber',
                    'file_hash': hashlib.md5(pdf_content).hexdigest()
                }
                
        except Exception as plumber_error:
            logging.error(f"pdfplumber extraction failed for {filename}: {plumber_error}")
            # Try PyMuPDF as fallback
            if PYMUPDF_AVAILABLE:
                return self._extract_pages_with_pymupdf(pdf_content, filename)
            else:
                return self._extract_with_fallback(pdf_content, filename)

    def _extract_pages_with_pymupdf(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF as fallback"""
        try:
            pdf_document = fitz.open("pdf", pdf_content)
            pages_text = []
            pages_metadata = []
            
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    pages_text.append(page_text)
                    pages_metadata.append({
                        'page_number': page_num + 1,
                        'char_count': len(page_text),
                        'rect': page.rect
                    })
                except Exception as page_error:
                    logging.warning(f"Error extracting page {page_num + 1}: {page_error}")
                    pages_text.append("")
                    pages_metadata.append({'page_number': page_num + 1, 'char_count': 0})
            
            pdf_document.close()
            full_text = '\n\n'.join(pages_text)
            
            return {
                'success': True,
                'filename': filename,
                'page_count': pdf_document.page_count,
                'extracted_pages': len(pages_text),
                'pages_text': pages_text,
                'pages_metadata': pages_metadata,
                'full_text': full_text,
                'extraction_method': 'pymupdf',
                'file_hash': hashlib.md5(pdf_content).hexdigest()
            }
            
        except Exception as pymupdf_error:
            logging.error(f"PyMuPDF extraction failed for {filename}: {pymupdf_error}")
            return self._extract_with_fallback(pdf_content, filename)

    def _extract_with_fallback(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Basic fallback extraction when PDF libraries fail"""
        try:
            # Try to extract basic text content
            content_str = pdf_content.decode('utf-8', errors='ignore')
            text_matches = re.findall(r'[A-Za-z].*?[.!?]', content_str)
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
            
            if self.debug_mode:
                logging.info(f"Found {len(sections)} sections using pattern matching")
            
            return sections
            
        except Exception as section_error:
            logging.error(f"Error finding sections: {section_error}")
            return []

    def _find_section_starts(self, text: str) -> List[Tuple[str, int]]:
        """Find section start positions"""
        section_starts = []
        
        for pattern in self.government_patterns['section_headers']:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                header = match.group(0)
                position = match.start()
                section_starts.append((header, position))
        
        # Sort by position
        section_starts.sort(key=lambda x: x[1])
        return section_starts

    def _classify_section_type(self, header: str) -> str:
        """Classify section type based on header"""
        header_lower = header.lower()
        
        if any(word in header_lower for word in ['recommendation', 'finding']):
            return 'recommendation'
        elif any(word in header_lower for word in ['response', 'government', 'implementation']):
            return 'response'
        elif any(word in header_lower for word in ['summary', 'conclusion']):
            return 'summary'
        else:
            return 'other'

    def _calculate_section_confidence(self, header: str, content: str) -> float:
        """Calculate confidence score for section"""
        confidence = 0.5
        
        # Header-based confidence
        if any(word in header.lower() for word in ['recommendation', 'response']):
            confidence += 0.3
        
        # Content-based confidence
        if len(content) > 500:
            confidence += 0.1
        if len(content) > 1000:
            confidence += 0.1
        
        return min(confidence, 1.0)

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            'documents_processed': self.extraction_stats['documents_processed'],
            'sections_extracted': self.extraction_stats['sections_extracted'],
            'recommendations_found': self.extraction_stats['recommendations_found'],
            'responses_found': self.extraction_stats['responses_found'],
            'last_extraction_time': self.extraction_stats['last_extraction_time'],
            'total_processing_time': self.extraction_stats['total_processing_time']
        }

    def compare_inquiry_and_response_documents(self, inquiry_text: str, response_text: str) -> Dict[str, Any]:
        """Compare inquiry recommendations with government responses"""
        try:
            inquiry_recs = self.extract_individual_recommendations(inquiry_text)
            response_items = self.extract_individual_responses(response_text)
            
            matched_pairs = []
            unmatched_recommendations = inquiry_recs[:]
            unmatched_responses = response_items[:]
            
            # Try to match recommendations to responses
            for rec in inquiry_recs:
                rec_num = rec.get('number', '')
                for resp in response_items:
                    resp_num = resp.get('number', '')
                    if rec_num == resp_num:
                        matched_pairs.append({
                            'recommendation': rec,
                            'response': resp,
                            'match_confidence': 1.0
                        })
                        if rec in unmatched_recommendations:
                            unmatched_recommendations.remove(rec)
                        if resp in unmatched_responses:
                            unmatched_responses.remove(resp)
                        break
            
            return {
                'matched_pairs': matched_pairs,
                'unmatched_recommendations': unmatched_recommendations,
                'unmatched_responses': unmatched_responses,
                'match_statistics': {
                    'total_recommendations': len(inquiry_recs),
                    'total_responses': len(response_items),
                    'matched_pairs': len(matched_pairs),
                    'match_rate': len(matched_pairs) / max(len(inquiry_recs), 1)
                }
            }
            
        except Exception as compare_error:
            logging.error(f"Error comparing documents: {compare_error}")
            return {
                'matched_pairs': [],
                'unmatched_recommendations': [],
                'unmatched_responses': [],
                'error': str(compare_error)
            }


# ===============================================
# STANDALONE FUNCTIONS
# ===============================================

def extract_sections_from_pdf(pdf_content: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
    """Standalone function for PDF section extraction"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_sections_from_pdf(pdf_content, filename)

def get_section_summary(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get summary statistics for extracted sections"""
    if not sections:
        return {'total_sections': 0}
    
    summary = {
        'total_sections': len(sections),
        'total_length': sum(section.get('length', 0) for section in sections),
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

def test_extraction(pdf_path: str = None, pdf_content: bytes = None) -> Dict[str, Any]:
    """Test the extraction functionality"""
    from pathlib import Path
    
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
