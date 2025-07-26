# ===============================================
# FILE: modules/enhanced_section_extractor.py (COMPLETE FIXED VERSION)
# ===============================================

import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import io

# NLTK for sentence tokenization with better fallback
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
    except LookupError:
        nltk.download('punkt', quiet=True)
        NLTK_AVAILABLE = True
    logging.info("✅ NLTK imported successfully")
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("⚠️ NLTK not available - using fallback sentence splitting")
    
    def sent_tokenize(text):
        """Fallback sentence tokenizer"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

# PDF processing libraries
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
    Enhanced Section Extractor with Fixed Hybrid Approach
    
    Combines original functionality with new generic sentence-based extraction.
    Maintains full backward compatibility while adding universal inquiry support.
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize with enhanced patterns for all inquiry types"""
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        # Original extraction stats (maintaining compatibility)
        self.extraction_stats = {
            'documents_processed': 0,
            'sections_extracted': 0,
            'recommendations_found': 0,
            'responses_found': 0,
            'last_extraction_time': None,
            'total_processing_time': 0
        }
        
        # FIXED: Simplified and more effective patterns for sentence-based extraction
        self.recommend_terms = [
            r'recommend',
            r'suggest',
            r'propose',
            r'advise',
            r'urge',
            r'should',
            r'must',
            r'ought\s+to',
            r'need\s+to',
            r'require',
            r'ensure',
            r'establish',
            r'implement'
        ]
        
        self.response_terms = [
            r'response',
            r'reply',
            r'answer',
            r'accept',
            r'reject',
            r'agree',
            r'disagree',
            r'implement',
            r'adopt',
            r'action',
            r'measure',
            r'step'
        ]
        
        # ORIGINAL: Maintain existing patterns for backward compatibility
        self.government_patterns = {
            'section_headers': [
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
                r'(?i)recommendation\s+(\d+)[:\.]?\s*(.+?)(?=recommendation\s+\d+|$)',
                r'(?i)(\d+)\.\s+(.+?)(?=\d+\.|$)',
                r'(?i)•\s*(.+?)(?=•|$)'
            ],
            'response_patterns': [
                r'(?i)response\s+to\s+recommendation\s+(\d+)[:\.]?\s*(.+?)(?=response\s+to|$)',
                r'(?i)(accepted|rejected|agreed)\s*[:\.]?\s*(.+?)(?=accepted|rejected|agreed|$)'
            ]
        }
        
        # ORIGINAL: Infected Blood Inquiry specific patterns (maintain compatibility)
        self.infected_blood_patterns = {
            'i_recommend_block': r'(?i)I\s+recommend:\s*\n(.*?)(?=\n\s*[A-Z][a-z]+\s+\d+|$)',
            'section_1_5': r'(?i)1\.5\s+recommendations?\s*\n(.*?)(?=\n\s*(?:\d+\.\d+|appendix|references|$))',
            'numbered_main': r'(?i)(?:^|\n)\s*(\d+)\.\s+([^\n]+)(?:\n((?:(?!\d+\.).)*?))?',
            'sub_lettered': r'(?i)\(([a-z])\)\s+([^\n]+(?:\n(?!\([a-z]\)|(?:\n\s*\d+\.)).)*)',
            'sub_roman': r'(?i)\(([ivx]+)\)\s+([^\n]+(?:\n(?!\([ivx]+\)|(?:\n\s*\d+\.)).)*)',
        }

    # ============================================================================
    # MAIN EXTRACTION METHODS - FIXED
    # ============================================================================

    def extract_individual_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """FIXED: Enhanced method - combines original + new hybrid approach"""
        if self.debug_mode:
            self.logger.info(f"Starting recommendation extraction on text length: {len(text)}")
        
        try:
            # Get results from both methods
            original_recs = self._extract_individual_recommendations_original(text)
            hybrid_recs = self._extract_recommendations_hybrid_fixed(text)
            
            # Combine and deduplicate
            combined_recs = self._combine_extraction_results_fixed(original_recs, hybrid_recs)
            
            self.extraction_stats['recommendations_found'] = len(combined_recs)
            
            if self.debug_mode:
                self.logger.info(f"Found {len(combined_recs)} recommendations ({len(original_recs)} original + {len(hybrid_recs)} hybrid)")
                for i, rec in enumerate(combined_recs):
                    self.logger.info(f"  {i+1}. {rec['text'][:100]}... (confidence: {rec.get('confidence', 0):.2f})")
            
            return combined_recs
            
        except Exception as e:
            self.logger.error(f"Error in extract_individual_recommendations: {e}")
            if self.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return []

    def extract_individual_responses(self, text: str) -> List[Dict[str, Any]]:
        """FIXED: Enhanced method - combines original + new hybrid approach"""
        if self.debug_mode:
            self.logger.info(f"Starting response extraction on text length: {len(text)}")
        
        try:
            # Get results from both methods
            original_resps = self._extract_individual_responses_original(text)
            hybrid_resps = self._extract_responses_hybrid_fixed(text)
            
            # Combine and deduplicate
            combined_resps = self._combine_extraction_results_fixed(original_resps, hybrid_resps)
            
            self.extraction_stats['responses_found'] = len(combined_resps)
            
            if self.debug_mode:
                self.logger.info(f"Found {len(combined_resps)} responses ({len(original_resps)} original + {len(hybrid_resps)} hybrid)")
            
            return combined_resps
            
        except Exception as e:
            self.logger.error(f"Error in extract_individual_responses: {e}")
            if self.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return []

    # ============================================================================
    # FIXED HYBRID EXTRACTION METHODS
    # ============================================================================

    def _extract_recommendations_hybrid_fixed(self, text: str) -> List[Dict[str, Any]]:
        """FIXED: Sentence-based recommendation extraction"""
        recommendations = []
        
        try:
            sentences = sent_tokenize(text)
            
            if self.debug_mode:
                self.logger.info(f"Split text into {len(sentences)} sentences for recommendation analysis")
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                
                # Check if sentence contains recommendation terms
                contains_rec_term = False
                matched_terms = []
                
                for term in self.recommend_terms:
                    if re.search(term, sentence, re.IGNORECASE):
                        contains_rec_term = True
                        matched_terms.append(term)
                
                if contains_rec_term:
                    # Calculate confidence
                    confidence = self._calculate_recommendation_confidence_fixed(sentence, matched_terms)
                    
                    if confidence > 0.5:  # Only include items with reasonable confidence
                        rec_type = self._classify_recommendation_type_fixed(sentence)
                        
                        recommendation = {
                            'id': f"hybrid_rec_{i+1}",
                            'text': sentence.strip(),
                            'type': rec_type,
                            'confidence': confidence,
                            'extraction_method': 'sentence_based',
                            'position': i,
                            'matched_terms': matched_terms,
                            'references': self._extract_references_fixed(sentence)
                        }
                        
                        recommendations.append(recommendation)
                        
                        if self.debug_mode:
                            self.logger.info(f"Hybrid found recommendation: {sentence[:50]}... (confidence: {confidence:.2f})")
        
        except Exception as e:
            self.logger.error(f"Error in hybrid recommendation extraction: {e}")
        
        return recommendations

    def _extract_responses_hybrid_fixed(self, text: str) -> List[Dict[str, Any]]:
        """FIXED: Sentence-based response extraction"""
        responses = []
        
        try:
            sentences = sent_tokenize(text)
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:
                    continue
                
                # Check if sentence contains response terms
                contains_resp_term = False
                matched_terms = []
                
                for term in self.response_terms:
                    if re.search(term, sentence, re.IGNORECASE):
                        contains_resp_term = True
                        matched_terms.append(term)
                
                if contains_resp_term:
                    # Calculate confidence
                    confidence = self._calculate_response_confidence_fixed(sentence, matched_terms)
                    
                    if confidence > 0.5:
                        resp_type = self._classify_response_type_fixed(sentence)
                        
                        response = {
                            'id': f"hybrid_resp_{i+1}",
                            'text': sentence.strip(),
                            'type': resp_type,
                            'confidence': confidence,
                            'extraction_method': 'sentence_based',
                            'position': i,
                            'matched_terms': matched_terms,
                            'references': self._extract_references_fixed(sentence)
                        }
                        
                        responses.append(response)
                        
                        if self.debug_mode:
                            self.logger.info(f"Hybrid found response: {sentence[:50]}... (confidence: {confidence:.2f})")
        
        except Exception as e:
            self.logger.error(f"Error in hybrid response extraction: {e}")
        
        return responses

    def _calculate_recommendation_confidence_fixed(self, text: str, matched_terms: List[str]) -> float:
        """FIXED: Calculate confidence score for recommendations"""
        if not text or not matched_terms:
            return 0.0
        
        confidence = 0.6  # Base confidence for having matched terms
        text_lower = text.lower()
        
        try:
            # Boost for strong recommendation indicators
            if 'i recommend' in text_lower:
                confidence += 0.3
            if re.search(r'recommendation\s+\d+', text_lower):
                confidence += 0.25
            if any(word in text_lower for word in ['should', 'must', 'ought']):
                confidence += 0.2
            if any(word in text_lower for word in ['ensure', 'establish', 'implement']):
                confidence += 0.15
            
            # Boost for multiple matched terms
            confidence += min(len(matched_terms) * 0.05, 0.15)
            
            # Reduce for weak contexts
            if any(weak in text_lower for weak in ['recommend reading', 'recommend visiting']):
                confidence -= 0.4
            
            # Boost for formal language
            if any(formal in text_lower for formal in ['government', 'committee', 'inquiry', 'report']):
                confidence += 0.1
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating recommendation confidence: {e}")
            return 0.6  # Return base confidence on error

    def _calculate_response_confidence_fixed(self, text: str, matched_terms: List[str]) -> float:
        """FIXED: Calculate confidence score for responses"""
        if not text or not matched_terms:
            return 0.0
        
        confidence = 0.6  # Base confidence
        text_lower = text.lower()
        
        try:
            # Boost for strong response indicators
            if 'government response' in text_lower or 'official response' in text_lower:
                confidence += 0.3
            if any(word in text_lower for word in ['accepted', 'rejected', 'agreed', 'disagreed']):
                confidence += 0.25
            if 'response to recommendation' in text_lower:
                confidence += 0.2
            if any(phrase in text_lower for phrase in ['accepted in full', 'accepted in principle', 'not accepted']):
                confidence += 0.25
            
            # Boost for multiple matched terms
            confidence += min(len(matched_terms) * 0.05, 0.15)
            
            # Boost for implementation language
            if any(word in text_lower for word in ['implement', 'action', 'measure']):
                confidence += 0.15
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating response confidence: {e}")
            return 0.6

    def _classify_recommendation_type_fixed(self, text: str) -> str:
        """FIXED: Classify recommendation type"""
        text_lower = text.lower()
        
        if 'i recommend' in text_lower:
            return 'direct_recommendation'
        elif re.search(r'recommendation\s+\d+', text_lower):
            return 'numbered_recommendation'
        elif any(word in text_lower for word in ['should', 'must', 'ought']):
            return 'prescriptive_recommendation'
        elif any(word in text_lower for word in ['suggest', 'propose']):
            return 'suggestive_recommendation'
        elif any(word in text_lower for word in ['ensure', 'establish', 'implement']):
            return 'action_recommendation'
        else:
            return 'general_recommendation'

    def _classify_response_type_fixed(self, text: str) -> str:
        """FIXED: Classify response type"""
        text_lower = text.lower()
        
        if 'accepted in full' in text_lower:
            return 'accepted_full'
        elif 'accepted in principle' in text_lower:
            return 'accepted_principle'
        elif 'not accepted' in text_lower or 'rejected' in text_lower:
            return 'rejected'
        elif 'partially accepted' in text_lower:
            return 'accepted_partial'
        elif 'government response' in text_lower:
            return 'government_response'
        elif 'will implement' in text_lower:
            return 'implementation_response'
        else:
            return 'general_response'

    def _extract_references_fixed(self, text: str) -> List[str]:
        """FIXED: Extract references from text"""
        references = []
        
        try:
            # Recommendation numbers
            rec_nums = re.findall(r'recommendation\s+(\d+)', text, re.IGNORECASE)
            references.extend([f"rec_{num}" for num in rec_nums])
            
            # Page references
            page_nums = re.findall(r'page\s+(\d+)', text, re.IGNORECASE)
            references.extend([f"page_{num}" for num in page_nums])
            
            # Paragraph references
            para_nums = re.findall(r'paragraph\s+(\d+)', text, re.IGNORECASE)
            references.extend([f"para_{num}" for num in para_nums])
            
            # Section references
            section_nums = re.findall(r'section\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            references.extend([f"section_{num}" for num in section_nums])
            
        except Exception as e:
            self.logger.error(f"Error extracting references: {e}")
        
        return list(set(references))

    def _combine_extraction_results_fixed(self, original_results: List[Dict], hybrid_results: List[Dict]) -> List[Dict[str, Any]]:
        """FIXED: Combine results from original and hybrid extraction methods"""
        combined = []
        seen_texts = set()
        
        try:
            # Combine all results
            all_results = original_results + hybrid_results
            
            # Sort by confidence (highest first)
            all_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            for result in all_results:
                # Create normalized text key for deduplication
                text_key = re.sub(r'\s+', ' ', result.get('text', '').lower().strip())[:100]
                
                if text_key not in seen_texts and len(text_key) > 10:
                    seen_texts.add(text_key)
                    combined.append(result)
        
        except Exception as e:
            self.logger.error(f"Error combining extraction results: {e}")
        
        return combined

    # ============================================================================
    # ORIGINAL METHODS - Maintaining backward compatibility
    # ============================================================================

    def _extract_individual_recommendations_original(self, text: str) -> List[Dict[str, Any]]:
        """Original recommendation extraction logic"""
        recommendations = []
        
        try:
            # Use original patterns
            for pattern in self.government_patterns['recommendation_patterns']:
                matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    if len(match.groups()) >= 2:
                        rec_id = match.group(1) if match.group(1) else str(len(recommendations) + 1)
                        rec_text = match.group(2).strip()
                    else:
                        rec_text = match.group(1).strip() if match.group(1) else match.group(0)
                        rec_id = str(len(recommendations) + 1)
                    
                    if len(rec_text) > 20:
                        recommendations.append({
                            'id': f"original_rec_{rec_id}",
                            'text': rec_text,
                            'type': 'original_pattern',
                            'confidence': 0.85,
                            'extraction_method': 'original_pattern',
                            'number': rec_id
                        })
            
            # Use Infected Blood specific patterns
            i_recommend_matches = re.finditer(self.infected_blood_patterns['i_recommend_block'], text, re.DOTALL)
            for match in i_recommend_matches:
                block_content = match.group(1).strip()
                if len(block_content) > 50:
                    recommendations.append({
                        'id': f"original_i_rec_{len(recommendations) + 1}",
                        'text': block_content,
                        'type': 'i_recommend_block',
                        'confidence': 0.95,
                        'extraction_method': 'original_pattern'
                    })
        
        except Exception as e:
            self.logger.error(f"Error in original recommendation extraction: {e}")
        
        return recommendations

    def _extract_individual_responses_original(self, text: str) -> List[Dict[str, Any]]:
        """Original response extraction logic"""
        responses = []
        
        try:
            for pattern in self.government_patterns['response_patterns']:
                matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    if len(match.groups()) >= 2:
                        resp_id = match.group(1) if match.group(1) else str(len(responses) + 1)
                        resp_text = match.group(2).strip()
                    else:
                        resp_text = match.group(1).strip() if match.group(1) else match.group(0)
                        resp_id = str(len(responses) + 1)
                    
                    if len(resp_text) > 20:
                        responses.append({
                            'id': f"original_resp_{resp_id}",
                            'text': resp_text,
                            'type': 'original_pattern',
                            'confidence': 0.85,
                            'extraction_method': 'original_pattern'
                        })
        
        except Exception as e:
            self.logger.error(f"Error in original response extraction: {e}")
        
        return responses

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Original method - get extraction statistics"""
        return {
            'documents_processed': self.extraction_stats['documents_processed'],
            'sections_extracted': self.extraction_stats['sections_extracted'],
            'recommendations_found': self.extraction_stats['recommendations_found'],
            'responses_found': self.extraction_stats['responses_found'],
            'last_extraction_time': self.extraction_stats['last_extraction_time'],
            'total_processing_time': self.extraction_stats['total_processing_time']
        }

    # ============================================================================
    # NEW CONVENIENCE METHODS
    # ============================================================================

    def quick_extract_recommendations(self, text: str) -> List[str]:
        """NEW: Quick function to get just recommendation text"""
        try:
            recommendations = self.extract_individual_recommendations(text)
            return [rec.get('text', '') for rec in recommendations if rec.get('confidence', 0) >= 0.6]
        except Exception as e:
            self.logger.error(f"Error in quick_extract_recommendations: {e}")
            return []

    def quick_extract_responses(self, text: str) -> List[str]:
        """NEW: Quick function to get just response text"""
        try:
            responses = self.extract_individual_responses(text)
            return [resp.get('text', '') for resp in responses if resp.get('confidence', 0) >= 0.6]
        except Exception as e:
            self.logger.error(f"Error in quick_extract_responses: {e}")
            return []

    def extract_all_content_hybrid(self, text: str, document_source: str = "", inquiry_type: str = "auto") -> Dict[str, Any]:
        """NEW: Main hybrid extraction method for any inquiry document"""
        if self.debug_mode:
            self.logger.info(f"Starting hybrid extraction for {document_source}")
        
        # Auto-detect inquiry type if not specified
        if inquiry_type == "auto":
            inquiry_type = self._detect_inquiry_type(text)
        
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        # Extract using both methods
        recommendations = self.extract_individual_recommendations(text)
        responses = self.extract_individual_responses(text)
        
        # Create results structure
        results = {
            'recommendations': recommendations,
            'responses': responses,
            'extraction_metadata': {
                'document_source': document_source,
                'inquiry_type': inquiry_type,
                'extraction_methods': ['sentence_based', 'pattern_based'],
                'total_recommendations': len(recommendations),
                'total_responses': len(responses),
                'confidence_stats': self._calculate_confidence_stats(recommendations, responses)
            }
        }
        
        if self.debug_mode:
            self._log_results_summary(results)
        
        return results

    # ============================================================================
    # PDF PROCESSING METHODS - Maintaining compatibility
    # ============================================================================

    def extract_sections_from_pdf(self, pdf_content: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """Original method - extract sections from PDF content (maintains compatibility)"""
        try:
            self.extraction_stats['documents_processed'] += 1
            start_time = datetime.now()
            
            # Try pdfplumber first, then PyMuPDF as fallback
            if PDFPLUMBER_AVAILABLE:
                result = self._extract_pages_with_pdfplumber(pdf_content, filename)
            elif PYMUPDF_AVAILABLE:
                result = self._extract_pages_with_pymupdf(pdf_content, filename)
            else:
                result = self._extract_with_fallback(pdf_content, filename)
            
            if result and result.get('full_text'):
                # Process the extracted text for sections
                sections = self._find_recommendation_response_sections(result['full_text'])
                result['sections'] = sections
                result['section_count'] = len(sections)
                
                # Extract individual items using enhanced methods
                recommendations = self.extract_individual_recommendations(result['full_text'])
                responses = self.extract_individual_responses(result['full_text'])
                
                result['individual_recommendations'] = recommendations
                result['individual_responses'] = responses
                
                # Update stats
                self.extraction_stats['recommendations_found'] = len(recommendations)
                self.extraction_stats['responses_found'] = len(responses)
                self.extraction_stats['sections_extracted'] = len(sections)
                self.extraction_stats['last_extraction_time'] = datetime.now()
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                self.extraction_stats['total_processing_time'] += processing_time
                
                result['success'] = True
                result['extraction_method'] = 'hybrid_enhanced'
                result['processing_time'] = processing_time
                
                # Add document analysis
                result['document_analysis'] = self._analyze_document_content(result['full_text'])
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in extract_sections_from_pdf: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'individual_recommendations': [],
                'individual_responses': [],
                'sections': []
            }

    def _extract_pages_with_pdfplumber(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Original PDF extraction with pdfplumber"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                full_text = ""
                page_data = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    full_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                    
                    page_data.append({
                        'page_number': page_num,
                        'text': page_text,
                        'word_count': len(page_text.split()),
                        'char_count': len(page_text)
                    })
                
                return {
                    'success': True,
                    'filename': filename,
                    'full_text': full_text,
                    'pages': page_data,
                    'total_pages': len(pdf.pages),
                    'extraction_method': 'pdfplumber',
                    'file_size_mb': len(pdf_content) / (1024 * 1024),
                    'metadata': {
                        'filename': filename,
                        'total_pages': len(pdf.pages),
                        'file_size_mb': round(len(pdf_content) / (1024 * 1024), 2)
                    }
                }
                
        except Exception as e:
            self.logger.error(f"pdfplumber extraction failed: {e}")
            return {'success': False, 'error': str(e), 'extraction_method': 'pdfplumber'}

    def _extract_pages_with_pymupdf(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Original PDF extraction with PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            full_text = ""
            page_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                page_data.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'word_count': len(page_text.split()),
                    'char_count': len(page_text)
                })
            
            total_pages = len(doc)
            doc.close()
            
            return {
                'success': True,
                'filename': filename,
                'full_text': full_text,
                'pages': page_data,
                'total_pages': total_pages,
                'extraction_method': 'pymupdf',
                'file_size_mb': len(pdf_content) / (1024 * 1024),
                'metadata': {
                    'filename': filename,
                    'total_pages': total_pages,
                    'file_size_mb': round(len(pdf_content) / (1024 * 1024), 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {e}")
            return {'success': False, 'error': str(e), 'extraction_method': 'pymupdf'}

    def _extract_with_fallback(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Fallback extraction method"""
        return {
            'success': False,
            'error': 'No PDF processing libraries available',
            'filename': filename,
            'full_text': '',
            'pages': [],
            'total_pages': 0,
            'extraction_method': 'fallback'
        }

    # ============================================================================
    # HELPER METHODS - Supporting functionality
    # ============================================================================

    def _detect_inquiry_type(self, text: str) -> str:
        """Auto-detect the type of inquiry document"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['infected blood', 'contaminated blood', 'haemophilia', 'hepatitis c']):
            return 'infected_blood'
        elif any(term in text_lower for term in ['public inquiry', 'statutory inquiry']):
            return 'public_inquiry'
        elif any(term in text_lower for term in ['grenfell', 'tower', 'fire']):
            return 'grenfell'
        else:
            return 'generic'

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix sentence boundaries
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', text)
        
        # Normalize numbered items
        text = re.sub(r'\n\s*(\d+)\.\s+', r'\n\1. ', text)
        text = re.sub(r'\n\s*\(([a-z])\)\s+', r'\n(\1) ', text)
        
        # Remove page headers/footers
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()

    def _analyze_document_content(self, text: str) -> Dict[str, Any]:
        """Analyze document content for metadata"""
        analysis = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'has_recommendations': any(re.search(term, text, re.IGNORECASE) for term in self.recommend_terms),
            'has_responses': any(re.search(term, text, re.IGNORECASE) for term in self.response_terms),
            'inquiry_type': self._detect_inquiry_type(text),
            'estimated_pages': len(text) // 3000  # Rough estimate
        }
        
        # Count potential recommendation and response indicators
        analysis['recommendation_indicators'] = len(re.findall(r'recommend', text, re.IGNORECASE))
        analysis['response_indicators'] = len(re.findall(r'response', text, re.IGNORECASE))
        
        return analysis

    def _find_recommendation_response_sections(self, text: str) -> List[Dict[str, Any]]:
        """Original section finding logic"""
        sections = []
        
        try:
            # Find section headers
            section_starts = self._find_section_starts(text)
            
            for i, (header, position) in enumerate(section_starts):
                # Determine section end
                if i + 1 < len(section_starts):
                    end_position = section_starts[i + 1][1]
                    content = text[position:end_position]
                else:
                    content = text[position:]
                
                # Extract section content (remove header)
                lines = content.split('\n')
                if lines:
                    content = '\n'.join(lines[1:])  # Skip header line
                
                section_type = self._classify_section_type(header)
                confidence = self._calculate_section_confidence(header, content)
                
                sections.append({
                    'title': header.strip(),
                    'content': content.strip(),
                    'type': section_type,
                    'page_start': self._estimate_page_number(position, text),
                    'page_end': self._estimate_page_number(position + len(content), text),
                    'word_count': len(content.split()),
                    'confidence': confidence,
                    'position': position,
                    'content_stats': {
                        'word_count': len(content.split()),
                        'char_count': len(content),
                        'numbered_items': len(re.findall(r'^\s*\d+\.', content, re.MULTILINE))
                    }
                })
            
            self.extraction_stats['sections_extracted'] = len(sections)
            
            if self.debug_mode:
                self.logger.info(f"Found {len(sections)} sections using pattern matching")
            
            return sections
            
        except Exception as section_error:
            self.logger.error(f"Error finding sections: {section_error}")
            return []

    def _find_section_starts(self, text: str) -> List[Tuple[str, int]]:
        """Find section start positions"""
        section_starts = []
        
        try:
            for pattern in self.government_patterns['section_headers']:
                matches = re.finditer(pattern, text, re.MULTILINE)
                for match in matches:
                    header = match.group(0)
                    position = match.start()
                    section_starts.append((header, position))
            
            # Sort by position
            section_starts.sort(key=lambda x: x[1])
        
        except Exception as e:
            self.logger.error(f"Error finding section starts: {e}")
        
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

    def _estimate_page_number(self, position: int, text: str) -> int:
        """Estimate page number based on text position"""
        # Count page markers before this position
        text_before = text[:position]
        page_markers = len(re.findall(r'--- Page \d+ ---', text_before))
        return max(1, page_markers)

    def _calculate_confidence_stats(self, recommendations: List[Dict], responses: List[Dict]) -> Dict[str, Any]:
        """Calculate confidence statistics"""
        def get_stats(items):
            if not items:
                return {'mean': 0, 'high_confidence': 0, 'medium_confidence': 0, 'low_confidence': 0}
            
            confidences = [item.get('confidence', 0) for item in items]
            mean_conf = sum(confidences) / len(confidences)
            high = len([c for c in confidences if c >= 0.8])
            medium = len([c for c in confidences if 0.6 <= c < 0.8])
            low = len([c for c in confidences if c < 0.6])
            
            return {
                'mean': round(mean_conf, 3),
                'high_confidence': high,
                'medium_confidence': medium,
                'low_confidence': low
            }
        
        return {
            'recommendations': get_stats(recommendations),
            'responses': get_stats(responses)
        }

    def _log_results_summary(self, results: Dict) -> None:
        """Log summary of extraction results"""
        metadata = results.get('extraction_metadata', {})
        
        self.logger.info(f"Hybrid Extraction Summary for {metadata.get('document_source', 'Unknown')}")
        self.logger.info(f"- Total Recommendations: {metadata.get('total_recommendations', 0)}")
        self.logger.info(f"- Total Responses: {metadata.get('total_responses', 0)}")
        self.logger.info(f"- Inquiry Type: {metadata.get('inquiry_type', 'Unknown')}")

    # ============================================================================
    # COMPATIBILITY METHODS - For existing UI integration
    # ============================================================================

    def compare_inquiry_and_response_documents(self, inquiry_text: str, response_text: str) -> Dict[str, Any]:
        """Compare inquiry recommendations with government responses"""
        try:
            # Extract from both documents
            inquiry_recs = self.extract_individual_recommendations(inquiry_text)
            response_items = self.extract_individual_responses(response_text)
            
            # Try to match recommendations with responses
            matched_pairs = []
            unmatched_recommendations = []
            unmatched_responses = []
            
            # Simple matching based on recommendation numbers and text similarity
            for rec in inquiry_recs:
                matched = False
                rec_refs = rec.get('references', [])
                
                for resp in response_items:
                    resp_refs = resp.get('references', [])
                    
                    # Check if references overlap
                    if any(ref in resp_refs for ref in rec_refs):
                        matched_pairs.append({
                            'recommendation': rec,
                            'response': resp,
                            'match_confidence': 0.9
                        })
                        matched = True
                        break
                
                if not matched:
                    unmatched_recommendations.append(rec)
            
            # Find unmatched responses
            matched_response_ids = {pair['response']['id'] for pair in matched_pairs}
            unmatched_responses = [resp for resp in response_items if resp['id'] not in matched_response_ids]
            
            return {
                'matched_pairs': matched_pairs,
                'unmatched_recommendations': unmatched_recommendations,
                'unmatched_responses': unmatched_responses,
                'summary': {
                    'total_recommendations': len(inquiry_recs),
                    'total_responses': len(response_items),
                    'matched_pairs': len(matched_pairs),
                    'match_rate': len(matched_pairs) / len(inquiry_recs) if inquiry_recs else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing documents: {e}")
            return {
                'matched_pairs': [],
                'unmatched_recommendations': [],
                'unmatched_responses': [],
                'error': str(e)
            }

# ============================================================================
# STANDALONE UTILITY FUNCTIONS - Maintaining compatibility
# ============================================================================

def extract_sections_from_pdf(pdf_content: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
    """Standalone function for PDF extraction"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_sections_from_pdf(pdf_content, filename)

def get_section_summary(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get summary of extracted sections"""
    return {
        'total_sections': len(sections),
        'recommendation_sections': len([s for s in sections if s.get('type') == 'recommendation']),
        'response_sections': len([s for s in sections if s.get('type') == 'response']),
        'other_sections': len([s for s in sections if s.get('type') not in ['recommendation', 'response']])
    }

def validate_section_extraction(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extraction results"""
    return {
        'is_valid': result.get('success', False),
        'has_recommendations': len(result.get('individual_recommendations', [])) > 0,
        'has_responses': len(result.get('individual_responses', [])) > 0,
        'extraction_method': result.get('extraction_method', 'unknown'),
        'sections_found': len(result.get('sections', [])),
        'processing_time': result.get('processing_time', 0)
    }

def extract_government_recommendations(text: str) -> List[Dict[str, Any]]:
    """Extract recommendations from government document text"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_individual_recommendations(text)

def extract_government_responses(text: str) -> List[Dict[str, Any]]:
    """Extract responses from government document text"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_individual_responses(text)

def test_extraction(pdf_path: str = None, pdf_content: bytes = None) -> Dict[str, Any]:
    """Test the extraction functionality"""
    if pdf_path:
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        filename = pdf_path.split('/')[-1]
    elif pdf_content:
        filename = "test_document.pdf"
    else:
        return {'error': 'No PDF provided for testing'}
    
    try:
        extractor = EnhancedSectionExtractor(debug_mode=True)
        result = extractor.extract_sections_from_pdf(pdf_content, filename)
        
        test_summary = {
            'test_passed': result.get('success', False),
            'sections_found': len(result.get('sections', [])),
            'recommendations_found': len(result.get('individual_recommendations', [])),
            'responses_found': len(result.get('individual_responses', [])),
            'extraction_method': result.get('extraction_method', 'unknown'),
            'processing_time': result.get('processing_time', 0),
            'statistics': extractor.get_extraction_statistics()
        }
        
        result['test_summary'] = test_summary
        return result
        
    except Exception as test_error:
        return {
            'error': str(test_error),
            'test_passed': False
        }

def debug_section_detection(text: str) -> Dict[str, Any]:
    """Debug section detection patterns"""
    extractor = EnhancedSectionExtractor(debug_mode=True)
    
    debug_info = {
        'text_length': len(text),
        'has_recommend_terms': any(re.search(pattern, text.lower()) for pattern in extractor.recommend_terms),
        'has_response_terms': any(re.search(pattern, text.lower()) for pattern in extractor.response_terms),
        'recommendation_sentences': len([s for s in sent_tokenize(text) if any(re.search(term, s, re.IGNORECASE) for term in extractor.recommend_terms)]),
        'response_sentences': len([s for s in sent_tokenize(text) if any(re.search(term, s, re.IGNORECASE) for term in extractor.response_terms)]),
        'inquiry_type': extractor._detect_inquiry_type(text),
        'document_analysis': extractor._analyze_document_content(text)
    }
    
    return debug_info

def test_individual_extraction(text: str) -> Dict[str, Any]:
    """Test individual recommendation and response extraction"""
    extractor = EnhancedSectionExtractor(debug_mode=True)
    
    recommendations = extractor.extract_individual_recommendations(text)
    responses = extractor.extract_individual_responses(text)
    
    return {
        'recommendations': recommendations,
        'responses': responses,
        'stats': {
            'total_recommendations': len(recommendations),
            'total_responses': len(responses),
            'high_confidence_recs': len([r for r in recommendations if r.get('confidence', 0) >= 0.8]),
            'high_confidence_resps': len([r for r in responses if r.get('confidence', 0) >= 0.8]),
            'extraction_methods': list(set([r.get('extraction_method', 'unknown') for r in recommendations + responses]))
        },
        'document_analysis': extractor._analyze_document_content(text)
    }

# ============================================================================
# NEW CONVENIENCE FUNCTIONS - Additional functionality
# ============================================================================

def quick_extract_from_text(text: str) -> Dict[str, List[str]]:
    """Quick extraction function for simple use cases"""
    extractor = EnhancedSectionExtractor()
    
    return {
        'recommendations': extractor.quick_extract_recommendations(text),
        'responses': extractor.quick_extract_responses(text)
    }

def extract_with_confidence_filter(text: str, min_confidence: float = 0.8) -> Dict[str, List[Dict[str, Any]]]:
    """Extract with confidence filtering"""
    extractor = EnhancedSectionExtractor()
    
    all_recs = extractor.extract_individual_recommendations(text)
    all_resps = extractor.extract_individual_responses(text)
    
    filtered_recs = [r for r in all_recs if r.get('confidence', 0) >= min_confidence]
    filtered_resps = [r for r in all_resps if r.get('confidence', 0) >= min_confidence]
    
    return {
        'recommendations': filtered_recs,
        'responses': filtered_resps,
        'filter_info': {
            'min_confidence': min_confidence,
            'original_rec_count': len(all_recs),
            'filtered_rec_count': len(filtered_recs),
            'original_resp_count': len(all_resps),
            'filtered_resp_count': len(filtered_resps)
        }
    }

def get_extraction_summary(text: str) -> Dict[str, Any]:
    """Get detailed extraction summary"""
    extractor = EnhancedSectionExtractor()
    
    # Extract everything
    recommendations = extractor.extract_individual_recommendations(text)
    responses = extractor.extract_individual_responses(text)
    
    # Calculate statistics
    rec_confidences = [r.get('confidence', 0) for r in recommendations]
    resp_confidences = [r.get('confidence', 0) for r in responses]
    
    return {
        'text_length': len(text),
        'total_recommendations': len(recommendations),
        'total_responses': len(responses),
        'recommendation_stats': {
            'mean_confidence': sum(rec_confidences) / len(rec_confidences) if rec_confidences else 0,
            'high_confidence': len([c for c in rec_confidences if c >= 0.8]),
            'medium_confidence': len([c for c in rec_confidences if 0.6 <= c < 0.8]),
            'low_confidence': len([c for c in rec_confidences if c < 0.6])
        },
        'response_stats': {
            'mean_confidence': sum(resp_confidences) / len(resp_confidences) if resp_confidences else 0,
            'high_confidence': len([c for c in resp_confidences if c >= 0.8]),
            'medium_confidence': len([c for c in resp_confidences if 0.6 <= c < 0.8]),
            'low_confidence': len([c for c in resp_confidences if c < 0.6])
        },
        'inquiry_type': extractor._detect_inquiry_type(text),
        'has_ibi_patterns': 'infected blood' in text.lower() or 'haemophilia' in text.lower(),
        'extraction_methods_used': list(set([r.get('extraction_method', 'unknown') for r in recommendations + responses]))
    }

def extract_by_inquiry_type(text: str, inquiry_type: str = "auto") -> Dict[str, List[Dict[str, Any]]]:
    """Extract using specific inquiry type approach"""
    extractor = EnhancedSectionExtractor(debug_mode=True)
    results = extractor.extract_all_content_hybrid(text, inquiry_type=inquiry_type)
    
    return {
        'recommendations': results.get('recommendations', []),
        'responses': results.get('responses', []),
        'metadata': results.get('extraction_metadata', {})
    }

# ============================================================================
# TESTING AND EXAMPLE FUNCTIONS
# ============================================================================

def test_extraction_simple(text: str = None) -> Dict[str, Any]:
    """Simple test function with debugging output"""
    if not text:
        text = """
        The committee recommends that the government should implement new safety measures immediately.
        I recommend establishing a monitoring system for compliance.
        It is recommended that annual reviews be conducted.
        
        Government Response:
        Recommendation 1 is accepted in full and will be implemented by March 2025.
        The government response is that recommendation 2 is accepted in principle.
        """
    
    print("=== Enhanced Section Extractor Test ===")
    print(f"Input text length: {len(text)} characters")
    
    extractor = EnhancedSectionExtractor(debug_mode=True)
    
    # Test recommendations
    print("\n--- Testing Recommendation Extraction ---")
    recommendations = extractor.extract_individual_recommendations(text)
    print(f"Recommendations found: {len(recommendations)}")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. Text: {rec['text'][:100]}...")
        print(f"     Confidence: {rec.get('confidence', 0):.2f}")
        print(f"     Type: {rec.get('type', 'unknown')}")
        print(f"     Method: {rec.get('extraction_method', 'unknown')}")
        if rec.get('matched_terms'):
            print(f"     Matched terms: {rec.get('matched_terms', [])}")
    
    # Test responses
    print("\n--- Testing Response Extraction ---")
    responses = extractor.extract_individual_responses(text)
    print(f"Responses found: {len(responses)}")
    for i, resp in enumerate(responses, 1):
        print(f"  {i}. Text: {resp['text'][:100]}...")
        print(f"     Confidence: {resp.get('confidence', 0):.2f}")
        print(f"     Type: {resp.get('type', 'unknown')}")
        print(f"     Method: {resp.get('extraction_method', 'unknown')}")
        if resp.get('matched_terms'):
            print(f"     Matched terms: {resp.get('matched_terms', [])}")
    
    # Get statistics
    stats = extractor.get_extraction_statistics()
    print(f"\n--- Extraction Statistics ---")
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Recommendations found: {stats['recommendations_found']}")
    print(f"Responses found: {stats['responses_found']}")
    
    return {
        'recommendations': recommendations,
        'responses': responses,
        'statistics': stats,
        'test_passed': len(recommendations) > 0 or len(responses) > 0
    }

def example_usage():
    """Example of how to use the enhanced extractor"""
    
    # Example text
    sample_text = """
    The inquiry committee recommends the following actions:
    
    1. The government should establish new safety protocols immediately.
    2. We recommend implementing a comprehensive monitoring system.
    3. It is recommended that annual reviews be conducted.
    
    Government Response:
    Recommendation 1 is accepted in full and will be implemented by March 2025.
    Recommendation 2 is accepted in principle, subject to budget approval.
    Recommendation 3 is not accepted due to resource constraints.
    """
    
    print("=== Enhanced Section Extractor - Example Usage ===")
    
    # Method 1: Quick extraction
    quick_results = quick_extract_from_text(sample_text)
    print(f"\nQuick Extraction:")
    print(f"- Recommendations: {len(quick_results['recommendations'])}")
    print(f"- Responses: {len(quick_results['responses'])}")
    
    # Method 2: Detailed extraction
    extractor = EnhancedSectionExtractor(debug_mode=True)
    recommendations = extractor.extract_individual_recommendations(sample_text)
    responses = extractor.extract_individual_responses(sample_text)
    
    print(f"\nDetailed Extraction:")
    print(f"- Recommendations found: {len(recommendations)}")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['text'][:100]}... (confidence: {rec.get('confidence', 0):.2f}, method: {rec.get('extraction_method', 'unknown')})")
    
    print(f"- Responses found: {len(responses)}")
    for i, resp in enumerate(responses, 1):
        print(f"  {i}. {resp['text'][:100]}... (confidence: {resp.get('confidence', 0):.2f}, method: {resp.get('extraction_method', 'unknown')})")
    
    # Method 3: Get summary
    summary = get_extraction_summary(sample_text)
    print(f"\nExtraction Summary:")
    print(f"- Text length: {summary['text_length']} characters")
    print(f"- Inquiry type: {summary['inquiry_type']}")
    print(f"- Mean recommendation confidence: {summary['recommendation_stats']['mean_confidence']:.2f}")
    print(f"- Mean response confidence: {summary['response_stats']['mean_confidence']:.2f}")
    print(f"- Extraction methods used: {summary['extraction_methods_used']}")

# ============================================================================
# MODULE EXPORTS - Maintaining compatibility
# ============================================================================

__all__ = [
    # Main class
    'EnhancedSectionExtractor',
    
    # Original standalone functions
    'extract_sections_from_pdf',
    'get_section_summary',
    'validate_section_extraction',
    'extract_government_recommendations',
    'extract_government_responses',
    
    # Testing and debugging
    'test_extraction',
    'debug_section_detection', 
    'test_individual_extraction',
    'test_extraction_simple',
    
    # NEW: Additional convenience functions
    'quick_extract_from_text',
    'extract_with_confidence_filter',
    'get_extraction_summary',
    'extract_by_inquiry_type',
    'example_usage',
    
    # Availability flags
    'PDFPLUMBER_AVAILABLE',
    'PYMUPDF_AVAILABLE',
    'NLTK_AVAILABLE'
]

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

if PDFPLUMBER_AVAILABLE and PYMUPDF_AVAILABLE:
    logging.info("✅ Enhanced Section Extractor loaded with full PDF support (pdfplumber + PyMuPDF)")
elif PDFPLUMBER_AVAILABLE:
    logging.info("✅ Enhanced Section Extractor loaded with pdfplumber support")
elif PYMUPDF_AVAILABLE:
    logging.info("✅ Enhanced Section Extractor loaded with PyMuPDF support")
else:
    logging.warning("⚠️ Enhanced Section Extractor loaded with fallback mode (no PDF libraries)")

if NLTK_AVAILABLE:
    logging.info("✅ NLTK sentence tokenization available")
else:
    logging.info("✅ Using fallback sentence tokenization")

logging.info("🎉 Enhanced Section Extractor with Fixed Hybrid Approach - Ready for any inquiry document!")

# Test on import (optional)
if __name__ == "__main__":
    example_usage()
    print("\n" + "="*60)
    test_extraction_simple()
