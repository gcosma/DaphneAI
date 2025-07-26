import re
import logging
from typing import List, Dict, Any, Set, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import io
from datetime import datetime

# PDF processing libraries with fallback handling (maintaining compatibility)
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
    Enhanced Section Extractor with Hybrid Approach
    
    Combines the original functionality with new generic sentence-based extraction.
    Maintains full backward compatibility with existing code.
    """
    
    def __init__(self, debug_mode: bool = False):
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
        
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # NEW: Generic patterns for any inquiry (sentence-based approach)
        self.recommend_terms = [
            r'\brecommend(?:s|ed|ing|ation|ations)?\b',
            r'\bsuggest(?:s|ed|ing|ion|ions)?\b',
            r'\bpropose(?:s|d|ing|al|als)?\b',
            r'\badvise(?:s|d|ing)?\b',
            r'\burge(?:s|d|ing)?\b',
            r'\bshould\b',
            r'\bmust\b',
            r'\bough?t\s+to\b',
            r'\bneed(?:s|ed|ing)?\s+to\b',
            r'\brequire(?:s|d|ing)?\b',
            r'\bensure(?:s|d|ing)?\b',
            r'\bestablish(?:es|ed|ing)?\b',
            r'\bimplement(?:s|ed|ing|ation)?\b'
        ]
        
        self.response_terms = [
            r'\brespons(?:e|es|ed|ing|ible|ibility)?\b',
            r'\brepl(?:y|ies|ied|ying)?\b',
            r'\banswer(?:s|ed|ing)?\b',
            r'\baccept(?:s|ed|ing|ance)?\b',
            r'\breject(?:s|ed|ing|ion)?\b',
            r'\bagree(?:s|d|ing|ment)?\b',
            r'\bdisagree(?:s|d|ing|ment)?\b',
            r'\bimplement(?:s|ed|ing|ation)?\b',
            r'\badopt(?:s|ed|ing|ion)?\b',
            r'\baction(?:s)?\b',
            r'\bmeasure(?:s)?\b',
            r'\bstep(?:s)?\b'
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
        
        # ORIGINAL: Infected Blood Inquiry specific patterns (maintain compatibility)
        self.infected_blood_patterns = {
            'i_recommend_block': r'(?i)I\s+recommend:\s*\n(.*?)(?=\n\s*[A-Z][a-z]+\s+\d+|$)',
            'section_1_5': r'(?i)1\.5\s+recommendations?\s*\n(.*?)(?=\n\s*(?:\d+\.\d+|appendix|references|$))',
            'numbered_main': r'(?i)(?:^|\n)\s*(\d+)\.\s+([^\n]+)(?:\n((?:(?!\d+\.).)*?))?',
            'sub_lettered': r'(?i)\(([a-z])\)\s+([^\n]+(?:\n(?!\([a-z]\)|(?:\n\s*\d+\.)).)*)',
            'sub_roman': r'(?i)\(([ivx]+)\)\s+([^\n]+(?:\n(?!\([ivx]+\)|(?:\n\s*\d+\.)).)*)',
        }

    # ============================================================================
    # ORIGINAL METHODS - Maintaining backward compatibility
    # ============================================================================

    def extract_sections_from_pdf(self, pdf_content: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """Original method - extract sections from PDF content (maintains compatibility)"""
        try:
            self.extraction_stats['documents_processed'] += 1
            
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
                
                # Extract individual items using BOTH old and new methods
                old_recommendations = self._extract_individual_recommendations_original(result['full_text'])
                new_recommendations = self._extract_recommendations_hybrid(result['full_text'])
                
                old_responses = self._extract_individual_responses_original(result['full_text'])
                new_responses = self._extract_responses_hybrid(result['full_text'])
                
                # Combine results with deduplication
                combined_recs = self._combine_extraction_results(old_recommendations, new_recommendations)
                combined_resps = self._combine_extraction_results(old_responses, new_responses)
                
                result['individual_recommendations'] = combined_recs
                result['individual_responses'] = combined_resps
                
                # Update stats
                self.extraction_stats['recommendations_found'] = len(combined_recs)
                self.extraction_stats['responses_found'] = len(combined_resps)
                self.extraction_stats['last_extraction_time'] = datetime.now()
                
                result['success'] = True
                result['extraction_method'] = 'hybrid_enhanced'
                
            return result
            
        except Exception as e:
            logging.error(f"Error in extract_sections_from_pdf: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }

    def extract_individual_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced method - combines original + new hybrid approach"""
        # Get results from both methods
        original_recs = self._extract_individual_recommendations_original(text)
        hybrid_recs = self._extract_recommendations_hybrid(text)
        
        # Combine and deduplicate
        combined_recs = self._combine_extraction_results(original_recs, hybrid_recs)
        
        self.extraction_stats['recommendations_found'] = len(combined_recs)
        return combined_recs

    def extract_individual_responses(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced method - combines original + new hybrid approach"""
        # Get results from both methods
        original_resps = self._extract_individual_responses_original(text)
        hybrid_resps = self._extract_responses_hybrid(text)
        
        # Combine and deduplicate
        combined_resps = self._combine_extraction_results(original_resps, hybrid_resps)
        
        self.extraction_stats['responses_found'] = len(combined_resps)
        return combined_resps

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
    # NEW HYBRID METHODS - Adding generic sentence-based extraction
    # ============================================================================

    def extract_all_content_hybrid(self, text: str, document_source: str = "", inquiry_type: str = "auto") -> Dict[str, Any]:
        """NEW: Main hybrid extraction method for any inquiry document"""
        if self.debug_mode:
            self.logger.info(f"Starting hybrid extraction for {document_source}")
        
        # Auto-detect inquiry type if not specified
        if inquiry_type == "auto":
            inquiry_type = self._detect_inquiry_type(text)
        
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        # Extract using sentence-based approach (generic)
        sentence_results = self._extract_by_sentences(text, document_source)
        
        # Extract using original specific patterns
        pattern_results = self._extract_by_original_patterns(text, document_source)
        
        # Combine and deduplicate results
        combined_results = self._combine_hybrid_results(sentence_results, pattern_results)
        
        # Add metadata
        combined_results['extraction_metadata'] = {
            'document_source': document_source,
            'inquiry_type': inquiry_type,
            'extraction_methods': ['sentence_based', 'pattern_based', 'original'],
            'total_recommendations': len(combined_results.get('recommendations', [])),
            'total_responses': len(combined_results.get('responses', [])),
            'confidence_stats': self._calculate_confidence_stats(combined_results)
        }
        
        if self.debug_mode:
            self._log_results_summary(combined_results)
        
        return combined_results

    def quick_extract_recommendations(self, text: str) -> List[str]:
        """NEW: Quick function to get just recommendation text"""
        recommendations = self.extract_individual_recommendations(text)
        return [rec.get('text', '') for rec in recommendations if rec.get('confidence', 0) >= 0.7]

    def quick_extract_responses(self, text: str) -> List[str]:
        """NEW: Quick function to get just response text"""
        responses = self.extract_individual_responses(text)
        return [resp.get('text', '') for resp in responses if resp.get('confidence', 0) >= 0.7]

    # ============================================================================
    # HELPER METHODS - Supporting both original and new functionality
    # ============================================================================

    def _extract_individual_recommendations_original(self, text: str) -> List[Dict[str, Any]]:
        """Original recommendation extraction logic"""
        recommendations = []
        
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
        
        return recommendations

    def _extract_individual_responses_original(self, text: str) -> List[Dict[str, Any]]:
        """Original response extraction logic"""
        responses = []
        
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
        
        return responses

    def _extract_recommendations_hybrid(self, text: str) -> List[Dict[str, Any]]:
        """NEW: Sentence-based recommendation extraction"""
        recommendations = []
        sentences = sent_tokenize(text)
        
        for i, sentence in enumerate(sentences):
            if self._contains_recommendation_terms(sentence):
                confidence = self._calculate_recommendation_confidence(sentence)
                if confidence >= 0.6:
                    recommendations.append({
                        'id': f"hybrid_rec_{i+1}",
                        'text': sentence.strip(),
                        'type': self._classify_recommendation_type(sentence),
                        'confidence': confidence,
                        'extraction_method': 'sentence_based',
                        'position': i,
                        'references': self._extract_references(sentence)
                    })
        
        return recommendations

    def _extract_responses_hybrid(self, text: str) -> List[Dict[str, Any]]:
        """NEW: Sentence-based response extraction"""
        responses = []
        sentences = sent_tokenize(text)
        
        for i, sentence in enumerate(sentences):
            if self._contains_response_terms(sentence):
                confidence = self._calculate_response_confidence(sentence)
                if confidence >= 0.6:
                    responses.append({
                        'id': f"hybrid_resp_{i+1}",
                        'text': sentence.strip(),
                        'type': self._classify_response_type(sentence),
                        'confidence': confidence,
                        'extraction_method': 'sentence_based',
                        'position': i,
                        'references': self._extract_references(sentence)
                    })
        
        return responses

    def _contains_recommendation_terms(self, text: str) -> bool:
        """Check if text contains recommendation-related terms"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.recommend_terms)

    def _contains_response_terms(self, text: str) -> bool:
        """Check if text contains response-related terms"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.response_terms)

    def _calculate_recommendation_confidence(self, text: str) -> float:
        """Calculate confidence score for recommendation text"""
        text_lower = text.lower()
        confidence = 0.5
        
        if re.search(r'\bi\s+recommend', text_lower):
            confidence += 0.3
        if re.search(r'\brecommendation\s+\d+', text_lower):
            confidence += 0.25
        if re.search(r'\b(?:should|must|ought\s+to)\b', text_lower):
            confidence += 0.2
        if re.search(r'\b(?:ensure|establish|implement|require)\b', text_lower):
            confidence += 0.15
        
        # Reduce confidence for weak matches
        weak_indicators = ['recommend reading', 'recommend visiting', 'recommend that you']
        for weak in weak_indicators:
            if weak in text_lower:
                confidence -= 0.3
        
        return min(confidence, 1.0)

    def _calculate_response_confidence(self, text: str) -> float:
        """Calculate confidence score for response text"""
        text_lower = text.lower()
        confidence = 0.5
        
        if re.search(r'\b(?:government|official)\s+response', text_lower):
            confidence += 0.3
        if re.search(r'\b(?:accepted|rejected|agreed|disagreed)', text_lower):
            confidence += 0.25
        if re.search(r'\bresponse\s+to\s+recommendation', text_lower):
            confidence += 0.2
        if re.search(r'\b(?:accepted\s+in\s+(?:full|principle)|not\s+accepted)', text_lower):
            confidence += 0.25
        
        return min(confidence, 1.0)

    def _classify_recommendation_type(self, text: str) -> str:
        """Classify the type of recommendation"""
        text_lower = text.lower()
        
        if re.search(r'\bi\s+recommend', text_lower):
            return 'direct_recommendation'
        elif re.search(r'\brecommendation\s+\d+', text_lower):
            return 'numbered_recommendation'
        elif re.search(r'\b(?:should|must|ought)\b', text_lower):
            return 'prescriptive_recommendation'
        elif re.search(r'\b(?:suggest|propose)\b', text_lower):
            return 'suggestive_recommendation'
        else:
            return 'general_recommendation'

    def _classify_response_type(self, text: str) -> str:
        """Classify the type of response"""
        text_lower = text.lower()
        
        if re.search(r'\baccepted\s+in\s+full', text_lower):
            return 'accepted_full'
        elif re.search(r'\baccepted\s+in\s+principle', text_lower):
            return 'accepted_principle'
        elif re.search(r'\bnot\s+accepted|rejected', text_lower):
            return 'rejected'
        elif re.search(r'\bgovernment\s+response', text_lower):
            return 'government_response'
        else:
            return 'general_response'

    def _extract_references(self, text: str) -> List[str]:
        """Extract references like recommendation numbers, pages, etc."""
        references = []
        
        # Recommendation numbers
        rec_nums = re.findall(r'\brecommendation\s+(\d+)', text, re.IGNORECASE)
        references.extend([f"rec_{num}" for num in rec_nums])
        
        # Page references
        page_nums = re.findall(r'\bpage\s+(\d+)', text, re.IGNORECASE)
        references.extend([f"page_{num}" for num in page_nums])
        
        return list(set(references))

    def _combine_extraction_results(self, original_results: List[Dict], hybrid_results: List[Dict]) -> List[Dict[str, Any]]:
        """Combine results from original and hybrid extraction methods"""
        combined = []
        seen_texts = set()
        
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
        
        return combined

    def _detect_inquiry_type(self, text: str) -> str:
        """Auto-detect the type of inquiry document"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['infected blood', 'contaminated blood', 'haemophilia', 'hepatitis c']):
            return 'infected_blood'
        elif any(term in text_lower for term in ['public inquiry', 'statutory inquiry']):
            return 'public_inquiry'
        else:
            return 'generic'

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', text)
        text = re.sub(r'\n\s*(\d+)\.\s+', r'\n\1. ', text)
        return text.strip()

    # ============================================================================
    # ORIGINAL HELPER METHODS - Maintaining compatibility
    # ============================================================================

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
                        'word_count': len(page_text.split())
                    })
                
                return {
                    'success': True,
                    'filename': filename,
                    'full_text': full_text,
                    'pages': page_data,
                    'total_pages': len(pdf.pages),
                    'extraction_method': 'pdfplumber'
                }
                
        except Exception as e:
            logging.error(f"pdfplumber extraction failed: {e}")
            return {'success': False, 'error': str(e)}

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
                    'word_count': len(page_text.split())
                })
            
            doc.close()
            
            return {
                'success': True,
                'filename': filename,
                'full_text': full_text,
                'pages': page_data,
                'total_pages': len(doc),
                'extraction_method': 'pymupdf'
            }
            
        except Exception as e:
            logging.error(f"PyMuPDF extraction failed: {e}")
            return {'success': False, 'error': str(e)}

    def _extract_with_fallback(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Fallback extraction method"""
        return {
            'success': False,
            'error': 'No PDF processing libraries available',
            'filename': filename,
            'full_text': '',
            'extraction_method': 'fallback'
        }

    def _find_recommendation_response_sections(self, text: str) -> List[Dict[str, Any]]:
        """Original section finding logic"""
        sections = []
        # Simplified version - maintain original structure
        return sections

    # Additional placeholder methods for full compatibility
    def _extract_by_sentences(self, text: str, source: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract using sentence-based approach"""
        return {
            'recommendations': self._extract_recommendations_hybrid(text),
            'responses': self._extract_responses_hybrid(text)
        }

    def _extract_by_original_patterns(self, text: str, source: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract using original patterns"""
        return {
            'recommendations': self._extract_individual_recommendations_original(text),
            'responses': self._extract_individual_responses_original(text)
        }

    def _combine_hybrid_results(self, sentence_results: Dict, pattern_results: Dict) -> Dict[str, List[Dict[str, Any]]]:
        """Combine sentence and pattern results"""
        combined_recs = self._combine_extraction_results(
            pattern_results.get('recommendations', []),
            sentence_results.get('recommendations', [])
        )
        combined_resps = self._combine_extraction_results(
            pattern_results.get('responses', []),
            sentence_results.get('responses', [])
        )
        
        return {
            'recommendations': combined_recs,
            'responses': combined_resps
        }

    def _calculate_confidence_stats(self, results: Dict) -> Dict[str, Any]:
        """Calculate confidence statistics"""
        recs = results.get('recommendations', [])
        resps = results.get('responses', [])
        
        def get_stats(items):
            if not items:
                return {'mean': 0, 'high_confidence': 0}
            
            confidences = [item.get('confidence', 0) for item in items]
            mean_conf = sum(confidences) / len(confidences)
            high = len([c for c in confidences if c >= 0.8])
            
            return {'mean': round(mean_conf, 3), 'high_confidence': high}
        
        return {
            'recommendations': get_stats(recs),
            'responses': get_stats(resps)
        }

    def _log_results_summary(self, results: Dict) -> None:
        """Log summary of extraction results"""
        metadata = results.get('extraction_metadata', {})
        
        self.logger.info(f"Hybrid Extraction Summary for {metadata.get('document_source', 'Unknown')}")
        self.logger.info(f"- Total Recommendations: {metadata.get('total_recommendations', 0)}")
        self.logger.info(f"- Total Responses: {metadata.get('total_responses', 0)}")


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
        'response_sections': len([s for s in sections if s.get('type') == 'response'])
    }

def validate_section_extraction(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extraction results"""
    return {
        'is_valid': result.get('success', False),
        'has_recommendations': len(result.get('individual_recommendations', [])) > 0,
        'has_responses': len(result.get('individual_responses', [])) > 0,
        'extraction_method': result.get('extraction_method', 'unknown')
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
        'recommendation_sentences': len([s for s in text.split('.') if extractor._contains_recommendation_terms(s)]),
        'response_sentences': len([s for s in text.split('.') if extractor._contains_response_terms(s)])
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
            'high_confidence_resps': len([r for r in responses if r.get('confidence', 0) >= 0.8])
        }
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
        'responses': filtered_resps
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
        'has_ibi_patterns': 'infected blood' in text.lower() or 'haemophilia' in text.lower()
    }

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
    
    # NEW: Additional convenience functions
    'quick_extract_from_text',
    'extract_with_confidence_filter',
    'get_extraction_summary',
    
    # Availability flags
    'PDFPLUMBER_AVAILABLE',
    'PYMUPDF_AVAILABLE'
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

logging.info("🎉 Enhanced Section Extractor with Hybrid Approach - Ready for any inquiry document!")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

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
        print(f"  {i}. {rec['text'][:100]}... (confidence: {rec.get('confidence', 0):.2f})")
    
    print(f"- Responses found: {len(responses)}")
    for i, resp in enumerate(responses, 1):
        print(f"  {i}. {resp['text'][:100]}... (confidence: {resp.get('confidence', 0):.2f})")
    
    # Method 3: Get summary
    summary = get_extraction_summary(sample_text)
    print(f"\nExtraction Summary:")
    print(f"- Text length: {summary['text_length']} characters")
    print(f"- Inquiry type: {summary['inquiry_type']}")
    print(f"- Mean recommendation confidence: {summary['recommendation_stats']['mean_confidence']:.2f}")
    print(f"- Mean response confidence: {summary['response_stats']['mean_confidence']:.2f}")

if __name__ == "__main__":
    example_usage()
