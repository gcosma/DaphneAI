# ===============================================
# UPDATED DOCUMENT PROCESSOR - INFECTED BLOOD INQUIRY OPTIMIZED
# modules/document_processor.py
# ===============================================

import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib

# PDF processing imports with fallback
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available")

class DocumentProcessor:
    """
    Enhanced document processor specifically optimized for Infected Blood Inquiry documents.
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the document processor"""
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        # Processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'total_pages_processed': 0,
            'extraction_errors': 0,
            'average_processing_time': 0.0
        }
        
        # Initialize patterns specifically for Infected Blood Inquiry
        self._initialize_infected_blood_patterns()
        
        self.logger.info("DocumentProcessor initialized for Infected Blood Inquiry extraction")

    def _initialize_infected_blood_patterns(self):
        """Initialize patterns specifically for Infected Blood Inquiry documents"""
        
        # Patterns for narrative/historical style recommendations
        self.narrative_recommendation_patterns = [
            # "it was agreed that..." patterns
            r'(?i)it\s+was\s+agreed\s+that\s+(?:officials?\s+)?(?:would\s+)?([^.!?]{20,200})[.!?]',
            
            # "option favoured" patterns  
            r'(?i)(?:the\s+)?option\s+(?:favoured|preferred|recommended)\s+(?:by\s+[^.!?]{1,50}\s+)?(?:was\s+)?([^.!?]{20,200})[.!?]',
            
            # "officials recommended" patterns
            r'(?i)officials?\s+(?:thought\s+that\s+[^.!?]{1,100}\s+and\s+)?recommended\s+([^.!?]{20,200})[.!?]',
            
            # General agreement/recommendation patterns
            r'(?i)(?:officials?\s+|ministers?\s+|government\s+)?(?:agreed|recommended|suggested|proposed)\s+(?:that\s+)?([^.!?]{20,200})[.!?]',
            
            # Seeking agreement patterns
            r'(?i)seeking\s+agreement\s+(?:to\s+)?(?:officials?\s+)?([^.!?]{20,200})[.!?]',
            
            # Minute/memo patterns
            r'(?i)(?:draft\s+)?minute\s+(?:for\s+[^.!?]{1,50}\s+)?(?:to\s+)?(?:send\s+to\s+[^.!?]{1,50}\s+)?seeking\s+([^.!?]{20,200})[.!?]'
        ]
        
        # Patterns for narrative/historical style responses
        self.narrative_response_patterns = [
            # Direct response patterns
            r'(?i)(?:John\s+Moore|Moore)\s+(?:was\s+)?(?:unconvinced|responded|replied|stated|wrote|said)(?:\s+that)?(?:\s+he\s+felt)?(?:\s+that)?\s+([^.!?]{20,200})[.!?]',
            
            # Written response patterns
            r'(?i)(?:he\s+)?wrote\s+(?:to\s+[^.!?]{1,50}\s+)?(?:on\s+[^.!?]{1,20}\s+)?(?:that\s+)?([^.!?]{20,200})[.!?]',
            
            # Response/responding patterns
            r'(?i)responding\s+that\s+([^.!?]{20,200})[.!?]',
            
            # Position/view patterns
            r'(?i)(?:remained\s+firm\s+in\s+his\s+view\s+that|his\s+position\s+was\s+that|his\s+view\s+that)\s+([^.!?]{20,200})[.!?]',
            
            # Advocated/recommended response patterns
            r'(?i)(?:John\s+Major|Major)\s+(?:who\s+was\s+[^.!?]{1,50}\s+)?advocated\s+([^.!?]{20,200})[.!?]',
            
            # General response patterns
            r'(?i)(?:government\s+|minister\s+|secretary\s+)?(?:response|reply)\s+(?:was\s+)?(?:that\s+)?([^.!?]{20,200})[.!?]'
        ]
        
        # Quote extraction patterns for direct statements
        self.quote_patterns = [
            # Direct quotes - these often contain key responses
            r'(?i)"([^"]{50,500})"',
            
            # Attributed quotes
            r'(?i)(?:he\s+|she\s+|they\s+)?(?:wrote|said|stated|declared):\s*"([^"]{50,500})"',
            
            # Quote with attribution
            r'(?i)"([^"]{50,500})"\s*(?:wrote|said|stated)\s+[A-Z][a-z]+\s+[A-Z][a-z]+'
        ]
        
        # Meeting/decision patterns
        self.decision_patterns = [
            # Meeting decisions
            r'(?i)following\s+(?:a\s+)?meeting\s+(?:between\s+[^.!?]{1,100}\s+)?(?:it\s+was\s+agreed\s+that\s+)?([^.!?]{20,200})[.!?]',
            
            # Background context that might contain recommendations
            r'(?i)it\s+was\s+against\s+this\s+background\s+(?:that\s+)?([^.!?]{20,200})[.!?]',
            
            # Campaign/pressure patterns
            r'(?i)(?:campaign|pressure|lobby)\s+(?:for\s+|to\s+)?([^.!?]{20,200})[.!?]'
        ]

    def extract_text_from_pdf(self, pdf_path: str, extract_sections_only: bool = True) -> Optional[Dict[str, Any]]:
        """
        Main extraction function optimized for Infected Blood Inquiry documents.
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting Infected Blood Inquiry extraction for: {pdf_path}")
            
            # Step 1: Extract basic text from all pages
            text_result = self._extract_basic_text(pdf_path)
            if not text_result:
                self.logger.error(f"Failed to extract basic text from {pdf_path}")
                return self._create_error_result(pdf_path, "Failed to extract basic text")
            
            full_text = text_result['text']
            pages_data = text_result['pages']
            
            # Step 2: Extract narrative recommendations and responses
            recommendations = self._extract_narrative_recommendations(full_text)
            responses = self._extract_narrative_responses(full_text)
            
            # Step 3: Extract quotes (often contain key content)
            quote_content = self._extract_quotes(full_text)
            
            # Step 4: Extract metadata
            metadata = self._extract_metadata_with_pymupdf(pdf_path) if PYMUPDF_AVAILABLE else {}
            
            # Step 5: Analyze document structure
            document_analysis = self._analyze_infected_blood_document(full_text)
            
            # Step 6: Create comprehensive result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'filename': Path(pdf_path).name,
                'text': full_text,
                'content': full_text,  # Alias for compatibility
                'sections': [],  # Will be populated by section analysis
                'recommendations': recommendations,
                'responses': responses,
                'quotes': quote_content,
                'metadata': {
                    **metadata,
                    'processing_mode': 'infected_blood_narrative',
                    'processing_time_seconds': processing_time,
                    'extraction_method': 'narrative_pattern_based',
                    'document_analysis': document_analysis,
                    'filename': Path(pdf_path).name,
                    'file_path': str(pdf_path)
                },
                'processed_at': datetime.now().isoformat(),
                'success': True,
                'extractor_version': 'InfectedBlood_v1.0',
                'statistics': {
                    'total_text_length': len(full_text),
                    'recommendations_found': len(recommendations),
                    'responses_found': len(responses),
                    'quotes_found': len(quote_content),
                    'pages_processed': len(pages_data)
                }
            }
            
            # Update processing statistics
            self.processing_stats['documents_processed'] += 1
            self.processing_stats['total_pages_processed'] += len(pages_data)
            
            self.logger.info(f"Successfully processed {Path(pdf_path).name} - Found {len(recommendations)} recommendations, {len(responses)} responses")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            self.processing_stats['extraction_errors'] += 1
            return self._create_error_result(pdf_path, str(e))

    def _extract_narrative_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """Extract recommendations from narrative text"""
        recommendations = []
        seen_content = set()
        
        for pattern in self.narrative_recommendation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                if len(match.groups()) > 0:
                    rec_text = match.group(1).strip()
                else:
                    rec_text = match.group(0).strip()
                
                # Clean up the text
                rec_text = self._clean_extracted_text(rec_text)
                
                # Filter by length and content quality
                if len(rec_text) < 30 or len(rec_text) > 500:
                    continue
                
                # Check for duplicates
                text_hash = hashlib.md5(rec_text.lower().encode()).hexdigest()
                if text_hash in seen_content:
                    continue
                seen_content.add(text_hash)
                
                # Calculate confidence based on pattern match and content
                confidence = self._calculate_confidence(rec_text, pattern, 'recommendation')
                
                if confidence > 0.3:  # Lower threshold for narrative content
                    recommendations.append({
                        'id': f"narrative_rec_{len(recommendations) + 1}",
                        'content': rec_text,
                        'text': rec_text,  # Alias for compatibility
                        'type': 'narrative_recommendation',
                        'confidence': confidence,
                        'extraction_method': 'narrative_pattern',
                        'pattern_used': pattern[:50] + "..." if len(pattern) > 50 else pattern,
                        'context': self._get_context(text, match.start(), match.end())
                    })
        
        # Also check decision patterns
        for pattern in self.decision_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                if len(match.groups()) > 0:
                    rec_text = match.group(1).strip()
                else:
                    rec_text = match.group(0).strip()
                
                rec_text = self._clean_extracted_text(rec_text)
                
                if 30 <= len(rec_text) <= 500:
                    text_hash = hashlib.md5(rec_text.lower().encode()).hexdigest()
                    if text_hash not in seen_content:
                        seen_content.add(text_hash)
                        
                        confidence = self._calculate_confidence(rec_text, pattern, 'recommendation')
                        
                        if confidence > 0.3:
                            recommendations.append({
                                'id': f"decision_rec_{len(recommendations) + 1}",
                                'content': rec_text,
                                'text': rec_text,
                                'type': 'decision_recommendation',
                                'confidence': confidence,
                                'extraction_method': 'decision_pattern',
                                'pattern_used': pattern[:50] + "...",
                                'context': self._get_context(text, match.start(), match.end())
                            })
        
        return recommendations

    def _extract_narrative_responses(self, text: str) -> List[Dict[str, Any]]:
        """Extract responses from narrative text"""
        responses = []
        seen_content = set()
        
        for pattern in self.narrative_response_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                if len(match.groups()) > 0:
                    resp_text = match.group(1).strip()
                else:
                    resp_text = match.group(0).strip()
                
                # Clean up the text
                resp_text = self._clean_extracted_text(resp_text)
                
                # Filter by length and content quality
                if len(resp_text) < 30 or len(resp_text) > 500:
                    continue
                
                # Check for duplicates
                text_hash = hashlib.md5(resp_text.lower().encode()).hexdigest()
                if text_hash in seen_content:
                    continue
                seen_content.add(text_hash)
                
                # Calculate confidence
                confidence = self._calculate_confidence(resp_text, pattern, 'response')
                
                if confidence > 0.3:
                    responses.append({
                        'id': f"narrative_resp_{len(responses) + 1}",
                        'content': resp_text,
                        'text': resp_text,  # Alias for compatibility
                        'type': 'narrative_response',
                        'confidence': confidence,
                        'extraction_method': 'narrative_pattern',
                        'pattern_used': pattern[:50] + "..." if len(pattern) > 50 else pattern,
                        'context': self._get_context(text, match.start(), match.end())
                    })
        
        return responses

    def _extract_quotes(self, text: str) -> List[Dict[str, Any]]:
        """Extract quoted content which often contains key statements"""
        quotes = []
        seen_content = set()
        
        for pattern in self.quote_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                quote_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                
                # Clean up the text
                quote_text = self._clean_extracted_text(quote_text)
                
                # Filter by length
                if len(quote_text) < 50 or len(quote_text) > 500:
                    continue
                
                # Check for duplicates
                text_hash = hashlib.md5(quote_text.lower().encode()).hexdigest()
                if text_hash in seen_content:
                    continue
                seen_content.add(text_hash)
                
                # Quotes often contain important responses or policy statements
                confidence = 0.8  # High confidence for quoted material
                
                quotes.append({
                    'id': f"quote_{len(quotes) + 1}",
                    'content': quote_text,
                    'text': quote_text,
                    'type': 'quoted_statement',
                    'confidence': confidence,
                    'extraction_method': 'quote_pattern',
                    'context': self._get_context(text, match.start(), match.end())
                })
        
        return quotes

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'\d+\s*$', '', text)  # Remove trailing page numbers
        text = re.sub(r'^[^\w]+', '', text)  # Remove leading non-word characters
        text = re.sub(r'[^\w\s.!?,:;()-]+$', '', text)  # Remove trailing special chars
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?', ':')):
            text += '.'
        
        return text.strip()

    def _calculate_confidence(self, text: str, pattern: str, extraction_type: str) -> float:
        """Calculate confidence score for extracted content"""
        confidence = 0.5  # Base confidence
        
        text_lower = text.lower()
        
        # Length factor
        if 50 <= len(text) <= 200:
            confidence += 0.2
        elif len(text) > 200:
            confidence += 0.1
        
        # Content quality factors
        if extraction_type == 'recommendation':
            # Look for recommendation indicators
            rec_indicators = ['agree', 'recommend', 'should', 'propose', 'suggest', 'option', 'work', 'provide']
            matches = sum(1 for indicator in rec_indicators if indicator in text_lower)
            confidence += min(matches * 0.05, 0.2)
            
        elif extraction_type == 'response':
            # Look for response indicators
            resp_indicators = ['respond', 'wrote', 'said', 'view', 'feel', 'position', 'unconvinced', 'maintain']
            matches = sum(1 for indicator in resp_indicators if indicator in text_lower)
            confidence += min(matches * 0.05, 0.2)
        
        # Government/official language
        if any(term in text_lower for term in ['government', 'minister', 'secretary', 'official', 'department']):
            confidence += 0.1
        
        # Proper sentence structure
        if text.strip().endswith(('.', '!', '?')):
            confidence += 0.05
        
        return min(confidence, 1.0)

    def _get_context(self, full_text: str, start_pos: int, end_pos: int, window: int = 150) -> str:
        """Get surrounding context for extracted content"""
        context_start = max(0, start_pos - window)
        context_end = min(len(full_text), end_pos + window)
        return full_text[context_start:context_end].strip()

    def _analyze_infected_blood_document(self, text: str) -> Dict[str, Any]:
        """Analyze document structure specifically for Infected Blood Inquiry"""
        analysis = {
            'document_type': 'infected_blood_inquiry',
            'is_government_response': 'government response' in text.lower(),
            'contains_recommendations': len(re.findall(r'\brecommend', text, re.IGNORECASE)) > 0,
            'contains_responses': len(re.findall(r'\brespond', text, re.IGNORECASE)) > 0,
            'has_quotes': len(re.findall(r'"[^"]{50,}"', text)) > 0,
            'mentions_compensation': 'compensation' in text.lower(),
            'mentions_haemophilia': 'haemophilia' in text.lower() or 'haemophiliacs' in text.lower(),
            'has_dates': len(re.findall(r'\b\d{1,2}\s+\w+\s+\d{4}\b', text)) > 0,
            'word_count': len(text.split()),
            'narrative_style': True  # This is narrative historical text
        }
        
        return analysis

    def _extract_basic_text(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Extract basic text from PDF using available libraries"""
        try:
            if PDFPLUMBER_AVAILABLE:
                return self._extract_with_pdfplumber(pdf_path)
            elif PYMUPDF_AVAILABLE:
                return self._extract_with_pymupdf(pdf_path)
            else:
                self.logger.error("No PDF processing libraries available")
                return None
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return None

    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using pdfplumber"""
        all_text = []
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean page text
                        page_text = re.sub(r'\s+', ' ', page_text)
                        all_text.append(page_text)
                        pages_data.append({
                            'page_number': page_num,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num}: {e}")
        
        return {
            'text': '\n'.join(all_text),
            'pages': pages_data,
            'total_pages': len(pages_data)
        }

    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF"""
        all_text = []
        pages_data = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    page_text = re.sub(r'\s+', ' ', page_text)
                    all_text.append(page_text)
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
            except Exception as e:
                self.logger.warning(f"Error extracting page {page_num + 1}: {e}")
        
        doc.close()
        
        return {
            'text': '\n'.join(all_text),
            'pages': pages_data,
            'total_pages': len(pages_data)
        }

    def _extract_metadata_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF"""
        metadata = {}
        try:
            doc = fitz.open(pdf_path)
            pdf_metadata = doc.metadata
            metadata.update({
                'title': pdf_metadata.get('title', ''),
                'author': pdf_metadata.get('author', ''),
                'subject': pdf_metadata.get('subject', ''),
                'creator': pdf_metadata.get('creator', ''),
                'producer': pdf_metadata.get('producer', ''),
                'creation_date': pdf_metadata.get('creationDate', ''),
                'modification_date': pdf_metadata.get('modDate', ''),
                'page_count': len(doc)
            })
            doc.close()
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {e}")
        
        return metadata

    def _create_error_result(self, pdf_path: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'filename': Path(pdf_path).name,
            'text': '',
            'content': '',
            'sections': [],
            'recommendations': [],
            'responses': [],
            'quotes': [],
            'metadata': {
                'error': error_message,
                'filename': Path(pdf_path).name,
                'file_path': str(pdf_path)
            },
            'processed_at': datetime.now().isoformat(),
            'success': False,
            'error': error_message,
            'extractor_version': 'InfectedBlood_v1.0'
        }
