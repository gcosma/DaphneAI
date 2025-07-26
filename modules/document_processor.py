# ===============================================
# DOCUMENT PROCESSOR - SIMPLIFIED WITH INTELLIGENT EXTRACTION
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

if not PDFPLUMBER_AVAILABLE and not PYMUPDF_AVAILABLE:
    logging.error("No PDF processing libraries available! Install pdfplumber or PyMuPDF")

class DocumentProcessor:
    """
    Enhanced document processor for government inquiry documents.
    Always uses intelligent section extraction optimized for recommendations and responses.
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
        
        # Enhanced patterns for government documents
        self._initialize_extraction_patterns()
        
        self.logger.info("DocumentProcessor initialized with intelligent extraction")

    def _initialize_extraction_patterns(self):
        """Initialize comprehensive patterns for UK government documents"""
        
        # Recommendation section patterns
        self.recommendation_patterns = {
            'section_headers': [
                r'(?i)^(?:chapter\s+\d+\s*:\s*)?recommendations?\s*$',
                r'(?i)^(?:chapter\s+\d+\s+)?(?:summary\s+of\s+)?recommendations?\s*$',
                r'(?i)^recommendations?\s+for\s+(?:the\s+)?government\s*$',
                r'(?i)^recommendations?\s+and\s+conclusions?\s*$',
                r'(?i)^our\s+recommendations?\s*$',
                r'(?i)^key\s+recommendations?\s*$',
                r'(?i)^main\s+recommendations?\s*$',
                r'(?i)^(?:final\s+)?recommendations?\s+to\s+government\s*$'
            ],
            'individual_items': [
                r'(?i)^recommendation\s+(\d+(?:\.\d+)*)\s*[:\.\-]?\s*(.+)$',
                r'(?i)^(\d+(?:\.\d+)*)\.\s+(.+)$',
                r'(?i)^(\d+(?:\.\d+)*)\s+(.+)$',
                r'(?i)^[•\-\*]\s*recommendation\s+(\d+(?:\.\d+)*)\s*[:\.\-]?\s*(.+)$',
                r'(?i)^that\s+(.+)$',
                r'(?i)^the\s+government\s+should\s+(.+)$',
                r'(?i)^we\s+recommend\s+that\s+(.+)$'
            ]
        }
        
        # Response section patterns
        self.response_patterns = {
            'section_headers': [
                r'(?i)^(?:government\s+)?responses?\s+to\s+recommendations?\s*$',
                r'(?i)^(?:the\s+)?government\s+responds?\s*$',
                r'(?i)^responses?\s+to\s+(?:the\s+)?inquiry\s*$',
                r'(?i)^(?:detailed\s+)?responses?\s*$',
                r'(?i)^government\s+response\s*$',
                r'(?i)^response\s+to\s+recommendation\s*',
                r'(?i)^our\s+response\s*$'
            ],
            'individual_items': [
                r'(?i)^response\s+to\s+recommendation\s+(\d+(?:\.\d+)*)\s*[:\.\-]?\s*(.+)$',
                r'(?i)^recommendation\s+(\d+(?:\.\d+)*)\s*[:\.\-]?\s*response\s*[:\.\-]?\s*(.+)$',
                r'(?i)^(\d+(?:\.\d+)*)\.\s*response\s*[:\.\-]?\s*(.+)$',
                r'(?i)^(\d+(?:\.\d+)*)\s*[:\.\-]\s*(.+)$',
                r'(?i)^accepted\s*[:\.\-]?\s*(.+)$',
                r'(?i)^not\s+accepted\s*[:\.\-]?\s*(.+)$',
                r'(?i)^partially\s+accepted\s*[:\.\-]?\s*(.+)$'
            ]
        }
        
        # Section boundary markers - when to stop extraction
        self.section_boundaries = [
            r'(?i)^(?:appendix|annex|bibliography|references|index|glossary)\s*$',
            r'(?i)^(?:next\s+steps?|way\s+forward|conclusion)\s*$',
            r'(?i)^(?:signed|dated\s+this|chair\s+of)\s*',
            r'(?i)^(?:minister\s+for|secretary\s+of\s+state)\s*$',
            r'(?i)^(?:contact\s+details?|further\s+information)\s*$',
            r'(?i)^(?:©\s*crown\s+copyright|published\s+by)\s*',
            r'(?i)^(?:isbn\s+|printed\s+in\s+|printed\s+on\s+paper)\s*',
        ]

    def extract_text_from_pdf(self, pdf_path: str, extract_sections_only: bool = True) -> Optional[Dict[str, Any]]:
        """
        Main extraction function with intelligent section processing.
        
        Args:
            pdf_path: Path to the PDF file
            extract_sections_only: Always True - uses intelligent extraction
            
        Returns:
            Dictionary containing extracted text, sections, and metadata
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting intelligent extraction for: {pdf_path}")
            
            # Step 1: Extract basic text from all pages
            text_result = self._extract_basic_text(pdf_path)
            if not text_result:
                self.logger.error(f"Failed to extract basic text from {pdf_path}")
                return self._create_error_result(pdf_path, "Failed to extract basic text")
            
            full_text = text_result['text']
            pages_data = text_result['pages']
            
            # Step 2: Always use intelligent section extraction
            sections = self._extract_sections_enhanced(pdf_path, full_text, pages_data)
            self.logger.info(f"Found {len(sections)} sections")
            
            # Step 3: Extract individual recommendations and responses
            recommendations = self._extract_individual_recommendations(full_text)
            responses = self._extract_individual_responses(full_text)
            
            # Step 4: Extract metadata
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            
            # Step 5: Analyze document structure
            document_analysis = self._analyze_government_document_structure(full_text)
            
            # Step 6: Create comprehensive result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'filename': Path(pdf_path).name,
                'text': full_text,
                'content': full_text,  # Alias for compatibility
                'sections': sections,
                'recommendations': recommendations,
                'responses': responses,
                'metadata': {
                    **metadata,
                    'processing_mode': 'intelligent_sections',
                    'processing_time_seconds': processing_time,
                    'extraction_method': 'enhanced_pattern_based',
                    'document_analysis': document_analysis,
                    'filename': Path(pdf_path).name,
                    'file_path': str(pdf_path)
                },
                'processed_at': datetime.now().isoformat(),
                'success': True,
                'extractor_version': 'Enhanced_v3.0',
                'statistics': {
                    'total_text_length': len(full_text),
                    'sections_found': len(sections),
                    'recommendations_found': len(recommendations),
                    'responses_found': len(responses),
                    'pages_processed': len(pages_data)
                }
            }
            
            # Update processing statistics
            self.processing_stats['documents_processed'] += 1
            self.processing_stats['total_pages_processed'] += len(pages_data)
            
            self.logger.info(f"Successfully processed {Path(pdf_path).name} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            self.processing_stats['extraction_errors'] += 1
            return self._create_error_result(pdf_path, str(e))

    def _create_error_result(self, pdf_path: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'filename': Path(pdf_path).name,
            'text': '',
            'content': '',
            'sections': [],
            'recommendations': [],
            'responses': [],
            'metadata': {
                'error': error_message,
                'filename': Path(pdf_path).name,
                'file_path': str(pdf_path)
            },
            'processed_at': datetime.now().isoformat(),
            'success': False,
            'error': error_message,
            'extractor_version': 'Enhanced_v3.0'
        }

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
        """Extract text using pdfplumber (preferred method)"""
        all_text = []
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text() or ""
                    all_text.append(page_text)
                    
                    pages_data.append({
                        'page_number': page_num,
                        'text': page_text,
                        'char_count': len(page_text),
                        'bbox': page.bbox
                    })
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num}: {e}")
                    pages_data.append({
                        'page_number': page_num,
                        'text': "",
                        'char_count': 0,
                        'error': str(e)
                    })
        
        return {
            'text': '\n'.join(all_text),
            'pages': pages_data,
            'total_pages': len(pages_data),
            'extraction_method': 'pdfplumber'
        }

    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF as fallback"""
        all_text = []
        pages_data = []
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                all_text.append(page_text)
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text),
                    'rect': page.rect
                })
            except Exception as e:
                self.logger.warning(f"Error extracting page {page_num + 1}: {e}")
                pages_data.append({
                    'page_number': page_num + 1,
                    'text': "",
                    'char_count': 0,
                    'error': str(e)
                })
        
        doc.close()
        
        return {
            'text': '\n'.join(all_text),
            'pages': pages_data,
            'total_pages': len(pages_data),
            'extraction_method': 'pymupdf'
        }

    def _extract_sections_enhanced(self, pdf_path: str, full_text: str, pages_data: List[Dict]) -> List[Dict[str, Any]]:
        """Enhanced section extraction for government documents"""
        sections = []
        lines = full_text.split('\n')
        
        current_section = None
        current_content = []
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for section headers
            section_type = self._identify_section_type(line_stripped)
            
            if section_type:
                # Save previous section
                if current_section and current_content:
                    current_section['content'] = '\n'.join(current_content).strip()
                    current_section['word_count'] = len(current_section['content'].split())
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'type': section_type,
                    'title': line_stripped,
                    'start_line': line_num,
                    'page_start': self._get_page_for_line(line_num, pages_data),
                    'content': '',
                    'word_count': 0
                }
                current_content = []
                
            elif current_section:
                # Check for section boundaries
                if self._is_section_boundary(line_stripped):
                    # End current section
                    if current_content:
                        current_section['content'] = '\n'.join(current_content).strip()
                        current_section['word_count'] = len(current_section['content'].split())
                        current_section['end_line'] = line_num
                        current_section['page_end'] = self._get_page_for_line(line_num, pages_data)
                        sections.append(current_section)
                    
                    current_section = None
                    current_content = []
                else:
                    # Add to current section
                    current_content.append(line)
        
        # Don't forget the last section
        if current_section and current_content:
            current_section['content'] = '\n'.join(current_content).strip()
            current_section['word_count'] = len(current_section['content'].split())
            current_section['end_line'] = len(lines)
            current_section['page_end'] = len(pages_data)
            sections.append(current_section)
        
        return sections

    def _identify_section_type(self, line: str) -> Optional[str]:
        """Identify the type of section based on header patterns"""
        line_clean = line.strip()
        
        # Check recommendation patterns
        for pattern in self.recommendation_patterns['section_headers']:
            if re.match(pattern, line_clean):
                return 'recommendations'
        
        # Check response patterns
        for pattern in self.response_patterns['section_headers']:
            if re.match(pattern, line_clean):
                return 'responses'
        
        return None

    def _is_section_boundary(self, line: str) -> bool:
        """Check if line indicates a section boundary"""
        for pattern in self.section_boundaries:
            if re.match(pattern, line):
                return True
        return False

    def _get_page_for_line(self, line_num: int, pages_data: List[Dict]) -> int:
        """Estimate page number for a given line"""
        if not pages_data:
            return 1
        
        total_lines = sum(page.get('text', '').count('\n') + 1 for page in pages_data)
        if total_lines == 0:
            return 1
        
        lines_per_page = total_lines / len(pages_data)
        estimated_page = int(line_num / lines_per_page) + 1
        
        return min(estimated_page, len(pages_data))

    def _extract_individual_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual numbered recommendations"""
        recommendations = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            for pattern in self.recommendation_patterns['individual_items']:
                match = re.match(pattern, line_stripped)
                if match:
                    if len(match.groups()) >= 2:
                        rec_number = match.group(1)
                        rec_text = match.group(2)
                    else:
                        rec_number = str(len(recommendations) + 1)
                        rec_text = match.group(1)
                    
                    recommendations.append({
                        'number': rec_number,
                        'text': rec_text.strip(),
                        'line_number': line_num,
                        'confidence': 0.8,
                        'extraction_method': 'pattern_based'
                    })
                    break
        
        return recommendations

    def _extract_individual_responses(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual numbered responses"""
        responses = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            for pattern in self.response_patterns['individual_items']:
                match = re.match(pattern, line_stripped)
                if match:
                    if len(match.groups()) >= 2:
                        resp_number = match.group(1)
                        resp_text = match.group(2)
                    else:
                        resp_number = str(len(responses) + 1)
                        resp_text = match.group(1)
                    
                    responses.append({
                        'number': resp_number,
                        'text': resp_text.strip(),
                        'line_number': line_num,
                        'confidence': 0.8,
                        'extraction_method': 'pattern_based'
                    })
                    break
        
        return responses

    def _extract_metadata_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF if available"""
        metadata = {
            'total_pages': 0,
            'file_size_mb': 0,
            'title': '',
            'author': '',
            'subject': '',
            'creator': '',
            'producer': '',
            'creation_date': '',
            'modification_date': ''
        }
        
        try:
            # File size
            file_size = Path(pdf_path).stat().st_size
            metadata['file_size_mb'] = file_size / (1024 * 1024)
            
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(pdf_path)
                metadata['total_pages'] = len(doc)
                
                # Document metadata
                doc_metadata = doc.metadata
                metadata.update({
                    'title': doc_metadata.get('title', ''),
                    'author': doc_metadata.get('author', ''),
                    'subject': doc_metadata.get('subject', ''),
                    'creator': doc_metadata.get('creator', ''),
                    'producer': doc_metadata.get('producer', ''),
                    'creation_date': doc_metadata.get('creationDate', ''),
                    'modification_date': doc_metadata.get('modDate', '')
                })
                
                doc.close()
        
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {e}")
        
        return metadata

    def _analyze_government_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure for government documents"""
        analysis = {
            'document_type': 'unknown',
            'has_table_of_contents': False,
            'has_numbered_sections': False,
            'recommendation_mentions': 0,
            'response_mentions': 0,
            'estimated_recommendations': 0,
            'estimated_responses': 0
        }
        
        text_lower = text.lower()
        
        # Count mentions
        analysis['recommendation_mentions'] = len(re.findall(r'\brecommendation\b', text_lower))
        analysis['response_mentions'] = len(re.findall(r'\bresponse\b', text_lower))
        
        # Estimate counts
        analysis['estimated_recommendations'] = len(re.findall(r'recommendation\s+\d+', text_lower))
        analysis['estimated_responses'] = len(re.findall(r'response\s+to\s+recommendation\s+\d+', text_lower))
        
        # Document type inference
        if analysis['recommendation_mentions'] > 10 and analysis['response_mentions'] > 10:
            analysis['document_type'] = 'combined_report'
        elif analysis['recommendation_mentions'] > 5:
            analysis['document_type'] = 'inquiry_report'
        elif analysis['response_mentions'] > 5:
            analysis['document_type'] = 'government_response'
        
        # Structure checks
        analysis['has_table_of_contents'] = bool(re.search(r'(?i)\btable\s+of\s+contents?\b', text))
        analysis['has_numbered_sections'] = bool(re.search(r'\d+\.\d+\s+[A-Z]', text))
        
        return analysis

    # ===============================================
    # PUBLIC UTILITY METHODS
    # ===============================================

    def get_document_summary(self, pdf_path: str) -> Dict[str, Any]:
        """Get a quick summary of document without full processing"""
        try:
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            
            # Quick text sample
            text_result = self._extract_basic_text(pdf_path)
            if text_result:
                text_sample = text_result['text'][:1000]
                structure = self._analyze_government_document_structure(text_sample)
                
                return {
                    'filename': Path(pdf_path).name,
                    'file_size_mb': metadata['file_size_mb'],
                    'total_pages': metadata['total_pages'],
                    'document_type': structure['document_type'],
                    'has_recommendations': structure['recommendation_mentions'] > 0,
                    'has_responses': structure['response_mentions'] > 0,
                    'summary_generated_at': datetime.now().isoformat()
                }
        
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
        
        return {
            'filename': Path(pdf_path).name,
            'error': 'Could not generate summary',
            'summary_generated_at': datetime.now().isoformat()
        }

    def validate_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """Validate extraction quality"""
        try:
            result = self.extract_text_from_pdf(pdf_path)
            
            if not result or not result.get('success'):
                return {
                    'is_valid': False,
                    'quality_score': 0.0,
                    'issues': ['Extraction failed'],
                    'validation_timestamp': datetime.now().isoformat()
                }
            
            issues = []
            quality_score = 1.0
            
            # Check text length
            text_length = len(result.get('text', ''))
            if text_length < 100:
                issues.append('Text too short')
                quality_score -= 0.3
            
            # Check for sections
            sections = result.get('sections', [])
            if not sections:
                issues.append('No sections found')
                quality_score -= 0.2
            
            # Check for recommendations or responses
            recommendations = result.get('recommendations', [])
            responses = result.get('responses', [])
            
            if not recommendations and not responses:
                issues.append('No recommendations or responses found')
                quality_score -= 0.3
            
            return {
                'is_valid': quality_score > 0.5,
                'quality_score': max(0.0, quality_score),
                'issues': issues,
                'text_length': text_length,
                'sections_found': len(sections),
                'recommendations_found': len(recommendations),
                'responses_found': len(responses),
                'validation_timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': [f'Validation error: {str(e)}'],
                'validation_timestamp': datetime.now().isoformat()
            }

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            'statistics_timestamp': datetime.now().isoformat()
        }

# ===============================================
# CONVENIENCE FUNCTIONS
# ===============================================

def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Convenience function for simple extraction"""
    processor = DocumentProcessor()
    return processor.extract_text_from_pdf(pdf_path)

def get_document_summary(pdf_path: str) -> Dict[str, Any]:
    """Get document summary"""
    processor = DocumentProcessor()
    return processor.get_document_summary(pdf_path)

def validate_document_extraction(pdf_path: str) -> bool:
    """Quick validation check - returns True if extraction was successful"""
    processor = DocumentProcessor()
    validation = processor.validate_extraction(pdf_path)
    return validation.get('is_valid', False) and validation.get('quality_score', 0) > 0.5

def extract_recommendations_and_responses_only(pdf_path: str) -> Dict[str, Any]:
    """Convenience function - always uses intelligent extraction"""
    processor = DocumentProcessor()
    return processor.extract_text_from_pdf(pdf_path)

def detect_government_document_type(pdf_path: str) -> Dict[str, Any]:
    """Detect government document type"""
    processor = DocumentProcessor()
    summary = processor.get_document_summary(pdf_path)
    return {
        'document_type': summary.get('document_type', 'unknown'),
        'filename': summary.get('filename', 'unknown'),
        'has_recommendations': summary.get('has_recommendations', False),
        'has_responses': summary.get('has_responses', False)
    }

# ===============================================
# TESTING AND DEBUG FUNCTIONS
# ===============================================

def test_extraction(pdf_path: str, verbose: bool = True):
    """Test extraction with a PDF file"""
    processor = DocumentProcessor(debug_mode=verbose)
    
    if verbose:
        print(f"Testing intelligent extraction for: {pdf_path}")
        print("=" * 50)
    
    # Get summary
    summary = processor.get_document_summary(pdf_path)
    if verbose:
        print("Document Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # Try extraction
    result = processor.extract_text_from_pdf(pdf_path)
    if result and result.get('success'):
        text = result.get('text', '')
        sections = result.get('sections', [])
        recommendations = result.get('recommendations', [])
        responses = result.get('responses', [])
        
        if verbose:
            print(f"\nExtraction successful!")
            print(f"Text length: {len(text):,} characters")
            print(f"Sections found: {len(sections)}")
            print(f"Individual recommendations: {len(recommendations)}")
            print(f"Individual responses: {len(responses)}")
            
            for section in sections:
                print(f"  - {section['type']}: {section['title']} ({len(section['content'])} chars)")
        
        return True
    else:
        if verbose:
            print("\nExtraction failed!")
            if result and 'error' in result:
                print(f"Error: {result['error']}")
        return False

def debug_document_processing(pdf_path: str):
    """Debug document processing step by step"""
    processor = DocumentProcessor(debug_mode=True)
    
    print(f"DEBUG: Processing {pdf_path}")
    print("=" * 60)
    
    # Step 1: Basic text extraction
    print("Step 1: Basic text extraction...")
    basic_result = processor._extract_basic_text(pdf_path)
    if basic_result:
        print(f"  ✅ Extracted {len(basic_result['text']):,} characters from {basic_result['total_pages']} pages")
        
        # Show first 300 characters
        preview = basic_result['text'][:300].replace('\n', ' ')
        print(f"  📄 Preview: {preview}...")
    else:
        print("  ❌ Basic text extraction failed")
        return
    
    # Step 2: Document analysis
    print("\nStep 2: Document structure analysis...")
    analysis = processor._analyze_government_document_structure(basic_result['text'])
    print(f"  📊 Document type: {analysis['document_type']}")
    print(f"  📋 Has table of contents: {analysis['has_table_of_contents']}")
    print(f"  🔢 Has numbered sections: {analysis['has_numbered_sections']}")
    print(f"  📝 Recommendation mentions: {analysis.get('recommendation_mentions', 0)}")
    print(f"  📋 Response mentions: {analysis.get('response_mentions', 0)}")
    
    # Step 3: Full extraction test
    print("\nStep 3: Full extraction test...")
    result = processor.extract_text_from_pdf(pdf_path)
    if result and result.get('success'):
        sections = result.get('sections', [])
        recommendations = result.get('recommendations', [])
        responses = result.get('responses', [])
        
        print(f"  ✅ Extraction successful!")
        print(f"  📋 Sections: {len(sections)}")
        print(f"  📝 Recommendations: {len(recommendations)}")
        print(f"  📋 Responses: {len(responses)}")
    else:
        print(f"  ❌ Full extraction failed")
        if result and 'error' in result:
            print(f"  Error: {result['error']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if '--debug' in sys.argv:
            debug_document_processing(sys.argv[1])
        else:
            test_extraction(sys.argv[1])
    else:
        print("Usage: python document_processor.py <pdf_file_path> [--debug]")
        print("Example: python document_processor.py 'Government_Response_to_IBI.pdf'")
        print("Debug:   python document_processor.py 'Government_Response_to_IBI.pdf' --debug")
