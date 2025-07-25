# ===============================================
# FILE: modules/document_processor.py (COMPLETE ENHANCED VERSION)
# ===============================================

import logging
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pdfplumber
import fitz  # PyMuPDF

# Set up logging
logging.basicConfig(level=logging.INFO)

class DocumentProcessor:
    """
    Enhanced Document processor with comprehensive support for government inquiry documents.
    
    Key features:
    - Enhanced section detection for government documents
    - Better pattern matching for table-of-contents based documents
    - Improved handling of numbered recommendations and responses
    - Fixed variable scoping issues
    - Support for Cabinet Office, inquiry reports, and government response documents
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced patterns for government document section detection
        self.government_section_patterns = {
            'recommendation_sections': [
                # Standard recommendation sections in inquiry reports
                r'(?i)^(?:\d+\.?\d*\s*)?(?:recommendations?|conclusions and recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:summary of recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:key recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:main recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:principal recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:findings and recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:inquiry recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:final recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:our recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:recommendations for (?:action|change|improvement))\s*$',
                
                # Table of contents style entries
                r'(?i)^(?:\d+\.\d+\s+)?(?:overview and )?recommendations?\s*$',
                r'(?i)^(?:\d+\.\d+\s+)?(?:list of )?recommendations?\s*$',
            ],
            'response_sections': [
                # Government response sections
                r'(?i)^(?:\d+\.?\d*\s*)?(?:government response)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:official response)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:response to (?:the )?(?:inquiry )?recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:departmental response)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:ministry response)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:cabinet office response)\s*$',
                
                # Implementation and action sections
                r'(?i)^(?:\d+\.?\d*\s*)?(?:implementation (?:plan|strategy))\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:action plan)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:government action)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:next steps?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:progress (?:report|update))\s*$',
                
                # Response to specific recommendations
                r'(?i)^(?:\d+\.?\d*\s*)?(?:response to recommendation\s+\d+)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:recommendation\s+\d+\s*[-:]?\s*response)\s*$',
                
                # SPECIFIC PATTERNS for documents like Infected Blood Inquiry response
                r'(?i)^(?:\d+\.?\d*\s*)?(?:responses? to individual recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:detailed responses?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:government responses? to recommendations?)\s*$',
                r'(?i)^(?:\d+\.?\d*\s*)?(?:recommendation\s+\d+)\s*$',  # Just "Recommendation 1", "Recommendation 2" etc.
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
        Main extraction function with enhanced government document support
        
        Args:
            pdf_path: Path to the PDF file
            extract_sections_only: If True, extract only recommendation/response sections
            
        Returns:
            Dictionary containing extracted text, sections, and metadata
        """
        try:
            self.logger.info(f"Starting extraction for: {pdf_path}")
            
            # Step 1: Extract basic text from all pages
            text_result = self._extract_basic_text(pdf_path)
            if not text_result:
                self.logger.error(f"Failed to extract basic text from {pdf_path}")
                return {
                'filename': Path(pdf_path).name,
                'file_size_mb': metadata.get('file_size_mb', 0),
                'total_pages': metadata.get('page_count', 0),
                'is_valid': validation.get('is_valid', False),
                'quality_score': validation.get('quality_score', 0),
                'text_length': validation.get('text_length', 0),
                'issues': validation.get('issues', []),
                'recommendations': validation.get('recommendations', []),
                'document_indicators': validation.get('document_indicators', {}),
                **sections_info
            }
            
        except Exception as summary_error:
            self.logger.error(f"Error getting document summary: {summary_error}")
            return {
                'filename': Path(pdf_path).name,
                'error': str(summary_error)
            }

    def extract_sections_with_details(self, pdf_path: str) -> Dict[str, Any]:
        """Extract sections with detailed analysis"""
        try:
            # Get full extraction
            result = self.extract_text_from_pdf(pdf_path, extract_sections_only=True)
            
            if not result or not result.get('success'):
                return {
                    'sections': [],
                    'summary': {'error': 'Extraction failed'},
                    'metadata': {'error': 'Could not process document'}
                }
            
            sections = result.get('sections', [])
            
            # Add additional details to each section
            detailed_sections = []
            for section in sections:
                detailed_section = section.copy()
                
                # Add content analysis
                content = section.get('content', '')
                detailed_section.update({
                    'content_preview': content[:200] + '...' if len(content) > 200 else content,
                    'has_numbered_items': bool(re.search(r'^\s*\d+\.', content, re.MULTILINE)),
                    'has_bullet_points': bool(re.search(r'[•·‣⁃]', content)),
                    'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
                    'sentence_count': len([s for s in re.split(r'[.!?]+', content) if s.strip()])
                })
                
                detailed_sections.append(detailed_section)
            
            return {
                'sections': detailed_sections,
                'summary': {
                    'total_sections': len(detailed_sections),
                    'recommendation_sections': len([s for s in detailed_sections if s['type'] == 'recommendation']),
                    'response_sections': len([s for s in detailed_sections if s['type'] == 'response']),
                    'total_content_length': sum(len(s.get('content', '')) for s in detailed_sections),
                    'average_section_length': sum(len(s.get('content', '')) for s in detailed_sections) / len(detailed_sections) if detailed_sections else 0
                },
                'metadata': result.get('metadata', {}),
                'document_analysis': result.get('document_analysis', {}),
                'recommendations': result.get('recommendations', []),
                'responses': result.get('responses', [])
            }
            
        except Exception as details_error:
            self.logger.error(f"Error extracting sections with details: {details_error}")
            return {
                'sections': [],
                'summary': {'error': str(details_error)},
                'metadata': {'error': str(details_error)}
            }

    def detect_document_type(self, pdf_path: str) -> Dict[str, Any]:
        """Detect the type of government document"""
        try:
            basic_result = self._extract_basic_text(pdf_path)
            if not basic_result:
                return {'type': 'unknown', 'confidence': 0.0, 'error': 'Failed to extract text'}
            
            text = basic_result['text']
            analysis = self._analyze_government_document_structure(text)
            
            return {
                'type': analysis['document_type'],
                'inquiry_type': analysis.get('potential_inquiry_type', 'unknown'),
                'has_recommendations': analysis.get('recommendation_mentions', 0) > 0,
                'has_responses': analysis.get('response_mentions', 0) > 0,
                'structure_type': 'table_of_contents' if analysis['has_table_of_contents'] else 'standard',
                'confidence': self._calculate_document_type_confidence(analysis),
                'analysis': analysis
            }
            
        except Exception as detection_error:
            self.logger.error(f"Error detecting document type: {detection_error}")
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'error': str(detection_error)
            }

    def _calculate_document_type_confidence(self, analysis: Dict) -> float:
        """Calculate confidence in document type detection"""
        confidence = 0.0
        
        # Base confidence from document type detection
        if analysis['document_type'] != 'unknown':
            confidence += 0.4
        
        # Bonus for specific inquiry type
        if analysis.get('potential_inquiry_type') != 'unknown':
            confidence += 0.2
        
        # Bonus for structure indicators
        if analysis['has_table_of_contents']:
            confidence += 0.2
        if analysis['has_numbered_sections']:
            confidence += 0.1
        
        # Bonus for content indicators
        if analysis.get('recommendation_mentions', 0) > 0:
            confidence += 0.1
        if analysis.get('response_mentions', 0) > 0:
            confidence += 0.1
        
        return min(1.0, confidence)


# ===============================================
# UTILITY FUNCTIONS FOR BACKWARD COMPATIBILITY
# ===============================================

def extract_text_from_pdf(pdf_path: str, extract_sections_only: bool = True) -> Optional[Dict[str, Any]]:
    """Standalone function for text extraction - maintains backward compatibility"""
    processor = DocumentProcessor()
    return processor.extract_text_from_pdf(pdf_path, extract_sections_only)

def validate_pdf_file(pdf_path: str) -> bool:
    """Quick validation check"""
    processor = DocumentProcessor()
    validation = processor.validate_extraction(pdf_path)
    return validation.get('is_valid', False)

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
    """Convenience function to extract only recommendations and responses sections"""
    processor = DocumentProcessor()
    return processor.extract_text_from_pdf(pdf_path, extract_sections_only=True)

def get_sections_with_page_numbers(pdf_path: str) -> List[Dict[str, Any]]:
    """Get a list of extracted sections with page numbers"""
    processor = DocumentProcessor()
    result = processor.extract_sections_with_details(pdf_path)
    
    sections_with_pages = []
    for section in result.get('sections', []):
        sections_with_pages.append({
            'type': section['type'],
            'title': section['title'],
            'content': section['content'],
            'page_start': section['page_start'],
            'page_end': section['page_end'],
            'page_range': f"{section['page_start']}-{section['page_end']}",
            'word_count': section.get('content_stats', {}).get('word_count', 0)
        })
    
    return sections_with_pages

def detect_government_document_type(pdf_path: str) -> Dict[str, Any]:
    """Detect government document type"""
    processor = DocumentProcessor()
    return processor.detect_document_type(pdf_path)

# ===============================================
# ENHANCED EXTRACTION FUNCTIONS
# ===============================================

def extract_government_recommendations(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract individual recommendations from government documents"""
    processor = DocumentProcessor()
    result = processor.extract_text_from_pdf(pdf_path, extract_sections_only=True)
    
    if result and result.get('success'):
        return result.get('recommendations', [])
    return []

def extract_government_responses(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract individual responses from government documents"""
    processor = DocumentProcessor()
    result = processor.extract_text_from_pdf(pdf_path, extract_sections_only=True)
    
    if result and result.get('success'):
        return result.get('responses', [])
    return []

def get_comprehensive_document_analysis(pdf_path: str) -> Dict[str, Any]:
    """Get comprehensive analysis of a government document"""
    processor = DocumentProcessor()
    
    # Get full extraction
    extraction_result = processor.extract_text_from_pdf(pdf_path, extract_sections_only=True)
    
    # Get document type
    document_type = processor.detect_document_type(pdf_path)
    
    # Get summary
    summary = processor.get_document_summary(pdf_path)
    
    return {
        'extraction_result': extraction_result,
        'document_type': document_type,
        'summary': summary,
        'analysis_timestamp': datetime.now().isoformat(),
        'success': extraction_result.get('success', False) if extraction_result else False
    }

# ===============================================
# TESTING AND DEBUG FUNCTIONS
# ===============================================

def test_extraction(pdf_path: str, verbose: bool = True):
    """Test extraction with a PDF file"""
    processor = DocumentProcessor()
    
    if verbose:
        print(f"Testing extraction for: {pdf_path}")
        print("=" * 50)
    
    # Get summary
    summary = processor.get_document_summary(pdf_path)
    if verbose:
        print("Document Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # Try extraction
    result = processor.extract_text_from_pdf(pdf_path, extract_sections_only=True)
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
    processor = DocumentProcessor()
    
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
    
    # Step 3: Section detection
    print("\nStep 3: Section header detection...")
    lines = basic_result['text'].split('\n')
    section_headers = processor._find_section_headers(lines)
    print(f"  ✅ Found {len(section_headers)} section headers")
    
    for line_num, section_type, title in section_headers:
        print(f"    Line {line_num}: {section_type} - {title}")
    
    # Step 4: Full extraction test
    print("\nStep 4: Full extraction test...")
    result = processor.extract_text_from_pdf(pdf_path, extract_sections_only=True)
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
        print("Debug:   python document_processor.py 'Government_Response_to_IBI.pdf' --debug") None
            
            full_text = text_result['text']
            pages_data = text_result['pages']
            
            # Step 2: Extract sections if requested
            sections = []
            if extract_sections_only:
                sections = self._extract_sections_enhanced(pdf_path, full_text, pages_data)
                self.logger.info(f"Found {len(sections)} sections")
            
            # Step 3: Extract metadata
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            
            # Step 4: Analyze document structure for government documents
            document_analysis = self._analyze_government_document_structure(full_text)
            
            # Step 5: Extract individual recommendations and responses if sections found
            recommendations = []
            responses = []
            if sections:
                recommendations = self._extract_individual_recommendations(sections)
                responses = self._extract_individual_responses(sections)
            
            # Build comprehensive result
            result = {
                'filename': Path(pdf_path).name,
                'text': full_text,
                'content': full_text,  # For backward compatibility
                'sections': sections,
                'recommendations': recommendations,
                'responses': responses,
                'metadata': metadata,
                'document_analysis': document_analysis,
                'pages_data': pages_data,
                'processed_at': datetime.now().isoformat(),
                'extraction_mode': 'sections_only' if extract_sections_only else 'full_document',
                'success': True,
                'extractor_version': 'Enhanced_v3.0'
            }
            
            self.logger.info(f"Extraction completed successfully for {pdf_path}")
            return result
            
        except Exception as extraction_error:
            self.logger.error(f"Error extracting text from {pdf_path}: {extraction_error}", exc_info=True)
            return {
                'filename': Path(pdf_path).name,
                'text': '',
                'content': '',
                'sections': [],
                'recommendations': [],
                'responses': [],
                'metadata': {'error': str(extraction_error)},
                'processed_at': datetime.now().isoformat(),
                'success': False,
                'error': str(extraction_error),
                'extractor_version': 'Enhanced_v3.0'
            }

    def _extract_basic_text(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Extract basic text from PDF using pdfplumber"""
        try:
            full_text_parts = []
            pages_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = self._clean_text(page_text)
                            full_text_parts.append(cleaned_text)
                            pages_data.append({
                                'page_number': page_num,
                                'text': cleaned_text,
                                'raw_text': page_text,
                                'char_count': len(cleaned_text)
                            })
                    except Exception as page_error:
                        self.logger.warning(f"Error extracting page {page_num}: {page_error}")
                        continue
            
            full_text = '\n'.join(full_text_parts)
            
            if not full_text.strip():
                self.logger.warning("No text extracted from PDF")
                return None
            
            return {
                'text': full_text,
                'pages': pages_data,
                'total_pages': len(pages_data),
                'total_chars': len(full_text)
            }
            
        except Exception as text_error:
            self.logger.error(f"Error in basic text extraction: {text_error}")
            return None

    def _extract_sections_enhanced(self, pdf_path: str, full_text: str, pages_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract sections using enhanced government document patterns"""
        sections = []
        
        try:
            # Split text into lines for processing
            lines = full_text.split('\n')
            
            # Find section headers
            section_headers = self._find_section_headers(lines)
            
            if not section_headers:
                self.logger.info("No section headers found using government patterns")
                return sections
            
            # Extract content for each section
            for i, (line_num, section_type, title) in enumerate(section_headers):
                # Determine end of section
                if i + 1 < len(section_headers):
                    end_line = section_headers[i + 1][0]
                else:
                    end_line = len(lines)
                
                # Extract section content
                section_content = self._extract_section_content(lines, line_num, end_line)
                
                if len(section_content.strip()) > 100:  # Only include substantial sections
                    # Calculate page range
                    page_start, page_end = self._calculate_page_range(line_num, end_line, pages_data)
                    
                    # Calculate content statistics
                    content_stats = self._calculate_content_stats(section_content)
                    
                    section = {
                        'type': section_type,
                        'title': title,
                        'content': section_content,
                        'page_start': page_start,
                        'page_end': page_end,
                        'line_start': line_num,
                        'line_end': end_line,
                        'content_stats': content_stats,
                        'extraction_method': 'government_patterns',
                        'confidence': self._calculate_section_confidence(section_content, section_type)
                    }
                    
                    sections.append(section)
                    self.logger.info(f"Extracted {section_type} section: {title}")
            
        except Exception as section_error:
            self.logger.error(f"Error extracting sections: {section_error}")
        
        return sections

    def _find_section_headers(self, lines: List[str]) -> List[Tuple[int, str, str]]:
        """Find section headers in the document"""
        section_headers = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 3:
                continue
            
            # Check against recommendation patterns
            for pattern in self.government_section_patterns['recommendation_sections']:
                if re.match(pattern, line_stripped):
                    section_headers.append((i, 'recommendation', line_stripped))
                    self.logger.debug(f"Found recommendation section at line {i}: {line_stripped}")
                    break
            
            # Check against response patterns
            for pattern in self.government_section_patterns['response_sections']:
                if re.match(pattern, line_stripped):
                    section_headers.append((i, 'response', line_stripped))
                    self.logger.debug(f"Found response section at line {i}: {line_stripped}")
                    break
        
        # Sort by line number
        section_headers.sort(key=lambda x: x[0])
        
        return section_headers

    def _extract_section_content(self, lines: List[str], start_line: int, end_line: int) -> str:
        """Extract content between start and end lines"""
        content_lines = []
        
        # Start from the line after the header
        for i in range(start_line + 1, min(end_line, len(lines))):
            line = lines[i].strip()
            
            # Check if we've hit a section boundary
            if self._is_section_boundary(line):
                break
            
            content_lines.append(lines[i])
        
        return '\n'.join(content_lines).strip()

    def _is_section_boundary(self, line: str) -> bool:
        """Check if line marks end of a section"""
        for pattern in self.section_boundaries:
            if re.match(pattern, line):
                return True
        return False

    def _calculate_page_range(self, start_line: int, end_line: int, pages_data: List[Dict]) -> Tuple[int, int]:
        """Calculate page range for a section based on line numbers"""
        if not pages_data:
            return 1, 1
        
        # Simple estimation based on character positions
        total_chars = sum(page['char_count'] for page in pages_data)
        avg_chars_per_page = total_chars / len(pages_data) if pages_data else 1
        
        # Estimate page based on line position
        estimated_start_page = max(1, int((start_line / (end_line or 1)) * len(pages_data)))
        estimated_end_page = max(estimated_start_page, min(len(pages_data), estimated_start_page + 1))
        
        return estimated_start_page, estimated_end_page

    def _calculate_content_stats(self, content: str) -> Dict[str, Any]:
        """Calculate statistics for section content"""
        if not content:
            return {'word_count': 0, 'char_count': 0, 'line_count': 0}
        
        words = content.split()
        lines = content.split('\n')
        
        # Count numbered items
        numbered_items = len(re.findall(r'^\s*\d+\.\s', content, re.MULTILINE))
        lettered_items = len(re.findall(r'^\s*[a-z]\.\s', content, re.MULTILINE))
        bullet_points = len(re.findall(r'[•·‣⁃]', content))
        
        return {
            'word_count': len(words),
            'char_count': len(content),
            'line_count': len(lines),
            'numbered_items': numbered_items,
            'lettered_items': lettered_items,
            'bullet_points': bullet_points,
            'avg_words_per_line': len(words) / len(lines) if lines else 0,
            'has_structured_content': numbered_items > 0 or lettered_items > 0 or bullet_points > 0
        }

    def _calculate_section_confidence(self, content: str, section_type: str) -> float:
        """Calculate confidence score for section extraction"""
        confidence = 0.5  # Base confidence
        
        # Content length bonus
        if len(content) > 1000:
            confidence += 0.3
        elif len(content) > 500:
            confidence += 0.2
        elif len(content) > 200:
            confidence += 0.1
        
        # Section-specific indicators
        if section_type == 'recommendation':
            if re.search(r'(?i)\b(?:recommend|should|must|ought|propose)\b', content):
                confidence += 0.2
            if re.search(r'^\s*\d+[\.\)]\s', content, re.MULTILINE):
                confidence += 0.1
        
        elif section_type == 'response':
            if re.search(r'(?i)\b(?:accept|reject|agree|disagree|implement|action)\b', content):
                confidence += 0.2
            if re.search(r'(?i)\b(?:government|department|ministry|cabinet)\b', content):
                confidence += 0.1
        
        return min(1.0, confidence)

    def _analyze_government_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure specifically for government documents"""
        analysis = {
            'document_type': 'unknown',
            'has_table_of_contents': False,
            'has_numbered_sections': False,
            'potential_inquiry_type': 'unknown',
            'structure_indicators': []
        }
        
        text_lower = text.lower()
        
        # Detect document type
        if 'government response to' in text_lower and 'inquiry' in text_lower:
            analysis['document_type'] = 'government_response'
        elif 'overview and recommendations' in text_lower:
            analysis['document_type'] = 'inquiry_report'
        elif 'infected blood inquiry' in text_lower:
            analysis['potential_inquiry_type'] = 'infected_blood'
        
        # Check for table of contents
        if re.search(r'(?i)(?:table\s+of\s+)?contents?|list\s+of\s+chapters?', text):
            analysis['has_table_of_contents'] = True
            analysis['structure_indicators'].append('table_of_contents')
        
        # Check for numbered sections
        if len(re.findall(r'^\s*\d+\.\d+\s+[A-Z]', text, re.MULTILINE)) >= 3:
            analysis['has_numbered_sections'] = True
            analysis['structure_indicators'].append('numbered_sections')
        
        # Count potential recommendations and responses
        rec_mentions = len(re.findall(r'(?i)recommendation', text))
        resp_mentions = len(re.findall(r'(?i)response', text))
        
        analysis['recommendation_mentions'] = rec_mentions
        analysis['response_mentions'] = resp_mentions
        
        return analysis

    def _extract_individual_recommendations(self, sections: List[Dict]) -> List[Dict[str, Any]]:
        """Extract individual numbered recommendations from sections"""
        recommendations = []
        
        for section in sections:
            if section['type'] == 'recommendation':
                content = section['content']
                
                # Look for numbered recommendations
                numbered_patterns = [
                    r'(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.|$)',
                    r'(?:^|\n)\s*([a-z])\.\s+(.+?)(?=\n\s*[a-z]\.|$)',
                    r'(?:^|\n)\s*\((\d+)\)\s+(.+?)(?=\n\s*\(\d+\)|$)'
                ]
                
                for pattern in numbered_patterns:
                    matches = re.finditer(pattern, content, re.DOTALL)
                    for match in matches:
                        rec_num = match.group(1)
                        rec_text = match.group(2).strip()
                        
                        if len(rec_text) > 50:  # Only substantial recommendations
                            recommendations.append({
                                'id': f"rec_{rec_num}",
                                'number': rec_num,
                                'text': rec_text,
                                'source_section': section['title'],
                                'page_start': section['page_start'],
                                'page_end': section['page_end'],
                                'extraction_method': 'numbered_pattern',
                                'confidence': 0.9
                            })
                
                # If no numbered recommendations, treat whole section as one
                if not any(rec['source_section'] == section['title'] for rec in recommendations):
                    if len(content) > 100:
                        recommendations.append({
                            'id': f"rec_section_{len(recommendations) + 1}",
                            'number': str(len(recommendations) + 1),
                            'text': content,
                            'source_section': section['title'],
                            'page_start': section['page_start'],
                            'page_end': section['page_end'],
                            'extraction_method': 'full_section',
                            'confidence': 0.7
                        })
        
        return recommendations

    def _extract_individual_responses(self, sections: List[Dict]) -> List[Dict[str, Any]]:
        """Extract individual government responses from sections"""
        responses = []
        
        for section in sections:
            if section['type'] == 'response':
                content = section['content']
                
                # Look for responses to specific recommendations
                response_patterns = [
                    r'(?:^|\n)\s*(?:Recommendation\s+)?(\d+)[:\.\)]\s*(.+?)(?=\n\s*(?:Recommendation\s+)?\d+[:\.\)]|$)',
                    r'(?:^|\n)\s*(?:Response\s+to\s+)?Recommendation\s+(\d+)[:\.]?\s*(.+?)(?=\n\s*(?:Response\s+to\s+)?Recommendation\s+\d+|$)'
                ]
                
                for pattern in response_patterns:
                    matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                    for match in matches:
                        rec_num = match.group(1)
                        response_text = match.group(2).strip()
                        
                        if len(response_text) > 30:  # Only substantial responses
                            response_type = self._classify_response_type(response_text)
                            
                            responses.append({
                                'id': f"resp_{rec_num}",
                                'recommendation_number': rec_num,
                                'response_text': response_text,
                                'response_type': response_type,
                                'source_section': section['title'],
                                'page_start': section['page_start'],
                                'page_end': section['page_end'],
                                'extraction_method': 'numbered_response',
                                'confidence': 0.9
                            })
        
        return responses

    def _classify_response_type(self, response_text: str) -> str:
        """Classify government response type"""
        response_lower = response_text.lower()
        
        if any(term in response_lower for term in ['accept', 'agree', 'support', 'endorse', 'welcome']):
            return 'accepted'
        elif any(term in response_lower for term in ['reject', 'disagree', 'oppose', 'decline']):
            return 'rejected'
        elif any(term in response_lower for term in ['partially', 'in part', 'some aspects']):
            return 'partially_accepted'
        elif any(term in response_lower for term in ['under consideration', 'reviewing', 'exploring']):
            return 'under_consideration'
        elif any(term in response_lower for term in ['already implemented', 'already in place', 'existing']):
            return 'already_implemented'
        elif any(term in response_lower for term in ['will implement', 'plan to', 'intend to']):
            return 'implementation_planned'
        else:
            return 'unclear'

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces
        text = re.sub(r'(\d+)([A-Z])', r'\1. \2', text)  # Fix numbered lists
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Fix sentence boundaries
        
        # Remove page headers/footers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r'^.*?confidential.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^.*?© crown copyright.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text.strip()

    def _extract_metadata_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document metadata using PyMuPDF"""
        metadata = {
            'filename': Path(pdf_path).name,
            'processed_at': datetime.now().isoformat()
        }
        
        try:
            with fitz.open(pdf_path) as doc:
                metadata.update({
                    'page_count': len(doc),
                    'file_size_mb': round(Path(pdf_path).stat().st_size / (1024 * 1024), 2),
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'creation_date': doc.metadata.get('creationDate', ''),
                    'modification_date': doc.metadata.get('modDate', '')
                })
                
        except Exception as metadata_error:
            self.logger.error(f"Error extracting metadata: {metadata_error}")
            metadata['error'] = str(metadata_error)
        
        return metadata

    def validate_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """Validate the quality of extraction"""
        try:
            # Quick extraction test
            basic_result = self._extract_basic_text(pdf_path)
            
            if not basic_result:
                return {
                    'is_valid': False,
                    'quality_score': 0.0,
                    'issues': ['Failed to extract any text'],
                    'recommendations': ['Check if file is corrupted or password protected'],
                    'text_length': 0
                }
            
            text = basic_result['text']
            text_length = len(text)
            
            # Basic validation checks
            issues = []
            recommendations = []
            
            if text_length < 500:
                issues.append('Very little text extracted')
                recommendations.append('File may be scanned image or corrupted')
            
            if text_length > 0:
                # Check for potential sections
                has_recommendations = bool(re.search(r'(?i)recommendation', text))
                has_responses = bool(re.search(r'(?i)response', text))
                has_government_markers = bool(re.search(r'(?i)(?:government|cabinet|ministry)', text))
                
                if not has_recommendations and not has_responses:
                    issues.append('No recommendations or responses detected')
                    recommendations.append('File may not contain expected content')
                
                # Check for government document indicators
                document_indicators = {
                    'has_recommendations': has_recommendations,
                    'has_responses': has_responses,
                    'has_government_markers': has_government_markers,
                    'has_numbered_items': bool(re.search(r'^\s*\d+\.\s', text, re.MULTILINE)),
                    'has_table_structure': bool(re.search(r'(?i)contents?|chapters?', text))
                }
            else:
                document_indicators = {}
            
            # Calculate quality score
            quality_score = min(1.0, text_length / 10000.0)  # Normalize to 10k chars = 1.0
            
            # Bonus for government document indicators
            if document_indicators.get('has_recommendations') or document_indicators.get('has_responses'):
                quality_score += 0.1
            if document_indicators.get('has_government_markers'):
                quality_score += 0.1
            
            quality_score = min(1.0, quality_score)
            
            return {
                'is_valid': quality_score > 0.1,
                'quality_score': quality_score,
                'text_length': text_length,
                'issues': issues,
                'recommendations': recommendations,
                'document_indicators': document_indicators
            }
            
        except Exception as validation_error:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': [f'Validation failed: {str(validation_error)}'],
                'recommendations': ['Check file permissions and format'],
                'error': str(validation_error),
                'text_length': 0
            }

    def get_document_summary(self, pdf_path: str) -> Dict[str, Any]:
        """Get a comprehensive summary of the document"""
        try:
            # Get basic info
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            validation = self.validate_extraction(pdf_path)
            
            # Try to get section preview
            sections_info = {'sections_found': 0, 'section_types': []}
            try:
                # Quick section detection
                basic_result = self._extract_basic_text(pdf_path)
                if basic_result:
                    lines = basic_result['text'].split('\n')
                    section_headers = self._find_section_headers(lines)
                    
                    sections_info = {
                        'potential_sections': len(section_headers),
                        'section_types': [header[1] for header in section_headers],
                        'section_titles': [header[2] for header in section_headers]
                    }
                    
            except Exception as sections_error:
                self.logger.warning(f"Could not analyze sections: {sections_error}")
            
            return
