# ===============================================
# FILE: modules/enhanced_section_extractor.py (COMPLETE VERSION)
# ===============================================

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
from datetime import datetime

class EnhancedSectionExtractor:
    """
    Enhanced document processor that extracts recommendations and responses sections
    from UK Government inquiry reports and response documents.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Section identifiers for inquiry recommendations
        self.recommendation_section_patterns = [
            # Standard recommendation sections in inquiry reports
            r'(?i)^(?:\d+\.?\s*)?(?:recommendations?|conclusions and recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:summary of recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:key recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:main recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:principal recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:findings and recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:inquiry recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:final recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:our recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:recommendations for (?:action|change|improvement))\s*$',
            
            # Additional patterns for government inquiry documents
            r'(?i)^(?:\d+\.?\s*)?(?:recommendations to government)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:what needs to be done)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:action required)\s*$',
        ]
        
        # Section identifiers for government responses
        self.response_section_patterns = [
            # Government response sections
            r'(?i)^(?:\d+\.?\s*)?(?:government response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:official response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:response to (?:the )?(?:inquiry )?recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:departmental response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:ministry response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:cabinet office response)\s*$',
            
            # Implementation and action sections
            r'(?i)^(?:\d+\.?\s*)?(?:implementation (?:plan|strategy))\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:action plan)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:government action)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:next steps?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:progress (?:report|update))\s*$',
            
            # Response to specific recommendations
            r'(?i)^(?:\d+\.?\s*)?(?:response to recommendation\s+\d+)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:recommendation\s+\d+\s*[-:]?\s*response)\s*$',
            
            # Acceptance/rejection responses
            r'(?i)^(?:\d+\.?\s*)?(?:accepted recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:government position)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:our response)\s*$',
        ]
        
        # Section end markers
        self.section_end_markers = [
            # General document end markers
            r'(?i)^(?:conclusion|summary|appendix|annex|bibliography|references|index)\s*$',
            r'(?i)^(?:part|chapter|section)\s+[ivxlcdm]+\s*$',  # Roman numerals
            r'(?i)^(?:part|chapter|section)\s+\d+\s*$',  # Numbers
            
            # Government document specific end markers
            r'(?i)^(?:next steps?|way forward|future work)\s*$',
            r'(?i)^(?:contact details?|further information)\s*$',
            r'(?i)^(?:glossary|abbreviations|acronyms)\s*$',
            r'(?i)^(?:published by|crown copyright|©)\s*',
            
            # Inquiry specific end markers
            r'(?i)^(?:signed|dated this|chair of (?:the )?inquiry)\s*',
            r'(?i)^(?:minister for|secretary of state)\s*',
            
            # Additional end markers
            r'(?i)^(?:acknowledgements?|thanks?|about (?:the )?(?:inquiry|commission))\s*$',
            r'(?i)^(?:terms of reference)\s*$',
        ]

    def extract_sections_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract recommendations and responses sections from PDF with enhanced detection
        """
        try:
            self.logger.info(f"Starting section extraction from: {pdf_path}")
            
            # Extract text with page numbers using pdfplumber
            pages_data = self._extract_pages_with_pdfplumber(pdf_path)
            
            if not pages_data:
                self.logger.warning("No pages extracted, falling back to PyMuPDF")
                pages_data = self._extract_pages_with_pymupdf(pdf_path)
            
            if not pages_data:
                self.logger.error("Failed to extract any pages from PDF")
                return {'sections': [], 'error': 'No pages could be extracted'}
            
            # Find relevant sections
            sections = self._find_recommendation_response_sections(pages_data)
            
            # Get document metadata
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            
            # Calculate summary statistics
            summary = self._calculate_sections_summary(sections, metadata)
            
            self.logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
            
            return {
                'sections': sections,
                'extraction_metadata': {
                    'total_pages': len(pages_data),
                    'sections_found': len(sections),
                    'extraction_method': 'enhanced_section_extractor_v2.0'
                },
                'metadata': metadata,
                'summary': summary,
                'extraction_date': datetime.now().isoformat(),
                'extractor_version': 'Enhanced_v2.0_GovernmentReports'
            }
            
        except Exception as e:
            self.logger.error(f"Error in section extraction: {str(e)}", exc_info=True)
            return {'sections': [], 'error': str(e)}

    def _extract_pages_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page using pdfplumber"""
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Try multiple extraction methods
                        text = page.extract_text()
                        
                        if not text:
                            # Try with different settings
                            text = page.extract_text(
                                x_tolerance=3,
                                y_tolerance=3,
                                layout=True
                            )
                        
                        if not text:
                            # Try extracting from tables
                            tables = page.extract_tables()
                            if tables:
                                table_text = ""
                                for table in tables:
                                    for row in table:
                                        if row:
                                            table_text += " ".join([cell or "" for cell in row]) + "\n"
                                text = table_text
                        
                        if text:
                            cleaned_text = self._clean_text(text)
                            pages_data.append({
                                'page_number': page_num,
                                'text': cleaned_text,
                                'extraction_method': 'pdfplumber'
                            })
                        else:
                            self.logger.warning(f"No text extracted from page {page_num}")
                            
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {page_num} with pdfplumber: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error opening PDF with pdfplumber: {e}")
            return []
        
        self.logger.info(f"Extracted {len(pages_data)} pages with pdfplumber")
        return pages_data

    def _extract_pages_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page using PyMuPDF as fallback"""
        pages_data = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    
                    # Try different extraction methods
                    text = page.get_text()
                    
                    if not text:
                        # Try dict method for structured extraction
                        text_dict = page.get_text("dict")
                        text = self._extract_text_from_dict(text_dict)
                    
                    if not text:
                        # Try blocks method
                        blocks = page.get_text("blocks")
                        text = "\n".join([block[4] for block in blocks if len(block) > 4])
                    
                    if text:
                        cleaned_text = self._clean_text(text)
                        pages_data.append({
                            'page_number': page_num + 1,
                            'text': cleaned_text,
                            'extraction_method': 'pymupdf'
                        })
                    else:
                        self.logger.warning(f"No text extracted from page {page_num + 1}")
                        
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num + 1} with PyMuPDF: {e}")
                    continue
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error opening PDF with PyMuPDF: {e}")
            return []
        
        self.logger.info(f"Extracted {len(pages_data)} pages with PyMuPDF")
        return pages_data

    def _extract_text_from_dict(self, text_dict: Dict) -> str:
        """Extract text from PyMuPDF dict structure"""
        text = ""
        
        try:
            if "blocks" in text_dict:
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    if "text" in span:
                                        text += span["text"]
                                text += "\n"
        except Exception as e:
            self.logger.warning(f"Error extracting text from dict: {e}")
        
        return text

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)  # Fix sentence boundaries
        text = re.sub(r'(\d)([A-Z])', r'\1. \2', text)  # Fix numbered lists
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Fix sentence boundaries
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r'^.*?confidential.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^.*?© crown copyright.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^.*?printed in.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text.strip()

    def _find_recommendation_response_sections(self, pages_data: List[Dict]) -> List[Dict[str, Any]]:
        """Find and extract recommendation and response sections"""
        sections = []
        
        # Combine all text for section detection
        full_text = "\n".join([page['text'] for page in pages_data])
        lines = full_text.split('\n')
        
        # Find section boundaries
        section_starts = self._find_section_starts(lines)
        
        if not section_starts:
            self.logger.warning("No recommendation or response sections found")
            return []
        
        # Extract each section
        for i, (start_line, section_type, title) in enumerate(section_starts):
            # Determine end of section
            if i + 1 < len(section_starts):
                end_line = section_starts[i + 1][0]
            else:
                # Look for section end markers
                end_line = self._find_section_end(lines, start_line)
                if end_line == -1:
                    end_line = len(lines)
            
            # Extract section content
            section_lines = lines[start_line:end_line]
            section_content = "\n".join(section_lines).strip()
            
            if len(section_content) > 100:  # Only include substantial sections
                # Find page range for this section
                page_start, page_end = self._find_page_range_for_section(
                    start_line, end_line, pages_data
                )
                
                # Calculate content statistics
                content_stats = self._calculate_content_stats(section_content)
                
                sections.append({
                    'type': section_type,
                    'title': title,
                    'content': section_content,
                    'page_start': page_start,
                    'page_end': page_end,
                    'line_start': start_line,
                    'line_end': end_line,
                    'content_stats': content_stats
                })
        
        return sections

    def _find_section_starts(self, lines: List[str]) -> List[Tuple[int, str, str]]:
        """Find the start lines of recommendation and response sections"""
        section_starts = []
        
        for line_num, line in enumerate(lines):
            line_clean = line.strip()
            
            if not line_clean or len(line_clean) < 3:
                continue
            
            # Check for recommendation sections
            for pattern in self.recommendation_section_patterns:
                if re.match(pattern, line_clean):
                    section_starts.append((line_num, 'recommendations', line_clean))
                    self.logger.info(f"Found recommendation section at line {line_num}: {line_clean}")
                    break
            
            # Check for response sections
            for pattern in self.response_section_patterns:
                if re.match(pattern, line_clean):
                    section_starts.append((line_num, 'responses', line_clean))
                    self.logger.info(f"Found response section at line {line_num}: {line_clean}")
                    break
        
        # Sort by line number
        section_starts.sort(key=lambda x: x[0])
        
        return section_starts

    def _find_section_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of a section based on end markers"""
        for line_num in range(start_line + 1, len(lines)):
            line_clean = lines[line_num].strip()
            
            if not line_clean:
                continue
            
            # Check against end markers
            for pattern in self.section_end_markers:
                if re.match(pattern, line_clean):
                    return line_num
        
        return -1  # No end marker found

    def _find_page_range_for_section(self, start_line: int, end_line: int, 
                                   pages_data: List[Dict]) -> Tuple[int, int]:
        """Find which pages contain the section content"""
        current_line = 0
        page_start = 1
        page_end = 1
        
        for page in pages_data:
            page_lines = page['text'].count('\n') + 1
            
            # Check if section starts in this page
            if current_line <= start_line < current_line + page_lines:
                page_start = page['page_number']
            
            # Check if section ends in this page
            if current_line <= end_line < current_line + page_lines:
                page_end = page['page_number']
                break
            
            current_line += page_lines
        
        return page_start, page_end

    def _calculate_content_stats(self, content: str) -> Dict[str, Any]:
        """Calculate statistics for section content"""
        if not content:
            return {'word_count': 0, 'char_count': 0, 'numbered_items': 0}
        
        # Count numbered items (potential recommendations/responses)
        numbered_patterns = [
            r'(?i)(?:^|\n)\s*(?:recommendation\s+)?\d+[\.:]\s',
            r'(?i)(?:^|\n)\s*\d+\.\s+[A-Z]',
            r'(?i)(?:^|\n)\s*\([a-z]\)\s',
            r'(?i)(?:^|\n)\s*[a-z][\.:]\s',
        ]
        
        numbered_items = 0
        for pattern in numbered_patterns:
            numbered_items += len(re.findall(pattern, content))
        
        return {
            'word_count': len(content.split()),
            'char_count': len(content),
            'line_count': len(content.split('\n')),
            'numbered_items': numbered_items,
            'bullet_points': len(re.findall(r'(?:^|\n)\s*[•·\-\*]\s', content)),
            'paragraphs': len([p for p in content.split('\n\n') if p.strip()])
        }

    def _calculate_sections_summary(self, sections: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Calculate summary statistics for extracted sections"""
        if not sections:
            return {
                'total_sections': 0,
                'recommendations_sections': 0,
                'responses_sections': 0,
                'total_content_length': 0,
                'avg_section_length': 0
            }
        
        recommendations_count = len([s for s in sections if s['type'] == 'recommendations'])
        responses_count = len([s for s in sections if s['type'] == 'responses'])
        total_content_length = sum([len(s['content']) for s in sections])
        
        return {
            'total_sections': len(sections),
            'recommendations_sections': recommendations_count,
            'responses_sections': responses_count,
            'total_content_length': total_content_length,
            'avg_section_length': total_content_length // len(sections) if sections else 0,
            'pages_covered': len(set([s['page_start'] for s in sections] + [s['page_end'] for s in sections])),
            'total_numbered_items': sum([s['content_stats']['numbered_items'] for s in sections])
        }

    def _extract_metadata_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            
            metadata.update({
                "page_count": doc.page_count,
                "file_size": Path(pdf_path).stat().st_size,
                "file_size_mb": Path(pdf_path).stat().st_size / (1024 * 1024),
                "processed_at": datetime.now().isoformat(),
                "filename": Path(pdf_path).name
            })
            
            doc.close()
            return metadata
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return {
                "processed_at": datetime.now().isoformat(),
                "filename": Path(pdf_path).name
            }

    def validate_sections(self, sections: List[Dict]) -> Dict[str, Any]:
        """Validate extracted sections for quality"""
        validation_results = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        if not sections:
            validation_results['is_valid'] = False
            validation_results['issues'].append('No sections extracted')
            validation_results['recommendations'].append('Check document format and content')
            return validation_results
        
        # Calculate quality metrics
        total_content = sum([len(s['content']) for s in sections])
        avg_section_length = total_content / len(sections)
        
        # Check for quality issues
        if avg_section_length < 200:
            validation_results['issues'].append('Sections appear to be very short')
            validation_results['recommendations'].append('Verify section extraction patterns')
        
        if total_content < 1000:
            validation_results['issues'].append('Very little content extracted')
            validation_results['recommendations'].append('Check if document contains expected sections')
        
        # Calculate quality score (0-1)
        quality_score = min(1.0, total_content / 5000.0)  # Normalize to 5000 chars = 1.0
        validation_results['quality_score'] = quality_score
        
        # Set validity based on score
        validation_results['is_valid'] = quality_score > 0.2
        
        return validation_results

    def get_section_types_found(self, pdf_path: str) -> Dict[str, int]:
        """Get count of each section type found in document"""
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            sections = result.get('sections', [])
            
            type_counts = {}
            for section in sections:
                section_type = section['type']
                type_counts[section_type] = type_counts.get(section_type, 0) + 1
            
            return type_counts
            
        except Exception as e:
            self.logger.error(f"Error getting section types: {e}")
            return {}

    def extract_section_titles_only(self, pdf_path: str) -> List[str]:
        """Extract just the titles of recommendation/response sections"""
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            sections = result.get('sections', [])
            
            return [section['title'] for section in sections if section.get('title')]
            
        except Exception as e:
            self.logger.error(f"Error extracting section titles: {e}")
            return []

    def get_document_overview(self, pdf_path: str) -> Dict[str, Any]:
        """Get comprehensive overview of document sections and metadata"""
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            
            if result.get('error'):
                return {
                    'filename': Path(pdf_path).name,
                    'error': result['error'],
                    'extraction_success': False
                }
            
            sections_info = []
            for section in result.get('sections', []):
                sections_info.append({
                    'type': section['type'],
                    'title': section['title'],
                    'pages': f"{section['page_start']}-{section['page_end']}",
                    'word_count': section.get('content_stats', {}).get('word_count', 0),
                    'numbered_items': section.get('content_stats', {}).get('numbered_items', 0)
                })
            
            return {
                'filename': result.get('metadata', {}).get('filename', ''),
                'total_pages': result.get('metadata', {}).get('page_count', 0),
                'file_size_mb': result.get('metadata', {}).get('file_size_mb', 0),
                'sections_found': sections_info,
                'extraction_success': len(result.get('sections', [])) > 0,
                'document_type': self._infer_document_type_from_sections(result.get('sections', []))
            }
            
        except Exception as e:
            return {
                'filename': Path(pdf_path).name,
                'error': str(e),
                'extraction_success': False
            }

    def _infer_document_type_from_sections(self, sections: List[Dict]) -> str:
        """Infer document type from extracted sections"""
        rec_sections = len([s for s in sections if s['type'] == 'recommendations'])
        resp_sections = len([s for s in sections if s['type'] == 'responses'])
        
        if rec_sections > 0 and resp_sections > 0:
            return 'combined_document'
        elif rec_sections > 0:
            return 'inquiry_report'
        elif resp_sections > 0:
            return 'government_response'
        else:
            return 'unknown'

    def extract_numbered_recommendations(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract individual numbered recommendations from sections"""
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            sections = result.get('sections', [])
            
            recommendations = []
            
            for section in sections:
                if section['type'] == 'recommendations':
                    content = section['content']
                    
                    # Find numbered recommendations
                    numbered_items = re.findall(
                        r'(?i)((?:recommendation\s+)?\d+[\.:]\s*[^\n\r]+(?:\n(?!\s*\d+[\.:]).*?)*)',
                        content,
                        re.MULTILINE | re.DOTALL
                    )
                    
                    for i, item in enumerate(numbered_items, 1):
                        recommendations.append({
                            'section_title': section['title'],
                            'recommendation_number': i,
                            'text': item.strip(),
                            'page_start': section['page_start'],
                            'page_end': section['page_end'],
                            'word_count': len(item.split())
                        })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error extracting numbered recommendations: {e}")
            return []

    def compare_documents(self, inquiry_pdf: str, response_pdf: str) -> Dict[str, Any]:
        """Compare an inquiry document with its government response"""
        try:
            # Extract from both documents
            inquiry_result = self.extract_sections_from_pdf(inquiry_pdf)
            response_result = self.extract_sections_from_pdf(response_pdf)
            
            # Get recommendations from inquiry
            inquiry_recs = [s for s in inquiry_result.get('sections', []) if s['type'] == 'recommendations']
            
            # Get responses from response document
            response_sections = [s for s in response_result.get('sections', []) if s['type'] == 'responses']
            
            comparison = {
                'inquiry_document': {
                    'filename': Path(inquiry_pdf).name,
                    'recommendations_sections': len(inquiry_recs),
                    'total_recommendations': sum([s['content_stats']['numbered_items'] for s in inquiry_recs])
                },
                'response_document': {
                    'filename': Path(response_pdf).name,
                    'response_sections': len(response_sections),
                    'total_responses': sum([s['content_stats']['numbered_items'] for s in response_sections])
                },
                'coverage_analysis': {
                    'has_matching_structure': len(inquiry_recs) > 0 and len(response_sections) > 0,
                    'recommendation_response_ratio': len(response_sections) / max(len(inquiry_recs), 1)
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing documents: {e}")
            return {'error': str(e)}


# Utility functions for standalone usage
def extract_sections_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Standalone function to extract sections from PDF"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_sections_from_pdf(pdf_path)


def get_section_summary(pdf_path: str) -> Dict[str, Any]:
    """Get a quick summary of sections in a PDF"""
    extractor = EnhancedSectionExtractor()
    result = extractor.extract_sections_from_pdf(pdf_path)
    
    return {
        'filename': Path(pdf_path).name,
        'sections_found': len(result.get('sections', [])),
        'section_types': extractor.get_section_types_found(pdf_path),
        'extraction_successful': len(result.get('sections', [])) > 0,
        'error': result.get('error', None)
    }


def validate_section_extraction(pdf_path: str) -> bool:
    """Quick validation - returns True if sections were successfully extracted"""
    extractor = EnhancedSectionExtractor()
    result = extractor.extract_sections_from_pdf(pdf_path)
    return len(result.get('sections', [])) > 0


# Test function
def test_extraction(pdf_path: str):
    """Test section extraction with detailed output"""
    extractor = EnhancedSectionExtractor()
    
    print(f"Testing section extraction for: {pdf_path}")
    print("=" * 60)
    
    # Extract sections
    result = extractor.extract_sections_from_pdf(pdf_path)
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
        return
    
    sections = result.get('sections', [])
    
    if not sections:
        print("❌ No sections found")
        return
    
    print(f"✅ Found {len(sections)} sections:")
    print()
    
    for i, section in enumerate(sections, 1):
        print(f"Section {i}: {section['type'].upper()}")
        print(f"  Title: {section['title']}")
        print(f"  Pages: {section['page_start']}-{section['page_end']}")
        print(f"  Content length: {len(section['content'])} characters")
        print(f"  Word count: {section['content_stats']['word_count']}")
        print(f"  Numbered items: {section['content_stats']['numbered_items']}")
        print(f"  First 100 chars: {section['content'][:100]}...")
        print("-" * 40)
    
    # Show summary
    summary = result.get('summary', {})
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Validate
    validation = extractor.validate_sections(sections)
    print(f"\nValidation:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Quality Score: {validation['quality_score']:.2f}")
    if validation['issues']:
        print(f"  Issues: {', '.join(validation['issues'])}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_extraction(sys.argv[1])
    else:
        print("Usage: python enhanced_section_extractor.py <pdf_file_path>")
