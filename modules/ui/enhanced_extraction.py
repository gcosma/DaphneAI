# ===============================================
# FILE: modules/enhanced_section_extractor.py (UPDATED FOR GOVERNMENT REPORTS)
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
        
        # UPDATED: Section identifiers for inquiry recommendations
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
            
            # REMOVED coroner-specific patterns - no longer needed
            # r'(?i)^(?:\d+\.?\s*)?(?:coroner\'s concerns?)\s*$',
            # r'(?i)^(?:\d+\.?\s*)?(?:matters? of concern)\s*$',
        ]
        
        # UPDATED: Section identifiers for government responses
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
        ]
        
        # UPDATED: Section end markers
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
        ]

    def extract_sections_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract recommendations and responses sections from PDF with enhanced detection
        """
        try:
            # Extract text with page numbers using pdfplumber
            pages_data = self._extract_pages_with_pdfplumber(pdf_path)
            
            # Find relevant sections
            sections = self._find_recommendation_response_sections(pages_data)
            
            # Get document metadata
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            
            # Calculate summary statistics
            summary = self._calculate_sections_summary(sections, metadata)
            
            return {
                'sections': sections,
                'metadata': metadata,
                'summary': summary,
                'extraction_date': datetime.now().isoformat(),
                'extractor_version': 'Enhanced_v2.0_Gov_Reports'
            }
            
        except Exception as e:
            self.logger.error(f"Section extraction failed for {pdf_path}: {e}")
            return {
                'sections': [],
                'metadata': {},
                'summary': {},
                'error': str(e)
            }

    def _extract_pages_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page with pdfplumber for better accuracy"""
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text
                        text = page.extract_text()
                        if not text:
                            continue
                        
                        # Clean and normalize text
                        text = self._normalize_page_text(text)
                        
                        pages_data.append({
                            'page_number': page_num,
                            'text': text,
                            'char_count': len(text),
                            'word_count': len(text.split()) if text else 0
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error opening PDF with pdfplumber: {e}")
            
        return pages_data

    def _normalize_page_text(self, text: str) -> str:
        """Normalize text extracted from PDF pages"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces
        text = re.sub(r'(\d+)\.([A-Z])', r'\1. \2', text)  # Fix numbered lists
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Fix sentence boundaries
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'^\d+\s*, '', text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r'^.*?confidential.*?, '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^.*?© crown copyright.*?, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text.strip()

    def _find_recommendation_response_sections(self, pages_data: List[Dict]) -> List[Dict[str, Any]]:
        """Find and extract recommendation and response sections"""
        sections = []
        
        # Combine all text for section detection
        full_text = "\n".join([page['text'] for page in pages_data])
        lines = full_text.split('\n')
        
        # Find section boundaries
        section_starts = self._find_section_starts(lines)
        
        # Extract each section
        for i, (start_line, section_type, title) in enumerate(section_starts):
            # Determine end of section
            if i + 1 < len(section_starts):
                end_line = section_starts[i + 1][0]
            else:
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
            
            # Check for recommendation sections
            for pattern in self.recommendation_section_patterns:
                if re.match(pattern, line_clean):
                    section_starts.append((line_num, 'recommendations', line_clean))
                    break
            
            # Check for response sections
            for pattern in self.response_section_patterns:
                if re.match(pattern, line_clean):
                    section_starts.append((line_num, 'responses', line_clean))
                    break
        
        # Sort by line number
        section_starts.sort(key=lambda x: x[0])
        
        return section_starts

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
            r'(?i)(?:^|\n)\s*[a-z][\.:]\s'
        ]
        
        numbered_items = 0
        for pattern in numbered_patterns:
            numbered_items += len(re.findall(pattern, content))
        
        return {
            'word_count': len(content.split()),
            'char_count': len(content),
            'numbered_items': numbered_items,
            'lines': len(content.split('\n'))
        }

    def _extract_metadata_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document metadata using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            return {
                'filename': Path(pdf_path).name,
                'total_pages': doc.page_count,
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'file_size_bytes': Path(pdf_path).stat().st_size,
                'file_size_mb': round(Path(pdf_path).stat().st_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return {
                'filename': Path(pdf_path).name,
                'total_pages': 0,
                'error': str(e)
            }

    def _calculate_sections_summary(self, sections: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Calculate summary statistics for extracted sections"""
        if not sections:
            return {
                'total_sections': 0,
                'recommendations_sections': 0,
                'responses_sections': 0,
                'total_pages_covered': {'total_pages': 0, 'page_range': 'none'},
                'sections_summary': {
                    'total_sections': 0,
                    'recommendations_sections': 0,
                    'responses_sections': 0
                }
            }
        
        # Count sections by type
        rec_sections = len([s for s in sections if s['type'] == 'recommendations'])
        resp_sections = len([s for s in sections if s['type'] == 'responses'])
        
        # Calculate page coverage
        all_pages = []
        for section in sections:
            for page in range(section['page_start'], section['page_end'] + 1):
                if page not in all_pages:
                    all_pages.append(page)
        
        page_range = f"{min(all_pages)}-{max(all_pages)}" if all_pages else "none"
        
        # Calculate total content statistics
        total_words = sum(s.get('content_stats', {}).get('word_count', 0) for s in sections)
        total_items = sum(s.get('content_stats', {}).get('numbered_items', 0) for s in sections)
        
        return {
            'total_sections': len(sections),
            'recommendations_sections': rec_sections,
            'responses_sections': resp_sections,
            'total_pages_covered': {
                'total_pages': len(all_pages),
                'page_range': page_range,
                'pages_list': sorted(all_pages)
            },
            'content_summary': {
                'total_words': total_words,
                'total_numbered_items': total_items,
                'average_words_per_section': total_words // len(sections) if sections else 0
            },
            'sections_summary': {
                'total_sections': len(sections),
                'recommendations_sections': rec_sections,
                'responses_sections': resp_sections
            },
            'document_info': {
                'filename': metadata.get('filename', ''),
                'total_document_pages': metadata.get('total_pages', 0),
                'coverage_percentage': (len(all_pages) / metadata.get('total_pages', 1)) * 100 if metadata.get('total_pages') else 0
            }
        }

    def validate_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """Validate the quality of section extraction"""
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            sections = result.get('sections', [])
            summary = result.get('summary', {})
            
            validation = {
                'is_valid': False,
                'quality_score': 0,
                'issues': [],
                'recommendations': [],
                'sections_found': len(sections),
                'document_type': 'unknown'
            }
            
            # Determine document type based on sections found
            rec_sections = len([s for s in sections if s['type'] == 'recommendations'])
            resp_sections = len([s for s in sections if s['type'] == 'responses'])
            
            if rec_sections > 0 and resp_sections > 0:
                validation['document_type'] = 'combined_inquiry_and_response'
            elif rec_sections > 0:
                validation['document_type'] = 'inquiry_report'
            elif resp_sections > 0:
                validation['document_type'] = 'government_response'
            
            # Quality assessment
            if not sections:
                validation['issues'].append("No recommendation or response sections found")
                validation['recommendations'].append("Try full document extraction or check document format")
                return validation
            
            # Calculate quality score (0-100)
            scores = []
            
            # Section coverage score
            total_pages = result.get('metadata', {}).get('total_pages', 1)
            pages_covered = summary.get('total_pages_covered', {}).get('total_pages', 0)
            coverage_score = min(100, (pages_covered / total_pages) * 100) if total_pages > 0 else 0
            scores.append(coverage_score)
            
            # Content quality score
            total_words = summary.get('content_summary', {}).get('total_words', 0)
            content_score = min(100, total_words / 100)  # 1 point per 100 words, max 100
            scores.append(content_score)
            
            # Section diversity score
            diversity_score = min(100, len(sections) * 25)  # 25 points per section, max 100
            scores.append(diversity_score)
            
            validation['quality_score'] = sum(scores) / len(scores) if scores else 0
            validation['is_valid'] = validation['quality_score'] >= 30
            
            # Provide recommendations based on results
            if validation['quality_score'] < 50:
                validation['recommendations'].append("Consider using full document extraction for better coverage")
            
            if rec_sections == 0:
                validation['recommendations'].append("No recommendation sections found - check if this is a response document")
            
            if resp_sections == 0:
                validation['recommendations'].append("No response sections found - check if this is an original inquiry report")
            
            if validation['quality_score'] >= 70:
                validation['recommendations'].append("Good extraction quality - proceed with recommendation extraction")
            
            return validation
            
        except Exception as e:
            return {
                'is_valid': False,
                'quality_score': 0,
                'issues': [f"Validation failed: {str(e)}"],
                'recommendations': ["Check document format and try again"],
                'error': str(e)
            }

    def get_document_summary(self, pdf_path: str) -> Dict[str, Any]:
        """Get a quick summary of document content for preview"""
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            
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
                'total_pages': result.get('metadata', {}).get('total_pages', 0),
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


# Test function
def test_section_extraction():
    """Test the enhanced section extractor"""
    extractor = EnhancedSectionExtractor()
    
    # This would be called with an actual PDF path
    print("Enhanced Section Extractor for Government Reports - Ready!")
    print("Features:")
    print("- Extracts inquiry recommendations and government responses")
    print("- Accurate page number tracking")
    print("- Content statistics and validation")
    print("- Support for combined documents")
    
    return extractor

if __name__ == "__main__":
    test_section_extraction()
