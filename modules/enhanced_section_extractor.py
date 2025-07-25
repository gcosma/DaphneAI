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
    
    Designed specifically for documents like:
    - "Government Response to the Infected Blood Inquiry"
    - "Volume 1 - Overview and Recommendations"
    - Cabinet Office reports with numbered recommendations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced section identifiers for inquiry recommendations
        self.recommendation_section_patterns = [
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
            
            # Table of contents style patterns
            r'(?i)^(?:\d+\.\d+\s+)?(?:overview and )?recommendations?\s*$',
            r'(?i)^(?:\d+\.\d+\s+)?(?:list of )?recommendations?\s*$',
            
            # Additional patterns for government inquiry documents
            r'(?i)^(?:\d+\.?\d*\s*)?(?:recommendations to government)\s*$',
            r'(?i)^(?:\d+\.?\d*\s*)?(?:what needs to be done)\s*$',
            r'(?i)^(?:\d+\.?\d*\s*)?(?:action required)\s*$',
        ]
        
        # Enhanced section identifiers for government responses
        self.response_section_patterns = [
            # Government response sections (primary patterns)
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
            
            # Acceptance/rejection responses
            r'(?i)^(?:\d+\.?\d*\s*)?(?:accepted recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\d*\s*)?(?:government position)\s*$',
            r'(?i)^(?:\d+\.?\d*\s*)?(?:our response)\s*$',
            
            # SPECIFIC PATTERNS for the Infected Blood Inquiry response format
            r'(?i)^(?:\d+\.?\d*\s*)?(?:responses? to individual recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\d*\s*)?(?:detailed responses?)\s*$',
            r'(?i)^(?:\d+\.?\d*\s*)?(?:government responses? to recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\d*\s*)?(?:recommendation\s+\d+)\s*$',  # Just "Recommendation 1", "Recommendation 2" etc.
            
            # Table format responses (for documents with tabular responses)
            r'(?i)^(?:\d+\.?\d*\s*)?(?:responses? table)\s*$',
            r'(?i)^(?:\d+\.?\d*\s*)?(?:summary of responses?)\s*$',
        ]
        
        # Enhanced section end markers
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
            r'(?i)^(?:further reading|related publications?)\s*$',
        ]

    def extract_sections_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract recommendations and responses sections from PDF with enhanced detection
        """
        try:
            self.logger.info(f"Starting enhanced section extraction from: {pdf_path}")
            
            # Extract text with page numbers using pdfplumber
            pages_data = self._extract_pages_with_pdfplumber(pdf_path)
            
            if not pages_data:
                self.logger.warning("No pages extracted with pdfplumber, trying PyMuPDF fallback")
                pages_data = self._extract_pages_with_pymupdf(pdf_path)
            
            if not pages_data:
                self.logger.error("Failed to extract any pages from PDF")
                return {
                    'sections': [],
                    'error': 'No pages could be extracted from PDF',
                    'success': False
                }
            
            # Find relevant sections
            sections = self._find_recommendation_response_sections(pages_data)
            
            # Get document metadata
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            
            # Calculate summary statistics
            summary = self._calculate_sections_summary(sections, metadata)
            
            self.logger.info(f"Enhanced extraction completed: {len(sections)} sections found")
            
            return {
                'sections': sections,
                'extraction_metadata': {
                    'total_pages': len(pages_data),
                    'sections_found': len(sections),
                    'extraction_method': 'enhanced_section_extractor_v3.0'
                },
                'metadata': metadata,
                'summary': summary,
                'extraction_date': datetime.now().isoformat(),
                'extractor_version': 'Enhanced_v3.0_Government_Reports',
                'success': True
            }
            
        except Exception as extraction_error:
            self.logger.error(f"Error extracting sections from {pdf_path}: {extraction_error}", exc_info=True)
            return {
                'sections': [],
                'metadata': {'error': str(extraction_error)},
                'summary': {'error': 'Extraction failed'},
                'extraction_date': datetime.now().isoformat(),
                'extractor_version': 'Enhanced_v3.0_Government_Reports',
                'success': False,
                'error': str(extraction_error)
            }

    def _extract_pages_with_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Extract text from each page with pdfplumber"""
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text
                        text = page.extract_text()
                        if text:
                            # Clean and normalize text
                            cleaned_text = self._clean_extracted_text(text)
                            pages_data.append({
                                'page_number': page_num,
                                'text': cleaned_text,
                                'raw_text': text,
                                'char_count': len(cleaned_text)
                            })
                    except Exception as page_error:
                        self.logger.warning(f"Error extracting page {page_num} with pdfplumber: {page_error}")
                        continue
                        
        except Exception as pdf_error:
            self.logger.error(f"Error opening PDF with pdfplumber: {pdf_error}")
            raise
        
        return pages_data

    def _extract_pages_with_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Fallback extraction using PyMuPDF"""
        pages_data = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    try:
                        page = doc[page_num]
                        text = page.get_text()
                        if text:
                            cleaned_text = self._clean_extracted_text(text)
                            pages_data.append({
                                'page_number': page_num + 1,
                                'text': cleaned_text,
                                'raw_text': text,
                                'char_count': len(cleaned_text)
                            })
                    except Exception as page_error:
                        self.logger.warning(f"Error extracting page {page_num + 1} with PyMuPDF: {page_error}")
                        continue
                        
        except Exception as pdf_error:
            self.logger.error(f"Error opening PDF with PyMuPDF: {pdf_error}")
            raise
        
        return pages_data

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

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces
        text = re.sub(r'(\d+)([A-Z])', r'\1. \2', text)  # Fix numbered lists
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Fix sentence boundaries
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r'^.*?confidential.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^.*?© crown copyright.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
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
            self.logger.warning("No section headers found")
            return sections
        
        # Extract each section
        for i, (start_line, section_type, title) in enumerate(section_starts):
            # Determine end of section
            if i + 1 < len(section_starts):
                end_line = section_starts[i + 1][0]
            else:
                end_line = len(lines)
            
            # Extract section content
            section_lines = lines[start_line + 1:end_line]  # Skip the header line
            section_content = "\n".join(section_lines).strip()
            
            if len(section_content) > 100:  # Only include substantial sections
                # Find page range for this section
                page_start, page_end = self._find_page_range_for_section(
                    start_line, end_line, pages_data
                )
                
                # Calculate content statistics
                content_stats = self._calculate_content_stats(section_content)
                
                # Calculate extraction confidence
                extraction_confidence = self._calculate_extraction_confidence(section_content, section_type)
                
                section = {
                    'type': section_type,
                    'title': title,
                    'content': section_content,
                    'page_start': page_start,
                    'page_end': page_end,
                    'line_start': start_line,
                    'line_end': end_line,
                    'content_stats': content_stats,
                    'extraction_confidence': extraction_confidence,
                    'extraction_method': 'enhanced_pattern_matching'
                }
                
                sections.append(section)
                self.logger.info(f"Extracted {section_type} section: {title} (confidence: {extraction_confidence:.2f})")
        
        return sections

    def _find_section_starts(self, lines: List[str]) -> List[Tuple[int, str, str]]:
        """Find the start of recommendation and response sections"""
        section_starts = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 3:
                continue
            
            # Check for recommendation sections
            for pattern in self.recommendation_section_patterns:
                if re.match(pattern, line_stripped):
                    section_starts.append((i, 'recommendation', line_stripped))
                    self.logger.debug(f"Found recommendation section at line {i}: {line_stripped}")
                    break
            
            # Check for response sections
            for pattern in self.response_section_patterns:
                if re.match(pattern, line_stripped):
                    section_starts.append((i, 'response', line_stripped))
                    self.logger.debug(f"Found response section at line {i}: {line_stripped}")
                    break
        
        # Sort by line number
        section_starts.sort(key=lambda x: x[0])
        
        return section_starts

    def _find_page_range_for_section(self, start_line: int, end_line: int, pages_data: List[Dict]) -> Tuple[int, int]:
        """Find the page range for a section based on line numbers"""
        if not pages_data:
            return 1, 1
        
        # Calculate character positions for better page mapping
        current_char = 0
        section_start_char = 0
        section_end_char = 0
        
        # Get full text and calculate section character positions
        full_text = "\n".join([page['text'] for page in pages_data])
        lines = full_text.split('\n')
        
        for i, line in enumerate(lines):
            if i == start_line:
                section_start_char = current_char
            if i == end_line:
                section_end_char = current_char
                break
            current_char += len(line) + 1  # +1 for newline
        
        # Find pages that contain these character ranges
        page_start = 1
        page_end = 1
        current_char = 0
        
        for page in pages_data:
            page_char_count = page['char_count']
            
            if current_char <= section_start_char < current_char + page_char_count:
                page_start = page['page_number']
            
            if current_char <= section_end_char < current_char + page_char_count:
                page_end = page['page_number']
            
            current_char += page_char_count
        
        return page_start, max(page_start, page_end)

    def _calculate_content_stats(self, content: str) -> Dict[str, Any]:
        """Calculate statistics for section content"""
        if not content:
            return {'word_count': 0, 'char_count': 0, 'line_count': 0, 'numbered_items': 0}
        
        words = content.split()
        lines = content.split('\n')
        
        # Count different types of numbered items
        numbered_items = len(re.findall(r'^\s*\d+[\.\)]\s', content, re.MULTILINE))
        lettered_items = len(re.findall(r'^\s*[a-z][\.\)]\s', content, re.MULTILINE))
        bullet_points = len(re.findall(r'[•·‣⁃]', content))
        
        # Look for recommendation/response specific patterns
        recommendation_indicators = len(re.findall(r'(?i)\b(?:recommend|should|must|ought)\b', content))
        response_indicators = len(re.findall(r'(?i)\b(?:accept|reject|agree|implement)\b', content))
        
        return {
            'word_count': len(words),
            'char_count': len(content),
            'line_count': len(lines),
            'numbered_items': numbered_items,
            'lettered_items': lettered_items,
            'bullet_points': bullet_points,
            'avg_words_per_line': len(words) / len(lines) if lines else 0,
            'recommendation_indicators': recommendation_indicators,
            'response_indicators': response_indicators,
            'has_structured_content': numbered_items > 0 or lettered_items > 0 or bullet_points > 0
        }

    def _calculate_extraction_confidence(self, content: str, section_type: str) -> float:
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
            if re.search(r'(?i)\b(?:inquiry|commission|investigation)\b', content):
                confidence += 0.1
        
        elif section_type == 'response':
            if re.search(r'(?i)\b(?:accept|reject|agree|disagree|implement|action)\b', content):
                confidence += 0.2
            if re.search(r'(?i)\b(?:government|department|ministry|cabinet)\b', content):
                confidence += 0.1
            if re.search(r'(?i)\b(?:recommendation\s+\d+)\b', content):
                confidence += 0.1
        
        return min(1.0, confidence)

    def _calculate_sections_summary(self, sections: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Calculate summary statistics for extracted sections"""
        if not sections:
            return {
                'total_sections': 0,
                'recommendation_sections': 0,
                'response_sections': 0,
                'total_content_length': 0,
                'avg_section_length': 0,
                'quality_score': 0.0,
                'extraction_success': False
            }
        
        recommendation_count = len([s for s in sections if s['type'] == 'recommendation'])
        response_count = len([s for s in sections if s['type'] == 'response'])
        total_content = sum([len(s['content']) for s in sections])
        avg_length = total_content / len(sections) if sections else 0
        
        # Calculate quality score based on content amount and structure
        quality_factors = []
        
        # Content amount factor
        content_factor = min(1.0, total_content / 5000.0)
        quality_factors.append(content_factor)
        
        # Section balance factor
        if recommendation_count > 0 and response_count > 0:
            balance_factor = 1.0
        elif recommendation_count > 0 or response_count > 0:
            balance_factor = 0.7
        else:
            balance_factor = 0.0
        quality_factors.append(balance_factor)
        
        # Structure factor (numbered items, etc.)
        total_numbered_items = sum([s.get('content_stats', {}).get('numbered_items', 0) for s in sections])
        structure_factor = min(1.0, total_numbered_items / 10.0)
        quality_factors.append(structure_factor)
        
        # Average confidence factor
        avg_confidence = sum([s.get('extraction_confidence', 0) for s in sections]) / len(sections)
        quality_factors.append(avg_confidence)
        
        quality_score = sum(quality_factors) / len(quality_factors)
        
        return {
            'total_sections': len(sections),
            'recommendation_sections': recommendation_count,
            'response_sections': response_count,
            'total_content_length': total_content,
            'avg_section_length': avg_length,
            'total_numbered_items': total_numbered_items,
            'avg_confidence': avg_confidence,
            'quality_score': quality_score,
            'extraction_success': quality_score > 0.3
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
        
        # Check section types
        section_types = [s['type'] for s in sections]
        if 'recommendation' not in section_types and 'response' not in section_types:
            validation_results['issues'].append('No recommendations or responses found')
            validation_results['recommendations'].append('Verify document contains the expected content')
        
        # Check confidence scores
        low_confidence_sections = [s for s in sections if s.get('extraction_confidence', 0) < 0.5]
        if len(low_confidence_sections) > len(sections) / 2:
            validation_results['issues'].append('Many sections have low confidence scores')
            validation_results['recommendations'].append('Review extraction patterns for this document type')
        
        # Calculate overall quality score (0-1)
        quality_score = min(1.0, total_content / 5000.0)  # Normalize to 5000 chars = 1.0
        if len(validation_results['issues']) == 0:
            quality_score += 0.2  # Bonus for no issues
        
        validation_results['quality_score'] = quality_score
        
        # Set validity based on score and issues
        validation_results['is_valid'] = quality_score > 0.2 and len(validation_results['issues']) <= 2
        
        return validation_results

    def get_section_types_found(self, pdf_path: str) -> Dict[str, int]:
        """Get count of each section type found in document"""
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            sections = result.get('sections', [])
            
            type_counts = {}
            for section in sections:
                section_type = section.get('type', 'unknown')
                type_counts[section_type] = type_counts.get(section_type, 0) + 1
            
            return type_counts
            
        except Exception as count_error:
            self.logger.error(f"Error counting section types: {count_error}")
            return {}

    def extract_individual_recommendations(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract individual recommendations from the document
        ENHANCED: Better pattern matching for numbered recommendations
        """
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            sections = result.get('sections', [])
            
            recommendations = []
            
            for section in sections:
                if section['type'] == 'recommendation':
                    content = section['content']
                    
                    # Look for numbered recommendations within the section
                    numbered_patterns = [
                        r'(?:^|\n)\s*(\d+)[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|\n\s*[A-Z][a-z]+\s*\d+|$)',
                        r'(?:^|\n)\s*Recommendation\s+(\d+):\s*(.+?)(?=\n\s*Recommendation\s+\d+|$)',
                        r'(?:^|\n)\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|$)',
                        r'(?:^|\n)\s*([a-z])[\.\)]\s+(.+?)(?=\n\s*[a-z][\.\)]|$)'
                    ]
                    
                    for pattern in numbered_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
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
                    
                    # If no numbered recommendations found, treat entire section as one recommendation
                    if not any(rec for rec in recommendations if rec['source_section'] == section['title']):
                        if len(content) > 100:
                            recommendations.append({
                                'id': f"section_{len(recommendations) + 1}",
                                'number': str(len(recommendations) + 1),
                                'text': content,
                                'source_section': section['title'],
                                'page_start': section['page_start'],
                                'page_end': section['page_end'],
                                'extraction_method': 'full_section',
                                'confidence': 0.7
                            })
            
            return recommendations
            
        except Exception as extract_error:
            self.logger.error(f"Error extracting individual recommendations: {extract_error}")
            return []

    def extract_individual_responses(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract individual responses from the document
        ENHANCED: Better handling of tabular and numbered responses
        """
        try:
            result = self.extract_sections_from_pdf(pdf_path)
            sections = result.get('sections', [])
            
            responses = []
            
            for section in sections:
                if section['type'] == 'response':
                    content = section['content']
                    
                    # Look for responses to specific recommendations
                    response_patterns = [
                        r'(?:^|\n)\s*(?:Response\s+to\s+)?Recommendation\s+(\d+)[:\.]?\s*(.+?)(?=\n\s*(?:Response\s+to\s+)?Recommendation\s+\d+|$)',
                        r'(?:^|\n)\s*(\d+)[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|$)',
                        r'(?:^|\n)\s*Rec(?:ommendation)?\s+(\d+)[:\-]\s*(.+?)(?=\n\s*Rec(?:ommendation)?\s+\d+|$)',
                        r'(?:^|\n)\s*(\d+)\s*[-:]\s*(.+?)(?=\n\s*\d+\s*[-:]|$)'
                    ]
                    
                    for pattern in response_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                        for match in matches:
                            rec_num = match.group(1)
                            response_text = match.group(2).strip()
                            
                            if len(response_text) > 30:  # Only substantial responses
                                # Determine response type (accepted, rejected, etc.)
                                response_type = self._classify_response_type(response_text)
                                
                                responses.append({
                                    'id': f"resp_{rec_num}",
                                    'recommendation_number': rec_num,
                                    'response_text': response_text,
                                    'response_type': response_type,
                                    'source_section': section['title'],
                                    'page_start': section['page_start'],
                                    'page_end': section['page_end'],
                                    'extraction_method': 'numbered_pattern',
                                    'confidence': 0.9
                                })
            
            return responses
            
        except Exception as extract_error:
            self.logger.error(f"Error extracting individual responses: {extract_error}")
            return []

    def _classify_response_type(self, response_text: str) -> str:
        """Classify the type of response (accepted, rejected, etc.)"""
        response_lower = response_text.lower()
        
        # Acceptance indicators
        if any(term in response_lower for term in [
            'accept', 'agree', 'support', 'endorse', 'welcome', 'committed to'
        ]):
            return 'accepted'
        
        # Rejection indicators  
        elif any(term in response_lower for term in [
            'reject', 'disagree', 'oppose', 'decline', 'not accept'
        ]):
            return 'rejected'
        
        # Partial acceptance
        elif any(term in response_lower for term in [
            'partially', 'in part', 'some aspects', 'partially accept'
        ]):
            return 'partially_accepted'
        
        # Under consideration
        elif any(term in response_lower for term in [
            'under consideration', 'reviewing', 'exploring', 'considering'
        ]):
            return 'under_consideration'
        
        # Already implemented
        elif any(term in response_lower for term in [
            'already implemented', 'already in place', 'existing', 'current practice'
        ]):
            return 'already_implemented'
        
        # Implementation planned
        elif any(term in response_lower for term in [
            'will implement', 'plan to', 'intend to', 'committed to implementing'
        ]):
            return 'implementation_planned'
        
        else:
            return 'unclear'

    def compare_inquiry_and_response_documents(self, inquiry_pdf: str, response_pdf: str) -> Dict[str, Any]:
        """
        Compare an inquiry report with its corresponding government response
        """
        try:
            self.logger.info(f"Comparing inquiry document {inquiry_pdf} with response {response_pdf}")
            
            # Extract from both documents
            inquiry_result = self.extract_sections_from_pdf(inquiry_pdf)
            response_result = self.extract_sections_from_pdf(response_pdf)
            
            inquiry_sections = inquiry_result.get('sections', [])
            response_sections = response_result.get('sections', [])
            
            # Extract individual items
            inquiry_recs = self.extract_individual_recommendations(inquiry_pdf)
            response_items = self.extract_individual_responses(response_pdf)
            
            # Analyze coverage
            comparison = {
                'inquiry_document': {
                    'filename': Path(inquiry_pdf).name,
                    'recommendation_sections': len([s for s in inquiry_sections if s['type'] == 'recommendation']),
                    'individual_recommendations': len(inquiry_recs)
                },
                'response_document': {
                    'filename': Path(response_pdf).name,
                    'response_sections': len([s for s in response_sections if s['type'] == 'response']),
                    'individual_responses': len(response_items)
                },
                'coverage_analysis': {
                    'has_matching_structure': len(inquiry_recs) > 0 and len(response_items) > 0,
                    'recommendation_response_ratio': len(response_items) / max(len(inquiry_recs), 1),
                    'potential_matches': self._find_potential_matches(inquiry_recs, response_items)
                },
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            return comparison
            
        except Exception as comparison_error:
            self.logger.error(f"Error comparing documents: {comparison_error}")
            return {'error': str(comparison_error)}

    def _find_potential_matches(self, recommendations: List[Dict], responses: List[Dict]) -> List[Dict]:
        """Find potential matches between recommendations and responses"""
        matches = []
        
        for rec in recommendations:
            rec_num = rec.get('number', '')
            
            # Look for response with matching number
            matching_responses = [
                resp for resp in responses 
                if resp.get('recommendation_number', '') == rec_num
            ]
            
            if matching_responses:
                for response in matching_responses:
                    matches.append({
                        'recommendation_id': rec['id'],
                        'recommendation_number': rec_num,
                        'response_id': response['id'],
                        'response_type': response.get('response_type', 'unclear'),
                        'match_confidence': 0.9 if rec_num == response.get('recommendation_number') else 0.5
                    })
        
        return matches


# ===============================================
# UTILITY FUNCTIONS FOR STANDALONE USAGE
# ===============================================

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
        'extraction_successful': result.get('success', False),
        'error': result.get('error', None)
    }

def validate_section_extraction(pdf_path: str) -> bool:
    """Quick validation - returns True if sections were successfully extracted"""
    extractor = EnhancedSectionExtractor()
    result = extractor.extract_sections_from_pdf(pdf_path)
    return result.get('success', False) and len(result.get('sections', [])) > 0

def extract_government_recommendations(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract individual recommendations from government documents"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_individual_recommendations(pdf_path)

def extract_government_responses(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract individual responses from government documents"""
    extractor = EnhancedSectionExtractor()
    return extractor.extract_individual_responses(pdf_path)

def compare_inquiry_and_response(inquiry_pdf: str, response_pdf: str) -> Dict[str, Any]:
    """Compare inquiry document with government response"""
    extractor = EnhancedSectionExtractor()
    return extractor.compare_inquiry_and_response_documents(inquiry_pdf, response_pdf)

# ===============================================
# TESTING AND DEBUG FUNCTIONS
# ===============================================

def test_extraction(pdf_path: str):
    """Test section extraction with detailed output"""
    extractor = EnhancedSectionExtractor()
    
    print(f"Testing enhanced section extraction for: {pdf_path}")
    print("=" * 60)
    
    # Extract sections
    result = extractor.extract_sections_from_pdf(pdf_path)
    
    if not result.get('success'):
        print(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
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
        print(f"  Confidence: {section['extraction_confidence']:.2f}")
        print(f"  Preview: {section['content'][:150]}...")
        print("-" * 40)
    
    # Show summary
    summary = result.get('summary', {})
    print("\nExtraction Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Validate
    validation = extractor.validate_sections(sections)
    print(f"\nValidation Results:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Quality Score: {validation['quality_score']:.2f}")
    if validation['issues']:
        print(f"  Issues: {', '.join(validation['issues'])}")
    if validation['recommendations']:
        print(f"  Recommendations: {', '.join(validation['recommendations'])}")

def debug_section_detection(pdf_path: str):
    """Debug section detection step by step"""
    extractor = EnhancedSectionExtractor()
    
    print(f"DEBUG: Section detection for {pdf_path}")
    print("=" * 60)
    
    # Step 1: Extract pages
    print("Step 1: Extracting pages...")
    pages_data = extractor._extract_pages_with_pdfplumber(pdf_path)
    print(f"  ✅ Extracted {len(pages_data)} pages")
    
    # Step 2: Find section headers
    print("\nStep 2: Finding section headers...")
    full_text = "\n".join([page['text'] for page in pages_data])
    lines = full_text.split('\n')
    section_starts = extractor._find_section_starts(lines)
    
    print(f"  ✅ Found {len(section_starts)} section headers:")
    for line_num, section_type, title in section_starts:
        print(f"    Line {line_num}: {section_type} - {title}")
    
    # Step 3: Show text around each section header
    print("\nStep 3: Context around section headers...")
    for line_num, section_type, title in section_starts:
        print(f"\n--- {section_type.upper()}: {title} ---")
        start_context = max(0, line_num - 2)
        end_context = min(len(lines), line_num + 5)
        for i in range(start_context, end_context):
            marker = ">>>" if i == line_num else "   "
            print(f"{marker} {i:3d}: {lines[i]}")

def test_individual_extraction(pdf_path: str):
    """Test extraction of individual recommendations and responses"""
    extractor = EnhancedSectionExtractor()
    
    print(f"Testing individual item extraction for: {pdf_path}")
    print("=" * 60)
    
    # Test recommendations
    recommendations = extractor.extract_individual_recommendations(pdf_path)
    print(f"Individual Recommendations Found: {len(recommendations)}")
    for rec in recommendations[:5]:  # Show first 5
        print(f"  {rec['number']}: {rec['text'][:100]}...")
    
    print()
    
    # Test responses
    responses = extractor.extract_individual_responses(pdf_path)
    print(f"Individual Responses Found: {len(responses)}")
    for resp in responses[:5]:  # Show first 5
        print(f"  Rec {resp['recommendation_number']} ({resp['response_type']}): {resp['response_text'][:100]}...")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if '--debug' in sys.argv:
            debug_section_detection(sys.argv[1])
        elif '--individual' in sys.argv:
            test_individual_extraction(sys.argv[1])
        else:
            test_extraction(sys.argv[1])
    else:
        print("Usage: python enhanced_section_extractor.py <pdf_file_path> [--debug|--individual]")
        print("Example: python enhanced_section_extractor.py 'Government_Response_to_IBI.pdf'")
        print("Debug:   python enhanced_section_extractor.py 'Government_Response_to_IBI.pdf' --debug")
        print("Individual: python enhanced_section_extractor.py 'Government_Response_to_IBI.pdf' --individual")
