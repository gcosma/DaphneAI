# ===============================================
# FILE: modules/enhanced_section_extractor.py
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
    Enhanced document processor that extracts ONLY recommendations and responses sections
    with accurate page number tracking.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Section identifiers for recommendations
        self.recommendation_section_patterns = [
            # Standard recommendation sections
            r'(?i)^(?:\d+\.?\s*)?(?:recommendations?|conclusions and recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:summary of recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:key recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:main recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:principal recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:findings and recommendations?)\s*$',
            
            # UK Government specific patterns
            r'(?i)^(?:\d+\.?\s*)?(?:coroner\'s concerns?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:matters? of concern)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:the matters? of concern)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:concerns? identified)\s*$',
            
            # Inquiry specific patterns
            r'(?i)^(?:\d+\.?\s*)?(?:inquiry recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:report recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:panel recommendations?)\s*$'
        ]
        
        # Section identifiers for responses
        self.response_section_patterns = [
            # Government responses
            r'(?i)^(?:\d+\.?\s*)?(?:government response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:official response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:departmental response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:ministry response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:response to recommendations?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:action should be taken)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:your response)\s*$',
            
            # Organization responses
            r'(?i)^(?:\d+\.?\s*)?(?:trust response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:hospital response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:organization response)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:provider response)\s*$',
            
            # Implementation sections
            r'(?i)^(?:\d+\.?\s*)?(?:implementation plan)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:action plan)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:next steps?)\s*$'
        ]
        
        # Section end markers
        self.section_end_patterns = [
            # Document structure markers
            r'(?i)^(?:\d+\.?\s*)?(?:conclusions?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:summary)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:appendix|appendices)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:references?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:bibliography)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:glossary)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:index)\s*$',
            
            # Administrative sections
            r'(?i)^(?:\d+\.?\s*)?(?:copies?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:signed:?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:dated this)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:contact details?)\s*$',
            r'(?i)^(?:\d+\.?\s*)?(?:further information)\s*$'
        ]

    def extract_sections_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract only recommendations and responses sections with page numbers
        """
        try:
            # Try both extraction methods
            pdfplumber_result = self._extract_sections_with_pdfplumber(pdf_path)
            pymupdf_result = self._extract_sections_with_pymupdf(pdf_path)
            
            # Combine results, preferring pdfplumber for better text extraction
            final_result = self._merge_extraction_results(pdfplumber_result, pymupdf_result)
            
            # Add metadata
            final_result['extraction_metadata'] = {
                'file_path': pdf_path,
                'filename': Path(pdf_path).name,
                'extraction_date': datetime.now().isoformat(),
                'extractor_version': 'Enhanced_v2.0',
                'total_sections_found': len(final_result.get('sections', []))
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error extracting sections from {pdf_path}: {e}")
            return {
                'sections': [],
                'extraction_metadata': {
                    'error': str(e),
                    'extraction_date': datetime.now().isoformat()
                }
            }

    def _extract_sections_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract sections using pdfplumber with detailed page tracking"""
        sections = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                    
                    # Split page into lines for section detection
                    lines = page_text.split('\n')
                    current_section = None
                    section_content = []
                    
                    for line_num, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check if this line starts a new section we care about
                        section_type = self._identify_section_type(line)
                        
                        if section_type:
                            # Save previous section if it exists
                            if current_section and section_content:
                                sections.append({
                                    'type': current_section['type'],
                                    'title': current_section['title'],
                                    'content': '\n'.join(section_content).strip(),
                                    'page_start': current_section['page_start'],
                                    'page_end': page_num,
                                    'extraction_method': 'pdfplumber'
                                })
                            
                            # Start new section
                            current_section = {
                                'type': section_type,
                                'title': line,
                                'page_start': page_num
                            }
                            section_content = []
                        
                        elif current_section:
                            # Check if this line ends the current section
                            if self._is_section_end(line):
                                # End current section
                                if section_content:
                                    sections.append({
                                        'type': current_section['type'],
                                        'title': current_section['title'],
                                        'content': '\n'.join(section_content).strip(),
                                        'page_start': current_section['page_start'],
                                        'page_end': page_num,
                                        'extraction_method': 'pdfplumber'
                                    })
                                current_section = None
                                section_content = []
                            else:
                                # Add to current section content
                                section_content.append(line)
                
                # Handle final section if document ends
                if current_section and section_content:
                    sections.append({
                        'type': current_section['type'],
                        'title': current_section['title'],
                        'content': '\n'.join(section_content).strip(),
                        'page_start': current_section['page_start'],
                        'page_end': page_num,
                        'extraction_method': 'pdfplumber'
                    })
            
            return {'sections': sections}
            
        except Exception as e:
            self.logger.error(f"Error with pdfplumber section extraction: {e}")
            return {'sections': []}

    def _extract_sections_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract sections using PyMuPDF as backup method"""
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                if not page_text:
                    continue
                
                lines = page_text.split('\n')
                current_section = None
                section_content = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    section_type = self._identify_section_type(line)
                    
                    if section_type:
                        # Save previous section
                        if current_section and section_content:
                            sections.append({
                                'type': current_section['type'],
                                'title': current_section['title'],
                                'content': '\n'.join(section_content).strip(),
                                'page_start': current_section['page_start'],
                                'page_end': page_num + 1,
                                'extraction_method': 'pymupdf'
                            })
                        
                        # Start new section
                        current_section = {
                            'type': section_type,
                            'title': line,
                            'page_start': page_num + 1
                        }
                        section_content = []
                    
                    elif current_section:
                        if self._is_section_end(line):
                            # End current section
                            if section_content:
                                sections.append({
                                    'type': current_section['type'],
                                    'title': current_section['title'],
                                    'content': '\n'.join(section_content).strip(),
                                    'page_start': current_section['page_start'],
                                    'page_end': page_num + 1,
                                    'extraction_method': 'pymupdf'
                                })
                            current_section = None
                            section_content = []
                        else:
                            section_content.append(line)
                
                # Handle final section
                if current_section and section_content:
                    sections.append({
                        'type': current_section['type'],
                        'title': current_section['title'],
                        'content': '\n'.join(section_content).strip(),
                        'page_start': current_section['page_start'],
                        'page_end': page_num + 1,
                        'extraction_method': 'pymupdf'
                    })
            
            doc.close()
            return {'sections': sections}
            
        except Exception as e:
            self.logger.error(f"Error with pymupdf section extraction: {e}")
            return {'sections': []}

    def _identify_section_type(self, line: str) -> Optional[str]:
        """Identify if a line starts a recommendations or responses section"""
        # Check recommendations patterns
        for pattern in self.recommendation_section_patterns:
            if re.match(pattern, line.strip()):
                return 'recommendations'
        
        # Check responses patterns
        for pattern in self.response_section_patterns:
            if re.match(pattern, line.strip()):
                return 'responses'
        
        return None

    def _is_section_end(self, line: str) -> bool:
        """Check if a line indicates the end of a section"""
        for pattern in self.section_end_patterns:
            if re.match(pattern, line.strip()):
                return True
        return False

    def _merge_extraction_results(self, pdfplumber_result: Dict, pymupdf_result: Dict) -> Dict[str, Any]:
        """Merge results from both extraction methods, preferring pdfplumber"""
        # Start with pdfplumber results
        final_sections = pdfplumber_result.get('sections', [])
        pymupdf_sections = pymupdf_result.get('sections', [])
        
        # If pdfplumber found nothing, use pymupdf results
        if not final_sections and pymupdf_sections:
            final_sections = pymupdf_sections
        
        # Remove duplicates and ensure quality
        final_sections = self._deduplicate_sections(final_sections)
        
        return {
            'sections': final_sections,
            'extraction_summary': {
                'pdfplumber_sections': len(pdfplumber_result.get('sections', [])),
                'pymupdf_sections': len(pymupdf_sections),
                'final_sections': len(final_sections)
            }
        }

    def _deduplicate_sections(self, sections: List[Dict]) -> List[Dict]:
        """Remove duplicate sections and improve quality"""
        if not sections:
            return []
        
        # Sort by page start for consistency
        sections.sort(key=lambda x: x.get('page_start', 0))
        
        # Remove very short sections (likely false positives)
        quality_sections = []
        for section in sections:
            content = section.get('content', '')
            if len(content.strip()) > 50:  # Minimum content length
                quality_sections.append(section)
        
        # Remove exact duplicates based on content similarity
        unique_sections = []
        for section in quality_sections:
            is_duplicate = False
            for existing in unique_sections:
                if self._sections_are_similar(section['content'], existing['content']):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sections.append(section)
        
        return unique_sections

    def _sections_are_similar(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """Check if two sections are similar enough to be considered duplicates"""
        if not content1 or not content2:
            return False
        
        # Simple similarity check based on length and common words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold

    def extract_and_save_sections(self, pdf_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract sections and optionally save results to file
        """
        result = self.extract_sections_from_pdf(pdf_path)
        
        if output_path:
            try:
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Results saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving results: {e}")
        
        return result

    def batch_extract_sections(self, pdf_directory: str) -> Dict[str, Any]:
        """
        Extract sections from multiple PDFs in a directory
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise ValueError(f"Directory {pdf_directory} does not exist")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_directory}")
        
        batch_results = {
            'processed_files': {},
            'summary': {
                'total_files': len(pdf_files),
                'successful_extractions': 0,
                'failed_extractions': 0,
                'total_sections_found': 0
            }
        }
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing {pdf_file.name}")
            
            try:
                result = self.extract_sections_from_pdf(str(pdf_file))
                sections = result.get('sections', [])
                
                batch_results['processed_files'][pdf_file.name] = result
                batch_results['summary']['successful_extractions'] += 1
                batch_results['summary']['total_sections_found'] += len(sections)
                
                self.logger.info(f"✓ {pdf_file.name}: Found {len(sections)} sections")
                
            except Exception as e:
                self.logger.error(f"✗ {pdf_file.name}: {e}")
                batch_results['processed_files'][pdf_file.name] = {
                    'sections': [],
                    'extraction_metadata': {'error': str(e)}
                }
                batch_results['summary']['failed_extractions'] += 1
        
        # Add batch summary
        batch_results['summary']['extraction_date'] = datetime.now().isoformat()
        
        self.logger.info(f"Batch processing complete: {batch_results['summary']}")
        return batch_results


# Example usage and testing functions
def main():
    """Example usage of the enhanced section extractor"""
    import sys
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Extract recommendations and responses sections from PDFs')
    parser.add_argument('input_path', help='PDF file or directory path')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--batch', '-b', action='store_true', help='Process directory of PDFs')
    
    args = parser.parse_args()
    
    extractor = EnhancedSectionExtractor()
    
    try:
        if args.batch:
            # Batch process directory
            results = extractor.batch_extract_sections(args.input_path)
            print(f"\n📊 Batch Processing Results:")
            print(f"Files processed: {results['summary']['successful_extractions']}/{results['summary']['total_files']}")
            print(f"Total sections found: {results['summary']['total_sections_found']}")
            
            # Show details for each file
            for filename, file_result in results['processed_files'].items():
                sections = file_result.get('sections', [])
                if sections:
                    print(f"\n📄 {filename}:")
                    for section in sections:
                        print(f"  - {section['type'].title()}: {section['title']} (Pages {section['page_start']}-{section['page_end']})")
                else:
                    error = file_result.get('extraction_metadata', {}).get('error', 'No sections found')
                    print(f"\n❌ {filename}: {error}")
            
        else:
            # Process single file
            results = extractor.extract_sections_from_pdf(args.input_path)
            sections = results.get('sections', [])
            
            print(f"\n📄 Document: {Path(args.input_path).name}")
            print(f"Sections found: {len(sections)}")
            
            for section in sections:
                print(f"\n📋 {section['type'].title()} Section:")
                print(f"  Title: {section['title']}")
                print(f"  Pages: {section['page_start']}-{section['page_end']}")
                print(f"  Content length: {len(section['content'])} characters")
                print(f"  Preview: {section['content'][:200]}...")
        
        # Save results if output path specified
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to {args.output}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
