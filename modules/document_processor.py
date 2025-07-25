# ===============================================
# FILE: modules/document_processor.py (UPDATED VERSION)
# ===============================================
from pathlib import Path
import pdfplumber
import fitz
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import gc
import re

class DocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Import the enhanced section extractor
        try:
            from enhanced_section_extractor import EnhancedSectionExtractor
            self.section_extractor = EnhancedSectionExtractor()
        except ImportError:
            self.logger.warning("Enhanced section extractor not available, using basic extraction")
            self.section_extractor = None
    
    def extract_text_from_pdf(self, pdf_path: str, extract_sections_only: bool = True) -> Optional[Dict[str, Any]]:
        """
        Extract text from PDF with option to extract only recommendations/responses sections
        
        Args:
            pdf_path: Path to PDF file
            extract_sections_only: If True, extracts only recommendations and responses sections
                                  If False, extracts full document (legacy behavior)
        """
        try:
            if extract_sections_only and self.section_extractor:
                # Use enhanced section extraction
                result = self.section_extractor.extract_sections_from_pdf(pdf_path)
                
                # Convert sections to text content with page info
                text_content = self._sections_to_text(result.get('sections', []))
                metadata = self._extract_metadata_with_pymupdf(pdf_path)
                
                # Add section metadata
                metadata['sections_found'] = len(result.get('sections', []))
                metadata['extraction_method'] = 'enhanced_sections'
                metadata['sections_metadata'] = result.get('extraction_metadata', {})
                
                return {
                    "content": text_content,
                    "metadata": metadata,
                    "source": Path(pdf_path).name,
                    "sections": result.get('sections', []),
                    "extraction_type": "sections_only"
                }
            else:
                # Use legacy full document extraction
                text_content = self._extract_with_pdfplumber(pdf_path)
                if not text_content:
                    self.logger.warning(f"No text from pdfplumber for {pdf_path}, trying pymupdf")
                    text_content = self._extract_with_pymupdf(pdf_path)
                
                metadata = self._extract_metadata_with_pymupdf(pdf_path)
                metadata['extraction_method'] = 'full_document'
                
                if not text_content:
                    self.logger.warning(f"No text extracted from {pdf_path} by any method")
                    text_content = ""
                
                return {
                    "content": text_content,
                    "metadata": metadata,
                    "source": Path(pdf_path).name,
                    "extraction_type": "full_document"
                }
                
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
    def _sections_to_text(self, sections: List[Dict[str, Any]]) -> str:
        """
        Convert extracted sections to formatted text with page numbers
        """
        if not sections:
            return ""
        
        formatted_text = []
        
        for section in sections:
            section_type = section.get('type', 'unknown')
            title = section.get('title', '')
            content = section.get('content', '')
            page_start = section.get('page_start', 'unknown')
            page_end = section.get('page_end', 'unknown')
            
            # Format section header with page info
            if page_start == page_end:
                page_info = f"(Page {page_start})"
            else:
                page_info = f"(Pages {page_start}-{page_end})"
            
            section_header = f"\n=== {section_type.upper()} SECTION {page_info} ===\n"
            section_header += f"Title: {title}\n"
            section_header += f"{'='*50}\n"
            
            formatted_text.append(section_header)
            formatted_text.append(content)
            formatted_text.append("\n" + "="*50 + "\n")
        
        return "\n".join(formatted_text)
    
    def extract_sections_with_details(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract sections with detailed information including page numbers and content
        Returns structured data about each section found
        """
        if not self.section_extractor:
            self.logger.error("Enhanced section extractor not available")
            return {'sections': [], 'error': 'Section extractor not available'}
        
        try:
            result = self.section_extractor.extract_sections_from_pdf(pdf_path)
            sections = result.get('sections', [])
            
            # Add additional processing for each section
            enhanced_sections = []
            for section in sections:
                enhanced_section = section.copy()
                
                # Add content statistics
                content = section.get('content', '')
                enhanced_section['content_stats'] = {
                    'character_count': len(content),
                    'word_count': len(content.split()) if content else 0,
                    'line_count': len(content.split('\n')) if content else 0,
                    'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]) if content else 0
                }
                
                # Extract key information from content
                enhanced_section['key_info'] = self._extract_key_information(content)
                
                enhanced_sections.append(enhanced_section)
            
            return {
                'sections': enhanced_sections,
                'extraction_metadata': result.get('extraction_metadata', {}),
                'summary': {
                    'total_sections': len(enhanced_sections),
                    'recommendations_sections': len([s for s in enhanced_sections if s['type'] == 'recommendations']),
                    'responses_sections': len([s for s in enhanced_sections if s['type'] == 'responses']),
                    'total_pages_covered': self._calculate_pages_covered(enhanced_sections)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting sections with details: {e}")
            return {'sections': [], 'error': str(e)}
    
    def _extract_key_information(self, content: str) -> Dict[str, Any]:
        """Extract key information from section content"""
        if not content:
            return {}
        
        key_info = {}
        
        # Count numbered items (recommendations/responses)
        numbered_items = len(re.findall(r'^\d+\.', content, re.MULTILINE))
        key_info['numbered_items_count'] = numbered_items
        
        # Count lettered sub-items
        lettered_items = len(re.findall(r'^\([a-z]\)', content, re.MULTILINE))
        key_info['lettered_items_count'] = lettered_items
        
        # Extract reference numbers if present
        refs = re.findall(r'(?:ref|reference)\.?\s*:?\s*([-\w\d]+)', content, re.IGNORECASE)
        key_info['references'] = list(set(refs)) if refs else []
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', content)
        key_info['dates_mentioned'] = list(set(dates)) if dates else []
        
        # Check for specific keywords
        keywords = {
            'should': len(re.findall(r'\bshould\b', content, re.IGNORECASE)),
            'must': len(re.findall(r'\bmust\b', content, re.IGNORECASE)),
            'recommend': len(re.findall(r'\brecommend', content, re.IGNORECASE)),
            'accepted': len(re.findall(r'\baccepted\b', content, re.IGNORECASE)),
            'implemented': len(re.findall(r'\bimplemented?\b', content, re.IGNORECASE)),
            'rejected': len(re.findall(r'\brejected?\b', content, re.IGNORECASE)),
            'action': len(re.findall(r'\baction\b', content, re.IGNORECASE))
        }
        key_info['keyword_counts'] = keywords
        
        return key_info
    
    def _calculate_pages_covered(self, sections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate total pages covered by all sections"""
        if not sections:
            return {'total_pages': 0, 'page_range': ''}
        
        all_pages = set()
        for section in sections:
            start = section.get('page_start')
            end = section.get('page_end')
            if start and end:
                try:
                    start_num = int(start)
                    end_num = int(end)
                    all_pages.update(range(start_num, end_num + 1))
                except (ValueError, TypeError):
                    continue
        
        if all_pages:
            sorted_pages = sorted(all_pages)
            return {
                'total_pages': len(all_pages),
                'page_range': f"{min(sorted_pages)}-{max(sorted_pages)}"
            }
        else:
            return {'total_pages': 0, 'page_range': ''}
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber with page-level logging"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        self.logger.warning(f"No text extracted on page {i+1} of {pdf_path}")
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error with pdfplumber extraction: {e}")
            return ""
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (fitz)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error with pymupdf extraction: {e}")
            return ""
    
    def _extract_metadata_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            
            metadata.update({
                "page_count": doc.page_count,
                "file_size": Path(pdf_path).stat().st_size,
                "processed_at": datetime.now().isoformat()
            })
            
            doc.close()
            return metadata
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return {"processed_at": datetime.now().isoformat()}
    
    def batch_process_documents(self, pdf_directory: str, extract_sections_only: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs in directory
        
        Args:
            pdf_directory: Directory containing PDF files
            extract_sections_only: If True, extracts only recommendations/responses sections
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            self.logger.error(f"Directory {pdf_directory} does not exist")
            return []
        
        processed_docs = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing {pdf_file.name}")
            
            try:
                doc_data = self.extract_text_from_pdf(str(pdf_file), extract_sections_only)
                if doc_data:
                    processed_docs.append(doc_data)
                    
                    # Log section summary if available
                    if 'sections' in doc_data:
                        sections = doc_data['sections']
                        rec_sections = len([s for s in sections if s['type'] == 'recommendations'])
                        resp_sections = len([s for s in sections if s['type'] == 'responses'])
                        self.logger.info(f"  Found {rec_sections} recommendation sections and {resp_sections} response sections")
                else:
                    self.logger.warning(f"Failed to extract data from {pdf_file.name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
            
            # Memory cleanup
            gc.collect()
        
        self.logger.info(f"Successfully processed {len(processed_docs)} PDFs from {pdf_directory}")
        return processed_docs
    
    def get_document_summary(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get a summary of document content and structure
        """
        try:
            # Extract sections with details
            sections_result = self.extract_sections_with_details(pdf_path)
            
            # Get basic metadata
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            
            summary = {
                'filename': Path(pdf_path).name,
                'file_size_mb': round(metadata.get('file_size', 0) / (1024 * 1024), 2),
                'total_pages': metadata.get('page_count', 0),
                'sections_summary': sections_result.get('summary', {}),
                'sections_found': []
            }
            
            # Add details for each section
            for section in sections_result.get('sections', []):
                section_summary = {
                    'type': section['type'],
                    'title': section['title'],
                    'pages': f"{section['page_start']}-{section['page_end']}",
                    'word_count': section.get('content_stats', {}).get('word_count', 0),
                    'numbered_items': section.get('key_info', {}).get('numbered_items_count', 0)
                }
                summary['sections_found'].append(section_summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting document summary: {e}")
            return {'error': str(e)}
    
    def validate_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """
        Validate the quality of section extraction
        """
        try:
            sections_result = self.extract_sections_with_details(pdf_path)
            sections = sections_result.get('sections', [])
            
            validation = {
                'is_valid': True,
                'quality_score': 0.0,
                'issues': [],
                'recommendations': []
            }
            
            if not sections:
                validation['is_valid'] = False
                validation['issues'].append("No recommendations or responses sections found")
                validation['quality_score'] = 0.0
                return validation
            
            # Check section quality
            total_content_length = 0
            has_recommendations = False
            has_responses = False
            
            for section in sections:
                content = section.get('content', '')
                total_content_length += len(content)
                
                if section['type'] == 'recommendations':
                    has_recommendations = True
                elif section['type'] == 'responses':
                    has_responses = True
                
                # Check for very short sections
                if len(content) < 100:
                    validation['issues'].append(f"Short {section['type']} section (< 100 chars)")
            
            # Calculate quality score
            score = 0.0
            if has_recommendations:
                score += 0.5
            if has_responses:
                score += 0.3
            if total_content_length > 500:
                score += 0.2
            
            validation['quality_score'] = min(score, 1.0)
            
            # Provide recommendations
            if not has_recommendations:
                validation['recommendations'].append("Consider checking if document contains recommendation sections")
            if not has_responses:
                validation['recommendations'].append("No response sections found - this may be normal for some documents")
            if total_content_length < 200:
                validation['recommendations'].append("Very little content extracted - check document quality")
            
            return validation
            
        except Exception as e:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': [f"Validation error: {e}"],
                'recommendations': ["Check document format and try again"]
            }


# Utility functions for integration with existing codebase
def extract_recommendations_and_responses_only(pdf_path: str) -> Dict[str, Any]:
    """
    Convenience function to extract only recommendations and responses sections
    This replaces the old extract_text_from_pdf when you only want specific sections
    """
    processor = DocumentProcessor()
    return processor.extract_text_from_pdf(pdf_path, extract_sections_only=True)


def get_sections_with_page_numbers(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Get a list of extracted sections with page numbers
    Returns: List of sections with type, content, and page range
    """
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


# Basic logging setup (can be customized in your main app)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        processor = DocumentProcessor()
        
        print(f"Processing: {pdf_path}")
        print("="*50)
        
        # Get summary
        summary = processor.get_document_summary(pdf_path)
        print("Document Summary:")
        print(f"  File: {summary.get('filename')}")
        print(f"  Size: {summary.get('file_size_mb')} MB")
        print(f"  Pages: {summary.get('total_pages')}")
        print(f"  Sections found: {len(summary.get('sections_found', []))}")
        
        # Show sections
        for section in summary.get('sections_found', []):
            print(f"\n📋 {section['type'].title()} Section:")
            print(f"  Title: {section['title']}")
            print(f"  Pages: {section['pages']}")
            print(f"  Words: {section['word_count']}")
            print(f"  Items: {section['numbered_items']}")
        
        # Validate extraction
        validation = processor.validate_extraction(pdf_path)
        print(f"\nValidation Score: {validation['quality_score']:.2f}")
        if validation['issues']:
            print("Issues:", ", ".join(validation['issues']))
        if validation['recommendations']:
            print("Recommendations:", ", ".join(validation['recommendations']))
    else:
        print("Usage: python document_processor.py <pdf_file_path>")


# Additional utility functions for backwards compatibility
def process_document_legacy(pdf_path: str) -> Dict[str, Any]:
    """
    Legacy function for full document processing (backwards compatibility)
    """
    processor = DocumentProcessor()
    return processor.extract_text_from_pdf(pdf_path, extract_sections_only=False)


def get_document_sections_summary(pdf_path: str) -> Dict[str, Any]:
    """
    Get a quick summary of sections found in a document
    """
    processor = DocumentProcessor()
    return processor.get_document_summary(pdf_path)


def validate_document_extraction(pdf_path: str) -> bool:
    """
    Quick validation check - returns True if extraction was successful
    """
    processor = DocumentProcessor()
    validation = processor.validate_extraction(pdf_path)
    return validation.get('is_valid', False) and validation.get('quality_score', 0) > 0.5


class BatchDocumentProcessor:
    """
    Enhanced batch processor for handling multiple documents with section extraction
    """
    
    def __init__(self, extract_sections_only: bool = True):
        self.processor = DocumentProcessor()
        self.extract_sections_only = extract_sections_only
        self.logger = logging.getLogger(__name__)
    
    def process_directory(self, directory_path: str, file_pattern: str = "*.pdf") -> Dict[str, Any]:
        """
        Process all PDFs in a directory with comprehensive reporting
        
        Args:
            directory_path: Path to directory containing PDFs
            file_pattern: File pattern to match (default: "*.pdf")
        
        Returns:
            Comprehensive results including success/failure counts and section summaries
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        pdf_files = list(directory.glob(file_pattern))
        if not pdf_files:
            self.logger.warning(f"No files matching {file_pattern} found in {directory_path}")
            return {
                'processed_documents': [],
                'summary': {
                    'total_files': 0,
                    'successful': 0,
                    'failed': 0,
                    'total_sections': 0
                }
            }
        
        results = {
            'processed_documents': [],
            'failed_documents': [],
            'summary': {
                'total_files': len(pdf_files),
                'successful': 0,
                'failed': 0,
                'total_sections': 0,
                'total_recommendations_sections': 0,
                'total_responses_sections': 0,
                'processing_start': datetime.now().isoformat()
            }
        }
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing {pdf_file.name}")
            
            try:
                doc_data = self.processor.extract_text_from_pdf(
                    str(pdf_file), 
                    extract_sections_only=self.extract_sections_only
                )
                
                if doc_data:
                    # Add processing metadata
                    doc_data['processing_metadata'] = {
                        'file_path': str(pdf_file),
                        'processing_time': datetime.now().isoformat(),
                        'extraction_successful': True
                    }
                    
                    results['processed_documents'].append(doc_data)
                    results['summary']['successful'] += 1
                    
                    # Count sections if available
                    if 'sections' in doc_data:
                        sections = doc_data['sections']
                        results['summary']['total_sections'] += len(sections)
                        results['summary']['total_recommendations_sections'] += len([
                            s for s in sections if s['type'] == 'recommendations'
                        ])
                        results['summary']['total_responses_sections'] += len([
                            s for s in sections if s['type'] == 'responses'
                        ])
                        
                        self.logger.info(f"  ✓ Found {len(sections)} relevant sections")
                    else:
                        self.logger.info(f"  ✓ Extracted full document")
                else:
                    raise Exception("No data extracted")
                    
            except Exception as e:
                self.logger.error(f"  ✗ Failed to process {pdf_file.name}: {e}")
                results['failed_documents'].append({
                    'filename': pdf_file.name,
                    'file_path': str(pdf_file),
                    'error': str(e),
                    'processing_time': datetime.now().isoformat()
                })
                results['summary']['failed'] += 1
        
        # Add completion metadata
        results['summary']['processing_end'] = datetime.now().isoformat()
        results['summary']['success_rate'] = (
            results['summary']['successful'] / results['summary']['total_files'] * 100
            if results['summary']['total_files'] > 0 else 0
        )
        
        self.logger.info(f"Batch processing complete: {results['summary']['successful']}/{results['summary']['total_files']} successful")
        return results
    
    def export_results(self, results: Dict[str, Any], output_path: str, format: str = 'json') -> bool:
        """
        Export processing results to file
        
        Args:
            results: Results from process_directory
            output_path: Path for output file
            format: Export format ('json', 'csv', or 'excel')
        """
        try:
            output_file = Path(output_path)
            
            if format.lower() == 'json':
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == 'csv':
                import pandas as pd
                
                # Create summary DataFrame
                summary_data = []
                for doc in results['processed_documents']:
                    sections = doc.get('sections', [])
                    summary_data.append({
                        'filename': doc['source'],
                        'total_sections': len(sections),
                        'recommendations_sections': len([s for s in sections if s['type'] == 'recommendations']),
                        'responses_sections': len([s for s in sections if s['type'] == 'responses']),
                        'extraction_type': doc.get('extraction_type', 'unknown')
                    })
                
                df = pd.DataFrame(summary_data)
                df.to_csv(output_file, index=False)
                
            elif format.lower() == 'excel':
                import pandas as pd
                
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = []
                    sections_data = []
                    
                    for doc in results['processed_documents']:
                        sections = doc.get('sections', [])
                        summary_data.append({
                            'filename': doc['source'],
                            'total_sections': len(sections),
                            'recommendations_sections': len([s for s in sections if s['type'] == 'recommendations']),
                            'responses_sections': len([s for s in sections if s['type'] == 'responses']),
                            'total_pages': doc.get('metadata', {}).get('page_count', 0),
                            'file_size_mb': round(doc.get('metadata', {}).get('file_size', 0) / (1024*1024), 2)
                        })
                        
                        # Sections details
                        for section in sections:
                            sections_data.append({
                                'filename': doc['source'],
                                'section_type': section['type'],
                                'section_title': section['title'],
                                'page_start': section['page_start'],
                                'page_end': section['page_end'],
                                'word_count': section.get('content_stats', {}).get('word_count', 0),
                                'numbered_items': section.get('key_info', {}).get('numbered_items_count', 0)
                            })
                    
                    # Write sheets
                    pd.DataFrame(summary_data).to_sheet(writer, sheet_name='Document_Summary', index=False)
                    pd.DataFrame(sections_data).to_sheet(writer, sheet_name='Sections_Detail', index=False)
                    pd.DataFrame([results['summary']]).to_sheet(writer, sheet_name='Processing_Summary', index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Results exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return False


# Configuration and settings
class DocumentProcessorConfig:
    """Configuration class for document processor settings"""
    
    def __init__(self):
        # Default settings
        self.extract_sections_only = True
        self.validate_extraction = True
        self.min_section_length = 50  # Minimum characters for valid section
        self.max_file_size_mb = 100
        
        # Logging settings
        self.log_level = logging.INFO
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Section extraction settings
        self.section_patterns_custom = []  # Additional custom patterns
        self.section_end_markers_custom = []  # Additional end markers
    
    def setup_logging(self):
        """Setup logging with configured settings"""
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('document_processor.log')
            ]
        )
    
    def validate_file_size(self, file_path: str) -> bool:
        """Check if file size is within limits"""
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        return file_size_mb <= self.max_file_size_mb


# Error handling and custom exceptions
class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

class SectionExtractionError(DocumentProcessingError):
    """Exception raised when section extraction fails"""
    pass

class ValidationError(DocumentProcessingError):
    """Exception raised when document validation fails"""
    pass


# Factory function for easy instantiation
def create_document_processor(config: Optional[DocumentProcessorConfig] = None) -> DocumentProcessor:
    """
    Factory function to create document processor with optional configuration
    """
    if config:
        config.setup_logging()
    
    return DocumentProcessor()


# Example usage and testing
def test_section_extraction():
    """Test function to verify section extraction works correctly"""
    import tempfile
    import os
    
    # This would be used for testing with actual PDF files
    test_pdf_path = "test_document.pdf"  # Replace with actual test file
    
    if os.path.exists(test_pdf_path):
        processor = DocumentProcessor()
        
        print("Testing section extraction...")
        result = processor.extract_text_from_pdf(test_pdf_path, extract_sections_only=True)
        
        if result and result.get('sections'):
            print(f"✅ Success! Found {len(result['sections'])} sections:")
            for section in result['sections']:
                print(f"  - {section['type']}: {section['title']} (Pages {section['page_start']}-{section['page_end']})")
        else:
            print("❌ No sections found or extraction failed")
            
        # Test validation
        validation = processor.validate_extraction(test_pdf_path)
        print(f"Validation score: {validation['quality_score']:.2f}")
        
    else:
        print(f"Test file {test_pdf_path} not found")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            sys.exit(1)
        
        processor = DocumentProcessor()
        
        print(f"Processing: {pdf_path}")
        print("="*50)
        
        try:
            # Get summary
            summary = processor.get_document_summary(pdf_path)
            print("📊 Document Summary:")
            print(f"  File: {summary.get('filename', 'Unknown')}")
            print(f"  Size: {summary.get('file_size_mb', 0)} MB")
            print(f"  Pages: {summary.get('total_pages', 0)}")
            print(f"  Sections found: {len(summary.get('sections_found', []))}")
            
            # Show sections with details
            sections_found = summary.get('sections_found', [])
            if sections_found:
                print("\n📋 Sections Found:")
                for i, section in enumerate(sections_found, 1):
                    print(f"  {i}. {section['type'].title()} Section:")
                    print(f"     Title: {section['title']}")
                    print(f"     Pages: {section['pages']}")
                    print(f"     Words: {section['word_count']:,}")
                    print(f"     Numbered items: {section['numbered_items']}")
            else:
                print("\n⚠️  No recommendations or responses sections found")
            
            # Validate extraction
            print("\n🔍 Validation Results:")
            validation = processor.validate_extraction(pdf_path)
            print(f"  Quality Score: {validation['quality_score']:.2f}/1.0")
            print(f"  Valid Extraction: {'✅ Yes' if validation['is_valid'] else '❌ No'}")
            
            if validation['issues']:
                print("  Issues found:")
                for issue in validation['issues']:
                    print(f"    - {issue}")
            
            if validation['recommendations']:
                print("  Recommendations:")
                for rec in validation['recommendations']:
                    print(f"    - {rec}")
            
            # Show extraction details
            print(f"\n📈 Processing Summary:")
            sections_summary = summary.get('sections_summary', {})
            print(f"  Total sections: {sections_summary.get('total_sections', 0)}")
            print(f"  Recommendations sections: {sections_summary.get('recommendations_sections', 0)}")
            print(f"  Responses sections: {sections_summary.get('responses_sections', 0)}")
            
            pages_info = sections_summary.get('total_pages_covered', {})
            if pages_info.get('total_pages', 0) > 0:
                print(f"  Pages covered: {pages_info['total_pages']} ({pages_info.get('page_range', 'unknown range')})")
                
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("📖 Usage Examples:")
        print("  python document_processor.py <pdf_file_path>")
        print("  python document_processor.py report.pdf")
        print("\n🧪 Run test:")
        print("  python -c 'from document_processor import test_section_extraction; test_section_extraction()'")
        print("\n📚 Import in your code:")
        print("  from document_processor import DocumentProcessor, extract_recommendations_and_responses_only")
