# ===============================================
# FILE: modules/document_processor.py (FIXED VERSION)
# ===============================================
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import gc
import re
import tempfile
import os

class DocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Import the enhanced section extractor if available
        try:
            from .enhanced_section_extractor import EnhancedSectionExtractor
            self.section_extractor = EnhancedSectionExtractor()
            self.logger.info("Enhanced section extractor loaded successfully")
        except ImportError:
            self.logger.warning("Enhanced section extractor not available, using basic extraction")
            self.section_extractor = None
    
    def extract_text_from_pdf(self, pdf_path: str, extract_sections_only: bool = True) -> Optional[Dict[str, Any]]:
        """
        Extract text from PDF with robust fallback methods
        
        Args:
            pdf_path: Path to PDF file  
            extract_sections_only: If True, extracts only recommendations and responses sections
                                  If False, extracts full document
        """
        try:
            self.logger.info(f"Starting text extraction from: {pdf_path}")
            
            # Validate file exists and is readable
            if not os.path.exists(pdf_path):
                self.logger.error(f"File does not exist: {pdf_path}")
                return None
                
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                self.logger.error(f"File is empty: {pdf_path}")
                return None
                
            self.logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            if extract_sections_only and self.section_extractor:
                # Use enhanced section extraction
                self.logger.info("Using enhanced section extraction")
                result = self.section_extractor.extract_sections_from_pdf(pdf_path)
                
                if result and result.get('sections'):
                    # Convert sections to text content
                    text_content = self._sections_to_text(result.get('sections', []))
                    metadata = self._extract_metadata_with_pymupdf(pdf_path)
                    
                    metadata['sections_found'] = len(result.get('sections', []))
                    metadata['extraction_method'] = 'enhanced_sections'
                    metadata['sections_metadata'] = result.get('extraction_metadata', {})
                    
                    return {
                        "text": text_content,
                        "content": text_content,  # For backwards compatibility
                        "metadata": metadata,
                        "source": Path(pdf_path).name,
                        "filename": Path(pdf_path).name,
                        "sections": result.get('sections', []),
                        "extraction_type": "sections_only"
                    }
                else:
                    self.logger.warning("No sections found with enhanced extractor, falling back to full document")
            
            # Use full document extraction with multiple fallback methods
            self.logger.info("Attempting full document extraction")
            text_content = self._extract_text_with_fallbacks(pdf_path)
            
            if not text_content or len(text_content.strip()) < 50:
                self.logger.error(f"Insufficient text extracted from {pdf_path} (length: {len(text_content) if text_content else 0})")
                return None
            
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            metadata['extraction_method'] = 'full_document_fallback'
            metadata['text_length'] = len(text_content)
            
            self.logger.info(f"Successfully extracted {len(text_content)} characters")
            
            return {
                "text": text_content,
                "content": text_content,  # For backwards compatibility
                "metadata": metadata,
                "source": Path(pdf_path).name,
                "filename": Path(pdf_path).name,
                "extraction_type": "full_document"
            }
                
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
            return None
    
    def _extract_text_with_fallbacks(self, pdf_path: str) -> str:
        """
        Extract text using multiple fallback methods
        """
        methods = [
            ("pdfplumber", self._extract_with_pdfplumber),
            ("pymupdf", self._extract_with_pymupdf),
            ("pymupdf_detailed", self._extract_with_pymupdf_detailed),
            ("pdfplumber_detailed", self._extract_with_pdfplumber_detailed)
        ]
        
        for method_name, method_func in methods:
            try:
                self.logger.info(f"Trying extraction method: {method_name}")
                text = method_func(pdf_path)
                
                if text and len(text.strip()) > 50:
                    self.logger.info(f"Success with {method_name}: {len(text)} characters extracted")
                    return text.strip()
                else:
                    self.logger.warning(f"{method_name} extracted insufficient text: {len(text) if text else 0} characters")
                    
            except Exception as e:
                self.logger.warning(f"Method {method_name} failed: {str(e)}")
                continue
        
        self.logger.error("All extraction methods failed")
        return ""
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (basic)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {i+1}: {e}")
                        continue
            
            self.logger.info(f"pdfplumber extracted {len(text)} characters from {pdf_path}")
            return text.strip()
        except Exception as e:
            self.logger.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    def _extract_with_pdfplumber_detailed(self, pdf_path: str) -> str:
        """Extract text using pdfplumber with detailed settings"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        # Try different extraction methods
                        page_text = page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=True,
                            x_density=7.25,
                            y_density=13
                        )
                        
                        if not page_text:
                            # Try extracting from tables if regular text fails
                            tables = page.extract_tables()
                            if tables:
                                table_text = ""
                                for table in tables:
                                    for row in table:
                                        if row:
                                            table_text += " ".join([cell or "" for cell in row]) + "\n"
                                page_text = table_text
                        
                        if page_text:
                            text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {i+1} with detailed method: {e}")
                        continue
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"pdfplumber detailed extraction failed: {e}")
            return ""
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (basic)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text and page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            gc.collect()
            
            self.logger.info(f"pymupdf extracted {len(text)} characters")
            return text.strip()
        except Exception as e:
            self.logger.error(f"pymupdf extraction failed: {e}")
            return ""
    
    def _extract_with_pymupdf_detailed(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF with detailed options"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    
                    # Try different text extraction methods
                    methods = [
                        ("dict", lambda p: p.get_text("dict")),
                        ("blocks", lambda p: p.get_text("blocks")),
                        ("text", lambda p: p.get_text()),
                        ("html", lambda p: p.get_text("html")),
                    ]
                    
                    page_text = ""
                    for method_name, method_func in methods:
                        try:
                            result = method_func(page)
                            
                            if method_name == "dict":
                                # Extract text from dict structure
                                if isinstance(result, dict) and "blocks" in result:
                                    for block in result["blocks"]:
                                        if "lines" in block:
                                            for line in block["lines"]:
                                                if "spans" in line:
                                                    for span in line["spans"]:
                                                        if "text" in span:
                                                            page_text += span["text"]
                                                    page_text += "\n"
                            elif method_name == "blocks":
                                # Extract from blocks
                                if isinstance(result, list):
                                    for block in result:
                                        if len(block) > 4:  # Block has text
                                            page_text += block[4] + "\n"
                            elif method_name in ["text", "html"]:
                                # Direct text
                                if isinstance(result, str) and result.strip():
                                    page_text = result
                            
                            if page_text and len(page_text.strip()) > 10:
                                break  # Use first successful method
                                
                        except Exception as e:
                            self.logger.warning(f"PyMuPDF method {method_name} failed on page {page_num + 1}: {e}")
                            continue
                    
                    if page_text and page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        
                except Exception as e:
                    self.logger.warning(f"Error with detailed extraction on page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            gc.collect()
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"pymupdf detailed extraction failed: {e}")
            return ""
    
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
    
    def _sections_to_text(self, sections: List[Dict[str, Any]]) -> str:
        """Convert extracted sections to formatted text with page numbers"""
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
            if title:
                section_header += f"Title: {title}\n"
            section_header += f"{'='*50}\n"
            
            formatted_text.append(section_header)
            formatted_text.append(content)
            formatted_text.append("\n" + "="*50 + "\n")
        
        return "\n".join(formatted_text)
    
    def validate_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """Validate that extraction was successful"""
        try:
            # Try to extract text
            result = self.extract_text_from_pdf(pdf_path, extract_sections_only=False)
            
            if not result:
                return {
                    'is_valid': False,
                    'quality_score': 0.0,
                    'issues': ['No text could be extracted'],
                    'recommendations': ['Check if file is corrupted', 'Verify file is a text-based PDF']
                }
            
            text = result.get('text', '') or result.get('content', '')
            text_length = len(text.strip()) if text else 0
            
            # Calculate quality score
            quality_score = min(1.0, text_length / 1000.0)  # Normalize to 1000 chars = 1.0 score
            
            issues = []
            recommendations = []
            
            if text_length < 100:
                issues.append('Very little text extracted')
                recommendations.append('File may be scanned image - consider OCR')
            elif text_length < 500:
                issues.append('Limited text extracted')
                recommendations.append('File may have formatting issues')
            
            # Check for common issues
            if text and len(set(text.replace(' ', '').replace('\n', ''))) < 10:
                issues.append('Text appears to be corrupted or repetitive')
                recommendations.append('Check source file quality')
            
            return {
                'is_valid': text_length > 50,
                'quality_score': quality_score,
                'text_length': text_length,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'issues': [f'Validation failed: {str(e)}'],
                'recommendations': ['Check file permissions and format']
            }
    
    def get_document_summary(self, pdf_path: str) -> Dict[str, Any]:
        """Get a summary of the document"""
        try:
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            validation = self.validate_extraction(pdf_path)
            
            return {
                'filename': Path(pdf_path).name,
                'file_size_mb': metadata.get('file_size_mb', 0),
                'total_pages': metadata.get('page_count', 0),
                'is_valid': validation.get('is_valid', False),
                'quality_score': validation.get('quality_score', 0),
                'text_length': validation.get('text_length', 0),
                'issues': validation.get('issues', []),
                'recommendations': validation.get('recommendations', [])
            }
        except Exception as e:
            self.logger.error(f"Error getting document summary: {e}")
            return {
                'filename': Path(pdf_path).name,
                'error': str(e)
            }


# Utility functions for backwards compatibility
def extract_text_from_pdf(pdf_path: str, extract_sections_only: bool = True) -> Optional[Dict[str, Any]]:
    """Standalone function for text extraction"""
    processor = DocumentProcessor()
    return processor.extract_text_from_pdf(pdf_path, extract_sections_only)


def validate_pdf_file(pdf_path: str) -> bool:
    """Quick validation check"""
    processor = DocumentProcessor()
    validation = processor.validate_extraction(pdf_path)
    return validation.get('is_valid', False)


# Test function
def test_extraction(pdf_path: str):
    """Test extraction with a PDF file"""
    processor = DocumentProcessor()
    
    print(f"Testing extraction for: {pdf_path}")
    print("=" * 50)
    
    # Get summary
    summary = processor.get_document_summary(pdf_path)
    print("Document Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Try extraction
    result = processor.extract_text_from_pdf(pdf_path)
    if result:
        text = result.get('text', '') or result.get('content', '')
        print(f"\nExtraction successful!")
        print(f"Text length: {len(text)}")
        print(f"First 200 characters:\n{text[:200]}...")
    else:
        print("\nExtraction failed!")


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
                    'word_count': len(content.split()) if content else 0,
                    'char_count': len(content) if content else 0,
                    'line_count': len(content.split('\n')) if content else 0
                }
                
                # Add key information
                enhanced_section['key_info'] = {
                    'numbered_items_count': len(re.findall(r'^\s*\d+\.', content, re.MULTILINE)) if content else 0,
                    'bullet_points_count': len(re.findall(r'^\s*[•·\-\*]', content, re.MULTILINE)) if content else 0
                }
                
                enhanced_sections.append(enhanced_section)
            
            return {
                'sections': enhanced_sections,
                'extraction_metadata': result.get('extraction_metadata', {}),
                'total_sections': len(enhanced_sections)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting sections with details: {e}")
            return {'sections': [], 'error': str(e)}
    
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
                        sections_count = len(doc_data['sections'])
                        self.logger.info(f"Extracted {sections_count} sections from {pdf_file.name}")
                else:
                    self.logger.warning(f"Failed to process {pdf_file.name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file.name}: {e}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_docs)} out of {len(pdf_files)} files")
        return processed_docs


# Configuration and settings
class DocumentProcessorConfig:
    """Configuration class for document processor settings"""
    
    def __init__(self):
        # Default settings
        self.extract_sections_only = True
        self.validate_extraction = True
        self.min_section_length = 50  # Minimum characters for valid section
        self.max_file_size_mb = 500
        
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
        
        processed_documents = []
        failed_files = []
        total_sections = 0
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing {pdf_file.name}")
                result = self.processor.extract_text_from_pdf(str(pdf_file), self.extract_sections_only)
                
                if result:
                    processed_documents.append(result)
                    sections_count = len(result.get('sections', []))
                    total_sections += sections_count
                    self.logger.info(f"Successfully processed {pdf_file.name}: {sections_count} sections")
                else:
                    failed_files.append(pdf_file.name)
                    self.logger.warning(f"Failed to process {pdf_file.name}")
                    
            except Exception as e:
                failed_files.append(pdf_file.name)
                self.logger.error(f"Error processing {pdf_file.name}: {e}")
        
        summary = {
            'total_files': len(pdf_files),
            'successful': len(processed_documents),
            'failed': len(failed_files),
            'total_sections': total_sections,
            'failed_files': failed_files,
            'processing_time': datetime.now().isoformat()
        }
        
        return {
            'processed_documents': processed_documents,
            'summary': summary
        }
    
    def export_results(self, results: Dict[str, Any], output_file: str, format: str = "json") -> bool:
        """
        Export processing results to file
        
        Args:
            results: Results from process_directory
            output_file: Output file path
            format: Export format ("json", "csv", "excel")
        """
        try:
            if format == "json":
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            elif format == "csv":
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
                        'total_pages': doc.get('metadata', {}).get('page_count', 0)
                    })
                
                df = pd.DataFrame(summary_data)
                df.to_csv(output_file, index=False)
            
            elif format == "excel":
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
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Document_Summary', index=False)
                    pd.DataFrame(sections_data).to_excel(writer, sheet_name='Sections_Detail', index=False)
                    pd.DataFrame([results['summary']]).to_excel(writer, sheet_name='Processing_Summary', index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Results exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return False


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


# Test and example functions
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
            print(f"✅ Success! Found {len(result['sections'])} sections")
            for section in result['sections']:
                print(f"  - {section['type']}: {section['title']} (Pages {section['page_start']}-{section['page_end']})")
        else:
            print("❌ No sections found or extraction failed")
    else:
        print(f"Test file {test_pdf_path} not found")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_extraction(sys.argv[1])
    else:
        print("Usage: python document_processor.py <pdf_file_path>")
