# modules/document_processor.py
# COMPLETE ENHANCED FILE - 900MB Document Support with Government Response Optimization

import logging
import re
import io
import gc
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Iterator
import hashlib
import tempfile
import mmap
from contextlib import contextmanager

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

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

class LargeDocumentProcessor:
    """
    Enhanced document processor for very large files (up to 900MB)
    Optimized for UK Government Response documents with memory-efficient processing
    """
    
    def __init__(self, debug_mode: bool = False, max_memory_mb: int = 2048):
        """Initialize the large document processor"""
        self.debug_mode = debug_mode
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
        
        # Memory management settings
        self.chunk_size_mb = min(50, max_memory_mb // 4)  # Process in 50MB chunks max
        self.page_batch_size = 10  # Process 10 pages at a time
        
        # Processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'total_pages_processed': 0,
            'extraction_errors': 0,
            'total_processing_time': 0.0,
            'memory_peak_mb': 0.0
        }
        
        # Initialize patterns for Government Response documents
        self._initialize_government_response_patterns()
        
        self.logger.info(f"LargeDocumentProcessor initialized - Max memory: {max_memory_mb}MB")

    def _initialize_government_response_patterns(self):
        """Initialize patterns specifically for Government Response documents"""
        
        # Recommendation patterns from actual Government Response documents
        self.recommendation_patterns = [
            # Numbered recommendations: "Recommendation 6a) iii)"
            r'(?i)(Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s*[:\.)]\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
            
            # Sub-recommendations with complex numbering
            r'(?i)Recommendation\s+(\d+[a-z]*\s*\)\s*[iv]+\s*\))\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
            
            # Progress/review recommendations
            r'(?i)(Progress\s+(?:in\s+)?implementation|Review\s+of\s+progress)\s+(?:towards\s+|of\s+)?([^.!?]{20,400}[.!?])',
            
            # Framework/system establishment
            r'(?i)(?:framework|system|process|mechanism)\s+(?:should|must|be)\s+(?:established|created|implemented)\s+([^.!?]{20,400}[.!?])',
            
            # Training and education requirements
            r'(?i)(?:training|education|guidance)\s+(?:should|must)\s+be\s+(?:provided|delivered|enhanced)\s+([^.!?]{20,400}[.!?])',
            
            # Action requirements
            r'(?i)(?:action|steps|measures)\s+(?:should|must)\s+be\s+taken\s+([^.!?]{20,400}[.!?])',
        ]
        
        # Response patterns from actual Government Response documents
        self.response_patterns = [
            # Core acceptance patterns - most important
            r'(?i)(?:This\s+recommendation|Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s+is\s+(accepted\s+in\s+(?:full|principle)|not\s+accepted|partially\s+accepted|rejected)\s*(?:by\s+(?:the\s+)?(?:UK\s+Government|Scottish\s+Government|Welsh\s+Government|Northern\s+Ireland\s+Executive|Government))?\.?\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
            
            # "Accepting in principle" patterns
            r'(?i)(Accepting\s+in\s+principle)\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
            
            # Implementation will begin patterns
            r'(?i)(Implementation\s+will\s+begin|Implementation\s+is\s+underway|Work\s+is\s+ongoing)\s*([^.!?]{20,500}[.!?])',
            
            # Government will establish patterns
            r'(?i)(?:The\s+(?:UK\s+)?Government|We|The\s+(?:Department|Ministry))\s+will\s+(establish|implement|develop|create)\s+([^.!?]{20,500}[.!?])',
            
            # Cross-nation acceptance patterns
            r'(?i)(?:UK\s+Government|Scottish\s+Government|Welsh\s+Government|Northern\s+Ireland\s+Executive)\s+(?:accepts?|agrees?|will)\s+([^.!?]{20,500}[.!?])',
            
            # Parts of these recommendations pattern
            r'(?i)(?:These\s+recommendations|Parts\s+of\s+these\s+recommendations)\s+are\s+(?:accepted|being\s+taken\s+forward)\s*([^.!?]{20,500}[.!?])',
            
            # Review and assessment patterns
            r'(?i)(?:review|assessment|evaluation)\s+(?:has\s+been|is\s+being|will\s+be)\s+(?:conducted|undertaken|scheduled)\s*([^.!?]{20,500}[.!?])',
            
            # Funding and resource commitments
            r'(?i)(?:funding|resources?|investment)\s+(?:will\s+be|has\s+been)\s+(?:provided|allocated|committed)\s*([^.!?]{20,500}[.!?])',
            
            # Timeline commitments
            r'(?i)(?:within\s+\d+\s+(?:months?|years?)|by\s+\d+|during\s+the\s+(?:first|next))\s+([^.!?]{20,500}[.!?])',
            
            # Quality and safety responses
            r'(?i)(?:quality\s+and\s+safety|patient\s+safety|safety\s+management)\s+(?:system|framework|approach)\s*([^.!?]{20,500}[.!?])',
        ]

    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Monitor memory usage during operations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = max(start_memory, end_memory)
            
            self.processing_stats['memory_peak_mb'] = max(
                self.processing_stats['memory_peak_mb'], 
                peak_memory
            )
            
            if self.debug_mode:
                self.logger.info(f"Memory usage for {operation_name}: {start_memory:.1f}MB -> {end_memory:.1f}MB")

    def process_large_pdf(self, file_path_or_bytes, filename: str = None) -> Dict[str, Any]:
        """
        Process large PDF files (up to 900MB) with memory optimization
        
        Args:
            file_path_or_bytes: File path string or bytes object
            filename: Optional filename for metadata
            
        Returns:
            Dict with extracted content and metadata
        """
        start_time = datetime.now()
        
        try:
            with self.memory_monitor("large_pdf_processing"):
                # Determine if input is file path or bytes
                if isinstance(file_path_or_bytes, (str, Path)):
                    file_path = Path(file_path_or_bytes)
                    file_size = file_path.stat().st_size
                    filename = filename or file_path.name
                    
                    # Memory-mapped file reading for very large files
                    result = self._process_pdf_file_path(file_path)
                    
                else:
                    # Bytes input (from Streamlit uploads)
                    file_size = len(file_path_or_bytes)
                    filename = filename or "uploaded_document.pdf"
                    
                    # Save to temporary file for memory-mapped processing
                    result = self._process_pdf_bytes(file_path_or_bytes, filename)
                
                # Add metadata
                result.update({
                    'filename': filename,
                    'file_size_mb': round(file_size / 1024 / 1024, 2),
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                    'processed_at': datetime.now().isoformat(),
                    'processor_version': 'large_document_v2.0',
                    'memory_optimized': True
                })
                
                self.processing_stats['documents_processed'] += 1
                self.processing_stats['total_processing_time'] += result['processing_time_seconds']
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error processing large PDF {filename}: {e}")
            self.processing_stats['extraction_errors'] += 1
            
            return {
                'filename': filename or 'unknown',
                'error': str(e),
                'processing_status': 'error',
                'processed_at': datetime.now().isoformat()
            }

    def _process_pdf_file_path(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF from file path using memory mapping"""
        
        if PYMUPDF_AVAILABLE:
            return self._process_with_pymupdf_mmap(file_path)
        elif PDFPLUMBER_AVAILABLE:
            return self._process_with_pdfplumber_chunked(file_path)
        else:
            raise ImportError("No PDF processing library available")

    def _process_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Process PDF from bytes using temporary file"""
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            try:
                # Write bytes to temporary file in chunks
                chunk_size = 8192
                for i in range(0, len(pdf_bytes), chunk_size):
                    chunk = pdf_bytes[i:i + chunk_size]
                    temp_file.write(chunk)
                
                temp_file.flush()
                temp_path = Path(temp_file.name)
                
                # Process the temporary file
                result = self._process_pdf_file_path(temp_path)
                
                return result
                
            finally:
                # Clean up temporary file
                try:
                    temp_path.unlink()
                except:
                    pass

    def _process_with_pymupdf_mmap(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF using PyMuPDF with memory mapping for large files"""
        
        full_text = []
        total_pages = 0
        processing_errors = []
        
        try:
            # Open document with memory mapping
            doc = fitz.open(str(file_path))
            total_pages = doc.page_count
            
            self.logger.info(f"Processing {total_pages} pages with PyMuPDF (memory mapped)")
            
            # Process pages in batches to manage memory
            for page_start in range(0, total_pages, self.page_batch_size):
                page_end = min(page_start + self.page_batch_size, total_pages)
                
                try:
                    batch_text = self._process_page_batch_pymupdf(doc, page_start, page_end)
                    full_text.extend(batch_text)
                    
                    # Update progress if Streamlit is available
                    if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
                        progress = page_end / total_pages
                        if 'pdf_progress_bar' in st.session_state:
                            st.session_state.pdf_progress_bar.progress(progress)
                    
                    # Force garbage collection every batch
                    if page_start % (self.page_batch_size * 4) == 0:
                        gc.collect()
                        
                except Exception as e:
                    error_msg = f"Error processing pages {page_start}-{page_end}: {e}"
                    self.logger.warning(error_msg)
                    processing_errors.append(error_msg)
            
            doc.close()
            
            # Combine all text
            combined_text = '\n\n'.join(full_text)
            
            # Extract recommendations and responses using government patterns
            extraction_results = self._extract_government_content(combined_text)
            
            return {
                'content': combined_text,
                'page_count': total_pages,
                'content_length': len(combined_text),
                'extraction_method': 'pymupdf_mmap',
                'processing_errors': processing_errors,
                'processing_status': 'completed',
                'recommendations_found': len(extraction_results.get('recommendations', [])),
                'responses_found': len(extraction_results.get('responses', [])),
                'extraction_results': extraction_results
            }
            
        except Exception as e:
            self.logger.error(f"PyMuPDF processing failed: {e}")
            raise

    def _process_page_batch_pymupdf(self, doc, start_page: int, end_page: int) -> List[str]:
        """Process a batch of pages with PyMuPDF"""
        batch_text = []
        
        for page_num in range(start_page, end_page):
            try:
                page = doc[page_num]
                text = page.get_text()
                
                # Clean and optimize text
                cleaned_text = self._clean_extracted_text(text)
                
                if cleaned_text.strip():
                    batch_text.append(cleaned_text)
                    
                # Update page processing count
                self.processing_stats['total_pages_processed'] += 1
                
            except Exception as e:
                self.logger.warning(f"Error processing page {page_num}: {e}")
                continue
        
        return batch_text

    def _process_with_pdfplumber_chunked(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF using pdfplumber with chunked reading"""
        
        full_text = []
        total_pages = 0
        processing_errors = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                self.logger.info(f"Processing {total_pages} pages with pdfplumber (chunked)")
                
                # Process pages in batches
                for page_start in range(0, total_pages, self.page_batch_size):
                    page_end = min(page_start + self.page_batch_size, total_pages)
                    
                    try:
                        batch_text = self._process_page_batch_pdfplumber(pdf, page_start, page_end)
                        full_text.extend(batch_text)
                        
                        # Force garbage collection
                        if page_start % (self.page_batch_size * 4) == 0:
                            gc.collect()
                            
                    except Exception as e:
                        error_msg = f"Error processing pages {page_start}-{page_end}: {e}"
                        self.logger.warning(error_msg)
                        processing_errors.append(error_msg)
                
                # Combine all text
                combined_text = '\n\n'.join(full_text)
                
                # Extract government content
                extraction_results = self._extract_government_content(combined_text)
                
                return {
                    'content': combined_text,
                    'page_count': total_pages,
                    'content_length': len(combined_text),
                    'extraction_method': 'pdfplumber_chunked',
                    'processing_errors': processing_errors,
                    'processing_status': 'completed',
                    'recommendations_found': len(extraction_results.get('recommendations', [])),
                    'responses_found': len(extraction_results.get('responses', [])),
                    'extraction_results': extraction_results
                }
                
        except Exception as e:
            self.logger.error(f"PDFplumber processing failed: {e}")
            raise

    def _process_page_batch_pdfplumber(self, pdf, start_page: int, end_page: int) -> List[str]:
        """Process a batch of pages with pdfplumber"""
        batch_text = []
        
        for page_num in range(start_page, end_page):
            try:
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if text:
                    # Clean and optimize text
                    cleaned_text = self._clean_extracted_text(text)
                    
                    if cleaned_text.strip():
                        batch_text.append(cleaned_text)
                
                # Update page processing count
                self.processing_stats['total_pages_processed'] += 1
                
            except Exception as e:
                self.logger.warning(f"Error processing page {page_num}: {e}")
                continue
        
        return batch_text

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and optimize extracted text for government documents"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and footers
        text = re.sub(r'\b(?:page|p\.)\s*\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*(?:of|/)\s*\d+\b', '', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'©\s*Crown\s*copyright\s*\d{4}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'E\d{8}\s+\d{2}/\d{2}', '', text)  # E-numbers
        
        # Remove URLs but keep government domains
        text = re.sub(r'https?://(?!.*\.gov\.uk)[^\s]+', '', text)
        
        # Clean up multiple spaces and newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

    def _extract_government_content(self, text: str) -> Dict[str, List[Dict]]:
        """Extract recommendations and responses using government-specific patterns"""
        
        recommendations = []
        responses = []
        
        # Extract recommendations
        for i, pattern in enumerate(self.recommendation_patterns):
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        rec_id = match.group(1).strip()
                        rec_content = match.group(2).strip()
                    else:
                        rec_id = f"rec_{len(recommendations)}"
                        rec_content = match.group().strip()
                    
                    if len(rec_content) > 20:  # Filter short matches
                        recommendations.append({
                            'id': rec_id,
                            'content': rec_content,
                            'pattern_index': i,
                            'confidence': 0.85,
                            'extraction_method': 'government_pattern',
                            'start_position': match.start(),
                            'end_position': match.end()
                        })
                        
            except re.error as e:
                self.logger.warning(f"Regex error in recommendation pattern {i}: {e}")
        
        # Extract responses
        for i, pattern in enumerate(self.response_patterns):
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        response_type = match.group(1).strip()
                        response_content = match.group(2).strip()
                        full_content = f"{response_type} {response_content}".strip()
                    else:
                        full_content = match.group().strip()
                        response_type = "general_response"
                    
                    if len(full_content) > 20:  # Filter short matches
                        responses.append({
                            'id': f"resp_{len(responses)}",
                            'content': full_content,
                            'response_type': response_type,
                            'pattern_index': i,
                            'confidence': 0.85,
                            'extraction_method': 'government_pattern',
                            'start_position': match.start(),
                            'end_position': match.end()
                        })
                        
            except re.error as e:
                self.logger.warning(f"Regex error in response pattern {i}: {e}")
        
        # Remove duplicates
        recommendations = self._remove_duplicate_extractions(recommendations)
        responses = self._remove_duplicate_extractions(responses)
        
        return {
            'recommendations': recommendations,
            'responses': responses
        }

    def _remove_duplicate_extractions(self, extractions: List[Dict]) -> List[Dict]:
        """Remove duplicate extractions based on content similarity"""
        if not extractions:
            return []
        
        unique_extractions = []
        seen_content = set()
        
        for extraction in extractions:
            content = extraction.get('content', '')
            # Normalize content for comparison
            normalized = re.sub(r'\s+', ' ', content.lower().strip())
            
            # Skip if we've seen very similar content
            is_duplicate = False
            for seen in seen_content:
                if len(seen) > 0 and len(normalized) > 0:
                    # Calculate simple similarity
                    shorter = min(len(seen), len(normalized))
                    longer = max(len(seen), len(normalized))
                    if shorter / longer > 0.85:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_content.add(normalized)
                unique_extractions.append(extraction)
        
        # Sort by confidence and position
        unique_extractions.sort(
            key=lambda x: (x.get('confidence', 0), -x.get('start_position', 0)), 
            reverse=True
        )
        
        return unique_extractions

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_time = (
            self.processing_stats['total_processing_time'] / 
            max(self.processing_stats['documents_processed'], 1)
        )
        
        return {
            'documents_processed': self.processing_stats['documents_processed'],
            'total_pages_processed': self.processing_stats['total_pages_processed'],
            'extraction_errors': self.processing_stats['extraction_errors'],
            'average_processing_time_seconds': round(avg_time, 2),
            'memory_peak_mb': round(self.processing_stats['memory_peak_mb'], 2),
            'processor_version': 'large_document_v2.0'
        }

    def test_patterns(self, sample_text: str = None) -> Dict[str, Any]:
        """Test extraction patterns on sample text"""
        
        if not sample_text:
            sample_text = """
            Recommendation 6a) iii) (accepted in principle by NI Executive)
            
            This recommendation is accepted in principle by the UK Government, 
            the Scottish Government, the Welsh Government, and the Northern Ireland Executive.
            
            Implementation will begin immediately through existing NHS structures.
            
            Recommendation 7b) Progress in implementation of the Transfusion 2024 recommendations be 
            reviewed, and next steps be determined and promulgated.
            
            This recommendation is accepted in full by the Scottish Government.
            
            The Government will establish a review process within 12 months.
            
            Accepting in principle
            Recommendation 4a) iv)
            Recommendation 4a) v)
            """
        
        results = self._extract_government_content(sample_text)
        
        return {
            'test_text_length': len(sample_text),
            'recommendations_found': len(results['recommendations']),
            'responses_found': len(results['responses']),
            'extraction_results': results,
            'pattern_coverage': {
                'recommendation_patterns': len(self.recommendation_patterns),
                'response_patterns': len(self.response_patterns)
            }
        }

# Backward compatibility
DocumentProcessor = LargeDocumentProcessor

# Export all classes and functions
__all__ = [
    'LargeDocumentProcessor',
    'DocumentProcessor',
    'PDFPLUMBER_AVAILABLE',
    'PYMUPDF_AVAILABLE'
]
