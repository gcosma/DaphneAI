# ===============================================
# FILE: modules/document_processor.py
# ===============================================
from pathlib import Path
import pdfplumber
import fitz
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import gc

class DocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Extract text from PDF with metadata"""
        try:
            text_content = self._extract_with_pdfplumber(pdf_path)
            if not text_content:
                self.logger.warning(f"No text from pdfplumber for {pdf_path}, trying pymupdf")
                text_content = self._extract_with_pymupdf(pdf_path)
            
            metadata = self._extract_metadata_with_pymupdf(pdf_path)
            
            if not text_content:
                self.logger.warning(f"No text extracted from {pdf_path} by any method")
                text_content = ""
            
            return {
                "content": text_content,
                "metadata": metadata,
                "source": Path(pdf_path).name
            }
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
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
    
    def batch_process_documents(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """Process multiple PDFs in directory"""
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            self.logger.error(f"Directory {pdf_directory} does not exist")
            return []
        
        processed_docs = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing {pdf_file.name}")
            doc_data = self.extract_text_from_pdf(str(pdf_file))
            if doc_data:
                processed_docs.append(doc_data)
            else:
                self.logger.warning(f"Failed to extract data from {pdf_file.name}")
            
            # Memory cleanup
            gc.collect()
        
        self.logger.info(f"Processed {len(processed_docs)} PDFs from {pdf_directory}")
        return processed_docs

# Basic logging setup (can be customized in your main app)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
