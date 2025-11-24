# modules/document_processor.py
# Simple, clean document processor for Streamlit document search
# WITH CHARACTER CLEANING to fix encoding issues

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
import logging
import re
# PDF processing with fallbacks
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available")

# DOCX processing
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not available")

def clean_extracted_text(text: str) -> str:
    """Clean text extracted from documents - remove special characters"""
    if not text:
        return ""
    
    # Replace smart quotes with regular quotes
    text = text.replace('"', '"').replace('"', '"')  # Smart double quotes
    text = text.replace(''', "'").replace(''', "'")  # Smart single quotes
    
    # Replace other problematic characters
    text = text.replace('–', '-').replace('—', '-')  # Em and en dashes
    text = text.replace('…', '...')  # Ellipsis
    
    # Remove non-breaking spaces and other unicode spaces
    text = text.replace('\u00A0', ' ')  # Non-breaking space
    text = text.replace('\u2009', ' ')  # Thin space
    text = text.replace('\u200B', '')   # Zero-width space
    text = text.replace('\u202F', ' ')  # Narrow no-break space
    text = text.replace('\u2028', '\n')  # Line separator
    text = text.replace('\u2029', '\n\n')  # Paragraph separator
    
    # Fix ligatures that sometimes appear in PDFs
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('ﬀ', 'ff').replace('ﬃ', 'ffi').replace('ﬄ', 'ffl')
    
    # Fix multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix multiple newlines (more than 2 in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def process_uploaded_files(uploaded_files: List) -> List[Dict[str, Any]]:
    """
    Simple document processor for Streamlit uploads
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        
    Returns:
        List of processed documents with text and metadata
    """
    if not uploaded_files:
        return []
    
    processed_docs = []
    
    for uploaded_file in uploaded_files:
        try:
            filename = uploaded_file.name
            file_size = len(uploaded_file.getvalue())
            
            # Process based on file type
            if filename.lower().endswith('.pdf'):
                text = extract_pdf_text(uploaded_file)
            elif filename.lower().endswith('.txt'):
                text = extract_txt_text(uploaded_file)
            elif filename.lower().endswith('.docx'):
                text = extract_docx_text(uploaded_file)
            else:
                # Unsupported file type
                processed_docs.append({
                    'filename': filename,
                    'error': f'Unsupported file type: {filename.split(".")[-1]}',
                    'file_type': filename.split('.')[-1].lower() if '.' in filename else 'unknown',
                    'processed_at': datetime.now().isoformat()
                })
                continue
            
            # Clean the extracted text
            text = clean_extracted_text(text)
            
            # Create document record
            doc = {
                'filename': filename,
                'text': text,
                'word_count': len(text.split()) if text else 0,
                'file_type': filename.split('.')[-1].lower() if '.' in filename else 'unknown',
                'file_size_mb': round(file_size / 1024 / 1024, 2),
                'processed_at': datetime.now().isoformat(),
                'processing_status': 'success'
            }
            
            processed_docs.append(doc)
            logging.info(f"Successfully processed {filename}: {doc['word_count']} words")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            processed_docs.append({
                'filename': filename,
                'error': str(e),
                'file_type': filename.split('.')[-1].lower() if '.' in filename else 'unknown',
                'processing_status': 'error',
                'processed_at': datetime.now().isoformat()
            })
    
    return processed_docs

def extract_pdf_text(uploaded_file) -> str:
    """Extract text from PDF using available library"""
    if not PDFPLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
        raise ImportError("No PDF library available. Install: pip install pdfplumber")
    
    try:
        # Try pdfplumber first (better for most documents)
        if PDFPLUMBER_AVAILABLE:
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                # Join pages with double newline
                full_text = '\n\n'.join(text_parts)
                
                # Fix multiple spaces between words (reduce to single space)
                full_text = re.sub(r' {2,}', ' ', full_text)
                
                # Fix issue where words are on separate lines
                # Replace single newlines with spaces, but keep double newlines (paragraphs)
                full_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', full_text)
                
                # Clean the text before returning
                return clean_extracted_text(full_text)
        
        # Fallback to PyPDF2
        elif PYPDF2_AVAILABLE:
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = '\n\n'.join(text_parts)
            
            # Fix multiple spaces
            full_text = re.sub(r' {2,}', ' ', full_text)
            
            # Fix single-word-per-line issue
            full_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', full_text)
            
            # Clean the text before returning
            return clean_extracted_text(full_text)
    
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        raise Exception(f"Failed to extract PDF text: {str(e)}")
        


def extract_txt_text(uploaded_file) -> str:
    """Extract text from TXT file with encoding detection"""
    try:
        # Try UTF-8 first
        text = uploaded_file.getvalue().decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Try with latin-1 encoding
            text = uploaded_file.getvalue().decode('latin-1')
        except UnicodeDecodeError:
            # Last resort: ignore errors
            text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
    
    # Clean the text before returning
    return clean_extracted_text(text)

def extract_docx_text(uploaded_file) -> str:
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not installed. Install: pip install python-docx")
    
    try:
        import docx
        from io import BytesIO
        
        doc = docx.Document(BytesIO(uploaded_file.getvalue()))
        text_parts = []
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_parts.append(cell.text)
        
        full_text = '\n'.join(text_parts)
        
        # Clean the text before returning
        return clean_extracted_text(full_text)
    
    except Exception as e:
        logging.error(f"DOCX extraction error: {e}")
        raise Exception(f"Failed to extract DOCX text: {str(e)}")

def get_processing_stats():
    """Get processing statistics (simple version)"""
    return {
        'documents_processed': 0,
        'total_processing_time': 0.0,
        'pdf_library': 'pdfplumber' if PDFPLUMBER_AVAILABLE else 'PyPDF2' if PYPDF2_AVAILABLE else 'none',
        'docx_library': 'python-docx' if DOCX_AVAILABLE else 'none'
    }

def check_dependencies():
    """Check which document processing libraries are available"""
    deps = {
        'pdfplumber': PDFPLUMBER_AVAILABLE,
        'PyPDF2': PYPDF2_AVAILABLE,
        'python-docx': DOCX_AVAILABLE
    }
    return deps
