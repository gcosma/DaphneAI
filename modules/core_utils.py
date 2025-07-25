# ===============================================
# FILE: modules/core_utils.py (COMPLETE VERSION WITH 500MB)
# ===============================================

import re
import logging
import unicodedata
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import nltk
from typing import Union
import urllib3
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter
import math
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components
import time
import os
import zipfile
import asyncio
from pathlib import Path

# Configure logging (can be centralized in the main app file later)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

def clean_text(text: str) -> str:
    """Clean text while preserving structure and metadata formatting"""
    if not text:
        return ""

    try:
        text = str(text)
        text = unicodedata.normalize("NFKD", text)

        replacements = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "â€¦": "...",
            'â€"': "-",
            "â€¢": "•",
            "Â": " ",
            "\u200b": "",
            "\uf0b7": "",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2022": "•",
        }

        for encoded, replacement in replacements.items():
            text = text.replace(encoded, replacement)

        text = re.sub(r"<[^>]+>", "", text)
        text = "".join(
            char if char.isprintable() or char == "\n" else " " for char in text
        )
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n+", "\n", text)

        return text.strip()

    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return ""

def clean_text_for_modeling(text: str) -> str:
    """Clean text for BERT modeling"""
    if not text:
        return ""

    try:
        text = clean_text(text)
        text = text.lower()

        text = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", "", text)
        text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
        text = re.sub(r"\bref\w*\s*[-:\s]?\s*\w+[-\d]+\b", "", text, flags=re.IGNORECASE)
        
        legal_terms = r"\b(coroner|inquest|hearing|evidence|witness|statement|report|dated|signed)\b"
        text = re.sub(legal_terms, "", text, flags=re.IGNORECASE)

        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = " ".join(word for word in text.split() if len(word) > 2)

        return text.strip() if len(text.split()) >= 3 else ""

    except Exception as e:
        logging.error(f"Error in clean_text_for_modeling: {e}")
        return ""

def extract_concern_text(content: str) -> str:
    """
    ENHANCED: Extract complete concern text from PFD report content with robust section handling
    
    This replaces the previous version that was failing on PDF extraction.
    
    Key improvements:
    1. Multiple robust patterns for different document formats
    2. Better handling of OCR issues and text variations
    3. Improved text normalization and cleaning
    4. More comprehensive end markers detection
    5. Better post-processing of extracted text
    """
    if pd.isna(content) or not isinstance(content, str):
        return ""

    try:
        # Enhanced concern section identifiers - handles more variations
        concern_patterns = [
            # Standard patterns with optional punctuation and spacing
            r"CORONER'S\s+CONCERNS?:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"MATTERS?\s+OF\s+CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"The\s+MATTERS?\s+OF\s+CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            
            # Handle variations with "are" after the identifier
            r"CORONER'S\s+CONCERNS?\s+are:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"MATTERS?\s+OF\s+CONCERN\s+are:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"CONCERNS?\s+IDENTIFIED:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            
            # Handle OCR issues where spaces are missing
            r"TheMATTERS?\s*OF\s*CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)"
        ]
        
        # Enhanced text normalization to handle PDF extraction issues
        content_norm = _normalize_pdf_text(content)
        
        # Try each pattern and keep the longest/best match
        best_concerns_text = ""
        best_confidence = 0.0
        
        for pattern in concern_patterns:
            try:
                # Use case-insensitive and multiline matching
                matches = re.finditer(pattern, content_norm, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                
                for match in matches:
                    extracted_text = match.group(1).strip()
                    
                    if extracted_text:
                        # Calculate confidence score for this extraction
                        confidence = _calculate_extraction_confidence(extracted_text, pattern)
                        
                        # Keep the best extraction based on length and confidence
                        if (len(extracted_text) > len(best_concerns_text) and confidence >= best_confidence) or confidence > best_confidence + 0.1:
                            best_concerns_text = extracted_text
                            best_confidence = confidence
                            
            except re.error as e:
                logging.warning(f"Regex error with pattern: {pattern[:50]}... Error: {e}")
                continue
        
        # Post-process the extracted text
        if best_concerns_text:
            return _post_process_concerns_text(best_concerns_text)
        
        return ""
        
    except Exception as e:
        logging.error(f"Error in extract_concern_text: {e}")
        return ""

def _normalize_pdf_text(text: str) -> str:
    """
    Normalize PDF text to handle common extraction issues
    """
    if not text:
        return ""
    
    # Basic normalization - collapse whitespace but preserve structure
    text = ' '.join(text.split())
    
    # Fix common OCR issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add spaces between words
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add spaces before numbers
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Add spaces after numbers
    
    # Fix common character substitutions
    text = text.replace('l', 'I')  # OCR often confuses l and I
    text = text.replace('0', 'O')  # OCR often confuses 0 and O in words
    
    # Normalize punctuation spacing
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    
    return text

def _calculate_extraction_confidence(text: str, pattern: str) -> float:
    """
    Calculate confidence score for extracted text
    """
    confidence = 0.5  # Base confidence
    
    # Length bonus (longer extractions generally more complete)
    if len(text) > 100:
        confidence += 0.2
    if len(text) > 300:
        confidence += 0.1
    
    # Content quality indicators
    if any(word in text.lower() for word in ['concern', 'risk', 'death', 'future']):
        confidence += 0.1
    
    # Structure indicators (numbered points, paragraphs)
    if re.search(r'\d+\.', text):
        confidence += 0.1
    
    return min(confidence, 1.0)

def _post_process_concerns_text(text: str) -> str:
    """
    Clean and post-process extracted concerns text
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Try to end at a complete sentence if text seems cut off
    if not text.endswith(('.', '!', '?', ':')):
        # Find the last complete sentence
        last_period = text.rfind('.')
        last_exclamation = text.rfind('!')
        last_question = text.rfind('?')
        
        last_punctuation = max(last_period, last_exclamation, last_question)
        
        # If we found punctuation and it's in the latter part of the text, cut there
        if last_punctuation != -1 and last_punctuation > len(text) * 0.7:
            text = text[:last_punctuation + 1].strip()
    
    # Remove any leftover artifacts
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'^[:\-\s]+', '', text)  # Remove leading punctuation
    
    return text.strip()

class SecurityValidator:
    """Security validation for file operations"""
    
    @staticmethod
    def validate_file_upload(file_content: bytes, filename: str) -> bool:
        """Validate uploaded files for security"""
        MAX_SIZE = 500 * 1024 * 1024  # ✅ UPDATED: 500MB (was 100MB)
        if len(file_content) > MAX_SIZE:
            raise ValueError(f"File too large: {len(file_content)/1024/1024:.1f}MB > {MAX_SIZE/1024/1024}MB")
        
        allowed_extensions = {'.pdf'}
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise ValueError(f"Invalid file type: {file_ext}")
        
        if not file_content.startswith(b'%PDF'):
            raise ValueError("Invalid PDF file format")
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        import os
        filename = os.path.basename(filename)
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        return filename

# Additional helper functions for metadata extraction
def extract_metadata(content: str) -> Dict[str, str]:
    """
    Extract metadata from document content (ref, names, dates, etc.)
    """
    metadata = {}
    
    if not content or pd.isna(content):
        return metadata
    
    try:
        # Reference number patterns
        ref_patterns = [
            r"(?:Ref|Reference)\.?\s*:?\s*([-\d\w]+)",
            r"(?:Case|Matter)\s+(?:Ref|Reference|No)\.?\s*:?\s*([-\d\w]+)",
            r"(?:PFD|Regulation\s+28)\s*[-:\s]?\s*([-\d\w]+)"
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["ref"] = match.group(1).strip()
                break
        
        # Deceased name
        name_patterns = [
            r"Deceased\s+name:?\s*([^\n]+)",
            r"Name\s+of\s+deceased:?\s*([^\n]+)",
            r"(?:Mr|Mrs|Miss|Ms|Dr)\s+([A-Za-z\s]+?)(?:\s+died|\s+was)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["deceased_name"] = match.group(1).strip()
                break
        
        # Date patterns
        date_patterns = [
            r"Date\s+of\s+report:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"Dated\s+this\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["date_of_report"] = match.group(1).strip()
                break
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        return metadata

def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process scraped data with cleaning and validation"""
    if df is None or len(df) == 0:
        return df
    
    try:
        # Clean text fields
        if 'Content' in df.columns:
            df['Content'] = df['Content'].apply(lambda x: clean_text(str(x)) if pd.notna(x) else "")
        
        if 'Title' in df.columns:
            df['Title'] = df['Title'].apply(lambda x: clean_text(str(x)) if pd.notna(x) else "")
        
        # Extract concerns if not already present
        if 'Extracted_Concerns' not in df.columns and 'Content' in df.columns:
            df['Extracted_Concerns'] = df['Content'].apply(extract_concern_text)
        
        # Process dates
        if 'date_of_report' in df.columns:
            df['date_of_report'] = pd.to_datetime(df['date_of_report'], errors='coerce')
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove duplicates if ref column exists
        if 'ref' in df.columns:
            df = df.drop_duplicates(subset=['ref'], keep='first')
        
        return df
        
    except Exception as e:
        logging.error(f"Error processing scraped data: {e}")
        return df

def is_response_document(row: pd.Series) -> bool:
    """Check if a document is a response document"""
    try:
        for i in range(1, 10):  # Check PDF_1 to PDF_9
            pdf_name = str(row.get(f"PDF_{i}_Name", "")).lower()
            if "response" in pdf_name or "reply" in pdf_name:
                return True
            pdf_type = str(row.get(f"PDF_{i}_Type", "")).lower() # Check type too
            if pdf_type == "response":
                return True

        title = str(row.get("Title", "")).lower()
        if any(word in title for word in ["response", "reply", "answered"]):
            return True

        content = str(row.get("Content", "")).lower()
        return any(
            phrase in content for phrase in [
                "in response to", "responding to", "reply to", "response to",
                "following the regulation 28", "following receipt of the regulation 28"
            ]
        )
    except Exception as e:
        logging.error(f"Error checking response type: {e}")
        return False

def filter_by_categories(df: pd.DataFrame, selected_categories: List[str]) -> pd.DataFrame:
    """Filter DataFrame by categories"""
    if not selected_categories or df.empty:
        return df
    
    try:
        # Handle categories column
        if 'categories' in df.columns:
            mask = df['categories'].apply(
                lambda x: any(cat in str(x).lower() for cat in [c.lower() for c in selected_categories])
                if pd.notna(x) else False
            )
            return df[mask]
        
        return df
        
    except Exception as e:
        logging.error(f"Error filtering by categories: {e}")
        return df

def export_to_excel(df: pd.DataFrame, filename: str = None) -> bytes:
    """
    Export DataFrame to Excel format with proper formatting
    
    Args:
        df: DataFrame to export
        filename: Optional filename (not used, kept for compatibility)
        
    Returns:
        Excel file as bytes
    """
    try:
        # Create Excel buffer
        excel_buffer = io.BytesIO()
        
        # Create workbook and worksheet
        wb = Workbook()
        ws = wb.active
        ws.title = "Document Analysis"
        
        # Add DataFrame to worksheet
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
        
        # Format headers
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            # Set reasonable width limits
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = max(adjusted_width, 10)
        
        # Set row height for header
        ws.row_dimensions[1].height = 20
        
        # Add data formatting
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        
        # Save to buffer
        wb.save(excel_buffer)
        excel_buffer.seek(0)
        
        return excel_buffer.getvalue()
        
    except Exception as e:
        logging.error(f"Error exporting to Excel: {e}")
        # Fallback to simple Excel export
        try:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_buffer.seek(0)
            return excel_buffer.getvalue()
        except Exception as fallback_error:
            logging.error(f"Fallback Excel export also failed: {fallback_error}")
            raise Exception("Excel export failed")

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate DataFrame and return validation results
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_required_columns": [],
        "data_quality_score": 0.0
    }
    
    try:
        # Check for required columns
        required_columns = ["Title", "Content"]
        for col in required_columns:
            if col not in df.columns:
                validation_results["missing_required_columns"].append(col)
                validation_results["errors"].append(f"Missing required column: {col}")
                validation_results["is_valid"] = False
        
        # Check for empty DataFrame
        if len(df) == 0:
            validation_results["errors"].append("DataFrame is empty")
            validation_results["is_valid"] = False
            return validation_results
        
        # Calculate data quality metrics
        quality_scores = []
        
        # Title completeness
        if "Title" in df.columns:
            title_completeness = df["Title"].notna().mean()
            quality_scores.append(title_completeness)
            if title_completeness < 0.9:
                validation_results["warnings"].append(f"Title completeness: {title_completeness:.1%}")
        
        # Content completeness
        if "Content" in df.columns:
            content_completeness = df["Content"].notna().mean()
            quality_scores.append(content_completeness)
            if content_completeness < 0.8:
                validation_results["warnings"].append(f"Content completeness: {content_completeness:.1%}")
        
        # Date completeness
        if "date_of_report" in df.columns:
            date_completeness = df["date_of_report"].notna().mean()
            quality_scores.append(date_completeness)
            if date_completeness < 0.7:
                validation_results["warnings"].append(f"Date completeness: {date_completeness:.1%}")
        
        # Calculate overall quality score
        if quality_scores:
            validation_results["data_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        # Check for duplicates
        if "Title" in df.columns and "ref" in df.columns:
            duplicate_count = df.duplicated(subset=["Title", "ref"]).sum()
            if duplicate_count > 0:
                validation_results["warnings"].append(f"Found {duplicate_count} potential duplicates")
        
        # Check data types
        if "date_of_report" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["date_of_report"]):
                validation_results["warnings"].append("Date column is not in datetime format")
        
        return validation_results
        
    except Exception as e:
        validation_results["errors"].append(f"Validation error: {str(e)}")
        validation_results["is_valid"] = False
        logging.error(f"Data validation error: {e}")
        return validation_results

def format_date_uk(date_obj):
    """Convert datetime object to UK date format string"""
    if pd.isna(date_obj): 
        return ""
    try:
        if isinstance(date_obj, str):
            date_obj = pd.to_datetime(date_obj, errors='coerce')
        if pd.isna(date_obj): 
            return "" # if conversion failed
        return date_obj.strftime("%d/%m/%Y")
    except:
        return str(date_obj) # fallback to string representation

def create_document_identifier(row: pd.Series) -> str:
    """Create a unique identifier for a document."""
    title = str(row.get("Title", "")).strip()
    ref = str(row.get("ref", "")).strip()
    deceased = str(row.get("deceased_name", "")).strip()
    return f"{title}_{ref}_{deceased}"

def deduplicate_documents(data: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate documents while preserving unique entries."""
    try:
        if data is None or len(data) == 0:
            return data
        
        # Create document identifiers
        data["doc_id"] = data.apply(create_document_identifier, axis=1)
        
        # Remove duplicates based on identifier
        data_deduped = data.drop_duplicates(subset=["doc_id"], keep="first")
        
        # Remove the temporary identifier column
        data_deduped = data_deduped.drop(columns=["doc_id"])
        
        logging.info(f"Deduplicated {len(data)} -> {len(data_deduped)} documents")
        return data_deduped
        
    except Exception as e:
        logging.error(f"Error in deduplication: {e}")
        return data

def combine_document_text(row: pd.Series) -> str:
    """Combine all text content from a document row for analysis."""
    text_parts = []
    if pd.notna(row.get("Title")): 
        text_parts.append(str(row["Title"]))
    if pd.notna(row.get("Content")): 
        text_parts.append(str(row["Content"]))
    
    # Iterate through potential PDF content columns
    for i in range(1, 10): # Assuming up to PDF_9_Content
        pdf_col_name = f"PDF_{i}_Content"
        if pdf_col_name in row.index and pd.notna(row.get(pdf_col_name)):
            text_parts.append(str(row[pdf_col_name]))
            
    return " ".join(text_parts)

def filter_by_document_types(df: pd.DataFrame, doc_types: List[str]) -> pd.DataFrame:
    """Filter DataFrame by document types (Report/Response)"""
    if not doc_types: 
        return df
    
    is_response_series = df.apply(is_response_document, axis=1)
    
    if "Report" in doc_types and "Response" not in doc_types:
        return df[~is_response_series]
    elif "Response" in doc_types and "Report" not in doc_types:
        return df[is_response_series]
    # If both are selected or neither (though 'if not doc_types' handles neither), return all
    return df

def perform_advanced_keyword_search(df: pd.DataFrame, keywords: List[str], search_columns: List[str] = None) -> pd.DataFrame:
    """Perform advanced keyword search across specified columns"""
    if not keywords or df.empty:
        return df
    
    if search_columns is None:
        search_columns = ['Title', 'Content', 'Extracted_Concerns']
    
    # Available columns in dataframe
    available_columns = [col for col in search_columns if col in df.columns]
    
    if not available_columns:
        return df
    
    try:
        # Create search mask
        mask = pd.Series([False] * len(df))
        
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if not keyword_lower:
                continue
                
            # Search in each column
            for col in available_columns:
                col_mask = df[col].astype(str).str.lower().str.contains(
                    keyword_lower, case=False, na=False, regex=False
                )
                mask = mask | col_mask
        
        return df[mask]
        
    except Exception as e:
        logging.error(f"Error in keyword search: {e}")
        return df

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global headers for all requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://judiciary.uk/",
}

# Export functions for compatibility with existing codebase
__all__ = [
    'clean_text',
    'clean_text_for_modeling', 
    'extract_concern_text',
    'SecurityValidator',
    'extract_metadata',
    'process_scraped_data',
    'is_response_document',
    'filter_by_categories',
    'export_to_excel',
    'validate_data',
    'format_date_uk',
    'create_document_identifier',
    'deduplicate_documents',
    'combine_document_text',
    'filter_by_document_types',
    'perform_advanced_keyword_search'
]
