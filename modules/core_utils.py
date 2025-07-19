# ===============================================
# FILE: modules/core_utils.py (UPDATED VERSION)
# ===============================================
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
import json
from datetime import datetime
import re
import unicodedata
import ssl
import nltk

# Setup NLTK with SSL handling
def ensure_nltk_data():
    """Ensure NLTK data is available with SSL handling"""
    try:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        required_data = ['punkt', 'stopwords']
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except Exception as e:
                    logging.warning(f"Failed to download NLTK data {data_name}: {e}")
    except Exception as e:
        logging.warning(f"NLTK setup failed: {e}")

# Initialize NLTK
ensure_nltk_data()

@dataclass
class Recommendation:
    id: str
    text: str
    document_source: str
    section_title: str
    page_number: Optional[int] = None
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Response:
    id: str
    text: str
    recommendation_id: str
    status: str
    document_source: str
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class AnnotationResult:
    framework: str
    theme: str
    confidence: float
    matched_keywords: List[str]
    semantic_similarity: float
    combined_score: float
    sentence_positions: List[tuple] = field(default_factory=list)

def clean_text(text: str) -> str:
    """Clean text while preserving structure"""
    if not text:
        return ""

    try:
        text = str(text)
        text = unicodedata.normalize("NFKD", text)

        replacements = {
            "â€™": "'", "â€œ": '"', "â€": '"', "â€¦": "...",
            'â€"': "-", "â€¢": "•", "Â": " ", "\u200b": "",
            "\uf0b7": "", "\u2019": "'", "\u201c": '"',
            "\u201d": '"', "\u2013": "-", "\u2022": "•",
        }

        for encoded, replacement in replacements.items():
            text = text.replace(encoded, replacement)

        text = re.sub(r"<[^>]+>", "", text)
        text = "".join(char if char.isprintable() or char == "\n" else " " for char in text)
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
            
            # Alternative section headers
            r"HEALTHCARE\s+SAFETY\s+CONCERNS?:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"SAFETY\s+CONCERNS?:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"PATIENT\s+SAFETY\s+ISSUES?:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"HSIB\s+FINDINGS:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"INVESTIGATION\s+FINDINGS:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"THE\s+CORONER'S\s+MATTER\s+OF\s+CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
            r"CONCERNS?\s+AND\s+RECOMMENDATIONS?:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|COPIES|SIGNED:|DATED\s+THIS|NEXT\s+STEPS|YOU\s+ARE\s+UNDER\s+A\s+DUTY|$)",
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
    
    return text.strip()

def _calculate_extraction_confidence(text: str, pattern: str) -> float:
    """
    Calculate confidence score for extracted concern text
    """
    base_confidence = 0.6
    
    # Length indicators (longer text often more complete)
    if len(text) > 50:
        base_confidence += 0.1
    if len(text) > 200:
        base_confidence += 0.1
    if len(text) > 500:
        base_confidence += 0.1
    
    # Structure indicators
    structure_indicators = [
        (r'\d+\.', 0.1),  # Numbered points (1., 2., etc.)
        (r'[A-Z][a-z]+:', 0.05),  # Section headers (Patient:, etc.)
        (r'(?:should|must|ought\s+to|need\s+to)', 0.05),  # Action words
        (r'(?:concern|issue|problem|risk|safety|patient)', 0.05),  # Key concern words
        (r'(?:hospital|medical|treatment|care)', 0.03),  # Healthcare context
        (r'(?:therefore|however|furthermore|additionally)', 0.03),  # Connective words
    ]
    
    for indicator_pattern, confidence_boost in structure_indicators:
        if re.search(indicator_pattern, text, re.IGNORECASE):
            base_confidence += confidence_boost
    
    # Pattern-specific confidence adjustments
    if "CORONER'S CONCERNS" in pattern.upper():
        base_confidence += 0.1  # Most reliable pattern
    elif "MATTERS OF CONCERN" in pattern.upper():
        base_confidence += 0.08  # Very reliable
    elif "SAFETY CONCERNS" in pattern.upper():
        base_confidence += 0.05  # Moderately reliable
    
    return min(base_confidence, 1.0)

def _post_process_concerns_text(text: str) -> str:
    """
    Post-process extracted concerns text to clean it up
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove any remaining end markers that might have been captured
    end_markers = [
        "ACTION SHOULD BE TAKEN",
        "CONCLUSIONS",
        "YOUR RESPONSE", 
        "COPIES",
        "SIGNED:",
        "DATED THIS",
        "NEXT STEPS",
        "YOU ARE UNDER A DUTY",
        "RESPONSE REQUIRED",
        "CHIEF CORONER",
        "REGULATION 28"
    ]
    
    for marker in end_markers:
        # Case-insensitive removal
        marker_pos = text.upper().find(marker.upper())
        if marker_pos != -1:
            text = text[:marker_pos].strip()
    
    # Clean up incomplete sentences at the end
    if text and not text.endswith(('.', '!', '?', ':')):
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
        MAX_SIZE = 100 * 1024 * 1024  # 100MB
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
            r"(?:The\s+)?deceased:?\s*([^\n]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up the name
                name = re.sub(r'\s*\([^)]*\)', '', name)  # Remove parenthetical info
                name = re.sub(r'\s*,.*$', '', name)  # Remove comma and everything after
                metadata["deceased_name"] = name.strip()
                break
        
        # Coroner name
        coroner_patterns = [
            r"Coroner(?:\'?s)?\s+name:?\s*([^\n]+)",
            r"(?:Senior\s+)?Coroner:?\s*([^\n]+)",
            r"(?:HM\s+)?Coroner\s+for\s+[^:]+:?\s*([^\n]+)"
        ]
        
        for pattern in coroner_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["coroner_name"] = match.group(1).strip()
                break
        
        # Coroner area
        area_patterns = [
            r"Coroner(?:\'?s)?\s+Area:?\s*([^\n]+)",
            r"Area\s+of\s+Jurisdiction:?\s*([^\n]+)",
            r"(?:HM\s+)?Coroner\s+for\s+([^:\n]+)"
        ]
        
        for pattern in area_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["coroner_area"] = match.group(1).strip()
                break
        
        # Date of report
        date_patterns = [
            r"(?:Date|Dated):?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"(?:Date|Dated):?\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})",
            r"Report\s+dated:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})"  # Any date pattern
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["date_of_report"] = match.group(1).strip()
                break
        
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
    
    return metadata

def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean scraped data with metadata extraction and concern extraction"""
    try:
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # Create a copy
        df = df.copy()

        # Extract metadata from Content field if it exists
        if "Content" in df.columns:
            # Process each row
            processed_rows = []
            for _, row in df.iterrows():
                # Start with original row data
                processed_row = row.to_dict()

                # Extract metadata using existing function
                content = str(row.get("Content", ""))
                metadata = extract_metadata(content)

                # Extract concerns text using the enhanced function
                processed_row["Extracted_Concerns"] = extract_concern_text(content)

                # Update row with metadata
                processed_row.update(metadata)
                processed_rows.append(processed_row)

            # Create new DataFrame from processed rows
            result = pd.DataFrame(processed_rows)
        else:
            result = df.copy()

        # Convert date_of_report to datetime with UK format handling
        if "date_of_report" in result.columns:

            def parse_date(date_str):
                if pd.isna(date_str):
                    return pd.NaT

                date_str = str(date_str).strip()

                # If already in DD/MM/YYYY format
                if re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
                    return pd.to_datetime(date_str, format="%d/%m/%Y")

                # Remove ordinal indicators
                date_str = re.sub(r"(\d)(st|nd|rd|th)", r"\1", date_str)

                # Try different formats
                formats = ["%Y-%m-%d", "%d-%m-%Y", "%d %B %Y", "%d %b %Y"]

                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except ValueError:
                        continue

                try:
                    return pd.to_datetime(date_str)
                except:
                    return pd.NaT

            result["date_of_report"] = result["date_of_report"].apply(parse_date)

        return result

    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df

def is_response_document(row: pd.Series) -> bool:
    """Check if a document is a response based on its metadata and content."""
    try:
        title = str(row.get('Title', '')).lower()
        url = str(row.get('URL', '')).lower()
        content = str(row.get('Content', '')).lower()
        
        # Response indicators in title/URL
        response_keywords = [
            'response', 'reply', 'answer', 'action taken',
            'implementation', 'progress', 'update'
        ]
        
        for keyword in response_keywords:
            if keyword in title or keyword in url:
                return True
        
        # Check content for response patterns
        response_patterns = [
            r'response\s+to\s+regulation\s+28',
            r'action\s+taken',
            r'in\s+response\s+to',
            r'following\s+the\s+regulation\s+28',
            r'we\s+have\s+implemented',
            r'steps\s+taken'
        ]
        
        for pattern in response_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
        
    except Exception as e:
        logging.error(f"Error in is_response_document: {e}")
        return False

def format_date_uk(date_obj) -> str:
    """Format date in UK format (DD/MM/YYYY)"""
    if pd.isna(date_obj):
        return ""
    
    try:
        if isinstance(date_obj, str):
            # Try to parse if it's a string
            date_obj = pd.to_datetime(date_obj)
        
        return date_obj.strftime("%d/%m/%Y")
    except:
        return str(date_obj)

def export_to_excel(data: pd.DataFrame, filename: str = None) -> bytes:
    """Export DataFrame to Excel format"""
    if filename is None:
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Data', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    except Exception as e:
        logging.error(f"Error exporting to Excel: {e}")
        raise

# Additional utility functions for PFD categories
def get_pfd_categories() -> List[str]:
    """Get standard PFD report categories - these are the actual UK coroner report categories"""
    return [
        "Accident at Work and Health and Safety related deaths",
        "Alcohol drug and medication related deaths",
        "Care Home Health related deaths", 
        "Child Death from 2015",
        "Community health care and emergency services related deaths",
        "Emergency services related deaths 2019 onwards",
        "Hospital Death Clinical Procedures and medical management related deaths",
        "Mental Health related deaths", 
        "Other related deaths", 
        "Police related deaths",
        "Product related deaths", 
        "Railway related deaths", 
        "Road Highways Safety related deaths",
        "Service Personnel related deaths", 
        "State Custody related deaths", 
        "Suicide from 2015",
        "Wales prevention of future deaths reports 2019 onwards"
    ]
