# ===============================================
# FILE: modules/core_utils.py
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
