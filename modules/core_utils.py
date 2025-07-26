# modules/core_utils.py
# COMPLETE CORE UTILITIES FILE - With all missing classes and functions

import logging
import re
import os
import time
import hashlib
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import streamlit as st

# ===============================================
# MISSING DATACLASSES - CRITICAL FIXES
# ===============================================

@dataclass
class AnnotationResult:
    """
    Result from annotation process - THIS WAS MISSING AND CAUSING ERRORS
    """
    content_item: Any
    content_type: str
    annotations: Dict[str, List[Dict[str, Any]]]
    annotation_time: str
    confidence_score: float = 0.0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'content_type': self.content_type,
            'annotations': self.annotations,
            'annotation_time': self.annotation_time,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time
        }
    
    def get_themes_by_framework(self, framework_name: str) -> List[Dict[str, Any]]:
        """Get themes for a specific framework"""
        return self.annotations.get(framework_name, [])
    
    def get_all_themes(self) -> List[str]:
        """Get all theme names across all frameworks"""
        themes = []
        for framework_annotations in self.annotations.values():
            for annotation in framework_annotations:
                if 'theme' in annotation:
                    themes.append(annotation['theme'])
        return list(set(themes))

@dataclass
class Recommendation:
    """Recommendation data model"""
    text: str
    document_source: str
    section_title: str = ""
    recommendation_number: str = ""
    confidence: float = 0.0
    extraction_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and clean up recommendation data"""
        self.text = str(self.text).strip()
        self.document_source = str(self.document_source).strip()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class Response:
    """Response data model"""
    text: str
    document_source: str
    response_type: str = "unknown"  # accepted, rejected, partial, etc.
    related_recommendation: str = ""
    confidence: float = 0.0
    extraction_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_recommendations: int = 0
    annotated_recommendations: int = 0
    matched_recommendations: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add an error to the stats"""
        self.errors.append(f"{datetime.now().isoformat()}: {error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class ExtractionConfig:
    """Configuration for recommendation extraction"""
    method: str = "hybrid"  # "ai", "pattern", "hybrid"
    confidence_threshold: float = 0.6
    max_recommendations: int = 100
    include_context: bool = True
    min_text_length: int = 20
    
    def validate(self) -> bool:
        """Validate configuration"""
        return (
            self.confidence_threshold >= 0.0 and self.confidence_threshold <= 1.0 and
            self.max_recommendations > 0 and
            self.min_text_length > 0
        )

# ===============================================
# SECURITY AND VALIDATION
# ===============================================

class SecurityValidator:
    """Security validation utilities with support for large files up to Streamlit limits"""
    
    def __init__(self, max_file_size_mb: int = 200):  # Default to Streamlit's 200MB limit
        self.logger = logging.getLogger(__name__)
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self.allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        self.chunk_size = 8192  # 8KB chunks for memory-efficient processing
        
        # Log the current limit
        self.logger.info(f"SecurityValidator initialized with {max_file_size_mb}MB file size limit")
        
        # Warn if limit is higher than Streamlit default
        if max_file_size_mb > 200:
            self.logger.warning(
                f"File size limit ({max_file_size_mb}MB) exceeds Streamlit's default 200MB. "
                "Ensure server.maxUploadSize is configured in .streamlit/config.toml"
            )
        
    def validate_file_upload(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """Validate uploaded file for security - supports up to 500MB"""
        try:
            # Size check
            file_size_mb = len(file_content) / (1024 * 1024)
            if len(file_content) > self.max_file_size:
                return False, f"File too large: {file_size_mb:.1f}MB > {self.max_file_size/(1024*1024)}MB"
            
            # Extension check
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.allowed_extensions:
                return False, f"File type not allowed: {file_ext}"
            
            # Basic content validation for PDFs
            if file_ext == '.pdf':
                if not file_content.startswith(b'%PDF'):
                    return False, "Invalid PDF file format"
                
                # Additional PDF validation for large files
                if len(file_content) > 50 * 1024 * 1024:  # 50MB+
                    if not self._validate_large_pdf(file_content):
                        return False, "Large PDF file appears corrupted"
            
            self.logger.info(f"File validation passed: {filename} ({file_size_mb:.1f}MB)")
            return True, f"File validation passed ({file_size_mb:.1f}MB)"
            
        except Exception as e:
            self.logger.error(f"File validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_large_pdf(self, file_content: bytes) -> bool:
        """Validate large PDF files using streaming approach"""
        try:
            # Check for PDF trailer at end
            last_chunk = file_content[-1024:] if len(file_content) > 1024 else file_content
            if b'%%EOF' not in last_chunk and b'endobj' not in last_chunk:
                return False
            
            # Check for essential PDF structures in chunks
            chunk_size = 64 * 1024  # 64KB chunks
            essential_found = {'xref': False, 'obj': False}
            
            for i in range(0, min(len(file_content), 5 * chunk_size), chunk_size):
                chunk = file_content[i:i + chunk_size]
                if b'xref' in chunk:
                    essential_found['xref'] = True
                if b'obj' in chunk:
                    essential_found['obj'] = True
                
                if all(essential_found.values()):
                    break
            
            return any(essential_found.values())  # At least one essential structure found
            
        except Exception as e:
            self.logger.error(f"Large PDF validation error: {e}")
            return False
    
    def validate_file_stream(self, file_stream, filename: str, max_size_mb: int = 500) -> Tuple[bool, str]:
        """Validate file stream without loading entire file into memory"""
        try:
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.allowed_extensions:
                return False, f"File type not allowed: {file_ext}"
            
            # Read first chunk to validate format
            first_chunk = file_stream.read(8192)
            file_stream.seek(0)  # Reset stream position
            
            if file_ext == '.pdf' and not first_chunk.startswith(b'%PDF'):
                return False, "Invalid PDF file format"
            
            # Estimate file size by seeking to end
            file_stream.seek(0, 2)  # Seek to end
            file_size = file_stream.tell()
            file_stream.seek(0)  # Reset to beginning
            
            file_size_mb = file_size / (1024 * 1024)
            if file_size > max_size_mb * 1024 * 1024:
                return False, f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
            
            return True, f"Stream validation passed ({file_size_mb:.1f}MB)"
            
        except Exception as e:
            self.logger.error(f"Stream validation error: {e}")
            return False, f"Stream validation error: {str(e)}"
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text input"""
        if not isinstance(text, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', text)
        sanitized = re.sub(r'\s+', ' ', sanitized)
        return sanitized.strip()
    
    def validate_user_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate user input data - supports large content up to 500MB"""
        errors = []
        
        # Check for required fields
        required_fields = ['filename', 'content']
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate content length - increased limits for large files
        if 'content' in input_data:
            content = str(input_data['content'])
            if len(content) < 10:
                errors.append("Content too short (minimum 10 characters)")
            elif len(content) > 500 * 1024 * 1024:  # 500MB text limit
                errors.append("Content too long (maximum 500MB)")
        
        return len(errors) == 0, errors

# ===============================================
# LARGE FILE PROCESSING UTILITIES
# ===============================================

class LargeFileProcessor:
    """Handle processing of large files up to 500MB efficiently"""
    
    def __init__(self, chunk_size: int = 64 * 1024):  # 64KB default chunk size
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def process_large_text_file(self, file_content: str, processor_func) -> List[Any]:
        """Process large text files in chunks to avoid memory issues"""
        if len(file_content) < 1024 * 1024:  # Less than 1MB, process normally
            return processor_func(file_content)
        
        results = []
        total_chunks = len(file_content) // self.chunk_size + 1
        
        for i in range(0, len(file_content), self.chunk_size):
            chunk = file_content[i:i + self.chunk_size]
            
            # Ensure we don't cut words in half
            if i + self.chunk_size < len(file_content):
                # Find the last complete sentence or word boundary
                last_sentence = chunk.rfind('.')
                last_space = chunk.rfind(' ')
                
                if last_sentence > len(chunk) - 100:  # Sentence boundary within last 100 chars
                    chunk = chunk[:last_sentence + 1]
                elif last_space > len(chunk) - 50:   # Word boundary within last 50 chars
                    chunk = chunk[:last_space]
            
            try:
                chunk_results = processor_func(chunk)
                if chunk_results:
                    results.extend(chunk_results if isinstance(chunk_results, list) else [chunk_results])
            except Exception as e:
                self.logger.warning(f"Error processing chunk {i//self.chunk_size + 1}/{total_chunks}: {e}")
                continue
        
        return results
    
    def extract_text_with_overlap(self, content: str, pattern: str, overlap_size: int = 1000) -> List[str]:
        """Extract text patterns from large content with overlapping chunks"""
        if len(content) < self.chunk_size:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            return matches
        
        results = []
        seen_matches = set()
        
        for i in range(0, len(content), self.chunk_size - overlap_size):
            chunk_end = min(i + self.chunk_size, len(content))
            chunk = content[i:chunk_end]
            
            try:
                matches = re.findall(pattern, chunk, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Simple deduplication based on first 100 characters
                    match_key = match[:100] if isinstance(match, str) else str(match)[:100]
                    if match_key not in seen_matches:
                        seen_matches.add(match_key)
                        results.append(match)
            except Exception as e:
                self.logger.warning(f"Pattern matching error in chunk: {e}")
                continue
        
        return results
    
    def memory_efficient_deduplication(self, items: List[Dict[str, Any]], 
                                     key_field: str = 'text', 
                                     similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Memory-efficient deduplication for large datasets"""
        if len(items) < 1000:  # Small dataset, use normal deduplication
            return deduplicate_recommendations(items, similarity_threshold)
        
        # For large datasets, use hash-based deduplication first
        hash_groups = {}
        for item in items:
            text = str(item.get(key_field, ''))
            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]  # First 16 chars of hash
            
            if text_hash not in hash_groups:
                hash_groups[text_hash] = []
            hash_groups[text_hash].append(item)
        
        # Then apply similarity deduplication within each hash group
        deduplicated = []
        for group in hash_groups.values():
            if len(group) == 1:
                deduplicated.extend(group)
            else:
                # Apply detailed deduplication only to items with same hash prefix
                group_deduplicated = deduplicate_recommendations(group, similarity_threshold)
                deduplicated.extend(group_deduplicated)
        
        return deduplicated

# Global large file processor instance
large_file_processor = LargeFileProcessor()

# ===============================================
# LOGGING AND MONITORING
# ===============================================

def log_user_action(action: str, details: Dict[str, Any] = None, user_id: str = "anonymous"):
    """Log user actions for monitoring"""
    logger = logging.getLogger("user_actions")
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'action': action,
        'details': details or {}
    }
    
    logger.info(f"USER_ACTION: {log_entry}")

def setup_logging(log_level: str = "INFO"):
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', mode='a')
        ]
    )

# ===============================================
# DATA PROCESSING UTILITIES
# ===============================================

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    cleaned = re.sub(r'[^\w\s\.,!?;:\-\(\)"\']', '', cleaned)
    
    # Normalize quotes
    cleaned = re.sub(r'["""]', '"', cleaned)
    cleaned = re.sub(r'[''']', "'", cleaned)
    
    return cleaned.strip()

def extract_recommendation_number(text: str) -> Optional[str]:
    """Extract recommendation number from text"""
    patterns = [
        r'Recommendation\s+(\d+[a-z]*)',
        r'R(\d+[a-z]*)',
        r'(\d+[a-z]*)\s*[:\.)]\s*',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate basic text similarity score"""
    if not text1 or not text2:
        return 0.0
    
    # Simple word overlap similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def deduplicate_recommendations(recommendations: List[Dict[str, Any]], similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Remove duplicate recommendations based on text similarity"""
    if not recommendations:
        return []
    
    unique_recommendations = []
    
    for rec in recommendations:
        is_duplicate = False
        rec_text = rec.get('text', '')
        
        for unique_rec in unique_recommendations:
            unique_text = unique_rec.get('text', '')
            similarity = calculate_text_similarity(rec_text, unique_text)
            
            if similarity > similarity_threshold:
                is_duplicate = True
                # Keep the one with higher confidence
                if rec.get('confidence', 0) > unique_rec.get('confidence', 0):
                    unique_recommendations.remove(unique_rec)
                    unique_recommendations.append(rec)
                break
        
        if not is_duplicate:
            unique_recommendations.append(rec)
    
    return unique_recommendations

def group_by_document_type(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group data by document type"""
    try:
        if 'document_type' not in data.columns:
            return {'all': data}
        
        groups = {}
        for doc_type in data['document_type'].unique():
            groups[doc_type] = data[data['document_type'] == doc_type]
        
        return groups
        
    except Exception as e:
        logging.error(f"Error grouping by document type: {e}")
        return {'all': data}

def is_response_document(row: pd.Series) -> bool:
    """Determine if a document row represents a response document"""
    try:
        # Check filename for response indicators
        filename = str(row.get('filename', '')).lower()
        response_indicators = ['response', 'reply', 'government response', 'answer']
        
        for indicator in response_indicators:
            if indicator in filename:
                return True
        
        # Check title
        title = str(row.get('title', '')).lower()
        for indicator in response_indicators:
            if indicator in title:
                return True
        
        # Check content for response patterns
        content = str(row.get('content', '')).lower()
        if any(phrase in content for phrase in ['accepts this recommendation', 'government response', 'in response to']):
            return True
        
        return False
        
    except Exception as e:
        logging.error(f"Error determining document type: {e}")
        return False

def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """Validate DataFrame has required structure"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, missing_columns
    
    return True, []

def safe_convert_to_dataframe(data: Any) -> pd.DataFrame:
    """Safely convert various data types to DataFrame"""
    try:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame({'data': data})
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error converting to DataFrame: {e}")
        return pd.DataFrame()

def deduplicate_dataframe_content(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate content from DataFrame"""
    try:
        if data.empty:
            return data
        
        # Create a hash column for deduplication
        if 'content' in data.columns:
            data['content_hash'] = data['content'].apply(
                lambda x: hashlib.md5(str(x).encode()).hexdigest() if pd.notna(x) else ''
            )
            
            # Remove duplicates based on content hash
            data_deduped = data.drop_duplicates(subset=['content_hash'], keep='first')
            data_deduped = data_deduped.drop(columns=['content_hash'])
            
            logging.info(f"Deduplicated {len(data)} -> {len(data_deduped)} documents")
            return data_deduped
        
        return data
        
    except Exception as dedup_error:
        logging.error(f"Error in deduplication: {dedup_error}")
        return data

def combine_document_text(row: pd.Series) -> str:
    """Combine all text content from a document row for analysis"""
    text_parts = []
    
    try:
        if pd.notna(row.get("Title")): 
            text_parts.append(str(row["Title"]))
        if pd.notna(row.get("Content")): 
            text_parts.append(str(row["Content"]))
        
        # Iterate through potential PDF content columns
        for i in range(1, 10):  # Assuming up to PDF_9_Content
            pdf_col_name = f"PDF_{i}_Content"
            if pdf_col_name in row.index and pd.notna(row.get(pdf_col_name)):
                text_parts.append(str(row[pdf_col_name]))
                
        return " ".join(text_parts)
        
    except Exception as combine_error:
        logging.error(f"Error combining document text: {combine_error}")
        return ""

def filter_by_document_types(df: pd.DataFrame, doc_types: List[str]) -> pd.DataFrame:
    """Filter DataFrame by document types (Report/Response)"""
    if not doc_types: 
        return df
    
    try:
        is_response_series = df.apply(is_response_document, axis=1)
        
        if "Report" in doc_types and "Response" not in doc_types:
            return df[~is_response_series]
        elif "Response" in doc_types and "Report" not in doc_types:
            return df[is_response_series]
        # If both are selected, return all
        return df
        
    except Exception as filter_error:
        logging.error(f"Error filtering by document types: {filter_error}")
        return df

def filter_by_categories(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """Filter DataFrame by categories if category column exists"""
    try:
        if not categories or 'Category' not in df.columns:
            return df
        
        return df[df['Category'].isin(categories)]
        
    except Exception as filter_error:
        logging.error(f"Error filtering by categories: {filter_error}")
        return df

# ===============================================
# SESSION STATE MANAGEMENT
# ===============================================

def initialize_session_state():
    """Initialize Streamlit session state with default values"""
    default_values = {
        'uploaded_documents': [],
        'extracted_recommendations': [],
        'extracted_responses': [],
        'annotation_results': {},
        'matching_results': {},
        'processing_stats': ProcessingStats(),
        'extraction_config': ExtractionConfig(),
        'current_framework': None,
        'annotation_frameworks': {},
        'search_results': [],
        'vector_store_manager': None,
        'bert_annotator': None,
        'last_action': None,
        'debug_mode': False
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_session_state():
    """Clear all session state data"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

def get_session_stats() -> Dict[str, Any]:
    """Get current session statistics"""
    return {
        'documents_uploaded': len(st.session_state.get('uploaded_documents', [])),
        'recommendations_extracted': len(st.session_state.get('extracted_recommendations', [])),
        'responses_extracted': len(st.session_state.get('extracted_responses', [])),
        'annotations_completed': len(st.session_state.get('annotation_results', {})),
        'frameworks_loaded': len(st.session_state.get('annotation_frameworks', {})),
        'last_action': st.session_state.get('last_action', 'None')
    }

# ===============================================
# ERROR HANDLING AND RECOVERY
# ===============================================

def handle_processing_error(error: Exception, context: str) -> Dict[str, Any]:
    """Handle processing errors gracefully"""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'timestamp': datetime.now().isoformat()
    }
    
    logging.error(f"Processing error in {context}: {error}")
    
    # Update session state with error
    if 'processing_stats' in st.session_state:
        st.session_state.processing_stats.add_error(f"{context}: {str(error)}")
    
    return error_info

def validate_processing_pipeline() -> Tuple[bool, List[str]]:
    """Validate that the processing pipeline is ready"""
    issues = []
    
    # Check session state
    if 'uploaded_documents' not in st.session_state:
        issues.append("Session state not properly initialized")
    
    # Check uploaded documents
    docs = st.session_state.get('uploaded_documents', [])
    if not docs:
        issues.append("No documents uploaded")
    
    # Check document content
    valid_docs = 0
    for doc in docs:
        if isinstance(doc, dict) and doc.get('content'):
            valid_docs += 1
    
    if valid_docs == 0:
        issues.append("No documents contain readable content")
    
    return len(issues) == 0, issues

# ===============================================
# PERFORMANCE MONITORING
# ===============================================

class PerformanceMonitor:
    """Monitor application performance"""
    
    def __init__(self):
        self.start_times = {}
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = duration
            del self.start_times[operation]
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all performance metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# ===============================================
# EXPORT FUNCTIONS
# ===============================================

__all__ = [
    # Data classes
    'AnnotationResult',
    'Recommendation', 
    'Response',
    'ProcessingStats',
    'ExtractionConfig',
    
    # Security and validation
    'SecurityValidator',
    'log_user_action',
    'setup_logging',
    
    # Large file processing
    'LargeFileProcessor',
    'large_file_processor',
    
    # Data processing
    'clean_extracted_text',
    'extract_recommendation_number',
    'calculate_text_similarity',
    'deduplicate_recommendations',
    'group_by_document_type',
    'is_response_document',
    'validate_dataframe_structure',
    'safe_convert_to_dataframe',
    'deduplicate_dataframe_content',
    'combine_document_text',
    'filter_by_document_types',
    'filter_by_categories',
    
    # Session state management
    'initialize_session_state',
    'clear_session_state',
    'get_session_stats',
    
    # Error handling
    'handle_processing_error',
    'validate_processing_pipeline',
    
    # Performance monitoring
    'PerformanceMonitor',
    'performance_monitor'
]
