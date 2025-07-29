# modules/ui/recommendation_alignment_core.py
"""
Core Functions for Recommendation-Response Alignment System
Handles data structures, configuration, and core utilities
"""

import re
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# ADVANCED DATA STRUCTURES
# =============================================================================

@dataclass
class ContentItem:
    """Enhanced data structure for recommendations and responses"""
    id: str
    document: Dict[str, Any]
    sentence: str
    context: str
    patterns_matched: List[Dict[str, Any]]
    confidence_score: float
    content_type: str
    topic_classification: str
    urgency_score: float
    formal_language_score: float
    position_info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_flags: Dict[str, bool] = field(default_factory=dict)
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AlignmentMatch:
    """Enhanced data structure for alignment matches"""
    recommendation: ContentItem
    response: ContentItem
    similarity_scores: Dict[str, float]
    combined_score: float
    match_quality: str
    confidence_level: str
    explanation: str
    validation_results: Dict[str, Any]
    cross_document: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class AlignmentConfig:
    """Centralized configuration for alignment system"""
    
    # Quality thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.4
    HIGH_QUALITY_THRESHOLD = 0.7
    EXCELLENT_QUALITY_THRESHOLD = 0.85
    
    # Text processing
    MIN_SENTENCE_LENGTH = 25
    MIN_WORD_COUNT = 6
    MAX_SENTENCE_LENGTH = 2000
    CONTEXT_WINDOW_SIZE = 3
    
    # Similarity weights
    SIMILARITY_WEIGHTS = {
        'semantic_similarity': 0.35,
        'contextual_similarity': 0.20,
        'document_relevance': 0.25,
        'topic_alignment': 0.15,
        'temporal_proximity': 0.03,
        'cross_reference': 0.02
    }
    
    # Pattern recognition
    RECOMMENDATION_PATTERNS = {
        'strong': [
            'we recommend', 'committee recommends', 'report recommends',
            'it is recommended', 'recommendation that', 'we suggest',
            'committee suggests', 'we propose', 'we advise'
        ],
        'moderate': [
            'should be', 'must be', 'ought to be', 'needs to be',
            'requires', 'call for', 'urge that'
        ],
        'weak': [
            'consider', 'might', 'could', 'perhaps', 'possibly'
        ]
    }
    
    RESPONSE_PATTERNS = {
        'strong': [
            'government accepts', 'department agrees', 'we accept',
            'we agree', 'will implement', 'has implemented',
            'government response', 'official response'
        ],
        'moderate': [
            'in response', 'our response', 'we consider',
            'under review', 'being considered', 'will consider'
        ],
        'weak': [
            'note', 'acknowledge', 'aware of', 'understand'
        ]
    }
    
    # 🛡️ CRITICAL: Narrative exclusion patterns (prevents your false positive issue)
    NARRATIVE_EXCLUSION_PATTERNS = [
        'was advised', 'were advised', 'meeting was advised',
        'directors were advised', 'members were advised',
        'they were told', 'he was told', 'she was told',
        'it was suggested to', 'professor ', 'dr ',
        'witness statement', 'written statement of',
        'infected blood inquiry', 'the report', 'volume ',
        'commentary on the government response',
        'see section', 'see page', 'see appendix'
    ]

# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def clean_text_for_processing(text: str) -> str:
    """Clean text for optimal processing"""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR issues
    # Fix common OCR quote issues
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    
    # Handle hyphenated line breaks
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    return text.strip()

def split_sentences_with_intelligence(text: str) -> List[str]:
    """Intelligent sentence splitting for government documents"""
    
    # Preprocess text
    text = clean_text_for_processing(text)
    
    # Advanced sentence boundary detection
    # Handle government-specific patterns like numbered recommendations
    sentence_boundaries = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\d+\.)\s+(?=[A-Z])|(?<=\w\.)\s+(?=[A-Z][^A-Z])'
    
    sentences = re.split(sentence_boundaries, text)
    
    # Post-process sentences
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Length and quality checks
        if (len(sentence) >= AlignmentConfig.MIN_SENTENCE_LENGTH and 
            len(sentence.split()) >= AlignmentConfig.MIN_WORD_COUNT and
            len(sentence) <= AlignmentConfig.MAX_SENTENCE_LENGTH):
            
            # Remove obvious metadata and navigation
            if not is_metadata_sentence(sentence):
                clean_sentences.append(sentence)
    
    return clean_sentences

def passes_basic_validation(sentence: str) -> bool:
    """Basic validation with enhanced checks"""
    
    if not sentence or len(sentence.strip()) < AlignmentConfig.MIN_SENTENCE_LENGTH:
        return False
    
    words = sentence.split()
    if len(words) < AlignmentConfig.MIN_WORD_COUNT:
        return False
    
    # Check for invalid patterns
    invalid_patterns = [
        r'^\d+\s*$',  # Just numbers
        r'^page \d+',  # Page numbers
        r'^figure \d+',  # Figure references
        r'^table \d+',  # Table references
        r'^\w+\s*:\s*$',  # Headers only
        r'^see (page|section|appendix)',  # Cross-references
        r'^\([a-z]\)\s*$',  # List markers only
        r'^\d{3,5}\s+[A-Z]\w+$'  # Legal numbering only
    ]
    
    sentence_lower = sentence.lower()
    for pattern in invalid_patterns:
        if re.match(pattern, sentence_lower):
            return False
    
    # Content quality check
    meaningful_word_ratio = calculate_meaningful_word_ratio(sentence)
    if meaningful_word_ratio < 0.3:
        return False
    
    return True

def is_narrative_content(sentence: str, filtering_level: str) -> bool:
    """🛡️ CRITICAL: Advanced narrative content detection - FIXES YOUR ISSUE"""
    
    sentence_lower = sentence.lower()
    
    # Aggressive filtering (recommended) - DIRECTLY FIXES YOUR ISSUE
    if filtering_level == "Aggressive":
        exclusion_patterns = AlignmentConfig.NARRATIVE_EXCLUSION_PATTERNS + [
            'the meeting was advised',  # DIRECT FIX FOR YOUR SPECIFIC ISSUE
            'meeting was advised',      # Alternative form
            'directors were advised',   # Similar pattern
            'professor ', 'dr ', 'mr ', 'mrs ', 'ms ',
            'witness number', 'witn', 'para ', 'paragraph ',
            'statement of', 'written by', 'evidence from'
        ]
    
    # Moderate filtering
    elif filtering_level == "Moderate":
        exclusion_patterns = AlignmentConfig.NARRATIVE_EXCLUSION_PATTERNS
    
    # Conservative filtering
    elif filtering_level == "Conservative":
        exclusion_patterns = [
            'was advised', 'were advised', 'meeting was advised',
            'witness statement', 'written statement'
        ]
    
    else:  # Disabled
        return False
    
    # Check for exclusion patterns
    for pattern in exclusion_patterns:
        if pattern in sentence_lower:
            return True
    
    # Additional narrative indicators
    narrative_score = 0
    
    # Personal pronouns in narrative context
    if re.search(r'\b(he|she|they) (said|told|advised|suggested)\b', sentence_lower):
        narrative_score += 0.5
    
    # Past tense reporting verbs
    reporting_verbs = ['explained', 'described', 'mentioned', 'indicated', 'stated', 'noted']
    for verb in reporting_verbs:
        if verb in sentence_lower:
            narrative_score += 0.2
    
    return narrative_score > 0.7

def is_metadata_sentence(sentence: str) -> bool:
    """Enhanced metadata sentence detection"""
    
    sentence_lower = sentence.lower().strip()
    
    # Common metadata patterns
    metadata_patterns = [
        r'^infected blood inquiry\b',
        r'^the report\b',
        r'^volume \d+',
        r'^chapter \d+',
        r'^section \d+',
        r'^appendix \w+',
        r'^page \d+',
        r'^table of contents',
        r'^bibliography',
        r'^references',
        r'^acknowledgments',
        r'^executive summary$',
        r'^introduction$',
        r'^methodology$',
        r'^conclusions?$',
        r'^\d{1,4}\s*$'  # Just numbers
    ]
    
    for pattern in metadata_patterns:
        if re.match(pattern, sentence_lower):
            return True
    
    # Check for document navigation
    if any(phrase in sentence_lower for phrase in [
        'see section', 'see page', 'see appendix', 'see volume',
        'refer to', 'as described in', 'mentioned above',
        'discussed below', 'outlined in'
    ]):
        return True
    
    return False

def calculate_meaningful_word_ratio(sentence: str) -> float:
    """Calculate ratio of meaningful words in sentence"""
    
    words = sentence.split()
    if not words:
        return 0.0
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    meaningful_words = [word for word in words 
                       if word.lower() not in stop_words and len(word) > 2]
    
    return len(meaningful_words) / len(words)

def calculate_character_position(text: str, sentence: str) -> int:
    """Calculate accurate character position of sentence in text"""
    
    try:
        position = text.find(sentence)
        return position if position != -1 else 0
    except Exception:
        return 0

def estimate_page_number_intelligent(char_position: int, text: str) -> int:
    """Intelligent page number estimation"""
    
    if char_position <= 0:
        return 1
    
    # Look for actual page markers first
    text_before = text[:char_position + 200]  # Include some text after position
    
    # Common page marker patterns
    page_patterns = [
        r'page\s+(\d+)',
        r'p\.?\s*(\d+)',
        r'^\s*(\d+)\s*$',  # Standalone numbers
        r'page\s*(\d+)\s*of\s*\d+'
    ]
    
    page_numbers = []
    for pattern in page_patterns:
        matches = re.findall(pattern, text_before, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            try:
                page_num = int(match)
                if 1 <= page_num <= 10000:  # Reasonable page range
                    page_numbers.append(page_num)
            except ValueError:
                continue
    
    if page_numbers:
        return max(page_numbers)  # Use the highest page number found
    
    # Fallback to character-based estimation
    # Government documents typically have 2000-2500 characters per page
    estimated_page = max(1, char_position // 2200 + 1)
    
    return min(estimated_page, 9999)  # Cap at reasonable maximum

# =============================================================================
# ENHANCED DOCUMENT METADATA FUNCTIONS
# =============================================================================

def enhance_document_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance document with intelligent metadata extraction"""
    
    enhanced_doc = doc.copy()
    text = doc.get('text', '')
    filename = doc.get('filename', '')
    
    # Extract comprehensive metadata
    metadata = {
        'original_filename': filename,
        'document_type': classify_document_type_advanced(filename, text),
        'authority_level': determine_authority_level_advanced(filename, text),
        'estimated_pages': max(1, len(text) // 2000),
        'word_count': len(text.split()),
        'character_count': len(text),
        'language_formality': calculate_language_formality(text),
        'content_density': calculate_content_density(text),
        'processing_timestamp': datetime.now().isoformat(),
        'content_hash': hashlib.md5(text.encode()).hexdigest()[:16],
        'quality_score': assess_document_quality(text)
    }
    
    enhanced_doc['enhanced_metadata'] = metadata
    
    return enhanced_doc

def classify_document_type_advanced(filename: str, text: str) -> str:
    """Advanced document type classification with content analysis"""
    
    filename_lower = filename.lower()
    text_sample = text[:2000].lower()  # First 2000 chars for classification
    
    # Government response documents
    if any(indicator in filename_lower for indicator in ['response', 'government']):
        if 'committee' in text_sample or 'recommendation' in text_sample:
            return 'Government Response to Committee'
        elif 'inquiry' in text_sample:
            return 'Government Response to Inquiry'
        else:
            return 'Government Response Document'
    
    # Committee and inquiry reports
    elif any(indicator in filename_lower for indicator in ['committee', 'inquiry']):
        if 'recommendation' in text_sample:
            return 'Committee Report with Recommendations'
        else:
            return 'Committee/Inquiry Report'
    
    # Multi-volume sets
    elif 'volume' in filename_lower:
        volume_match = re.search(r'volume[_\s]*(\d+)', filename_lower)
        if volume_match:
            return f'Multi-Volume Report (Volume {volume_match.group(1)})'
        else:
            return 'Multi-Volume Report'
    
    # Parliamentary documents
    elif any(indicator in filename_lower for indicator in ['parliament', 'mp', 'house of']):
        return 'Parliamentary Document'
    
    # Policy documents
    elif any(indicator in filename_lower for indicator in ['policy', 'framework', 'strategy']):
        return 'Policy Document'
    
    # Analyze content for classification
    else:
        if text_sample.count('recommend') > 5:
            return 'Document with Recommendations'
        elif text_sample.count('response') > 5:
            return 'Response Document'
        elif text_sample.count('policy') > 5:
            return 'Policy-Related Document'
        else:
            return 'General Government Document'

def determine_authority_level_advanced(filename: str, text: str) -> str:
    """Determine document authority level with enhanced analysis"""
    
    filename_lower = filename.lower()
    text_sample = text[:1000].lower()
    
    # Highest authority indicators
    high_authority_indicators = [
        'prime minister', 'secretary of state', 'cabinet',
        'parliament', 'house of commons', 'house of lords',
        'crown copyright', 'her majesty', 'his majesty'
    ]
    
    # Medium authority indicators  
    medium_authority_indicators = [
        'department', 'ministry', 'government',
        'official', 'public inquiry', 'committee'
    ]
    
    if any(indicator in filename_lower or indicator in text_sample 
           for indicator in high_authority_indicators):
        return 'High Authority'
    elif any(indicator in filename_lower or indicator in text_sample 
             for indicator in medium_authority_indicators):
        return 'Medium Authority'
    else:
        return 'Standard Authority'

def calculate_language_formality(text: str) -> float:
    """Calculate language formality score"""
    
    if not text:
        return 0.0
    
    formal_indicators = [
        'committee', 'government', 'department', 'ministry',
        'recommendation', 'implementation', 'consideration',
        'furthermore', 'therefore', 'consequently',
        'shall', 'pursuant', 'whereas', 'aforementioned'
    ]
    
    informal_indicators = [
        'basically', 'really', 'pretty much', 'kind of',
        'stuff', 'things', 'okay', 'yeah'
    ]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
    informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
    
    # Normalize by text length
    formal_score = formal_count / len(words) * 1000 if words else 0
    informal_penalty = informal_count / len(words) * 1000 if words else 0
    
    return min(max(formal_score - informal_penalty, 0), 1.0)

def calculate_content_density(text: str) -> float:
    """Calculate content density (meaningful words per total words)"""
    
    if not text:
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    # Common stop words and meaningless words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'
    }
    
    meaningful_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
    
    return len(meaningful_words) / len(words)

def assess_document_quality(text: str) -> float:
    """Assess overall document quality"""
    
    if not text:
        return 0.0
    
    quality_factors = {
        'length': min(len(text) / 10000, 1.0),  # Prefer longer documents
        'content_density': calculate_content_density(text),
        'formality': calculate_language_formality(text),
    }
    
    return sum(quality_factors.values()) / len(quality_factors)

# Export all functions
__all__ = [
    'ContentItem', 'AlignmentMatch', 'AlignmentConfig',
    'clean_text_for_processing', 'split_sentences_with_intelligence',
    'passes_basic_validation', 'is_narrative_content', 'is_metadata_sentence',
    'calculate_meaningful_word_ratio', 'calculate_character_position',
    'estimate_page_number_intelligent', 'enhance_document_metadata',
    'classify_document_type_advanced', 'determine_authority_level_advanced'
]
