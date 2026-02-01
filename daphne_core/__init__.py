"""
Daphne Core - Recommendation Extraction and Alignment Engine

v4.1 - Fixed regex patterns for decimal IDs, improved status classification,
       and restored missing utility functions.

Main components:
- recommendation_extractor: Extract recommendations from documents
- response_extractor: Extract responses from government/HSIB documents  
- alignment_engine: Match recommendations with responses, classify status
- format_detection: Detect document format for appropriate extraction strategy

Usage:
    from daphne_core import (
        extract_recommendations,
        extract_responses,
        align_recommendations_responses,
        classify_response_status,
        detect_document_format,
    )
    
    # Extract recommendations
    recs = extract_recommendations(recommendation_text)
    
    # Extract responses
    responses = extract_responses(response_text)
    
    # Align and classify
    aligned = align_recommendations_responses(recs, responses)
    
    for item in aligned:
        print(f"Rec {item['rec_id']}: {item['status']}")
"""

__version__ = '4.1.0'
__author__ = 'Daphne Project'

# Import main functions for easy access
from .recommendation_extractor import (
    extract_recommendations,
    StrictRecommendationExtractor,
    AdvancedRecommendationExtractor,  # Alias for backward compat
)

from .response_extractor import (
    extract_responses,
    extract_government_responses,
    extract_decimal_government_responses,
    extract_hsib_responses,
    extract_trust_responses,
    extract_hssib_org_structured_responses,
    extract_org_based_hsib_responses,
    get_responses_for_document,
    is_decimal_government_response,
    is_hsib_response_document,
    is_trust_response_document,
)

from .alignment_engine import (
    # Classes
    AlignmentEngine,
    StatusClassifier,
    RecommendationResponseMatcher,
    
    # Main alignment functions
    align_recommendations_responses,
    align_recommendations_with_responses,  # Backward compat alias
    classify_response_status,
    get_status_classification_details,
    
    # Utility functions (restored in v4.1)
    calculate_simple_similarity,
    classify_content_type,
    determine_alignment_status,
    find_pattern_matches,
    extract_response_sentences,
    
    # Constants
    STOP_WORDS,
)

from .format_detection import (
    detect_document_format,
    detect_response_document_format,
    get_format_type,
    get_recommended_extractor,
    is_structured_document,
    FormatDetectionResult,
)

__all__ = [
    # Version
    '__version__',
    
    # Recommendation extraction
    'extract_recommendations',
    'StrictRecommendationExtractor',
    'AdvancedRecommendationExtractor',
    
    # Response extraction
    'extract_responses',
    'extract_government_responses',
    'extract_decimal_government_responses',
    'extract_hsib_responses',
    'extract_trust_responses',
    'extract_hssib_org_structured_responses',
    'extract_org_based_hsib_responses',
    'get_responses_for_document',
    'is_decimal_government_response',
    'is_hsib_response_document',
    'is_trust_response_document',
    
    # Alignment - Classes
    'AlignmentEngine',
    'StatusClassifier',
    'RecommendationResponseMatcher',
    
    # Alignment - Functions
    'align_recommendations_responses',
    'align_recommendations_with_responses',  # Backward compat alias
    'classify_response_status',
    'get_status_classification_details',
    
    # Alignment - Utility functions (restored in v4.1)
    'calculate_simple_similarity',
    'classify_content_type',
    'determine_alignment_status',
    'find_pattern_matches',
    'extract_response_sentences',
    
    # Alignment - Constants
    'STOP_WORDS',
    
    # Format detection
    'detect_document_format',
    'detect_response_document_format',
    'get_format_type',
    'get_recommended_extractor',
    'is_structured_document',
    'FormatDetectionResult',
]
