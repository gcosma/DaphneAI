# modules/__init__.py
# Main package initialization for document search application
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Package metadata
__version__ = "2.1.0"  # Bumped for strict extractor
__description__ = "Advanced Document Search System with RAG + Smart Search"
__author__ = "DaphneAI Team"

# =============================================================================
# RECOMMENDATION EXTRACTOR - UPDATED to use StrictRecommendationExtractor
# =============================================================================
# Priority: Use the strict extractor (simple_recommendation_extractor.py)
# which eliminates ~90% of false positives from PDF artifacts, URLs, timestamps

try:
    # FIRST: Try to import the STRICT extractor (the good one)
    from .simple_recommendation_extractor import (
        extract_recommendations,
        StrictRecommendationExtractor,
    )
    # Alias for backward compatibility
    AdvancedRecommendationExtractor = StrictRecommendationExtractor
    RECOMMENDATION_EXTRACTOR_AVAILABLE = True
    logging.info("Using StrictRecommendationExtractor (improved filtering)")
    
except ImportError as e:
    logging.warning(f"Could not import simple_recommendation_extractor: {e}")
    
    # FALLBACK: Try the old extractor
    try:
        from .extractors.recommendation_extractor import (
            extract_recommendations,
            AdvancedRecommendationExtractor
        )
        RECOMMENDATION_EXTRACTOR_AVAILABLE = True
        logging.warning("Using legacy AdvancedRecommendationExtractor (may have false positives)")
        
    except ImportError as e2:
        logging.warning(f"Could not import any recommendation_extractor: {e2}")
        RECOMMENDATION_EXTRACTOR_AVAILABLE = False
        
        def extract_recommendations(text, min_confidence=0.7):
            logging.error("Recommendation extractor not available")
            return []
        
        class AdvancedRecommendationExtractor:
            def __init__(self):
                pass
            def extract_recommendations(self, text, min_confidence=0.7):
                return []
            def get_statistics(self, recommendations):
                return {'total': 0, 'methods': {}, 'top_verbs': {}, 'avg_confidence': 0}

# =============================================================================
# CORE UTILITIES
# =============================================================================
try:
    from .core_utils import setup_logging, log_action, search_analytics
    CORE_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import core_utils: {e}")
    CORE_UTILS_AVAILABLE = False
    
    def setup_logging():
        return logging.getLogger(__name__)
    
    def log_action(action, data=None):
        logging.info(f"Action: {action}, Data: {data}")
    
    def search_analytics():
        return {'total_searches': 0, 'total_uploads': 0}

# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================
try:
    from .document_processor import process_uploaded_files, get_processing_stats, check_dependencies
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import document_processor: {e}")
    DOCUMENT_PROCESSOR_AVAILABLE = False
    
    def process_uploaded_files(files):
        return [{'filename': f.name, 'error': 'Document processor not available'} for f in files]
    
    def get_processing_stats():
        return {'documents_processed': 0}
    
    def check_dependencies():
        return {'pdfplumber': False, 'PyPDF2': False, 'python-docx': False}

# =============================================================================
# UI COMPONENTS
# =============================================================================
try:
    from .ui import render_search_interface, render_recommendation_alignment_interface
    UI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import UI components: {e}")
    UI_AVAILABLE = False
    
    def render_search_interface(documents):
        import streamlit as st
        st.error("Search interface components not available")
        st.info("Basic search functionality is still available through the simple interface")
    
    def render_recommendation_alignment_interface(documents):
        import streamlit as st
        st.error("Recommendation alignment components not available")

# =============================================================================
# INTEGRATION HELPER
# =============================================================================
try:
    from .integration_helper import (
        setup_search_tab, 
        prepare_documents_for_search, 
        extract_text_from_file,
        render_analytics_tab
    )
    INTEGRATION_HELPER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import integration_helper: {e}")
    INTEGRATION_HELPER_AVAILABLE = False
    
    def setup_search_tab():
        import streamlit as st
        st.error("Integration helper not available")
    
    def prepare_documents_for_search(files, extract_func):
        return [{'filename': f.name, 'text': 'Basic text extraction'} for f in files]
    
    def extract_text_from_file(file):
        return "Text extraction not available"
    
    def render_analytics_tab():
        import streamlit as st
        st.info("Analytics not available")

# =============================================================================
# PACKAGE EXPORTS
# =============================================================================
__all__ = [
    # Recommendation extractor (now uses strict version)
    'extract_recommendations',
    'AdvancedRecommendationExtractor',  # Alias to StrictRecommendationExtractor
    
    # Core utilities
    'setup_logging',
    'log_action', 
    'search_analytics',
    
    # Document processing
    'process_uploaded_files',
    'get_processing_stats',
    'check_dependencies',
    
    # UI components
    'render_search_interface',
    'render_recommendation_alignment_interface',
    
    # Integration helpers
    'setup_search_tab',
    'prepare_documents_for_search',
    'extract_text_from_file',
    'render_analytics_tab',
    
    # Availability flags
    'CORE_UTILS_AVAILABLE',
    'DOCUMENT_PROCESSOR_AVAILABLE',
    'UI_AVAILABLE',
    'INTEGRATION_HELPER_AVAILABLE',
    'RECOMMENDATION_EXTRACTOR_AVAILABLE'
]

# =============================================================================
# PACKAGE STATUS
# =============================================================================
logger = logging.getLogger(__name__)
logger.info(f"Initialized document search package v{__version__}")
logger.info(f"Core utils: {'✓' if CORE_UTILS_AVAILABLE else '✗'}")
logger.info(f"Document processor: {'✓' if DOCUMENT_PROCESSOR_AVAILABLE else '✗'}")
logger.info(f"UI components: {'✓' if UI_AVAILABLE else '✗'}")
logger.info(f"Integration helper: {'✓' if INTEGRATION_HELPER_AVAILABLE else '✗'}")
logger.info(f"Recommendation extractor: {'✓' if RECOMMENDATION_EXTRACTOR_AVAILABLE else '✗'}")


def get_package_status():
    """Get the status of all package components"""
    return {
        'version': __version__,
        'core_utils': CORE_UTILS_AVAILABLE,
        'document_processor': DOCUMENT_PROCESSOR_AVAILABLE, 
        'ui_components': UI_AVAILABLE,
        'integration_helper': INTEGRATION_HELPER_AVAILABLE,
        'recommendation_extractor': RECOMMENDATION_EXTRACTOR_AVAILABLE,
        'all_available': all([
            CORE_UTILS_AVAILABLE,
            DOCUMENT_PROCESSOR_AVAILABLE, 
            UI_AVAILABLE,
            INTEGRATION_HELPER_AVAILABLE,
            RECOMMENDATION_EXTRACTOR_AVAILABLE
        ])
    }
