# modules/__init__.py
# Main package initialization for document search application

import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__version__ = "1.0.0"
__description__ = "Advanced Document Search System with RAG + Smart Search"
__author__ = "Your Name"

# Import functions with proper error handling
try:
    from .core_utils import setup_logging, log_action, search_analytics
    CORE_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import core_utils: {e}")
    CORE_UTILS_AVAILABLE = False
    
    # Provide dummy functions
    def setup_logging():
        return logging.getLogger(__name__)
    
    def log_action(action, data=None):
        logging.info(f"Action: {action}, Data: {data}")
    
    def search_analytics():
        return {'total_searches': 0, 'total_uploads': 0}

try:
    from .document_processor import process_uploaded_files, get_processing_stats, check_dependencies
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import document_processor: {e}")
    DOCUMENT_PROCESSOR_AVAILABLE = False
    
    # Provide dummy functions
    def process_uploaded_files(files):
        return [{'filename': f.name, 'error': 'Document processor not available'} for f in files]
    
    def get_processing_stats():
        return {'documents_processed': 0}
    
    def check_dependencies():
        return {'pdfplumber': False, 'PyPDF2': False, 'python-docx': False}

# UI components - optional import
try:
    from .ui.search_components import render_search_interface
    UI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import UI components: {e}")
    UI_AVAILABLE = False
    
    def render_search_interface(documents):
        import streamlit as st
        st.error("Search interface components not available")
        st.info("Basic search functionality is still available through the simple interface")

# Package exports
__all__ = [
    'setup_logging',
    'log_action', 
    'search_analytics',
    'process_uploaded_files',
    'get_processing_stats',
    'check_dependencies',
    'render_search_interface',
    'CORE_UTILS_AVAILABLE',
    'DOCUMENT_PROCESSOR_AVAILABLE',
    'UI_AVAILABLE'
]

# Log package initialization
logger = logging.getLogger(__name__)
logger.info(f"Initialized document search package v{__version__}")
logger.info(f"Core utils: {'✓' if CORE_UTILS_AVAILABLE else '✗'}")
logger.info(f"Document processor: {'✓' if DOCUMENT_PROCESSOR_AVAILABLE else '✗'}")
logger.info(f"UI components: {'✓' if UI_AVAILABLE else '✗'}")
