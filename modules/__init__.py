# modules/__init__.py
# Main package initialization

import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__version__ = "1.0.0"
__description__ = "Advanced RAG + Smart Search System"
__author__ = "Your Name"

# Package exports
from .core_utils import setup_logging, log_action, search_analytics
from .document_processor import process_uploaded_files, get_processing_stats
from .ui import render_search_interface

__all__ = [
    'setup_logging',
    'log_action', 
    'search_analytics',
    'process_uploaded_files',
    'get_processing_stats',
    'render_search_interface'
]
