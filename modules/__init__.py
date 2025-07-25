# ===============================================
# FILE: modules/__init__.py
# ===============================================

"""
Recommendation-Response Tracker - Core Modules Package

This package contains the core modules for the Recommendation-Response Tracker application.
"""

__version__ = "2.0.0"
__description__ = "AI-powered document analysis for UK Government inquiry reports"

import logging

# Setup logging for the modules package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package-level imports for convenience
try:
    from .document_processor import DocumentProcessor
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSOR_AVAILABLE = False
    logging.warning("DocumentProcessor not available")

try:
    from .enhanced_section_extractor import EnhancedSectionExtractor
    ENHANCED_SECTION_EXTRACTOR_AVAILABLE = True
except ImportError:
    ENHANCED_SECTION_EXTRACTOR_AVAILABLE = False
    logging.warning("EnhancedSectionExtractor not available")

# Package availability status
__all__ = [
    '__version__',
    '__description__',
    'DOCUMENT_PROCESSOR_AVAILABLE',
    'ENHANCED_SECTION_EXTRACTOR_AVAILABLE'
]

if DOCUMENT_PROCESSOR_AVAILABLE:
    __all__.append('DocumentProcessor')

if ENHANCED_SECTION_EXTRACTOR_AVAILABLE:
    __all__.append('EnhancedSectionExtractor')

logging.info(f"✅ Modules package initialized - Version {__version__}")
