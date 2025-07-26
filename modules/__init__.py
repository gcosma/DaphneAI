# ===============================================
# COMPLETE modules/__init__.py
# Main modules package initialization
# ===============================================

"""
Recommendation-Response Tracker - Core Modules Package

This package contains the core modules for the Recommendation-Response Tracker application.
"""

__version__ = "2.0.0"
__description__ = "AI-powered document analysis for UK Government inquiry reports"

import logging
from datetime import datetime

# Setup logging for the modules package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package-level imports for convenience
try:
    from .document_processor import DocumentProcessor
    DOCUMENT_PROCESSOR_AVAILABLE = True
    logging.info("✅ DocumentProcessor available")
except ImportError:
    DOCUMENT_PROCESSOR_AVAILABLE = False
    logging.warning("⚠️ DocumentProcessor not available")

try:
    from .enhanced_section_extractor import EnhancedSectionExtractor
    ENHANCED_SECTION_EXTRACTOR_AVAILABLE = True
    logging.info("✅ EnhancedSectionExtractor available")
except ImportError:
    ENHANCED_SECTION_EXTRACTOR_AVAILABLE = False
    logging.warning("⚠️ EnhancedSectionExtractor not available")

try:
    from .llm_extractor import LLMRecommendationExtractor
    LLM_EXTRACTOR_AVAILABLE = True
    logging.info("✅ LLMRecommendationExtractor available")
except ImportError:
    LLM_EXTRACTOR_AVAILABLE = False
    logging.warning("⚠️ LLMRecommendationExtractor not available")

try:
    from .core_utils import SecurityValidator, log_user_action
    CORE_UTILS_AVAILABLE = True
    logging.info("✅ Core utilities available")
except ImportError:
    CORE_UTILS_AVAILABLE = False
    logging.warning("⚠️ Core utilities not available")

# Package availability status
PACKAGE_STATUS = {
    'document_processor': DOCUMENT_PROCESSOR_AVAILABLE,
    'enhanced_extractor': ENHANCED_SECTION_EXTRACTOR_AVAILABLE,
    'llm_extractor': LLM_EXTRACTOR_AVAILABLE,
    'core_utils': CORE_UTILS_AVAILABLE,
    'version': __version__,
    'initialized_at': datetime.now().isoformat()
}

# Package-level exports
__all__ = [
    '__version__',
    '__description__',
    'PACKAGE_STATUS',
    'DOCUMENT_PROCESSOR_AVAILABLE',
    'ENHANCED_SECTION_EXTRACTOR_AVAILABLE',
    'LLM_EXTRACTOR_AVAILABLE',
    'CORE_UTILS_AVAILABLE'
]

# Conditional exports based on availability
if DOCUMENT_PROCESSOR_AVAILABLE:
    __all__.append('DocumentProcessor')

if ENHANCED_SECTION_EXTRACTOR_AVAILABLE:
    __all__.append('EnhancedSectionExtractor')

if LLM_EXTRACTOR_AVAILABLE:
    __all__.append('LLMRecommendationExtractor')

if CORE_UTILS_AVAILABLE:
    __all__.extend(['SecurityValidator', 'log_user_action'])

# Package initialization message
logging.info(f"🚀 Modules package initialized - Version {__version__}")
logging.info(f"📊 Available components: {sum(PACKAGE_STATUS[k] for k in PACKAGE_STATUS if isinstance(PACKAGE_STATUS[k], bool))}/4")
