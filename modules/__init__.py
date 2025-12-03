# modules/__init__.py
# Shim that re-exports the canonical daphne_core package
import logging

import daphne_core as _core

# Package metadata
__version__ = _core.__version__
__description__ = _core.__description__
__author__ = _core.__author__

# Re-export core symbols
from daphne_core import *  # type: ignore  # noqa: F401,F403

# Mirror the export list
__all__ = _core.__all__

# =============================================================================
# PACKAGE STATUS
# =============================================================================
logger = logging.getLogger(__name__)
logger.info(f"Initialized document search package v{__version__}")
logger.info(f"Core utils: {'✓' if getattr(_core, 'CORE_UTILS_AVAILABLE', False) else '✗'}")
logger.info(
    f"Document processor: {'✓' if getattr(_core, 'DOCUMENT_PROCESSOR_AVAILABLE', False) else '✗'}"
)
logger.info(f"UI components: {'✓' if getattr(_core, 'UI_AVAILABLE', False) else '✗'}")
logger.info(
    f"Integration helper: {'✓' if getattr(_core, 'INTEGRATION_HELPER_AVAILABLE', False) else '✗'}"
)
logger.info(
    f"Recommendation extractor: {'✓' if getattr(_core, 'RECOMMENDATION_EXTRACTOR_AVAILABLE', False) else '✗'}"
)


def get_package_status():
    """Get the status of all package components"""
    return _core.get_package_status()
