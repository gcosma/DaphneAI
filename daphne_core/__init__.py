"""
Core logic package for DaphneAI.

This package is a thin wrapper around the legacy ``modules`` package.
It provides a clearer, domain-specific namespace (``daphne_core``)
without changing any behaviour.
"""

from __future__ import annotations

import modules as _legacy_modules  # type: ignore

# Re-export the public API from the existing ``modules`` package.
# This respects whatever ``__all__`` is defined there.
from modules import *  # type: ignore  # noqa: F401,F403

# Preserve the original export list where possible.
__all__ = getattr(_legacy_modules, "__all__", [])  # type: ignore[attr-defined]

# Mirror common metadata if present.
if hasattr(_legacy_modules, "__version__"):
    __version__ = _legacy_modules.__version__  # type: ignore[attr-defined]
if hasattr(_legacy_modules, "__description__"):
    __description__ = _legacy_modules.__description__  # type: ignore[attr-defined]
if hasattr(_legacy_modules, "__author__"):
    __author__ = _legacy_modules.__author__  # type: ignore[attr-defined]

