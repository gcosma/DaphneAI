"""
Thin wrapper around ``modules.recommendation_extractor``.

All implementation lives in the legacy module; this file only provides
the new ``daphne_core`` namespace for imports.
"""

from modules.recommendation_extractor import *  # type: ignore  # noqa: F401,F403

