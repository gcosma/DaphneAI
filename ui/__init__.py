"""
Streamlit UI package for DaphneAI.

This is a thin wrapper around the legacy ``modules.ui`` package so that
the application can import ``ui.*`` without changing any behaviour.
"""

from modules.ui import *  # type: ignore  # noqa: F401,F403

