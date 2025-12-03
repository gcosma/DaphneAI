"""
Lightweight core utilities for logging and analytics.

This preserves the minimal behaviour that previously lived as fallbacks in
``modules.__init__`` so imports succeed without altering downstream logic.
"""

import logging
from typing import Any, Dict


def setup_logging() -> logging.Logger:
    return logging.getLogger(__name__)


def log_action(action: str, data: Any = None) -> None:
    logging.info("Action: %s, Data: %s", action, data)


def search_analytics() -> Dict[str, int]:
    return {"total_searches": 0, "total_uploads": 0}
