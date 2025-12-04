"""Legacy shim for display functions split across smaller modules."""

from __future__ import annotations

# Core shared helpers
from .display_shared import (  # noqa: F401
    STOP_WORDS,
    clean_html_artifacts,
    create_highlighting_legend,
    debug_highlighting_issue,
    format_as_beautiful_paragraphs,
    format_text_as_clean_paragraphs,
    get_meaningful_words,
    highlight_government_terms,
    highlight_with_bold_method,
    highlight_with_brackets_method,
    highlight_with_capitalization_method,
    highlight_with_color_codes,
    highlight_with_context_awareness,
    highlight_with_html_method,
    render_highlighted_content_with_legend,
    smart_highlight_text,
    test_highlighting_methods,
)

# Search display functions
from .display_search import (  # noqa: F401
    copy_results_beautiful,
    display_manual_search_results_beautiful,
    display_search_results_beautiful,
    display_single_search_result_beautiful,
    export_results_csv_beautiful,
)

# Alignment display (already refactored)
from .alignment_display import (  # noqa: F401
    display_alignment_results_beautiful,
    display_single_alignment_beautiful,
    show_alignment_feature_info_beautiful,
)

__all__ = [
    # Search
    "display_search_results_beautiful",
    "display_single_search_result_beautiful",
    "display_manual_search_results_beautiful",
    "copy_results_beautiful",
    "export_results_csv_beautiful",
    # Alignment
    "display_alignment_results_beautiful",
    "display_single_alignment_beautiful",
    "show_alignment_feature_info_beautiful",
    # Shared/highlighting
    "STOP_WORDS",
    "get_meaningful_words",
    "clean_html_artifacts",
    "format_text_as_clean_paragraphs",
    "format_as_beautiful_paragraphs",
    "smart_highlight_text",
    "highlight_government_terms",
    "highlight_with_html_method",
    "highlight_with_bold_method",
    "highlight_with_capitalization_method",
    "highlight_with_brackets_method",
    "highlight_with_color_codes",
    "highlight_with_context_awareness",
    "render_highlighted_content_with_legend",
    "create_highlighting_legend",
    "test_highlighting_methods",
    "debug_highlighting_issue",
]
