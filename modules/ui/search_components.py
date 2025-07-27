# modules/ui/search_components.py - Main Search Components (Modularized)
"""
Enhanced search system with modular architecture.

This module provides the main interfaces for document search and 
recommendation-response alignment, importing functionality from specialized modules.
"""

# Import main interfaces
from .search_interface import render_search_interface
from .recommendation_alignment import render_recommendation_alignment_interface

# Import utility functions that may be needed by other modules
from .search_utils import (
    STOP_WORDS,
    filter_stop_words,
    preprocess_query,
    estimate_page_number,
    check_rag_availability,
    remove_overlapping_matches,
    highlight_search_terms
)

from .search_methods import (
    execute_search,
    exact_search,
    smart_search_filtered,
    fuzzy_search_filtered,
    semantic_search,
    hybrid_search_filtered
)

from .result_display import (
    display_results_grouped,
    display_single_result,
    copy_all_results,
    export_results_csv,
    generate_search_report
)

# Version info
__version__ = "2.0.0"
__author__ = "DaphneAI Team"

# Module documentation
__doc__ = """
DaphneAI Search Components - Modular Architecture

This package provides enhanced document search capabilities with the following features:

🔍 **Search Methods:**
- Smart Search: Enhanced keyword matching with word variations
- Exact Match: Precise phrase matching
- Fuzzy Search: Handles typos and misspellings
- AI Semantic: AI-powered concept matching
- Hybrid: Combines multiple methods

🏛️ **Government Features:**
- Recommendation-Response Alignment
- AI-powered summary generation
- Export capabilities for analysis

📁 **Module Structure:**
- search_interface.py: Main search UI
- search_methods.py: Search algorithm implementations
- search_utils.py: Utility functions and constants
- result_display.py: Results formatting and display
- recommendation_alignment.py: Government analysis features

🚀 **Usage:**
```python
from modules.ui.search_components import (
    render_search_interface,
    render_recommendation_alignment_interface
)

# In your Streamlit app
render_search_interface(documents)
render_recommendation_alignment_interface(documents)
```

✅ **Key Improvements in v2.0:**
- Modular architecture for better maintainability
- Enhanced word variation matching
- Fixed stop word filtering
- Improved highlighting system
- Better error handling and fallbacks
- Comprehensive semantic search with fallbacks
"""

# Export all main functions for backward compatibility
__all__ = [
    # Main interfaces
    'render_search_interface',
    'render_recommendation_alignment_interface',
    
    # Search utilities
    'STOP_WORDS',
    'filter_stop_words',
    'preprocess_query',
    'estimate_page_number',
    'check_rag_availability',
    'remove_overlapping_matches',
    'highlight_search_terms',
    
    # Search methods
    'execute_search',
    'exact_search',
    'smart_search_filtered',
    'fuzzy_search_filtered',
    'semantic_search',
    'hybrid_search_filtered',
    
    # Result display
    'display_results_grouped',
    'display_single_result',
    'copy_all_results',
    'export_results_csv',
    'generate_search_report'
]

# Configuration
SEARCH_CONFIG = {
    'default_max_results': 5,
    'semantic_similarity_threshold': 0.3,
    'fuzzy_similarity_threshold': 0.6,
    'page_character_estimate': 2000,
    'context_window_size': 150,
    'alignment_confidence_threshold': 0.4
}

def get_search_info():
    """Get information about the search system"""
    return {
        'version': __version__,
        'modules': len(__all__),
        'features': [
            'Enhanced word variation matching',
            'Stop word filtering',
            'AI semantic search with fallbacks',
            'Recommendation-response alignment',
            'Multiple export formats',
            'Modular architecture'
        ],
        'config': SEARCH_CONFIG
    }

def test_module_imports():
    """Test that all modules import correctly"""
    try:
        from .search_interface import render_search_interface
        from .recommendation_alignment import render_recommendation_alignment_interface
        from .search_methods import execute_search
        from .search_utils import check_rag_availability
        from .result_display import display_results_grouped
        
        return {
            'status': 'success',
            'message': 'All modules imported successfully',
            'available_functions': len(__all__)
        }
    except ImportError as e:
        return {
            'status': 'error',
            'message': f'Import error: {str(e)}',
            'available_functions': 0
        }

# Backward compatibility aliases
render_search_components = render_search_interface  # Old function name
render_recommendation_components = render_recommendation_alignment_interface  # Old function name
