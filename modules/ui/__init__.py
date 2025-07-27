# modules/ui/__init__.py
"""
UI components package for DaphneAI
Provides search interface and display components
"""

__version__ = "1.0.0"

# Try to import main components
try:
    from .search_components import render_search_interface, get_search_stats
    UI_COMPONENTS_AVAILABLE = True
except ImportError as e:
    UI_COMPONENTS_AVAILABLE = False
    
    # Provide fallback function
    def render_search_interface(documents):
        import streamlit as st
        st.error("Search interface not available")
        st.info("Please check if all dependencies are installed")
    
    def get_search_stats():
        return {'total_searches': 0}

__all__ = [
    'render_search_interface',
    'get_search_stats',
    'UI_COMPONENTS_AVAILABLE'
]

# ==========================================
# modules/search/__init__.py
"""
Search engines package for DaphneAI
Provides various search algorithms including RAG
"""

__version__ = "1.0.0"

# Try to import search engines
try:
    from .rag_search import RAGSearchEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    
    # Provide dummy class
    class RAGSearchEngine:
        def __init__(self, *args, **kwargs):
            self.available = False
        
        def search(self, *args, **kwargs):
            return []

__all__ = [
    'RAGSearchEngine',
    'RAG_AVAILABLE'
]
