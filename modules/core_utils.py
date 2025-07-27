# modules/core_utils.py
# Core utilities for the document search application

import logging
import sys
from datetime import datetime
from typing import Dict, Any, List
import streamlit as st

def setup_logging(level=logging.INFO):
    """Setup application logging"""
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

def log_action(action: str, data: Dict[str, Any] = None):
    """Log user actions for analytics"""
    logger = logging.getLogger(__name__)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'data': data or {}
    }
    
    logger.info(f"Action: {action} | Data: {data}")
    
    # Store in session state for analytics
    if 'action_log' not in st.session_state:
        st.session_state.action_log = []
    
    st.session_state.action_log.append(log_entry)
    
    # Keep only last 100 actions
    if len(st.session_state.action_log) > 100:
        st.session_state.action_log = st.session_state.action_log[-100:]

def search_analytics() -> Dict[str, Any]:
    """Get search analytics from session state"""
    if 'action_log' not in st.session_state:
        return {
            'total_searches': 0,
            'total_uploads': 0,
            'avg_search_time': 0,
            'popular_search_terms': []
        }
    
    actions = st.session_state.action_log
    
    # Count different action types
    searches = [a for a in actions if a['action'] == 'search_performed']
    uploads = [a for a in actions if a['action'] == 'document_uploaded']
    
    # Calculate average search time
    search_times = [a['data'].get('search_time', 0) for a in searches if 'search_time' in a.get('data', {})]
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    
    # Get popular search terms
    search_terms = [a['data'].get('query', '') for a in searches if a.get('data', {}).get('query')]
    term_counts = {}
    for term in search_terms:
        term_counts[term] = term_counts.get(term, 0) + 1
    
    popular_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'total_searches': len(searches),
        'total_uploads': len(uploads),
        'avg_search_time': round(avg_search_time, 3),
        'popular_search_terms': popular_terms,
        'total_actions': len(actions)
    }

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def format_processing_time(seconds: float) -> str:
    """Format processing time in human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove null characters
    text = text.replace('\x00', '')
    
    return text.strip()

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length with ellipsis"""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    import platform
    import sys
    
    try:
        import streamlit as st
        st_version = st.__version__
    except:
        st_version = "unknown"
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'streamlit_version': st_version,
        'timestamp': datetime.now().isoformat()
    }
