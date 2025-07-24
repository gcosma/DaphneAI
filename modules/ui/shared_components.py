# ===============================================
# FILE: modules/ui/shared_components.py (FIXED VERSION)
# ===============================================
import streamlit as st
import logging
from typing import Tuple, List, Dict, Any
from datetime import datetime
import json
import os

def initialize_session_state():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
        # Document management
        st.session_state.uploaded_documents = []
        
        # Extraction results - Updated for recommendations
        st.session_state.extracted_recommendations = []
        st.session_state.extracted_concerns = []  # Keep for backward compatibility
        st.session_state.extraction_results = {}
        
        # Analysis results
        st.session_state.annotation_results = {}
        st.session_state.matching_results = {}
        st.session_state.search_history = []
        st.session_state.search_results = {}
        
        # Component instances
        st.session_state.vector_store_manager = None
        st.session_state.rag_engine = None
        st.session_state.bert_annotator = None
        st.session_state.recommendation_matcher = None
        
        # Processing states
        st.session_state.processing_status = "idle"
        st.session_state.last_processing_time = None
        st.session_state.error_messages = []
        
        # UI states
        st.session_state.selected_frameworks = []
        st.session_state.current_tab = "upload"
        st.session_state.export_ready = False
        
        # New extraction-specific states
        st.session_state.selected_extraction_docs = []
        st.session_state.extraction_method_used = None
        st.session_state.last_extraction_timestamp = None
        
        # AI availability status
        st.session_state.ai_available = bool(os.getenv('OPENAI_API_KEY'))
        st.session_state.use_mock_ai = not st.session_state.ai_available

def render_header():
    """Render the main application header with status indicators"""
    st.title("📋 Recommendation-Response Tracker")
    st.markdown("""
    **AI-Powered Document Analysis System** for UK Government Inquiries and Reviews
    
    Extract recommendations, analyze responses, and track implementation across inquiry reports.
    """)
    
    # Show system status
    render_status_indicators()

def render_status_indicators():
    """Show current system status indicators"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        doc_count = len(st.session_state.get('uploaded_documents', []))
        st.metric("📁 Documents", doc_count)
    
    with col2:
        rec_count = len(st.session_state.get('extracted_recommendations', []))
        st.metric("💡 Recommendations", rec_count)
    
    with col3:
        annotation_count = len(st.session_state.get('annotation_results', {}))
        st.metric("🏷️ Annotations", annotation_count)
    
    with col4:
        match_count = len(st.session_state.get('matching_results', {}))
        st.metric("🔗 Matches", match_count)

def render_navigation_tabs():
    """Render the main navigation tabs"""
    tabs = st.tabs([
        "📁 Upload", 
        "🔍 Extract", 
        "🏷️ Annotate", 
        "🔗 Match", 
        "🔎 Search", 
        "📊 Dashboard"
    ])
    
    return tabs

# ===============================================
# MISSING FUNCTIONS - NOW ADDED
# ===============================================

def render_sidebar_info():
    """Render sidebar information and controls"""
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # API Key status - OPTIONAL
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            st.success("✅ OpenAI API Key: Configured")
            st.info("🤖 AI-powered features available")
        else:
            st.info("ℹ️ OpenAI API Key: Not Set")
            st.caption("App works without API key. AI features will use mock data.")
            with st.expander("🔑 Want to add OpenAI API Key?"):
                st.markdown("""
                **To enable AI features:**
                1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
                2. Set environment variable: `OPENAI_API_KEY=your_key`
                3. Or upload a `.env` file with your key
                
                **Current functionality without API key:**
                - ✅ Document upload and processing
                - ✅ Pattern-based extraction  
                - ✅ Mock AI responses for testing
                - ❌ Real AI-powered extraction
                - ❌ BERT annotation
                """)
        
        # Processing status
        status = st.session_state.get('processing_status', 'idle')
        if status == 'idle':
            st.info("🟡 Status: Ready")
        elif status == 'processing':
            st.warning("🟠 Status: Processing...")
        elif status == 'error':
            st.error("🔴 Status: Error")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        st.write(f"Documents: {len(st.session_state.get('uploaded_documents', []))}")
        st.write(f"Recommendations: {len(st.session_state.get('extracted_recommendations', []))}")
        st.write(f"Annotations: {len(st.session_state.get('annotation_results', {}))}")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        # Processing settings
        batch_size = st.slider("Batch Size", 1, 20, 10)
        st.session_state.batch_size = batch_size
        
        # Model settings
        if st.checkbox("Use Advanced AI", value=True):
            st.session_state.use_advanced_ai = True
            model_temp = st.slider("Temperature", 0.0, 1.0, 0.1)
            st.session_state.model_temperature = model_temp
        else:
            st.session_state.use_advanced_ai = False
        
        # Debug mode
        if st.checkbox("Debug Mode"):
            st.session_state.debug_mode = True
            display_debug_info()
        else:
            st.session_state.debug_mode = False

def show_error_messages():
    """Show any pending error messages"""
    if 'error_messages' in st.session_state and st.session_state.error_messages:
        for error in st.session_state.error_messages[-3:]:  # Show last 3 errors
            if error['type'] == 'error':
                st.error(error['message'])
            elif error['type'] == 'warning':
                st.warning(error['message'])
            elif error['type'] == 'info':
                st.info(error['message'])
            elif error['type'] == 'success':
                st.success(error['message'])

def clear_error_messages():
    """Clear all error messages"""
    if 'error_messages' in st.session_state:
        st.session_state.error_messages = []

def add_error_message(message: str, error_type: str = "error"):
    """Add an error message to the session state"""
    if 'error_messages' not in st.session_state:
        st.session_state.error_messages = []
    
    st.session_state.error_messages.append({
        'message': message,
        'type': error_type,
        'timestamp': datetime.now().isoformat()
    })

def display_error_messages():
    """Display any pending error messages (alias for show_error_messages)"""
    show_error_messages()

def render_sidebar_controls():
    """Render sidebar controls for global settings"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Processing settings
        st.subheader("Processing Options")
        batch_size = st.slider("Batch Size", 1, 20, 10)
        st.session_state.batch_size = batch_size
        
        # Model settings
        st.subheader("AI Model Settings")
        if st.checkbox("Use Advanced AI", value=True):
            st.session_state.use_advanced_ai = True
            model_temp = st.slider("Temperature", 0.0, 1.0, 0.1)
            st.session_state.model_temperature = model_temp
        else:
            st.session_state.use_advanced_ai = False

def validate_api_keys():
    """Validate required API keys - OPTIONAL for this app"""
    missing_keys = []
    
    # Check OpenAI API key - but it's OPTIONAL
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key or openai_key.strip() == "":
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        st.info(f"ℹ️ Optional API keys not set: {', '.join(missing_keys)}")
        st.success("✅ App will work with limited functionality (pattern-based extraction only)")
        return False  # False means "no API key" but app still works
    
    st.success("✅ All API keys configured - Full AI features available")
    return True

def check_ai_availability():
    """Check if AI features are available"""
    api_key = os.getenv('OPENAI_API_KEY')
    return bool(api_key and api_key.strip())

def show_ai_status_message():
    """Show current AI availability status"""
    if check_ai_availability():
        st.success("🤖 AI features are available")
    else:
        st.info("🔧 Running in pattern-matching mode (no OpenAI API key)")
        st.caption("Upload documents and try pattern-based extraction!")

def log_user_action(action: str, details: str = ""):
    """Log user actions for debugging"""
    timestamp = datetime.now().isoformat()
    logging.info(f"[{timestamp}] User Action: {action} - {details}")
    
    # Also store in session state for debugging
    if 'user_actions' not in st.session_state:
        st.session_state.user_actions = []
    
    st.session_state.user_actions.append({
        'timestamp': timestamp,
        'action': action,
        'details': details
    })
    
    # Keep only last 50 actions
    if len(st.session_state.user_actions) > 50:
        st.session_state.user_actions = st.session_state.user_actions[-50:]

def display_debug_info():
    """Display debug information (only in development)"""
    if st.session_state.get('debug_mode', False):
        with st.expander("🐛 Debug Information"):
            st.json({
                'session_keys': list(st.session_state.keys()),
                'document_count': len(st.session_state.get('uploaded_documents', [])),
                'recommendation_count': len(st.session_state.get('extracted_recommendations', [])),
                'processing_status': st.session_state.get('processing_status', 'unknown'),
                'recent_actions': st.session_state.get('user_actions', [])[-5:]
            })

# Utility functions for specific data types
def safe_get_text_length(text: str) -> int:
    """Safely get text length"""
    return len(text) if text else 0

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if not text:
        return ""
    return text[:max_length] + "..." if len(text) > max_length else text

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

def show_progress_indicator(current: int = None, total: int = None, message: str = "Processing..."):
    """Show a progress indicator - supports both simple spinner and progress bar modes"""
    
    # If called with 3 arguments (current, total, message) - show progress bar
    if current is not None and total is not None:
        if total > 0:
            progress = current / total
            st.progress(progress, text=f"{message}: {current}/{total}")
        else:
            st.info(f"{message}...")
        return None  # No context manager for progress bar mode
    
    # If called with just message or no arguments - show spinner
    elif isinstance(current, str):
        # Handle case where first arg is actually the message
        return st.spinner(current)
    else:
        # Default spinner mode
        return st.spinner(message)

def render_progress_indicator(current: int, total: int, description: str = "Processing"):
    """Render a progress indicator (alternative function name for compatibility)"""
    if total > 0:
        progress = current / total
        st.progress(progress, text=f"{description}: {current}/{total}")
    else:
        st.info(f"{description}...")

def generate_mock_recommendations(doc_name: str, num_recommendations: int = 3):
    """Generate mock recommendations when no API key is available"""
    mock_recommendations = [
        {
            'id': f"REC-{i+1}",
            'text': f"Sample recommendation {i+1} extracted from {doc_name}. This is a demonstration of pattern-based extraction working without an API key.",
            'type': 'recommendation',
            'source_document': doc_name,
            'page_number': i + 1,
            'confidence': 0.85,
            'extraction_method': 'pattern_based'
        }
        for i in range(num_recommendations)
    ]
    return mock_recommendations

def get_mock_annotation_results(recommendations: list):
    """Generate mock annotation results"""
    mock_themes = ['Communication', 'Process Improvement', 'Training', 'Technology', 'Policy']
    
    results = {}
    for i, rec in enumerate(recommendations):
        theme = mock_themes[i % len(mock_themes)]
        results[rec.get('id', f'rec_{i}')] = {
            'primary_theme': theme,
            'confidence': 0.78,
            'all_themes': {theme: 0.78},
            'method': 'mock_bert'
        }
    
    return results
