# ===============================================
# FILE: modules/ui/shared_components.py
# ===============================================

import streamlit as st
import logging
from typing import Tuple

def initialize_session_state():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.uploaded_documents = []
        st.session_state.extracted_recommendations = []
        st.session_state.extracted_concerns = []
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

def render_header():
    """Render the main application header with status indicators"""
    st.title("📋 Recommendation-Response Tracker")
    st.markdown("""
    **AI-Powered Document Analysis System**
    
    Upload documents → Extract recommendations → Annotate with concepts → Find responses → Analyze patterns
    """)
    
    # Create status indicator columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        docs_count = len(st.session_state.get('uploaded_documents', []))
        if docs_count > 0:
            st.metric("📁 Documents", docs_count, delta=None)
        else:
            st.metric("📁 Documents", "0", help="Upload PDF documents to start")
    
    with col2:
        recs_count = len(st.session_state.get('extracted_recommendations', []))
        if recs_count > 0:
            st.metric("🔍 Recommendations", recs_count)
        else:
            st.metric("🔍 Recommendations", "0", help="Extract recommendations from documents")
    
    with col3:
        concerns_count = len(st.session_state.get('extracted_concerns', []))
        if concerns_count > 0:
            st.metric("⚠️ Concerns", concerns_count)
        else:
            st.metric("⚠️ Concerns", "0", help="Concerns identified in documents")
    
    with col4:
        annotations_count = len(st.session_state.get('annotation_results', {}))
        if annotations_count > 0:
            st.metric("🏷️ Annotated", annotations_count)
        else:
            st.metric("🏷️ Annotated", "0", help="Apply concept annotations")
    
    with col5:
        matches_count = len(st.session_state.get('matching_results', {}))
        if matches_count > 0:
            st.metric("🔗 Matched", matches_count)
        else:
            st.metric("🔗 Matched", "0", help="Find responses to recommendations")
    
    # Show processing status if active
    if st.session_state.get('processing_status') != "idle":
        status = st.session_state.processing_status
        if status == "processing":
            st.info("🔄 Processing documents...")
        elif status == "extracting":
            st.info("🔍 Extracting recommendations and concerns...")
        elif status == "annotating":
            st.info("🏷️ Annotating with concepts...")
        elif status == "matching":
            st.info("🔗 Finding responses...")
        elif status == "indexing":
            st.info("📚 Indexing documents for search...")
    
    # Show any error messages
    if st.session_state.get('error_messages'):
        for error in st.session_state.error_messages[-3:]:  # Show last 3 errors
            st.error(f"❌ {error}")

def render_navigation_tabs() -> Tuple:
    """Render navigation tabs and return tab objects"""
    try:
        tabs = st.tabs([
            "📁 Upload Documents",
            "🔍 Extract Content", 
            "🏷️ Concept Annotation",
            "🔗 Find Responses",
            "🔎 Smart Search",
            "📊 Dashboard"
        ])
        
        # Store current tab in session state for persistence
        if len(tabs) == 6:
            return tabs
        else:
            # Fallback if tabs creation fails
            st.error("Error creating navigation tabs. Using simplified interface.")
            return (st.container(), st.container(), st.container(), st.container(), st.container(), st.container())
            
    except Exception as e:
        logging.error(f"Error creating navigation tabs: {e}")
        st.error("Navigation error. Please refresh the page.")
        return (st.container(), st.container(), st.container(), st.container(), st.container(), st.container())

def show_system_status():
    """Show detailed system status in sidebar"""
    with st.sidebar:
        st.subheader("🖥️ System Status")
        
        # Check component status
        components_status = {
            "Vector Store": st.session_state.vector_store_manager is not None,
            "RAG Engine": st.session_state.rag_engine is not None,
            "BERT Annotator": st.session_state.bert_annotator is not None,
            "Recommendation Matcher": st.session_state.recommendation_matcher is not None
        }
        
        for component, status in components_status.items():
            if status:
                st.success(f"✅ {component}")
            else:
                st.warning(f"⚠️ {component} (not initialized)")
        
        # Show memory usage if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent:.1f}%")
        except ImportError:
            pass
        
        # Show last processing time
        if st.session_state.get('last_processing_time'):
            st.info(f"Last processed: {st.session_state.last_processing_time}")

def clear_error_messages():
    """Clear error messages from session state"""
    if st.button("🗑️ Clear Errors"):
        st.session_state.error_messages = []
        st.rerun()

def add_error_message(message: str):
    """Add error message to session state"""
    if 'error_messages' not in st.session_state:
        st.session_state.error_messages = []
    
    st.session_state.error_messages.append(message)
    
    # Keep only last 10 errors
    if len(st.session_state.error_messages) > 10:
        st.session_state.error_messages = st.session_state.error_messages[-10:]

def show_progress_indicator(current_step: int, total_steps: int, step_name: str):
    """Show progress indicator for multi-step processes"""
    progress = current_step / total_steps
    st.progress(progress, text=f"Step {current_step}/{total_steps}: {step_name}")

def render_help_section():
    """Render help and instructions"""
    with st.sidebar:
        with st.expander("ℹ️ Help & Instructions"):
            st.markdown("""
            ### Getting Started
            1. **Upload Documents**: Add PDF files containing recommendations and responses
            2. **Extract Content**: Use AI to identify recommendations and concerns
            3. **Annotate**: Apply concept frameworks to understand themes
            4. **Find Responses**: Match recommendations to their responses
            5. **Search**: Use the smart search to explore your documents
            6. **Analyze**: View insights and export results
            
            ### Tips
            - Upload both recommendation and response documents for best results
            - Use the concept annotation to understand patterns
            - The smart search uses AI to find relevant content
            - Export your results for further analysis
            
            ### Troubleshooting
            - If processing fails, try smaller batches of documents
            - Check that PDFs contain readable text (not just images)
            - Ensure you have a stable internet connection for AI features
            """)

def show_keyboard_shortcuts():
    """Show keyboard shortcuts in sidebar"""
    with st.sidebar:
        with st.expander("⌨️ Keyboard Shortcuts"):
            st.markdown("""
            - `Ctrl + R`: Refresh page
            - `Ctrl + Enter`: Run current action
            - `Tab`: Navigate between elements
            - `Escape`: Close current dialog
            """)

def render_footer():
    """Render footer with app information"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Recommendation-Response Tracker**")
        st.markdown("AI-powered document analysis system")
    
    with col2:
        st.markdown("**Technologies:**")
        st.markdown("Streamlit • OpenAI • BERT • RAG")
    
    with col3:
        st.markdown("**Version:** 1.0.0")
        st.markdown("Built with ❤️ for document analysis")
