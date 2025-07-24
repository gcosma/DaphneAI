# ===============================================
# FILE: modules/ui/shared_components.py
# ===============================================
import streamlit as st
import logging
from typing import Tuple, List, Dict, Any
from datetime import datetime
import json

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

def add_error_message(message: str, error_type: str = "error"):
    """Add an error message to the session state"""
    if 'error_messages' not in st.session_state:
        st.session_state.error_messages = []
    
    st.session_state.error_messages.append({
        'message': message,
        'type': error_type,
        'timestamp': datetime.now().isoformat()
    })

def clear_error_messages():
    """Clear all error messages"""
    st.session_state.error_messages = []

def display_error_messages():
    """Display any pending error messages"""
    if 'error_messages' in st.session_state and st.session_state.error_messages:
        for error in st.session_state.error_messages[-3:]:  # Show last 3 errors
            if error['type'] == 'error':
                st.error(error['message'])
            elif error['type'] == 'warning':
                st.warning(error['message'])
            elif error['type'] == 'info':
                st.info(error['message'])

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
            model_temp = st.slider("Temperature", 0.0, 1.0, 0.3)
            st.session_state.model_temperature = model_temp
        else:
            st.session_state.use_advanced_ai = False
        
        # Export settings
        st.subheader("Export Options")
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"])
        st.session_state.export_format = export_format
        
        # Clear data button
        st.subheader("Data Management")
        if st.button("🗑️ Clear All Data", type="secondary"):
            clear_session_data()
            st.rerun()

def clear_session_data():
    """Clear all session data except settings"""
    keys_to_keep = [
        'initialized', 'batch_size', 'use_advanced_ai', 
        'model_temperature', 'export_format'
    ]
    
    keys_to_clear = [key for key in st.session_state.keys() if key not in keys_to_keep]
    
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Re-initialize
    initialize_session_state()
    add_error_message("All data cleared successfully", "info")

def render_document_selector(documents: List[Dict], key_prefix: str = "doc_select") -> List[Dict]:
    """Render a multi-select widget for documents"""
    if not documents:
        st.info("No documents available. Please upload documents first.")
        return []
    
    st.subheader("📄 Select Documents")
    
    # Select all/none controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All", key=f"{key_prefix}_all"):
            st.session_state[f"{key_prefix}_selected"] = list(range(len(documents)))
    
    with col2:
        if st.button("Select None", key=f"{key_prefix}_none"):
            st.session_state[f"{key_prefix}_selected"] = []
    
    # Initialize selection state
    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = []
    
    # Document checkboxes
    selected_docs = []
    for i, doc in enumerate(documents):
        is_selected = i in st.session_state[f"{key_prefix}_selected"]
        
        if st.checkbox(
            f"📄 {doc.get('name', f'Document {i+1}')} ({len(doc.get('content', ''))} chars)",
            value=is_selected,
            key=f"{key_prefix}_{i}"
        ):
            if i not in st.session_state[f"{key_prefix}_selected"]:
                st.session_state[f"{key_prefix}_selected"].append(i)
            selected_docs.append(doc)
        else:
            if i in st.session_state[f"{key_prefix}_selected"]:
                st.session_state[f"{key_prefix}_selected"].remove(i)
    
    return selected_docs

def render_progress_indicator(current: int, total: int, description: str = "Processing"):
    """Render a progress indicator"""
    if total > 0:
        progress = current / total
        st.progress(progress, text=f"{description}: {current}/{total}")
    else:
        st.info(f"{description}...")

def format_extraction_results(results: Dict[str, Any]) -> str:
    """Format extraction results for display"""
    if not results:
        return "No results available"
    
    formatted = []
    
    # Basic stats
    if 'recommendations' in results:
        formatted.append(f"**Found {len(results['recommendations'])} recommendations**")
    
    if 'stats' in results:
        stats = results['stats']
        formatted.append(f"- AI Method: {stats.get('ai_count', 0)} results")
        formatted.append(f"- Pattern Method: {stats.get('pattern_count', 0)} results")
        formatted.append(f"- Quality Score: {stats.get('avg_quality', 0):.1f}/100")
    
    # Sample recommendations
    if 'recommendations' in results and results['recommendations']:
        formatted.append("\n**Sample Recommendations:**")
        for i, rec in enumerate(results['recommendations'][:3]):  # Show first 3
            formatted.append(f"{i+1}. [{rec.get('id', 'N/A')}] {rec.get('text', '')[:100]}...")
    
    return "\n".join(formatted)

def export_data(data: Any, filename: str, format_type: str = "JSON"):
    """Export data in the specified format"""
    try:
        if format_type == "JSON":
            json_str = json.dumps(data, indent=2, default=str)
            st.download_button(
                label=f"📥 Download {filename}.json",
                data=json_str,
                file_name=f"{filename}.json",
                mime="application/json"
            )
        
        elif format_type == "CSV":
            import pandas as pd
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                csv_str = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {filename}.csv",
                    data=csv_str,
                    file_name=f"{filename}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("CSV export requires list data format")
        
        elif format_type == "Excel":
            import pandas as pd
            import io
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)
                
                st.download_button(
                    label=f"📥 Download {filename}.xlsx",
                    data=buffer.getvalue(),
                    file_name=f"{filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Excel export requires list data format")
    
    except Exception as e:
        add_error_message(f"Export failed: {str(e)}", "error")

def render_data_summary():
    """Render a summary of current data state"""
    st.subheader("📊 Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Documents Uploaded", len(st.session_state.get('uploaded_documents', [])))
        st.metric("Recommendations Extracted", len(st.session_state.get('extracted_recommendations', [])))
    
    with col2:
        st.metric("Annotations Created", len(st.session_state.get('annotation_results', {})))
        st.metric("Matches Found", len(st.session_state.get('matching_results', {})))
    
    # Processing status
    status = st.session_state.get('processing_status', 'idle')
    if status == 'idle':
        st.success("✅ System Ready")
    elif status == 'processing':
        st.warning("⏳ Processing...")
    elif status == 'error':
        st.error("❌ Error State")

def validate_api_keys():
    """Validate required API keys"""
    import os
    
    missing_keys = []
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key or openai_key.strip() == "":
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        st.warning(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
        st.info("Some features may not work without proper API configuration.")
        return False
    
    return True

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
    import os
    
    if os.getenv('DEBUG', 'False').lower() == 'true':
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
