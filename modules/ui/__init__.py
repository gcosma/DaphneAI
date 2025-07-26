# ===============================================
# FILE: modules/ui/__init__.py (COMPLETE FIXED VERSION)
# Main UI module initialization with all components
# ===============================================

import streamlit as st
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# IMPORT ALL UI COMPONENTS WITH FALLBACKS
# ===============================================

# Upload components
try:
    from .upload_components import render_upload_tab
    UPLOAD_COMPONENTS_AVAILABLE = True
    logger.info("✅ Upload components imported successfully")
except ImportError as e:
    UPLOAD_COMPONENTS_AVAILABLE = False
    logger.error(f"❌ Failed to import upload components: {e}")
    
    def render_upload_tab():
        st.header("📁 Document Upload")
        st.error("❌ Upload component not available")
        st.info("Please check modules/ui/upload_components.py")

# Extraction components
try:
    from .extraction_components import render_extraction_tab
    EXTRACTION_COMPONENTS_AVAILABLE = True
    logger.info("✅ Enhanced extraction components imported successfully")
except ImportError as e:
    EXTRACTION_COMPONENTS_AVAILABLE = False
    logger.error(f"❌ Failed to import extraction components: {e}")
    
    def render_extraction_tab():
        st.header("🔍 Content Extraction")
        st.error("❌ Extraction component not available")
        st.info("Please check modules/ui/extraction_components.py")

# Annotation components  
try:
    from .annotation_components import render_annotation_tab
    ANNOTATION_COMPONENTS_AVAILABLE = True
    logger.info("✅ Annotation components imported successfully")
except ImportError as e:
    ANNOTATION_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Annotation components not available: {e}")
    
    def render_annotation_tab():
        st.header("🏷️ Concept Annotation")
        st.info("🚧 Annotation component not yet available")

# Matching components
try:
    from .matching_components import render_matching_tab
    MATCHING_COMPONENTS_AVAILABLE = True
    logger.info("✅ Matching components imported successfully")
except ImportError as e:
    MATCHING_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Matching components not available: {e}")
    
    def render_matching_tab():
        st.header("🔗 Response Matching")
        st.info("🚧 Matching component not yet available")

# Search components
try:
    from .search_components import render_search_tab
    SEARCH_COMPONENTS_AVAILABLE = True
    logger.info("✅ Search components imported successfully")
except ImportError as e:
    SEARCH_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Search components not available: {e}")
    
    def render_search_tab():
        st.header("🔎 Smart Search")
        st.info("🚧 Search component not yet available")

# Dashboard components
try:
    from .dashboard_components import render_dashboard_tab
    DASHBOARD_COMPONENTS_AVAILABLE = True
    logger.info("✅ Dashboard components imported successfully")
except ImportError as e:
    DASHBOARD_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Dashboard components not available: {e}")
    
    def render_dashboard_tab():
        st.header("📊 Analytics Dashboard")
        st.info("🚧 Dashboard component not yet available")

# ===============================================
# CORE UI FUNCTIONS
# ===============================================

def render_header():
    """Render the application header with status indicators"""
    st.set_page_config(
        page_title="DaphneAI - Government Document Analyzer",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ DaphneAI")
    st.markdown("### AI-Powered Government Document Analysis Platform")
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        uploaded_count = len(st.session_state.get('uploaded_documents', []))
        st.metric("📁 Documents", uploaded_count)
    
    with col2:
        rec_count = len(st.session_state.get('extracted_recommendations', []))
        st.metric("📋 Recommendations", rec_count)
    
    with col3:
        resp_count = len(st.session_state.get('extracted_responses', []))
        st.metric("💬 Responses", resp_count)
    
    with col4:
        status = st.session_state.get('processing_status', 'idle')
        status_emoji = "✅" if status == "complete" else "⏳" if status == "processing" else "⭕"
        st.metric("⚡ Status", f"{status_emoji} {status.title()}")
    
    st.divider()

def render_navigation_tabs():
    """Render the main navigation tabs"""
    tabs = st.tabs([
        "📁 Upload Documents",
        "🔍 Extract Content", 
        "🏷️ Concept Annotation",
        "🔗 Find Responses",
        "🔎 Smart Search",
        "📊 Analytics Dashboard"
    ])
    
    # Store current tab index for state management
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    
    # Render each tab with proper error handling
    with tabs[0]:  # Upload Documents
        try:
            render_upload_tab()
        except Exception as e:
            st.error(f"❌ Error in Upload tab: {str(e)}")
            logger.error(f"Upload tab error: {e}")
    
    with tabs[1]:  # Extract Content
        try:
            render_extraction_tab()
        except Exception as e:
            st.error(f"❌ Error in Extraction tab: {str(e)}")
            logger.error(f"Extraction tab error: {e}")
    
    with tabs[2]:  # Concept Annotation
        try:
            render_annotation_tab()
        except Exception as e:
            st.error(f"❌ Error in Annotation tab: {str(e)}")
            logger.error(f"Annotation tab error: {e}")
    
    with tabs[3]:  # Find Responses
        try:
            render_matching_tab()
        except Exception as e:
            st.error(f"❌ Error in Matching tab: {str(e)}")
            logger.error(f"Matching tab error: {e}")
    
    with tabs[4]:  # Smart Search
        try:
            render_search_tab()
        except Exception as e:
            st.error(f"❌ Error in Search tab: {str(e)}")
            logger.error(f"Search tab error: {e}")
    
    with tabs[5]:  # Analytics Dashboard
        try:
            render_dashboard_tab()
        except Exception as e:
            st.error(f"❌ Error in Dashboard tab: {str(e)}")
            logger.error(f"Dashboard tab error: {e}")

def render_sidebar():
    """Render the application sidebar with settings and status"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System status
        st.subheader("📊 System Status")
        
        # Component availability status
        components_status = {
            "Upload": UPLOAD_COMPONENTS_AVAILABLE,
            "Extraction": EXTRACTION_COMPONENTS_AVAILABLE,
            "Annotation": ANNOTATION_COMPONENTS_AVAILABLE,
            "Matching": MATCHING_COMPONENTS_AVAILABLE,
            "Search": SEARCH_COMPONENTS_AVAILABLE,
            "Dashboard": DASHBOARD_COMPONENTS_AVAILABLE
        }
        
        for component, available in components_status.items():
            status_emoji = "✅" if available else "❌"
            st.write(f"{status_emoji} {component}")
        
        st.divider()
        
        # Processing settings
        st.subheader("🔧 Processing Settings")
        
        # API Configuration
        with st.expander("🔑 API Configuration"):
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key for AI processing"
            )
            if api_key:
                st.session_state['openai_api_key'] = api_key
                st.success("✅ API Key configured")
        
        # Processing parameters
        with st.expander("⚙️ Processing Parameters"):
            st.session_state['chunk_size'] = st.slider(
                "Text Chunk Size",
                min_value=500,
                max_value=4000,
                value=st.session_state.get('chunk_size', 2000),
                help="Size of text chunks for processing"
            )
            
            st.session_state['overlap_size'] = st.slider(
                "Chunk Overlap",
                min_value=50,
                max_value=500,
                value=st.session_state.get('overlap_size', 200),
                help="Overlap between consecutive chunks"
            )
            
            st.session_state['similarity_threshold'] = st.slider(
                "Similarity Threshold",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.get('similarity_threshold', 0.7),
                step=0.05,
                help="Minimum similarity for matching"
            )
        
        # Export options
        st.divider()
        st.subheader("📤 Export Options")
        
        if st.button("📋 Export Recommendations"):
            export_recommendations()
        
        if st.button("💬 Export Responses"):
            export_responses()
        
        if st.button("📊 Export Full Report"):
            export_full_report()
        
        # Debug information
        if st.checkbox("🐛 Debug Mode"):
            st.subheader("🔍 Debug Information")
            st.json({
                "session_keys": list(st.session_state.keys()),
                "uploaded_docs": len(st.session_state.get('uploaded_documents', [])),
                "extracted_recs": len(st.session_state.get('extracted_recommendations', [])),
                "extracted_responses": len(st.session_state.get('extracted_responses', [])),
                "components_available": components_status
            })

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'uploaded_documents': [],
        'extracted_recommendations': [],
        'extracted_responses': [],
        'processing_status': 'idle',
        'chunk_size': 2000,
        'overlap_size': 200,
        'similarity_threshold': 0.7,
        'current_tab': 0,
        'openai_api_key': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def export_recommendations():
    """Export recommendations to CSV"""
    try:
        import pandas as pd
        from io import StringIO
        
        recommendations = st.session_state.get('extracted_recommendations', [])
        if not recommendations:
            st.warning("⚠️ No recommendations to export")
            return
        
        df = pd.DataFrame(recommendations)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Recommendations CSV",
            data=csv_buffer.getvalue(),
            file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_responses():
    """Export responses to CSV"""
    try:
        import pandas as pd
        from io import StringIO
        
        responses = st.session_state.get('extracted_responses', [])
        if not responses:
            st.warning("⚠️ No responses to export")
            return
        
        df = pd.DataFrame(responses)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Responses CSV",
            data=csv_buffer.getvalue(),
            file_name=f"responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_full_report():
    """Export full analysis report"""
    try:
        from io import StringIO
        
        report = StringIO()
        report.write("# DaphneAI Analysis Report\n\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        report.write("## Summary Statistics\n\n")
        report.write(f"- Documents processed: {len(st.session_state.get('uploaded_documents', []))}\n")
        report.write(f"- Recommendations extracted: {len(st.session_state.get('extracted_recommendations', []))}\n")
        report.write(f"- Responses extracted: {len(st.session_state.get('extracted_responses', []))}\n\n")
        
        # Recommendations section
        recommendations = st.session_state.get('extracted_recommendations', [])
        if recommendations:
            report.write("## Extracted Recommendations\n\n")
            for i, rec in enumerate(recommendations, 1):
                report.write(f"### Recommendation {i}\n")
                report.write(f"**Text:** {rec.get('text', 'N/A')}\n")
                report.write(f"**Source:** {rec.get('source', 'N/A')}\n")
                report.write(f"**Confidence:** {rec.get('confidence', 'N/A')}\n\n")
        
        # Responses section
        responses = st.session_state.get('extracted_responses', [])
        if responses:
            report.write("## Extracted Responses\n\n")
            for i, resp in enumerate(responses, 1):
                report.write(f"### Response {i}\n")
                report.write(f"**Text:** {resp.get('text', 'N/A')}\n")
                report.write(f"**Source:** {resp.get('source', 'N/A')}\n")
                report.write(f"**Confidence:** {resp.get('confidence', 'N/A')}\n\n")
        
        st.download_button(
            label="📥 Download Full Report",
            data=report.getvalue(),
            file_name=f"daphne_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        
    except Exception as e:
        st.error(f"❌ Report generation failed: {str(e)}")

def render_footer():
    """Render application footer"""
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏛️ DaphneAI**")
        st.markdown("Government Document Analysis Platform")
    
    with col2:
        st.markdown("**📊 Statistics**")
        st.markdown(f"Session started: {datetime.now().strftime('%H:%M:%S')}")
    
    with col3:
        st.markdown("**ℹ️ Help**")
        if st.button("📚 View Documentation"):
            st.info("📖 Documentation coming soon!")

# ===============================================
# MAIN APPLICATION FUNCTION
# ===============================================

def main():
    """Main application entry point"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Render header
        render_header()
        
        # Render sidebar
        render_sidebar()
        
        # Render main navigation
        render_navigation_tabs()
        
        # Render footer
        render_footer()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Main application error: {e}")
        
        if st.button("🔄 Restart Application"):
            st.rerun()

# ===============================================
# MODULE EXPORTS
# ===============================================

__all__ = [
    'main',
    'render_header',
    'render_navigation_tabs',
    'render_sidebar',
    'initialize_session_state',
    'render_upload_tab',
    'render_extraction_tab',
    'render_annotation_tab',
    'render_matching_tab',
    'render_search_tab',
    'render_dashboard_tab'
]

if __name__ == "__main__":
    main()
