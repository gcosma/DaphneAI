# ===============================================
# FILE: app.py (Main Streamlit Application)
# ===============================================

import streamlit as st
import sys
import logging
from pathlib import Path
import os
from modules.ui import (
    initialize_session_state,
    render_header,
    render_navigation_tabs,
    render_upload_tab,
    render_extraction_tab,
    render_annotation_tab,
    render_matching_tab,
    render_smart_search_tab,
    render_dashboard_tab
)

# Configure page
st.set_page_config(
    page_title="Recommendation-Response Tracker",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main application entry point"""
    setup_logging()
    
    try:
        # Import components after path setup
        from streamlit_components import (
            initialize_session_state,
            render_header,
            render_navigation_tabs,
            render_upload_tab,
            render_extraction_tab,
            render_annotation_tab,
            render_matching_tab,
            render_dashboard_tab
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Render header
        render_header()
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = render_navigation_tabs()
        
        # Render tab content
        with tab1:
            render_upload_tab()
        with tab2:
            render_extraction_tab()
        with tab3:
            render_annotation_tab()
        with tab4:
            render_matching_tab()
        with tab5:
            render_dashboard_tab()
            
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logging.error(f"Main app error: {e}", exc_info=True)
        
        # Fallback UI
        st.markdown("## 🔧 System Recovery")
        st.info("The application encountered an error. Please refresh the page or check the logs.")

if __name__ == "__main__":
    main()
