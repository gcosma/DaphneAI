# ===============================================
# FILE: app.py (Main Streamlit Application)
# ===============================================

import streamlit as st
import sys
import logging
from pathlib import Path
import os
from datetime import datetime

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
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noisy library logs
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import openai
    except ImportError:
        missing_deps.append("openai")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    if missing_deps:
        st.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        st.info("Please install missing packages using: pip install " + " ".join(missing_deps))
        return False
    
    return True

def load_custom_css():
    """Load custom CSS styling"""
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

def show_welcome_message():
    """Show welcome message for new users"""
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
        
        st.balloons()
        
        st.info("""
        🎉 **Welcome to the Recommendation-Response Tracker!**
        
        This tool helps you analyze UK Government inquiry reports and responses:
        
        1. **Upload** your PDF documents (inquiry reports, government responses)
        2. **Extract** recommendations using AI or pattern matching
        3. **Analyze** and match recommendations to responses
        4. **Export** your results for further analysis
        
        Start by uploading some documents in the **Upload** tab!
        """)

def main():
    """Main application entry point"""
    setup_logging()
    
    # Check dependencies first
    if not check_dependencies():
        st.stop()
    
    # Load custom styling
    load_custom_css()
    
    try:
        # Import components after path setup
        from ui.shared_components import (
            initialize_session_state,
            render_header,
            render_navigation_tabs,
            render_sidebar_info,
            show_error_messages,
            clear_error_messages
        )
        
        from ui.upload_components import render_upload_tab
        from ui.extraction_components import render_extraction_tab
        from ui.annotation_components import render_annotation_tab
        from ui.matching_components import render_matching_tab
        from ui.search_components import render_smart_search_tab
        from ui.dashboard_components import render_dashboard_tab
        
        # Initialize session state
        initialize_session_state()
        
        # Show welcome message for first-time users
        show_welcome_message()
        
        # Render header and sidebar
        render_header()
        render_sidebar_info()
        
        # Show any error messages
        show_error_messages()
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = render_navigation_tabs()
        
        # Render tab content
        with tab1:
            try:
                render_upload_tab()
            except Exception as e:
                st.error(f"Upload tab error: {e}")
                logging.error(f"Upload tab error: {e}", exc_info=True)
        
        with tab2:
            try:
                render_extraction_tab()
            except Exception as e:
                st.error(f"Extraction tab error: {e}")
                logging.error(f"Extraction tab error: {e}", exc_info=True)
        
        with tab3:
            try:
                render_annotation_tab()
            except Exception as e:
                st.error(f"Annotation tab error: {e}")
                logging.error(f"Annotation tab error: {e}", exc_info=True)
        
        with tab4:
            try:
                render_matching_tab()
            except Exception as e:
                st.error(f"Matching tab error: {e}")
                logging.error(f"Matching tab error: {e}", exc_info=True)
        
        with tab5:
            try:
                render_smart_search_tab()
            except Exception as e:
                st.error(f"Search tab error: {e}")
                logging.error(f"Search tab error: {e}", exc_info=True)
        
        with tab6:
            try:
                render_dashboard_tab()
            except Exception as e:
                st.error(f"Dashboard tab error: {e}")
                logging.error(f"Dashboard tab error: {e}", exc_info=True)
        
        # Clear error messages after display
        if st.session_state.get('error_messages'):
            clear_error_messages()
            
    except ImportError as e:
        st.error(f"❌ Import error: {e}")
        st.info("""
        **Troubleshooting Steps:**
        1. Ensure all files are in the correct locations
        2. Check that the `modules/` directory exists
        3. Verify all required Python packages are installed
        4. Try refreshing the page
        """)
        logging.error(f"Import error: {e}", exc_info=True)
        
        # Show basic fallback UI
        show_fallback_ui()
        
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logging.error(f"Main app error: {e}", exc_info=True)
        
        # Show recovery options
        show_recovery_ui(e)

def show_fallback_ui():
    """Show a basic fallback UI when imports fail"""
    st.markdown("## 🔧 System Recovery Mode")
    
    st.warning("""
    The application is running in recovery mode due to missing components.
    """)
    
    st.markdown("### 📋 System Check")
    
    # Check file structure
    required_files = [
        "modules/uk_inquiry_extractor.py",
        "modules/ui/shared_components.py",
        "modules/ui/extraction_components.py",
        "modules/ui/upload_components.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            st.success(f"✅ {file_path}")
        else:
            st.error(f"❌ {file_path} - Missing")
    
    # Check Python packages
    st.markdown("### 📦 Package Check")
    
    required_packages = ["streamlit", "pandas", "openai", "logging"]
    
    for package in required_packages:
        try:
            __import__(package)
            st.success(f"✅ {package}")
        except ImportError:
            st.error(f"❌ {package} - Not installed")
    
    if st.button("🔄 Retry Application"):
        st.rerun()

def show_recovery_ui(error):
    """Show recovery options when the app encounters errors"""
    st.markdown("## 🚨 Application Error Recovery")
    
    st.error(f"Error details: {str(error)}")
    
    st.markdown("### 🔧 Recovery Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Restart Application", use_container_width=True):
            # Clear session state and restart
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
    
    with col3:
        if st.button("📝 View Logs", use_container_width=True):
            if Path("app.log").exists():
                with open("app.log", "r") as f:
                    logs = f.read().split("\n")[-50:]  # Last 50 lines
                st.text_area("Recent logs:", "\n".join(logs), height=300)
    
    st.markdown("### 💡 Common Solutions")
    st.markdown("""
    1. **Refresh the page** - Often resolves temporary issues
    2. **Check file permissions** - Ensure the app can read/write files
    3. **Verify API keys** - Check OpenAI API key is set correctly
    4. **Update dependencies** - Run `pip install -r requirements.txt`
    5. **Restart the server** - Stop and restart the Streamlit application
    """)

def show_system_info():
    """Show system information for debugging"""
    if st.checkbox("🔍 Show System Info"):
        st.markdown("### System Information")
        
        import platform
        import sys
        
        info = {
            "Python Version": sys.version,
            "Platform": platform.platform(),
            "Streamlit Version": st.__version__,
            "Current Directory": str(Path.cwd()),
            "App Start Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for key, value in info.items():
            st.write(f"**{key}:** {value}")
        
        # Environment variables (safely)
        st.write("**Environment Variables:**")
        env_vars = ["OPENAI_API_KEY", "PYTHONPATH", "PATH"]
        for var in env_vars:
            value = os.getenv(var, "Not set")
            if var == "OPENAI_API_KEY" and value != "Not set":
                value = f"Set (length: {len(value)})"
            st.write(f"• {var}: {value}")

if __name__ == "__main__":
    # Add system info option
    with st.sidebar:
        show_system_info()
    
    # Run main application
    main()
