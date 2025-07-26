# ===============================================
# COMPLETE APP.PY - RECOMMENDATION-RESPONSE TRACKER
# Production-ready main application file
# ===============================================

import streamlit as st
import sys
import logging
import os
import traceback
from pathlib import Path
from datetime import datetime
import platform

# ===============================================
# PAGE CONFIGURATION (MUST BE FIRST)
# ===============================================

st.set_page_config(
    page_title="Recommendation-Response Tracker",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/recommendation-response-tracker',
        'Report a bug': 'https://github.com/your-repo/recommendation-response-tracker/issues',
        'About': "Recommendation-Response Tracker v2.0 - AI-powered document analysis for UK Government inquiry reports"
    }
)

# ===============================================
# PATH AND MODULE SETUP
# ===============================================

# Add modules to Python path
current_dir = Path(__file__).parent
modules_dir = current_dir / "modules"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(modules_dir))

# ===============================================
# LOGGING CONFIGURATION
# ===============================================

def setup_logging():
    """Setup comprehensive application logging"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("🚀 Application starting up")
    logging.info(f"Python version: {platform.python_version()}")
    logging.info(f"Streamlit version: {st.__version__}")

# Setup logging immediately
setup_logging()

# ===============================================
# SESSION STATE INITIALIZATION
# ===============================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
        # Document management
        st.session_state.uploaded_documents = []
        
        # Extraction results
        st.session_state.extracted_recommendations = []
        st.session_state.extracted_concerns = []  # Keep for backward compatibility
        st.session_state.extraction_results = {}
        
        # Analysis results
        st.session_state.annotation_results = {}
        st.session_state.matching_results = {}
        st.session_state.search_history = []
        st.session_state.search_results = {}
        
        # Processing states
        st.session_state.processing_status = "idle"
        st.session_state.last_processing_time = None
        st.session_state.error_messages = []
        
        # UI states
        st.session_state.selected_frameworks = []
        st.session_state.current_tab = "upload"
        st.session_state.export_ready = False
        
        # AI availability status
        st.session_state.ai_available = bool(os.getenv('OPENAI_API_KEY'))
        st.session_state.use_mock_ai = not st.session_state.ai_available
        
        logging.info("✅ Session state initialized")

# ===============================================
# DEPENDENCY CHECKS
# ===============================================

def check_critical_dependencies():
    """Check if critical dependencies are available"""
    critical_deps = {
        'streamlit': st,
        'pandas': None,
        'pathlib': Path,
        'logging': logging,
        'os': os,
        'sys': sys
    }
    
    missing_deps = []
    
    # Check pandas separately
    try:
        import pandas as pd
        critical_deps['pandas'] = pd
    except ImportError:
        missing_deps.append('pandas')
    
    if missing_deps:
        st.error(f"❌ Critical dependencies missing: {', '.join(missing_deps)}")
        st.info("Install with: `pip install pandas`")
        st.stop()
    
    logging.info("✅ Critical dependencies check passed")

def check_optional_dependencies():
    """Check optional dependencies and show status"""
    optional_deps = {
        'sentence-transformers': '🤖 Free AI Features',
        'transformers': '🔬 BERT Analysis',
        'scikit-learn': '📊 Clustering & Analytics',
        'torch': '🧠 Neural Networks',
        'openai': '🚀 GPT Integration'
    }
    
    available_deps = []
    missing_deps = []
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep.replace('-', '_'))
            available_deps.append(f"{description} ✅")
        except ImportError:
            missing_deps.append(f"{description} ❌")
    
    # Show status in sidebar
    with st.sidebar:
        st.markdown("### 🔧 AI Capabilities")
        for dep in available_deps:
            st.success(dep)
        for dep in missing_deps:
            st.warning(dep)
        
        if missing_deps:
            st.info("Install for enhanced features:")
            st.code("pip install sentence-transformers scikit-learn transformers torch")

# ===============================================
# FILE STRUCTURE VALIDATION
# ===============================================

def check_file_structure():
    """Validate that required files and directories exist"""
    required_structure = {
        'modules/': 'Module directory',
        'modules/ui/': 'UI components directory',
        'modules/ui/__init__.py': 'UI package init',
    }
    
    missing_items = []
    
    for item, description in required_structure.items():
        path = Path(item)
        if not path.exists():
            missing_items.append(f"{item} ({description})")
    
    if missing_items:
        st.error("❌ Required files/directories missing:")
        for item in missing_items:
            st.write(f"- {item}")
        return False
    
    logging.info("✅ File structure validation passed")
    return True

# ===============================================
# UI COMPONENT IMPORTS WITH FALLBACKS
# ===============================================

def import_ui_components():
    """Import UI components with comprehensive fallbacks"""
    components = {}
    
    # Upload Component
    try:
        from modules.ui.upload_components import render_upload_tab
        components['upload'] = render_upload_tab
        logging.info("✅ Upload components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Upload components not available: {e}")
        def upload_fallback():
            st.header("📁 Document Upload")
            st.info("🚧 Upload component not yet available")
            st.markdown("""
            **What this tab will do:**
            - Upload PDF documents containing government inquiry reports
            - Upload government response documents
            - Extract and validate document content
            - Prepare documents for AI analysis
            
            **Current status:** Component under development
            """)
            
            # Basic file uploader as fallback
            uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload government inquiry reports or response documents"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} files uploaded")
                # Store basic file info
                st.session_state.uploaded_documents = [
                    {
                        'filename': file.name,
                        'size': len(file.read()),
                        'type': 'pdf',
                        'uploaded_at': datetime.now().isoformat()
                    }
                    for file in uploaded_files
                ]
                
        components['upload'] = upload_fallback
    
    # Enhanced Extraction Component
    try:
        from modules.ui.extraction_components import render_extraction_tab
        components['extraction'] = render_extraction_tab
        logging.info("✅ Enhanced extraction components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Enhanced extraction components not available: {e}")
        def extraction_fallback():
            st.header("🔍 Recommendation & Response Extraction")
            st.error("❌ Enhanced extraction component not available")
            st.markdown("""
            **Required file missing:** `modules/ui/extraction_components.py`
            
            **To fix this:**
            1. Create the file `modules/ui/extraction_components.py`
            2. Copy the complete enhanced extraction code
            3. Restart the application
            
            **What enhanced extraction provides:**
            - 🧠 Smart Complete Extraction - captures full recommendations
            - 🤖 AI-Powered extraction using OpenAI GPT
            - 🔬 BERT semantic analysis
            - 📊 Advanced downloads and analytics
            - 🆓 Free AI options (no API required)
            """)
            
            # Basic fallback extraction
            if st.button("📁 Go to Upload Tab"):
                st.session_state.active_tab = 'Upload'
                st.rerun()
                
        components['extraction'] = extraction_fallback
    
    # Annotation Component
    try:
        from modules.ui.annotation_components import render_annotation_tab
        components['annotation'] = render_annotation_tab
        logging.info("✅ Annotation components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Annotation components not available: {e}")
        def annotation_fallback():
            st.header("🏷️ Concept Annotation")
            st.info("🚧 Annotation component not yet available")
            st.markdown("""
            **What this tab will do:**
            - Annotate recommendations with conceptual themes using BERT
            - Apply multiple annotation frameworks (I-SIRch, House of Commons, etc.)
            - Generate semantic embeddings for document sections
            - Provide confidence scoring for annotations
            
            **Current status:** Component under development
            """)
            
        components['annotation'] = annotation_fallback
    
    # Matching Component
    try:
        from modules.ui.matching_components import render_matching_tab
        components['matching'] = render_matching_tab
        logging.info("✅ Matching components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Matching components not available: {e}")
        def matching_fallback():
            st.header("🔗 Response Matching")
            st.info("🚧 Matching component not yet available")
            st.markdown("""
            **What this tab will do:**
            - Match recommendations to their corresponding government responses
            - Use semantic similarity and concept overlap for matching
            - Provide confidence scores for matches
            - Export matched recommendation-response pairs
            
            **Current status:** Component under development
            """)
            
        components['matching'] = matching_fallback
    
    # Search Component
    try:
        from modules.ui.search_components import render_search_tab
        components['search'] = render_search_tab
        logging.info("✅ Search components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Search components not available: {e}")
        def search_fallback():
            st.header("🔎 Smart Search")
            st.info("🚧 Search component not yet available")
            st.markdown("""
            **What this tab will do:**
            - Semantic search across all uploaded documents
            - RAG-powered query answering
            - Vector similarity search for recommendations
            - Search history and saved queries
            
            **Current status:** Component under development
            """)
            
        components['search'] = search_fallback
    
    # Dashboard Component  
    try:
        from modules.ui.dashboard_components import render_dashboard_tab
        components['dashboard'] = render_dashboard_tab
        logging.info("✅ Dashboard components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Dashboard components not available: {e}")
        def dashboard_fallback():
            st.header("📊 Analytics Dashboard")
            st.info("🚧 Dashboard component not yet available")
            st.markdown("""
            **What this tab will do:**
            - Visual analytics of recommendations and responses
            - Trend analysis across government departments
            - Implementation tracking and compliance metrics
            - Export analytics reports
            
            **Current status:** Component under development
            """)
            
            # Basic metrics as fallback
            if st.session_state.get('uploaded_documents'):
                st.metric("Documents Uploaded", len(st.session_state.uploaded_documents))
            
            if st.session_state.get('extraction_results'):
                results = st.session_state.extraction_results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recommendations", len(results.get('recommendations', [])))
                with col2:
                    st.metric("Responses", len(results.get('responses', [])))
                    
        components['dashboard'] = dashboard_fallback
    
    return components

# ===============================================
# HEADER AND NAVIGATION
# ===============================================

def render_header():
    """Render application header"""
    st.title("📋 Recommendation-Response Tracker")
    st.markdown("""
    **AI-Powered Document Analysis System** for UK Government Inquiries and Reviews
    
    Extract recommendations, analyze responses, and track implementation across inquiry reports.
    """)
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.get('uploaded_documents'):
            st.success(f"📁 {len(st.session_state.uploaded_documents)} documents loaded")
        else:
            st.info("📁 No documents uploaded")
    
    with col2:
        if st.session_state.get('extraction_results'):
            results = st.session_state.extraction_results
            total_items = len(results.get('recommendations', [])) + len(results.get('responses', []))
            st.success(f"🔍 {total_items} items extracted")
        else:
            st.info("🔍 No extractions yet")
    
    with col3:
        ai_status = "🤖 AI Available" if st.session_state.get('ai_available') else "🆓 Free AI Only"
        st.info(ai_status)

def show_welcome_message():
    """Show welcome message for new users"""
    if not st.session_state.get('uploaded_documents') and not st.session_state.get('welcome_dismissed'):
        with st.expander("👋 Welcome to Recommendation-Response Tracker", expanded=True):
            st.markdown("""
            **Get started in 3 easy steps:**
            
            1. **📁 Upload** - Upload PDF documents (inquiry reports, government responses)
            2. **🔍 Extract** - Use AI to extract recommendations and responses  
            3. **📊 Analyze** - Annotate, match, and explore your data
            
            **Key Features:**
            - 🧠 **Smart AI Extraction** - Captures complete recommendations, not fragments
            - 🆓 **Free AI Options** - No API keys required for excellent results
            - 🔬 **BERT Analysis** - Semantic understanding and concept annotation
            - 📊 **Advanced Analytics** - Comprehensive analysis and reporting
            
            **Ready to start?** Go to the **📁 Upload** tab to begin!
            """)
            
            if st.button("✅ Got it, let's start!"):
                st.session_state.welcome_dismissed = True
                st.rerun()

# ===============================================
# CUSTOM CSS STYLING
# ===============================================

def load_custom_css():
    """Load custom CSS for better styling"""
    st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        color: #1f1f1f;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ff6b6b;
        color: white;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 0.5rem;
        border-left: 4px solid;
        padding: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border-radius: 0.5rem;
        border: 2px dashed #d1d5db;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #ff6b6b;
        border-radius: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ===============================================
# ERROR HANDLING
# ===============================================

def handle_critical_error(error, context="Application"):
    """Handle critical errors with user-friendly messages"""
    error_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logging.critical(f"💥 Critical error in {context}: {error}", exc_info=True)
    
    st.error(f"🚨 Critical Error in {context}")
    
    with st.expander("🔧 Error Details", expanded=False):
        st.code(f"Error ID: {error_id}")
        st.code(f"Error: {str(error)}")
        st.code(f"Context: {context}")
        st.code(f"Time: {datetime.now().isoformat()}")
    
    st.markdown("""
    ### 💡 Troubleshooting Steps
    
    1. **Refresh the page** - Resolves most temporary issues
    2. **Clear browser cache** - Fixes persistent loading problems  
    3. **Check file structure** - Ensure all required files are present
    4. **Verify dependencies** - Run `pip install -r requirements.txt`
    5. **Restart application** - Stop and restart Streamlit
    
    **If problems persist:**
    - Check the log files for detailed error information
    - Verify all required modules are in the modules directory
    - Ensure Python environment is properly configured
    """)

# ===============================================
# MAIN APPLICATION FUNCTION
# ===============================================

def main():
    """Main application entry point"""
    try:
        # Load custom styling
        load_custom_css()
        
        # Check critical dependencies
        check_critical_dependencies()
        
        # Check optional dependencies
        check_optional_dependencies()
        
        # Check file structure
        if not check_file_structure():
            st.error("❌ Critical files are missing. Please check the installation.")
            st.stop()
        
        # Initialize session state
        initialize_session_state()
        
        # Import UI components with fallbacks
        ui_components = import_ui_components()
        
        # Render header
        render_header()
        
        # Show welcome message
        show_welcome_message()
        
        # Main navigation tabs
        tab_names = ["📁 Upload", "🔍 Extract", "🏷️ Annotate", "🔗 Match", "🔎 Search", "📊 Dashboard"]
        tabs = st.tabs(tab_names)
        
        # Render each tab with error handling
        with tabs[0]:
            try:
                ui_components['upload']()
            except Exception as e:
                handle_critical_error(e, "Upload Tab")
        
        with tabs[1]:
            try:
                ui_components['extraction']()
            except Exception as e:
                handle_critical_error(e, "Extraction Tab")
        
        with tabs[2]:
            try:
                ui_components['annotation']()
            except Exception as e:
                handle_critical_error(e, "Annotation Tab")
        
        with tabs[3]:
            try:
                ui_components['matching']()
            except Exception as e:
                handle_critical_error(e, "Matching Tab")
        
        with tabs[4]:
            try:
                ui_components['search']()
            except Exception as e:
                handle_critical_error(e, "Search Tab")
        
        with tabs[5]:
            try:
                ui_components['dashboard']()
            except Exception as e:
                handle_critical_error(e, "Dashboard Tab")
        
        # Footer information
        with st.sidebar:
            st.markdown("---")
            st.markdown("### ℹ️ System Information")
            st.info(f"""
            **Version:** 2.0.0  
            **Python:** {platform.python_version()}  
            **Streamlit:** {st.__version__}  
            **Status:** {st.session_state.get('processing_status', 'idle').title()}
            """)
            
            # Show recent activity
            if st.session_state.get('last_processing_time'):
                st.success(f"Last activity: {st.session_state.last_processing_time}")
        
        logging.info("✅ Application main loop completed successfully")
        
    except Exception as e:
        handle_critical_error(e, "Main Application")

# ===============================================
# APPLICATION ENTRY POINT
# ===============================================

if __name__ == "__main__":
    try:
        # Run main application
        main()
        
    except Exception as e:
        # Last resort error handling
        st.error("🚨 Critical Error: Application failed to start")
        st.code(f"Error: {str(e)}")
        st.code(traceback.format_exc())
        
        # Provide emergency recovery
        st.markdown("""
        ### 🆘 Emergency Recovery
        
        If you see this error, try the following:
        
        1. **Refresh the browser page**
        2. **Restart the Streamlit server**: `streamlit run app.py`
        3. **Check Python environment**: Ensure all dependencies are installed
        4. **Verify file structure**: Ensure all required files are present
        5. **Check permissions**: Ensure the application has read/write access
        
        **Technical Details:**
        - Check the console output for more detailed error messages
        - Verify that all imports are working correctly
        - Ensure the modules directory and all subdirectories exist
        """)
        
        logging.critical(f"💥 Application startup failed: {e}", exc_info=True)
