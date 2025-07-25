# ===============================================
# FILE: app.py (COMPLETE AND ROBUST MAIN APPLICATION)
# ===============================================

import streamlit as st
import sys
import logging
import os
import traceback
from pathlib import Path
from datetime import datetime
import platform
import importlib

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
            logging.FileHandler(logs_dir / 'app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from external libraries
    external_loggers = ['openai', 'httpx', 'urllib3', 'requests', 'matplotlib', 'PIL']
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logging.info("🚀 Application logging initialized")
    logging.info(f"📁 Working directory: {Path.cwd()}")
    logging.info(f"🐍 Python version: {sys.version}")

# Initialize logging immediately
setup_logging()

# ===============================================
# DEPENDENCY AND ENVIRONMENT CHECKING
# ===============================================

def check_critical_dependencies():
    """Check critical dependencies required for basic functionality"""
    critical_packages = {
        'streamlit': 'Web application framework',
        'pathlib': 'File path operations',
        'logging': 'Application logging',
        'datetime': 'Date and time operations',
        'os': 'Operating system interface',
        'sys': 'System-specific parameters'
    }
    
    missing_critical = []
    
    for package, description in critical_packages.items():
        try:
            importlib.import_module(package)
        except ImportError:
            missing_critical.append(f"{package} - {description}")
    
    if missing_critical:
        st.error("❌ Critical dependencies missing - application cannot start:")
        for dep in missing_critical:
            st.caption(f"• {dep}")
        st.stop()
    
    logging.info("✅ Critical dependencies check passed")
    return True

def check_optional_dependencies():
    """Check optional dependencies and show status"""
    optional_packages = {
        'pandas': 'Data manipulation and analysis',
        'openai': 'AI-powered text extraction', 
        'pdfplumber': 'PDF text extraction',
        'fitz': 'PyMuPDF PDF processing',
        'transformers': 'BERT-based annotation',
        'chromadb': 'Vector database for search',
        'plotly': 'Interactive visualizations'
    }
    
    available_packages = []
    missing_packages = []
    
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
            available_packages.append(f"{package} - {description}")
        except ImportError:
            missing_packages.append(f"{package} - {description}")
    
    # Store in session state for display
    st.session_state.available_packages = available_packages
    st.session_state.missing_packages = missing_packages
    
    logging.info(f"✅ {len(available_packages)} optional packages available")
    logging.info(f"⚠️ {len(missing_packages)} optional packages missing")
    
    return len(available_packages), len(missing_packages)

def check_file_structure():
    """Check critical file structure"""
    critical_files = [
        "modules/__init__.py",
        "modules/document_processor.py",
        "modules/enhanced_section_extractor.py",
        "modules/ui/__init__.py",
        "modules/ui/upload_components.py"
    ]
    
    optional_files = [
        "modules/ui/extraction_components.py",
        "modules/ui/annotation_components.py", 
        "modules/ui/matching_components.py",
        "modules/ui/search_components.py",
        "modules/ui/dashboard_components.py",
        "modules/core_utils.py",
        "requirements.txt",
        ".env"
    ]
    
    missing_critical = []
    missing_optional = []
    
    for file_path in critical_files:
        if not Path(file_path).exists():
            missing_critical.append(file_path)
    
    for file_path in optional_files:
        if not Path(file_path).exists():
            missing_optional.append(file_path)
    
    # Store results
    st.session_state.missing_critical_files = missing_critical
    st.session_state.missing_optional_files = missing_optional
    
    if missing_critical:
        logging.error(f"❌ Missing critical files: {missing_critical}")
        return False
    
    logging.info("✅ File structure check passed")
    return True

# ===============================================
# SESSION STATE MANAGEMENT
# ===============================================

def initialize_session_state():
    """Initialize comprehensive session state"""
    default_values = {
        # Core data
        'uploaded_documents': [],
        'extracted_recommendations': [],
        'annotated_recommendations': [],
        'matched_responses': [],
        
        # Application state
        'app_initialized': True,
        'first_visit': True,
        'last_activity': datetime.now().isoformat(),
        'session_start_time': datetime.now().isoformat(),
        
        # Error tracking
        'upload_errors': [],
        'processing_errors': [],
        'system_errors': [],
        
        # Processing status
        'processing_status': {},
        'current_operation': None,
        'operation_progress': 0,
        
        # User preferences
        'user_preferences': {
            'extraction_mode': 'sections_only',
            'annotation_framework': 'isirch',
            'matching_threshold': 0.7,
            'auto_save': True,
            'debug_mode': False,
            'theme': 'light'
        },
        
        # Component availability
        'components_status': {},
        'features_available': {},
        
        # History tracking
        'extraction_history': [],
        'annotation_history': [],
        'matching_history': [],
        'search_history': [],
        
        # API and configuration
        'api_key_validated': False,
        'openai_available': False,
        'model_loaded': {},
        
        # UI state
        'active_tab': 'Upload',
        'sidebar_expanded': True,
        'show_debug': False,
        'show_advanced': False
    }
    
    # Initialize any missing keys
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Update activity timestamp
    st.session_state.last_activity = datetime.now().isoformat()
    
    logging.info("✅ Session state initialized with all required keys")

# ===============================================
# UI COMPONENT IMPORTS WITH ROBUST FALLBACKS
# ===============================================

def import_ui_components():
    """Import UI components with comprehensive fallback handling"""
    components = {}
    
    # Upload Component (CRITICAL - this one must work)
    try:
        from modules.ui.upload_components import render_upload_tab
        components['upload'] = render_upload_tab
        st.session_state.components_status['upload'] = 'available'
        logging.info("✅ Upload components loaded successfully")
    except Exception as e:
        logging.error(f"❌ CRITICAL: Failed to import upload components: {e}")
        def critical_upload_fallback():
            st.header("📁 Document Upload & Management")
            st.error("❌ Upload component failed to load - this is critical for basic functionality")
            st.code(f"Import error: {str(e)}")
            st.info("Please check that `modules/ui/upload_components.py` exists and all dependencies are installed.")
            if st.button("🔄 Retry Import"):
                st.rerun()
        components['upload'] = critical_upload_fallback
        st.session_state.components_status['upload'] = 'failed'
    
    # Extraction Component
    try:
        from modules.ui.extraction_components import render_extraction_tab
        components['extraction'] = render_extraction_tab
        st.session_state.components_status['extraction'] = 'available'
        logging.info("✅ Extraction components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Extraction components not available: {e}")
        def extraction_fallback():
            st.header("🔍 Recommendation & Response Extraction")
            st.info("🚧 Extraction component not yet available")
            st.markdown("""
            **What this tab will do:**
            - Extract recommendations from uploaded documents using AI or pattern matching
            - Identify government responses to recommendations
            - Parse structured content from PDFs
            
            **Current status:** Component under development
            """)
            if st.button("📁 Go to Upload Tab"):
                st.session_state.active_tab = 'Upload'
                st.rerun()
        components['extraction'] = extraction_fallback
        st.session_state.components_status['extraction'] = 'fallback'
    
    # Annotation Component
    try:
        from modules.ui.annotation_components import render_annotation_tab
        components['annotation'] = render_annotation_tab
        st.session_state.components_status['annotation'] = 'available'
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
            - Generate semantic tags for better analysis
            
            **Current status:** Component under development
            """)
        components['annotation'] = annotation_fallback
        st.session_state.components_status['annotation'] = 'fallback'
    
    # Matching Component
    try:
        from modules.ui.matching_components import render_matching_tab
        components['matching'] = render_matching_tab
        st.session_state.components_status['matching'] = 'available'
        logging.info("✅ Matching components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Matching components not available: {e}")
        def matching_fallback():
            st.header("🔗 Find Responses to Recommendations")
            st.info("🚧 Matching component not yet available")
            st.markdown("""
            **What this tab will do:**
            - Match recommendations to their corresponding government responses
            - Use semantic similarity and concept overlap
            - Provide confidence scores for matches
            
            **Current status:** Component under development
            """)
        components['matching'] = matching_fallback
        st.session_state.components_status['matching'] = 'fallback'
    
    # Search Component
    try:
        from modules.ui.search_components import render_smart_search_tab
        components['search'] = render_smart_search_tab
        st.session_state.components_status['search'] = 'available'
        logging.info("✅ Search components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Search components not available: {e}")
        def search_fallback():
            st.header("🔎 Smart Search")
            st.info("🚧 Search component not yet available")
            st.markdown("""
            **What this tab will do:**
            - Semantic search across all documents
            - RAG-powered question answering
            - Advanced filtering and discovery
            
            **Current status:** Component under development
            """)
        components['search'] = search_fallback
        st.session_state.components_status['search'] = 'fallback'
    
    # Dashboard Component
    try:
        from modules.ui.dashboard_components import render_dashboard_tab
        components['dashboard'] = render_dashboard_tab
        st.session_state.components_status['dashboard'] = 'available'
        logging.info("✅ Dashboard components loaded")
    except Exception as e:
        logging.warning(f"⚠️ Dashboard components not available: {e}")
        def dashboard_fallback():
            st.header("📊 Analytics Dashboard")
            st.info("🚧 Dashboard component not yet available")
            
            # Show basic stats from session state
            uploaded_docs = st.session_state.get('uploaded_documents', [])
            if uploaded_docs:
                st.subheader("📈 Basic Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Documents", len(uploaded_docs))
                with col2:
                    total_sections = sum(len(doc.get('sections', [])) for doc in uploaded_docs)
                    st.metric("📋 Sections", total_sections)
                with col3:
                    total_size = sum(doc.get('file_size', 0) for doc in uploaded_docs)
                    size_mb = total_size / (1024 * 1024)
                    st.metric("💾 Total Size", f"{size_mb:.1f} MB")
            else:
                st.info("📄 No documents uploaded yet. Go to Upload tab to get started.")
                
        components['dashboard'] = dashboard_fallback
        st.session_state.components_status['dashboard'] = 'fallback'
    
    return components

# ===============================================
# STYLING AND UI ENHANCEMENTS
# ===============================================

def load_custom_css():
    """Load comprehensive custom CSS styling"""
    st.markdown("""
    <style>
    /* Main application styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 0 24px;
        background-color: white;
        border-radius: 0.375rem;
        border: 2px solid transparent;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f8f9fa;
        border-color: #dee2e6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
        border-color: #1f77b4 !important;
    }
    
    /* Enhanced metrics and cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
    .status-fallback { background-color: #6c757d; }
    
    /* Enhanced message styling */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.375rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.375rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.375rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.375rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    
    /* Button enhancements */
    .stButton > button {
        border-radius: 0.375rem;
        border: 1px solid #dee2e6;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: #e9ecef;
    }
    
    .stProgress .st-bp {
        background-color: #1f77b4;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1f77b4, #17a2b8);
        color: white;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render enhanced application header"""
    st.markdown("""
    <div class="main-header">
        <h1>📋 Recommendation-Response Tracker</h1>
        <p>AI-powered analysis of UK Government inquiry reports and responses</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================================
# SIDEBAR AND STATUS DISPLAY
# ===============================================

def render_sidebar_info():
    """Render comprehensive sidebar information"""
    st.sidebar.markdown("## 🏠 Application Status")
    
    # Document metrics
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    extracted_recs = st.session_state.get('extracted_recommendations', [])
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📄 Documents", len(uploaded_docs))
    with col2:
        st.metric("📋 Recommendations", len(extracted_recs))
    
    # Component status
    st.sidebar.markdown("### 🔧 Component Status")
    
    components_status = st.session_state.get('components_status', {})
    
    for component, status in components_status.items():
        if status == 'available':
            status_class = "status-online"
            status_text = "Online"
        elif status == 'fallback':
            status_class = "status-warning" 
            status_text = "Fallback"
        else:
            status_class = "status-offline"
            status_text = "Offline"
        
        st.sidebar.markdown(f"""
        <div>
            <span class="status-indicator {status_class}"></span>
            {component.title()}: {status_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Package availability
    if st.sidebar.checkbox("📦 Show Package Status"):
        available = st.session_state.get('available_packages', [])
        missing = st.session_state.get('missing_packages', [])
        
        if available:
            st.sidebar.success(f"✅ {len(available)} packages available")
        if missing:
            st.sidebar.warning(f"⚠️ {len(missing)} packages missing")
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    if st.sidebar.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared!")
    
    if st.sidebar.button("🗑️ Clear Session"):
        for key in list(st.session_state.keys()):
            if key not in ['components_status', 'available_packages', 'missing_packages']:
                del st.session_state[key]
        st.sidebar.success("Session cleared!")
        st.rerun()

# ===============================================
# WELCOME AND ONBOARDING
# ===============================================

def show_welcome_message():
    """Show comprehensive welcome message"""
    if st.session_state.get('first_visit', True):
        st.balloons()
        
        st.markdown("""
        <div class="success-message">
            <h3>🎉 Welcome to the Recommendation-Response Tracker!</h3>
            <p>This powerful tool helps you analyze UK Government inquiry reports and their responses:</p>
            <ol>
                <li><strong>📁 Upload</strong> PDF documents (inquiry reports and government responses)</li>
                <li><strong>🔍 Extract</strong> recommendations and responses using AI or pattern matching</li>
                <li><strong>🏷️ Annotate</strong> with conceptual themes using BERT models</li>
                <li><strong>🔗 Match</strong> recommendations to their corresponding responses</li>
                <li><strong>🔎 Search</strong> through all content with semantic search</li>
                <li><strong>📊 Analyze</strong> results with interactive dashboards</li>
            </ol>
            <p><strong>👈 Start by uploading documents in the Upload tab!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.first_visit = False

# ===============================================
# SYSTEM DIAGNOSTICS AND DEBUGGING
# ===============================================

def show_system_diagnostics():
    """Show comprehensive system diagnostics"""
    if st.sidebar.checkbox("🔍 System Diagnostics"):
        st.sidebar.markdown("### 📊 System Info")
        
        # Basic system information
        info = {
            "Python": f"{sys.version.split()[0]}",
            "Streamlit": st.__version__,
            "Platform": platform.system(),
            "Session Keys": len(st.session_state.keys())
        }
        
        for key, value in info.items():
            st.sidebar.text(f"{key}: {value}")
        
        # File structure status
        if st.sidebar.button("📁 Check Files"):
            missing_critical = st.session_state.get('missing_critical_files', [])
            missing_optional = st.session_state.get('missing_optional_files', [])
            
            if not missing_critical:
                st.sidebar.success("✅ All critical files present")
            else:
                st.sidebar.error(f"❌ Missing {len(missing_critical)} critical files")
            
            if missing_optional:
                st.sidebar.warning(f"⚠️ Missing {len(missing_optional)} optional files")
        
        # Download logs
        if st.sidebar.button("📥 Download Logs"):
            log_file = Path("logs/app.log")
            if log_file.exists():
                with open(log_file, "r") as f:
                    logs = f.read()
                st.sidebar.download_button(
                    "💾 Download",
                    logs,
                    f"app_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
            else:
                st.sidebar.warning("No log file found")

# ===============================================
# ERROR HANDLING AND RECOVERY
# ===============================================

def show_recovery_options(error):
    """Show comprehensive recovery options when errors occur"""
    st.markdown("""
    <div class="error-message">
        <h3>🚨 Application Error Detected</h3>
        <p>An error occurred, but recovery options are available below.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.error(f"**Error Details:** {str(error)}")
    
    with st.expander("🔧 Recovery Options", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Restart App", use_container_width=True):
                # Clear everything and restart
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Data", use_container_width=True):
                # Clear data but keep settings
                data_keys = ['uploaded_documents', 'extracted_recommendations', 
                            'annotated_recommendations', 'matched_responses']
                for key in data_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Data cleared!")
        
        with col3:
            if st.button("💾 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
        
        with col4:
            if st.button("📋 Show Details", use_container_width=True):
                st.code(traceback.format_exc())
    
    st.markdown("""
    ### 💡 Troubleshooting Guide
    
    **Common solutions:**
    1. **Refresh the page** - Resolves most temporary issues
    2. **Clear browser cache** - Fixes persistent loading problems
    3. **Check file permissions** - Ensure app can read/write files
    4. **Verify dependencies** - Run `pip install -r requirements.txt`
    5. **Restart Streamlit** - Stop and restart the application
    
    **If problems persist:**
    - Check the log files for detailed error information
    - Verify all required files are present in the modules directory
    - Ensure Python environment is properly configured
    """)

# ===============================================
# MAIN APPLICATION FUNCTION
# ===============================================

def main():
    """Main application entry point with comprehensive error handling"""
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
                st.error(f"Error in Upload tab: {str(e)}")
                logging.error(f"Upload tab error: {e}", exc_info=True)
        
        with tabs[1]:
            try:
                ui_components['extraction']()
            except Exception as e:
                st.error(f"Error in Extraction tab: {str(e)}")
                logging.error(f"Extraction tab error: {e}", exc_info=True)
        
        with tabs[2]:
            try:
                ui_components['annotation']()
            except Exception as e:
                st.error(f"Error in Annotation tab: {str(e)}")
                logging.error(f"Annotation tab error: {e}", exc_info=True)
        
        with tabs[3]:
            try:
                ui_components['matching']()
            except Exception as e:
                st.error(f"Error in Matching tab: {str(e)}")
                logging.error(f"Matching tab error: {e}", exc_info=True)
        
        with tabs[4]:
            try:
                ui_components['search']()
            except Exception as e:
                st.error(f"Error in Search tab: {str(e)}")
                logging.error(f"Search tab error: {e}", exc_info=True)
        
        with tabs[5]:
            try:
                ui_components['dashboard']()
            except Exception as e:
                st.error(f"Error in Dashboard tab: {str(e)}")
                logging.error(f"Dashboard tab error: {e}", exc_info=True)
        
        # Render sidebar
        with st.sidebar:
            render_sidebar_info()
            show_system_diagnostics()
        
        # Update session activity
        st.session_state.last_activity = datetime.now().isoformat()
        
        logging.info("✅ Application main loop completed successfully")
        
    except Exception as e:
        logging.error(f"❌ Critical application error: {e}", exc_info=True)
        st.session_state.system_errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        show_recovery_options(e)

# ===============================================
# ADVANCED FEATURES AND UTILITIES
# ===============================================

def setup_advanced_features():
    """Setup advanced features and integrations"""
    
    # Environment variables and API keys
    if 'OPENAI_API_KEY' in os.environ:
        st.session_state.openai_available = True
        logging.info("✅ OpenAI API key detected")
    
    # Development mode
    if os.getenv('DEV_MODE', 'false').lower() == 'true':
        st.session_state.user_preferences['debug_mode'] = True
        logging.info("🔧 Development mode enabled")
    
    # Performance monitoring
    if not hasattr(st.session_state, 'performance_metrics'):
        st.session_state.performance_metrics = {
            'app_start_time': datetime.now(),
            'total_uploads': 0,
            'total_extractions': 0,
            'average_processing_time': 0
        }

def export_session_data():
    """Export session data for backup/restore"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'uploaded_documents': st.session_state.get('uploaded_documents', []),
        'extracted_recommendations': st.session_state.get('extracted_recommendations', []),
        'annotated_recommendations': st.session_state.get('annotated_recommendations', []),
        'matched_responses': st.session_state.get('matched_responses', []),
        'user_preferences': st.session_state.get('user_preferences', {}),
        'processing_history': {
            'extraction_history': st.session_state.get('extraction_history', []),
            'annotation_history': st.session_state.get('annotation_history', []),
            'matching_history': st.session_state.get('matching_history', [])
        }
    }
    return export_data

def health_check():
    """Perform comprehensive application health check"""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'components': {},
        'dependencies': {},
        'performance': {},
        'errors': []
    }
    
    # Check components
    components_status = st.session_state.get('components_status', {})
    for component, status in components_status.items():
        health_status['components'][component] = {
            'status': status,
            'last_check': datetime.now().isoformat()
        }
    
    # Check dependencies
    try:
        import pandas
        health_status['dependencies']['pandas'] = 'available'
    except ImportError:
        health_status['dependencies']['pandas'] = 'missing'
    
    try:
        import openai
        health_status['dependencies']['openai'] = 'available'
    except ImportError:
        health_status['dependencies']['openai'] = 'missing'
    
    # Check performance
    if hasattr(st.session_state, 'performance_metrics'):
        metrics = st.session_state.performance_metrics
        uptime = datetime.now() - metrics.get('app_start_time', datetime.now())
        health_status['performance'] = {
            'uptime_seconds': uptime.total_seconds(),
            'total_uploads': metrics.get('total_uploads', 0),
            'total_extractions': metrics.get('total_extractions', 0)
        }
    
    # Check for errors
    system_errors = st.session_state.get('system_errors', [])
    if system_errors:
        health_status['errors'] = system_errors[-5:]  # Last 5 errors
        health_status['overall_status'] = 'degraded'
    
    return health_status

# ===============================================
# API ENDPOINTS AND INTEGRATIONS
# ===============================================

def setup_api_integrations():
    """Setup external API integrations"""
    
    # OpenAI Integration
    if st.session_state.get('openai_available', False):
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                # Set up OpenAI client
                st.session_state.openai_client = openai.OpenAI(api_key=api_key)
                st.session_state.api_key_validated = True
                logging.info("✅ OpenAI client initialized")
        except Exception as e:
            logging.warning(f"⚠️ OpenAI setup failed: {e}")
            st.session_state.api_key_validated = False

def validate_environment():
    """Validate environment configuration"""
    validation_results = {
        'python_version': sys.version_info >= (3, 8),
        'required_directories': True,
        'write_permissions': True,
        'memory_available': True
    }
    
    # Check required directories
    required_dirs = ['modules', 'modules/ui', 'logs', 'data']
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                validation_results['required_directories'] = False
    
    # Check write permissions
    try:
        test_file = Path('logs/test_write.tmp')
        test_file.write_text('test')
        test_file.unlink()
    except Exception:
        validation_results['write_permissions'] = False
    
    return validation_results

# ===============================================
# CLEANUP AND MAINTENANCE
# ===============================================

def cleanup_old_files():
    """Cleanup old log files and temporary data"""
    try:
        logs_dir = Path('logs')
        if logs_dir.exists():
            # Remove log files older than 7 days
            cutoff_time = datetime.now().timestamp() - (7 * 24 * 60 * 60)
            for log_file in logs_dir.glob('*.log'):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    logging.info(f"🧹 Cleaned up old log file: {log_file}")
        
        # Clean up temporary files
        temp_dir = Path('temp')
        if temp_dir.exists():
            for temp_file in temp_dir.glob('*'):
                if temp_file.stat().st_mtime < cutoff_time:
                    temp_file.unlink()
                    logging.info(f"🧹 Cleaned up temp file: {temp_file}")
    
    except Exception as e:
        logging.warning(f"⚠️ Cleanup failed: {e}")

def optimize_session_state():
    """Optimize session state by removing old data"""
    try:
        # Limit history sizes
        max_history_size = 100
        
        history_keys = ['extraction_history', 'annotation_history', 'matching_history', 'search_history']
        for key in history_keys:
            if key in st.session_state and len(st.session_state[key]) > max_history_size:
                st.session_state[key] = st.session_state[key][-max_history_size:]
        
        # Limit error logs
        error_keys = ['upload_errors', 'processing_errors', 'system_errors']
        for key in error_keys:
            if key in st.session_state and len(st.session_state[key]) > 50:
                st.session_state[key] = st.session_state[key][-50:]
        
        logging.info("✅ Session state optimized")
    
    except Exception as e:
        logging.warning(f"⚠️ Session optimization failed: {e}")

# ===============================================
# APPLICATION ENTRY POINT
# ===============================================

if __name__ == "__main__":
    try:
        # Setup advanced features
        setup_advanced_features()
        
        # Setup API integrations
        setup_api_integrations()
        
        # Validate environment
        env_validation = validate_environment()
        if not all(env_validation.values()):
            st.warning("⚠️ Some environment checks failed. Application may not work correctly.")
            for check, result in env_validation.items():
                if not result:
                    st.error(f"❌ {check.replace('_', ' ').title()} check failed")
        
        # Run maintenance tasks
        cleanup_old_files()
        optimize_session_state()
        
        # Start main application
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
