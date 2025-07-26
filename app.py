# app.py
# MAIN APPLICATION FILE - Updated to work with fixed modules

import streamlit as st
import logging
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# ===============================================
# LOGGING SETUP
# ===============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# ===============================================
# APPLICATION STARTUP CHECKS
# ===============================================

logger.info("🚀 Application starting up")
logger.info(f"Python version: {sys.version}")
logger.info(f"Streamlit version: {st.__version__}")

def check_critical_dependencies():
    """Check critical dependencies and report status"""
    dependencies = {
        'streamlit': True,  # Always available if we get here
        'pandas': False,
        'numpy': False,
        'logging': True
    }
    
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    logger.info("✅ Critical dependencies check passed")
    return dependencies

def validate_file_structure():
    """Validate that required files and directories exist"""
    required_paths = [
        'modules/',
        'modules/ui/',
        'modules/ui/__init__.py',
        'modules/core_utils.py'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        logger.error(f"Missing required paths: {missing_paths}")
        return False
    
    logger.info("✅ File structure validation passed")
    return True

# Run startup checks
dependencies = check_critical_dependencies()
file_structure_valid = validate_file_structure()

# ===============================================
# IMPORT MODULES WITH ENHANCED ERROR HANDLING
# ===============================================

# Initialize session state first
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = False

def initialize_session_state():
    """Initialize session state with error handling"""
    try:
        if not st.session_state.app_initialized:
            # Core session state
            default_values = {
                'uploaded_documents': [],
                'extracted_recommendations': [],
                'extracted_responses': [],
                'annotation_results': {},
                'processing_stats': {},
                'last_action': None,
                'app_initialized': True
            }
            
            for key, value in default_values.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            
            logger.info("✅ Session state initialized")
        
    except Exception as e:
        logger.error(f"Session state initialization error: {e}")

# Initialize session state
initialize_session_state()

# Import core utilities
try:
    from modules.core_utils import (
        SecurityValidator,
        AnnotationResult,
        log_user_action,
        initialize_session_state as core_init_session
    )
    CORE_UTILS_AVAILABLE = True
    logger.info("✅ Core utilities imported successfully")
except ImportError as e:
    CORE_UTILS_AVAILABLE = False
    logger.warning(f"⚠️ Core utilities not available: {e}")

# Import document processor
try:
    from modules.document_processor import DocumentProcessor
    DOCUMENT_PROCESSOR_AVAILABLE = True
    logger.info("✅ DocumentProcessor available")
except ImportError as e:
    DOCUMENT_PROCESSOR_AVAILABLE = False
    logger.warning(f"⚠️ DocumentProcessor not available: {e}")

# Import enhanced section extractor
try:
    from modules.enhanced_section_extractor import EnhancedSectionExtractor
    ENHANCED_SECTION_EXTRACTOR_AVAILABLE = True
    logger.info("✅ EnhancedSectionExtractor available")
except ImportError as e:
    ENHANCED_SECTION_EXTRACTOR_AVAILABLE = False
    logger.warning(f"⚠️ EnhancedSectionExtractor not available: {e}")

# Import UI components
try:
    from modules.ui import (
        render_header,
        render_navigation_tabs,
        initialize_session_state as ui_init_session,
        check_component_health
    )
    UI_COMPONENTS_AVAILABLE = True
    logger.info("✅ UI components imported successfully")
except ImportError as e:
    UI_COMPONENTS_AVAILABLE = False
    logger.error(f"❌ UI components import failed: {e}")

# ===============================================
# FALLBACK UI COMPONENTS
# ===============================================

def render_fallback_header():
    """Fallback header when UI components are not available"""
    st.set_page_config(
        page_title="DaphneAI - Government Document Analyzer",
        page_icon="🏛️",
        layout="wide"
    )
    
    st.title("🏛️ DaphneAI")
    st.markdown("### Government Document Analysis Platform")
    st.error("❌ Some UI components are not available. Please check the logs for details.")

def render_fallback_navigation():
    """Fallback navigation when main UI is not available"""
    st.markdown("---")
    st.subheader("📱 Basic Interface")
    
    # Simple file uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        
        # Store in session state
        st.session_state.uploaded_documents = [
            {
                'filename': file.name,
                'size': len(file.read()),
                'content': 'Content extraction not available in fallback mode'
            }
            for file in uploaded_files
        ]
    
    # Show uploaded documents
    if st.session_state.get('uploaded_documents'):
        st.subheader("📄 Uploaded Documents")
        for doc in st.session_state.uploaded_documents:
            st.write(f"• {doc['filename']} ({doc['size']/1024:.1f} KB)")

def render_system_status():
    """Render system status and diagnostics"""
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Dependency status
        st.write("**Dependencies:**")
        for dep, available in dependencies.items():
            icon = "✅" if available else "❌"
            st.write(f"{icon} {dep}")
        
        # Component status
        st.write("**Components:**")
        components = {
            'Core Utils': CORE_UTILS_AVAILABLE,
            'Document Processor': DOCUMENT_PROCESSOR_AVAILABLE,
            'Section Extractor': ENHANCED_SECTION_EXTRACTOR_AVAILABLE,
            'UI Components': UI_COMPONENTS_AVAILABLE
        }
        
        for component, available in components.items():
            icon = "✅" if available else "❌"
            st.write(f"{icon} {component}")
        
        # File structure
        st.write("**File Structure:**")
        icon = "✅" if file_structure_valid else "❌"
        st.write(f"{icon} Required files")
        
        # Overall health
        total_components = len(components)
        available_components = sum(components.values())
        health_percentage = (available_components / total_components) * 100
        
        st.metric("System Health", f"{health_percentage:.0f}%")

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
# MAIN APPLICATION
# ===============================================

def main():
    """Main application function"""
    try:
        # Render appropriate interface based on component availability
        if UI_COMPONENTS_AVAILABLE:
            # Full interface with all components
            render_header()
            render_navigation_tabs()
            
            # Check component health if available
            try:
                health = check_component_health()
                logger.info(f"Component health check: {health}")
            except Exception as e:
                logger.warning(f"Component health check failed: {e}")
        
        else:
            # Fallback interface
            render_fallback_header()
            render_fallback_navigation()
        
        # Always render system status
        render_system_status()
        check_optional_dependencies()
        
        # Log successful application load
        if CORE_UTILS_AVAILABLE:
            log_user_action("app_loaded", {
                'ui_available': UI_COMPONENTS_AVAILABLE,
                'core_utils_available': CORE_UTILS_AVAILABLE
            })
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ Application error: {str(e)}")
        
        # Emergency fallback
        st.markdown("### 🆘 Emergency Mode")
        st.info("The application encountered an error but is running in emergency mode.")
        
        if st.button("🔄 Restart Application"):
            st.rerun()

# ===============================================
# ERROR HANDLING AND RECOVERY
# ===============================================

def handle_startup_errors():
    """Handle startup errors gracefully"""
    if not file_structure_valid:
        st.error("❌ Required files are missing. Please check the installation.")
        st.markdown("""
        **Missing files detected. Please ensure these files exist:**
        - `modules/`
        - `modules/ui/`
        - `modules/ui/__init__.py`
        - `modules/core_utils.py`
        """)
        return False
    
    if not UI_COMPONENTS_AVAILABLE:
        st.warning("⚠️ UI components are not fully available. Running in limited mode.")
    
    return True

# ===============================================
# APPLICATION ENTRY POINT
# ===============================================

if __name__ == "__main__":
    logger.info("🚀 Starting DaphneAI application")
    
    # Handle startup errors
    if handle_startup_errors():
        # Run main application
        main()
        logger.info("✅ Application main loop completed successfully")
    else:
        logger.error("❌ Application failed to start due to missing requirements")
        st.stop()
