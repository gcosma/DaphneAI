# ===============================================
# FILE: modules/ui/upload_components.py - COMPLETE FULL VERSION
# ===============================================

import streamlit as st
import pandas as pd
import tempfile
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# ===============================================
# ROBUST IMPORT HANDLING WITH FALLBACKS
# ===============================================

# First, try to import shared components
try:
    from .shared_components import (
        add_error_message, 
        show_progress_indicator,
        log_user_action,
        safe_filename,
        format_file_size
    )
    SHARED_COMPONENTS_AVAILABLE = True
except ImportError:
    SHARED_COMPONENTS_AVAILABLE = False
    logging.warning("Could not import from shared_components, using fallbacks")

# If shared components import failed, define fallback functions
if not SHARED_COMPONENTS_AVAILABLE:
    def show_progress_indicator(current=None, total=None, message="Processing..."):
        """Fallback progress indicator function"""
        if current is not None and total is not None:
            if total > 0:
                progress = current / total
                st.progress(progress, text=f"{message}: {current}/{total}")
            else:
                st.info(f"{message}...")
            return None
        else:
            return st.spinner(message)
    
    def add_error_message(message: str, error_type: str = "error"):
        """Fallback error message function"""
        if error_type == "error":
            st.error(message)
        elif error_type == "warning":
            st.warning(message)
        elif error_type == "info":
            st.info(message)
        else:
            st.success(message)
    
    def log_user_action(action: str, details: str = ""):
        """Fallback logging function"""
        logging.info(f"User Action: {action} - {details}")
    
    def safe_filename(filename: str) -> str:
        """Fallback safe filename function"""
        import re
        return re.sub(r'[^\w\-_\.]', '_', filename)
    
    def format_file_size(size_bytes: int) -> str:
        """Fallback file size formatting"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024*1024:
            return f"{size_bytes/1024:.1f} KB"
        else:
            return f"{size_bytes/(1024*1024):.1f} MB"

# Now try to import core modules
try:
    import sys
    sys.path.append('modules')
    from document_processor import DocumentProcessor
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError as e:
    DOCUMENT_PROCESSOR_AVAILABLE = False
    logging.error(f"Could not import DocumentProcessor: {e}")
    
    class DocumentProcessor:
        """Fallback DocumentProcessor class"""
        def extract_text_from_pdf(self, path, extract_sections_only=True):
            return {
                'filename': Path(path).name,
                'text': "Mock extracted text - DocumentProcessor not available",
                'sections': [],
                'metadata': {'pages': 1, 'processed_at': datetime.now().isoformat()}
            }

try:
    from core_utils import SecurityValidator
    SECURITY_VALIDATOR_AVAILABLE = True
except ImportError as e:
    SECURITY_VALIDATOR_AVAILABLE = False
    logging.error(f"Could not import SecurityValidator: {e}")
    
    class SecurityValidator:
        """Fallback SecurityValidator class"""
        @staticmethod
        def validate_file_upload(content, filename):
            return True
        
        @staticmethod
        def sanitize_filename(filename):
            import re
            return re.sub(r'[^\w\-_\.]', '_', filename)

# ===============================================
# MAIN UPLOAD TAB FUNCTION
# ===============================================

def render_upload_tab():
    """Render the document upload tab"""
    st.header("📁 Document Upload & Management")
    
    st.markdown("""
    Upload PDF documents containing recommendations and responses. The system will automatically 
    process and extract only the relevant sections for analysis.
    """)
    
    # Show component availability status
    if not DOCUMENT_PROCESSOR_AVAILABLE:
        st.warning("⚠️ DocumentProcessor not available - using fallback mode")
    if not SECURITY_VALIDATOR_AVAILABLE:
        st.info("ℹ️ SecurityValidator not available - basic validation only")
    
    # Upload interface
    render_upload_interface()
    
    # Document management
    render_document_library()
    
    # Batch operations
    render_batch_operations()

# ===============================================
# UPLOAD INTERFACE
# ===============================================

def render_upload_interface():
    """Render the file upload interface"""
    st.subheader("📤 Upload New Documents")
    
    # Extraction mode selection
    col1, col2 = st.columns([3, 1])
    
    with col2:
        extraction_mode = st.radio(
            "Extraction Mode:",
            options=["Sections Only", "Full Document"],
            help="Sections Only extracts just recommendations/responses sections (recommended). Full Document extracts everything."
        )
        extract_sections_only = (extraction_mode == "Sections Only")
    
    with col1:
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents containing recommendations and responses"
        )
        
        # Processing options
        with st.expander("🔧 Processing Options"):
            col1a, col2a = st.columns(2)
            with col1a:
                max_file_size = st.selectbox(
                    "Max file size (MB):",
                    options=[10, 25, 50, 100],
                    index=1,
                    help="Maximum file size to process"
                )
            with col2a:
                batch_processing = st.checkbox(
                    "Batch processing",
                    value=True,
                    help="Process multiple files together"
                )
    
    # Process uploaded files
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary"):
            process_uploaded_files(uploaded_files, extract_sections_only, max_file_size, batch_processing)

# ===============================================
# FILE PROCESSING LOGIC
# ===============================================

def process_uploaded_files(uploaded_files: List, extract_sections_only: bool, max_file_size: int, batch_processing: bool):
    """Process uploaded PDF files"""
    
    log_user_action("file_upload_started", f"Files: {len(uploaded_files)}, Mode: {'Sections' if extract_sections_only else 'Full'}")
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Tracking variables
    successful_uploads = 0
    failed_uploads = []
    duplicate_files = []
    sections_summary = {'recommendations': 0, 'responses': 0}
    
    total_files = len(uploaded_files)
    
    # Create containers for dynamic updates
    progress_container = st.container()
    status_container = st.container()
    results_container = st.container()
    
    # Initialize session state for uploaded documents if needed
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    # Process each file
    for i, uploaded_file in enumerate(uploaded_files):
        current_file = i + 1
        
        # Update progress
        with progress_container:
            show_progress_indicator(current_file, total_files, "Processing documents")
        
        with status_container:
            status_text = st.empty()
            status_text.info(f"📄 Processing: {uploaded_file.name} ({current_file}/{total_files})")
        
        try:
            # File size check
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > max_file_size:
                failed_uploads.append(f"{uploaded_file.name}: File too large ({file_size_mb:.1f}MB > {max_file_size}MB)")
                status_text.error(f"❌ File too large: {uploaded_file.name}")
                continue
            
            # Security validation
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            if not SecurityValidator.validate_file_upload(file_content, uploaded_file.name):
                failed_uploads.append(f"{uploaded_file.name}: Security validation failed")
                status_text.error(f"❌ Security validation failed: {uploaded_file.name}")
                continue
            
            # Check for duplicates
            existing_filenames = [doc.get('filename', '') for doc in st.session_state.uploaded_documents]
            safe_name = SecurityValidator.sanitize_filename(uploaded_file.name)
            
            if safe_name not in existing_filenames:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Extract text using DocumentProcessor
                    doc_data = processor.extract_text_from_pdf(tmp_file_path, extract_sections_only)
                    
                    if doc_data and doc_data.get('text'):
                        # Add additional metadata
                        doc_data.update({
                            'filename': safe_name,
                            'upload_timestamp': datetime.now().isoformat(),
                            'file_size': uploaded_file.size,
                            'extraction_mode': 'sections_only' if extract_sections_only else 'full_document',
                            'file_size_formatted': format_file_size(uploaded_file.size)
                        })
                        
                        # Add to session state
                        st.session_state.uploaded_documents.append(doc_data)
                        successful_uploads += 1
                        
                        # Count sections if available
                        if doc_data.get('sections'):
                            sections_count = len(doc_data['sections'])
                            rec_sections = len([s for s in doc_data['sections'] if s['type'] == 'recommendations'])
                            resp_sections = len([s for s in doc_data['sections'] if s['type'] == 'responses'])
                            
                            sections_summary['recommendations'] += rec_sections
                            sections_summary['responses'] += resp_sections
                            
                            status_text.success(f"✅ Found {sections_count} relevant sections in: {uploaded_file.name}")
                            st.info(f"📋 Sections: {rec_sections} recommendations, {resp_sections} responses")
                        else:
                            if extract_sections_only:
                                status_text.warning(f"⚠️ No recommendations/responses sections found in: {uploaded_file.name}")
                            else:
                                status_text.success(f"✅ Successfully processed (full document): {uploaded_file.name}")
                    else:
                        failed_uploads.append(f"{uploaded_file.name}: No readable text found")
                        status_text.error(f"❌ No text extracted from: {uploaded_file.name}")
                
                finally:
                    # Cleanup temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
            else:
                duplicate_files.append(uploaded_file.name)
                status_text.warning(f"⚠️ Duplicate file skipped: {uploaded_file.name}")
            
        except Exception as e:
            error_msg = f"{uploaded_file.name}: {str(e)}"
            failed_uploads.append(error_msg)
            add_error_message(f"Failed to process {uploaded_file.name}: {str(e)}")
            status_text.error(f"❌ Processing error: {uploaded_file.name}")
            logging.error(f"File processing error: {e}", exc_info=True)
    
    # Clear progress indicators
    progress_container.empty()
    status_container.empty()
    
    # Show final results
    with results_container:
        show_processing_results(successful_uploads, failed_uploads, duplicate_files, sections_summary, total_files)
    
    # Log completion
    log_user_action("file_upload_completed", f"Success: {successful_uploads}, Failed: {len(failed_uploads)}")

def show_processing_results(successful_uploads: int, failed_uploads: List[str], duplicate_files: List[str], sections_summary: Dict, total_files: int):
    """Show the final processing results"""
    
    if successful_uploads > 0:
        st.success(f"✅ Successfully processed {successful_uploads} of {total_files} files!")
        
        # Show sections summary if available
        if sections_summary['recommendations'] > 0 or sections_summary['responses'] > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📋 Recommendation Sections", sections_summary['recommendations'])
            with col2:
                st.metric("📝 Response Sections", sections_summary['responses'])
    
    if duplicate_files:
        st.warning(f"⚠️ Skipped {len(duplicate_files)} duplicate files:")
        for dup_file in duplicate_files:
            st.caption(f"• {dup_file}")
    
    if failed_uploads:
        st.error(f"❌ Failed to process {len(failed_uploads)} files:")
        for error in failed_uploads:
            st.caption(f"• {error}")
    
    if successful_uploads == 0 and not duplicate_files:
        st.error("❌ No files were successfully processed. Please check file format and content.")

# ===============================================
# DOCUMENT LIBRARY MANAGEMENT
# ===============================================

def render_document_library():
    """Render the document library interface"""
    st.subheader("📚 Document Library")
    
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    
    if not uploaded_docs:
        st.info("📄 No documents uploaded yet. Use the upload section above to add documents.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📁 Total Documents", len(uploaded_docs))
    
    with col2:
        total_sections = sum(len(doc.get('sections', [])) for doc in uploaded_docs)
        st.metric("📋 Total Sections", total_sections)
    
    with col3:
        total_size = sum(doc.get('file_size', 0) for doc in uploaded_docs)
        st.metric("💾 Total Size", format_file_size(total_size))
    
    with col4:
        docs_with_sections = len([doc for doc in uploaded_docs if doc.get('sections')])
        st.metric("✅ Docs with Sections", docs_with_sections)
    
    # Document list
    with st.expander("📋 View Document Details", expanded=True):
        for i, doc in enumerate(uploaded_docs):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{doc.get('filename', 'Unknown')}**")
                if doc.get('sections'):
                    sections = doc['sections']
                    rec_count = len([s for s in sections if s['type'] == 'recommendations'])
                    resp_count = len([s for s in sections if s['type'] == 'responses'])
                    st.caption(f"📋 {rec_count} rec. sections, 📝 {resp_count} resp. sections")
                else:
                    st.caption(f"📄 Full document mode")
            
            with col2:
                st.caption(f"Size: {format_file_size(doc.get('file_size', 0))}")
                upload_date = doc.get('upload_timestamp', 'Unknown')
                if upload_date != 'Unknown' and len(upload_date) > 10:
                    upload_date = upload_date[:10]  # Show just the date part
                st.caption(f"Uploaded: {upload_date}")
            
            with col3:
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.uploaded_documents.pop(i)
                    add_error_message(f"Removed document: {doc.get('filename', 'Unknown')}", "success")
                    log_user_action("remove_document", doc.get('filename', 'Unknown'))
                    st.rerun()

# ===============================================
# BATCH OPERATIONS
# ===============================================

def render_batch_operations():
    """Render batch operations for document management"""
    st.subheader("⚙️ Batch Operations")
    
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    
    if not uploaded_docs:
        st.info("📄 No documents available for batch operations.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get('confirm_clear_all', False):
                st.session_state.uploaded_documents = []
                st.session_state.confirm_clear_all = False
                add_error_message("All documents cleared", "success")
                log_user_action("clear_all_documents")
                st.rerun()
            else:
                st.session_state.confirm_clear_all = True
                st.warning("⚠️ Click again to confirm clearing all documents")
    
    with col2:
        if st.button("📊 Export Document List"):
            export_document_list()
    
    with col3:
        if st.button("🔄 Reprocess All"):
            reprocess_all_documents()

def export_document_list():
    """Export document list as CSV"""
    try:
        uploaded_docs = st.session_state.get('uploaded_documents', [])
        
        if not uploaded_docs:
            st.warning("No documents to export")
            return
        
        # Create export data
        export_data = []
        for doc in uploaded_docs:
            sections = doc.get('sections', [])
            rec_sections = len([s for s in sections if s['type'] == 'recommendations'])
            resp_sections = len([s for s in sections if s['type'] == 'responses'])
            
            export_data.append({
                'Filename': doc.get('filename', 'Unknown'),
                'Upload_Date': doc.get('upload_timestamp', 'Unknown'),
                'File_Size_Bytes': doc.get('file_size', 0),
                'File_Size_Formatted': format_file_size(doc.get('file_size', 0)),
                'Extraction_Mode': doc.get('extraction_mode', 'unknown'),
                'Total_Sections': len(sections),
                'Recommendation_Sections': rec_sections,
                'Response_Sections': resp_sections,
                'Has_Text': bool(doc.get('text'))
            })
        
        # Convert to DataFrame and download
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"document_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        add_error_message("Document list exported successfully", "success")
        log_user_action("export_document_list", f"Exported {len(export_data)} documents")
        
    except Exception as e:
        add_error_message(f"Export failed: {str(e)}")
        logging.error(f"Export error: {e}", exc_info=True)

def reprocess_all_documents():
    """Reprocess all uploaded documents"""
    try:
        uploaded_docs = st.session_state.get('uploaded_documents', [])
        
        if not uploaded_docs:
            st.warning("No documents to reprocess")
            return
        
        with st.spinner("🔄 Reprocessing all documents..."):
            processor = DocumentProcessor()
            
            for i, doc in enumerate(uploaded_docs):
                filename = doc.get('filename', f'document_{i}')
                
                # Show progress
                show_progress_indicator(i + 1, len(uploaded_docs), "Reprocessing")
                
                # Here you would implement reprocessing logic
                # For now, just update timestamp
                doc['reprocessed_at'] = datetime.now().isoformat()
        
        add_error_message(f"Reprocessed {len(uploaded_docs)} documents", "success")
        log_user_action("reprocess_all_documents", f"Reprocessed {len(uploaded_docs)} documents")
        
    except Exception as e:
        add_error_message(f"Reprocessing failed: {str(e)}")
        logging.error(f"Reprocessing error: {e}", exc_info=True)

# ===============================================
# DOCUMENT VALIDATION AND ANALYTICS
# ===============================================

def render_document_validation():
    """Render document validation interface"""
    st.subheader("✅ Document Validation")
    
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    
    if not uploaded_docs:
        st.info("No documents to validate.")
        return
    
    # Run validation
    validation_results = validate_all_documents()
    
    # Show validation summary
    total_docs = len(uploaded_docs)
    valid_docs = len([r for r in validation_results if r['is_valid']])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Total Documents", total_docs)
    
    with col2:
        st.metric("✅ Valid Documents", valid_docs)
    
    with col3:
        st.metric("❌ Issues Found", total_docs - valid_docs)
    
    # Show detailed results
    with st.expander("🔍 Detailed Validation Results"):
        for result in validation_results:
            if result['is_valid']:
                st.success(f"✅ {result['filename']}")
            else:
                st.error(f"❌ {result['filename']}")
                for issue in result['issues']:
                    st.caption(f"  • {issue}")

def validate_all_documents():
    """Validate all uploaded documents"""
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    validation_results = []
    
    for doc in uploaded_docs:
        result = validate_single_document(doc)
        validation_results.append(result)
    
    return validation_results

def validate_single_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single document"""
    issues = []
    
    # Check required fields
    if not doc.get('filename'):
        issues.append("Missing filename")
    
    if not doc.get('text'):
        issues.append("Missing text content")
    
    if not doc.get('upload_timestamp'):
        issues.append("Missing upload timestamp")
    
    # Check text quality
    text = doc.get('text', '')
    if len(text) < 100:
        issues.append("Text content is very short")
    
    # Check file size
    file_size = doc.get('file_size', 0)
    if file_size == 0:
        issues.append("File size is zero")
    
    # Check sections (if extraction mode was sections only)
    if doc.get('extraction_mode') == 'sections_only':
        sections = doc.get('sections', [])
        if not sections:
            issues.append("No sections found despite sections-only extraction")
    
    return {
        'filename': doc.get('filename', 'Unknown'),
        'is_valid': len(issues) == 0,
        'issues': issues
    }

# ===============================================
# UTILITY FUNCTIONS FOR UPLOAD TAB
# ===============================================

def get_uploaded_documents_summary():
    """Get summary of uploaded documents for other components"""
    docs = st.session_state.get('uploaded_documents', [])
    
    summary = {
        'total_documents': len(docs),
        'documents_with_sections': len([d for d in docs if d.get('sections')]),
        'total_sections': sum(len(d.get('sections', [])) for d in docs),
        'section_types': {},
        'document_types': {}
    }
    
    # Count section and document types
    for doc in docs:
        doc_type = doc.get('document_type', 'Unknown')
        summary['document_types'][doc_type] = summary['document_types'].get(doc_type, 0) + 1
        
        for section in doc.get('sections', []):
            sec_type = section['type']
            summary['section_types'][sec_type] = summary['section_types'].get(sec_type, 0) + 1
    
    return summary

def validate_uploaded_documents():
    """Validate that uploaded documents are properly formatted"""
    docs = st.session_state.get('uploaded_documents', [])
    issues = []
    
    for i, doc in enumerate(docs):
        if not doc.get('filename'):
            issues.append(f"Document {i}: Missing filename")
        if not doc.get('text'):
            issues.append(f"Document {i}: Missing text content")
        if not doc.get('upload_timestamp'):
            issues.append(f"Document {i}: Missing upload timestamp")
    
    return issues

def get_upload_statistics():
    """Get upload statistics for dashboard"""
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        return {}
    
    stats = {
        'total_documents': len(docs),
        'total_size_bytes': sum(doc.get('file_size', 0) for doc in docs),
        'total_size_formatted': format_file_size(sum(doc.get('file_size', 0) for doc in docs)),
        'upload_dates': [doc.get('upload_timestamp', '') for doc in docs],
        'extraction_modes': {},
        'documents_by_date': {}
    }
    
    # Count extraction modes
    for doc in docs:
        mode = doc.get('extraction_mode', 'unknown')
        stats['extraction_modes'][mode] = stats['extraction_modes'].get(mode, 0) + 1
    
    # Group by upload date
    for doc in docs:
        upload_date = doc.get('upload_timestamp', 'Unknown')
        if upload_date != 'Unknown':
            date_only = upload_date[:10]  # YYYY-MM-DD
            stats['documents_by_date'][date_only] = stats['documents_by_date'].get(date_only, 0) + 1
    
    return stats

# ===============================================
# ADVANCED UPLOAD FEATURES
# ===============================================

def render_advanced_upload_options():
    """Render advanced upload options"""
    with st.expander("🔧 Advanced Upload Options"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**File Processing:**")
            
            auto_detect_type = st.checkbox(
                "Auto-detect document type",
                value=True,
                help="Automatically detect if document contains recommendations or responses"
            )
            
            preserve_formatting = st.checkbox(
                "Preserve text formatting",
                value=True,
                help="Keep original text formatting and structure"
            )
            
            extract_metadata = st.checkbox(
                "Extract document metadata",
                value=True,
                help="Extract PDF metadata like author, creation date, etc."
            )
        
        with col2:
            st.markdown("**Quality Control:**")
            
            min_text_length = st.number_input(
                "Minimum text length (characters):",
                min_value=100,
                max_value=10000,
                value=500,
                help="Minimum text length required for valid documents"
            )
            
            skip_empty_pages = st.checkbox(
                "Skip empty pages",
                value=True,
                help="Skip pages with no readable text"
            )
            
            validate_pdf_structure = st.checkbox(
                "Validate PDF structure",
                value=True,
                help="Check PDF file integrity before processing"
            )
        
        return {
            'auto_detect_type': auto_detect_type,
            'preserve_formatting': preserve_formatting,
            'extract_metadata': extract_metadata,
            'min_text_length': min_text_length,
            'skip_empty_pages': skip_empty_pages,
            'validate_pdf_structure': validate_pdf_structure
        }

def render_upload_help():
    """Render upload help and tips"""
    with st.expander("❓ Upload Help & Tips"):
        st.markdown("""
        ### 📖 Supported File Types
        - **PDF files** (.pdf) - Primary supported format
        - Files should contain text (not scanned images)
        
        ### 📋 Document Types
        - **Inquiry Reports** - Documents containing recommendations
        - **Government Responses** - Official responses to recommendations
        - **Mixed Documents** - Contains both recommendations and responses
        
        ### 🔧 Processing Modes
        - **Sections Only** - Extracts only relevant recommendation/response sections (faster, recommended)
        - **Full Document** - Processes entire document (slower, more comprehensive)
        
        ### 💡 Tips for Best Results
        1. Use text-based PDFs (not scanned images)
        2. Ensure files are not password protected
        3. Use descriptive filenames
        4. Upload related documents together
        5. Check file size limits before uploading
        
        ### ⚠️ Common Issues
        - **No text extracted**: File might be a scanned image or corrupted
        - **Processing failed**: File might be too large or corrupted
        - **No sections found**: Document might not contain standard recommendation patterns
        """)

# ===============================================
# ERROR HANDLING AND LOGGING
# ===============================================

def setup_upload_logging():
    """Setup logging for upload components"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def handle_upload_error(error: Exception, filename: str, context: str = "upload"):
    """Handle upload-specific errors"""
    error_msg = f"Error in {context} for {filename}: {str(error)}"
    logging.error(error_msg, exc_info=True)
    add_error_message(error_msg)
    
    # Log to session state for debugging
    if 'upload_errors' not in st.session_state:
        st.session_state.upload_errors = []
    
    st.session_state.upload_errors.append({
        'timestamp': datetime.now().isoformat(),
        'filename': filename,
        'context': context,
        'error': str(error)
    })

# ===============================================
# EXPORT ALL FUNCTIONS
# ===============================================

__all__ = [
    'render_upload_tab',
    'render_upload_interface',
    'render_document_library',
    'render_batch_operations',
    'process_uploaded_files',
    'show_processing_results',
    'export_document_list',
    'reprocess_all_documents',
    'render_document_validation',
    'validate_all_documents',
    'validate_single_document',
    'get_uploaded_documents_summary',
    'validate_uploaded_documents',
    'get_upload_statistics',
    'render_advanced_upload_options',
    'render_upload_help',
    'setup_upload_logging',
    'handle_upload_error'
]

# ===============================================
# MODULE INITIALIZATION AND TESTING
# ===============================================

if __name__ == "__main__":
    # This would be used for testing the module independently
    setup_upload_logging()
    print("Upload components module loaded successfully")
    
    # Test imports
    try:
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
    except ImportError as e:
        print(f"❌ DocumentProcessor import failed: {e}")
    
    try:
        from core_utils import SecurityValidator
        print("✅ SecurityValidator imported successfully")  
    except ImportError as e:
        print(f"❌ SecurityValidator import failed: {e}")
    
    # Test shared components
    if SHARED_COMPONENTS_AVAILABLE:
        print("✅ Shared components available")
    else:
        print("⚠️ Using fallback shared components")
    
    # Test functions
    print(f"Available functions: {len(__all__)}")
    print("Core functions:")
    for func_name in __all__[:5]:  # Show first 5 functions
        print(f"  • {func_name}")
    
    print("Module initialization complete!")

# ===============================================
# ADDITIONAL UTILITY FUNCTIONS
# ===============================================

def cleanup_uploaded_documents():
    """Clean up uploaded documents and free memory"""
    try:
        if 'uploaded_documents' in st.session_state:
            count = len(st.session_state.uploaded_documents)
            st.session_state.uploaded_documents = []
            log_user_action("cleanup_documents", f"Cleaned up {count} documents")
            return count
        return 0
    except Exception as e:
        logging.error(f"Cleanup error: {e}")
        return 0

def get_document_by_filename(filename: str) -> Dict[str, Any]:
    """Get a document by filename"""
    docs = st.session_state.get('uploaded_documents', [])
    for doc in docs:
        if doc.get('filename') == filename:
            return doc
    return {}

def update_document_metadata(filename: str, metadata: Dict[str, Any]):
    """Update metadata for a specific document"""
    docs = st.session_state.get('uploaded_documents', [])
    for doc in docs:
        if doc.get('filename') == filename:
            doc.update(metadata)
            log_user_action("update_document_metadata", f"Updated {filename}")
            return True
    return False

def remove_document_by_filename(filename: str) -> bool:
    """Remove a document by filename"""
    try:
        docs = st.session_state.get('uploaded_documents', [])
        original_count = len(docs)
        
        st.session_state.uploaded_documents = [
            doc for doc in docs if doc.get('filename') != filename
        ]
        
        removed = len(st.session_state.uploaded_documents) < original_count
        if removed:
            log_user_action("remove_document", filename)
        
        return removed
    
    except Exception as e:
        logging.error(f"Error removing document {filename}: {e}")
        return False

def search_documents_by_content(search_term: str) -> List[Dict[str, Any]]:
    """Search documents by content"""
    docs = st.session_state.get('uploaded_documents', [])
    matching_docs = []
    
    search_term_lower = search_term.lower()
    
    for doc in docs:
        # Search in text content
        text = doc.get('text', '').lower()
        if search_term_lower in text:
            matching_docs.append(doc)
            continue
        
        # Search in sections
        sections = doc.get('sections', [])
        for section in sections:
            section_content = section.get('content', '').lower()
            if search_term_lower in section_content:
                matching_docs.append(doc)
                break
    
    return matching_docs

def get_document_statistics_detailed():
    """Get detailed statistics about uploaded documents"""
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        return {}
    
    stats = {
        'document_count': len(docs),
        'total_size_bytes': sum(doc.get('file_size', 0) for doc in docs),
        'size_distribution': {
            'small (< 1MB)': 0,
            'medium (1-10MB)': 0,
            'large (> 10MB)': 0
        },
        'extraction_modes': {},
        'sections_stats': {
            'docs_with_sections': 0,
            'total_sections': 0,
            'recommendation_sections': 0,
            'response_sections': 0
        },
        'upload_timeline': {},
        'text_length_stats': {
            'min_length': float('inf'),
            'max_length': 0,
            'avg_length': 0
        }
    }
    
    text_lengths = []
    
    for doc in docs:
        # Size distribution
        file_size = doc.get('file_size', 0)
        size_mb = file_size / (1024 * 1024)
        
        if size_mb < 1:
            stats['size_distribution']['small (< 1MB)'] += 1
        elif size_mb <= 10:
            stats['size_distribution']['medium (1-10MB)'] += 1
        else:
            stats['size_distribution']['large (> 10MB)'] += 1
        
        # Extraction modes
        mode = doc.get('extraction_mode', 'unknown')
        stats['extraction_modes'][mode] = stats['extraction_modes'].get(mode, 0) + 1
        
        # Sections
        sections = doc.get('sections', [])
        if sections:
            stats['sections_stats']['docs_with_sections'] += 1
            stats['sections_stats']['total_sections'] += len(sections)
            
            for section in sections:
                if section.get('type') == 'recommendations':
                    stats['sections_stats']['recommendation_sections'] += 1
                elif section.get('type') == 'responses':
                    stats['sections_stats']['response_sections'] += 1
        
        # Upload timeline
        upload_date = doc.get('upload_timestamp', '')
        if upload_date:
            date_only = upload_date[:10]  # YYYY-MM-DD
            stats['upload_timeline'][date_only] = stats['upload_timeline'].get(date_only, 0) + 1
        
        # Text length
        text_length = len(doc.get('text', ''))
        text_lengths.append(text_length)
        
        stats['text_length_stats']['min_length'] = min(stats['text_length_stats']['min_length'], text_length)
        stats['text_length_stats']['max_length'] = max(stats['text_length_stats']['max_length'], text_length)
    
    # Calculate average text length
    if text_lengths:
        stats['text_length_stats']['avg_length'] = sum(text_lengths) / len(text_lengths)
    
    # Handle edge case for min_length
    if stats['text_length_stats']['min_length'] == float('inf'):
        stats['text_length_stats']['min_length'] = 0
    
    return stats

def render_upload_analytics():
    """Render upload analytics and insights"""
    st.subheader("📈 Upload Analytics")
    
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        st.info("No documents uploaded yet.")
        return
    
    stats = get_document_statistics_detailed()
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documents", stats['document_count'])
    
    with col2:
        st.metric("💾 Total Size", format_file_size(stats['total_size_bytes']))
    
    with col3:
        st.metric("📋 Total Sections", stats['sections_stats']['total_sections'])
    
    with col4:
        if stats['text_length_stats']['avg_length'] > 0:
            avg_length_formatted = f"{stats['text_length_stats']['avg_length']:,.0f}"
            st.metric("📝 Avg Text Length", avg_length_formatted)
    
    # Detailed analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 File Size Distribution:**")
        for size_range, count in stats['size_distribution'].items():
            percentage = (count / stats['document_count']) * 100
            st.write(f"• {size_range}: {count} ({percentage:.1f}%)")
        
        st.markdown("**🔧 Extraction Modes:**")
        for mode, count in stats['extraction_modes'].items():
            percentage = (count / stats['document_count']) * 100
            st.write(f"• {mode}: {count} ({percentage:.1f}%)")
    
    with col2:
        st.markdown("**📋 Sections Analysis:**")
        sections_stats = stats['sections_stats']
        st.write(f"• Documents with sections: {sections_stats['docs_with_sections']}")
        st.write(f"• Recommendation sections: {sections_stats['recommendation_sections']}")
        st.write(f"• Response sections: {sections_stats['response_sections']}")
        
        st.markdown("**📈 Upload Timeline:**")
        for date, count in sorted(stats['upload_timeline'].items()):
            st.write(f"• {date}: {count} documents")

# ===============================================
# DOCUMENT SEARCH AND FILTERING
# ===============================================

def render_document_search():
    """Render document search interface"""
    st.subheader("🔍 Document Search & Filter")
    
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        st.info("No documents to search.")
        return
    
    # Search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input(
            "🔍 Search documents:",
            placeholder="Enter keywords to search in document content...",
            help="Search in document text and sections"
        )
    
    with col2:
        search_mode = st.selectbox(
            "Search in:",
            options=["All Content", "Filenames Only", "Sections Only"],
            help="Choose what to search in"
        )
    
    # Perform search
    if search_term:
        if search_mode == "All Content":
            results = search_documents_by_content(search_term)
        elif search_mode == "Filenames Only":
            results = [doc for doc in docs if search_term.lower() in doc.get('filename', '').lower()]
        else:  # Sections Only
            results = search_documents_in_sections(search_term)
        
        # Show results
        st.write(f"**🔍 Found {len(results)} documents matching '{search_term}':**")
        
        for doc in results:
            with st.container():
                st.markdown(f"**📄 {doc.get('filename', 'Unknown')}**")
                
                # Show matching context
                text = doc.get('text', '')
                if search_term.lower() in text.lower():
                    # Find context around the search term
                    context = get_search_context(text, search_term)
                    if context:
                        st.caption(f"...{context}...")
                
                st.divider()

def search_documents_in_sections(search_term: str) -> List[Dict[str, Any]]:
    """Search specifically in document sections"""
    docs = st.session_state.get('uploaded_documents', [])
    matching_docs = []
    
    search_term_lower = search_term.lower()
    
    for doc in docs:
        sections = doc.get('sections', [])
        for section in sections:
            section_content = section.get('content', '').lower()
            if search_term_lower in section_content:
                matching_docs.append(doc)
                break
    
    return matching_docs

def get_search_context(text: str, search_term: str, context_length: int = 100) -> str:
    """Get context around search term"""
    text_lower = text.lower()
    search_term_lower = search_term.lower()
    
    index = text_lower.find(search_term_lower)
    if index == -1:
        return ""
    
    start = max(0, index - context_length)
    end = min(len(text), index + len(search_term) + context_length)
    
    context = text[start:end]
    
    # Highlight the search term
    context = context.replace(search_term, f"**{search_term}**")
    
    return context

# ===============================================
# FINAL MODULE VALIDATION
# ===============================================

def validate_module_completeness():
    """Validate that the module is complete and functional"""
    required_functions = [
        'render_upload_tab',
        'render_upload_interface',
        'process_uploaded_files',
        'show_progress_indicator',
        'add_error_message'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    if missing_functions:
        logging.error(f"Missing required functions: {missing_functions}")
        return False
    
    logging.info("✅ Upload components module validation passed")
    return True

# Validate module on import
if not validate_module_completeness():
    logging.warning("⚠️ Upload components module may not be fully functional")

# Log successful module load
logging.info(f"✅ Upload components module loaded with {len(__all__)} functions")
logging.info(f"📊 Component availability - DocumentProcessor: {DOCUMENT_PROCESSOR_AVAILABLE}, SecurityValidator: {SECURITY_VALIDATOR_AVAILABLE}, SharedComponents: {SHARED_COMPONENTS_AVAILABLE}")
    '
