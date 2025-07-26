# ===============================================
# UPLOAD COMPONENTS MODULE - COMPLETE IMPLEMENTATION
# Fixed Import Issues & Robust Error Handling
# 
# This module provides comprehensive document upload functionality
# for the Recommendation-Response Tracker system.
#
# Key Features:
# - Intelligent import fallbacks for DocumentProcessor
# - Robust PDF text extraction with multiple library support
# - Comprehensive error handling and user feedback
# - Batch processing capabilities
# - Document library management
# - Export functionality for metadata and results
#
# Dependencies (with fallbacks):
# - Primary: modules.document_processor.DocumentProcessor
# - Fallback: pdfplumber for PDF processing
# - Final fallback: PyMuPDF (fitz) for PDF processing
# - Core utilities with mock implementations if unavailable
#
# Author: Recommendation-Response Tracker Team
# Version: 2.0 - Complete with fallbacks
# Last Updated: 2025
# ===============================================

import streamlit as st
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import os
import tempfile

# Configure logging first
logging.basicConfig(level=logging.INFO)

# Import core modules with better error handling
DOCUMENT_PROCESSOR_AVAILABLE = False
SECURITY_VALIDATOR_AVAILABLE = False
CORE_UTILS_AVAILABLE = False

# Try importing DocumentProcessor with multiple fallback paths
try:
    # Try direct import first
    from modules.document_processor import DocumentProcessor
    DOCUMENT_PROCESSOR_AVAILABLE = True
    logging.info("✅ DocumentProcessor imported successfully")
except ImportError:
    try:
        # Try alternative import path
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from modules.document_processor import DocumentProcessor
        DOCUMENT_PROCESSOR_AVAILABLE = True
        logging.info("✅ DocumentProcessor imported via alternative path")
    except ImportError:
        try:
            # Try relative import
            from .document_processor import DocumentProcessor
            DOCUMENT_PROCESSOR_AVAILABLE = True
            logging.info("✅ DocumentProcessor imported via relative path")
        except ImportError:
            logging.error("❌ DocumentProcessor not available - will use fallback")
            # Create fallback DocumentProcessor
            class DocumentProcessor:
                def __init__(self):
                    self.logger = logging.getLogger(__name__)
                
                def extract_text_from_pdf(self, pdf_path: str, **kwargs) -> Dict[str, Any]:
                    """Fallback extraction method"""
                    try:
                        # Try basic text extraction with pdfplumber
                        try:
                            import pdfplumber
                            with pdfplumber.open(pdf_path) as pdf:
                                text = ""
                                for page in pdf.pages:
                                    page_text = page.extract_text() or ""
                                    text += page_text + "\n"
                                
                                return {
                                    'success': True,
                                    'filename': Path(pdf_path).name,
                                    'text': text,
                                    'content': text,
                                    'sections': [],
                                    'recommendations': [],
                                    'responses': [],
                                    'metadata': {
                                        'total_pages': len(pdf.pages),
                                        'extraction_method': 'fallback_pdfplumber',
                                        'text_length': len(text)
                                    },
                                    'processed_at': datetime.now().isoformat(),
                                    'extractor_version': 'Fallback_v1.0'
                                }
                        except ImportError:
                            # Try PyMuPDF as final fallback
                            try:
                                import fitz
                                doc = fitz.open(pdf_path)
                                text = ""
                                for page_num in range(len(doc)):
                                    page = doc[page_num]
                                    text += page.get_text() + "\n"
                                doc.close()
                                
                                return {
                                    'success': True,
                                    'filename': Path(pdf_path).name,
                                    'text': text,
                                    'content': text,
                                    'sections': [],
                                    'recommendations': [],
                                    'responses': [],
                                    'metadata': {
                                        'total_pages': len(doc),
                                        'extraction_method': 'fallback_pymupdf',
                                        'text_length': len(text)
                                    },
                                    'processed_at': datetime.now().isoformat(),
                                    'extractor_version': 'Fallback_v1.0'
                                }
                            except ImportError:
                                return {
                                    'success': False,
                                    'error': 'No PDF processing libraries available',
                                    'filename': Path(pdf_path).name,
                                    'text': '',
                                    'content': '',
                                    'sections': [],
                                    'recommendations': [],
                                    'responses': [],
                                    'metadata': {'error': 'No PDF libraries'},
                                    'processed_at': datetime.now().isoformat(),
                                    'extractor_version': 'Fallback_v1.0'
                                }
                    except Exception as e:
                        return {
                            'success': False,
                            'error': str(e),
                            'filename': Path(pdf_path).name,
                            'text': '',
                            'content': '',
                            'sections': [],
                            'recommendations': [],
                            'responses': [],
                            'metadata': {'error': str(e)},
                            'processed_at': datetime.now().isoformat(),
                            'extractor_version': 'Fallback_v1.0'
                        }

# Try importing other utilities
try:
    from modules.core_utils import (
        SecurityValidator, 
        log_user_action, 
        extract_government_document_metadata,
        detect_inquiry_document_structure
    )
    SECURITY_VALIDATOR_AVAILABLE = True
    CORE_UTILS_AVAILABLE = True
    logging.info("✅ Core utilities imported successfully")
except ImportError as import_error:
    logging.warning(f"⚠️ Core utilities not available: {import_error}")
    
    # Fallback implementations
    class SecurityValidator:
        @staticmethod
        def validate_text_input(text, max_length=10000):
            return str(text)[:max_length] if text else ""
    
    def log_user_action(action, details):
        logging.info(f"User action: {action} - {details}")
    
    def extract_government_document_metadata(content):
        return {'document_type': 'unknown', 'extraction_method': 'fallback'}
    
    def detect_inquiry_document_structure(content):
        return {
            'document_structure': 'unknown', 
            'has_recommendations': 'recommendation' in content.lower() if content else False,
            'has_responses': 'response' in content.lower() if content else False
        }

# ===============================================
# MAIN UPLOAD TAB FUNCTION
# ===============================================

def render_upload_tab():
    """Render the main upload tab for document management"""
    st.header("📁 Document Upload & Management")
    
    st.markdown("""
    Upload PDF documents containing recommendations and responses. The system will automatically 
    process and extract only the relevant sections for analysis.
    """)
    
    # Show component availability status
    col1, col2 = st.columns(2)
    with col1:
        if DOCUMENT_PROCESSOR_AVAILABLE:
            st.success("✅ DocumentProcessor available")
        else:
            st.warning("⚠️ DocumentProcessor using fallback mode")
    
    with col2:
        if CORE_UTILS_AVAILABLE:
            st.success("✅ Core utilities available")
        else:
            st.info("ℹ️ Core utilities using fallback mode")
    
    # Main interface components
    render_upload_interface()
    render_document_library()
    render_batch_operations()

# ===============================================
# UPLOAD INTERFACE - FIXED
# ===============================================

def render_upload_interface():
    """Render the simplified file upload interface"""
    st.subheader("📤 Upload New Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents containing recommendations and responses. The system will automatically extract relevant sections."
    )
    
    # Processing options in an expander
    with st.expander("🔧 Processing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_file_size = st.selectbox(
                "Max file size (MB):",
                options=[10, 25, 50, 100, 200],
                index=2,
                help="Maximum file size to process"
            )
        
        with col2:
            batch_processing = st.checkbox(
                "Batch processing",
                value=True,
                help="Process multiple files together for better performance"
            )
    
    # Process uploaded files
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary"):
            process_uploaded_files(uploaded_files, max_file_size, batch_processing)

# ===============================================
# FIXED FILE PROCESSING LOGIC
# ===============================================

def process_uploaded_files(uploaded_files: List, max_file_size: int, batch_processing: bool):
    """Process uploaded PDF files with robust error handling"""
    
    log_user_action("file_upload_started", f"Files: {len(uploaded_files)}, Batch: {batch_processing}")
    
    # Initialize processor with error handling
    try:
        processor = DocumentProcessor()
        processor_status = "✅ Ready"
    except Exception as e:
        logging.error(f"Error initializing DocumentProcessor: {e}")
        processor = DocumentProcessor()  # Fallback will be used
        processor_status = "⚠️ Using fallback mode"
    
    # Initialize session state
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    successful_uploads = 0
    failed_uploads = 0
    processing_results = []
    
    total_files = len(uploaded_files)
    
    # Create temp directory
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    status_text.text(f"Processor status: {processor_status}")
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files}) - {processor_status}")
        
        temp_path = None
        
        try:
            # Validate file size
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > max_file_size:
                raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_file_size}MB)")
            
            # Save uploaded file temporarily with better error handling
            temp_path = save_uploaded_file_temporarily(uploaded_file)
            
            if not temp_path or not temp_path.exists():
                raise ValueError("Failed to save uploaded file")
            
            # Process with intelligent extraction
            extraction_result = processor.extract_text_from_pdf(str(temp_path))
            
            if extraction_result and extraction_result.get('success'):
                # Prepare document data for session state
                doc_data = {
                    'filename': uploaded_file.name,
                    'file_size_mb': file_size_mb,
                    'upload_timestamp': datetime.now().isoformat(),
                    'extraction_result': extraction_result,
                    'processing_mode': 'intelligent_sections',
                    'temp_path': str(temp_path),
                    'status': 'processed',
                    'metadata': extraction_result.get('metadata', {}),
                    'sections_count': len(extraction_result.get('sections', [])),
                    'recommendations_count': len(extraction_result.get('recommendations', [])),
                    'responses_count': len(extraction_result.get('responses', [])),
                    'processor_status': processor_status,
                    'text_length': len(extraction_result.get('text', '')),
                    'extraction_method': extraction_result.get('metadata', {}).get('extraction_method', 'unknown')
                }
                
                # Add to session state
                st.session_state.uploaded_documents.append(doc_data)
                successful_uploads += 1
                
                processing_results.append({
                    'filename': uploaded_file.name,
                    'status': '✅ Success',
                    'size_mb': f"{file_size_mb:.1f}",
                    'sections': len(extraction_result.get('sections', [])),
                    'recommendations': len(extraction_result.get('recommendations', [])),
                    'responses': len(extraction_result.get('responses', [])),
                    'text_length': f"{len(extraction_result.get('text', '')):,}",
                    'method': extraction_result.get('metadata', {}).get('extraction_method', 'unknown')
                })
                
            else:
                error_msg = extraction_result.get('error', 'Unknown processing error') if extraction_result else 'Failed to extract text'
                failed_uploads += 1
                processing_results.append({
                    'filename': uploaded_file.name,
                    'status': f'❌ Failed: {error_msg}',
                    'size_mb': f"{file_size_mb:.1f}",
                    'sections': 0,
                    'recommendations': 0,
                    'responses': 0,
                    'text_length': '0',
                    'method': 'failed'
                })
                
        except Exception as e:
            failed_uploads += 1
            error_msg = str(e)
            processing_results.append({
                'filename': uploaded_file.name,
                'status': f'❌ Error: {error_msg}',
                'size_mb': 'N/A',
                'sections': 0,
                'recommendations': 0,
                'responses': 0,
                'text_length': '0',
                'method': 'error'
            })
            logging.error(f"Error processing {uploaded_file.name}: {e}")
        
        finally:
            # Clean up temp file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logging.warning(f"Could not clean up temp file {temp_path}: {cleanup_error}")
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Display results
    with results_container:
        st.subheader("📊 Processing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("Successful", successful_uploads, delta=successful_uploads)
        with col3:
            st.metric("Failed", failed_uploads, delta=-failed_uploads if failed_uploads > 0 else 0)
        with col4:
            success_rate = (successful_uploads / total_files * 100) if total_files > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Detailed results table
        if processing_results:
            results_df = pd.DataFrame(processing_results)
            st.dataframe(results_df, use_container_width=True)
        
        # Success message
        if successful_uploads > 0:
            st.success(f"✅ Successfully processed {successful_uploads} document(s)! You can now proceed to the Extract tab.")
            log_user_action("file_upload_completed", f"Success: {successful_uploads}, Failed: {failed_uploads}")
        
        # Troubleshooting info
        if failed_uploads > 0:
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Common issues:**
                - **DocumentProcessor not available**: Check if required libraries (pdfplumber, PyMuPDF) are installed
                - **Large file size**: Try reducing file size or increasing the limit
                - **Corrupted PDF**: Try with a different PDF file
                - **Memory issues**: Try processing files one at a time
                
                **Current system status:**
                - DocumentProcessor: {'Available' if DOCUMENT_PROCESSOR_AVAILABLE else 'Fallback mode'}
                - Core utilities: {'Available' if CORE_UTILS_AVAILABLE else 'Fallback mode'}
                """)

def save_uploaded_file_temporarily(uploaded_file) -> Optional[Path]:
    """Save uploaded file to temporary location with better error handling"""
    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Create unique filename to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in '._-')
        temp_path = temp_dir / f"{timestamp}_{safe_filename}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        return temp_path
        
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        return None

# ===============================================
# DOCUMENT LIBRARY - ENHANCED
# ===============================================

def render_document_library():
    """Render the document library showing uploaded documents"""
    st.subheader("📚 Document Library")
    
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    
    if not uploaded_docs:
        st.info("📝 No documents uploaded yet. Upload some PDFs above to get started!")
        return
    
    # Create library display
    library_data = []
    for i, doc in enumerate(uploaded_docs):
        library_data.append({
            'Index': i,
            'Filename': doc.get('filename', 'Unknown'),
            'Size (MB)': f"{doc.get('file_size_mb', 0):.1f}",
            'Upload Time': doc.get('upload_timestamp', '').split('T')[0] if doc.get('upload_timestamp') else 'Unknown',
            'Status': '✅ Processed' if doc.get('status') == 'processed' else '⚠️ Pending',
            'Sections': doc.get('sections_count', 0),
            'Recommendations': doc.get('recommendations_count', 0),
            'Responses': doc.get('responses_count', 0),
            'Text Length': f"{doc.get('text_length', 0):,}",
            'Method': doc.get('extraction_method', 'unknown')
        })
    
    # Display as dataframe
    library_df = pd.DataFrame(library_data)
    st.dataframe(library_df, use_container_width=True)
    
    # Document actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Library"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Documents"):
            if st.session_state.get('confirm_clear_all'):
                st.session_state.uploaded_documents = []
                st.session_state.confirm_clear_all = False
                st.success("All documents cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear_all = True
                st.warning("Click again to confirm clearing all documents")
    
    with col3:
        if len(uploaded_docs) > 0:
            selected_doc_index = st.selectbox(
                "Select document to view details:",
                options=range(len(uploaded_docs)),
                format_func=lambda x: uploaded_docs[x].get('filename', f'Document {x}')
            )
            
            if st.button("👁️ View Details"):
                show_document_details(uploaded_docs[selected_doc_index])

def show_document_details(doc_data: Dict[str, Any]):
    """Show detailed information about a document"""
    st.subheader(f"📄 Document Details: {doc_data.get('filename', 'Unknown')}")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**File Information:**")
        st.write(f"• Filename: {doc_data.get('filename', 'Unknown')}")
        st.write(f"• Size: {doc_data.get('file_size_mb', 0):.1f} MB")
        st.write(f"• Upload Time: {doc_data.get('upload_timestamp', 'Unknown')}")
        st.write(f"• Processing Mode: {doc_data.get('processing_mode', 'Unknown')}")
        st.write(f"• Processor Status: {doc_data.get('processor_status', 'Unknown')}")
        st.write(f"• Text Length: {doc_data.get('text_length', 0):,} characters")
    
    with col2:
        st.write("**Extraction Results:**")
        st.write(f"• Status: {doc_data.get('status', 'Unknown')}")
        st.write(f"• Sections Found: {doc_data.get('sections_count', 0)}")
        st.write(f"• Recommendations: {doc_data.get('recommendations_count', 0)}")
        st.write(f"• Responses: {doc_data.get('responses_count', 0)}")
        st.write(f"• Extraction Method: {doc_data.get('extraction_method', 'Unknown')}")
    
    # Metadata
    if doc_data.get('metadata'):
        with st.expander("🔍 Document Metadata"):
            metadata = doc_data['metadata']
            for key, value in metadata.items():
                st.write(f"**{key}:** {value}")
    
    # Show text preview if available
    extraction_result = doc_data.get('extraction_result', {})
    text_content = extraction_result.get('text', '')
    if text_content:
        with st.expander("📄 Text Preview (First 1000 characters)"):
            st.text(text_content[:1000] + "..." if len(text_content) > 1000 else text_content)

# ===============================================
# BATCH OPERATIONS
# ===============================================

def render_batch_operations():
    """Render batch operations for multiple documents"""
    st.subheader("⚡ Batch Operations")
    
    uploaded_docs = st.session_state.get('uploaded_documents', [])
    
    if len(uploaded_docs) < 2:
        st.info("📝 Upload at least 2 documents to use batch operations.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Operations:**")
        
        if st.button("📊 Generate Summary Report"):
            generate_batch_summary_report(uploaded_docs)
        
        if st.button("📥 Export All Metadata"):
            export_batch_metadata(uploaded_docs)
    
    with col2:
        st.write("**Batch Statistics:**")
        total_sections = sum(doc.get('sections_count', 0) for doc in uploaded_docs)
        total_recommendations = sum(doc.get('recommendations_count', 0) for doc in uploaded_docs)
        total_responses = sum(doc.get('responses_count', 0) for doc in uploaded_docs)
        total_text_length = sum(doc.get('text_length', 0) for doc in uploaded_docs)
        
        st.metric("Total Documents", len(uploaded_docs))
        st.metric("Total Sections", total_sections)
        st.metric("Total Recommendations", total_recommendations)
        st.metric("Total Responses", total_responses)
        st.metric("Total Text Length", f"{total_text_length:,}")

def generate_batch_summary_report(uploaded_docs: List[Dict[str, Any]]):
    """Generate a summary report for all uploaded documents"""
    st.subheader("📈 Batch Summary Report")
    
    # Calculate summary statistics
    total_docs = len(uploaded_docs)
    total_size_mb = sum(doc.get('file_size_mb', 0) for doc in uploaded_docs)
    total_sections = sum(doc.get('sections_count', 0) for doc in uploaded_docs)
    total_recommendations = sum(doc.get('recommendations_count', 0) for doc in uploaded_docs)
    total_responses = sum(doc.get('responses_count', 0) for doc in uploaded_docs)
    total_text_length = sum(doc.get('text_length', 0) for doc in uploaded_docs)
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents", total_docs)
        st.metric("Total Size", f"{total_size_mb:.1f} MB")
    
    with col2:
        st.metric("Sections", total_sections)
        st.metric("Avg per Doc", f"{total_sections/total_docs:.1f}")
    
    with col3:
        st.metric("Recommendations", total_recommendations)
        st.metric("Responses", total_responses)
    
    # Document breakdown
    breakdown_data = []
    for doc in uploaded_docs:
        breakdown_data.append({
            'Document': doc.get('filename', 'Unknown'),
            'Size (MB)': f"{doc.get('file_size_mb', 0):.1f}",
            'Sections': doc.get('sections_count', 0),
            'Recommendations': doc.get('recommendations_count', 0),
            'Responses': doc.get('responses_count', 0),
            'Text Length': f"{doc.get('text_length', 0):,}",
            'Upload Date': doc.get('upload_timestamp', '').split('T')[0] if doc.get('upload_timestamp') else 'Unknown',
            'Method': doc.get('extraction_method', 'unknown')
        })
    
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True)

def export_batch_metadata(uploaded_docs: List[Dict[str, Any]]):
    """Export metadata for all documents"""
    export_data = []
    
    for doc in uploaded_docs:
        export_data.append({
            'filename': doc.get('filename', 'Unknown'),
            'file_size_mb': doc.get('file_size_mb', 0),
            'upload_timestamp': doc.get('upload_timestamp', ''),
            'processing_mode': doc.get('processing_mode', 'Unknown'),
            'status': doc.get('status', 'Unknown'),
            'sections_count': doc.get('sections_count', 0),
            'recommendations_count': doc.get('recommendations_count', 0),
            'responses_count': doc.get('responses_count', 0),
            'text_length': doc.get('text_length', 0),
            'extraction_method': doc.get('extraction_method', 'unknown'),
            'processor_status': doc.get('processor_status', 'unknown')
        })
    
    export_df = pd.DataFrame(export_data)
    
    # Convert to CSV for download
    csv_data = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Metadata CSV",
        data=csv_data,
        file_name=f"document_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ Metadata export ready for download!")

# ===============================================
# MAIN EXPORT
# ===============================================

__all__ = [
    'render_upload_tab',
    'render_upload_interface', 
    'render_document_library',
    'render_batch_operations',
    'process_uploaded_files',
    'save_uploaded_file_temporarily',
    'show_document_details',
    'generate_batch_summary_report',
    'export_batch_metadata'
]
