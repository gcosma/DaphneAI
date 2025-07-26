# ===============================================
# COMPLETE modules/ui/upload_components.py
# Document upload and management functionality
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import io
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import PDF processing with fallbacks
try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except ImportError:
    PDF_PLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available")

class DocumentProcessor:
    """Handle document upload and processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_types = ['pdf']
        self.max_file_size = 200 * 1024 * 1024  # 200MB
    
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """Process a single uploaded file"""
        try:
            # Basic file info
            file_info = {
                'filename': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type,
                'uploaded_at': datetime.now().isoformat(),
                'processing_status': 'processing'
            }
            
            # Check file size
            if uploaded_file.size > self.max_file_size:
                file_info['error'] = f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Max size is 200MB."
                file_info['processing_status'] = 'error'
                return file_info
            
            # Generate file hash for deduplication
            file_content = uploaded_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            file_info['file_hash'] = file_hash
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Extract text content
            if uploaded_file.type == 'application/pdf':
                extraction_result = self._extract_pdf_content(uploaded_file)
                file_info['extraction_result'] = extraction_result
            else:
                file_info['error'] = f"Unsupported file type: {uploaded_file.type}"
                file_info['processing_status'] = 'error'
                return file_info
            
            # Success
            file_info['processing_status'] = 'completed'
            return file_info
            
        except Exception as e:
            self.logger.error(f"Error processing file {uploaded_file.name}: {e}")
            file_info['error'] = str(e)
            file_info['processing_status'] = 'error'
            return file_info
    
    def _extract_pdf_content(self, pdf_file) -> Dict[str, Any]:
        """Extract content from PDF file"""
        extraction_result = {
            'text': '',
            'page_count': 0,
            'metadata': {},
            'extraction_method': 'none',
            'success': False
        }
        
        # Try pdfplumber first
        if PDF_PLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            pages_text.append(text)
                    
                    extraction_result.update({
                        'text': '\n'.join(pages_text),
                        'page_count': len(pdf.pages),
                        'metadata': pdf.metadata or {},
                        'extraction_method': 'pdfplumber',
                        'success': True
                    })
                    return extraction_result
            except Exception as e:
                self.logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Try PyMuPDF as fallback
        if PYMUPDF_AVAILABLE:
            try:
                pdf_file.seek(0)
                pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
                pages_text = []
                
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    if text:
                        pages_text.append(text)
                
                extraction_result.update({
                    'text': '\n'.join(pages_text),
                    'page_count': pdf_document.page_count,
                    'metadata': pdf_document.metadata,
                    'extraction_method': 'pymupdf',
                    'success': True
                })
                pdf_document.close()
                return extraction_result
                
            except Exception as e:
                self.logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # If both methods fail
        extraction_result['error'] = "Could not extract text from PDF"
        return extraction_result

def render_upload_tab():
    """Main upload tab interface"""
    st.header("📁 Document Upload & Management")
    
    st.markdown("""
    Upload PDF documents containing recommendations and responses. The system will automatically 
    process and extract only the relevant sections for analysis.
    """)
    
    # Status indicators
    col1, col2 = st.columns(2)
    
    with col1:
        if PDF_PLUMBER_AVAILABLE:
            st.success("✅ DocumentProcessor available")
        else:
            st.warning("⚠️ Limited PDF processing available")
    
    with col2:
        if st.session_state.get('uploaded_documents'):
            st.info(f"📚 Core utilities using fallback mode")
        else:
            st.info("🔧 Core utilities using fallback mode")
    
    # Document upload section
    render_upload_interface()
    
    # Document library
    render_document_library()
    
    # Batch operations
    render_batch_operations()

def render_upload_interface():
    """File upload interface"""
    st.subheader("☁️ Upload New Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload government inquiry reports, response documents, or related PDFs. Maximum 200MB per file.",
        key="file_uploader"
    )
    
    # Processing options
    with st.expander("⚙️ Processing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            extract_text = st.checkbox("Extract full text", value=True, 
                                     help="Extract all text content from PDFs")
            validate_documents = st.checkbox("Validate document structure", value=True,
                                           help="Check if documents contain recommendations or responses")
        
        with col2:
            remove_duplicates = st.checkbox("Remove duplicates", value=True,
                                          help="Skip files that have already been uploaded")
            auto_categorize = st.checkbox("Auto-categorize documents", value=True,
                                        help="Automatically detect document types")
    
    # Process uploaded files
    if uploaded_files:
        process_uploaded_files(uploaded_files, {
            'extract_text': extract_text,
            'validate_documents': validate_documents,
            'remove_duplicates': remove_duplicates,
            'auto_categorize': auto_categorize
        })

def process_uploaded_files(uploaded_files: List, options: Dict[str, bool]):
    """Process the uploaded files"""
    
    processor = DocumentProcessor()
    
    # Initialize uploaded documents list if not exists
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    new_documents = []
    duplicate_count = 0
    error_count = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Process the file
        file_info = processor.process_uploaded_file(uploaded_file)
        
        # Check for duplicates if enabled
        if options.get('remove_duplicates', True):
            existing_hashes = [doc.get('file_hash') for doc in st.session_state.uploaded_documents]
            if file_info.get('file_hash') in existing_hashes:
                duplicate_count += 1
                continue
        
        # Check for errors
        if file_info.get('processing_status') == 'error':
            error_count += 1
            st.error(f"❌ Error processing {uploaded_file.name}: {file_info.get('error', 'Unknown error')}")
            continue
        
        # Auto-categorize if enabled
        if options.get('auto_categorize', True):
            file_info['document_type'] = categorize_document(file_info)
        
        new_documents.append(file_info)
    
    # Add new documents to session state
    st.session_state.uploaded_documents.extend(new_documents)
    
    # Show results
    status_text.text("✅ Processing completed!")
    progress_bar.progress(1.0)
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ Processed", len(new_documents))
    with col2:
        st.metric("🔄 Duplicates", duplicate_count)
    with col3:
        st.metric("❌ Errors", error_count)
    with col4:
        st.metric("📚 Total", len(st.session_state.uploaded_documents))
    
    if new_documents:
        st.success(f"Successfully uploaded {len(new_documents)} new documents!")
        
        # Show preview of new documents
        with st.expander("📋 Preview New Documents"):
            for doc in new_documents[-3:]:  # Show last 3
                st.write(f"**{doc['filename']}** ({doc['size'] / 1024:.1f} KB)")
                if doc.get('extraction_result', {}).get('text'):
                    preview_text = doc['extraction_result']['text'][:200] + "..."
                    st.text(preview_text)
                st.write("---")

def categorize_document(file_info: Dict[str, Any]) -> str:
    """Auto-categorize document based on content"""
    try:
        text = file_info.get('extraction_result', {}).get('text', '').lower()
        
        if not text:
            return 'unknown'
        
        # Check for government response indicators
        response_indicators = [
            'government response',
            'official response',
            'ministerial response',
            'department response',
            'accepted',
            'rejected',
            'not accepted',
            'partially accepted'
        ]
        
        if any(indicator in text for indicator in response_indicators):
            return 'government_response'
        
        # Check for inquiry report indicators
        inquiry_indicators = [
            'inquiry report',
            'investigation report',
            'final report',
            'recommendations',
            'the inquiry recommends',
            'committee recommends'
        ]
        
        if any(indicator in text for indicator in inquiry_indicators):
            return 'inquiry_report'
        
        # Check for specific document patterns
        if 'recommendation' in text and text.count('recommendation') > 3:
            return 'document_with_recommendations'
        
        return 'government_document'
        
    except Exception as e:
        logging.warning(f"Document categorization failed: {e}")
        return 'unknown'

def render_document_library():
    """Display uploaded documents library"""
    st.subheader("📚 Document Library")
    
    if not st.session_state.get('uploaded_documents'):
        st.info("📄 No documents uploaded yet. Upload some PDFs above to get started!")
        return
    
    documents = st.session_state.uploaded_documents
    
    # Library stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(documents))
    
    with col2:
        total_size = sum(doc.get('size', 0) for doc in documents)
        st.metric("Total Size", f"{total_size / 1024 / 1024:.1f} MB")
    
    with col3:
        successful_extractions = sum(1 for doc in documents 
                                   if doc.get('extraction_result', {}).get('success', False))
        st.metric("Text Extracted", successful_extractions)
    
    with col4:
        doc_types = [doc.get('document_type', 'unknown') for doc in documents]
        unique_types = len(set(doc_types))
        st.metric("Document Types", unique_types)
    
    # Document list
    st.markdown("#### 📋 Document List")
    
    # Create DataFrame for display
    doc_data = []
    for i, doc in enumerate(documents):
        extraction = doc.get('extraction_result', {})
        doc_data.append({
            'Index': i + 1,
            'Filename': doc.get('filename', 'Unknown'),
            'Size (KB)': f"{doc.get('size', 0) / 1024:.1f}",
            'Type': doc.get('document_type', 'unknown').replace('_', ' ').title(),
            'Pages': extraction.get('page_count', 0),
            'Text Extracted': '✅' if extraction.get('success') else '❌',
            'Upload Date': doc.get('uploaded_at', '')[:10] if doc.get('uploaded_at') else 'Unknown'
        })
    
    if doc_data:
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
        
        # Document actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Preview Documents"):
                render_document_preview()
        
        with col2:
            if st.button("📊 Export Document List"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"document_list_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🗑️ Clear All Documents"):
                if st.session_state.get('confirm_clear'):
                    st.session_state.uploaded_documents = []
                    st.success("All documents cleared!")
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm deletion")

def render_document_preview():
    """Show document preview modal"""
    documents = st.session_state.uploaded_documents
    
    if not documents:
        return
    
    # Document selector
    doc_names = [f"{i+1}. {doc.get('filename', 'Unknown')}" for i, doc in enumerate(documents)]
    selected_doc_name = st.selectbox("Select document to preview:", doc_names)
    
    if selected_doc_name:
        doc_index = int(selected_doc_name.split('.')[0]) - 1
        selected_doc = documents[doc_index]
        
        # Document info
        st.write("**Document Information:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Filename:** {selected_doc.get('filename')}")
            st.write(f"**Size:** {selected_doc.get('size', 0) / 1024:.1f} KB")
            st.write(f"**Type:** {selected_doc.get('document_type', 'unknown').replace('_', ' ').title()}")
        
        with col2:
            extraction = selected_doc.get('extraction_result', {})
            st.write(f"**Pages:** {extraction.get('page_count', 0)}")
            st.write(f"**Extraction Method:** {extraction.get('extraction_method', 'none')}")
            st.write(f"**Text Extracted:** {'✅ Yes' if extraction.get('success') else '❌ No'}")
        
        # Text content preview
        if extraction.get('text'):
            st.write("**Text Content Preview:**")
            preview_text = extraction['text'][:1000] + "..." if len(extraction['text']) > 1000 else extraction['text']
            st.text_area("Content", preview_text, height=200, disabled=True)
        else:
            st.warning("No text content available for preview")

def render_batch_operations():
    """Batch operations on uploaded documents"""
    st.subheader("⚡ Batch Operations")
    
    if not st.session_state.get('uploaded_documents'):
        st.info("📄 Upload at least 2 documents to use batch operations.")
        return
    
    documents = st.session_state.uploaded_documents
    
    if len(documents) < 2:
        st.info("📄 Upload at least 2 documents to use batch operations.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Re-process All"):
            st.info("Re-processing functionality would be implemented here")
    
    with col2:
        if st.button("📊 Analyze All"):
            st.info("Batch analysis functionality would be implemented here")
    
    with col3:
        if st.button("📦 Export All"):
            # Create combined export
            all_text = []
            for doc in documents:
                extraction = doc.get('extraction_result', {})
                if extraction.get('text'):
                    all_text.append(f"=== {doc.get('filename')} ===\n{extraction['text']}\n\n")
            
            if all_text:
                combined_text = '\n'.join(all_text)
                st.download_button(
                    label="💾 Download Combined Text",
                    data=combined_text,
                    file_name=f"all_documents_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No text content available for export")

# Export functions
__all__ = [
    'render_upload_tab',
    'DocumentProcessor',
    'categorize_document'
]
