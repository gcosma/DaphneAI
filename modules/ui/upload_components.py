# ===============================================
# FILE: modules/ui/upload_components.py
# Upload Components for DaphneAI Document Analysis
# ===============================================

import streamlit as st
import pandas as pd
import logging
import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document class fallback
try:
    from langchain.schema import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        class Document:
            def __init__(self, page_content: str, metadata: Dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

# PDF processing libraries with fallbacks
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Core utilities
try:
    from modules.core_utils import SecurityValidator, log_user_action
    CORE_UTILS_AVAILABLE = True
except ImportError:
    CORE_UTILS_AVAILABLE = False
    class SecurityValidator:
        def validate_file_upload(self, content: bytes, filename: str) -> Tuple[bool, str]:
            max_size = 200 * 1024 * 1024  # 200MB
            if len(content) > max_size:
                return False, f"File too large: {len(content)/1024/1024:.1f}MB > 200MB"
            return True, "Valid file"
    
    def log_user_action(action: str, data: Dict = None):
        logger.info(f"Action: {action}, Data: {data}")

# ===============================================
# DOCUMENT PROCESSING FUNCTIONS
# ===============================================

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF using available libraries"""
    try:
        # Try pdfplumber first
        if PDFPLUMBER_AVAILABLE:
            pdf_content = io.BytesIO(uploaded_file.read())
            with pdfplumber.open(pdf_content) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        
        # Fallback to PyMuPDF
        elif PYMUPDF_AVAILABLE:
            uploaded_file.seek(0)
            pdf_content = uploaded_file.read()
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text() + "\n"
            pdf_document.close()
            return text
        
        else:
            return "PDF processing libraries not available. Please install pdfplumber or PyMuPDF."
            
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return f"Error extracting PDF: {str(e)}"

def process_uploaded_documents(uploaded_files) -> List[Dict]:
    """Process uploaded files and create document objects"""
    documents = []
    security_validator = SecurityValidator()
    
    for uploaded_file in uploaded_files:
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            
            # Security validation
            is_valid, message = security_validator.validate_file_upload(file_content, uploaded_file.name)
            
            if not is_valid:
                st.error(f"❌ {uploaded_file.name}: {message}")
                continue
            
            # Extract text based on file type
            uploaded_file.seek(0)
            
            if uploaded_file.name.lower().endswith('.pdf'):
                text_content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.lower().endswith(('.txt', '.md')):
                text_content = str(file_content, 'utf-8', errors='ignore')
            else:
                text_content = f"Unsupported file type: {uploaded_file.name}"
            
            # Create document metadata
            doc_metadata = {
                'filename': uploaded_file.name,
                'file_size': len(file_content),
                'file_type': uploaded_file.type or 'unknown',
                'upload_timestamp': datetime.now().isoformat(),
                'text_length': len(text_content),
                'processed': True
            }
            
            # Create document object
            document = {
                'filename': uploaded_file.name,
                'content': text_content,
                'metadata': doc_metadata,
                'doc_object': Document(page_content=text_content, metadata=doc_metadata)
            }
            
            documents.append(document)
            logger.info(f"✅ Processed: {uploaded_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    return documents

# ===============================================
# MAIN UPLOAD TAB COMPONENT
# ===============================================

def render_upload_tab():
    """Render the document upload interface"""
    st.header("📁 Document Upload")
    st.markdown("""
    Upload government inquiry reports, response documents, or related materials for analysis.
    Supported formats: PDF, TXT, MD
    """)
    
    # File upload interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'md', 'docx'],
            accept_multiple_files=True,
            help="Upload multiple documents for batch processing"
        )
    
    with col2:
        st.markdown("### 📊 Upload Status")
        current_docs = len(st.session_state.get('uploaded_documents', []))
        st.metric("Documents", current_docs)
        
        if st.button("🗑️ Clear All", help="Remove all uploaded documents"):
            st.session_state.uploaded_documents = []
            st.success("✅ All documents cleared")
            st.rerun()
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            processed_docs = process_uploaded_documents(uploaded_files)
            
            if processed_docs:
                # Update session state
                if 'uploaded_documents' not in st.session_state:
                    st.session_state.uploaded_documents = []
                
                # Add new documents (avoid duplicates)
                existing_filenames = {doc['filename'] for doc in st.session_state.uploaded_documents}
                new_docs = [doc for doc in processed_docs if doc['filename'] not in existing_filenames]
                
                st.session_state.uploaded_documents.extend(new_docs)
                
                # Success message
                st.success(f"✅ Successfully processed {len(new_docs)} new documents")
                
                # Log action
                if CORE_UTILS_AVAILABLE:
                    log_user_action("documents_uploaded", {
                        'count': len(new_docs),
                        'filenames': [doc['filename'] for doc in new_docs]
                    })
    
    # Display uploaded documents
    if st.session_state.get('uploaded_documents'):
        st.markdown("### 📚 Uploaded Documents")
        
        documents_data = []
        for doc in st.session_state.uploaded_documents:
            documents_data.append({
                'Filename': doc['filename'],
                'Size (KB)': f"{doc['metadata']['file_size'] / 1024:.1f}",
                'Type': doc['metadata']['file_type'],
                'Text Length': f"{doc['metadata']['text_length']:,} chars",
                'Uploaded': doc['metadata']['upload_timestamp'][:19].replace('T', ' ')
            })
        
        df = pd.DataFrame(documents_data)
        st.dataframe(df, use_container_width=True)
        
        # Document actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Preview Documents"):
                st.session_state.show_preview = True
        
        with col2:
            if st.button("📊 Document Statistics"):
                show_document_statistics()
        
        with col3:
            if st.button("📥 Export Document List"):
                export_document_list(documents_data)
    
    else:
        st.info("📤 No documents uploaded yet. Use the file uploader above to get started.")
    
    # Document preview
    if st.session_state.get('show_preview', False):
        show_document_preview()

def show_document_statistics():
    """Display statistics about uploaded documents"""
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        st.warning("No documents to analyze")
        return
    
    st.markdown("### 📊 Document Statistics")
    
    # Calculate statistics
    total_docs = len(docs)
    total_size = sum(doc['metadata']['file_size'] for doc in docs)
    total_text = sum(doc['metadata']['text_length'] for doc in docs)
    avg_text_length = total_text / total_docs if total_docs > 0 else 0
    
    # File type distribution
    file_types = {}
    for doc in docs:
        file_type = doc['metadata']['file_type']
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", total_docs)
    
    with col2:
        st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
    
    with col3:
        st.metric("Total Text", f"{total_text:,} chars")
    
    with col4:
        st.metric("Avg Text Length", f"{avg_text_length:,.0f} chars")
    
    # File type breakdown
    st.markdown("#### File Type Distribution")
    for file_type, count in file_types.items():
        st.write(f"**{file_type}**: {count} files")

def show_document_preview():
    """Show preview of uploaded documents"""
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        st.warning("No documents to preview")
        return
    
    st.markdown("### 🔍 Document Preview")
    
    # Document selector
    doc_names = [doc['filename'] for doc in docs]
    selected_doc_name = st.selectbox("Select document to preview:", doc_names)
    
    # Find selected document
    selected_doc = next((doc for doc in docs if doc['filename'] == selected_doc_name), None)
    
    if selected_doc:
        # Document metadata
        st.markdown(f"**Filename:** {selected_doc['filename']}")
        st.markdown(f"**Size:** {selected_doc['metadata']['file_size']} bytes")
        st.markdown(f"**Text Length:** {selected_doc['metadata']['text_length']} characters")
        
        # Content preview
        content = selected_doc['content']
        preview_length = min(2000, len(content))
        
        st.markdown("#### Content Preview (first 2000 characters):")
        st.text_area(
            "Document Content",
            content[:preview_length],
            height=300,
            disabled=True
        )
        
        if len(content) > preview_length:
            st.info(f"Showing first {preview_length} of {len(content)} characters")
    
    # Close preview button
    if st.button("❌ Close Preview"):
        st.session_state.show_preview = False
        st.rerun()

def export_document_list(documents_data):
    """Export document list as CSV"""
    try:
        df = pd.DataFrame(documents_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"document_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Document list ready for download")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

# ===============================================
# INITIALIZATION
# ===============================================

# Initialize session state for upload tab
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False

logger.info("✅ Upload components initialized")
