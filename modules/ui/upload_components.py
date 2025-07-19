# ===============================================
# FILE: modules/ui/upload_components.py
# ===============================================

import streamlit as st
import pandas as pd
import tempfile
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Import required modules with error handling
try:
    import sys
    sys.path.append('modules')
    from document_processor import DocumentProcessor
    from core_utils import SecurityValidator
    from .shared_components import add_error_message, show_progress_indicator
except ImportError as e:
    logging.error(f"Import error in upload_components: {e}")
    # Create mock classes for development
    class DocumentProcessor:
        def extract_text_from_pdf(self, path): return None
    class SecurityValidator:
        @staticmethod
        def validate_file_upload(content, filename): return True
        @staticmethod
        def sanitize_filename(filename): return filename

def render_upload_tab():
    """Render the document upload tab"""
    st.header("📁 Document Upload & Management")
    
    st.markdown("""
    Upload PDF documents containing recommendations and responses. The system will automatically 
    process and categorize them for analysis.
    """)
    
    # Upload interface
    render_upload_interface()
    
    # Document management
    render_document_library()
    
    # Batch operations
    render_batch_operations()

def render_upload_interface():
    """Render the file upload interface"""
    st.subheader("📤 Upload New Documents")
    
    # Create upload columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Select one or more PDF documents to upload and process",
            key="document_uploader"
        )
    
    with col2:
        st.markdown("**Upload Guidelines:**")
        st.markdown("• Max file size: 100MB")
        st.markdown("• Supported format: PDF")
        st.markdown("• Text must be readable (not scanned images)")
        st.markdown("• Multiple files supported")
    
    # Process uploaded files
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            process_uploaded_files(uploaded_files)

def process_uploaded_files(uploaded_files: List):
    """Process uploaded PDF files with comprehensive error handling"""
    processor = DocumentProcessor()
    validator = SecurityValidator()
    
    if not uploaded_files:
        st.warning("No files selected for processing.")
        return
    
    # Initialize progress tracking
    total_files = len(uploaded_files)
    progress_container = st.container()
    status_container = st.container()
    
    successful_uploads = 0
    failed_uploads = []
    
    # Process each file
    for i, uploaded_file in enumerate(uploaded_files):
        current_step = i + 1
        
        with progress_container:
            show_progress_indicator(current_step, total_files, f"Processing {uploaded_file.name}")
        
        with status_container:
            status_text = st.empty()
            status_text.info(f"📄 Processing: {uploaded_file.name}")
        
        try:
            # Read file content
            file_content = uploaded_file.read()
            
            # Security validation
            try:
                validator.validate_file_upload(file_content, uploaded_file.name)
            except ValueError as e:
                failed_uploads.append(f"{uploaded_file.name}: {str(e)}")
                status_text.error(f"❌ Security validation failed: {uploaded_file.name}")
                continue
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract text and metadata
                doc_data = processor.extract_text_from_pdf(tmp_file_path)
                
                if doc_data and doc_data.get('content'):
                    # Determine document type
                    doc_type = determine_document_type(doc_data['content'])
                    
                    # Create document info
                    doc_info = {
                        'filename': validator.sanitize_filename(uploaded_file.name),
                        'original_filename': uploaded_file.name,
                        'content': doc_data['content'],
                        'metadata': doc_data.get('metadata', {}),
                        'document_type': doc_type,
                        'upload_time': datetime.now().isoformat(),
                        'file_size': len(file_content),
                        'processing_status': 'completed'
                    }
                    
                    # Check for duplicates
                    existing_names = [doc['filename'] for doc in st.session_state.uploaded_documents]
                    if doc_info['filename'] not in existing_names:
                        st.session_state.uploaded_documents.append(doc_info)
                        successful_uploads += 1
                        status_text.success(f"✅ Successfully processed: {uploaded_file.name}")
                    else:
                        status_text.warning(f"⚠️ Duplicate file skipped: {uploaded_file.name}")
                
                else:
                    failed_uploads.append(f"{uploaded_file.name}: No readable text found")
                    status_text.error(f"❌ No text extracted from: {uploaded_file.name}")
                
            finally:
                # Cleanup temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            
        except Exception as e:
            error_msg = f"{uploaded_file.name}: {str(e)}"
            failed_uploads.append(error_msg)
            add_error_message(f"Failed to process {uploaded_file.name}: {str(e)}")
            status_text.error(f"❌ Processing error: {uploaded_file.name}")
            logging.error(f"File processing error: {e}", exc_info=True)
    
    # Final status update
    progress_container.empty()
    status_container.empty()
    
    # Show results summary
    if successful_uploads > 0:
        st.success(f"✅ Successfully processed {successful_uploads} of {total_files} files!")
    
    if failed_uploads:
        st.error(f"❌ Failed to process {len(failed_uploads)} files:")
        for error in failed_uploads:
            st.write(f"• {error}")
    
    # Trigger rerun to update document library
    if successful_uploads > 0:
        st.rerun()

def determine_document_type(content: str) -> str:
    """Intelligently determine document type based on content analysis"""
    if not content:
        return 'Unknown'
    
    content_lower = content.lower()
    
    # Response indicators (stronger signals)
    response_indicators = [
        'in response to', 'responding to', 'implementation of', 
        'accepted recommendation', 'rejected recommendation', 'under review', 
        'action taken', 'actions completed', 'following the recommendation', 
        'as recommended', 'we have implemented', 'steps taken',
        'progress report', 'status update', 'implementation plan'
    ]
    
    # Recommendation indicators
    recommendation_indicators = [
        'recommendation', 'recommend that', 'should implement',
        'must establish', 'needs to', 'ought to', 'suggests that',
        'proposes that', 'advises that', 'urges that'
    ]
    
    # Concern indicators
    concern_indicators = [
        'concern about', 'worried about', 'issue with', 'problem identified',
        'risk of', 'difficulty in', 'challenge faced', 'failure to'
    ]
    
    # Count occurrences
    response_score = sum(1 for indicator in response_indicators if indicator in content_lower)
    recommendation_score = sum(1 for indicator in recommendation_indicators if indicator in content_lower)
    concern_score = sum(1 for indicator in concern_indicators if indicator in content_lower)
    
    # Determine type based on scores
    if response_score > max(recommendation_score, concern_score):
        return 'Response Document'
    elif recommendation_score > max(response_score, concern_score):
        return 'Recommendation Document'
    elif concern_score > max(response_score, recommendation_score):
        return 'Concern Document'
    elif recommendation_score > 0:
        return 'Mixed Content'
    else:
        return 'General Document'

def render_document_library():
    """Render the document library with management options"""
    if not st.session_state.uploaded_documents:
        st.info("📝 No documents uploaded yet. Upload some PDF files to get started!")
        return
    
    st.subheader("📚 Document Library")
    
    # Create document summary
    docs_data = []
    for i, doc in enumerate(st.session_state.uploaded_documents):
        docs_data.append({
            "Index": i,
            "Filename": doc.get('filename', 'Unknown'),
            "Type": doc.get('document_type', 'Unknown'),
            "Pages": doc.get('metadata', {}).get('page_count', 'N/A'),
            "Size (KB)": round(doc.get('file_size', 0) / 1024, 1),
            "Upload Time": doc.get('upload_time', '')[:19] if doc.get('upload_time') else 'Unknown',
            "Status": "✅ Ready" if doc.get('processing_status') == 'completed' else "⚠️ Processing"
        })
    
    # Display as interactive dataframe
    df = pd.DataFrame(docs_data)
    
    # Document type filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        type_filter = st.selectbox(
            "Filter by Type:",
            options=['All'] + sorted(df['Type'].unique().tolist()),
            key="doc_type_filter"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            options=['Upload Time', 'Filename', 'Size (KB)', 'Type'],
            key="doc_sort_by"
        )
    
    with col3:
        sort_order = st.selectbox(
            "Order:",
            options=['Descending', 'Ascending'],
            key="doc_sort_order"
        )
    
    # Apply filters and sorting
    filtered_df = df.copy()
    
    if type_filter != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == type_filter]
    
    # Sort dataframe
    ascending = sort_order == 'Ascending'
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    # Display filtered dataframe
    if not filtered_df.empty:
        st.dataframe(
            filtered_df.drop('Index', axis=1),  # Hide index column
            use_container_width=True,
            hide_index=True
        )
        
        # Selection for detailed view
        selected_doc_name = st.selectbox(
            "View Details:",
            options=['Select a document...'] + filtered_df['Filename'].tolist(),
            key="selected_doc_detail"
        )
        
        if selected_doc_name != 'Select a document...':
            display_document_details(selected_doc_name)
    
    else:
        st.warning(f"No documents found matching filter: {type_filter}")

def display_document_details(filename: str):
    """Display detailed information for a selected document"""
    # Find the document
    doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == filename), None)
    
    if not doc:
        st.error(f"Document not found: {filename}")
        return
    
    with st.expander(f"📄 Document Details: {filename}", expanded=True):
        # Create detail columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"• **Original Name:** {doc.get('original_filename', 'N/A')}")
            st.write(f"• **Document Type:** {doc.get('document_type', 'Unknown')}")
            st.write(f"• **File Size:** {doc.get('file_size', 0):,} bytes")
            st.write(f"• **Upload Time:** {doc.get('upload_time', 'Unknown')}")
        
        with col2:
            metadata = doc.get('metadata', {})
            st.write("**Document Metadata:**")
            st.write(f"• **Page Count:** {metadata.get('page_count', 'N/A')}")
            st.write(f"• **Author:** {metadata.get('author', 'N/A')}")
            st.write(f"• **Title:** {metadata.get('title', 'N/A')}")
            st.write(f"• **Created:** {metadata.get('creationDate', 'N/A')}")
        
        # Content preview
        content = doc.get('content', '')
        if content:
            st.write("**Content Preview:**")
            preview_length = 500
            preview_text = content[:preview_length]
            if len(content) > preview_length:
                preview_text += "..."
            
            st.text_area(
                "First 500 characters:",
                value=preview_text,
                height=100,
                disabled=True,
                key=f"preview_{filename}"
            )
            
            st.write(f"**Total Content Length:** {len(content):,} characters")
        else:
            st.warning("No content available for this document.")

def render_batch_operations():
    """Render batch operations for document management"""
    if not st.session_state.uploaded_documents:
        return
    
    st.subheader("🔧 Batch Operations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Document List", use_container_width=True):
            export_document_list()
    
    with col2:
        if st.button("🔄 Reprocess All", use_container_width=True):
            reprocess_all_documents()
    
    with col3:
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            clear_all_documents()
    
    with col4:
        if st.button("📋 Document Statistics", use_container_width=True):
            show_document_statistics()

def export_document_list():
    """Export document list as CSV"""
    if not st.session_state.uploaded_documents:
        st.warning("No documents to export.")
        return
    
    # Prepare export data
    export_data = []
    for doc in st.session_state.uploaded_documents:
        export_data.append({
            'Filename': doc.get('filename', ''),
            'Original_Filename': doc.get('original_filename', ''),
            'Document_Type': doc.get('document_type', ''),
            'File_Size_Bytes': doc.get('file_size', 0),
            'Upload_Time': doc.get('upload_time', ''),
            'Page_Count': doc.get('metadata', {}).get('page_count', ''),
            'Content_Length': len(doc.get('content', '')),
            'Processing_Status': doc.get('processing_status', '')
        })
    
    # Create DataFrame and CSV
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    # Provide download
    st.download_button(
        label="📥 Download Document List CSV",
        data=csv,
        file_name=f"document_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.success(f"✅ Document list ready for download ({len(export_data)} documents)")

def reprocess_all_documents():
    """Reprocess all uploaded documents"""
    if st.button("⚠️ Confirm Reprocessing", type="secondary"):
        st.warning("This will reprocess all documents and may take some time.")
        
        # Reset processing status
        for doc in st.session_state.uploaded_documents:
            doc['processing_status'] = 'pending'
        
        st.info("🔄 Documents marked for reprocessing. This feature is under development.")
        # TODO: Implement actual reprocessing logic

def clear_all_documents():
    """Clear all uploaded documents with confirmation"""
    st.warning("⚠️ This will permanently delete all uploaded documents from the session.")
    
    if st.button("🗑️ Confirm Clear All", type="secondary"):
        st.session_state.uploaded_documents = []
        st.session_state.extracted_recommendations = []
        st.session_state.extracted_concerns = []
        st.session_state.annotation_results = {}
        st.session_state.matching_results = {}
        st.success("✅ All documents cleared successfully.")
        st.rerun()

def show_document_statistics():
    """Display comprehensive document statistics"""
    if not st.session_state.uploaded_documents:
        st.info("No documents to analyze.")
        return
    
    docs = st.session_state.uploaded_documents
    
    # Calculate statistics
    total_docs = len(docs)
    total_size = sum(doc.get('file_size', 0) for doc in docs)
    total_pages = sum(
        doc.get('metadata', {}).get('page_count', 0) 
        for doc in docs 
        if isinstance(doc.get('metadata', {}).get('page_count'), int)
    )
    total_content = sum(len(doc.get('content', '')) for doc in docs)
    
    # Document type distribution
    type_counts = {}
    for doc in docs:
        doc_type = doc.get('document_type', 'Unknown')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    # Display statistics
    st.markdown("### 📊 Document Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
    with col3:
        st.metric("Total Pages", total_pages if total_pages > 0 else "N/A")
    with col4:
        st.metric("Avg Size", f"{(total_size / total_docs) / 1024:.1f} KB" if total_docs > 0 else "0 KB")
    
    # Document type distribution chart
    if type_counts:
        st.markdown("### 📈 Document Type Distribution")
        type_df = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])
        st.bar_chart(type_df.set_index('Type'))
    
    # Content length distribution
    st.markdown("### 📝 Content Analysis")
    content_lengths = [len(doc.get('content', '')) for doc in docs]
    avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Content Length", f"{avg_content_length:,.0f} chars")
    with col2:
        st.metric("Total Content", f"{total_content:,.0f} chars")
    
    # Show content length histogram
    if content_lengths:
        content_df = pd.DataFrame({'Content Length': content_lengths})
        st.bar_chart(content_df)
