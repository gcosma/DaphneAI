# ===============================================
# FILE: modules/ui/upload_components.py (UPDATED VERSION)
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
        def extract_text_from_pdf(self, path, extract_sections_only=True): return None
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
    process and extract only the relevant sections for analysis.
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
    
    # NEW: Add extraction mode selection
    col1, col2 = st.columns([3, 1])
    
    with col2:
        extraction_mode = st.radio(
            "Extraction Mode:",
            options=["Sections Only", "Full Document"],
            help="Sections Only extracts just recommendations/responses sections (recommended). Full Document extracts everything.",
            key="extraction_mode"
        )
        
        # Store the choice for processing
        st.session_state.extract_sections_only = (extraction_mode == "Sections Only")
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Select one or more PDF documents to upload and process",
            key="document_uploader"
        )
    
    # Guidelines
    st.markdown("**Upload Guidelines:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("• Max file size: 500MB")
    with col2:
        st.markdown("• Supported format: PDF")
    with col3:
        st.markdown("• Text must be readable")
    with col4:
        st.markdown("• Multiple files supported")
    
    # Process uploaded files
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            process_uploaded_files(uploaded_files)

def process_uploaded_files(uploaded_files: List):
    """Process uploaded PDF files with section extraction"""
    processor = DocumentProcessor()
    validator = SecurityValidator()
    
    if not uploaded_files:
        st.warning("No files selected for processing.")
        return
    
    # Get user preference for extraction mode
    extract_sections_only = st.session_state.get('extract_sections_only', True)
    
    # Initialize progress tracking
    total_files = len(uploaded_files)
    progress_container = st.container()
    status_container = st.container()
    
    successful_uploads = 0
    failed_uploads = []
    sections_summary = {'total_sections': 0, 'recommendations': 0, 'responses': 0}
    
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
                # ✅ UPDATED: Extract text with section extraction option
                doc_data = processor.extract_text_from_pdf(tmp_file_path, extract_sections_only=extract_sections_only)
                
                if doc_data and doc_data.get('content'):
                    # Determine document type
                    doc_type = determine_document_type(doc_data['content'])
                    
                    # ✅ UPDATED: Create document info with section support
                    doc_info = {
                        'filename': validator.sanitize_filename(uploaded_file.name),
                        'original_filename': uploaded_file.name,
                        'content': doc_data['content'],
                        'metadata': doc_data.get('metadata', {}),
                        'sections': doc_data.get('sections', []),  # NEW: Add sections data
                        'extraction_type': doc_data.get('extraction_type', 'unknown'),  # NEW: Track extraction type
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
                        
                        # ✅ UPDATED: Better success messaging with section info
                        sections = doc_data.get('sections', [])
                        if sections:
                            sections_count = len(sections)
                            rec_sections = len([s for s in sections if s['type'] == 'recommendations'])
                            resp_sections = len([s for s in sections if s['type'] == 'responses'])
                            
                            sections_summary['total_sections'] += sections_count
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
    
    # ✅ UPDATED: Enhanced results summary with section information
    if successful_uploads > 0:
        st.success(f"✅ Successfully processed {successful_uploads} of {total_files} files!")
        
        if extract_sections_only and sections_summary['total_sections'] > 0:
            st.info(f"""
            📊 **Sections Summary:**
            • Total sections found: {sections_summary['total_sections']}
            • Recommendations sections: {sections_summary['recommendations']}
            • Responses sections: {sections_summary['responses']}
            """)
    
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
        'proposes that', 'advises that', 'urges that', 'coroner\'s concerns',
        'matters of concern'
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
    """Render the document library with enhanced section information"""
    if not st.session_state.uploaded_documents:
        st.info("📝 No documents uploaded yet. Upload some PDF files to get started!")
        return
    
    st.subheader("📚 Document Library")
    
    # ✅ UPDATED: Enhanced document summary with section information
    docs_data = []
    for i, doc in enumerate(st.session_state.uploaded_documents):
        sections = doc.get('sections', [])
        sections_info = f"{len(sections)} sections" if sections else "Full document"
        
        docs_data.append({
            "Index": i,
            "Filename": doc.get('filename', 'Unknown'),
            "Type": doc.get('document_type', 'Unknown'),
            "Extraction": doc.get('extraction_type', 'unknown'),
            "Sections": sections_info,
            "Pages": doc.get('metadata', {}).get('page_count', 'N/A'),
            "Size (KB)": round(doc.get('file_size', 0) / 1024, 1),
            "Upload Time": doc.get('upload_time', '')[:19] if doc.get('upload_time') else 'Unknown',
            "Status": "✅ Ready" if doc.get('processing_status') == 'completed' else "⚠️ Processing"
        })
    
    # Display as interactive dataframe
    df = pd.DataFrame(docs_data)
    
    # Enhanced filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        type_filter = st.selectbox(
            "Filter by Type:",
            options=['All'] + sorted(df['Type'].unique().tolist()),
            key="doc_type_filter"
        )
    
    with col2:
        extraction_filter = st.selectbox(
            "Filter by Extraction:",
            options=['All'] + sorted(df['Extraction'].unique().tolist()),
            key="doc_extraction_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            options=['Upload Time', 'Filename', 'Size (KB)', 'Type', 'Sections'],
            key="doc_sort_by"
        )
    
    with col4:
        sort_order = st.selectbox(
            "Order:",
            options=['Descending', 'Ascending'],
            key="doc_sort_order"
        )
    
    # Apply filters and sorting
    filtered_df = df.copy()
    
    if type_filter != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == type_filter]
    
    if extraction_filter != 'All':
        filtered_df = filtered_df[filtered_df['Extraction'] == extraction_filter]
    
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
        st.warning(f"No documents found matching filters.")

def display_document_details(filename: str):
    """Display detailed information for a selected document with section support"""
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
            st.write(f"• **Extraction Type:** {doc.get('extraction_type', 'Unknown')}")
            st.write(f"• **File Size:** {doc.get('file_size', 0):,} bytes")
            st.write(f"• **Upload Time:** {doc.get('upload_time', 'Unknown')}")
        
        with col2:
            metadata = doc.get('metadata', {})
            st.write("**Document Metadata:**")
            st.write(f"• **Page Count:** {metadata.get('page_count', 'N/A')}")
            st.write(f"• **Author:** {metadata.get('author', 'N/A')}")
            st.write(f"• **Title:** {metadata.get('title', 'N/A')}")
            st.write(f"• **Created:** {metadata.get('creationDate', 'N/A')}")
        
        # ✅ NEW: Show sections information if available
        if 'sections' in doc and doc['sections']:
            st.write("**📋 Sections Found:**")
            sections = doc['sections']
            
            for i, section in enumerate(sections, 1):
                with st.container():
                    st.write(f"**Section {i}: {section['type'].title()}**")
                    
                    # Create columns for section details
                    sec_col1, sec_col2 = st.columns(2)
                    
                    with sec_col1:
                        st.write(f"• **Title:** {section['title']}")
                        st.write(f"• **Pages:** {section['page_start']}-{section['page_end']}")
                    
                    with sec_col2:
                        content_stats = section.get('content_stats', {})
                        st.write(f"• **Words:** {content_stats.get('word_count', 'Unknown')}")
                        st.write(f"• **Characters:** {content_stats.get('character_count', 'Unknown')}")
                    
                    # Show content preview
                    content_preview = section.get('content', '')[:200]
                    if content_preview:
                        st.text_area(
                            f"Preview (Section {i}):",
                            value=content_preview + "..." if len(section.get('content', '')) > 200 else content_preview,
                            height=80,
                            disabled=True,
                            key=f"section_preview_{filename}_{i}"
                        )
                    
                    st.write("---")
                    
        elif doc.get('extraction_type') == 'sections_only':
            st.warning("📋 Document was processed for sections but none were found.")
            st.markdown("""
            **Possible reasons:**
            • Document doesn't contain standard recommendation/response sections
            • Sections might be formatted differently than expected
            • Try switching to 'Full Document' extraction mode
            """)
        
        # Content preview (full content)
        content = doc.get('content', '')
        if content:
            st.write("**📄 Content Preview:**")
            preview_length = 500
            preview_text = content[:preview_length]
            if len(content) > preview_length:
                preview_text += "..."
            
            st.text_area(
                "First 500 characters:",
                value=preview_text,
                height=100,
                disabled=True,
                key=f"full_preview_{filename}"
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
    """Export document list as CSV with section information"""
    if not st.session_state.uploaded_documents:
        st.warning("No documents to export.")
        return
    
    # ✅ UPDATED: Include section information in export
    export_data = []
    for doc in st.session_state.uploaded_documents:
        sections = doc.get('sections', [])
        
        base_info = {
            'filename': doc.get('filename', ''),
            'document_type': doc.get('document_type', ''),
            'extraction_type': doc.get('extraction_type', ''),
            'file_size_kb': round(doc.get('file_size', 0) / 1024, 1),
            'upload_time': doc.get('upload_time', ''),
            'page_count': doc.get('metadata', {}).get('page_count', ''),
            'total_sections': len(sections),
            'recommendations_sections': len([s for s in sections if s['type'] == 'recommendations']),
            'responses_sections': len([s for s in sections if s['type'] == 'responses'])
        }
        
        if sections:
            # Add detailed section information
            for i, section in enumerate(sections, 1):
                section_info = base_info.copy()
                section_info.update({
                    'section_number': i,
                    'section_type': section['type'],
                    'section_title': section['title'],
                    'section_pages': f"{section['page_start']}-{section['page_end']}",
                    'section_words': section.get('content_stats', {}).get('word_count', 0)
                })
                export_data.append(section_info)
        else:
            # No sections, just add the base info
            export_data.append(base_info)
    
    # Create DataFrame and export
    df = pd.DataFrame(export_data)
    csv_data = df.to_csv(index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"document_analysis_{timestamp}.csv"
    
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )
    
    st.success(f"✅ Document list exported! {len(export_data)} rows generated.")

def reprocess_all_documents():
    """Reprocess all uploaded documents with current settings"""
    if st.button("⚠️ Confirm Reprocess All", type="secondary"):
        st.warning("This will reprocess all documents. Current analysis will be lost.")
        
        # Clear current data
        st.session_state.uploaded_documents = []
        st.session_state.extracted_recommendations = []
        st.session_state.extracted_concerns = []
        
        st.success("✅ Documents cleared. Re-upload your files to reprocess with current settings.")
        st.rerun()

def clear_all_documents():
    """Clear all uploaded documents"""
    if st.button("⚠️ Confirm Clear All", type="secondary"):
        st.session_state.uploaded_documents = []
        st.session_state.extracted_recommendations = []
        st.session_state.extracted_concerns = []
        st.success("✅ All documents cleared!")
        st.rerun()

def show_document_statistics():
    """Show comprehensive document statistics"""
    docs = st.session_state.uploaded_documents
    
    if not docs:
        st.warning("No documents to analyze.")
        return
    
    # ✅ UPDATED: Enhanced statistics with section information
    st.subheader("📊 Document Statistics")
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(docs))
    
    with col2:
        total_size = sum(doc.get('file_size', 0) for doc in docs)
        st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
    
    with col3:
        total_pages = sum(doc.get('metadata', {}).get('page_count', 0) for doc in docs)
        st.metric("Total Pages", total_pages)
    
    with col4:
        total_sections = sum(len(doc.get('sections', [])) for doc in docs)
        st.metric("Total Sections", total_sections)
    
    # Section breakdown
    if total_sections > 0:
        st.subheader("📋 Section Analysis")
        
        section_types = {}
        extraction_types = {}
        
        for doc in docs:
            # Count extraction types
            ext_type = doc.get('extraction_type', 'unknown')
            extraction_types[ext_type] = extraction_types.get(ext_type, 0) + 1
            
            # Count section types
            for section in doc.get('sections', []):
                sec_type = section['type']
                section_types[sec_type] = section_types.get(sec_type, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Extraction Types:**")
            for ext_type, count in extraction_types.items():
                st.write(f"• {ext_type.replace('_', ' ').title()}: {count}")
        
        with col2:
            st.write("**Section Types:**")
            for sec_type, count in section_types.items():
                st.write(f"• {sec_type.replace('_', ' ').title()}: {count}")

# ✅ NEW: Test function to verify section extraction
def test_section_extraction():
    """Test function to verify section extraction is working correctly"""
    st.subheader("🧪 Section Extraction Test")
    
    if st.button("Test Section Extraction"):
        if st.session_state.uploaded_documents:
            st.write("Testing section extraction on uploaded documents...")
            
            for doc in st.session_state.uploaded_documents[:3]:  # Test first 3 docs
                st.write(f"**Document**: {doc['filename']}")
                
                if 'sections' in doc and doc['sections']:
                    sections = doc['sections']
                    st.success(f"✅ Found {len(sections)} sections:")
                    for section in sections:
                        st.write(f"  - {section['type']}: Pages {section['page_start']}-{section['page_end']}")
                elif doc.get('extraction_type') == 'sections_only':
                    st.warning("❌ No sections found - document processed for sections but none detected")
                else:
                    st.info("ℹ️ Full document extraction - no section data available")
                
                st.write("---")
        else:
            st.warning("Upload some documents first to test extraction")

# Initialize session state for uploaded documents
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

# Initialize other required session state variables
if 'extracted_recommendations' not in st.session_state:
    st.session_state.extracted_recommendations = []

if 'extracted_concerns' not in st.session_state:
    st.session_state.extracted_concerns = []

if 'extract_sections_only' not in st.session_state:
    st.session_state.extract_sections_only = True  # Default to sections only


# Error handling and logging setup
def setup_upload_logging():
    """Setup logging for upload components"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Utility function for backwards compatibility
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


# Main module initialization
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
