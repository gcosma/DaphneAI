# modules/integration_helper.py
"""
Complete integration helper for DaphneAI document search
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

def prepare_documents_for_search(uploaded_files, text_extraction_function) -> List[Dict[str, Any]]:
    """
    Convert uploaded files to the format required by the search engine
    
    Args:
        uploaded_files: List of uploaded files from Streamlit
        text_extraction_function: Function that extracts text from files
    
    Returns:
        List of documents formatted for the search engine
    """
    documents = []
    
    for i, file in enumerate(uploaded_files):
        try:
            # Use the text extraction function
            extracted_text = text_extraction_function(file)
            
            # Format for search engine
            doc_data = {
                'id': f'doc_{i}_{file.name}',        # Unique ID
                'filename': file.name,               # Display name
                'text': extracted_text,              # Searchable content
                'word_count': len(extracted_text.split()) if extracted_text else 0,
                'file_type': file.name.split('.')[-1].lower() if '.' in file.name else 'unknown',
                'file_size_mb': round(len(file.getvalue()) / 1024 / 1024, 2),
                'document_type': classify_document_type(extracted_text),
                'metadata': {
                    'file_size': len(file.getvalue()),
                    'upload_date': datetime.now().isoformat(),
                    'file_type': file.type
                }
            }
            documents.append(doc_data)
            
            # Store in session state
            if 'documents' not in st.session_state:
                st.session_state.documents = []
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            continue
    
    # Store all documents in session state
    if documents:
        st.session_state.documents = documents
    
    return documents

def classify_document_type(text: str) -> str:
    """
    Classify document type based on content
    """
    if not text:
        return 'general'
    
    text_lower = text.lower()
    
    # Count keywords for different document types
    recommendation_keywords = ['recommend', 'propose', 'suggest', 'advise', 'should']
    response_keywords = ['accept', 'reject', 'agree', 'disagree', 'response', 'reply']
    policy_keywords = ['policy', 'framework', 'strategy', 'guidelines', 'procedures']
    
    rec_count = sum(1 for word in recommendation_keywords if word in text_lower)
    resp_count = sum(1 for word in response_keywords if word in text_lower)
    policy_count = sum(1 for word in policy_keywords if word in text_lower)
    
    # Return the type with the highest count
    if rec_count > resp_count and rec_count > policy_count:
        return 'recommendation'
    elif resp_count > policy_count:
        return 'response'
    elif policy_count > 0:
        return 'policy'
    else:
        return 'general'

def get_document_statistics() -> Dict[str, Any]:
    """
    Get statistics about loaded documents
    """
    if 'documents' not in st.session_state or not st.session_state.documents:
        return {
            'total_documents': 0,
            'document_types': {},
            'total_text_length': 0,
            'average_document_length': 0
        }
    
    documents = st.session_state.documents
    
    # Count by type
    type_counts = {}
    total_length = 0
    
    for doc in documents:
        doc_type = doc.get('document_type', 'general')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        total_length += len(doc.get('text', ''))
    
    return {
        'total_documents': len(documents),
        'document_types': type_counts,
        'total_text_length': total_length,
        'average_document_length': total_length // len(documents) if documents else 0
    }

def setup_search_tab():
    """
    Setup function for the search tab - uses the working search components
    """
    from modules.ui.search_components import render_search_interface
    
    # Check if documents are available
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 No documents available for search")
        st.info("Please upload and process documents first in the Upload tab.")
        
        # Show what format is expected
        with st.expander("📋 Expected Document Format"):
            st.code("""
            # Your documents should be stored as:
            st.session_state.documents = [
                {
                    'id': 'unique_id',
                    'filename': 'document.txt', 
                    'text': 'full document content...',
                    'word_count': 100,
                    'file_type': 'txt',
                    'document_type': 'recommendation'  # optional
                }
            ]
            """)
        return
    
    # Show document statistics
    stats = get_document_statistics()
    
    with st.expander("📊 Document Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", stats['total_documents'])
        
        with col2:
            st.metric("Total Text Length", f"{stats['total_text_length']:,} chars")
        
        with col3:
            st.metric("Average Length", f"{stats['average_document_length']:,} chars")
        
        if stats['document_types']:
            st.markdown("**Document Types:**")
            for doc_type, count in stats['document_types'].items():
                st.write(f"- {doc_type.title()}: {count}")
    
    # Render the search interface using the working components
    render_search_interface(st.session_state.documents)

# Text extraction functions for different file types
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        import PyPDF2
        from io import BytesIO
        
        pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        import docx
        from io import BytesIO
        
        doc = docx.Document(BytesIO(file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        return f"Error extracting TXT: {str(e)}"

def extract_text_from_file(file):
    """
    Universal file text extraction function
    """
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return extract_text_from_txt(file)
    else:
        return f"Unsupported file type: {file.type}"

def render_extract_tab():
    """Content extraction and viewing"""
    st.header("🔍 Content Extraction")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first.")
        return
    
    documents = st.session_state.documents
    
    st.success(f"✅ {len(documents)} documents available for extraction")
    
    # Document selector
    doc_names = [doc['filename'] for doc in documents]
    selected_doc_name = st.selectbox("Select document to view:", doc_names)
    
    # Find selected document
    selected_doc = next((doc for doc in documents if doc['filename'] == selected_doc_name), None)
    
    if selected_doc:
        # Document info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", selected_doc.get('word_count', 0))
        with col2:
            st.metric("File Type", selected_doc.get('file_type', 'unknown').upper())
        with col3:
            st.metric("Document Type", selected_doc.get('document_type', 'general').title())
        
        # Show content
        st.markdown("### 📄 Document Content")
        content = selected_doc.get('text', 'No content available')
        st.text_area("Content", content, height=400, disabled=True)

def render_analytics_tab():
    """Analytics and statistics"""
    st.header("📊 Analytics Dashboard")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 No documents to analyze.")
        return
    
    documents = st.session_state.documents
    stats = get_document_statistics()
    
    # Overview metrics
    st.markdown("### 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", stats['total_documents'])
    
    with col2:
        total_words = sum(doc.get('word_count', 0) for doc in documents)
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        avg_words = total_words // len(documents) if documents else 0
        st.metric("Avg Words/Doc", f"{avg_words:,}")
    
    with col4:
        total_size = sum(doc.get('file_size_mb', 0) for doc in documents)
        st.metric("Total Size", f"{total_size:.1f} MB")
    
    # Document types chart
    if stats['document_types']:
        st.markdown("### 📋 Document Types")
        import pandas as pd
        
        df = pd.DataFrame(list(stats['document_types'].items()), columns=['Type', 'Count'])
        st.bar_chart(df.set_index('Type'))
    
    # Document list
    st.markdown("### 📚 Document Details")
    doc_data = []
    for doc in documents:
        doc_data.append({
            'Filename': doc['filename'],
            'Type': doc.get('document_type', 'general').title(),
            'Words': doc.get('word_count', 0),
            'Size (MB)': doc.get('file_size_mb', 0),
            'File Type': doc.get('file_type', 'unknown').upper()
        })
    
    df = pd.DataFrame(doc_data)
    st.dataframe(df, use_container_width=True)

# Export functions
__all__ = [
    'prepare_documents_for_search',
    'classify_document_type', 
    'get_document_statistics',
    'setup_search_tab',
    'extract_text_from_file',
    'render_extract_tab',
    'render_analytics_tab'
]
