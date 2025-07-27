# integration_helper.py
"""
Helper functions to integrate the government search engine with your existing DaphneAI app
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

def prepare_documents_for_search(uploaded_files, text_extraction_function) -> List[Dict[str, Any]]:
    """
    Convert uploaded files to the format required by the search engine
    
    Args:
        uploaded_files: List of uploaded files from Streamlit
        text_extraction_function: Your existing function that extracts text from files
    
    Returns:
        List of documents formatted for the search engine
    """
    documents = []
    
    for i, file in enumerate(uploaded_files):
        try:
            # Use your existing text extraction function
            extracted_text = text_extraction_function(file)
            
            # Format for search engine
            doc_data = {
                'id': f'doc_{i}_{file.name}',        # Unique ID
                'filename': file.name,               # Display name
                'text': extracted_text,              # Searchable content
                'document_type': classify_document_type(extracted_text),
                'metadata': {
                    'file_size': file.size,
                    'upload_date': datetime.now().isoformat(),
                    'file_type': file.type
                }
            }
            documents.append(doc_data)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            continue
    
    # Store in session state for search engine
    st.session_state.documents = documents
    
    return documents

def classify_document_type(text: str) -> str:
    """
    Classify document type based on content
    
    Args:
        text: Document text content
    
    Returns:
        Document type: 'recommendation', 'response', 'policy', or 'general'
    """
    text_lower = text.lower()
    
    # Government recommendation indicators
    recommendation_keywords = [
        'recommend', 'recommendation', 'suggests', 'proposes', 
        'advises', 'committee recommends', 'we recommend'
    ]
    
    # Government response indicators  
    response_keywords = [
        'government response', 'accept', 'reject', 'government accepts',
        'government rejects', 'minister responds', 'department response'
    ]
    
    # Policy document indicators
    policy_keywords = [
        'policy', 'regulation', 'guideline', 'framework', 
        'strategy', 'directive', 'government policy'
    ]
    
    # Count occurrences
    recommendation_count = sum(1 for keyword in recommendation_keywords if keyword in text_lower)
    response_count = sum(1 for keyword in response_keywords if keyword in text_lower)
    policy_count = sum(1 for keyword in policy_keywords if keyword in text_lower)
    
    # Classify based on highest count
    if recommendation_count > response_count and recommendation_count > policy_count:
        return 'recommendation'
    elif response_count > recommendation_count and response_count > policy_count:
        return 'response'
    elif policy_count > 0:
        return 'policy'
    else:
        return 'general'

def get_document_statistics() -> Dict[str, Any]:
    """
    Get statistics about loaded documents
    
    Returns:
        Dictionary with document statistics
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
    Complete setup function for the search tab
    Call this function in your search tab
    """
    from .government_search_ui import render_government_search
    
    # Check if documents are available
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 No documents available for search")
        st.info("Please upload and process documents first in the Upload/Extract tabs.")
        
        # Show what format is expected
        with st.expander("📋 Expected Document Format"):
            st.code("""
            # Your documents should be stored as:
            st.session_state.documents = [
                {
                    'id': 'unique_id',
                    'filename': 'document.txt', 
                    'text': 'full document content...',
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
    
    # Render the main search interface
    render_government_search()

# Sample integration functions for common file types
def extract_text_from_pdf(file):
    """
    Sample PDF text extraction function
    Replace this with your existing PDF extraction logic
    """
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
    """
    Sample DOCX text extraction function  
    Replace this with your existing DOCX extraction logic
    """
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
    """
    Sample TXT text extraction function
    Replace this with your existing TXT extraction logic
    """
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        return f"Error extracting TXT: {str(e)}"

def extract_text_from_file(file):
    """
    Universal file text extraction function
    Add this to your existing code or replace with your extraction function
    """
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return extract_text_from_txt(file)
    else:
        return f"Unsupported file type: {file.type}"

# Export functions
__all__ = [
    'prepare_documents_for_search',
    'classify_document_type', 
    'get_document_statistics',
    'setup_search_tab',
    'extract_text_from_file'
]
