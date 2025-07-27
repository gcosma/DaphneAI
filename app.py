# app.py
"""
DaphneAI - Government Document Search and Analysis
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from modules.integration_helper import (
    setup_search_tab, 
    prepare_documents_for_search, 
    extract_text_from_file,
    get_document_statistics,
    render_analytics_tab
)

def main():
    """Main application"""
    st.set_page_config(
        page_title="DaphneAI - Government Document Search", 
        layout="wide"
    )
    
    st.title("🏛️ DaphneAI - Government Document Analysis")
    st.markdown("*Advanced document processing and search for government content*")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload", 
        "🔍 Extract", 
        "🔍 Search",
        "📊 Analytics"
    ])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_extract_tab()
    
    with tab3:
        setup_search_tab()
    
    with tab4:
        render_analytics_tab()

def render_upload_tab():
    """Document upload and processing"""
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files for analysis"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Files", type="primary"):
            with st.spinner("Processing documents..."):
                documents = prepare_documents_for_search(uploaded_files, extract_text_from_file)
                
                st.success(f"✅ Processed {len(documents)} documents")
                
                # Show document types
                doc_types = {}
                for doc in documents:
                    doc_type = doc.get('document_type', 'general')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                if doc_types:
                    st.info(f"Document types: {', '.join([f'{k}: {v}' for k, v in doc_types.items()])}")

def render_extract_tab():
    """Content extraction and viewing"""
    st.header("🔍 Content Extraction")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first.")
        return
    
    documents = st.session_state.documents
    
    # Document selector
    doc_names = [doc['filename'] for doc in documents]
    selected_doc = st.selectbox("Select document to view:", doc_names)
    
    if selected_doc:
        # Find selected document
        doc = next((d for d in documents if d['filename'] == selected_doc), None)
        
        if doc:
            # Document info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Type", doc.get('file_type', 'unknown').upper())
            with col2:
                st.metric("Word Count", f"{doc.get('word_count', 0):,}")
            with col3:
                st.metric("File Size", f"{doc.get('file_size_mb', 0):.1f} MB")
            
            # Document content
            st.subheader("📄 Document Content")
            
            # Show text with option to expand
            if 'text' in doc and doc['text']:
                text = doc['text']
                
                # Show preview by default
                preview_length = 1000
                if len(text) > preview_length:
                    st.text_area(
                        "Content Preview (first 1000 characters):",
                        text[:preview_length] + "...",
                        height=200,
                        disabled=True
                    )
                    
                    # Option to show full text
                    if st.button("📖 Show Full Content"):
                        st.text_area(
                            "Full Content:",
                            text,
                            height=400,
                            disabled=True
                        )
                else:
                    st.text_area(
                        "Full Content:",
                        text,
                        height=300,
                        disabled=True
                    )
            else:
                st.error("No text content available for this document")
        
        # Download options
        if st.button("💾 Download Extracted Text"):
            if doc and 'text' in doc:
                st.download_button(
                    label="Download as TXT",
                    data=doc['text'],
                    file_name=f"{doc['filename']}_extracted.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
