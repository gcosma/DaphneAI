# app.py
"""
DaphneAI - Government Document Search and Analysis
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from modules.integration_helper import setup_search_tab, prepare_documents_for_search, extract_text_from_file

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
    st.success(f"📄 {len(documents)} documents available")
    
    # Document selector
    doc_names = [doc['filename'] for doc in documents]
    selected_doc = st.selectbox("Select document to view:", doc_names)
    
    if selected_doc:
        # Find selected document
        doc = next(d for d in documents if d['filename'] == selected_doc)
        
        # Document info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Document Type", doc.get('document_type', 'general').title())
        with col2:
            st.metric("Text Length", f"{len(doc['text']):,} chars")
        with col3:
            st.metric("Word Count", f"{len(doc['text'].split()):,} words")
        
        # Document content
        st.subheader("📖 Document Content")
        st.text_area(
            "Full Text",
            doc['text'],
            height=400,
            help="Full extracted text content"
        )

def render_analytics_tab():
    """Analytics and statistics"""
    st.header("📊 Analytics Dashboard")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 No documents available for analysis.")
        return
    
    from modules.integration_helper import get_document_statistics
    stats = get_document_statistics()
    
    # Overview metrics
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", stats['total_documents'])
    
    with col2:
        st.metric("Total Characters", f"{stats['total_text_length']:,}")
    
    with col3:
        st.metric("Average Length", f"{stats['average_document_length']:,}")
    
    with col4:
        avg_words = stats['total_text_length'] // 5  # Rough word estimate
        st.metric("Est. Total Words", f"{avg_words:,}")
    
    # Document type breakdown
    if stats['document_types']:
        st.subheader("📋 Document Types")
        
        # Create chart data
        type_data = []
        for doc_type, count in stats['document_types'].items():
            type_data.append({
                'Type': doc_type.title(),
                'Count': count,
                'Percentage': round((count / stats['total_documents']) * 100, 1)
            })
        
        # Display as table and chart
        col1, col2 = st.columns(2)
        
        with col1:
            df = pd.DataFrame(type_data)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.bar_chart(df.set_index('Type')['Count'])
    
    # Search analytics (if available)
    if 'search_history' in st.session_state and st.session_state.search_history:
        st.subheader("🔍 Search Analytics")
        
        search_history = st.session_state.search_history
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Searches", len(search_history))
        
        with col2:
            recent_searches = search_history[-5:]
            st.write("**Recent Searches:**")
            for search in reversed(recent_searches):
                st.write(f"• {search['query']} ({search['results']} results)")

if __name__ == "__main__":
    main()
