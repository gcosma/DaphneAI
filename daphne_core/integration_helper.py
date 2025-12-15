# modules/integration_helper.py
# Complete integration helper with all functions

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Callable
import logging
import re
from pathlib import Path

def prepare_documents_for_search(uploaded_files: List, text_extractor: Callable) -> List[Dict[str, Any]]:
    """Process uploaded files and prepare for search"""
    if not uploaded_files:
        return []
    
    processed_docs = []
    upload_dir = Path("output/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        try:
            # Extract text using provided extractor
            text = text_extractor(uploaded_file)
            
            # Classify document type
            doc_type = classify_document_type(text, uploaded_file.name)
            
            file_type = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else 'unknown'

            # Optionally persist PDFs to disk so v2 pipelines can re-read
            # them directly (e.g. for unstructured-based preprocessing).
            pdf_path = None
            if file_type == 'pdf':
                pdf_path = upload_dir / uploaded_file.name
                # Overwrite if it already exists for this session.
                pdf_path.write_bytes(uploaded_file.getvalue())

            # Create document record
            doc = {
                'filename': uploaded_file.name,
                'text': text,
                'word_count': len(text.split()) if text else 0,
                'file_type': file_type,
                'file_size_mb': len(uploaded_file.getvalue()) / (1024 * 1024),
                'document_type': doc_type,
                'processed_at': datetime.now().isoformat(),
            }
            if pdf_path is not None:
                doc['pdf_path'] = str(pdf_path)
            
            processed_docs.append(doc)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            logging.error(f"Document processing error: {e}")
    
    # Store in session state
    if processed_docs:
        st.session_state.documents = processed_docs
        logging.info(f"Processed {len(processed_docs)} documents successfully")
    
    return processed_docs

def classify_document_type(text: str, filename: str) -> str:
    """Classify document type based on content and filename"""
    if not text:
        return 'unknown'
    
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Government document patterns
    gov_patterns = [
        r'\b(ministry|department|government|parliament|cabinet)\b',
        r'\b(policy|regulation|act|bill|statute)\b',
        r'\b(recommendation|directive|circular|guidance)\b',
        r'\b(official|public|crown|her majesty)\b'
    ]
    
    # Medical/Health patterns
    health_patterns = [
        r'\b(patient|medical|health|clinical|hospital)\b',
        r'\b(treatment|diagnosis|therapy|medication)\b',
        r'\b(nhs|healthcare|blood|infection)\b'
    ]
    
    # Legal patterns
    legal_patterns = [
        r'\b(court|judge|legal|law|litigation)\b',
        r'\b(evidence|testimony|witness|ruling)\b',
        r'\b(case|proceeding|hearing|trial)\b'
    ]
    
    # Check patterns
    if any(re.search(pattern, text_lower) for pattern in gov_patterns):
        return 'government'
    elif any(re.search(pattern, text_lower) for pattern in health_patterns):
        return 'healthcare'
    elif any(re.search(pattern, text_lower) for pattern in legal_patterns):
        return 'legal'
    elif 'report' in filename_lower or 'analysis' in filename_lower:
        return 'report'
    elif 'inquiry' in filename_lower or 'investigation' in filename_lower:
        return 'inquiry'
    else:
        return 'general'

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file - wrapper for document processor"""
    try:
        from .document_processor import extract_pdf_text, extract_txt_text, extract_docx_text
        
        filename = uploaded_file.name.lower()
        
        if filename.endswith('.pdf'):
            return extract_pdf_text(uploaded_file)
        elif filename.endswith('.txt'):
            return extract_txt_text(uploaded_file)
        elif filename.endswith('.docx'):
            return extract_docx_text(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
            
    except ImportError as e:
        st.error("Document processor not available")
        logging.error(f"Import error in text extraction: {e}")
        return ""
    except Exception as e:
        logging.error(f"Text extraction error: {e}")
        raise

def get_document_statistics() -> Dict[str, Any]:
    """Get statistics about processed documents"""
    if 'documents' not in st.session_state:
        return {
            'total_documents': 0,
            'total_words': 0,
            'document_types': {},
            'file_types': {},
            'avg_words_per_doc': 0
        }
    
    documents = st.session_state.documents
    
    # Calculate statistics
    total_words = sum(doc.get('word_count', 0) for doc in documents)
    doc_types = {}
    file_types = {}
    
    for doc in documents:
        # Document type counts
        doc_type = doc.get('document_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # File type counts
        file_type = doc.get('file_type', 'unknown')
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    return {
        'total_documents': len(documents),
        'total_words': total_words,
        'document_types': doc_types,
        'file_types': file_types,
        'avg_words_per_doc': total_words // len(documents) if documents else 0
    }

def setup_search_tab():
    """Setup the search tab with interface"""
    try:
        from ui.search_components import render_search_interface
        
        documents = st.session_state.get('documents', [])
        render_search_interface(documents)
        
    except ImportError:
        st.error("Search components not available")
        # Fallback basic search
        render_basic_search()

def render_basic_search():
    """Basic search interface fallback"""
    st.header("🔍 Basic Search")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("No documents uploaded. Please upload documents first.")
        return
    
    documents = st.session_state.documents
    query = st.text_input("Search documents:", placeholder="Enter search terms...")
    
    if query:
        results = []
        query_lower = query.lower()
        
        for doc in documents:
            if 'text' in doc and query_lower in doc['text'].lower():
                # Simple match
                count = doc['text'].lower().count(query_lower)
                results.append({
                    'filename': doc['filename'],
                    'matches': count,
                    'word_count': doc.get('word_count', 0)
                })
        
        if results:
            st.success(f"Found {len(results)} matching documents")
            for result in results:
                st.write(f"📄 {result['filename']} - {result['matches']} matches")
        else:
            st.warning("No matches found")

def render_analytics_tab():
    """Analytics and document statistics"""
    st.header("📊 Analytics Dashboard")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 No documents to analyze. Please upload documents first.")
        return
    
    documents = st.session_state.documents
    stats = get_document_statistics()
    
    # Overview metrics
    st.markdown("### 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", stats['total_documents'])
    
    with col2:
        st.metric("Total Words", f"{stats['total_words']:,}")
    
    with col3:
        st.metric("Avg Words/Doc", f"{stats['avg_words_per_doc']:,}")
    
    with col4:
        total_size = sum(doc.get('file_size_mb', 0) for doc in documents)
        st.metric("Total Size", f"{total_size:.1f} MB")
    
    # Document types chart
    if stats['document_types']:
        st.markdown("### 📋 Document Types")
        
        # Create dataframe for chart
        df_types = pd.DataFrame(
            list(stats['document_types'].items()), 
            columns=['Type', 'Count']
        )
        st.bar_chart(df_types.set_index('Type'))
    
    # File types distribution
    if stats['file_types']:
        st.markdown("### 📁 File Types")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart data
            for file_type, count in stats['file_types'].items():
                percentage = (count / stats['total_documents']) * 100
                st.write(f"**{file_type.upper()}**: {count} files ({percentage:.1f}%)")
        
        with col2:
            df_files = pd.DataFrame(
                list(stats['file_types'].items()),
                columns=['File Type', 'Count']
            )
            st.bar_chart(df_files.set_index('File Type'))
    
    # Document details table
    st.markdown("### 📚 Document Details")
    doc_data = []
    for doc in documents:
        doc_data.append({
            'Filename': doc['filename'],
            'Type': doc.get('document_type', 'general').title(),
            'Words': doc.get('word_count', 0),
            'Size (MB)': round(doc.get('file_size_mb', 0), 2),
            'File Type': doc.get('file_type', 'unknown').upper(),
            'Processed': doc.get('processed_at', 'Unknown')[:16]  # Just date and time
        })
    
    df = pd.DataFrame(doc_data)
    st.dataframe(df, use_container_width=True)
    
    # Search analytics (if available)
    if 'search_history' in st.session_state and st.session_state.search_history:
        st.markdown("### 🔍 Search Analytics")
        
        search_history = st.session_state.search_history
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Searches", len(search_history))
            
            # Recent queries
            recent_queries = [h.get('query', '') for h in search_history[-5:]]
            if recent_queries:
                st.write("**Recent Queries:**")
                for query in reversed(recent_queries):
                    if query:
                        st.write(f"• {query}")
        
        with col2:
            avg_results = sum(h.get('result_count', 0) for h in search_history) / len(search_history)
            st.metric("Avg Results", f"{avg_results:.1f}")
            
            avg_time = sum(h.get('search_time', 0) for h in search_history) / len(search_history)
            st.metric("Avg Search Time", f"{avg_time:.3f}s")
    
    # Export options
    st.markdown("### 💾 Export Options")
    if st.button("📊 Download Analytics Report"):
        # Create analytics report
        report = f"""DaphneAI Analytics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DOCUMENT SUMMARY:
- Total Documents: {stats['total_documents']}
- Total Words: {stats['total_words']:,}
- Average Words per Document: {stats['avg_words_per_doc']:,}
- Total Size: {sum(doc.get('file_size_mb', 0) for doc in documents):.1f} MB

DOCUMENT TYPES:
{chr(10).join([f'- {k}: {v}' for k, v in stats['document_types'].items()])}

FILE TYPES:
{chr(10).join([f'- {k}: {v}' for k, v in stats['file_types'].items()])}
"""
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Export functions
__all__ = [
    'prepare_documents_for_search',
    'classify_document_type', 
    'get_document_statistics',
    'setup_search_tab',
    'extract_text_from_file',
    'render_analytics_tab'
]
