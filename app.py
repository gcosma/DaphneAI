# app.py
# Clean, working document search application

import streamlit as st
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup
st.set_page_config(
    page_title="Document Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Document Processing
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    from io import BytesIO
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# AI Dependencies (optional)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

def initialize_session_state():
    """Initialize all session state variables"""
    # Basic state
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = {}
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # AI-related state (initialize as empty/None)
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = None
    
    if 'document_embeddings' not in st.session_state:
        st.session_state.document_embeddings = None
    
    # Analytics
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {
            'total_searches': 0,
            'files_processed': 0,
            'ai_searches': 0,
            'keyword_searches': 0
        }

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text"""
    filename = uploaded_file.name
    file_ext = Path(filename).suffix.lower()
    file_size = len(uploaded_file.getvalue())
    
    try:
        text_content = ""
        
        # PDF Processing
        if file_ext == '.pdf' and PDF_AVAILABLE:
            with pdfplumber.open(uploaded_file) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                text_content = '\n\n'.join(text_parts)
        
        # Word Document Processing
        elif file_ext == '.docx' and DOCX_AVAILABLE:
            doc = docx.Document(BytesIO(uploaded_file.getvalue()))
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            text_content = '\n'.join(text_parts)
        
        # Text File Processing
        elif file_ext == '.txt':
            text_content = uploaded_file.getvalue().decode('utf-8')
        
        else:
            return {
                'filename': filename,
                'error': f'Unsupported file type: {file_ext}',
                'file_type': file_ext[1:] if file_ext else 'unknown'
            }
        
        if not text_content.strip():
            return {
                'filename': filename,
                'error': 'No text content found',
                'file_type': file_ext[1:] if file_ext else 'unknown'
            }
        
        return {
            'filename': filename,
            'text': text_content,
            'word_count': len(text_content.split()),
            'character_count': len(text_content),
            'file_type': file_ext[1:] if file_ext else 'unknown',
            'file_size_mb': file_size / 1024 / 1024,
            'processed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'filename': filename,
            'error': str(e),
            'file_type': file_ext[1:] if file_ext else 'unknown'
        }

def load_ai_model():
    """Load AI model for semantic search"""
    if not AI_AVAILABLE:
        return None
    
    if st.session_state.ai_model is None:
        try:
            with st.spinner("🤖 Loading AI model..."):
                st.session_state.ai_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ AI model loaded")
        except Exception as e:
            st.error(f"Failed to load AI model: {e}")
            return None
    
    return st.session_state.ai_model

def create_embeddings(documents):
    """Create AI embeddings for documents"""
    model = load_ai_model()
    if not model:
        return None
    
    try:
        texts = []
        doc_map = []
        
        for i, doc in enumerate(documents):
            if 'text' in doc and doc['text']:
                # Split into chunks
                chunks = split_into_chunks(doc['text'])
                texts.extend(chunks)
                doc_map.extend([i] * len(chunks))
        
        if texts:
            with st.spinner("🔗 Creating embeddings..."):
                embeddings = model.encode(texts)
                
                return {
                    'embeddings': embeddings,
                    'texts': texts,
                    'doc_map': doc_map,
                    'documents': documents
                }
    except Exception as e:
        st.error(f"Failed to create embeddings: {e}")
        return None

def split_into_chunks(text, chunk_size=300):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def semantic_search(query, max_results=10):
    """Perform AI semantic search"""
    if not st.session_state.document_embeddings:
        return []
    
    model = st.session_state.ai_model
    if not model:
        return []
    
    try:
        # Get query embedding
        query_embedding = model.encode([query])
        
        # Calculate similarities
        embeddings_data = st.session_state.document_embeddings
        similarities = np.dot(query_embedding, embeddings_data['embeddings'].T)[0]
        
        # Get top results
        top_indices = similarities.argsort()[-max_results * 2:][::-1]
        
        # Group by document
        doc_results = {}
        for idx in top_indices:
            if similarities[idx] < 0.1:
                continue
                
            doc_idx = embeddings_data['doc_map'][idx]
            chunk_text = embeddings_data['texts'][idx]
            
            if doc_idx not in doc_results:
                doc_results[doc_idx] = {
                    'document': embeddings_data['documents'][doc_idx],
                    'similarity': similarities[idx],
                    'best_chunk': chunk_text,
                    'search_type': 'AI Semantic'
                }
            elif similarities[idx] > doc_results[doc_idx]['similarity']:
                doc_results[doc_idx]['similarity'] = similarities[idx]
                doc_results[doc_idx]['best_chunk'] = chunk_text
        
        # Convert to list and sort
        results = list(doc_results.values())
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:max_results]
        
    except Exception as e:
        st.error(f"Semantic search failed: {e}")
        return []

def keyword_search(query, documents, max_results=10):
    """Perform keyword search"""
    results = []
    query_lower = query.lower()
    query_words = query_lower.split()
    
    for doc in documents:
        if 'text' not in doc:
            continue
        
        text = doc['text']
        text_lower = text.lower()
        score = 0
        
        # Exact phrase matching
        if query_lower in text_lower:
            score += text_lower.count(query_lower) * 10
        
        # Word matching
        for word in query_words:
            if word in text_lower:
                score += text_lower.count(word) * 2
        
        if score > 0:
            # Extract snippet
            snippet = extract_snippet(text, query)
            
            results.append({
                'document': doc,
                'score': score,
                'snippet': snippet,
                'search_type': 'Keywords'
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

def extract_snippet(text, query, max_length=300):
    """Extract relevant snippet from text"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Find query in text
    index = text_lower.find(query_lower)
    if index == -1:
        # Find first word
        for word in query_lower.split():
            index = text_lower.find(word)
            if index != -1:
                break
    
    if index == -1:
        return text[:max_length] + "..."
    
    # Extract around found position
    start = max(0, index - max_length // 2)
    end = min(len(text), index + len(query) + max_length // 2)
    snippet = text[start:end]
    
    # Highlight query
    highlighted = re.sub(f'({re.escape(query)})', r'**\1**', snippet, flags=re.IGNORECASE)
    
    return ("..." if start > 0 else "") + highlighted + ("..." if end < len(text) else "")

def render_upload_section():
    """Render file upload interface"""
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, DOCX files"
    )
    
    if uploaded_files:
        if st.button("📤 Process Files", type="primary"):
            processed_docs = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                st.write(f"Processing {file.name}...")
                doc = process_uploaded_file(file)
                processed_docs.append(doc)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.documents = processed_docs
            st.session_state.document_embeddings = None  # Reset embeddings
            st.session_state.analytics['files_processed'] += len(uploaded_files)
            
            # Show results
            successful = [doc for doc in processed_docs if 'error' not in doc]
            failed = [doc for doc in processed_docs if 'error' in doc]
            
            if successful:
                st.success(f"✅ Successfully processed {len(successful)} files")
                
                # Show file details
                with st.expander("📊 File Details", expanded=True):
                    for doc in successful:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"📄 **{doc['filename']}**")
                        with col2:
                            st.write(f"📝 {doc['word_count']:,} words")
                        with col3:
                            st.write(f"💾 {doc['file_size_mb']:.1f} MB")
            
            if failed:
                st.error(f"❌ Failed to process {len(failed)} files")
                for doc in failed:
                    st.error(f"• {doc['filename']}: {doc['error']}")

def render_search_section():
    """Render search interface"""
    st.header("🔍 Search Documents")
    
    if not st.session_state.documents:
        st.info("📁 Please upload documents first")
        
        # Sample data option
        if st.button("🎯 Load Sample Data"):
            st.session_state.documents = [
                {
                    'filename': 'sample_document.txt',
                    'text': 'This is a sample document about government policy and healthcare recommendations. The department suggests implementing new safety measures and improving patient care protocols.',
                    'word_count': 24,
                    'file_type': 'txt'
                }
            ]
            st.success("✅ Sample data loaded")
            st.rerun()
        return
    
    # Show document count
    valid_docs = [doc for doc in st.session_state.documents if 'text' in doc]
    st.info(f"📚 Ready to search {len(valid_docs)} documents")
    
    # Search input
    query = st.text_input(
        "Search Query",
        placeholder="Enter your search terms...",
        help="Search for any content in your documents"
    )
    
    # Search options
    col1, col2 = st.columns(2)
    
    with col1:
        search_modes = ['Keywords', 'AI Semantic', 'Both']
        if not AI_AVAILABLE:
            search_modes = ['Keywords']
        
        search_mode = st.selectbox("Search Mode", search_modes)
    
    with col2:
        max_results = st.slider("Max Results", 1, 20, 10)
    
    # Perform search
    if query:
        start_time = time.time()
        
        all_results = []
        
        # Keyword search
        if search_mode in ['Keywords', 'Both']:
            keyword_results = keyword_search(query, valid_docs, max_results)
            all_results.extend(keyword_results)
        
        # AI search
        if search_mode in ['AI Semantic', 'Both'] and AI_AVAILABLE:
            # Build embeddings if needed
            if st.session_state.document_embeddings is None:
                with st.spinner("🧠 Building AI search index..."):
                    st.session_state.document_embeddings = create_embeddings(valid_docs)
            
            if st.session_state.document_embeddings:
                semantic_results = semantic_search(query, max_results)
                all_results.extend(semantic_results)
        
        search_time = time.time() - start_time
        
        # Remove duplicates and sort
        seen = set()
        unique_results = []
        for result in all_results:
            doc_name = result['document']['filename']
            if doc_name not in seen:
                seen.add(doc_name)
                unique_results.append(result)
        
        # Sort by score/similarity
        unique_results.sort(key=lambda x: x.get('similarity', x.get('score', 0)), reverse=True)
        final_results = unique_results[:max_results]
        
        # Update analytics
        st.session_state.analytics['total_searches'] += 1
        if 'AI' in search_mode:
            st.session_state.analytics['ai_searches'] += 1
        else:
            st.session_state.analytics['keyword_searches'] += 1
        
        # Display results
        if final_results:
            st.success(f"🎯 Found {len(final_results)} results in {search_time:.2f} seconds")
            
            for i, result in enumerate(final_results, 1):
                doc = result['document']
                
                with st.expander(f"📄 {i}. {doc['filename']}"):
                    # Document info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Type:** {doc.get('file_type', 'unknown').upper()}")
                    with col2:
                        st.write(f"**Words:** {doc.get('word_count', 0):,}")
                    with col3:
                        score = result.get('similarity', result.get('score', 0))
                        st.write(f"**Score:** {score:.3f}")
                    
                    # Search info
                    st.caption(f"Search Method: {result.get('search_type', 'Unknown')}")
                    
                    # Content snippet
                    if 'snippet' in result:
                        st.markdown("**Relevant excerpt:**")
                        st.markdown(result['snippet'])
                    elif 'best_chunk' in result:
                        st.markdown("**Relevant content:**")
                        st.markdown(result['best_chunk'][:500] + "...")
                    
                    # Full text option
                    if st.button(f"📖 Show Full Text", key=f"full_{i}"):
                        st.text_area("Full Content", doc['text'], height=300, key=f"content_{i}")
        
        else:
            st.warning(f"❌ No results found for '{query}'")

def render_analytics():
    """Render analytics dashboard"""
    st.header("📈 Analytics")
    
    analytics = st.session_state.analytics
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Searches", analytics['total_searches'])
    
    with col2:
        st.metric("Files Processed", analytics['files_processed'])
    
    with col3:
        st.metric("AI Searches", analytics['ai_searches'])
    
    with col4:
        st.metric("Keyword Searches", analytics['keyword_searches'])
    
    # Document stats
    if st.session_state.documents:
        st.subheader("📚 Document Statistics")
        
        valid_docs = [doc for doc in st.session_state.documents if 'text' in doc]
        
        if valid_docs:
            total_words = sum(doc.get('word_count', 0) for doc in valid_docs)
            total_size = sum(doc.get('file_size_mb', 0) for doc in valid_docs)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Documents:** {len(valid_docs)}")
                st.write(f"**Total Words:** {total_words:,}")
                st.write(f"**Total Size:** {total_size:.2f} MB")
            
            with col2:
                if valid_docs:
                    st.write(f"**Avg Words/Doc:** {total_words/len(valid_docs):,.0f}")
                    st.write(f"**Avg Size/Doc:** {total_size/len(valid_docs):.2f} MB")
                
                # File types
                file_types = {}
                for doc in valid_docs:
                    ft = doc.get('file_type', 'unknown')
                    file_types[ft] = file_types.get(ft, 0) + 1
                
                st.write("**File Types:**")
                for ft, count in file_types.items():
                    st.write(f"• {ft.upper()}: {count}")

def main():
    """Main application"""
    # Initialize session state first
    initialize_session_state()
    
    st.title("🔍 Document Search Engine")
    st.markdown("**Upload documents and search with AI-powered semantic understanding**")
    
    # Sidebar with system status
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Check capabilities
        if AI_AVAILABLE:
            st.success("🧠 AI Search Available")
        else:
            st.error("❌ AI Search Unavailable")
            st.caption("pip install sentence-transformers")
        
        if PDF_AVAILABLE:
            st.success("📄 PDF Processing Available")
        else:
            st.error("❌ PDF Processing Unavailable")
            st.caption("pip install pdfplumber")
        
        if DOCX_AVAILABLE:
            st.success("📝 DOCX Processing Available")
        else:
            st.warning("⚠️ DOCX Processing Unavailable")
            st.caption("pip install python-docx")
        
        # Quick stats
        if st.session_state.documents:
            st.markdown("### 📊 Quick Stats")
            valid_docs = [doc for doc in st.session_state.documents if 'text' in doc]
            st.metric("Documents", len(valid_docs))
            
            if st.session_state.document_embeddings:
                st.success("🔗 AI Index Built")
            else:
                st.info("🔗 AI Index Not Built")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "🔍 Search", "📈 Analytics"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_search_section()
    
    with tab3:
        render_analytics()

if __name__ == "__main__":
    main()
