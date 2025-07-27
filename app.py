"""
AI-Based Smart Document Search Engine
Single file Streamlit application
"""

import streamlit as st
import time
import logging
import io
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import re
import chardet

# Optional imports for AI features
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Optional imports for different file types
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

@dataclass
class SearchResult:
    """Structured search result"""
    filename: str
    content: str
    snippet: str
    score: float
    rank: int

class DocumentProcessor:
    """Process various document formats into searchable text"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_files(self, uploaded_files) -> List[Dict[str, Any]]:
        """Process uploaded files and extract text content"""
        documents = []
        
        for file in uploaded_files:
            try:
                doc = self._process_single_file(file)
                if doc:
                    documents.append(doc)
            except Exception as e:
                self.logger.error(f"Failed to process {file.name}: {e}")
                # Add error document
                documents.append({
                    'filename': file.name,
                    'text': '',
                    'error': str(e),
                    'file_size': len(file.getvalue()) if hasattr(file, 'getvalue') else 0
                })
        
        return documents
    
    def _process_single_file(self, file) -> Optional[Dict[str, Any]]:
        """Process a single uploaded file"""
        filename = file.name
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Get file size
        file_content = file.getvalue()
        file_size = len(file_content)
        
        # Reset file pointer
        file.seek(0)
        
        # Process based on file type
        text = ""
        if file_extension == 'pdf':
            text = self._extract_pdf_text(file)
        elif file_extension == 'docx':
            text = self._extract_docx_text(file)
        elif file_extension in ['txt', 'md', 'py', 'js', 'html', 'css', 'json']:
            text = self._extract_text_file(file)
        else:
            # Try to process as text file
            text = self._extract_text_file(file)
        
        if not text:
            return None
        
        return {
            'filename': filename,
            'text': text,
            'file_type': file_extension,
            'file_size': file_size,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def _extract_pdf_text(self, file) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise Exception("PDF processing not available. Install: pip install pdfplumber PyPDF2")
        
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            self.logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                file.seek(0)
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e2:
                self.logger.error(f"PyPDF2 also failed: {e2}")
                raise Exception(f"PDF extraction failed: {e2}")
        
        return text.strip()
    
    def _extract_docx_text(self, file) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise Exception("DOCX processing not available. Install: pip install python-docx")
        
        try:
            doc = Document(file)
            text = ""
            
            # Extract paragraph text
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Extract table text
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {e}")
    
    def _extract_text_file(self, file) -> str:
        """Extract text from plain text files with encoding detection"""
        try:
            # Try UTF-8 first
            content = file.read()
            if isinstance(content, bytes):
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    # Detect encoding
                    detected = chardet.detect(content)
                    encoding = detected.get('encoding', 'utf-8')
                    return content.decode(encoding, errors='ignore')
            else:
                return content
                
        except Exception as e:
            raise Exception(f"Text extraction failed: {e}")
    
    def get_supported_formats(self) -> Dict[str, bool]:
        """Get supported file formats and their availability"""
        return {
            'pdf': PDF_AVAILABLE,
            'docx': DOCX_AVAILABLE,
            'txt': True,
            'md': True,
            'py': True,
            'js': True,
            'html': True,
            'css': True,
            'json': True
        }

class SmartSearchEngine:
    """Unified AI-powered document search engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.documents = []
        self.document_embeddings = None
        
        # Initialize AI model if available
        if AI_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("AI search model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load AI model: {e}")
                self.model = None
        else:
            self.model = None
            self.logger.warning("AI libraries not available - using keyword search")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the search index"""
        self.documents = documents
        self._build_index()
    
    def _build_index(self) -> None:
        """Build search index"""
        if not self.documents:
            return
        
        # Extract text content
        texts = [doc.get('text', doc.get('content', '')) for doc in self.documents]
        
        # Build semantic embeddings if AI available
        if self.model and texts:
            try:
                self.document_embeddings = self.model.encode(texts)
                self.logger.info(f"Built AI index for {len(texts)} documents")
            except Exception as e:
                self.logger.error(f"Failed to build AI index: {e}")
                self.document_embeddings = None
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Smart search that combines AI semantic search with keyword matching
        """
        start_time = time.time()
        
        if not self.documents or not query.strip():
            return []
        
        # Use AI search if available, otherwise fallback to keyword search
        if self.model and self.document_embeddings is not None:
            results = self._ai_search(query, max_results)
        else:
            results = self._keyword_search(query, max_results)
        
        # Add rankings
        for i, result in enumerate(results):
            result.rank = i + 1
        
        search_time = time.time() - start_time
        self.logger.info(f"Search completed in {search_time:.3f}s - Query: '{query}', Results: {len(results)}")
        
        return results
    
    def _ai_search(self, query: str, max_results: int) -> List[SearchResult]:
        """AI-powered semantic search"""
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
            
            # Get top results with minimum similarity threshold
            results = []
            for idx, similarity in enumerate(similarities):
                if similarity > 0.1:  # Minimum relevance threshold
                    doc = self.documents[idx]
                    text = doc.get('text', '')
                    snippet = self._extract_snippet(text, query)
                    
                    result = SearchResult(
                        filename=doc.get('filename', f'Document {idx+1}'),
                        content=text,
                        snippet=snippet,
                        score=float(similarity),
                        rank=0  # Will be set later
                    )
                    results.append(result)
            
            # Sort by similarity score
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"AI search failed: {e}")
            return self._keyword_search(query, max_results)
    
    def _keyword_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Keyword-based search with intelligent scoring"""
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for idx, doc in enumerate(self.documents):
            text = doc.get('text', '')
            text_lower = text.lower()
            filename = doc.get('filename', '').lower()
            
            if not text:
                continue
            
            score = 0
            
            # Exact phrase matching (highest priority)
            if query_lower in text_lower:
                score += text_lower.count(query_lower) * 10
            
            # Individual word matching
            for word in query_words:
                if word in text_lower:
                    score += text_lower.count(word) * 2
            
            # Filename matching bonus
            if query_lower in filename:
                score += 20
            
            if score > 0:
                snippet = self._extract_snippet(text, query)
                
                result = SearchResult(
                    filename=doc.get('filename', f'Document {idx+1}'),
                    content=text,
                    snippet=snippet,
                    score=score,
                    rank=0  # Will be set later
                )
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def _extract_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """Extract relevant snippet from text around query terms"""
        if not text or not query:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find query position
        pos = text_lower.find(query_lower)
        if pos == -1:
            # Find first word
            for word in query_lower.split():
                pos = text_lower.find(word)
                if pos != -1:
                    break
        
        if pos == -1:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Extract snippet around found position
        start = max(0, pos - max_length // 2)
        end = min(len(text), pos + len(query) + max_length // 2)
        snippet = text[start:end]
        
        # Highlight query terms (basic highlighting)
        for word in query.split():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            snippet = pattern.sub(f"**{word}**", snippet)
        
        # Add ellipsis if needed
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        
        return prefix + snippet + suffix
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'total_documents': len(self.documents),
            'ai_enabled': self.model is not None,
            'index_built': self.document_embeddings is not None,
            'ai_available': AI_AVAILABLE
        }

# Streamlit UI Functions
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = SmartSearchEngine()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

def render_sidebar():
    """Render sidebar with system information"""
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        stats = st.session_state.search_engine.get_stats()
        
        # AI Status
        if stats['ai_available'] and stats['ai_enabled']:
            st.success("🤖 AI Search Ready")
        elif stats['ai_available']:
            st.warning("🤖 AI Available (No Index)")
        else:
            st.error("🤖 AI Unavailable")
            st.caption("Install: `pip install sentence-transformers scikit-learn`")
        
        # Document Stats
        st.markdown("### 📊 Document Stats")
        st.metric("Total Documents", stats['total_documents'])
        
        if stats['total_documents'] > 0:
            if stats['index_built']:
                st.success("✅ Search Index Built")
            else:
                st.info("⏳ Building Index...")
        
        # Supported Formats
        st.markdown("### 📁 Supported Formats")
        formats = st.session_state.processor.get_supported_formats()
        
        for fmt, available in formats.items():
            icon = "✅" if available else "❌"
            st.write(f"{icon} {fmt.upper()}")

def render_upload_section():
    """Render file upload interface"""
    st.header("📁 Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md', 'py', 'js', 'html', 'css', 'json'],
        help="Upload documents to search through. Supported formats: PDF, DOCX, TXT, MD, and more."
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            # Process files
            new_documents = st.session_state.processor.process_files(uploaded_files)
            
            # Add to existing documents
            st.session_state.documents.extend(new_documents)
            
            # Update search engine
            st.session_state.search_engine.add_documents(st.session_state.documents)
            
            st.success(f"✅ Processed {len(new_documents)} documents")
            
            # Show processing results
            for doc in new_documents:
                if 'error' in doc:
                    st.error(f"❌ {doc['filename']}: {doc['error']}")
                else:
                    st.info(f"✅ {doc['filename']} - {doc['word_count']} words")

def render_search_section():
    """Render search interface"""
    st.header("🔍 Smart Search")
    
    if not st.session_state.documents:
        st.warning("📝 Please upload documents first to enable search.")
        return
    
    # Search input
    query = st.text_input(
        "Search your documents",
        placeholder="Enter your search query...",
        help="Use natural language or keywords to find relevant information"
    )
    
    # Search button and options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    
    with col2:
        max_results = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
    
    # Perform search
    if search_button and query:
        with st.spinner("Searching..."):
            start_time = time.time()
            
            results = st.session_state.search_engine.search(query, max_results)
            
            search_time = time.time() - start_time
            
            # Add to search history
            st.session_state.search_history.append({
                'query': query,
                'results_count': len(results),
                'search_time': search_time
            })
        
        # Display results
        if results:
            st.success(f"Found {len(results)} results in {search_time:.2f} seconds")
            
            # Results
            for result in results:
                with st.expander(f"#{result.rank} - {result.filename} (Score: {result.score:.3f})"):
                    # Snippet
                    st.markdown("**Preview:**")
                    st.markdown(result.snippet)
                    
                    # Full content toggle
                    if st.button(f"Show full content", key=f"show_{result.rank}"):
                        st.markdown("**Full Content:**")
                        st.text_area(
                            "Document content",
                            value=result.content,
                            height=300,
                            key=f"content_{result.rank}"
                        )
        else:
            st.warning("No results found. Try different keywords or check your documents.")

def render_analytics():
    """Render search analytics"""
    st.header("📈 Search Analytics")
    
    if not st.session_state.search_history:
        st.info("No search history yet. Perform some searches to see analytics.")
        return
    
    # Basic stats
    total_searches = len(st.session_state.search_history)
    avg_results = sum(s['results_count'] for s in st.session_state.search_history) / total_searches
    avg_time = sum(s['search_time'] for s in st.session_state.search_history) / total_searches
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Searches", total_searches)
    
    with col2:
        st.metric("Avg Results", f"{avg_results:.1f}")
    
    with col3:
        st.metric("Avg Time", f"{avg_time:.2f}s")
    
    # Recent searches
    st.subheader("Recent Searches")
    
    for i, search in enumerate(reversed(st.session_state.search_history[-10:])):
        st.write(f"{i+1}. **{search['query']}** - {search['results_count']} results ({search['search_time']:.2f}s)")

def main():
    """Main application function"""
    # Set page config
    st.set_page_config(
        page_title="AI Document Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.title("🔍 AI Document Search Engine")
    st.markdown("**Upload documents and search with intelligent AI-powered ranking**")
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "🔍 Search", "📈 Analytics"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_search_section()
    
    with tab3:
        render_analytics()
    
    # Footer
    st.markdown("---")
    st.markdown("*AI-powered document search with semantic understanding*")

if __name__ == "__main__":
    main()
