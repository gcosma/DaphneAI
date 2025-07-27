"""
AI Document Search Engine - Clean Multi-Result Version
"""

import streamlit as st
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import re
import chardet

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

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
    filename: str
    content: str
    snippet: str
    score: float
    rank: int
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    explanation: str = ""

class DocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_files(self, uploaded_files) -> List[Dict[str, Any]]:
        documents = []
        for file in uploaded_files:
            try:
                doc = self._process_single_file(file)
                if doc:
                    documents.append(doc)
            except Exception as e:
                st.error(f"❌ Failed to process {file.name}: {str(e)}")
        return documents
    
    def _process_single_file(self, file) -> Optional[Dict[str, Any]]:
        filename = file.name
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        text = ""
        if file_extension == 'pdf':
            text = self._extract_pdf_text(file)
        elif file_extension == 'docx':
            text = self._extract_docx_text(file)
        else:
            text = self._extract_text_file(file)
        
        if not text:
            return None
        
        return {
            'filename': filename,
            'text': text,
            'file_type': file_extension,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def _extract_pdf_text(self, file) -> str:
        if not PDF_AVAILABLE:
            raise Exception("PDF processing not available. Install: pip install pdfplumber PyPDF2")
        
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    
    def _extract_docx_text(self, file) -> str:
        if not DOCX_AVAILABLE:
            raise Exception("DOCX processing not available. Install: pip install python-docx")
        
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _extract_text_file(self, file) -> str:
        content = file.read()
        if isinstance(content, bytes):
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                detected = chardet.detect(content)
                encoding = detected.get('encoding', 'utf-8')
                return content.decode(encoding, errors='ignore')
        return content

class MultiResultSearchEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.documents = []
        self.document_embeddings = None
        
        if AI_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("AI model loaded")
            except Exception as e:
                self.model = None
                self.logger.warning(f"AI model failed: {e}")
        else:
            self.model = None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents
        self._build_index()
    
    def _build_index(self) -> None:
        if not self.documents:
            return
        
        texts = [doc.get('text', '') for doc in self.documents]
        
        if self.model and texts:
            try:
                self.document_embeddings = self.model.encode(texts)
                self.logger.info(f"Built AI index for {len(texts)} documents")
            except Exception as e:
                self.logger.error(f"Failed to build AI index: {e}")
                self.document_embeddings = None
    
    def search(self, query: str, max_results: int = 15) -> List[SearchResult]:
        """Multi-algorithm search returning multiple results with scores"""
        if not self.documents or not query.strip():
            return []
        
        # Get semantic results
        semantic_results = {}
        if self.model and self.document_embeddings is not None:
            semantic_results = self._semantic_search(query, max_results * 2)
        
        # Get keyword results  
        keyword_results = self._keyword_search(query, max_results * 2)
        
        # Combine results
        all_results = {}
        
        # Add semantic results
        for result in semantic_results:
            all_results[result.filename] = result
        
        # Add/merge keyword results
        for result in keyword_results:
            if result.filename in all_results:
                existing = all_results[result.filename]
                existing.keyword_score = result.keyword_score
                existing.score = (existing.semantic_score * 0.6) + (result.keyword_score * 0.4)
                existing.explanation = f"Semantic: {existing.semantic_score:.3f}, Keyword: {result.keyword_score:.3f}"
            else:
                all_results[result.filename] = result
        
        # Sort by combined score
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Add rankings
        for i, result in enumerate(final_results[:max_results]):
            result.rank = i + 1
        
        return final_results[:max_results]
    
    def _semantic_search(self, query: str, max_results: int) -> List[SearchResult]:
        """AI semantic search with low threshold for more results"""
        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
            
            results = []
            for idx, similarity in enumerate(similarities):
                if similarity > 0.01:  # Very low threshold
                    doc = self.documents[idx]
                    snippet = self._extract_snippet(doc.get('text', ''), query)
                    
                    result = SearchResult(
                        filename=doc.get('filename', f'Document {idx+1}'),
                        content=doc.get('text', ''),
                        snippet=snippet,
                        score=float(similarity),
                        rank=0,
                        semantic_score=float(similarity),
                        explanation=f"Semantic similarity: {similarity:.3f}"
                    )
                    results.append(result)
            
            results.sort(key=lambda x: x.semantic_score, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Enhanced keyword search with detailed scoring"""
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for idx, doc in enumerate(self.documents):
            text = doc.get('text', '').lower()
            filename = doc.get('filename', '').lower()
            
            if not text:
                continue
            
            score = 0
            explanations = []
            
            # Exact phrase matching
            exact_count = text.count(query_lower)
            if exact_count > 0:
                score += exact_count * 20
                explanations.append(f"'{query}' x{exact_count}")
            
            # Individual word matching
            for word in query_words:
                word_count = text.count(word)
                if word_count > 0:
                    score += word_count * 5
                    explanations.append(f"'{word}' x{word_count}")
                
                # Filename bonus
                if word in filename:
                    score += 15
                    explanations.append(f"'{word}' in filename")
            
            if score > 0:
                # Normalize score to 0-1 range
                normalized_score = min(score / 100.0, 1.0)
                snippet = self._extract_snippet(doc.get('text', ''), query)
                
                result = SearchResult(
                    filename=doc.get('filename', f'Document {idx+1}'),
                    content=doc.get('text', ''),
                    snippet=snippet,
                    score=normalized_score,
                    rank=0,
                    keyword_score=normalized_score,
                    explanation="; ".join(explanations[:3])
                )
                results.append(result)
        
        results.sort(key=lambda x: x.keyword_score, reverse=True)
        return results[:max_results]
    
    def _extract_snippet(self, text: str, query: str, max_length: int = 300) -> str:
        """Extract relevant snippet with highlighting"""
        if not text or not query:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find query position
        pos = text_lower.find(query_lower)
        if pos == -1:
            for word in query_lower.split():
                pos = text_lower.find(word)
                if pos != -1:
                    break
        
        if pos == -1:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Extract snippet
        start = max(0, pos - max_length // 2)
        end = min(len(text), pos + len(query) + max_length // 2)
        snippet = text[start:end]
        
        # Highlight terms
        for word in query.split():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            snippet = pattern.sub(f"**{word.upper()}**", snippet)
        
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        
        return prefix + snippet + suffix
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_documents': len(self.documents),
            'ai_enabled': self.model is not None,
            'index_built': self.document_embeddings is not None,
            'ai_available': AI_AVAILABLE
        }

# UI Functions
def initialize_session_state():
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = MultiResultSearchEngine()
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'documents' not in st.session_state:
        st.session_state.documents = []

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        stats = st.session_state.search_engine.get_stats()
        
        if stats['ai_available'] and stats['ai_enabled']:
            st.success("🤖 AI Search Ready")
        else:
            st.error("🤖 AI Unavailable")
            st.caption("Install: `pip install sentence-transformers scikit-learn`")
        
        st.metric("Documents", stats['total_documents'])

def render_upload_section():
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md', 'py', 'js', 'html', 'css', 'json']
    )
    
    if uploaded_files:
        with st.spinner("Processing..."):
            new_documents = st.session_state.processor.process_files(uploaded_files)
            st.session_state.documents.extend(new_documents)
            st.session_state.search_engine.add_documents(st.session_state.documents)
            st.success(f"✅ Processed {len(new_documents)} documents")

def render_search_section():
    st.header("🔍 Multi-Result Search")
    
    if not st.session_state.documents:
        st.warning("📝 Please upload documents first.")
        return
    
    query = st.text_input("Search query", placeholder="Enter your search...")
    max_results = st.selectbox("Max Results", [10, 15, 20, 25], index=1)
    
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            results = st.session_state.search_engine.search(query, max_results)
        
        if results:
            st.success(f"✅ Found {len(results)} results")
            
            # Results summary
            scores = [r.score for r in results]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Score", f"{max(scores):.3f}")
            with col2:
                st.metric("Average", f"{sum(scores)/len(scores):.3f}")
            with col3:
                st.metric("Results", len(results))
            
            # Display results
            for result in results:
                with st.expander(f"#{result.rank} - {result.filename} (Score: {result.score:.3f})"):
                    
                    # Score details
                    score_cols = st.columns(3)
                    with score_cols[0]:
                        st.metric("Semantic", f"{result.semantic_score:.3f}" if result.semantic_score > 0 else "N/A")
                    with score_cols[1]:
                        st.metric("Keyword", f"{result.keyword_score:.3f}" if result.keyword_score > 0 else "N/A")
                    with score_cols[2]:
                        st.metric("Combined", f"{result.score:.3f}")
                    
                    # Explanation
                    if result.explanation:
                        st.info(f"💡 **Match details:** {result.explanation}")
                    
                    # Content
                    st.markdown("**Preview:**")
                    st.markdown(result.snippet)
                    
                    if st.button(f"📖 Show full content", key=f"show_{result.rank}"):
                        st.text_area("Full content", value=result.content, height=300, key=f"content_{result.rank}")
        else:
            st.warning("No results found. Try different keywords.")

def main():
    st.set_page_config(
        page_title="Multi-Result Document Search",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🔍 Multi-Result Document Search")
    st.markdown("**Advanced search returning multiple relevant results with similarity scores**")
    
    render_sidebar()
    
    tab1, tab2 = st.tabs(["📁 Upload", "🔍 Search"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_search_section()

if __name__ == "__main__":
    main()
