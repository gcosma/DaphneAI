"""
Enhanced AI Document Search Engine 
Based on your previous sophisticated search system
"""

import streamlit as st
import time
import logging
import pandas as pd
import json
import re
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import chardet
from datetime import datetime
from collections import Counter

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
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
    content_type: str = ""
    source: str = ""
    metadata: Dict[str, Any] = None
    explanation: str = ""

class AdvancedDocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_files(self, uploaded_files) -> List[Dict[str, Any]]:
        documents = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            st.write(f"Processing: {file.name}")
            
            try:
                doc = self._process_single_file(file)
                if doc:
                    documents.append(doc)
            except Exception as e:
                st.error(f"❌ Failed to process {file.name}: {str(e)}")
        
        progress_bar.empty()
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
        
        # Detect content type based on keywords
        content_type = self._detect_content_type(text)
        
        return {
            'filename': filename,
            'text': text,
            'file_type': file_extension,
            'content_type': content_type,
            'word_count': len(text.split()),
            'char_count': len(text),
            'upload_time': datetime.now().isoformat()
        }
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type based on keywords"""
        text_lower = text.lower()
        
        # Define keyword patterns
        patterns = {
            'concern': ['concern', 'worried', 'problem', 'issue', 'risk', 'failure', 'inadequate'],
            'recommendation': ['recommend', 'suggest', 'should', 'ought', 'advise', 'propose'],
            'response': ['response', 'reply', 'answer', 'acknowledge', 'address'],
            'policy': ['policy', 'procedure', 'protocol', 'guideline', 'standard'],
            'training': ['training', 'education', 'learning', 'course', 'workshop']
        }
        
        # Count matches for each type
        type_scores = {}
        for content_type, keywords in patterns.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            type_scores[content_type] = score
        
        # Return type with highest score, or 'general' if no clear match
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        return 'general'
    
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

class AdvancedSearchEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.documents = []
        self.document_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        if AI_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("AI model loaded successfully")
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
        
        # Build semantic embeddings
        if self.model and texts:
            try:
                with st.spinner("Building AI search index..."):
                    self.document_embeddings = self.model.encode(texts, show_progress_bar=False)
                self.logger.info(f"Built AI index for {len(texts)} documents")
            except Exception as e:
                self.logger.error(f"Failed to build AI index: {e}")
                self.document_embeddings = None
        
        # Build TF-IDF index
        if AI_AVAILABLE and texts:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    stop_words='english',
                    max_features=5000,
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.95
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                self.logger.info(f"Built TF-IDF index for {len(texts)} documents")
            except Exception as e:
                self.logger.error(f"Failed to build TF-IDF index: {e}")
    
    def search(self, query: str, search_type: str = "All Content", max_results: int = 10, 
               search_mode: str = "Hybrid Search") -> List[SearchResult]:
        """Advanced search with multiple modes and filtering"""
        
        if not self.documents or not query.strip():
            return []
        
        # Filter documents by content type
        filtered_indices = self._filter_documents_by_type(search_type)
        
        # Perform search based on mode
        if search_mode == "Semantic Search" and self.model:
            results = self._semantic_search(query, filtered_indices, max_results)
        elif search_mode == "Keyword Search":
            results = self._keyword_search(query, filtered_indices, max_results)
        else:  # Hybrid Search
            results = self._hybrid_search(query, filtered_indices, max_results)
        
        return results
    
    def _filter_documents_by_type(self, search_type: str) -> List[int]:
        """Filter document indices by content type"""
        if search_type == "All Content":
            return list(range(len(self.documents)))
        
        type_mapping = {
            "Recommendations Only": "recommendation",
            "Responses Only": "response",
            "Concerns Only": "concern",
            "Policies Only": "policy",
            "Training Only": "training"
        }
        
        target_type = type_mapping.get(search_type)
        if not target_type:
            return list(range(len(self.documents)))
        
        filtered_indices = []
        for i, doc in enumerate(self.documents):
            if doc.get('content_type') == target_type:
                filtered_indices.append(i)
        
        return filtered_indices if filtered_indices else list(range(len(self.documents)))
    
    def _semantic_search(self, query: str, doc_indices: List[int], max_results: int) -> List[SearchResult]:
        """Semantic search using AI embeddings"""
        try:
            query_embedding = self.model.encode([query])
            
            # Filter embeddings
            if doc_indices:
                filtered_embeddings = self.document_embeddings[doc_indices]
                similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
            else:
                similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
                doc_indices = list(range(len(self.documents)))
            
            results = []
            for i, similarity in enumerate(similarities):
                if similarity > 0.05:  # Low threshold for more results
                    doc_idx = doc_indices[i] if doc_indices else i
                    doc = self.documents[doc_idx]
                    
                    result = SearchResult(
                        filename=doc.get('filename', f'Document {doc_idx+1}'),
                        content=doc.get('text', ''),
                        snippet=self._extract_snippet(doc.get('text', ''), query),
                        score=float(similarity),
                        rank=0,
                        semantic_score=float(similarity),
                        content_type=doc.get('content_type', 'general'),
                        source=doc.get('filename', ''),
                        explanation=f"Semantic similarity: {similarity:.3f}"
                    )
                    results.append(result)
            
            results.sort(key=lambda x: x.semantic_score, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, doc_indices: List[int], max_results: int) -> List[SearchResult]:
        """Enhanced keyword search"""
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        indices_to_search = doc_indices if doc_indices else range(len(self.documents))
        
        for doc_idx in indices_to_search:
            doc = self.documents[doc_idx]
            text = doc.get('text', '').lower()
            filename = doc.get('filename', '').lower()
            
            if not text:
                continue
            
            score = 0
            explanations = []
            
            # Exact phrase matching
            exact_count = text.count(query_lower)
            if exact_count > 0:
                score += exact_count * 25
                explanations.append(f"'{query}' appears {exact_count} times")
            
            # Individual word matching
            word_matches = {}
            for word in query_words:
                word_count = text.count(word)
                if word_count > 0:
                    score += word_count * 5
                    word_matches[word] = word_count
                
                # Filename bonus
                if word in filename:
                    score += 20
                    explanations.append(f"'{word}' in filename")
            
            if word_matches:
                word_desc = ", ".join([f"'{w}': {c}" for w, c in word_matches.items()])
                explanations.append(f"Word matches: {word_desc}")
            
            if score > 0:
                normalized_score = min(score / 100.0, 1.0)
                
                result = SearchResult(
                    filename=doc.get('filename', f'Document {doc_idx+1}'),
                    content=doc.get('text', ''),
                    snippet=self._extract_snippet(doc.get('text', ''), query),
                    score=normalized_score,
                    rank=0,
                    keyword_score=normalized_score,
                    content_type=doc.get('content_type', 'general'),
                    source=doc.get('filename', ''),
                    explanation="; ".join(explanations[:2])
                )
                results.append(result)
        
        results.sort(key=lambda x: x.keyword_score, reverse=True)
        return results[:max_results]
    
    def _hybrid_search(self, query: str, doc_indices: List[int], max_results: int) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword approaches"""
        # Get results from both methods
        semantic_results = self._semantic_search(query, doc_indices, max_results * 2)
        keyword_results = self._keyword_search(query, doc_indices, max_results * 2)
        
        # Combine results
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            combined_results[result.filename] = result
        
        # Add/merge keyword results
        for result in keyword_results:
            if result.filename in combined_results:
                existing = combined_results[result.filename]
                existing.keyword_score = result.keyword_score
                existing.score = (existing.semantic_score * 0.6) + (result.keyword_score * 0.4)
                existing.explanation = f"Semantic: {existing.semantic_score:.3f}, Keyword: {result.keyword_score:.3f}"
            else:
                combined_results[result.filename] = result
        
        # Sort and return
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:max_results]
    
    def _extract_snippet(self, text: str, query: str, max_length: int = 400) -> str:
        """Extract relevant snippet with highlighting"""
        if not text or not query:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find best position
        pos = text_lower.find(query_lower)
        if pos == -1:
            for word in query_lower.split():
                pos = text_lower.find(word)
                if pos != -1:
                    break
        
        if pos == -1:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Extract snippet
        start = max(0, pos - max_length // 3)
        end = min(len(text), pos + len(query) + max_length * 2 // 3)
        snippet = text[start:end]
        
        # Highlight terms
        for word in query.split():
            if word.strip():
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                snippet = pattern.sub(f"**{word.upper()}**", snippet)
        
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        
        return prefix + snippet + suffix
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_documents': len(self.documents),
            'ai_enabled': self.model is not None,
            'index_built': self.document_embeddings is not None,
            'ai_available': AI_AVAILABLE
        }
        
        if self.documents:
            content_types = {}
            for doc in self.documents:
                content_type = doc.get('content_type', 'general')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            stats['content_types'] = content_types
        
        return stats

# UI Functions
def initialize_session_state():
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = AdvancedSearchEngine()
    if 'processor' not in st.session_state:
        st.session_state.processor = AdvancedDocumentProcessor()
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'search_results' not in st.session_state:
        st.session_state.search_results = {}

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        stats = st.session_state.search_engine.get_stats()
        
        if stats['ai_available'] and stats['ai_enabled']:
            st.success("🤖 AI Search Ready")
        else:
            st.error("🤖 AI Unavailable")
        
        st.metric("Documents", stats['total_documents'])
        
        if stats.get('content_types'):
            st.markdown("### 📊 Content Types")
            for content_type, count in stats['content_types'].items():
                st.write(f"• {content_type.title()}: {count}")

def render_upload_section():
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md', 'py', 'js', 'html', 'css', 'json']
    )
    
    if uploaded_files:
        if st.button("🚀 Process Files", type="primary"):
            with st.spinner("Processing documents..."):
                new_documents = st.session_state.processor.process_files(uploaded_files)
                st.session_state.documents.extend(new_documents)
                st.session_state.search_engine.add_documents(st.session_state.documents)
                st.success(f"✅ Processed {len(new_documents)} documents")

def render_search_section():
    st.header("🔍 Advanced Search Engine")
    
    if not st.session_state.documents:
        st.warning("📝 Please upload documents first.")
        return
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g. 'patient safety recommendations', 'communication protocols'",
            key="search_query"
        )
    
    with col2:
        search_mode = st.selectbox(
            "Search Mode",
            ["Hybrid Search", "Semantic Search", "Keyword Search"],
            key="search_mode"
        )
    
    # Search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_type = st.selectbox(
            "Content Type",
            ["All Content", "Recommendations Only", "Responses Only", "Concerns Only", "Policies Only", "Training Only"],
            key="search_type"
        )
    
    with col2:
        max_results = st.slider("Max Results", 5, 25, 10, key="max_results")
    
    with col3:
        include_score = st.checkbox("Show Scores", value=True, key="show_scores")
    
    # Search button
    if st.button("🔍 Search", type="primary") and query:
        perform_search(query, search_type, search_mode, max_results, include_score)
    
    # Display results
    display_search_results()
    
    # Search history
    render_search_history()

def perform_search(query: str, search_type: str, search_mode: str, max_results: int, include_score: bool):
    """Perform search and store results"""
    with st.spinner(f"Searching {search_type.lower()} using {search_mode.lower()}..."):
        try:
            results = st.session_state.search_engine.search(
                query, search_type, max_results, search_mode
            )
            
            # Store results
            st.session_state.search_results[query] = results
            st.session_state.last_search_query = query
            
            # Add to history
            st.session_state.search_history.insert(0, {
                'query': query,
                'type': search_type,
                'mode': search_mode,
                'results_count': len(results),
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10 searches
            st.session_state.search_history = st.session_state.search_history[:10]
            
            if results:
                st.success(f"✅ Found {len(results)} results")
            else:
                st.warning("No results found. Try different keywords or search type.")
                
        except Exception as e:
            st.error(f"Search failed: {e}")

def display_search_results():
    """Display search results with advanced features"""
    if not st.session_state.get('search_results'):
        return
    
    last_query = st.session_state.get('last_search_query')
    if not last_query or last_query not in st.session_state.search_results:
        return
    
    results = st.session_state.search_results[last_query]
    
    if not results:
        return
    
    st.subheader(f"📋 Search Results ({len(results)} found)")
    
    # Results summary
    if len(results) > 1:
        scores = [r.score for r in results]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Score", f"{max(scores):.3f}")
        with col2:
            st.metric("Average", f"{sum(scores)/len(scores):.3f}")
        with col3:
            content_types = [r.content_type for r in results]
            most_common = Counter(content_types).most_common(1)[0][0] if content_types else "Mixed"
            st.metric("Top Type", most_common.title())
    
    # Display results
    for i, result in enumerate(results):
        result.rank = i + 1
        
        with st.expander(f"#{result.rank} - {result.filename} (Score: {result.score:.3f})"):
            
            # Score breakdown
            if getattr(result, 'semantic_score', 0) > 0 or getattr(result, 'keyword_score', 0) > 0:
                score_cols = st.columns(3)
                with score_cols[0]:
                    semantic = getattr(result, 'semantic_score', 0)
                    st.metric("Semantic", f"{semantic:.3f}" if semantic > 0 else "N/A")
                with score_cols[1]:
                    keyword = getattr(result, 'keyword_score', 0)
                    st.metric("Keyword", f"{keyword:.3f}" if keyword > 0 else "N/A")
                with score_cols[2]:
                    st.metric("Content Type", result.content_type.title())
            
            # Explanation
            explanation = getattr(result, 'explanation', '')
            if explanation:
                st.info(f"💡 **Match details:** {explanation}")
            
            # Content preview
            st.markdown("**Preview:**")
            st.markdown(result.snippet)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📖 Full Content", key=f"show_{result.rank}"):
                    st.text_area("Full content", value=result.content, height=300, key=f"content_{result.rank}")
            
            with col2:
                if st.button(f"🔍 Similar", key=f"similar_{result.rank}"):
                    find_similar_content(result.content)
            
            with col3:
                if st.button(f"📊 Analyze", key=f"analyze_{result.rank}"):
                    analyze_content(result.content)

def find_similar_content(content: str):
    """Find similar content"""
    # Use first 200 characters as query
    query = content[:200]
    
    with st.spinner("Finding similar content..."):
        results = st.session_state.search_engine.search(query, "All Content", 3, "Semantic Search")
        
        if results:
            st.subheader("🔗 Similar Content:")
            for i, result in enumerate(results):
                with st.expander(f"Similar {i+1} - {result.filename}"):
                    st.markdown(result.snippet)
        else:
            st.info("No similar content found")

def analyze_content(content: str):
    """Basic content analysis"""
    st.subheader("📊 Content Analysis")
    
    words = content.split()
    sentences = re.split(r'[.!?]+', content)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", len(words))
    with col2:
        st.metric("Characters", len(content))
    with col3:
        st.metric("Sentences", len([s for s in sentences if s.strip()]))
    
    # Extract key terms
    word_freq = Counter([word.lower().strip('.,!?()[]') for word in words if len(word) > 3])
    common_words = {'the', 'and', 'that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use'}
    
    filtered_words = {word: count for word, count in word_freq.items() if word not in common_words}
    top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if top_words:
        st.subheader("🔤 Key Terms:")
        for word, count in top_words:
            st.write(f"• **{word}**: {count} times")

def render_search_history():
    """Render search history"""
    if not st.session_state.get('search_history'):
        return
    
    with st.expander("📚 Search History"):
        st.markdown("**Recent Searches:**")
        
        for i, search in enumerate(st.session_state.search_history):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                st.text(search['query'])
            with col2:
                st.caption(search['type'])
            with col3:
                st.caption(search['mode'])
            with col4:
                st.caption(f"{search['results_count']} results")
            with col5:
                if st.button("🔄", key=f"repeat_{i}", help="Repeat search"):
                    st.session_state.search_query = search['query']
                    st.rerun()

def main():
    st.set_page_config(
        page_title="Advanced Document Search",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🔍 Advanced Document Search Engine")
    st.markdown("**Professional document search with AI semantic understanding and advanced filtering**")
    
    render_sidebar()
    
    tab1, tab2 = st.tabs(["📁 Upload Documents", "🔍 Advanced Search"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_search_section()

if __name__ == "__main__":
    main()
