# modules/ui/search_components.py
# Advanced Search Engine with RAG + Smart Search

import streamlit as st
import pandas as pd
import re
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from modules.core_utils import log_action

logger = logging.getLogger(__name__)

# Check for RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

class AdvancedSearchEngine:
    """High-performance search engine with multiple search modes"""
    
    def __init__(self):
        self.search_history = []
        self.performance_stats = {}
        
    def search(self, query: str, documents: List[Dict], search_mode: str, 
               max_results: int = 20, filters: Dict = None) -> Dict:
        """Main search interface with performance tracking"""
        start_time = time.time()
        
        # Choose search method
        if search_mode == "rag_semantic" and RAG_AVAILABLE:
            results = self._rag_search(query, documents, max_results, filters)
        elif search_mode == "hybrid":
            results = self._hybrid_search(query, documents, max_results, filters)
        elif search_mode == "fuzzy":
            results = self._fuzzy_search(query, documents, max_results, filters)
        else:  # smart_pattern
            results = self._smart_pattern_search(query, documents, max_results, filters)
        
        # Performance tracking
        search_time = time.time() - start_time
        
        search_result = {
            'results': results,
            'query': query,
            'search_mode': search_mode,
            'total_found': len(results),
            'search_time': search_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to history
        self._add_to_history(search_result)
        
        return search_result
    
    def _rag_search(self, query: str, documents: List[Dict], max_results: int, filters: Dict) -> List[Dict]:
        """RAG semantic search with embeddings"""
        if 'rag_engine' not in st.session_state:
            st.session_state.rag_engine = RAGSearchEngine()
        
        rag_engine = st.session_state.rag_engine
        
        # Initialize and index documents if needed
        if not rag_engine.is_indexed(documents):
            with st.spinner("🤖 Building semantic index..."):
                rag_engine.index_documents(documents)
        
        return rag_engine.semantic_search(query, max_results, filters)
    
    def _smart_pattern_search(self, query: str, documents: List[Dict], max_results: int, filters: Dict) -> List[Dict]:
        """Advanced pattern-based search"""
        results = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        for doc in documents:
            if filters and not self._apply_filters(doc, filters):
                continue
                
            score_data = self._calculate_advanced_score(query, query_words, doc)
            
            if score_data['total_score'] > 0:
                results.append({
                    'document': doc,
                    'score': score_data['total_score'],
                    'score_breakdown': score_data,
                    'match_type': self._classify_match(score_data['total_score']),
                    'highlights': self._extract_highlights(query_lower, doc['text']),
                    'relevance_reason': self._explain_relevance(score_data)
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:max_results]
    
    def _hybrid_search(self, query: str, documents: List[Dict], max_results: int, filters: Dict) -> List[Dict]:
        """Combine RAG and pattern search for best results"""
        if not RAG_AVAILABLE:
            return self._smart_pattern_search(query, documents, max_results, filters)
        
        # Get results from both methods
        rag_results = self._rag_search(query, documents, max_results * 2, filters)
        pattern_results = self._smart_pattern_search(query, documents, max_results * 2, filters)
        
        # Merge and re-rank results
        combined_results = self._merge_search_results(rag_results, pattern_results)
        
        return combined_results[:max_results]
    
    def _fuzzy_search(self, query: str, documents: List[Dict], max_results: int, filters: Dict) -> List[Dict]:
        """Fuzzy search for typo tolerance"""
        results = []
        
        # Use TF-IDF for fuzzy matching
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
            doc_texts = [doc['text'] for doc in documents]
            tfidf_matrix = vectorizer.fit_transform(doc_texts + [query])
            
            # Calculate similarities
            query_vector = tfidf_matrix[-1]
            similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
            
            for i, doc in enumerate(documents):
                if filters and not self._apply_filters(doc, filters):
                    continue
                    
                similarity = similarities[i]
                if similarity > 0.05:  # Minimum threshold
                    results.append({
                        'document': doc,
                        'score': similarity,
                        'match_type': f"Fuzzy Match ({similarity:.2f})",
                        'highlights': self._extract_highlights(query.lower(), doc['text'])
                    })
            
            return sorted(results, key=lambda x: x['score'], reverse=True)[:max_results]
            
        except Exception as e:
            logger.error(f"Fuzzy search error: {e}")
            return self._smart_pattern_search(query, documents, max_results, filters)
    
    def _calculate_advanced_score(self, query: str, query_words: set, doc: Dict) -> Dict:
        """Advanced scoring with detailed breakdown"""
        text = doc['text'].lower()
        
        scores = {
            'exact_phrase': 0.0,
            'word_matches': 0.0,
            'key_terms': 0.0,
            'position_bonus': 0.0,
            'frequency_bonus': 0.0,
            'title_match': 0.0
        }
        
        # Exact phrase match (highest value)
        if query.lower() in text:
            scores['exact_phrase'] = 2.0
            # Position bonus - earlier matches score higher
            position = text.find(query.lower()) / len(text)
            scores['position_bonus'] = (1.0 - position) * 0.5
        
        # Word matches with frequency weighting
        text_words = re.findall(r'\b\w+\b', text)
        word_freq = {}
        for word in text_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        matched_words = query_words & set(text_words)
        if query_words:
            scores['word_matches'] = len(matched_words) / len(query_words) * 1.5
            
            # Frequency bonus for important words
            for word in matched_words:
                freq = word_freq.get(word, 0)
                if freq > 1:
                    scores['frequency_bonus'] += min(freq * 0.1, 0.5)
        
        # Key terms boost
        key_terms = {
            'recommend', 'implementation', 'accept', 'reject', 'consider',
            'policy', 'government', 'minister', 'department', 'action',
            'urgent', 'critical', 'important', 'priority'
        }
        
        for term in key_terms:
            if term in query.lower() and term in text:
                scores['key_terms'] += 0.2
        
        # Title/filename relevance
        filename = doc.get('filename', '').lower()
        if any(word in filename for word in query_words):
            scores['title_match'] = 0.3
        
        scores['total_score'] = sum(scores.values())
        return scores
    
    def _extract_highlights(self, query: str, text: str, context_length: int = 150) -> List[str]:
        """Extract highlighted text snippets"""
        highlights = []
        text_lower = text.lower()
        
        # Find exact phrase matches
        if query in text_lower:
            start = text_lower.find(query)
            snippet_start = max(0, start - context_length // 2)
            snippet_end = min(len(text), start + len(query) + context_length // 2)
            
            snippet = text[snippet_start:snippet_end]
            # Highlight the query
            highlighted = re.sub(
                re.escape(query), 
                f"**{query}**", 
                snippet, 
                flags=re.IGNORECASE
            )
            highlights.append(f"...{highlighted}...")
        
        # Find word matches if no exact phrase
        elif not highlights:
            query_words = query.split()
            for word in query_words:
                if word in text_lower:
                    start = text_lower.find(word)
                    snippet_start = max(0, start - context_length // 2)
                    snippet_end = min(len(text), start + len(word) + context_length // 2)
                    
                    snippet = text[snippet_start:snippet_end]
                    highlighted = re.sub(
                        re.escape(word), 
                        f"**{word}**", 
                        snippet, 
                        flags=re.IGNORECASE
                    )
                    highlights.append(f"...{highlighted}...")
                    break
        
        return highlights[:2]  # Limit to 2 highlights
    
    def _apply_filters(self, doc: Dict, filters: Dict) -> bool:
        """Apply search filters"""
        if not filters:
            return True
        
        # File type filter
        if 'file_type' in filters and filters['file_type']:
            if doc.get('file_type', '').lower() != filters['file_type'].lower():
                return False
        
        # Word count filter
        if 'min_words' in filters and filters['min_words']:
            if doc.get('word_count', 0) < filters['min_words']:
                return False
        
        if 'max_words' in filters and filters['max_words']:
            if doc.get('word_count', 0) > filters['max_words']:
                return False
        
        return True
    
    def _classify_match(self, score: float) -> str:
        """Classify match quality"""
        if score >= 2.5:
            return "🎯 Excellent Match"
        elif score >= 1.5:
            return "✅ Very Good Match"
        elif score >= 1.0:
            return "👍 Good Match"
        elif score >= 0.5:
            return "🔍 Fair Match"
        else:
            return "❓ Weak Match"
    
    def _explain_relevance(self, score_data: Dict) -> str:
        """Generate explanation for why this result is relevant"""
        explanations = []
        
        if score_data['exact_phrase'] > 0:
            explanations.append("Contains exact phrase")
        if score_data['word_matches'] > 0.5:
            explanations.append("High word match rate")
        if score_data['key_terms'] > 0:
            explanations.append("Contains key terms")
        if score_data['title_match'] > 0:
            explanations.append("Filename relevance")
        if score_data['position_bonus'] > 0.2:
            explanations.append("Early document position")
        
        return " • ".join(explanations) if explanations else "Basic relevance"
    
    def _merge_search_results(self, rag_results: List[Dict], pattern_results: List[Dict]) -> List[Dict]:
        """Intelligently merge RAG and pattern search results"""
        merged = {}
        
        # Add RAG results with boost
        for result in rag_results:
            doc_id = result['document']['filename']
            result['final_score'] = result.get('similarity', 0) * 1.2  # RAG boost
            merged[doc_id] = result
        
        # Add pattern results, combining scores if document already exists
        for result in pattern_results:
            doc_id = result['document']['filename']
            if doc_id in merged:
                # Combine scores
                merged[doc_id]['final_score'] += result['score'] * 0.8
                merged[doc_id]['match_type'] = "🔄 Hybrid Match"
            else:
                result['final_score'] = result['score']
                merged[doc_id] = result
        
        return sorted(merged.values(), key=lambda x: x['final_score'], reverse=True)
    
    def _add_to_history(self, search_result: Dict):
        """Add search to history with performance tracking"""
        self.search_history.append({
            'query': search_result['query'],
            'search_mode': search_result['search_mode'],
            'results_count': search_result['total_found'],
            'search_time': search_result['search_time'],
            'timestamp': search_result['timestamp']
        })
        
        # Keep last 100 searches
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]

class RAGSearchEngine:
    """Dedicated RAG engine for semantic search"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.document_embeddings = {}
        self.indexed_docs_hash = None
        
    def is_indexed(self, documents: List[Dict]) -> bool:
        """Check if documents are already indexed"""
        current_hash = hash(str([doc['filename'] for doc in documents]))
        return current_hash == self.indexed_docs_hash and bool(self.document_embeddings)
    
    def index_documents(self, documents: List[Dict]) -> bool:
        """Create semantic index of documents"""
        if not RAG_AVAILABLE:
            return False
        
        try:
            # Load model if needed
            if self.model is None:
                self.model = SentenceTransformer(self.model_name)
            
            # Create embeddings
            self.document_embeddings = {}
            
            for doc in documents:
                # Chunk large documents for better performance
                text_chunks = self._chunk_text(doc['text'])
                chunk_embeddings = self.model.encode(text_chunks)
                
                self.document_embeddings[doc['filename']] = {
                    'document': doc,
                    'chunks': text_chunks,
                    'embeddings': chunk_embeddings
                }
            
            # Store hash to track changes
            self.indexed_docs_hash = hash(str([doc['filename'] for doc in documents]))
            
            return True
            
        except Exception as e:
            logger.error(f"RAG indexing error: {e}")
            return False
    
    def semantic_search(self, query: str, max_results: int, filters: Dict) -> List[Dict]:
        """Perform semantic search"""
        if not self.model or not self.document_embeddings:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        results = []
        
        # Search through all documents
        for doc_id, doc_data in self.document_embeddings.items():
            document = doc_data['document']
            
            # Apply filters
            if filters and not self._apply_filters(document, filters):
                continue
            
            # Calculate best chunk similarity
            similarities = cosine_similarity(query_embedding, doc_data['embeddings'])[0]
            best_similarity = float(np.max(similarities))
            best_chunk_idx = int(np.argmax(similarities))
            
            if best_similarity > 0.1:  # Minimum threshold
                results.append({
                    'document': document,
                    'similarity': best_similarity,
                    'match_type': self._classify_semantic_match(best_similarity),
                    'best_chunk': doc_data['chunks'][best_chunk_idx],
                    'highlights': [f"...{doc_data['chunks'][best_chunk_idx][:200]}..."]
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:max_results]
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > start + chunk_size // 2:
                    chunk = text[start:start + last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def _classify_semantic_match(self, similarity: float) -> str:
        """Classify semantic match quality"""
        if similarity >= 0.85:
            return "🎯 Excellent Semantic Match"
        elif similarity >= 0.70:
            return "✅ Very Good Semantic Match"
        elif similarity >= 0.55:
            return "👍 Good Semantic Match"
        elif similarity >= 0.40:
            return "🔍 Fair Semantic Match"
        else:
            return "❓ Weak Semantic Match"
    
    def _apply_filters(self, doc: Dict, filters: Dict) -> bool:
        """Apply filters for RAG search"""
        # Same as main search engine
        if not filters:
            return True
        
        if 'file_type' in filters and filters['file_type']:
            if doc.get('file_type', '').lower() != filters['file_type'].lower():
                return False
        
        if 'min_words' in filters and filters['min_words']:
            if doc.get('word_count', 0) < filters['min_words']:
                return False
        
        if 'max_words' in filters and filters['max_words']:
            if doc.get('word_count', 0) > filters['max_words']:
                return False
        
        return True

def render_search_interface(documents: List[Dict]):
    """Advanced search interface"""
    st.header("🔍 Advanced Search Engine")
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = AdvancedSearchEngine()
    
    search_engine = st.session_state.search_engine
    
    # Search configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter your search query...",
            help="🔍 Smart tips: Use quotes for exact phrases, + for required words"
        )
    
    with col2:
        search_mode = st.selectbox(
            "Search Mode",
            [
                "🤖 RAG Semantic" if RAG_AVAILABLE else "🤖 RAG (Install Required)",
                "🔍 Smart Pattern",
                "🔄 Hybrid Search" if RAG_AVAILABLE else "🔄 Hybrid (Install Required)",
                "🎯 Fuzzy Search"
            ],
            help="Choose search algorithm"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_results = st.slider("Max Results", 5, 50, 15)
            
        with col2:
            file_type_filter = st.selectbox(
                "File Type",
                ["All", "PDF", "TXT", "DOCX"],
                help="Filter by document type"
            )
        
        with col3:
            word_range = st.slider(
                "Word Count Range",
                0, 10000, (0, 10000),
                help="Filter by document length"
            )
    
    # Build filters
    filters = {}
    if file_type_filter != "All":
        filters['file_type'] = file_type_filter.lower()
    if word_range[0] > 0:
        filters['min_words'] = word_range[0]
    if word_range[1] < 10000:
        filters['max_words'] = word_range[1]
    
    # Search execution
    if st.button("🚀 Search", type="primary") or search_query:
        if search_query.strip():
            # Determine search mode
            mode_map = {
                "🤖 RAG Semantic": "rag_semantic",
                "🔍 Smart Pattern": "smart_pattern", 
                "🔄 Hybrid Search": "hybrid",
                "🎯 Fuzzy Search": "fuzzy"
            }
            
            selected_mode = None
            for key in mode_map:
                if key in search_mode:
                    selected_mode = mode_map[key]
                    break
            
            selected_mode = selected_mode or "smart_pattern"
            
            # Check RAG availability
            if selected_mode in ["rag_semantic", "hybrid"] and not RAG_AVAILABLE:
                st.error("❌ RAG features require: `pip install sentence-transformers torch scikit-learn`")
                selected_mode = "smart_pattern"
            
            # Perform search
            with st.spinner(f"🔍 Searching with {search_mode}..."):
                search_result = search_engine.search(
                    search_query, documents, selected_mode, max_results, filters
                )
            
            # Display results
            render_search_results(search_result)
            
            # Log search
            log_action("advanced_search", {
                "query": search_query,
                "mode": selected_mode,
                "results": search_result['total_found'],
                "time": search_result['search_time']
            })
        
        else:
            st.warning("Please enter a search query")
    
    # Performance and history
    render_search_analytics(search_engine)

def render_search_results(search_result: Dict):
    """Display advanced search results"""
    results = search_result['results']
    query = search_result['query']
    
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    # Results header with performance info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Results Found", search_result['total_found'])
    with col2:
        st.metric("Search Time", f"{search_result['search_time']:.3f}s")
    with col3:
        st.metric("Search Mode", search_result['search_mode'].replace('_', ' ').title())
    
    st.markdown(f"### 🎯 Results for '{query}'")
    
    # Display options
    display_mode = st.radio(
        "View Mode",
        ["🃏 Cards", "📊 Table", "📝 Details"],
        horizontal=True
    )
    
    if display_mode == "📊 Table":
        # Compact table view
        table_data = []
        for result in results:
            doc = result['document']
            score_key = 'similarity' if 'similarity' in result else 'score'
            
            table_data.append({
                'Document': doc['filename'],
                'Score': f"{result[score_key]:.3f}",
                'Match Type': result['match_type'],
                'Words': doc.get('word_count', 0),
                'Type': doc.get('file_type', '').upper()
            })
        
        st.dataframe(table_data, use_container_width=True)
    
    else:
        # Card view
        for i, result in enumerate(results, 1):
            doc = result['document']
            score_key = 'similarity' if 'similarity' in result else 'score'
            score = result[score_key]
            
            with st.container():
                # Card header
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{i}. {doc['filename']}**")
                with col2:
                    st.metric("Score", f"{score:.3f}")
                with col3:
                    st.caption(result['match_type'])
                with col4:
                    st.caption(f"{doc.get('word_count', 0)} words")
                
                # Relevance explanation
                if 'relevance_reason' in result:
                    st.caption(f"🔍 {result['relevance_reason']}")
                
                # Content preview and highlights
                if display_mode == "📝 Details":
                    # Show full document preview
                    preview_text = doc['text'][:800] + "..." if len(doc['text']) > 800 else doc['text']
                    st.text_area("Content Preview", preview_text, height=120, key=f"preview_{i}")
                
                # Show highlights
                if result.get('highlights'):
                    st.markdown("**📍 Relevant Excerpts:**")
                    for highlight in result['highlights']:
                        st.markdown(f"> {highlight}")
                
                # Score breakdown for advanced users
                if 'score_breakdown' in result and display_mode == "📝 Details":
                    with st.expander("📊 Score Breakdown"):
                        breakdown = result['score_breakdown']
                        for component, value in breakdown.items():
                            if value > 0:
                                st.write(f"• {component.replace('_', ' ').title()}: {value:.3f}")
                
                st.markdown("---")

def render_search_analytics(search_engine: AdvancedSearchEngine):
    """Display search analytics and history"""
    if st.checkbox("📊 Show Analytics & History"):
        tab1, tab2 = st.tabs(["📈 Performance", "📝 Search History"])
        
        with tab1:
            # Performance metrics
            if search_engine.search_history:
                df = pd.DataFrame(search_engine.search_history)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⚡ Performance Metrics")
                    avg_time = df['search_time'].mean()
                    st.metric("Average Search Time", f"{avg_time:.3f}s")
                    st.metric("Total Searches", len(df))
                    
                    # Search mode distribution
                    mode_counts = df['search_mode'].value_counts()
                    st.bar_chart(mode_counts)
                
                with col2:
                    st.subheader("📊 Search Times")
                    st.line_chart(df.set_index('timestamp')['search_time'])
                    
                    # Results distribution
                    st.subheader("🎯 Results Found")
                    st.bar_chart(df['results_count'])
        
        with tab2:
            # Search history
            if search_engine.search_history:
                st.subheader("📝 Recent Searches")
                
                history_df = pd.DataFrame(search_engine.search_history)
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M:%S')
                
                # Show recent searches
                display_df = history_df[['timestamp', 'query', 'search_mode', 'results_count', 'search_time']].tail(20)
                display_df.columns = ['Time', 'Query', 'Mode', 'Results', 'Time (s)']
                
                st.dataframe(display_df, use_container_width=True)
                
                # Quick re-search buttons
                st.subheader("🔄 Quick Re-search")
                recent_queries = history_df['query'].tail(5).unique()
                
                cols = st.columns(min(len(recent_queries), 3))
                for i, query in enumerate(recent_queries):
                    if i < len(cols):
                        if cols[i].button(f"🔍 '{query[:20]}...'", key=f"rerun_{i}"):
                            st.session_state.quick_search = query
                            st.rerun()
                
                # Handle quick search
                if hasattr(st.session_state, 'quick_search'):
                    st.info(f"Searching for: {st.session_state.quick_search}")
                    del st.session_state.quick_search
            
            else:
                st.info("No search history yet. Perform some searches to see analytics!")
