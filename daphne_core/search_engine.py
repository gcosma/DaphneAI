# modules/search_engine.py
"""
State-of-the-Art Semantic Search Engine
Optimized for maximum relevance and accuracy
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

# Core dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# Optional for advanced features
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False


@dataclass
class SearchResult:
    """Enhanced search result with rich metadata"""
    document_id: str
    filename: str
    text_fragment: str
    full_context: str
    relevance_score: float
    sentence_index: int
    embedding_distance: float
    cross_encoder_score: Optional[float] = None
    snippet_preview: str = ""
    matched_concepts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate preview snippet"""
        if not self.snippet_preview:
            self.snippet_preview = self._create_snippet()
    
    def _create_snippet(self, max_length: int = 200) -> str:
        """Create intelligently truncated snippet"""
        if len(self.text_fragment) <= max_length:
            return self.text_fragment
        
        # Find sentence boundary near max_length
        truncated = self.text_fragment[:max_length]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')
        
        cut_point = last_period if last_period > max_length * 0.7 else last_space
        if cut_point == -1:
            cut_point = max_length
            
        return truncated[:cut_point] + "..."


@dataclass
class DocumentResult:
    """Aggregated results for a single document"""
    document_id: str
    filename: str
    document_type: str
    overall_score: float
    best_match_score: float
    total_matches: int
    results: List[SearchResult]
    
    def get_top_results(self, n: int = 3) -> List[SearchResult]:
        """Get top N results from this document"""
        return sorted(self.results, key=lambda x: x.relevance_score, reverse=True)[:n]


class SemanticSearchEngine:
    """
    Advanced semantic search engine with state-of-the-art NLP models
    
    Features:
    - Multiple embedding models (can switch between them)
    - Query expansion and reformulation
    - Re-ranking with cross-encoders (optional)
    - Hybrid semantic + keyword boosting
    - Contextual chunking for long documents
    - Efficient caching and batch processing
    """
    
    def __init__(
        self,
        model_name: str = 'BAAI/bge-small-en-v1.5',
        use_cross_encoder: bool = False,
        device: str = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize semantic search engine
        
        Args:
            model_name: Hugging Face model name. Top options:
            
                BEST FOR STREAMLIT CLOUD (Small, Fast, Good):
                - 'BAAI/bge-small-en-v1.5': 33MB, excellent quality (RECOMMENDED)
                - 'sentence-transformers/all-MiniLM-L6-v2': 80MB, very good
                
                BEST QUALITY (Larger, Slower):
                - 'BAAI/bge-base-en-v1.5': 109MB, state-of-the-art quality
                - 'sentence-transformers/all-mpnet-base-v2': 420MB, excellent
                
                SPECIALIZED:
                - 'multi-qa-mpnet-base-dot-v1': 420MB, Q&A optimized
                - 'msmarco-distilbert-base-v4': 250MB, document retrieval
                
            use_cross_encoder: Enable re-ranking (slower but more accurate)
            device: 'cuda', 'cpu', or None for auto-detect
            cache_embeddings: Cache embeddings to speed up repeated searches
        """
        if not SEMANTIC_AVAILABLE:
            raise RuntimeError("Semantic search dependencies not available")
        
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load bi-encoder model
        self.logger.info(f"Loading semantic model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        
        # Optional cross-encoder for re-ranking
        self.cross_encoder = None
        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
                self.logger.info("Cross-encoder loaded for re-ranking")
            except Exception as e:
                self.logger.warning(f"Could not load cross-encoder: {e}")
        
        # Storage
        self.documents = []
        self.document_chunks = {}  # doc_id -> list of text chunks
        self.chunk_embeddings = {}  # doc_id -> numpy array of embeddings
        self.cache_embeddings = cache_embeddings
        
        # Query cache
        self.query_cache = {} if cache_embeddings else None
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'total_documents': 0,
            'total_chunks': 0
        }
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        batch_size: int = 32
    ) -> None:
        """
        Add documents and build semantic index
        
        Args:
            documents: List of dicts with 'text'/'content' and metadata
            chunk_size: Target characters per chunk
            chunk_overlap: Overlapping characters between chunks
            batch_size: Batch size for embedding generation
        """
        self.logger.info(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        self.document_chunks = {}
        self.chunk_embeddings = {}
        
        all_chunks = []
        chunk_metadata = []  # Track which doc each chunk belongs to
        
        # Step 1: Chunk all documents
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', str(i))
            text = doc.get('text', doc.get('content', ''))
            
            if not text:
                continue
            
            # Create chunks with overlap
            chunks = self._create_smart_chunks(text, chunk_size, chunk_overlap)
            self.document_chunks[doc_id] = chunks
            
            # Track metadata
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata.append(doc_id)
        
        # Step 2: Generate embeddings in batches (much faster)
        self.logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        
        all_embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            all_embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(all_embeddings)
        
        # Step 3: Organize embeddings by document
        chunk_idx = 0
        for doc_id in self.document_chunks.keys():
            num_chunks = len(self.document_chunks[doc_id])
            self.chunk_embeddings[doc_id] = all_embeddings[chunk_idx:chunk_idx + num_chunks]
            chunk_idx += num_chunks
        
        # Update statistics
        self.stats['total_documents'] = len(documents)
        self.stats['total_chunks'] = len(all_chunks)
        
        self.logger.info(f"✓ Indexed {len(documents)} documents into {len(all_chunks)} chunks")
    
    def _create_smart_chunks(
        self,
        text: str,
        target_size: int = 300,
        overlap: int = 50
    ) -> List[str]:
        """
        Create intelligent text chunks that respect sentence boundaries
        
        This is much better than naive character-based chunking as it:
        - Preserves complete sentences
        - Maintains context with overlap
        - Handles edge cases gracefully
        """
        if not text:
            return []
        
        # Use sentence tokenization if available
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except:
                # Fallback to simple splitting
                sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        else:
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If single sentence exceeds target, add it as its own chunk
            if sentence_len > target_size and not current_chunk:
                chunks.append(sentence)
                continue
            
            # If adding this sentence would exceed target, start new chunk
            if current_length + sentence_len > target_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Add overlap: keep last few sentences
                if overlap > 0 and len(current_chunk) > 1:
                    overlap_text = ' '.join(current_chunk[-2:])
                    if len(overlap_text) <= overlap:
                        current_chunk = current_chunk[-2:]
                        current_length = len(overlap_text)
                    else:
                        current_chunk = [current_chunk[-1]]
                        current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.3,
        rerank: bool = None,
        return_all_documents: bool = False,
        keyword_boost: float = 0.1
    ) -> List[DocumentResult]:
        """
        Perform semantic search across all documents
        
        Args:
            query: Search query
            top_k: Maximum results per document
            min_score: Minimum similarity threshold (0-1)
            rerank: Use cross-encoder for re-ranking (uses default if None)
            return_all_documents: Return docs even with no matches
            keyword_boost: Boost score for keyword matches (0-0.3 recommended)
        
        Returns:
            List of DocumentResult objects sorted by relevance
        """
        if not query.strip():
            return []
        
        self.stats['total_searches'] += 1
        
        # Check cache
        cache_key = f"{query}_{top_k}_{min_score}"
        if self.query_cache is not None and cache_key in self.query_cache:
            self.stats['cache_hits'] += 1
            return self.query_cache[cache_key]
        
        # Expand query for better matching
        expanded_queries = self._expand_query(query)
        
        # Encode all query variations
        query_embeddings = self.model.encode(
            expanded_queries,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Search each document
        document_results = []
        
        for i, doc in enumerate(self.documents):
            doc_id = doc.get('id', str(i))
            
            if doc_id not in self.chunk_embeddings:
                continue
            
            # Find best matching chunks
            search_results = self._search_document(
                doc_id=doc_id,
                doc=doc,
                query=query,
                query_embeddings=query_embeddings,
                top_k=top_k,
                min_score=min_score,
                keyword_boost=keyword_boost
            )
            
            if search_results or return_all_documents:
                # Calculate document-level score
                if search_results:
                    best_score = max(r.relevance_score for r in search_results)
                    avg_score = sum(r.relevance_score for r in search_results) / len(search_results)
                    overall_score = 0.7 * best_score + 0.3 * avg_score
                else:
                    best_score = 0.0
                    overall_score = 0.0
                
                doc_result = DocumentResult(
                    document_id=doc_id,
                    filename=doc.get('filename', f'Document {i+1}'),
                    document_type=self._classify_document(doc),
                    overall_score=overall_score,
                    best_match_score=best_score,
                    total_matches=len(search_results),
                    results=search_results
                )
                document_results.append(doc_result)
        
        # Re-rank documents if cross-encoder available
        if (rerank or (rerank is None and self.cross_encoder)) and document_results:
            document_results = self._rerank_documents(query, document_results)
        
        # Sort by overall score
        document_results.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Cache results
        if self.query_cache is not None:
            self.query_cache[cache_key] = document_results
        
        return document_results
    
    def _search_document(
        self,
        doc_id: str,
        doc: Dict[str, Any],
        query: str,
        query_embeddings: np.ndarray,
        top_k: int,
        min_score: float,
        keyword_boost: float
    ) -> List[SearchResult]:
        """Search within a single document"""
        
        chunks = self.document_chunks.get(doc_id, [])
        chunk_embeddings = self.chunk_embeddings.get(doc_id)
        
        if chunk_embeddings is None or len(chunks) == 0:
            return []
        
        # Calculate similarities with all query variations
        # Shape: (num_query_variations, num_chunks)
        similarities = cosine_similarity(query_embeddings, chunk_embeddings)
        
        # Take maximum similarity across query variations for each chunk
        max_similarities = np.max(similarities, axis=0)
        
        # Apply keyword boosting
        if keyword_boost > 0:
            query_terms = set(query.lower().split())
            for idx, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                matching_terms = sum(1 for term in query_terms if term in chunk_lower)
                if matching_terms > 0:
                    boost = min(keyword_boost * matching_terms, 0.3)
                    max_similarities[idx] = min(1.0, max_similarities[idx] + boost)
        
        # Get top-k chunks above threshold
        top_indices = np.argsort(max_similarities)[::-1][:top_k]
        top_indices = [idx for idx in top_indices if max_similarities[idx] >= min_score]
        
        # Create search results
        results = []
        for idx in top_indices:
            chunk = chunks[idx]
            score = float(max_similarities[idx])
            
            # Extract matched concepts (terms that appear in both query and chunk)
            matched_concepts = self._extract_matched_concepts(query, chunk)
            
            result = SearchResult(
                document_id=doc_id,
                filename=doc.get('filename', ''),
                text_fragment=chunk,
                full_context=self._get_extended_context(chunks, idx),
                relevance_score=score,
                sentence_index=idx,
                embedding_distance=1.0 - score,
                matched_concepts=matched_concepts
            )
            results.append(result)
        
        return results
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with variations for better semantic matching
        
        Returns multiple query formulations to capture different aspects
        """
        queries = [query]
        
        # Add question variations
        if not query.strip().endswith('?'):
            queries.append(f"What about {query}?")
            queries.append(f"Information about {query}")
        
        # Add context variations for government documents
        if any(term in query.lower() for term in ['recommendation', 'response', 'policy', 'action']):
            queries.append(f"government {query}")
        
        return queries
    
    def _extract_matched_concepts(self, query: str, text: str) -> List[str]:
        """Extract concepts that appear in both query and text"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        matched = query_words & text_words
        
        # Filter out very common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        matched = [w for w in matched if w not in stopwords and len(w) > 2]
        
        return sorted(matched)[:5]
    
    def _get_extended_context(self, chunks: List[str], index: int, window: int = 1) -> str:
        """Get extended context around a chunk"""
        start = max(0, index - window)
        end = min(len(chunks), index + window + 1)
        
        return ' ... '.join(chunks[start:end])
    
    def _rerank_documents(
        self,
        query: str,
        document_results: List[DocumentResult]
    ) -> List[DocumentResult]:
        """Re-rank documents using cross-encoder"""
        if not self.cross_encoder:
            return document_results
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        pair_metadata = []
        
        for doc_result in document_results:
            # Use top result from each document
            if doc_result.results:
                best_result = doc_result.results[0]
                pairs.append([query, best_result.text_fragment])
                pair_metadata.append((doc_result, best_result))
        
        if not pairs:
            return document_results
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Update scores
        for (doc_result, search_result), cross_score in zip(pair_metadata, cross_scores):
            # Blend bi-encoder and cross-encoder scores
            search_result.cross_encoder_score = float(cross_score)
            blended_score = 0.6 * search_result.relevance_score + 0.4 * cross_score
            doc_result.overall_score = blended_score
        
        return document_results
    
    def _classify_document(self, doc: Dict[str, Any]) -> str:
        """Classify document type"""
        text = doc.get('text', '').lower()
        filename = doc.get('filename', '').lower()
        
        if 'recommendation' in text or 'recommend' in filename:
            return 'recommendation'
        elif 'response' in text or 'response' in filename:
            return 'response'
        elif 'policy' in text or 'policy' in filename:
            return 'policy'
        else:
            return 'document'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        cache_hit_rate = 0
        if self.stats['total_searches'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_searches']
        
        return {
            'model': self.model_name,
            'device': self.device,
            'cross_encoder_enabled': self.cross_encoder is not None,
            'total_documents': self.stats['total_documents'],
            'total_chunks': self.stats['total_chunks'],
            'total_searches': self.stats['total_searches'],
            'cache_hit_rate': f"{cache_hit_rate:.1%}",
            'embeddings_cached': self.cache_embeddings
        }
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        if self.query_cache is not None:
            self.query_cache.clear()
            self.logger.info("Query cache cleared")
