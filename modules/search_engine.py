"""
AI-Based Smart Document Search Engine
Single unified search with intelligent ranking
"""

import time
import logging
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
import re

# Optional imports for AI features
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

@dataclass
class SearchResult:
    """Structured search result"""
    filename: str
    content: str
    snippet: str
    score: float
    rank: int

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
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
        
        Returns:
            List of SearchResult objects, ranked by relevance
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
