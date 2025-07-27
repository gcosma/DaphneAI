# modules/search/rag_search.py
# RAG (Retrieval-Augmented Generation) Search Engine

import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    import torch
    RAG_DEPENDENCIES_AVAILABLE = True
except ImportError:
    RAG_DEPENDENCIES_AVAILABLE = False
    logging.warning("RAG dependencies not available. Install: pip install sentence-transformers torch")

class RAGSearchEngine:
    """Advanced semantic search using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.model = None
        self.document_embeddings = None
        self.document_chunks = []
        self.chunk_size = 500  # words per chunk
        self.overlap = 50  # word overlap between chunks
        
        if RAG_DEPENDENCIES_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.error("RAG dependencies not available")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            with st.spinner("Loading AI model for semantic search..."):
                self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"RAG model {self.model_name} loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load RAG model: {e}")
            self.model = None
    
    def _chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks for better semantic search"""
        chunks = []
        
        for doc in documents:
            if 'text' not in doc or not doc['text']:
                continue
            
            text = doc['text']
            words = text.split()
            
            # Create overlapping chunks
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunk = {
                    'text': chunk_text,
                    'document': doc,
                    'chunk_index': len(chunks),
                    'start_word': i,
                    'end_word': i + len(chunk_words)
                }
                chunks.append(chunk)
        
        return chunks
    
    def _build_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Build semantic index for documents"""
        if not self.model:
            return False
        
        try:
            # Chunk documents
            self.document_chunks = self._chunk_documents(documents)
            
            if not self.document_chunks:
                self.logger.warning("No text chunks found in documents")
                return False
            
            # Create embeddings
            chunk_texts = [chunk['text'] for chunk in self.document_chunks]
            
            with st.spinner(f"Building semantic index for {len(chunk_texts)} text chunks..."):
                self.document_embeddings = self.model.encode(
                    chunk_texts,
                    convert_to_tensor=True,
                    show_progress_bar=True
                )
            
            self.logger.info(f"Built semantic index with {len(chunk_texts)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build semantic index: {e}")
            return False
    
    def search(self, query: str, documents: List[Dict[str, Any]], max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using RAG"""
        if not RAG_DEPENDENCIES_AVAILABLE:
            st.error("RAG search not available. Install: pip install sentence-transformers torch")
            return []
        
        if not self.model:
            st.error("RAG model not loaded")
            return []
        
        # Build index if needed
        if self.document_embeddings is None or len(self.document_chunks) != sum(1 for doc in documents if doc.get('text')):
            if not self._build_index(documents):
                return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(query_embedding, self.document_embeddings)
            
            # Get top results
            top_indices = torch.argsort(similarities, descending=True)[:max_results * 2]  # Get more for deduplication
            
            # Process results
            results = []
            seen_documents = set()
            
            for idx in top_indices:
                chunk = self.document_chunks[idx.item()]
                similarity = similarities[idx.item()].item()
                
                # Skip very low similarity results
                if similarity < 0.1:
                    continue
                
                doc_filename = chunk['document']['filename']
                
                # Deduplicate by document (keep highest scoring chunk per document)
                if doc_filename in seen_documents:
                    continue
                seen_documents.add(doc_filename)
                
                result = {
                    'document': chunk['document'],
                    'similarity': similarity,
                    'snippet': self._extract_snippet(chunk, query),
                    'chunk_info': {
                        'start_word': chunk['start_word'],
                        'end_word': chunk['end_word'],
                        'chunk_index': chunk['chunk_index']
                    },
                    'search_type': 'rag_semantic'
                }
                
                results.append(result)
                
                if len(results) >= max_results:
                    break
            
            self.logger.info(f"RAG search returned {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            self.logger.error(f"RAG search failed: {e}")
            st.error(f"Semantic search error: {str(e)}")
            return []
    
    def _extract_snippet(self, chunk: Dict[str, Any], query: str, context_words: int = 50) -> str:
        """Extract relevant snippet from chunk"""
        text = chunk['text']
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find query in text
        best_pos = 0
        best_score = 0
        
        # Look for exact phrase match
        if query_lower in text_lower:
            best_pos = text_lower.find(query_lower)
        else:
            # Look for individual words
            query_words = query_lower.split()
            for word in query_words:
                pos = text_lower.find(word)
                if pos != -1:
                    # Score based on how many query words are nearby
                    local_score = sum(1 for qw in query_words if qw in text_lower[max(0, pos-100):pos+100])
                    if local_score > best_score:
                        best_score = local_score
                        best_pos = pos
        
        # Extract snippet around best position
        words = text.split()
        word_pos = len(text[:best_pos].split())
        
        start_word = max(0, word_pos - context_words)
        end_word = min(len(words), word_pos + context_words)
        
        snippet_words = words[start_word:end_word]
        snippet = ' '.join(snippet_words)
        
        # Add ellipsis if truncated
        if start_word > 0:
            snippet = "..." + snippet
        if end_word < len(words):
            snippet = snippet + "..."
        
        return snippet
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {'status': 'not_loaded', 'model_name': self.model_name}
        
        return {
            'status': 'loaded',
            'model_name': self.model_name,
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'max_sequence_length': self.model.max_seq_length,
            'chunks_indexed': len(self.document_chunks),
            'dependencies_available': RAG_DEPENDENCIES_AVAILABLE
        }
