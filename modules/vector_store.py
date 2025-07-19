# ===============================================
# FILE: modules/vector_store.py
# ===============================================

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import atexit
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    TEXT_SPLITTER_AVAILABLE = True
except ImportError:
    TEXT_SPLITTER_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class OptimizedEmbeddings:
    """Optimized embeddings with caching and batch processing"""
    def __init__(self):
        self.cache = {}
        self.cache_lock = threading.Lock()
        try:
            if os.getenv("OPENAI_API_KEY"):
                from openai import OpenAI
                self.client = OpenAI()
                self.has_api = True
            else:
                self.has_api = False
        except:
            self.has_api = False
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query with caching"""
        text_hash = hash(text)
        
        with self.cache_lock:
            if text_hash in self.cache:
                return self.cache[text_hash]
        
        if self.has_api:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embedding = response.data[0].embedding
            except:
                embedding = self._mock_embedding(text)
        else:
            embedding = self._mock_embedding(text)
        
        with self.cache_lock:
            self.cache[text_hash] = embedding
            # Limit cache size
            if len(self.cache) > 1000:
                self.cache.pop(next(iter(self.cache)))
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents efficiently"""
        if not texts:
            return []
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        with self.cache_lock:
            for i, text in enumerate(texts):
                text_hash = hash(text)
                if text_hash in self.cache:
                    embeddings.append(self.cache[text_hash])
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        
        # Process uncached texts in batches
        if uncached_texts:
            batch_size = 20  # Optimal batch size for OpenAI API
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                batch_embeddings = self._batch_embed(batch)
                
                # Update cache and results
                with self.cache_lock:
                    for j, emb in enumerate(batch_embeddings):
                        idx = uncached_indices[i + j]
                        embeddings[idx] = emb
                        self.cache[hash(uncached_texts[i + j])] = emb
        
        return [emb for emb in embeddings if emb is not None]
    
    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with API or fallback"""
        if self.has_api:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
                return [data.embedding for data in response.data]
            except:
                pass
        
        # Fallback to mock embeddings
        return [self._mock_embedding(text) for text in texts]
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding"""
        import hashlib
        hash_object = hashlib.md5(text.encode())
        hex_dig = hash_object.hexdigest()
        
        # Create 1536-dimensional embedding (OpenAI's dimension)
        embedding = []
        for i in range(0, min(len(hex_dig), 32), 2):
            val = int(hex_dig[i:i+2], 16) / 255.0
            embedding.extend([val] * 48)  # Repeat to reach 1536 dimensions
        
        # Pad or truncate to exactly 1536 dimensions
        while len(embedding) < 1536:
            embedding.append(0.5)
        
        return embedding[:1536]

class FastVectorStore:
    """High-performance vector store with FAISS backend"""
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.documents = []
        self.embeddings = []
        self.index = None
        self.index_lock = threading.Lock()
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
    def add_documents(self, docs: List[Document], embeddings: List[List[float]]):
        """Add documents with their embeddings"""
        with self.index_lock:
            start_idx = len(self.documents)
            self.documents.extend(docs)
            self.embeddings.extend(embeddings)
            
            if self.index and embeddings:
                # Normalize embeddings for cosine similarity
                emb_array = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(emb_array)
                self.index.add(emb_array)
    
    def similarity_search(self, query: str, k: int = 5, embedding_func=None) -> List[Document]:
        """Fast similarity search"""
        if not self.documents:
            return []
        
        if embedding_func:
            query_emb = embedding_func(query)
        else:
            return self.documents[:k]  # Fallback
        
        with self.index_lock:
            if self.index and len(self.embeddings) > 0:
                # Use FAISS for fast search
                query_vec = np.array([query_emb], dtype=np.float32)
                faiss.normalize_L2(query_vec)
                
                scores, indices = self.index.search(query_vec, min(k, len(self.documents)))
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.documents):
                        results.append(self.documents[idx])
                
                return results
            else:
                # Fallback to simple search
                return self._simple_similarity_search(query, query_emb, k)
    
    def similarity_search_with_score(self, query: str, k: int = 5, embedding_func=None) -> List[Tuple[Document, float]]:
        """Similarity search with scores"""
        if not self.documents:
            return []
        
        if embedding_func:
            query_emb = embedding_func(query)
        else:
            docs = self.similarity_search(query, k)
            return [(doc, 0.8) for doc in docs]
        
        with self.index_lock:
            if self.index and len(self.embeddings) > 0:
                query_vec = np.array([query_emb], dtype=np.float32)
                faiss.normalize_L2(query_vec)
                
                scores, indices = self.index.search(query_vec, min(k, len(self.documents)))
                
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.documents):
                        results.append((self.documents[idx], float(score)))
                
                return results
            else:
                return self._simple_similarity_search_with_score(query, query_emb, k)
    
    def _simple_similarity_search(self, query: str, query_emb: List[float], k: int) -> List[Document]:
        """Simple similarity search without FAISS"""
        if not self.embeddings:
            return self.documents[:k]
        
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_emb, doc_emb)
            similarities.append((sim, i))
        
        similarities.sort(reverse=True)
        return [self.documents[idx] for _, idx in similarities[:k]]
    
    def _simple_similarity_search_with_score(self, query: str, query_emb: List[float], k: int) -> List[Tuple[Document, float]]:
        """Simple similarity search with scores"""
        if not self.embeddings:
            return [(doc, 0.8) for doc in self.documents[:k]]
        
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_emb, doc_emb)
            similarities.append((sim, i))
        
        similarities.sort(reverse=True)
        return [(self.documents[idx], score) for score, idx in similarities[:k]]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        try:
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        except:
            return 0.0

class VectorStoreManager:
    """High-performance vector store manager"""
    
    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = OptimizedEmbeddings()
        self.vector_store = None
        self.fast_store = FastVectorStore()
        self.logger = logging.getLogger(__name__)
        
        # Text splitter
        if TEXT_SPLITTER_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Optimal size for search
                chunk_overlap=100,  # Reduced overlap for speed
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
        else:
            self.text_splitter = self._simple_text_splitter
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def _simple_text_splitter(self, text: str) -> List[str]:
        """Simple text splitter fallback"""
        chunks = []
        chunk_size = 800
        overlap = 100
        
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (". " if current_chunk else "") + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def _initialize_vector_store(self):
        """Initialize vector store"""
        try:
            if LANGCHAIN_AVAILABLE:
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings,
                    collection_name="recommendations_responses"
                )
                self.logger.info("Chroma vector store initialized")
            else:
                self.logger.warning("Using fast vector store - LangChain unavailable")
                
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents with optimized processing"""
        if not documents:
            self.logger.warning("No documents provided")
            return False
        
        try:
            start_time = time.time()
            all_docs = []
            all_texts = []
            
            # Process documents in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                chunk_futures = []
                
                for doc in documents:
                    if not doc.get('content'):
                        continue
                    
                    future = executor.submit(self._process_document, doc)
                    chunk_futures.append(future)
                
                # Collect results
                for future in chunk_futures:
                    try:
                        doc_chunks, chunk_texts = future.result(timeout=30)
                        all_docs.extend(doc_chunks)
                        all_texts.extend(chunk_texts)
                    except Exception as e:
                        self.logger.error(f"Error processing document chunk: {e}")
            
            if not all_docs:
                self.logger.warning("No valid document chunks to add")
                return False
            
            # Batch generate embeddings
            self.logger.info(f"Generating embeddings for {len(all_texts)} chunks...")
            embeddings = self.embeddings.embed_documents(all_texts)
            
            # Add to stores
            if self.vector_store and hasattr(self.vector_store, 'add_documents'):
                try:
                    self.vector_store.add_documents(all_docs)
                    if hasattr(self.vector_store, 'persist'):
                        self.vector_store.persist()
                except Exception as e:
                    self.logger.error(f"Error adding to Chroma: {e}")
            
            # Add to fast store
            self.fast_store.add_documents(all_docs, embeddings)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Added {len(all_docs)} chunks in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False
    
    def _process_document(self, doc: Dict[str, Any]) -> Tuple[List[Document], List[str]]:
        """Process single document into chunks"""
        docs = []
        texts = []
        
        content = doc['content']
        if TEXT_SPLITTER_AVAILABLE:
            chunks = self.text_splitter.split_text(content)
        else:
            chunks = self._simple_text_splitter(content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            
            # Clean and optimize chunk
            chunk = self._clean_chunk(chunk)
            
            metadata = {
                'source': doc.get('source', 'unknown'),
                'chunk_id': i,
                'document_type': doc.get('document_type', 'unknown'),
                'total_chunks': len(chunks),
                'content_type': self._detect_content_type(chunk)
            }
            
            if 'metadata' in doc:
                metadata.update(doc['metadata'])
            
            langchain_doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            
            docs.append(langchain_doc)
            texts.append(chunk)
        
        return docs, texts
    
    def _clean_chunk(self, chunk: str) -> str:
        """Clean chunk for better search"""
        import re
        
        # Remove excessive whitespace
        chunk = re.sub(r'\s+', ' ', chunk)
        
        # Remove page numbers and headers/footers
        chunk = re.sub(r'\b(page|p\.)\s*\d+\b', '', chunk, flags=re.IGNORECASE)
        
        # Remove common document artifacts
        chunk = re.sub(r'\b(confidential|draft|final)\b', '', chunk, flags=re.IGNORECASE)
        
        return chunk.strip()
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type for better filtering"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['recommendation', 'recommend', 'should']):
            return 'recommendation'
        elif any(word in text_lower for word in ['response', 'action taken', 'implemented']):
            return 'response'
        elif any(word in text_lower for word in ['concern', 'issue', 'problem']):
            return 'concern'
        else:
            return 'general'
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Fast similarity search"""
        try:
            # Use fast store first
            results = self.fast_store.similarity_search(
                query, k=k, embedding_func=self.embeddings.embed_query
            )
            
            # Apply filters if specified
            if filter_dict and results:
                filtered_results = []
                for doc in results:
                    match = True
                    for key, value in filter_dict.items():
                        if key in doc.metadata and doc.metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(doc)
                results = filtered_results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Similarity search with scores"""
        try:
            results = self.fast_store.similarity_search_with_score(
                query, k=k, embedding_func=self.embeddings.embed_query
            )
            
            # Apply filters
            if filter_dict and results:
                filtered_results = []
                for doc, score in results:
                    match = True
                    for key, value in filter_dict.items():
                        if key in doc.metadata and doc.metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append((doc, score))
                results = filtered_results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            stats = {
                'total_documents': len(self.fast_store.documents),
                'total_embeddings': len(self.fast_store.embeddings),
                'embedding_dimension': self.fast_store.dimension,
                'collection_name': 'recommendations_responses',
                'cache_size': len(self.embeddings.cache),
                'faiss_available': FAISS_AVAILABLE,
                'langchain_available': LANGCHAIN_AVAILABLE
            }
            
            # Content type distribution
            content_types = {}
            for doc in self.fast_store.documents:
                content_type = doc.metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            stats['content_distribution'] = content_types
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {
                'total_documents': 0, 
                'embedding_dimension': 'unknown', 
                'collection_name': 'error',
                'error': str(e)
            }
    
    def search_by_content_type(self, query: str, content_type: str, k: int = 5) -> List[Document]:
        """Search within specific content type"""
        filter_dict = {'content_type': content_type}
        return self.similarity_search(query, k=k, filter_dict=filter_dict)
    
    def cleanup(self):
        """Cleanup vector store connections"""
        try:
            if self.vector_store and hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
            if self.vector_store and hasattr(self.vector_store, 'close'):
                self.vector_store.close()
            self.logger.info("Vector store cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
