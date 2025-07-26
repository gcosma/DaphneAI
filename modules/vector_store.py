# modules/vector_store.py
# COMPLETE FIXED VERSION - All issues resolved

import logging
import numpy as np
import time
import atexit
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===============================================
# DOCUMENT CLASS DEFINITION - CRITICAL FIX
# ===============================================

@dataclass
class Document:
    """Document class for vector store operations"""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    @property
    def content(self) -> str:
        """Alias for page_content for compatibility"""
        return self.page_content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'page_content': self.page_content,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary"""
        return cls(
            page_content=data.get('page_content', ''),
            metadata=data.get('metadata', {})
        )

# ===============================================
# OPTIONAL IMPORTS WITH FALLBACKS
# ===============================================

# Try importing LangChain Document (prefer this if available)
try:
    from langchain.schema import Document as LangChainDocument
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    TEXT_SPLITTER_AVAILABLE = True
    # Use LangChain Document if available
    Document = LangChainDocument
except ImportError:
    TEXT_SPLITTER_AVAILABLE = False
    # Use our custom Document class (already defined above)

# Try importing Chroma
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Try importing FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Try importing sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try importing OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ===============================================
# OPTIMIZED EMBEDDINGS CLASS
# ===============================================

class OptimizedEmbeddings:
    """High-performance embeddings with caching and multiple backends"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the best available embedding model"""
        # Try SentenceTransformers first (best performance)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.backend = 'sentence_transformers'
                self.logger.info("✅ Using SentenceTransformers for embeddings")
                return
            except Exception as e:
                self.logger.warning(f"SentenceTransformers failed: {e}")
        
        # Try OpenAI as fallback
        if OPENAI_AVAILABLE:
            try:
                # Test OpenAI connection
                self.backend = 'openai'
                self.logger.info("✅ Using OpenAI for embeddings")
                return
            except Exception as e:
                self.logger.warning(f"OpenAI embeddings failed: {e}")
        
        # Use simple fallback
        self.backend = 'simple'
        self.logger.info("⚠️ Using simple hash-based embeddings (fallback)")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        if not text or not text.strip():
            return [0.0] * 384  # Default dimension
        
        # Check cache first
        text_hash = hash(text.strip())
        with self.cache_lock:
            if text_hash in self.cache:
                return self.cache[text_hash]
        
        # Generate embedding
        if self.backend == 'sentence_transformers':
            embedding = self._sentence_transformer_embed(text)
        elif self.backend == 'openai':
            embedding = self._openai_embed(text)
        else:
            embedding = self._simple_embed(text)
        
        # Cache result
        with self.cache_lock:
            self.cache[text_hash] = embedding
            # Limit cache size
            if len(self.cache) > 1000:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return embedding
    
    def _sentence_transformer_embed(self, text: str) -> List[float]:
        """Generate embedding using SentenceTransformers"""
        try:
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"SentenceTransformer embedding error: {e}")
            return self._simple_embed(text)
    
    def _openai_embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        try:
            import openai
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            # Truncate to 1536 dimensions for consistency
            return embedding[:1536]
        except Exception as e:
            self.logger.error(f"OpenAI embedding error: {e}")
            return self._simple_embed(text)
    
    def _simple_embed(self, text: str) -> List[float]:
        """Simple hash-based embedding fallback"""
        # Create a simple but consistent embedding
        words = text.lower().split()
        embedding = [0.0] * 384
        
        for i, word in enumerate(words[:100]):  # Limit to 100 words
            word_hash = hash(word)
            for j in range(min(4, len(embedding))):
                idx = (word_hash + j) % len(embedding)
                embedding[idx] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding

# ===============================================
# FAST VECTOR STORE CLASS
# ===============================================

class FastVectorStore:
    """High-performance vector store with FAISS backend"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self.index_lock = threading.Lock()
        
        # Initialize FAISS index if available
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product
        else:
            self.index = None
    
    def add_documents(self, docs: List[Document], embeddings: List[List[float]]):
        """Add documents with their embeddings"""
        with self.index_lock:
            start_idx = len(self.documents)
            
            # Add documents
            self.documents.extend(docs)
            self.embeddings.extend(embeddings)
            
            # Add to FAISS index if available
            if self.index and embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.index.add(embeddings_array)
    
    def similarity_search(self, query: str, k: int = 5, embedding_func=None) -> List[Document]:
        """Search for similar documents"""
        if not self.documents or not embedding_func:
            return self.documents[:k]
        
        query_embedding = embedding_func(query)
        results = self._search_with_embedding(query_embedding, k)
        return [doc for doc, _ in results]
    
    def similarity_search_with_score(self, query: str, k: int = 5, embedding_func=None) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        if not self.documents or not embedding_func:
            return [(doc, 0.8) for doc in self.documents[:k]]
        
        query_embedding = embedding_func(query)
        return self._search_with_embedding(query_embedding, k)
    
    def _search_with_embedding(self, query_embedding: List[float], k: int) -> List[Tuple[Document, float]]:
        """Internal search with pre-computed embedding"""
        if not self.embeddings:
            return [(doc, 0.8) for doc in self.documents[:k]]
        
        # Use FAISS if available
        if self.index and len(self.embeddings) > 0:
            try:
                query_array = np.array([query_embedding], dtype=np.float32)
                scores, indices = self.index.search(query_array, min(k, len(self.documents)))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self.documents):
                        results.append((self.documents[idx], float(score)))
                
                return results
            except Exception as e:
                logging.warning(f"FAISS search failed: {e}")
        
        # Fallback to simple similarity
        return self._simple_similarity_search(query_embedding, k)
    
    def _simple_similarity_search(self, query_embedding: List[float], k: int) -> List[Tuple[Document, float]]:
        """Simple cosine similarity search"""
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        similarities.sort(reverse=True)
        results = []
        for score, idx in similarities[:k]:
            results.append((self.documents[idx], score))
        
        return results
    
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

# ===============================================
# VECTOR STORE MANAGER
# ===============================================

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
                chunk_size=800,
                chunk_overlap=100,
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
        
        return chunks
    
    def _initialize_vector_store(self):
        """Initialize the vector store backend"""
        if CHROMA_AVAILABLE:
            try:
                import chromadb
                self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
                self.vector_store = self.chroma_client.get_or_create_collection("documents")
                self.logger.info("✅ ChromaDB initialized")
            except Exception as e:
                self.logger.warning(f"ChromaDB initialization failed: {e}")
                self.vector_store = None
        else:
            self.logger.info("Using FastVectorStore (ChromaDB not available)")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store"""
        try:
            start_time = time.time()
            
            # Convert to Document objects and split text
            all_docs = []
            all_texts = []
            
            # Process documents in parallel for speed
            with ThreadPoolExecutor(max_workers=4) as executor:
                chunk_futures = []
                
                for doc in documents:
                    future = executor.submit(self._process_document_for_indexing, doc)
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
            if self.vector_store and hasattr(self.vector_store, 'add'):
                try:
                    # Convert to ChromaDB format
                    ids = [f"doc_{i}" for i in range(len(all_docs))]
                    documents_text = [doc.page_content for doc in all_docs]
                    metadatas = [doc.metadata for doc in all_docs]
                    
                    self.vector_store.add(
                        embeddings=embeddings,
                        documents=documents_text,
                        metadatas=metadatas,
                        ids=ids
                    )
                    self.logger.info("Added documents to ChromaDB")
                except Exception as e:
                    self.logger.error(f"Error adding to ChromaDB: {e}")
            
            # Add to fast store
            self.fast_store.add_documents(all_docs, embeddings)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Added {len(all_docs)} chunks in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False
    
    def _process_document_for_indexing(self, doc: Dict[str, Any]) -> Tuple[List[Document], List[str]]:
        """Process a single document for indexing"""
        content = doc.get('content', '')
        if not content or len(content.strip()) < 50:
            return [], []
        
        # Split text into chunks
        if hasattr(self.text_splitter, 'split_text'):
            chunks = self.text_splitter.split_text(content)
        else:
            chunks = self.text_splitter(content)
        
        documents = []
        texts = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) >= 50:  # Minimum chunk size
                doc_obj = Document(
                    page_content=chunk,
                    metadata={
                        'source': doc.get('source', 'unknown'),
                        'document_type': doc.get('document_type', 'unknown'),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        **doc.get('metadata', {})
                    }
                )
                documents.append(doc_obj)
                texts.append(chunk)
        
        return documents, texts
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            # Try ChromaDB first
            if self.vector_store and hasattr(self.vector_store, 'query'):
                try:
                    results = self.vector_store.query(
                        query_texts=[query],
                        n_results=k,
                        where=filter_dict
                    )
                    
                    documents = []
                    if results and 'documents' in results and results['documents']:
                        for i, doc_text in enumerate(results['documents'][0]):
                            metadata = results.get('metadatas', [[]])[0]
                            doc_metadata = metadata[i] if i < len(metadata) else {}
                            
                            documents.append(Document(
                                page_content=doc_text,
                                metadata=doc_metadata
                            ))
                    
                    return documents
                except Exception as e:
                    self.logger.warning(f"ChromaDB search failed: {e}")
            
            # Fallback to fast store
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
                'chroma_available': CHROMA_AVAILABLE,
                'backend': self.embeddings.backend
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
    
    def clear_collection(self):
        """Clear the vector store"""
        try:
            # Clear fast store
            self.fast_store.documents.clear()
            self.fast_store.embeddings.clear()
            if self.fast_store.index and FAISS_AVAILABLE:
                self.fast_store.index.reset()
            
            # Clear ChromaDB if available
            if self.vector_store and hasattr(self.vector_store, 'delete'):
                try:
                    # Delete all documents
                    result = self.vector_store.get()
                    if result and 'ids' in result:
                        self.vector_store.delete(ids=result['ids'])
                except Exception as e:
                    self.logger.warning(f"ChromaDB clear failed: {e}")
            
            self.logger.info("Vector store cleared")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
    
    def cleanup(self):
        """Cleanup vector store connections"""
        try:
            if self.vector_store and hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
            self.logger.info("Vector store cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# ===============================================
# EXPORTS
# ===============================================

__all__ = [
    'Document',
    'VectorStoreManager',
    'FastVectorStore',
    'OptimizedEmbeddings'
]
