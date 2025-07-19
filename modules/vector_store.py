# ===============================================
# FILE: modules/vector_store.py
# ===============================================

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import atexit

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

class MockEmbeddings:
    """Mock embeddings for testing"""
    def embed_query(self, text: str) -> List[float]:
        import hashlib
        hash_object = hashlib.md5(text.encode())
        hex_dig = hash_object.hexdigest()
        return [float(int(hex_dig[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

class MockVectorStore:
    """Mock vector store for testing"""
    def __init__(self):
        self.documents = []
    
    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        return self.documents[:k]
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        docs = self.similarity_search(query, k)
        return [(doc, 0.8) for doc in docs]

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vector_store = None
        self.logger = logging.getLogger(__name__)
        
        if TEXT_SPLITTER_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
        else:
            self.text_splitter = self._simple_text_splitter
            
        self._initialize_embeddings()
        self._initialize_vector_store()
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def _simple_text_splitter(self, text: str) -> List[str]:
        """Simple text splitter fallback"""
        chunks = []
        chunk_size = 1000
        overlap = 200
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _initialize_embeddings(self):
        """Initialize embeddings"""
        try:
            if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                self.embeddings = OpenAIEmbeddings()
                self.logger.info("OpenAI embeddings initialized")
            else:
                self.logger.warning("Using mock embeddings - OpenAI key not found or LangChain unavailable")
                self.embeddings = MockEmbeddings()
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {e}")
            self.embeddings = MockEmbeddings()
    
    def _initialize_vector_store(self):
        """Initialize vector store"""
        try:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            if LANGCHAIN_AVAILABLE:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="recommendations_responses"
                )
                self.logger.info("Chroma vector store initialized")
            else:
                self.logger.warning("Using mock vector store - LangChain unavailable")
                self.vector_store = MockVectorStore()
                
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            self.vector_store = MockVectorStore()
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to vector store"""
        try:
            if not documents:
                self.logger.warning("No documents provided")
                return False
            
            langchain_docs = []
            
            for doc in documents:
                if not doc.get('content'):
                    continue
                
                if TEXT_SPLITTER_AVAILABLE:
                    chunks = self.text_splitter.split_text(doc['content'])
                else:
                    chunks = self._simple_text_splitter(doc['content'])
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:
                        continue
                        
                    metadata = {
                        'source': doc.get('source', 'unknown'),
                        'chunk_id': i,
                        'document_type': doc.get('document_type', 'unknown'),
                        'total_chunks': len(chunks)
                    }
                    
                    if 'metadata' in doc:
                        metadata.update(doc['metadata'])
                    
                    langchain_doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    langchain_docs.append(langchain_doc)
            
            if not langchain_docs:
                self.logger.warning("No valid document chunks to add")
                return False
            
            if hasattr(self.vector_store, 'add_documents'):
                self.vector_store.add_documents(langchain_docs)
                if hasattr(self.vector_store, 'persist'):
                    self.vector_store.persist()
            
            self.logger.info(f"Added {len(langchain_docs)} document chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if hasattr(self.vector_store, 'similarity_search'):
                return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        try:
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                return self.vector_store.similarity_search_with_score(query, k=k)
            else:
                docs = self.similarity_search(query, k)
                return [(doc, 0.8) for doc in docs]
        except Exception as e:
            self.logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            if hasattr(self.vector_store, '_collection'):
                collection = self.vector_store._collection
                stats = {
                    'total_documents': collection.count() if hasattr(collection, 'count') else 0,
                    'collection_name': getattr(collection, 'name', 'unknown')
                }
            else:
                stats = {
                    'total_documents': len(getattr(self.vector_store, 'documents', [])),
                    'collection_name': 'mock'
                }
            
            try:
                if self.embeddings and hasattr(self.embeddings, 'embed_query'):
                    test_embedding = self.embeddings.embed_query("test")
                    stats['embedding_dimension'] = len(test_embedding)
            except:
                stats['embedding_dimension'] = 'unknown'
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {'total_documents': 0, 'embedding_dimension': 'unknown', 'collection_name': 'error'}
    
    def cleanup(self):
        """Cleanup vector store connections"""
        try:
            if hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
            if hasattr(self.vector_store, 'close'):
                self.vector_store.close()
            self.logger.info("Vector store cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
