"""
AI-Based Smart Document Search Engine - Enhanced Version
Single file Streamlit application with advanced features
"""

import streamlit as st
import time
import logging
import io
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import re
import chardet
import hashlib
from datetime import datetime
import json

# Optional imports for AI features
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
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

# Environment variables and configuration
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 100))
ENABLE_CLUSTERING = os.getenv('ENABLE_CLUSTERING', 'true').lower() == 'true'
ENABLE_DUPLICATE_DETECTION = os.getenv('ENABLE_DUPLICATE_DETECTION', 'true').lower() == 'true'
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.1))

@dataclass
class SearchResult:
    """Enhanced search result with additional metadata"""
    filename: str
    content: str
    snippet: str
    score: float
    rank: int
    file_type: str = ""
    word_count: int = 0
    relevance_explanation: str = ""
    duplicate_of: Optional[str] = None

class AdvancedDocumentProcessor:
    """Enhanced document processor with advanced features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_hashes = set()  # For duplicate detection
        
    def process_files(self, uploaded_files) -> List[Dict[str, Any]]:
        """Process uploaded files with enhanced features"""
        documents = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            try:
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
                
                # Check file size
                file_size = len(file.getvalue())
                if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"⚠️ {file.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
                    continue
                
                doc = self._process_single_file(file)
                if doc:
                    # Check for duplicates
                    if ENABLE_DUPLICATE_DETECTION:
                        doc = self._check_duplicate(doc)
                    
                    documents.append(doc)
                    
            except Exception as e:
                self.logger.error(f"Failed to process {file.name}: {e}")
                st.error(f"❌ Failed to process {file.name}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        return documents
    
    def _process_single_file(self, file) -> Optional[Dict[str, Any]]:
        """Enhanced single file processing"""
        filename = file.name
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Get file content and hash
        file_content = file.getvalue()
        file_size = len(file_content)
        content_hash = hashlib.md5(file_content).hexdigest()
        
        # Reset file pointer
        file.seek(0)
        
        # Process based on file type
        text = ""
        if file_extension == 'pdf':
            text = self._extract_pdf_text(file)
        elif file_extension == 'docx':
            text = self._extract_docx_text(file)
        elif file_extension in ['txt', 'md', 'py', 'js', 'html', 'css', 'json', 'csv']:
            text = self._extract_text_file(file)
        else:
            text = self._extract_text_file(file)
        
        if not text:
            return None
        
        # Enhanced text analysis
        word_count = len(text.split())
        char_count = len(text)
        language = self._detect_language(text)
        
        return {
            'filename': filename,
            'text': text,
            'file_type': file_extension,
            'file_size': file_size,
            'content_hash': content_hash,
            'word_count': word_count,
            'char_count': char_count,
            'language': language,
            'upload_time': datetime.now().isoformat(),
            'quality_score': self._calculate_quality_score(text, word_count)
        }
    
    def _check_duplicate(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Check for duplicate documents"""
        content_hash = doc['content_hash']
        
        if content_hash in self.processed_hashes:
            doc['is_duplicate'] = True
            doc['duplicate_of'] = content_hash
        else:
            self.processed_hashes.add(content_hash)
            doc['is_duplicate'] = False
        
        return doc
    
    def _detect_language(self, text: str) -> str:
        """Basic language detection"""
        # Simple heuristic - could be enhanced with proper language detection library
        common_english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().split()[:100]  # Check first 100 words
        english_count = sum(1 for word in words if word in common_english_words)
        
        if len(words) > 0 and english_count / len(words) > 0.1:
            return "English"
        return "Unknown"
    
    def _calculate_quality_score(self, text: str, word_count: int) -> float:
        """Calculate document quality score"""
        score = 0.0
        
        # Length factor
        if 50 <= word_count <= 10000:
            score += 0.3
        elif word_count > 10000:
            score += 0.2
        
        # Text structure
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences > 0:
            avg_words_per_sentence = word_count / sentences
            if 5 <= avg_words_per_sentence <= 25:
                score += 0.3
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        if unique_chars > 20:
            score += 0.2
        
        # No excessive repetition
        words = text.split()
        unique_words = len(set(words))
        if len(words) > 0 and unique_words / len(words) > 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _extract_pdf_text(self, file) -> str:
        """Enhanced PDF text extraction"""
        if not PDF_AVAILABLE:
            raise Exception("PDF processing not available. Install: pip install pdfplumber PyPDF2")
        
        text = ""
        page_count = 0
        
        try:
            with pdfplumber.open(file) as pdf:
                page_count = len(pdf.pages)
                
                # Limit pages for very large PDFs
                max_pages = int(os.getenv('MAX_PDF_PAGES', 1000))
                pages_to_process = min(page_count, max_pages)
                
                for i, page in enumerate(pdf.pages[:pages_to_process]):
                    if i > 0 and i % 10 == 0:
                        st.text(f"Processing PDF page {i}/{pages_to_process}...")
                    
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
        except Exception as e:
            self.logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            try:
                file.seek(0)
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for i, page in enumerate(pdf_reader.pages):
                    if i > 0 and i % 10 == 0:
                        st.text(f"Processing PDF page {i}/{page_count}...")
                    
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
            except Exception as e2:
                raise Exception(f"PDF extraction failed: {e2}")
        
        return text.strip()
    
    def _extract_docx_text(self, file) -> str:
        """Enhanced DOCX text extraction"""
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
            
            # Extract headers and footers (if available)
            for section in doc.sections:
                if hasattr(section, 'header'):
                    text += section.header.text + "\n"
                if hasattr(section, 'footer'):
                    text += section.footer.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {e}")
    
    def _extract_text_file(self, file) -> str:
        """Enhanced text file extraction with better encoding detection"""
        try:
            content = file.read()
            if isinstance(content, bytes):
                # Try UTF-8 first
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    # Use chardet for encoding detection
                    detected = chardet.detect(content)
                    encoding = detected.get('encoding', 'utf-8')
                    confidence = detected.get('confidence', 0)
                    
                    if confidence > 0.7:
                        return content.decode(encoding, errors='ignore')
                    else:
                        # Try common encodings
                        for enc in ['latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                return content.decode(enc)
                            except:
                                continue
                        
                        # Last resort
                        return content.decode('utf-8', errors='ignore')
            else:
                return content
                
        except Exception as e:
            raise Exception(f"Text extraction failed: {e}")
    
    def get_supported_formats(self) -> Dict[str, bool]:
        """Get supported file formats"""
        return {
            'pdf': PDF_AVAILABLE,
            'docx': DOCX_AVAILABLE,
            'txt': True,
            'md': True,
            'py': True,
            'js': True,
            'html': True,
            'css': True,
            'json': True,
            'csv': True
        }

class EnhancedSearchEngine:
    """Enhanced search engine with advanced features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.documents = []
        self.document_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.clusters = None
        
        # Initialize AI model if available
        if AI_AVAILABLE:
            try:
                model_name = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
                self.model = SentenceTransformer(model_name)
                self.logger.info(f"AI search model '{model_name}' loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load AI model: {e}")
                self.model = None
        else:
            self.model = None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents with enhanced indexing"""
        self.documents = documents
        
        if documents:
            with st.spinner("Building advanced search index..."):
                self._build_enhanced_index()
    
    def _build_enhanced_index(self) -> None:
        """Build enhanced search indexes"""
        if not self.documents:
            return
        
        texts = [doc.get('text', '') for doc in self.documents]
        
        # Build semantic embeddings
        if self.model and texts:
            try:
                self.document_embeddings = self.model.encode(texts, show_progress_bar=True)
                self.logger.info(f"Built semantic embeddings for {len(texts)} documents")
                
                # Build clusters if enabled
                if ENABLE_CLUSTERING and len(texts) > 5:
                    self._build_clusters()
                    
            except Exception as e:
                self.logger.error(f"Failed to build embeddings: {e}")
        
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
    
    def _build_clusters(self) -> None:
        """Build document clusters for organization"""
        try:
            n_clusters = min(5, len(self.documents) // 2)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                self.clusters = kmeans.fit_predict(self.document_embeddings)
                
                # Add cluster info to documents
                for i, doc in enumerate(self.documents):
                    doc['cluster'] = int(self.clusters[i])
                    
                self.logger.info(f"Created {n_clusters} document clusters")
        except Exception as e:
            self.logger.error(f"Failed to build clusters: {e}")
    
    def search(self, query: str, max_results: int = 10, filter_options: Dict = None) -> List[SearchResult]:
        """Enhanced search with multiple algorithms and filtering"""
        start_time = time.time()
        
        if not self.documents or not query.strip():
            return []
        
        # Apply filters first
        filtered_docs = self._apply_filters(filter_options or {})
        
        # Get results from different search methods
        semantic_results = []
        keyword_results = []
        
        if self.model and self.document_embeddings is not None:
            semantic_results = self._semantic_search(query, filtered_docs, max_results * 2)
        
        keyword_results = self._enhanced_keyword_search(query, filtered_docs, max_results * 2)
        
        # Combine and re-rank results
        combined_results = self._combine_results(semantic_results, keyword_results, max_results)
        
        # Add explanations and rankings
        for i, result in enumerate(combined_results):
            result.rank = i + 1
            result.relevance_explanation = self._generate_explanation(result, query)
        
        search_time = time.time() - start_time
        self.logger.info(f"Enhanced search completed in {search_time:.3f}s")
        
        return combined_results
    
    def _apply_filters(self, filter_options: Dict) -> List[int]:
        """Apply filters and return document indices"""
        indices = list(range(len(self.documents)))
        
        if filter_options.get('file_type'):
            file_type = filter_options['file_type']
            indices = [i for i in indices if self.documents[i].get('file_type') == file_type]
        
        if filter_options.get('min_quality'):
            min_quality = filter_options['min_quality']
            indices = [i for i in indices if self.documents[i].get('quality_score', 0) >= min_quality]
        
        if filter_options.get('exclude_duplicates'):
            indices = [i for i in indices if not self.documents[i].get('is_duplicate', False)]
        
        return indices
    
    def _semantic_search(self, query: str, doc_indices: List[int], max_results: int) -> List[SearchResult]:
        """Enhanced semantic search"""
        try:
            query_embedding = self.model.encode([query])
            
            # Filter embeddings for allowed documents
            filtered_embeddings = self.document_embeddings[doc_indices]
            similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
            
            results = []
            for i, similarity in enumerate(similarities):
                if similarity > SIMILARITY_THRESHOLD:
                    doc_idx = doc_indices[i]
                    doc = self.documents[doc_idx]
                    
                    result = SearchResult(
                        filename=doc.get('filename', f'Document {doc_idx+1}'),
                        content=doc.get('text', ''),
                        snippet=self._extract_enhanced_snippet(doc.get('text', ''), query),
                        score=float(similarity),
                        rank=0,
                        file_type=doc.get('file_type', ''),
                        word_count=doc.get('word_count', 0)
                    )
                    results.append(result)
            
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _enhanced_keyword_search(self, query: str, doc_indices: List[int], max_results: int) -> List[SearchResult]:
        """Enhanced keyword search with better scoring"""
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for doc_idx in doc_indices:
            doc = self.documents[doc_idx]
            text = doc.get('text', '').lower()
            filename = doc.get('filename', '').lower()
            
            if not text:
                continue
            
            score = 0
            
            # Exact phrase matching (highest weight)
            exact_matches = text.count(query_lower)
            score += exact_matches * 15
            
            # Individual word matching with position weighting
            for word in query_words:
                word_count = text.count(word)
                score += word_count * 3
                
                # Bonus for words in filename
                if word in filename:
                    score += 10
                
                # Bonus for words at beginning of document
                first_100_words = ' '.join(text.split()[:100])
                if word in first_100_words:
                    score += 2
            
            # Quality score bonus
            quality_score = doc.get('quality_score', 0)
            score += quality_score * 5
            
            if score > 0:
                result = SearchResult(
                    filename=doc.get('filename', f'Document {doc_idx+1}'),
                    content=doc.get('text', ''),
                    snippet=self._extract_enhanced_snippet(doc.get('text', ''), query),
                    score=score,
                    rank=0,
                    file_type=doc.get('file_type', ''),
                    word_count=doc.get('word_count', 0)
                )
                results.append(result)
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def _combine_results(self, semantic_results: List[SearchResult], 
                        keyword_results: List[SearchResult], max_results: int) -> List[SearchResult]:
        """Intelligently combine results from different search methods"""
        # Create a mapping of filename to results
        all_results = {}
        
        # Add semantic results with weight
        for result in semantic_results:
            key = result.filename
            if key not in all_results:
                all_results[key] = result
                all_results[key].score = result.score * 0.7  # Semantic weight
            else:
                # Boost score if found in both
                all_results[key].score += result.score * 0.3
        
        # Add keyword results with weight
        for result in keyword_results:
            key = result.filename
            if key not in all_results:
                all_results[key] = result
                all_results[key].score = result.score * 0.3  # Keyword weight
            else:
                # Boost score if found in both
                all_results[key].score += result.score * 0.7
        
        # Sort and return top results
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:max_results]
    
    def _extract_enhanced_snippet(self, text: str, query: str, max_length: int = 300) -> str:
        """Extract enhanced snippet with better context"""
        if not text or not query:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find best position for snippet
        positions = []
        
        # Look for exact phrase
        pos = text_lower.find(query_lower)
        if pos != -1:
            positions.append(pos)
        
        # Look for individual words
        for word in query_lower.split():
            pos = text_lower.find(word)
            if pos != -1:
                positions.append(pos)
        
        if not positions:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Use the earliest position
        best_pos = min(positions)
        
        # Extract snippet with good context
        start = max(0, best_pos - max_length // 3)
        end = min(len(text), best_pos + max_length * 2 // 3)
        snippet = text[start:end]
        
        # Enhanced highlighting
        for word in query.split():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            snippet = pattern.sub(f"**{word.upper()}**", snippet)
        
        # Add ellipsis
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        
        return prefix + snippet + suffix
    
    def _generate_explanation(self, result: SearchResult, query: str) -> str:
        """Generate explanation for why this result was matched"""
        explanations = []
        
        text_lower = result.content.lower()
        query_lower = query.lower()
        
        # Check for exact phrase
        if query_lower in text_lower:
            explanations.append("Contains exact phrase")
        
        # Check for individual words
        words_found = []
        for word in query.split():
            if word.lower() in text_lower:
                words_found.append(word)
        
        if words_found:
            explanations.append(f"Contains keywords: {', '.join(words_found)}")
        
        # Check filename match
        if query_lower in result.filename.lower():
            explanations.append("Filename matches query")
        
        # Quality score
        if hasattr(result, 'word_count') and result.word_count > 100:
            explanations.append("High-quality document")
        
        return "; ".join(explanations) if explanations else "Semantic similarity"
    
    def get_clusters_info(self) -> Dict[str, Any]:
        """Get information about document clusters"""
        if not self.clusters:
            return {}
        
        cluster_info = {}
        for i in range(max(self.clusters) + 1):
            docs_in_cluster = [j for j, c in enumerate(self.clusters) if c == i]
            cluster_info[f"Cluster {i+1}"] = {
                'count': len(docs_in_cluster),
                'documents': [self.documents[j]['filename'] for j in docs_in_cluster[:5]]
            }
        
        return cluster_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced search engine statistics"""
        stats = {
            'total_documents': len(self.documents),
            'ai_enabled': self.model is not None,
            'index_built': self.document_embeddings is not None,
            'ai_available': AI_AVAILABLE,
            'clustering_enabled': ENABLE_CLUSTERING,
            'duplicate_detection_enabled': ENABLE_DUPLICATE_DETECTION
        }
        
        if self.documents:
            # Document type distribution
            type_counts = {}
            quality_scores = []
            duplicates = 0
            
            for doc in self.documents:
                doc_type = doc.get('file_type', 'unknown')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                
                if doc.get('quality_score'):
                    quality_scores.append(doc['quality_score'])
                
                if doc.get('is_duplicate'):
                    duplicates += 1
            
            stats.update({
                'file_types': type_counts,
                'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                'duplicates_found': duplicates,
                'clusters': len(set(self.clusters)) if self.clusters else 0
            })
        
        return stats

# Enhanced UI Functions
def initialize_session_state():
    """Initialize enhanced session state"""
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = EnhancedSearchEngine()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = AdvancedDocumentProcessor()
    
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'filter_options' not in st.session_state:
        st.session_state.filter_options = {}

def render_enhanced_sidebar():
    """Render enhanced sidebar with advanced options"""
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        stats = st.session_state.search_engine.get_stats()
        
        # AI Status with more detail
        if stats['ai_available'] and stats['ai_enabled']:
            st.success("🤖 AI Search Ready")
            if stats['index_built']:
                st.success("🔗 Semantic Index Built")
            if stats.get('clusters', 0) > 0:
                st.info(f"📊 {stats['clusters']} Document Clusters")
        elif stats['ai_available']:
            st.warning("🤖 AI Available (No Index)")
        else:
            st.error("🤖 AI Unavailable")
            st.caption("Install: `pip install sentence-transformers scikit-learn`")
        
        # Enhanced Document Stats
        st.markdown("### 📊 Document Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Docs", stats['total_documents'])
            if stats.get('duplicates_found', 0) > 0:
                st.metric("Duplicates", stats['duplicates_found'])
        
        with col2:
            if stats.get('avg_quality_score'):
                st.metric("Avg Quality", f"{stats['avg_quality_score']:.2f}")
        
        # File type distribution
        if stats.get('file_types'):
            st.markdown("**File Types:**")
            for file_type, count in stats['file_types'].items():
                st.write(f"• {file_type.upper()}: {count}")
        
        # Cluster information
        if stats.get('clusters', 0) > 0:
            st.markdown("### 📊 Document Clusters")
            cluster_info = st.session_state.search_engine.get_clusters_info()
            for cluster_name, info in cluster_info.items():
                st.write(f"**{cluster_name}**: {info['count']} docs")
        
        # Advanced filters
        st.markdown("### 🔍 Search Filters")
        
        # File type filter
        if stats.get('file_types'):
            file_types = ['All'] + list(stats['file_types'].keys())
            selected_type = st.selectbox("File Type", file_types)
            st.session_state.filter_options['file_type'] = selected_type if selected_type != 'All' else None
        
        # Quality filter
        min_quality = st.slider("Minimum Quality", 0.0, 1.0, 0.0, 0.1)
        st.session_state.filter_options['min_quality'] = min_quality if min_quality > 0 else None
        
        # Duplicate filter
        if stats.get('duplicates_found', 0) > 0:
            exclude_dupes = st.checkbox("Exclude Duplicates", value=True)
            st.session_state.filter_options['exclude_duplicates'] = exclude_dupes
        
        # Supported Formats
        st.markdown("### 📁 Supported Formats")
        formats = st.session_state.processor.get_supported_formats()
        
        for fmt, available in formats.items():
            icon = "✅" if available else "❌"
            st.write(f"{icon} {fmt.upper()}")

def render_enhanced_upload_section():
    """Render enhanced file upload interface"""
    st.header("📁 Upload Documents")
    
    # Upload options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader with enhanced options
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'csv'],
            help=f"Upload documents to search through. Max size: {MAX_FILE_SIZE_MB}MB per file."
        )
    
    with col2:
        st.markdown("**Upload Options:**")
        batch_process = st.checkbox("Batch Processing", value=True, help="Process multiple files efficiently")
        detect_duplicates = st.checkbox("Detect Duplicates", value=ENABLE_DUPLICATE_DETECTION)
    
    if uploaded_files:
        # Show upload summary
        total_size = sum(len(f.getvalue()) for f in uploaded_files)
        st.info(f"📊 {len(uploaded_files)} files selected, total size: {total_size / (1024*1024):.1f} MB")
        
        # Process files
        if st.button("🚀 Process Files", type="primary"):
            with st.spinner("Processing documents with advanced features..."):
                start_time = time.time()
                
                # Set processing options
                original_duplicate_setting = ENABLE_DUPLICATE_DETECTION
                if not detect_duplicates:
                    # Temporarily disable duplicate detection
                    import os
                    os.environ['ENABLE_DUPLICATE_DETECTION'] = 'false'
                
                try:
                    new_documents = st.session_state.processor.process_files(uploaded_files)
                    
                    # Add to existing documents
                    st.session_state.documents.extend(new_documents)
                    
                    # Update search engine with enhanced indexing
                    st.session_state.search_engine.add_documents(st.session_state.documents)
                    
                    processing_time = time.time() - start_time
                    
                    # Show detailed results
                    successful = len([d for d in new_documents if 'error' not in d])
                    failed = len(new_documents) - successful
                    duplicates = len([d for d in new_documents if d.get('is_duplicate')])
                    
                    st.success(f"✅ Processing complete in {processing_time:.1f}s")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Successful", successful)
                    with col2:
                        st.metric("Failed", failed)
                    with col3:
                        if detect_duplicates:
                            st.metric("Duplicates", duplicates)
                    
                    # Show processing details
                    with st.expander("📋 Processing Details"):
                        for doc in new_documents:
                            if 'error' in doc:
                                st.error(f"❌ {doc['filename']}: {doc['error']}")
                            else:
                                status_icons = []
                                if doc.get('is_duplicate'):
                                    status_icons.append("🔄")
                                if doc.get('quality_score', 0) > 0.7:
                                    status_icons.append("⭐")
                                
                                status = " ".join(status_icons)
                                st.success(f"✅ {doc['filename']} - {doc['word_count']} words, Quality: {doc.get('quality_score', 0):.2f} {status}")
                
                finally:
                    # Restore original setting
                    if not detect_duplicates:
                        os.environ['ENABLE_DUPLICATE_DETECTION'] = str(original_duplicate_setting).lower()

def render_enhanced_search_section():
    """Render enhanced search interface"""
    st.header("🔍 Advanced Smart Search")
    
    if not st.session_state.documents:
        st.warning("📝 Please upload documents first to enable search.")
        return
    
    # Search input with suggestions
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search your documents",
            placeholder="Enter your search query... (try natural language or keywords)",
            help="Use natural language, keywords, or phrases. Advanced AI understanding enabled."
        )
    
    with col2:
        search_mode = st.selectbox(
            "Search Mode", 
            ["Smart (Auto)", "Semantic Only", "Keyword Only"],
            help="Smart mode automatically combines semantic and keyword search"
        )
    
    # Advanced search options
    with st.expander("🔧 Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_results = st.selectbox("Max Results", [5, 10, 15, 20, 30], index=1)
            
        with col2:
            show_explanations = st.checkbox("Show Explanations", value=True)
            
        with col3:
            show_snippets_only = st.checkbox("Snippets Only", value=False)
    
    # Search suggestions
    if st.session_state.search_history:
        recent_queries = [h['query'] for h in st.session_state.search_history[-5:]]
        if recent_queries:
            st.markdown("**Recent searches:** " + " • ".join([f"`{q}`" for q in recent_queries]))
    
    # Perform search
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching with advanced algorithms..."):
            start_time = time.time()
            
            # Apply search filters
            filter_options = st.session_state.filter_options.copy()
            
            # Get results
            results = st.session_state.search_engine.search(
                query, 
                max_results=max_results, 
                filter_options=filter_options
            )
            
            search_time = time.time() - start_time
            
            # Add to search history
            st.session_state.search_history.append({
                'query': query,
                'results_count': len(results),
                'search_time': search_time,
                'timestamp': datetime.now().isoformat(),
                'mode': search_mode
            })
        
        # Display results with enhanced formatting
        if results:
            st.success(f"🎯 Found {len(results)} results in {search_time:.2f} seconds")
            
            # Results summary
            if len(results) > 5:
                file_types = {}
                avg_score = sum(r.score for r in results) / len(results)
                
                for result in results:
                    ft = result.file_type or 'unknown'
                    file_types[ft] = file_types.get(ft, 0) + 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Average relevance:** {avg_score:.3f}")
                with col2:
                    st.write(f"**File types:** {', '.join(f'{k}({v})' for k, v in file_types.items())}")
            
            # Display results
            for result in results:
                with st.expander(
                    f"#{result.rank} - {result.filename} "
                    f"(Score: {result.score:.3f}, {result.word_count} words)"
                ):
                    # Result metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if result.file_type:
                            st.badge(result.file_type.upper())
                    with col2:
                        st.write(f"📄 {result.word_count:,} words")
                    with col3:
                        st.write(f"🎯 Score: {result.score:.3f}")
                    
                    # Explanation
                    if show_explanations and result.relevance_explanation:
                        st.info(f"💡 **Why this matches:** {result.relevance_explanation}")
                    
                    # Content preview
                    st.markdown("**Preview:**")
                    st.markdown(result.snippet)
                    
                    # Full content toggle
                    if not show_snippets_only:
                        if st.button(f"📖 Show full content", key=f"show_{result.rank}"):
                            st.markdown("**Full Content:**")
                            st.text_area(
                                "Document content",
                                value=result.content,
                                height=400,
                                key=f"content_{result.rank}"
                            )
                    
                    # Export option
                    if st.button(f"💾 Export this result", key=f"export_{result.rank}"):
                        export_data = {
                            'filename': result.filename,
                            'query': query,
                            'score': result.score,
                            'snippet': result.snippet,
                            'content': result.content
                        }
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"search_result_{result.rank}.json",
                            mime="application/json",
                            key=f"download_{result.rank}"
                        )
        else:
            st.warning("🔍 No results found. Try:")
            st.write("• Different keywords or phrases")
            st.write("• Checking your filters")
            st.write("• Using broader search terms")
            st.write("• Trying semantic search mode")

def render_enhanced_analytics():
    """Render enhanced analytics dashboard"""
    st.header("📈 Advanced Analytics")
    
    # Quick stats
    stats = st.session_state.search_engine.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", stats['total_documents'])
    
    with col2:
        if st.session_state.search_history:
            total_searches = len(st.session_state.search_history)
            st.metric("Searches", total_searches)
        else:
            st.metric("Searches", 0)
    
    with col3:
        if stats.get('duplicates_found'):
            st.metric("Duplicates", stats['duplicates_found'])
        else:
            st.metric("Duplicates", 0)
    
    with col4:
        if stats.get('clusters'):
            st.metric("Clusters", stats['clusters'])
        else:
            st.metric("Clusters", 0)
    
    # Search performance
    if st.session_state.search_history:
        st.subheader("🚀 Search Performance")
        
        search_times = [s['search_time'] for s in st.session_state.search_history]
        result_counts = [s['results_count'] for s in st.session_state.search_history]
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_time = sum(search_times) / len(search_times)
            st.metric("Avg Search Time", f"{avg_time:.2f}s")
            st.write(f"Fastest: {min(search_times):.2f}s")
            st.write(f"Slowest: {max(search_times):.2f}s")
        
        with col2:
            avg_results = sum(result_counts) / len(result_counts)
            st.metric("Avg Results", f"{avg_results:.1f}")
            st.write(f"Best: {max(result_counts)} results")
            st.write(f"Worst: {min(result_counts)} results")
    
    # Document quality analysis
    if st.session_state.documents:
        st.subheader("📊 Document Quality Analysis")
        
        quality_scores = [doc.get('quality_score', 0) for doc in st.session_state.documents if doc.get('quality_score')]
        
        if quality_scores:
            col1, col2 = st.columns(2)
            
            with col1:
                avg_quality = sum(quality_scores) / len(quality_scores)
                st.metric("Avg Quality Score", f"{avg_quality:.2f}")
                
                high_quality = len([s for s in quality_scores if s > 0.7])
                st.write(f"High quality docs: {high_quality}")
                
            with col2:
                quality_distribution = {
                    'Excellent (0.8+)': len([s for s in quality_scores if s >= 0.8]),
                    'Good (0.6-0.8)': len([s for s in quality_scores if 0.6 <= s < 0.8]),
                    'Fair (0.4-0.6)': len([s for s in quality_scores if 0.4 <= s < 0.6]),
                    'Poor (<0.4)': len([s for s in quality_scores if s < 0.4])
                }
                
                for category, count in quality_distribution.items():
                    st.write(f"{category}: {count}")
    
    # Recent search history
    st.subheader("🔍 Recent Search History")
    
    if st.session_state.search_history:
        for i, search in enumerate(reversed(st.session_state.search_history[-10:])):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{search['query']}**")
            
            with col2:
                st.write(f"{search['results_count']} results")
            
            with col3:
                st.write(f"{search['search_time']:.2f}s")
    else:
        st.info("No search history yet. Perform some searches to see analytics.")
    
    # Export analytics
    if st.button("📊 Export Analytics"):
        analytics_data = {
            'system_stats': stats,
            'search_history': st.session_state.search_history,
            'generated_at': datetime.now().isoformat()
        }
        
        st.download_button(
            "Download Analytics JSON",
            data=json.dumps(analytics_data, indent=2),
            file_name=f"search_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Enhanced main application function"""
    # Set page config
    st.set_page_config(
        page_title="Advanced AI Document Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    initialize_session_state()
    
    # Header with enhanced branding
    st.title("🔍 Advanced AI Document Search Engine")
    st.markdown("**Next-generation document search with AI-powered semantic understanding, clustering, and advanced analytics**")
    
    # Feature highlights
    feature_cols = st.columns(4)
    
    with feature_cols[0]:
        st.info("🤖 **AI Semantic Search**\nUnderstand meaning, not just keywords")
    
    with feature_cols[1]:
        st.info("📊 **Smart Clustering**\nAutomatically organize documents")
    
    with feature_cols[2]:
        st.info("🔍 **Duplicate Detection**\nIdentify and manage duplicates")
    
    with feature_cols[3]:
        st.info("📈 **Advanced Analytics**\nTrack performance and insights")
    
    # Sidebar
    render_enhanced_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "🔍 Advanced Search", "📈 Analytics Dashboard"])
    
    with tab1:
        render_enhanced_upload_section()
    
    with tab2:
        render_enhanced_search_section()
    
    with tab3:
        render_enhanced_analytics()
    
    # Footer with system info
    st.markdown("---")
    
    footer_cols = st.columns(3)
    
    with footer_cols[0]:
        st.markdown("*Powered by Advanced AI and Machine Learning*")
    
    with footer_cols[1]:
        if AI_AVAILABLE:
            st.markdown("✅ **AI Features Active**")
        else:
            st.markdown("⚠️ **Basic Mode** - Install AI libraries for full features")
    
    with footer_cols[2]:
        if st.session_state.documents:
            st.markdown(f"📚 **{len(st.session_state.documents)} documents indexed**")

if __name__ == "__main__":
    main()
