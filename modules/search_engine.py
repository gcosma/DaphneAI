# government_search_engine.py
"""
Advanced Government Document Search Engine
Designed for searching recommendations, responses, and other government content
"""

import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Import stemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Optional AI imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

@dataclass
class SearchFragment:
    """Individual search result fragment"""
    document_id: str
    filename: str
    sentence: str
    context: str  # Surrounding sentences for context
    score: float
    match_type: str  # 'exact', 'semantic', 'partial'
    sentence_index: int
    highlights: List[str]

@dataclass
class DocumentSearchResult:
    """Complete search result for a document"""
    document_id: str
    filename: str
    total_score: float
    fragments: List[SearchFragment]
    document_type: str  # 'recommendation', 'response', 'policy', etc.

class GovernmentSearchEngine:
    """Advanced search engine for government documents"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.documents = []
        self.sentence_index = {}  # Maps doc_id -> list of sentences
        self.stemmed_sentence_index = {}  # Maps doc_id -> list of stemmed sentences
        self.semantic_model = None
        self.sentence_embeddings = {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Government-specific keywords
        self.gov_keywords = {
            'recommendation': ['recommend', 'recommendation', 'suggests', 'proposes', 'advises'],
            'response': ['response', 'accept', 'reject', 'implement', 'considers', 'agrees'],
            'policy': ['policy', 'regulation', 'guideline', 'framework', 'strategy'],
            'action': ['action', 'implementation', 'execute', 'deliver', 'establish'],
            'priority': ['urgent', 'priority', 'immediate', 'critical', 'essential']
        }
        
        # Initialize semantic model if available
        if AI_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Semantic search model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load semantic model: {e}")
                self.semantic_model = None
        else:
            self.logger.warning("AI libraries not available - using advanced keyword search only")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the search index"""
        self.documents = documents
        self._build_sentence_index()
        self._build_semantic_index()
    
    def _build_sentence_index(self) -> None:
        """Build sentence-level index for all documents with stemming"""
        self.sentence_index = {}
        self.stemmed_sentence_index = {}
        
        for i, doc in enumerate(self.documents):
            doc_id = doc.get('id', str(i))
            text = doc.get('text', doc.get('content', ''))
            
            if text:
                # Split into sentences
                sentences = sent_tokenize(text)
                self.sentence_index[doc_id] = sentences
                
                # Create stemmed versions
                stemmed_sentences = []
                for sentence in sentences:
                    words = word_tokenize(sentence.lower())
                    stemmed_words = [self.stemmer.stem(word) for word in words if word.isalnum()]
                    stemmed_sentences.append(' '.join(stemmed_words))
                
                self.stemmed_sentence_index[doc_id] = stemmed_sentences
                
        self.logger.info(f"Built sentence and stemmed index for {len(self.documents)} documents")
    
    def _build_semantic_index(self) -> None:
        """Build semantic embeddings for all sentences"""
        if not self.semantic_model:
            return
        
        try:
            self.sentence_embeddings = {}
            total_sentences = 0
            
            for doc_id, sentences in self.sentence_index.items():
                if sentences:
                    embeddings = self.semantic_model.encode(sentences)
                    self.sentence_embeddings[doc_id] = embeddings
                    total_sentences += len(sentences)
            
            self.logger.info(f"Built semantic embeddings for {total_sentences} sentences")
        except Exception as e:
            self.logger.error(f"Failed to build semantic index: {e}")
            self.sentence_embeddings = {}
    
    def search(self, query: str, search_type: str = 'comprehensive', 
               max_results_per_doc: int = 10, min_score: float = 0.1) -> List[DocumentSearchResult]:
        """
        Comprehensive search across all documents
        
        Args:
            query: Search query
            search_type: 'exact', 'semantic', 'comprehensive'
            max_results_per_doc: Maximum fragments per document
            min_score: Minimum relevance score threshold
        """
        start_time = time.time()
        
        if not query.strip():
            return []
        
        all_results = []
        
        for i, doc in enumerate(self.documents):
            doc_id = doc.get('id', str(i))
            
            if search_type == 'exact':
                fragments = self._exact_search_document(query, doc_id, doc)
            elif search_type == 'semantic':
                fragments = self._semantic_search_document(query, doc_id, doc)
            else:  # comprehensive
                exact_fragments = self._exact_search_document(query, doc_id, doc)
                semantic_fragments = self._semantic_search_document(query, doc_id, doc)
                fragments = self._merge_fragments(exact_fragments, semantic_fragments)
            
            # Filter by minimum score and limit results
            filtered_fragments = [f for f in fragments if f.score >= min_score]
            filtered_fragments.sort(key=lambda x: x.score, reverse=True)
            limited_fragments = filtered_fragments[:max_results_per_doc]
            
            if limited_fragments:
                total_score = sum(f.score for f in limited_fragments) / len(limited_fragments)
                doc_result = DocumentSearchResult(
                    document_id=doc_id,
                    filename=doc.get('filename', f'Document {i+1}'),
                    total_score=total_score,
                    fragments=limited_fragments,
                    document_type=self._classify_document_type(doc)
                )
                all_results.append(doc_result)
        
        # Sort documents by total score
        all_results.sort(key=lambda x: x.total_score, reverse=True)
        
        search_time = time.time() - start_time
        total_fragments = sum(len(result.fragments) for result in all_results)
        
        self.logger.info(f"Search completed in {search_time:.3f}s - Query: '{query}', "
                        f"Documents: {len(all_results)}, Fragments: {total_fragments}")
        
        return all_results
    
    def _exact_search_document(self, query: str, doc_id: str, doc: Dict) -> List[SearchFragment]:
        """Advanced exact search with stemming and context"""
        fragments = []
        sentences = self.sentence_index.get(doc_id, [])
        stemmed_sentences = self.stemmed_sentence_index.get(doc_id, [])
        
        if not sentences:
            return fragments
        
        # Prepare query variations for stemming-based matching
        query_variations = self._generate_stemmed_query_variations(query)
        
        for sentence_idx, (sentence, stemmed_sentence) in enumerate(zip(sentences, stemmed_sentences)):
            sentence_lower = sentence.lower()
            best_score = 0
            best_match_type = 'none'
            highlights = []
            
            # Test each query variation
            for variation, stemmed_variation, variation_score in query_variations:
                variation_lower = variation.lower()
                
                # 1. Exact phrase match (highest priority)
                if variation_lower in sentence_lower:
                    score = variation_score * 1.0
                    if score > best_score:
                        best_score = score
                        best_match_type = 'exact'
                        highlights = [variation]
                
                # 2. Stemmed phrase match
                if stemmed_variation in stemmed_sentence:
                    score = variation_score * 0.9
                    if score > best_score:
                        best_score = score
                        best_match_type = 'stemmed'
                        highlights = variation.split()
                
                # 3. Individual word matches (original and stemmed)
                original_words = variation.split()
                stemmed_words = stemmed_variation.split()
                
                word_matches = 0
                matched_words = []
                
                for orig_word, stem_word in zip(original_words, stemmed_words):
                    # Check original word
                    if orig_word.lower() in sentence_lower:
                        word_matches += 1
                        matched_words.append(orig_word)
                    # Check stemmed word
                    elif stem_word in stemmed_sentence:
                        word_matches += 1
                        # Find the original word in sentence that matches the stem
                        sentence_words = word_tokenize(sentence.lower())
                        for sent_word in sentence_words:
                            if self.stemmer.stem(sent_word) == stem_word:
                                matched_words.append(sent_word)
                                break
                
                if word_matches > 0:
                    word_score = (word_matches / len(original_words)) * variation_score * 0.7
                    if word_score > best_score:
                        best_score = word_score
                        best_match_type = 'partial'
                        highlights = matched_words
            
            # Government keyword boost
            gov_boost = self._calculate_government_relevance(sentence, query)
            best_score += gov_boost
            
            if best_score > 0:
                context = self._get_sentence_context(sentences, sentence_idx)
                
                fragment = SearchFragment(
                    document_id=doc_id,
                    filename=doc.get('filename', ''),
                    sentence=sentence,
                    context=context,
                    score=best_score,
                    match_type=best_match_type,
                    sentence_index=sentence_idx,
                    highlights=highlights
                )
                fragments.append(fragment)
        
        return fragments
    
    def _semantic_search_document(self, query: str, doc_id: str, doc: Dict) -> List[SearchFragment]:
        """Semantic search with contextual understanding"""
        fragments = []
        
        if not self.semantic_model or doc_id not in self.sentence_embeddings:
            return fragments
        
        try:
            sentences = self.sentence_index.get(doc_id, [])
            sentence_embeddings = self.sentence_embeddings[doc_id]
            
            # Encode query with government context
            gov_enhanced_query = self._enhance_query_with_context(query)
            query_embedding = self.semantic_model.encode([gov_enhanced_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
            
            for sentence_idx, (sentence, similarity) in enumerate(zip(sentences, similarities)):
                if similarity > 0.1:  # Minimum semantic threshold
                    context = self._get_sentence_context(sentences, sentence_idx)
                    
                    fragment = SearchFragment(
                        document_id=doc_id,
                        filename=doc.get('filename', ''),
                        sentence=sentence,
                        context=context,
                        score=float(similarity),
                        match_type='semantic',
                        sentence_index=sentence_idx,
                        highlights=self._extract_semantic_highlights(sentence, query)
                    )
                    fragments.append(fragment)
        
        except Exception as e:
            self.logger.error(f"Semantic search failed for doc {doc_id}: {e}")
        
        return fragments
    
    def _generate_stemmed_query_variations(self, query: str) -> List[Tuple[str, str, float]]:
        """Generate query variations with original, stemmed forms, and importance scores"""
        variations = []
        
        # Original query (highest priority)
        stemmed_query = self._stem_text(query)
        variations.append((query, stemmed_query, 1.0))
        
        # Remove common stopwords
        words = query.split()
        important_words = [w for w in words if w.lower() not in self.stop_words]
        if len(important_words) != len(words):
            important_query = ' '.join(important_words)
            stemmed_important = self._stem_text(important_query)
            variations.append((important_query, stemmed_important, 0.9))
        
        # Individual important words with their stems
        for word in important_words:
            if len(word) > 3:  # Skip very short words
                stemmed_word = self.stemmer.stem(word.lower())
                variations.append((word, stemmed_word, 0.6))
        
        # Government synonyms and variations
        gov_variations = self._generate_government_stemmed_variations(query)
        variations.extend(gov_variations)
        
        return variations
    
    def _stem_text(self, text: str) -> str:
        """Stem a text string"""
        words = word_tokenize(text.lower())
        stemmed_words = [self.stemmer.stem(word) for word in words if word.isalnum()]
        return ' '.join(stemmed_words)
    
    def _generate_government_stemmed_variations(self, query: str) -> List[Tuple[str, str, float]]:
        """Generate government-specific query variations with stemming"""
        variations = []
        query_lower = query.lower()
        
        # Synonym mapping for government terms
        gov_synonyms = {
            'recommendation': ['advice', 'suggestion', 'proposal', 'guidance'],
            'response': ['reply', 'answer', 'reaction', 'feedback'],
            'implement': ['execute', 'carry out', 'deliver', 'enact'],
            'policy': ['strategy', 'framework', 'guideline', 'directive'],
            'government': ['department', 'ministry', 'authority', 'administration'],
            'accept': ['approve', 'endorse', 'support', 'agree'],
            'reject': ['decline', 'refuse', 'dismiss', 'oppose']
        }
        
        for original, synonyms in gov_synonyms.items():
            if original in query_lower:
                for synonym in synonyms:
                    new_query = query_lower.replace(original, synonym)
                    stemmed_new_query = self._stem_text(new_query)
                    variations.append((new_query, stemmed_new_query, 0.8))
        
        return variations
    
    def _calculate_government_relevance(self, sentence: str, query: str) -> float:
        """Calculate additional relevance for government-specific content with stemming"""
        boost = 0.0
        sentence_lower = sentence.lower()
        query_lower = query.lower()
        
        # Stem both sentence and query for better matching
        stemmed_sentence = self._stem_text(sentence)
        stemmed_query = self._stem_text(query)
        
        # Boost for government keywords (check both original and stemmed)
        for category, keywords in self.gov_keywords.items():
            for keyword in keywords:
                stemmed_keyword = self.stemmer.stem(keyword)
                
                # Check original forms
                if keyword in sentence_lower:
                    if keyword in query_lower:
                        boost += 0.3  # Query contains government term
                    else:
                        boost += 0.1  # Sentence contains government term
                
                # Check stemmed forms
                elif stemmed_keyword in stemmed_sentence:
                    if stemmed_keyword in stemmed_query:
                        boost += 0.25  # Stemmed query matches stemmed sentence
                    else:
                        boost += 0.08  # Stemmed sentence contains government term
        
        # Boost for numbered recommendations/responses
        if re.search(r'\b\d+[.)\]]\s*(recommendation|response)', sentence_lower):
            boost += 0.2
        
        # Boost for action words near query terms (with stemming)
        action_words = ['accept', 'reject', 'implement', 'consider', 'propose']
        stemmed_action_words = [self.stemmer.stem(word) for word in action_words]
        
        query_words = query_lower.split()
        stemmed_query_words = stemmed_query.split()
        
        for action, stemmed_action in zip(action_words, stemmed_action_words):
            # Check original action words
            if action in sentence_lower and any(word in sentence_lower for word in query_words):
                boost += 0.15
            # Check stemmed action words
            elif stemmed_action in stemmed_sentence and any(word in stemmed_sentence for word in stemmed_query_words):
                boost += 0.12
        
        return min(boost, 0.5)  # Cap the boost
    
    def _enhance_query_with_context(self, query: str) -> str:
        """Enhance query with government context for better semantic search"""
        enhanced = query
        
        # Add government context if not present
        gov_terms = ['government', 'policy', 'recommendation', 'response']
        if not any(term in query.lower() for term in gov_terms):
            enhanced = f"government policy {query}"
        
        return enhanced
    
    def _extract_semantic_highlights(self, sentence: str, query: str) -> List[str]:
        """Extract relevant highlights from semantic matches with stemming awareness"""
        words = query.split()
        highlights = []
        
        # Find exact word matches
        for word in words:
            if len(word) > 2 and word.lower() in sentence.lower():
                highlights.append(word)
        
        # Find stemmed matches
        sentence_words = word_tokenize(sentence.lower())
        query_stems = [self.stemmer.stem(word.lower()) for word in words]
        
        for sent_word in sentence_words:
            if len(sent_word) > 3:
                sent_stem = self.stemmer.stem(sent_word)
                if sent_stem in query_stems and sent_word not in [h.lower() for h in highlights]:
                    highlights.append(sent_word)
        
        # Add semantically related terms (simplified)
        for word in sentence_words:
            if len(word) > 4 and word.lower() not in [h.lower() for h in highlights]:
                # Simple heuristic for semantic relation using stemming
                for query_word in words:
                    query_stem = self.stemmer.stem(query_word.lower())
                    word_stem = self.stemmer.stem(word.lower())
                    
                    # If stems are similar or word contains query stem
                    if (query_stem in word_stem or word_stem in query_stem) and len(query_word) > 3:
                        highlights.append(word)
                        break
        
        return highlights[:5]  # Limit highlights
    
    def _get_sentence_context(self, sentences: List[str], sentence_idx: int, 
                            context_window: int = 2) -> str:
        """Get surrounding sentences for context"""
        start = max(0, sentence_idx - context_window)
        end = min(len(sentences), sentence_idx + context_window + 1)
        
        context_sentences = sentences[start:end]
        return ' '.join(context_sentences)
    
    def _merge_fragments(self, exact_fragments: List[SearchFragment], 
                        semantic_fragments: List[SearchFragment]) -> List[SearchFragment]:
        """Merge exact and semantic fragments, avoiding duplicates"""
        merged = []
        exact_sentences = {f.sentence_index for f in exact_fragments}
        
        # Add all exact matches
        merged.extend(exact_fragments)
        
        # Add semantic matches that don't overlap with exact matches
        for semantic_frag in semantic_fragments:
            if semantic_frag.sentence_index not in exact_sentences:
                merged.append(semantic_frag)
        
        return merged
    
    def _classify_document_type(self, doc: Dict) -> str:
        """Classify document type based on content"""
        text = doc.get('text', '').lower()
        filename = doc.get('filename', '').lower()
        
        if 'recommendation' in text or 'recommend' in text:
            return 'recommendation'
        elif 'response' in text or 'accept' in text or 'reject' in text:
            return 'response'
        elif 'policy' in text or 'policy' in filename:
            return 'policy'
        else:
            return 'general'
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        total_sentences = sum(len(sentences) for sentences in self.sentence_index.values())
        
        return {
            'total_documents': len(self.documents),
            'total_sentences': total_sentences,
            'semantic_enabled': self.semantic_model is not None,
            'embeddings_built': len(self.sentence_embeddings) > 0,
            'average_sentences_per_doc': total_sentences / len(self.documents) if self.documents else 0
        }


class SearchResultDisplay:
    """Display and formatting utilities for search results"""
    
    @staticmethod
    def format_results_for_display(results: List[DocumentSearchResult]) -> Dict[str, Any]:
        """Format search results for UI display"""
        total_fragments = sum(len(result.fragments) for result in results)
        
        return {
            'total_documents': len(results),
            'total_fragments': total_fragments,
            'results': results,
            'by_document_type': SearchResultDisplay._group_by_document_type(results)
        }
    
    @staticmethod
    def _group_by_document_type(results: List[DocumentSearchResult]) -> Dict[str, List[DocumentSearchResult]]:
        """Group results by document type"""
        grouped = {}
        for result in results:
            doc_type = result.document_type
            if doc_type not in grouped:
                grouped[doc_type] = []
            grouped[doc_type].append(result)
        return grouped
    
    @staticmethod
    def highlight_text(text: str, highlights: List[str]) -> str:
        """Add highlighting to text"""
        highlighted = text
        for highlight in highlights:
            pattern = re.compile(re.escape(highlight), re.IGNORECASE)
            highlighted = pattern.sub(f'**{highlight}**', highlighted)
        return highlighted
