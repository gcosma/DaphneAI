# ===============================================
# RAG EXTRACTOR BACKEND MODULE
# modules/rag_extractor.py
# Intelligent RAG-based extraction engine
# ===============================================

import logging
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_rag_available() -> bool:
    """Check if RAG dependencies are available"""
    try:
        import sentence_transformers
        import sklearn
        return True
    except ImportError:
        return False

def get_rag_status() -> Dict[str, Any]:
    """Get detailed RAG system status"""
    status = {
        'dependencies_available': False,
        'models_loaded': False,
        'backend_ready': False
    }
    
    try:
        import sentence_transformers
        import sklearn
        status['dependencies_available'] = True
        
        # Try to load a small model to test
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        status['models_loaded'] = True
        status['backend_ready'] = True
        
    except Exception as e:
        logger.warning(f"RAG status check failed: {e}")
    
    return status

class IntelligentRAGExtractor:
    """Advanced RAG-based extraction with semantic understanding"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.recommendation_patterns = [
            r'(?:recommend|suggests?|should|must|ought to|proposes?)[^.!?]*[.!?]',
            r'(?:The|A|An)\s+(?:Department|Ministry|Government|Authority|Committee|Panel)\s+(?:should|must|ought to|recommends?)[^.!?]*[.!?]',
            r'(?:We|I)\s+(?:recommend|suggest|propose)[^.!?]*[.!?]',
            r'(?:It is|This is)\s+(?:recommended|suggested|proposed)[^.!?]*[.!?]',
            r'(?:Action|Steps?|Measures?)\s+(?:should|must|ought to)\s+be\s+taken[^.!?]*[.!?]'
        ]
        self.response_patterns = [
            r'(?:accept|accepted|agree|agreed|implement|implemented|will)[^.!?]*[.!?]',
            r'(?:The Government|We|The Department)\s+(?:accept|agree|will|has|have)[^.!?]*[.!?]',
            r'(?:In response|Response|Reply)[^.!?]*[.!?]',
            r'(?:This|That)\s+(?:recommendation|suggestion)\s+(?:is|has been|will be)[^.!?]*[.!?]',
            r'(?:Action|Implementation|Progress)[^.!?]*[.!?]'
        ]
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            if is_rag_available():
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("✅ RAG model initialized successfully")
            else:
                self.logger.warning("⚠️ RAG dependencies not available, using fallback mode")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RAG model: {e}")
            self.model = None
    
    def extract_with_rag(self, content: str, filename: str, chunk_size: int = 500, max_items: int = 50) -> Dict[str, Any]:
        """Main RAG extraction method"""
        try:
            self.logger.info(f"Starting RAG extraction for {filename}")
            
            # Step 1: Smart chunking
            chunks = self._smart_chunk_document(content, chunk_size)
            self.logger.info(f"Created {len(chunks)} semantic chunks")
            
            # Step 2: Find relevant chunks using semantic search
            recommendation_chunks = self._find_relevant_chunks(chunks, "recommendations suggestions should must")
            response_chunks = self._find_relevant_chunks(chunks, "response accept implement agreed government")
            
            self.logger.info(f"Found {len(recommendation_chunks)} recommendation chunks, {len(response_chunks)} response chunks")
            
            # Step 3: Extract from relevant chunks
            recommendations = self._extract_from_chunks(recommendation_chunks, 'recommendation', max_items // 2)
            responses = self._extract_from_chunks(response_chunks, 'response', max_items // 2)
            
            # Step 4: Post-process and validate
            recommendations = self._post_process_extractions(recommendations, 'recommendation')
            responses = self._post_process_extractions(responses, 'response')
            
            self.logger.info(f"Extracted {len(recommendations)} recommendations, {len(responses)} responses")
            
            return {
                'recommendations': recommendations,
                'responses': responses,
                'metadata': {
                    'chunks_processed': len(chunks),
                    'recommendation_chunks': len(recommendation_chunks),
                    'response_chunks': len(response_chunks),
                    'method': 'rag_intelligent',
                    'model_used': 'sentence-transformers/all-MiniLM-L6-v2' if self.model else 'fallback',
                    'extraction_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"RAG extraction failed: {e}")
            # Fallback to basic extraction
            return self._fallback_extraction(content, filename, max_items)
    
    def _smart_chunk_document(self, content: str, chunk_size: int) -> List[Dict[str, Any]]:
        """Create semantic chunks that preserve meaning"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'index': chunk_index,
                    'length': len(current_chunk),
                    'type': 'semantic'
                })
                current_chunk = para
                chunk_index += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'index': chunk_index,
                'length': len(current_chunk),
                'type': 'semantic'
            })
        
        return chunks
    
    def _find_relevant_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Find chunks most relevant to the query using semantic similarity"""
        if not self.model or not chunks:
            # Fallback: simple keyword matching
            return self._keyword_filter_chunks(chunks, query)
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get embeddings for query and chunks
            query_embedding = self.model.encode([query])
            chunk_texts = [chunk['content'] for chunk in chunks]
            chunk_embeddings = self.model.encode(chunk_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Rank chunks by similarity
            ranked_chunks = []
            for i, similarity in enumerate(similarities):
                if similarity > 0.3:  # Threshold for relevance
                    chunk = chunks[i].copy()
                    chunk['similarity_score'] = float(similarity)
                    ranked_chunks.append(chunk)
            
            # Sort by similarity score
            ranked_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return ranked_chunks[:10]  # Return top 10 most relevant chunks
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity failed, using keyword fallback: {e}")
            return self._keyword_filter_chunks(chunks, query)
    
    def _keyword_filter_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Fallback method using keyword matching"""
        keywords = query.lower().split()
        relevant_chunks = []
        
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            score = sum(1 for keyword in keywords if keyword in content_lower)
            
            if score > 0:
                chunk_copy = chunk.copy()
                chunk_copy['keyword_score'] = score
                relevant_chunks.append(chunk_copy)
        
        # Sort by keyword score
        relevant_chunks.sort(key=lambda x: x.get('keyword_score', 0), reverse=True)
        return relevant_chunks[:10]
    
    def _extract_from_chunks(self, chunks: List[Dict], extraction_type: str, max_items: int) -> List[Dict]:
        """Extract recommendations or responses from relevant chunks"""
        extractions = []
        patterns = self.recommendation_patterns if extraction_type == 'recommendation' else self.response_patterns
        
        for chunk in chunks:
            content = chunk['content']
            
            # Apply all patterns
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    matched_text = match.group().strip()
                    
                    # Skip if too short or already found
                    if len(matched_text) < 50:
                        continue
                    
                    # Check for duplicates
                    if any(self._is_similar_text(matched_text, existing['content']) 
                          for existing in extractions):
                        continue
                    
                    # Calculate confidence score
                    confidence = self._calculate_extraction_confidence(
                        matched_text, chunk, extraction_type
                    )
                    
                    extraction = {
                        'content': matched_text,
                        'confidence': confidence,
                        'chunk_index': chunk['index'],
                        'similarity_score': chunk.get('similarity_score', chunk.get('keyword_score', 0)),
                        'pattern_used': pattern[:50] + "..." if len(pattern) > 50 else pattern,
                        'extraction_type': extraction_type,
                        'context': content[max(0, match.start()-100):match.end()+100]
                    }
                    
                    extractions.append(extraction)
                    
                    if len(extractions) >= max_items:
                        break
                
                if len(extractions) >= max_items:
                    break
            
            if len(extractions) >= max_items:
                break
        
        return extractions
    
    def _calculate_extraction_confidence(self, text: str, chunk: Dict, extraction_type: str) -> float:
        """Calculate confidence score for an extraction"""
        confidence = 0.5  # Base confidence
        
        # Length factor
        if len(text) > 100:
            confidence += 0.1
        if len(text) > 200:
            confidence += 0.1
        
        # Semantic similarity factor
        similarity = chunk.get('similarity_score', chunk.get('keyword_score', 0))
        if isinstance(similarity, (int, float)):
            confidence += min(similarity * 0.3, 0.3)
        
        # Pattern quality factor
        if extraction_type == 'recommendation':
            keywords = ['recommend', 'should', 'must', 'suggest', 'propose']
        else:
            keywords = ['accept', 'implement', 'agree', 'response', 'action']
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        confidence += min(keyword_matches * 0.05, 0.2)
        
        # Structure factor
        if re.search(r'^\d+\.', text.strip()):  # Numbered item
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _is_similar_text(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar (basic implementation)"""
        # Simple similarity check based on word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))
        
        return similarity > 0.7
    
    def _post_process_extractions(self, extractions: List[Dict], extraction_type: str) -> List[Dict]:
        """Post-process extractions to improve quality"""
        if not extractions:
            return extractions
        
        # Sort by confidence
        extractions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates more thoroughly
        unique_extractions = []
        for extraction in extractions:
            is_duplicate = False
            for existing in unique_extractions:
                if self._is_similar_text(extraction['content'], existing['content']):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if extraction['confidence'] > existing['confidence']:
                        unique_extractions.remove(existing)
                        unique_extractions.append(extraction)
                    break
            
            if not is_duplicate:
                unique_extractions.append(extraction)
        
        # Add final confidence scores
        for extraction in unique_extractions:
            extraction['final_confidence'] = self._calculate_final_confidence(extraction)
        
        return unique_extractions
    
    def _calculate_final_confidence(self, extraction: Dict) -> float:
        """Calculate final confidence score with validation"""
        base_confidence = extraction.get('confidence', 0.5)
        
        # Content quality factors
        content = extraction.get('content', '')
        
        # Length quality
        length_score = min(len(content) / 200, 1.0) * 0.1
        
        # Completeness (ends with punctuation)
        completeness_score = 0.1 if content.strip().endswith(('.', '!', '?')) else 0
        
        # Structure quality (starts with capital, proper grammar indicators)
        structure_score = 0.1 if content.strip()[0].isupper() else 0
        
        final_confidence = min(base_confidence + length_score + completeness_score + structure_score, 1.0)
        
        return round(final_confidence, 3)
    
    def _fallback_extraction(self, content: str, filename: str, max_items: int) -> Dict[str, Any]:
        """Fallback extraction method when RAG is not available"""
        self.logger.info("Using fallback extraction method")
        
        recommendations = []
        responses = []
        
        # Basic pattern extraction
        for pattern in self.recommendation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                text = match.group().strip()
                if len(text) >= 50 and len(recommendations) < max_items // 2:
                    recommendations.append({
                        'content': text,
                        'confidence': 0.6,
                        'extraction_type': 'recommendation',
                        'method': 'fallback_pattern'
                    })
        
        for pattern in self.response_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                text = match.group().strip()
                if len(text) >= 50 and len(responses) < max_items // 2:
                    responses.append({
                        'content': text,
                        'confidence': 0.6,
                        'extraction_type': 'response',
                        'method': 'fallback_pattern'
                    })
        
        return {
            'recommendations': recommendations,
            'responses': responses,
            'metadata': {
                'method': 'fallback_pattern',
                'extraction_timestamp': datetime.now().isoformat(),
                'note': 'RAG not available, used basic patterns'
            }
        }

# Additional utility functions
def validate_rag_extraction(results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate RAG extraction results"""
    validation = {
        'total_items': len(results.get('recommendations', [])) + len(results.get('responses', [])),
        'avg_confidence': 0.0,
        'high_confidence_items': 0,
        'method_used': results.get('metadata', {}).get('method', 'unknown')
    }
    
    all_items = results.get('recommendations', []) + results.get('responses', [])
    
    if all_items:
        confidences = [item.get('final_confidence', item.get('confidence', 0)) for item in all_items]
        validation['avg_confidence'] = sum(confidences) / len(confidences)
        validation['high_confidence_items'] = sum(1 for c in confidences if c > 0.8)
    
    return validation

def merge_similar_extractions(extractions: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """Merge extractions that are very similar"""
    if not extractions:
        return extractions
    
    merged = []
    processed = set()
    
    for i, extraction in enumerate(extractions):
        if i in processed:
            continue
        
        similar_group = [extraction]
        processed.add(i)
        
        for j, other in enumerate(extractions[i+1:], i+1):
            if j in processed:
                continue
            
            # Check similarity
            extractor = IntelligentRAGExtractor()
            if extractor._is_similar_text(extraction['content'], other['content']):
                similar_group.append(other)
                processed.add(j)
        
        # Merge the group - keep the one with highest confidence
        best_extraction = max(similar_group, key=lambda x: x.get('final_confidence', x.get('confidence', 0)))
        merged.append(best_extraction)
    
    return merged

# Export key functions
__all__ = [
    'IntelligentRAGExtractor',
    'is_rag_available', 
    'get_rag_status',
    'validate_rag_extraction',
    'merge_similar_extractions'
]
