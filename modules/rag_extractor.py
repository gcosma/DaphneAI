# ===============================================
# MODULAR RAG EXTRACTOR - SEPARATE FILE
# modules/rag_extractor.py
# Clean, focused RAG implementation
# ===============================================

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import dependencies with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    RAG_AVAILABLE = True
    logging.info("✅ RAG dependencies available")
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("⚠️ RAG dependencies not available")

class IntelligentRAGExtractor:
    """
    Intelligent RAG-based extraction system
    Focused, clean implementation for maximum accuracy
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentence_model = None
        self.extraction_templates = self._load_templates()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize RAG models"""
        try:
            if RAG_AVAILABLE:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("✅ RAG models initialized")
            else:
                self.logger.warning("⚠️ RAG models not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG models: {e}")
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load extraction query templates"""
        return {
            'recommendation_queries': [
                "government inquiry recommendation committee recommends that",
                "department should must implement establish create",
                "recommendation numbered formal official action required"
            ],
            'response_queries': [
                "government response accepted rejected recommendation",
                "ministry department official response implementation",
                "accepted partially not accepted government position"
            ]
        }
    
    def extract_with_rag(self, document_text: str, filename: str = "", 
                        chunk_size: int = 500, max_items: int = 25) -> Dict[str, Any]:
        """
        Main RAG extraction function
        """
        try:
            # Step 1: Intelligent chunking
            chunks = self._smart_chunking(document_text, chunk_size)
            
            # Step 2: Create embeddings if possible
            embeddings = []
            if self.sentence_model and chunks:
                embeddings = self.sentence_model.encode(chunks)
            
            # Step 3: RAG extraction
            recommendations = self._rag_extract_content(
                document_text, chunks, embeddings, 'recommendation'
            )
            responses = self._rag_extract_content(
                document_text, chunks, embeddings, 'response'
            )
            
            # Step 4: Post-process and validate
            final_recommendations = self._post_process(recommendations, max_items)
            final_responses = self._post_process(responses, max_items)
            
            return {
                'recommendations': final_recommendations,
                'responses': final_responses,
                'metadata': {
                    'method': 'rag_intelligent',
                    'chunks_processed': len(chunks),
                    'rag_enabled': bool(self.sentence_model),
                    'filename': filename
                }
            }
            
        except Exception as e:
            self.logger.error(f"RAG extraction failed: {e}")
            return self._fallback_extraction(document_text)
    
    def _smart_chunking(self, text: str, chunk_size: int = 500) -> List[str]:
        """Smart chunking that preserves semantic boundaries"""
        # Split by double newlines (paragraphs) first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _rag_extract_content(self, full_text: str, chunks: List[str], 
                           embeddings: List, content_type: str) -> List[Dict]:
        """RAG-based content extraction"""
        extractions = []
        
        # Get relevant query templates
        query_key = f'{content_type}_queries'
        queries = self.extraction_templates.get(query_key, [])
        
        if self.sentence_model and len(embeddings) > 0 and queries:
            # Create query embeddings
            query_embeddings = self.sentence_model.encode(queries)
            
            # Find relevant chunks using semantic similarity
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk_embedding = embeddings[i].reshape(1, -1)
                    
                    # Calculate similarity to all queries
                    similarities = []
                    for query_emb in query_embeddings:
                        sim = cosine_similarity(chunk_embedding, query_emb.reshape(1, -1))[0][0]
                        similarities.append(sim)
                    
                    max_similarity = max(similarities)
                    
                    # Extract from relevant chunks
                    if max_similarity > 0.25:  # Similarity threshold
                        chunk_extractions = self._extract_from_chunk(
                            chunk, content_type, max_similarity
                        )
                        extractions.extend(chunk_extractions)
        
        # Also do pattern-based extraction as backup
        pattern_extractions = self._pattern_extract(full_text, content_type)
        
        # Merge results intelligently
        merged = self._merge_extractions(extractions, pattern_extractions)
        
        return merged
    
    def _extract_from_chunk(self, chunk: str, content_type: str, similarity: float) -> List[Dict]:
        """Extract content from a relevant chunk"""
        extractions = []
        
        # Define extraction patterns based on content type
        if content_type == 'recommendation':
            patterns = [
                r'(?i)recommendation\s+(\d+(?:\.\d+)*)[:\.\-]?\s*(.{15,300})',
                r'(?i)(?:we\s+)?recommend\s+that\s+(.{15,200})',
                r'(?i)(?:should|must|ought\s+to)\s+(.{15,150})',
                r'(?i)(?:the\s+)?(?:committee|inquiry|panel)\s+recommends?\s+(.{15,200})'
            ]
        else:  # response
            patterns = [
                r'(?i)(?:government\s+)?response[:\.\-]?\s*(.{15,200})',
                r'(?i)(?:accepted|rejected|not\s+accepted|partially\s+accepted)[:\.\-]?\s*(.{15,150})',
                r'(?i)the\s+government\s+(?:accepts?|rejects?|acknowledges?)\s+(.{15,150})'
            ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, chunk, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                # Extract the matched text
                if len(match.groups()) > 1:
                    text = match.group(2).strip()
                else:
                    text = match.group(1).strip()
                
                # Clean and validate
                text = self._clean_text(text)
                
                if self._is_valid_extraction(text):
                    confidence = min(0.9, 0.6 + similarity * 0.3)
                    
                    extraction = {
                        'text': text,
                        'confidence': confidence,
                        'extraction_method': 'rag_semantic',
                        'similarity_score': similarity,
                        'content_type': content_type,
                        'word_count': len(text.split()),
                        'char_count': len(text),
                        'extracted_at': datetime.now().isoformat()
                    }
                    
                    # Add response type for responses
                    if content_type == 'response':
                        extraction['response_type'] = self._classify_response(text)
                    
                    extractions.append(extraction)
        
        return extractions
    
    def _pattern_extract(self, text: str, content_type: str) -> List[Dict]:
        """Simple pattern-based extraction as backup"""
        extractions = []
        lines = text.split('\n')
        
        if content_type == 'recommendation':
            pattern = r'(?i)^\s*(?:recommendation\s+\d+|we\s+recommend|should\s+)'
        else:
            pattern = r'(?i)^\s*(?:response|government\s+response|accepted|rejected)'
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if re.match(pattern, line):
                # Collect multi-line content
                content_lines = [line]
                j = i + 1
                
                while j < len(lines) and j < i + 5:  # Max 5 lines
                    next_line = lines[j].strip()
                    if not next_line:  # Empty line
                        break
                    if re.match(r'(?i)^\s*(?:recommendation|response)', next_line):
                        break  # Next item
                    content_lines.append(next_line)
                    j += 1
                
                full_text = ' '.join(content_lines)
                full_text = self._clean_text(full_text)
                
                if self._is_valid_extraction(full_text):
                    extraction = {
                        'text': full_text,
                        'confidence': 0.7,
                        'extraction_method': 'pattern_backup',
                        'content_type': content_type,
                        'word_count': len(full_text.split()),
                        'char_count': len(full_text),
                        'extracted_at': datetime.now().isoformat()
                    }
                    
                    if content_type == 'response':
                        extraction['response_type'] = self._classify_response(full_text)
                    
                    extractions.append(extraction)
                
                i = j
            else:
                i += 1
        
        return extractions
    
    def _merge_extractions(self, rag_items: List[Dict], pattern_items: List[Dict]) -> List[Dict]:
        """Merge RAG and pattern extractions, avoiding duplicates"""
        merged = []
        
        # Add high-confidence RAG items first
        for item in rag_items:
            if item.get('confidence', 0) > 0.7:
                merged.append(item)
        
        # Add pattern items that don't significantly overlap
        for pattern_item in pattern_items:
            is_duplicate = False
            
            for existing in merged:
                similarity = self._calculate_overlap(
                    pattern_item.get('text', ''), 
                    existing.get('text', '')
                )
                if similarity > 0.7:  # High overlap
                    is_duplicate = True
                    # Boost confidence of existing item
                    existing['confidence'] = min(0.95, existing['confidence'] + 0.1)
                    existing['pattern_confirmed'] = True
                    break
            
            if not is_duplicate:
                merged.append(pattern_item)
        
        # Add remaining RAG items
        for item in rag_items:
            if item.get('confidence', 0) <= 0.7:
                is_duplicate = any(
                    self._calculate_overlap(item.get('text', ''), existing.get('text', '')) > 0.7
                    for existing in merged
                )
                if not is_duplicate:
                    merged.append(item)
        
        return merged
    
    def _post_process(self, extractions: List[Dict], max_items: int) -> List[Dict]:
        """Post-process and filter extractions"""
        # Calculate final confidence scores
        for extraction in extractions:
            base_confidence = extraction.get('confidence', 0)
            text = extraction.get('text', '')
            
            # Quality bonuses
            quality_bonus = 0
            if text.strip().endswith('.'):
                quality_bonus += 0.1
            if len(text.split()) > 15:
                quality_bonus += 0.1
            if extraction.get('pattern_confirmed'):
                quality_bonus += 0.1
            
            final_confidence = min(0.95, base_confidence + quality_bonus)
            extraction['final_confidence'] = final_confidence
        
        # Filter and sort
        valid_extractions = [
            ext for ext in extractions 
            if ext.get('final_confidence', 0) > 0.4
        ]
        
        # Sort by confidence
        valid_extractions.sort(key=lambda x: x.get('final_confidence', 0), reverse=True)
        
        # Add ranking
        for i, extraction in enumerate(valid_extractions[:max_items]):
            extraction['rank'] = i + 1
            extraction['extraction_id'] = f"{extraction['content_type'].upper()}_{i+1:03d}"
        
        return valid_extractions[:max_items]
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common artifacts
        text = re.sub(r'^\W+', '', text)  # Leading non-word chars
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        
        return text.strip()
    
    def _is_valid_extraction(self, text: str) -> bool:
        """Check if extraction is valid"""
        if not text or len(text.strip()) < 15:
            return False
        
        word_count = len(text.split())
        if word_count < 5:
            return False
        
        # Check for meaningful content
        if not any(c.isalpha() for c in text):
            return False
        
        return True
    
    def _classify_response(self, text: str) -> str:
        """Classify government response type"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['accept', 'accepted', 'agree']):
            if 'partially' in text_lower or 'in part' in text_lower:
                return 'partially_accepted'
            else:
                return 'accepted'
        elif any(word in text_lower for word in ['reject', 'rejected', 'not accept', 'decline']):
            return 'rejected'
        elif any(word in text_lower for word in ['consider', 'review', 'under consideration']):
            return 'under_consideration'
        else:
            return 'government_response'
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap percentage"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback when RAG fails"""
        # Simple pattern extraction
        recommendations = self._pattern_extract(text, 'recommendation')
        responses = self._pattern_extract(text, 'response')
        
        return {
            'recommendations': recommendations[:10],
            'responses': responses[:10],
            'metadata': {
                'method': 'pattern_fallback',
                'note': 'RAG unavailable, using pattern extraction'
            }
        }

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def is_rag_available() -> bool:
    """Check if RAG dependencies are available"""
    return RAG_AVAILABLE

def get_rag_status() -> Dict[str, Any]:
    """Get detailed RAG system status"""
    status = {
        'available': RAG_AVAILABLE,
        'models_loaded': False,
        'dependencies': {
            'sentence_transformers': False,
            'sklearn': False,
            'numpy': False
        }
    }
    
    if RAG_AVAILABLE:
        try:
            # Test model loading
            model = SentenceTransformer('all-MiniLM-L6-v2')
            status['models_loaded'] = True
            status['dependencies'] = {
                'sentence_transformers': True,
                'sklearn': True,
                'numpy': True
            }
        except Exception as e:
            logging.warning(f"RAG model loading failed: {e}")
    
    return status

# Export main classes and functions
__all__ = [
    'IntelligentRAGExtractor',
    'is_rag_available',
    'get_rag_status',
    'RAG_AVAILABLE'
]
