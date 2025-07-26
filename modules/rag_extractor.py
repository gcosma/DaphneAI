# modules/rag_extractor.py
# COMPLETE ENHANCED RAG EXTRACTOR - Government Response Document Optimization

import logging
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from dataclasses import dataclass

# Core dependencies check
def is_rag_available() -> bool:
    """Check if RAG dependencies are available"""
    try:
        import sentence_transformers
        import sklearn
        import numpy
        return True
    except ImportError:
        return False

def get_rag_status() -> Dict[str, Any]:
    """Get detailed RAG system status"""
    status = {
        'rag_available': is_rag_available(),
        'models_loaded': False,
        'embedding_model': None,
        'vector_store_ready': False,
        'dependencies': {
            'sentence_transformers': False,
            'sklearn': False,
            'numpy': False
        }
    }
    
    # Check individual dependencies
    try:
        import sentence_transformers
        status['dependencies']['sentence_transformers'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        status['dependencies']['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        status['dependencies']['numpy'] = True
    except ImportError:
        pass
    
    return status

@dataclass
class ExtractionResult:
    """Results from RAG extraction process"""
    content: str
    confidence: float
    extraction_type: str
    source_chunk: str
    semantic_score: float
    pattern_score: float
    context: str
    position: int
    validation_passed: bool

class IntelligentRAGExtractor:
    """
    Enhanced RAG (Retrieval-Augmented Generation) Extractor
    Optimized for UK Government Response documents with semantic understanding
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the RAG extractor"""
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        # Model initialization
        self.model = None
        self.model_loaded = False
        
        # Extraction patterns optimized for Government Response documents
        self._initialize_government_patterns()
        
        # Semantic keywords for content discovery
        self._initialize_semantic_keywords()
        
        # Quality thresholds
        self.min_semantic_score = 0.6
        self.min_pattern_score = 0.7
        self.min_combined_confidence = 0.65
        
        # Try to initialize the model
        self._initialize_model()
        
        self.logger.info(f"RAG Extractor initialized - Model loaded: {self.model_loaded}")

    def _initialize_government_patterns(self):
        """Initialize patterns specifically for Government Response documents"""
        
        # Recommendation patterns from your actual documents
        self.recommendation_patterns = [
            # Complex numbered recommendations: "Recommendation 6a) iii)"
            r'(?i)(Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s*[:\.)]\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
            
            # Sub-recommendations with complex numbering
            r'(?i)Recommendation\s+(\d+[a-z]*\s*\)\s*[iv]+\s*\))\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
            
            # Progress and review requirements
            r'(?i)(Progress\s+(?:in\s+)?implementation|Review\s+of\s+progress)\s+(?:towards\s+|of\s+)?([^.!?]{20,400}[.!?])',
            
            # Framework establishment requirements
            r'(?i)(?:framework|system|process|mechanism)\s+(?:should|must|be)\s+(?:established|created|implemented)\s+([^.!?]{20,400}[.!?])',
            
            # Training and education requirements
            r'(?i)(?:training|education|guidance)\s+(?:should|must)\s+be\s+(?:provided|delivered|enhanced)\s+([^.!?]{20,400}[.!?])',
            
            # Specific action requirements
            r'(?i)(?:action|steps|measures)\s+(?:should|must)\s+be\s+taken\s+([^.!?]{20,400}[.!?])',
            
            # Bodies and organizations recommendations
            r'(?i)(?:bodies|organizations?)\s+(?:should|must)\s+(?:ensure|establish|develop)\s+([^.!?]{20,400}[.!?])',
            
            # Funding and resource recommendations
            r'(?i)(?:funding|resources?)\s+(?:should|must)\s+be\s+(?:provided|allocated|increased)\s+([^.!?]{20,400}[.!?])',
        ]
        
        # Response patterns from your actual documents
        self.response_patterns = [
            # Core acceptance patterns - MOST IMPORTANT
            r'(?i)(?:This\s+recommendation|Recommendation\s+\d+[a-z]*(?:\s*[)\]])*(?:\s*[iv]+)*)\s+is\s+(accepted\s+in\s+(?:full|principle)|not\s+accepted|partially\s+accepted|rejected)\s*(?:by\s+(?:the\s+)?(?:UK\s+Government|Scottish\s+Government|Welsh\s+Government|Northern\s+Ireland\s+Executive|Government))?\.?\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
            
            # "Accepting in principle" section headers
            r'(?i)(Accepting\s+in\s+principle)\s*([^.!?]*(?:[.!?]|(?=Recommendation)|$))',
            
            # Implementation commitments
            r'(?i)(Implementation\s+will\s+begin|Implementation\s+is\s+underway|Work\s+is\s+ongoing)\s*([^.!?]{20,500}[.!?])',
            
            # Government establishment commitments
            r'(?i)(?:The\s+(?:UK\s+)?Government|We|The\s+(?:Department|Ministry))\s+will\s+(establish|implement|develop|create)\s+([^.!?]{20,500}[.!?])',
            
            # Cross-nation acceptance patterns
            r'(?i)(?:UK\s+Government|Scottish\s+Government|Welsh\s+Government|Northern\s+Ireland\s+Executive)\s+(?:accepts?|agrees?|will)\s+([^.!?]{20,500}[.!?])',
            
            # "Parts of these recommendations" patterns
            r'(?i)(?:These\s+recommendations|Parts\s+of\s+these\s+recommendations)\s+are\s+(?:accepted|being\s+taken\s+forward)\s*([^.!?]{20,500}[.!?])',
            
            # Review and assessment responses
            r'(?i)(?:review|assessment|evaluation)\s+(?:has\s+been|is\s+being|will\s+be)\s+(?:conducted|undertaken|scheduled)\s*([^.!?]{20,500}[.!?])',
            
            # Funding and resource commitments
            r'(?i)(?:funding|resources?|investment)\s+(?:will\s+be|has\s+been)\s+(?:provided|allocated|committed)\s*([^.!?]{20,500}[.!?])',
            
            # Timeline and deadline commitments
            r'(?i)(?:within\s+\d+\s+(?:months?|years?)|by\s+\d+|during\s+the\s+(?:first|next))\s+([^.!?]{20,500}[.!?])',
            
            # Quality and safety management responses
            r'(?i)(?:quality\s+and\s+safety|patient\s+safety|safety\s+management)\s+(?:system|framework|approach)\s*([^.!?]{20,500}[.!?])',
            
            # Training and education responses
            r'(?i)(?:training|education|guidance)\s+(?:will\s+be|is\s+being|has\s+been)\s+(?:provided|delivered|enhanced|improved)\s*([^.!?]{20,500}[.!?])',
            
            # Stakeholder involvement responses
            r'(?i)(?:working\s+with|in\s+collaboration\s+with|together\s+with)\s+(?:[^.!?]{5,100})\s+(?:to\s+)?([^.!?]{20,500}[.!?])',
        ]

    def _initialize_semantic_keywords(self):
        """Initialize semantic keywords for content discovery"""
        
        self.recommendation_keywords = [
            'recommendation', 'recommend', 'suggest', 'proposal', 'should', 'must', 'ought',
            'action required', 'steps needed', 'measures', 'implement', 'establish', 'develop',
            'framework', 'system', 'process', 'mechanism', 'training', 'education', 'guidance',
            'review', 'assess', 'evaluate', 'monitor', 'progress', 'improvement', 'enhance'
        ]
        
        self.response_keywords = [
            'accept', 'accepted', 'agree', 'agreed', 'response', 'implementation', 'will begin',
            'government', 'department', 'ministry', 'establish', 'create', 'develop', 'funding',
            'resources', 'investment', 'timeline', 'months', 'years', 'quality', 'safety',
            'principle', 'full', 'rejected', 'partially', 'uk government', 'scottish government',
            'welsh government', 'northern ireland', 'working with', 'collaboration'
        ]

    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        if not is_rag_available():
            self.logger.warning("⚠️ RAG dependencies not available")
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight but effective model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_loaded = True
            self.logger.info("✅ RAG model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RAG model: {e}")
            self.model = None
            self.model_loaded = False

    def extract_with_rag(self, content: str, filename: str, 
                        chunk_size: int = 500, max_items: int = 50) -> Dict[str, Any]:
        """
        Main RAG extraction method with semantic understanding
        
        Args:
            content: Document text content
            filename: Source filename
            chunk_size: Size of semantic chunks
            max_items: Maximum items to extract
            
        Returns:
            Dict with extracted recommendations and responses
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting RAG extraction for {filename}")
            
            # Step 1: Smart semantic chunking
            chunks = self._smart_chunk_document(content, chunk_size)
            self.logger.info(f"Created {len(chunks)} semantic chunks")
            
            if not chunks:
                return self._create_empty_result(filename, "No content chunks created")
            
            # Step 2: Find semantically relevant chunks
            recommendation_chunks = self._find_relevant_chunks(
                chunks, self.recommendation_keywords, 'recommendation'
            )
            response_chunks = self._find_relevant_chunks(
                chunks, self.response_keywords, 'response'
            )
            
            self.logger.info(f"Found {len(recommendation_chunks)} recommendation chunks, {len(response_chunks)} response chunks")
            
            # Step 3: Extract from relevant chunks using patterns + semantic analysis
            recommendations = self._extract_from_chunks(
                recommendation_chunks, 'recommendation', max_items // 2
            )
            responses = self._extract_from_chunks(
                response_chunks, 'response', max_items // 2
            )
            
            # Step 4: Post-process and validate results
            recommendations = self._post_process_extractions(recommendations, 'recommendation')
            responses = self._post_process_extractions(responses, 'response')
            
            # Step 5: Final quality scoring and ranking
            recommendations = self._rank_by_quality(recommendations)
            responses = self._rank_by_quality(responses)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'recommendations': recommendations[:max_items // 2],
                'responses': responses[:max_items // 2],
                'processing_results': [{
                    'filename': filename,
                    'total_chunks': len(chunks),
                    'recommendation_chunks': len(recommendation_chunks),
                    'response_chunks': len(response_chunks),
                    'processing_time_seconds': round(processing_time, 2),
                    'extraction_method': 'rag_intelligent',
                    'model_used': 'all-MiniLM-L6-v2' if self.model_loaded else 'fallback',
                    'quality_threshold': self.min_combined_confidence
                }],
                'extraction_method': 'rag_intelligent',
                'processing_timestamp': datetime.now().isoformat(),
                'model_info': {
                    'model_loaded': self.model_loaded,
                    'semantic_analysis': True,
                    'pattern_matching': True,
                    'quality_validation': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"RAG extraction failed for {filename}: {e}")
            return self._create_error_result(filename, str(e))

    def _smart_chunk_document(self, content: str, chunk_size: int) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks from document content"""
        
        if not content.strip():
            return []
        
        # Clean content first
        content = self._clean_content(content)
        
        chunks = []
        
        # Split by paragraphs first, then by sentences if too large
        paragraphs = re.split(r'\n\s*\n', content)
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
            
            if len(paragraph) <= chunk_size:
                # Paragraph fits in one chunk
                chunks.append({
                    'text': paragraph,
                    'chunk_id': len(chunks),
                    'paragraph_id': para_idx,
                    'chunk_type': 'paragraph',
                    'length': len(paragraph)
                })
            else:
                # Split large paragraph into sentence-based chunks
                sentences = re.split(r'[.!?]+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # Check if adding this sentence would exceed chunk size
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        # Save current chunk and start new one
                        if current_chunk.strip():
                            chunks.append({
                                'text': current_chunk.strip(),
                                'chunk_id': len(chunks),
                                'paragraph_id': para_idx,
                                'chunk_type': 'sentence_group',
                                'length': len(current_chunk)
                            })
                        current_chunk = sentence + ". "
                
                # Add final chunk if any content remains
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': len(chunks),
                        'paragraph_id': para_idx,
                        'chunk_type': 'sentence_group',
                        'length': len(current_chunk)
                    })
        
        return chunks

    def _find_relevant_chunks(self, chunks: List[Dict], keywords: List[str], 
                             content_type: str) -> List[Dict]:
        """Find chunks most likely to contain target content using semantic similarity"""
        
        if not chunks:
            return []
        
        relevant_chunks = []
        
        if self.model_loaded and self.model:
            # Use semantic similarity with sentence transformers
            try:
                # Create keyword query
                keyword_query = " ".join(keywords[:10])  # Use top 10 keywords
                
                # Get embeddings for query and chunks
                chunk_texts = [chunk['text'] for chunk in chunks]
                query_embedding = self.model.encode([keyword_query])
                chunk_embeddings = self.model.encode(chunk_texts)
                
                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                
                # Select chunks above threshold
                for i, similarity in enumerate(similarities):
                    if similarity >= self.min_semantic_score:
                        chunk = chunks[i].copy()
                        chunk['semantic_score'] = float(similarity)
                        chunk['content_type'] = content_type
                        relevant_chunks.append(chunk)
                
                # Sort by semantic score
                relevant_chunks.sort(key=lambda x: x['semantic_score'], reverse=True)
                
            except Exception as e:
                self.logger.warning(f"Semantic similarity failed, using keyword fallback: {e}")
                # Fall back to keyword matching
                relevant_chunks = self._keyword_based_chunk_selection(chunks, keywords, content_type)
        else:
            # Use keyword-based selection as fallback
            relevant_chunks = self._keyword_based_chunk_selection(chunks, keywords, content_type)
        
        return relevant_chunks[:20]  # Limit to top 20 chunks

    def _keyword_based_chunk_selection(self, chunks: List[Dict], keywords: List[str], 
                                     content_type: str) -> List[Dict]:
        """Fallback chunk selection using keyword matching"""
        
        relevant_chunks = []
        
        for chunk in chunks:
            text_lower = chunk['text'].lower()
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            
            if keyword_count > 0:
                # Calculate simple keyword-based score
                keyword_score = min(keyword_count / len(keywords), 1.0)
                
                if keyword_score >= 0.1:  # At least 10% keyword match
                    chunk_copy = chunk.copy()
                    chunk_copy['semantic_score'] = keyword_score
                    chunk_copy['content_type'] = content_type
                    relevant_chunks.append(chunk_copy)
        
        # Sort by keyword score
        relevant_chunks.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        return relevant_chunks

    def _extract_from_chunks(self, chunks: List[Dict], extraction_type: str, 
                           max_extractions: int) -> List[ExtractionResult]:
        """Extract content from relevant chunks using patterns and validation"""
        
        extractions = []
        patterns = (self.recommendation_patterns if extraction_type == 'recommendation' 
                   else self.response_patterns)
        
        for chunk in chunks:
            chunk_text = chunk['text']
            semantic_score = chunk.get('semantic_score', 0.0)
            
            # Apply patterns to this chunk
            for pattern_idx, pattern in enumerate(patterns):
                try:
                    matches = re.finditer(pattern, chunk_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    
                    for match in matches:
                        # Extract content from match
                        if len(match.groups()) >= 2:
                            # Pattern has capturing groups
                            if extraction_type == 'recommendation':
                                rec_id = match.group(1).strip()
                                content = match.group(2).strip()
                                full_content = f"{rec_id}: {content}" if rec_id else content
                            else:
                                response_type = match.group(1).strip()
                                content = match.group(2).strip()
                                full_content = f"{response_type} {content}".strip()
                        else:
                            # Use full match
                            full_content = match.group().strip()
                        
                        # Validate extraction
                        if self._validate_extraction(full_content, extraction_type):
                            # Calculate pattern score
                            pattern_score = self._calculate_pattern_score(
                                pattern_idx, len(patterns), full_content
                            )
                            
                            # Create extraction result
                            extraction = ExtractionResult(
                                content=full_content,
                                confidence=0.0,  # Will be calculated later
                                extraction_type=extraction_type,
                                source_chunk=chunk_text,
                                semantic_score=semantic_score,
                                pattern_score=pattern_score,
                                context=self._extract_context(chunk_text, match.start(), match.end()),
                                position=match.start(),
                                validation_passed=True
                            )
                            
                            extractions.append(extraction)
                            
                            if len(extractions) >= max_extractions:
                                return extractions
                                
                except re.error as e:
                    self.logger.warning(f"Regex error in pattern {pattern_idx}: {e}")
                    continue
        
        return extractions

    def _validate_extraction(self, content: str, extraction_type: str) -> bool:
        """Validate extracted content quality"""
        
        if not content or len(content.strip()) < 20:
            return False
        
        # Remove common false positives
        content_lower = content.lower().strip()
        
        # Skip page numbers, headers, footers
        if re.match(r'^(?:page|p\.)\s*\d+', content_lower):
            return False
        
        if re.match(r'^\d+\s*$', content_lower):
            return False
        
        # Skip copyright notices
        if 'crown copyright' in content_lower or 'ogl' in content_lower:
            return False
        
        # Skip very repetitive content
        words = content_lower.split()
        if len(set(words)) < len(words) * 0.5:  # Less than 50% unique words
            return False
        
        # Content-specific validation
        if extraction_type == 'recommendation':
            # Must have action words or requirement indicators
            action_indicators = ['should', 'must', 'recommend', 'suggest', 'establish', 'implement', 'develop']
            if not any(indicator in content_lower for indicator in action_indicators):
                return False
        
        elif extraction_type == 'response':
            # Must have response indicators
            response_indicators = ['accept', 'agree', 'implement', 'will', 'government', 'response']
            if not any(indicator in content_lower for indicator in response_indicators):
                return False
        
        return True

    def _calculate_pattern_score(self, pattern_idx: int, total_patterns: int, content: str) -> float:
        """Calculate pattern matching confidence score"""
        
        # Higher score for patterns that appear earlier (more specific)
        pattern_priority_score = (total_patterns - pattern_idx) / total_patterns
        
        # Content length bonus (longer matches often more complete)
        length_score = min(len(content) / 200, 1.0)  # Normalize to 200 chars
        
        # Keyword density score
        relevant_keywords = (self.recommendation_keywords + self.response_keywords)
        content_lower = content.lower()
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword in content_lower)
        keyword_score = min(keyword_matches / 10, 1.0)  # Normalize to 10 keywords
        
        # Combine scores
        pattern_score = (pattern_priority_score * 0.4 + 
                        length_score * 0.3 + 
                        keyword_score * 0.3)
        
        return min(pattern_score, 1.0)

    def _extract_context(self, chunk_text: str, start_pos: int, end_pos: int, 
                        context_window: int = 100) -> str:
        """Extract context around the matched content"""
        
        context_start = max(0, start_pos - context_window)
        context_end = min(len(chunk_text), end_pos + context_window)
        
        context = chunk_text[context_start:context_end].strip()
        
        # Clean up context
        context = re.sub(r'\s+', ' ', context)
        
        return context

    def _post_process_extractions(self, extractions: List[ExtractionResult], 
                                extraction_type: str) -> List[Dict[str, Any]]:
        """Post-process and convert extractions to final format"""
        
        processed = []
        seen_content = set()
        
        for extraction in extractions:
            # Calculate final confidence score
            final_confidence = (extraction.semantic_score * 0.4 + 
                              extraction.pattern_score * 0.6)
            
            # Skip low confidence extractions
            if final_confidence < self.min_combined_confidence:
                continue
            
            # Deduplicate based on content similarity
            content_normalized = re.sub(r'\s+', ' ', extraction.content.lower().strip())
            
            # Check for substantial duplicates
            is_duplicate = False
            for seen in seen_content:
                if self._calculate_text_similarity(content_normalized, seen) > 0.85:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            seen_content.add(content_normalized)
            
            # Create final extraction dict
            processed_extraction = {
                'content': extraction.content,
                'confidence': extraction.pattern_score,
                'semantic_score': extraction.semantic_score,
                'final_confidence': final_confidence,
                'extraction_method': 'rag_intelligent',
                'extraction_type': extraction_type,
                'context': extraction.context,
                'source_chunk_length': len(extraction.source_chunk),
                'position': extraction.position,
                'validation_passed': extraction.validation_passed
            }
            
            processed.append(processed_extraction)
        
        return processed

    def _rank_by_quality(self, extractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank extractions by overall quality score"""
        
        def quality_score(extraction):
            confidence = extraction.get('final_confidence', 0)
            length_bonus = min(len(extraction.get('content', '')) / 100, 0.2)
            return confidence + length_bonus
        
        extractions.sort(key=quality_score, reverse=True)
        return extractions

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        shorter = min(len(text1), len(text2))
        longer = max(len(text1), len(text2))
        
        if longer == 0:
            return 1.0
        
        # Calculate longest common subsequence ratio
        common_chars = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        return common_chars / longer

    def _clean_content(self, content: str) -> str:
        """Clean document content for better processing"""
        
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove page numbers and common PDF artifacts
        content = re.sub(r'\b(?:page|p\.)\s*\d+\b', '', content, flags=re.IGNORECASE)
        content = re.sub(r'©\s*Crown\s*copyright\s*\d{4}', '', content, flags=re.IGNORECASE)
        content = re.sub(r'E\d{8}\s+\d{2}/\d{2}', '', content)
        
        # Clean up multiple spaces and newlines
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        return content.strip()

    def _create_empty_result(self, filename: str, reason: str) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            'recommendations': [],
            'responses': [],
            'processing_results': [{
                'filename': filename,
                'error': reason,
                'extraction_method': 'rag_intelligent',
                'processing_timestamp': datetime.now().isoformat()
            }],
            'extraction_method': 'rag_intelligent',
            'model_info': {'model_loaded': self.model_loaded}
        }

    def _create_error_result(self, filename: str, error: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'recommendations': [],
            'responses': [],
            'processing_results': [{
                'filename': filename,
                'error': error,
                'extraction_method': 'rag_intelligent',
                'processing_timestamp': datetime.now().isoformat(),
                'status': 'error'
            }],
            'extraction_method': 'rag_intelligent',
            'model_info': {'model_loaded': self.model_loaded, 'error': error}
        }

    def test_extraction(self, sample_text: str = None) -> Dict[str, Any]:
        """Test RAG extraction on sample text"""
        
        if not sample_text:
            sample_text = """
            Recommendation 6a) iii) (accepted in principle by NI Executive)
            
            This recommendation is accepted in principle by the UK Government, 
            the Scottish Government, the Welsh Government, and the Northern Ireland Executive.
            
            Implementation will begin immediately through existing NHS structures.
            
            Recommendation 7b) Progress in implementation of the Transfusion 2024 recommendations be 
            reviewed, and next steps be determined and promulgated.
            
            This recommendation is accepted in full by the Scottish Government.
            
            The Government will establish a review process within 12 months.
            
            Accepting in principle
            Recommendation 4a) iv)
            Recommendation 4a) v)
            """
        
        return self.extract_with_rag(sample_text, "test_sample.txt", chunk_size=300, max_items=20)

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction performance statistics"""
        return {
            'model_loaded': self.model_loaded,
            'model_name': 'all-MiniLM-L6-v2' if self.model_loaded else 'fallback',
            'rag_available': is_rag_available(),
            'semantic_analysis_enabled': self.model_loaded,
            'pattern_count': {
                'recommendation_patterns': len(self.recommendation_patterns),
                'response_patterns': len(self.response_patterns)
            },
            'quality_thresholds': {
                'min_semantic_score': self.min_semantic_score,
                'min_pattern_score': self.min_pattern_score,
                'min_combined_confidence': self.min_combined_confidence
            },
            'keyword_sets': {
                'recommendation_keywords': len(self.recommendation_keywords),
                'response_keywords': len(self.response_keywords)
            }
        }

    def update_quality_thresholds(self, semantic_threshold: float = None, 
                                pattern_threshold: float = None, 
                                combined_threshold: float = None):
        """Update quality thresholds for extraction"""
        
        if semantic_threshold is not None:
            self.min_semantic_score = max(0.0, min(1.0, semantic_threshold))
        
        if pattern_threshold is not None:
            self.min_pattern_score = max(0.0, min(1.0, pattern_threshold))
        
        if combined_threshold is not None:
            self.min_combined_confidence = max(0.0, min(1.0, combined_threshold))
        
        self.logger.info(f"Updated thresholds - Semantic: {self.min_semantic_score}, "
                        f"Pattern: {self.min_pattern_score}, Combined: {self.min_combined_confidence}")

    def extract_with_custom_patterns(self, content: str, filename: str,
                                   custom_rec_patterns: List[str] = None,
                                   custom_resp_patterns: List[str] = None,
                                   **kwargs) -> Dict[str, Any]:
        """Extract using custom patterns in addition to built-in ones"""
        
        # Store original patterns
        original_rec_patterns = self.recommendation_patterns.copy()
        original_resp_patterns = self.response_patterns.copy()
        
        try:
            # Add custom patterns
            if custom_rec_patterns:
                self.recommendation_patterns.extend(custom_rec_patterns)
                self.logger.info(f"Added {len(custom_rec_patterns)} custom recommendation patterns")
            
            if custom_resp_patterns:
                self.response_patterns.extend(custom_resp_patterns)
                self.logger.info(f"Added {len(custom_resp_patterns)} custom response patterns")
            
            # Perform extraction
            result = self.extract_with_rag(content, filename, **kwargs)
            
            # Add custom pattern info to result
            result['model_info']['custom_patterns_used'] = {
                'recommendation_patterns': len(custom_rec_patterns) if custom_rec_patterns else 0,
                'response_patterns': len(custom_resp_patterns) if custom_resp_patterns else 0
            }
            
            return result
            
        finally:
            # Restore original patterns
            self.recommendation_patterns = original_rec_patterns
            self.response_patterns = original_resp_patterns

    def analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure for optimization insights"""
        
        if not content.strip():
            return {'error': 'No content provided'}
        
        # Basic structure analysis
        paragraphs = re.split(r'\n\s*\n', content)
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        # Pattern occurrence analysis
        rec_pattern_matches = []
        for i, pattern in enumerate(self.recommendation_patterns):
            try:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                if matches > 0:
                    rec_pattern_matches.append({'pattern_index': i, 'matches': matches})
            except re.error:
                continue
        
        resp_pattern_matches = []
        for i, pattern in enumerate(self.response_patterns):
            try:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                if matches > 0:
                    resp_pattern_matches.append({'pattern_index': i, 'matches': matches})
            except re.error:
                continue
        
        # Keyword density analysis
        content_lower = content.lower()
        rec_keyword_density = sum(1 for kw in self.recommendation_keywords if kw in content_lower)
        resp_keyword_density = sum(1 for kw in self.response_keywords if kw in content_lower)
        
        # Content quality indicators
        avg_paragraph_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / max(len([s for s in sentences if s.strip()]), 1)
        
        return {
            'document_stats': {
                'total_characters': len(content),
                'total_words': len(words),
                'total_paragraphs': len(paragraphs),
                'total_sentences': len([s for s in sentences if s.strip()]),
                'avg_paragraph_length': round(avg_paragraph_length, 2),
                'avg_sentence_length': round(avg_sentence_length, 2)
            },
            'pattern_analysis': {
                'recommendation_pattern_matches': rec_pattern_matches,
                'response_pattern_matches': resp_pattern_matches,
                'total_recommendation_matches': sum(p['matches'] for p in rec_pattern_matches),
                'total_response_matches': sum(p['matches'] for p in resp_pattern_matches)
            },
            'keyword_analysis': {
                'recommendation_keyword_density': rec_keyword_density,
                'response_keyword_density': resp_keyword_density,
                'recommendation_keyword_ratio': rec_keyword_density / len(self.recommendation_keywords),
                'response_keyword_ratio': resp_keyword_density / len(self.response_keywords)
            },
            'extraction_readiness': {
                'has_recommendations': len(rec_pattern_matches) > 0,
                'has_responses': len(resp_pattern_matches) > 0,
                'content_density': 'high' if len(words) > 1000 else 'medium' if len(words) > 300 else 'low',
                'structure_quality': 'good' if len(paragraphs) > 5 and avg_paragraph_length > 100 else 'fair'
            }
        }

    def optimize_for_document(self, content: str) -> Dict[str, Any]:
        """Analyze document and suggest optimization parameters"""
        
        analysis = self.analyze_document_structure(content)
        
        # Suggest optimal chunk size based on document structure
        avg_paragraph_length = analysis['document_stats']['avg_paragraph_length']
        
        if avg_paragraph_length > 800:
            suggested_chunk_size = 600
        elif avg_paragraph_length > 400:
            suggested_chunk_size = 500
        else:
            suggested_chunk_size = 400
        
        # Suggest thresholds based on keyword density
        rec_ratio = analysis['keyword_analysis']['recommendation_keyword_ratio']
        resp_ratio = analysis['keyword_analysis']['response_keyword_ratio']
        
        if rec_ratio > 0.3 or resp_ratio > 0.3:
            # High keyword density - can use higher thresholds
            suggested_semantic_threshold = 0.65
            suggested_combined_threshold = 0.7
        elif rec_ratio > 0.1 or resp_ratio > 0.1:
            # Medium keyword density - moderate thresholds
            suggested_semantic_threshold = 0.6
            suggested_combined_threshold = 0.65
        else:
            # Low keyword density - lower thresholds
            suggested_semantic_threshold = 0.55
            suggested_combined_threshold = 0.6
        
        recommendations = []
        
        # Content-specific recommendations
        if analysis['extraction_readiness']['content_density'] == 'low':
            recommendations.append("Document has low content density - consider combining with related documents")
        
        if not analysis['extraction_readiness']['has_recommendations']:
            recommendations.append("No clear recommendation patterns detected - may need custom patterns")
        
        if not analysis['extraction_readiness']['has_responses']:
            recommendations.append("No clear response patterns detected - may need custom patterns")
        
        if analysis['document_stats']['avg_paragraph_length'] < 50:
            recommendations.append("Very short paragraphs detected - may need different chunking strategy")
        
        return {
            'optimization_suggestions': {
                'chunk_size': suggested_chunk_size,
                'semantic_threshold': suggested_semantic_threshold,
                'combined_threshold': suggested_combined_threshold,
                'max_items': min(50, max(10, analysis['pattern_analysis']['total_recommendation_matches'] * 2))
            },
            'recommendations': recommendations,
            'document_analysis': analysis,
            'confidence_in_suggestions': 'high' if len(recommendations) <= 1 else 'medium' if len(recommendations) <= 3 else 'low'
        }

# Utility functions for external use
def create_rag_extractor(debug_mode: bool = False) -> IntelligentRAGExtractor:
    """Create and return a configured RAG extractor instance"""
    return IntelligentRAGExtractor(debug_mode=debug_mode)

def test_rag_system() -> Dict[str, Any]:
    """Test the RAG system with sample data"""
    
    extractor = IntelligentRAGExtractor(debug_mode=True)
    
    sample_text = """
    Recommendation 6a) iii) (accepted in principle by NI Executive)
    
    This recommendation is accepted in principle by the UK Government, 
    the Scottish Government, the Welsh Government, and the Northern Ireland Executive.
    
    Implementation will begin immediately through existing NHS structures.
    
    Recommendation 7b) Progress in implementation of the Transfusion 2024 recommendations be 
    reviewed, and next steps be determined and promulgated.
    
    This recommendation is accepted in full by the Scottish Government.
    
    The Government will establish a review process within 12 months.
    
    Accepting in principle
    Recommendation 4a) iv) Training should be provided to all staff members
    Recommendation 4a) v) A new framework must be established for quality assurance
    
    These recommendations are accepted and will be implemented through existing structures.
    """
    
    # Test extraction
    result = extractor.test_extraction(sample_text)
    
    # Test document analysis
    analysis = extractor.analyze_document_structure(sample_text)
    
    # Test optimization
    optimization = extractor.optimize_for_document(sample_text)
    
    return {
        'rag_status': get_rag_status(),
        'extraction_stats': extractor.get_extraction_stats(),
        'test_extraction': result,
        'document_analysis': analysis,
        'optimization_suggestions': optimization,
        'test_passed': len(result.get('recommendations', [])) > 0 and len(result.get('responses', [])) > 0
    }

# Export functions and classes
__all__ = [
    'IntelligentRAGExtractor',
    'is_rag_available',
    'get_rag_status',
    'ExtractionResult',
    'create_rag_extractor',
    'test_rag_system'
]
