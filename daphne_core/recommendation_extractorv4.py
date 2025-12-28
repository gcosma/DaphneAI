"""
Recommendation Extractor v4.0 - Semantic-First Architecture
Uses sentence transformers for intelligent extraction and deduplication.

v4.0 Changes:
- SEMANTIC deduplication using embeddings (catches duplicates with different footnotes)
- Intelligent recommendation text extraction (finds "That [Entity] should..." patterns)
- Semantic boundary detection (detects content-type shifts)
- Minimal regex (only for initial heading detection)
- Transformer-powered content analysis

Supported Formats:
- HSIB 2018 format: "Recommendation 2018/006:", "Recommendation 2018/007:"
- HSSIB 2023+ format: "Safety recommendation R/2023/220:"
- Standard government: "Recommendation 1", "Recommendation 12"
- Sentence-based fallback for unstructured documents

Usage:
    from recommendation_extractor import extract_recommendations
    
    recommendations = extract_recommendations(document_text, min_confidence=0.75)
    for rec in recommendations:
        print(f"{rec['rec_number']}: {rec['text'][:100]}...")
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, falling back to keyword-based methods")


class SemanticAnalyser:
    """
    Semantic analysis using sentence transformers.
    Provides embeddings, similarity, and content classification.
    """
    
    # Reference sentences for semantic comparison
    RECOMMENDATION_EXEMPLARS = [
        "That NHS England should implement new guidance.",
        "The Department should review current policies.",
        "Providers must ensure compliance with standards.",
        "We recommend establishing a new framework.",
        "The Trust should develop improved procedures.",
        "Clinical teams should adopt best practices.",
        "The Government should consider legislative changes.",
    ]
    
    NON_RECOMMENDATION_EXEMPLARS = [
        "This section describes the background.",
        "The following observations were noted.",
        "Local-level learning from this case.",
        "Background and context for the investigation.",
        "Safety observations from our review.",
        "Appendix containing supporting documents.",
        "References and endnotes section.",
        "Acknowledgements to contributors.",
    ]
    
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """Initialise with a sentence transformer model."""
        self.model = None
        self.rec_embeddings = None
        self.non_rec_embeddings = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = SentenceTransformer(model_name, device=device)
                # Pre-compute exemplar embeddings
                self.rec_embeddings = self.model.encode(
                    self.RECOMMENDATION_EXEMPLARS, 
                    convert_to_numpy=True
                )
                self.non_rec_embeddings = self.model.encode(
                    self.NON_RECOMMENDATION_EXEMPLARS,
                    convert_to_numpy=True
                )
                logger.info(f"SemanticAnalyser initialised with {model_name}")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}")
                self.model = None
    
    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode texts to embeddings."""
        if not self.model or not texts:
            return None
        try:
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.warning(f"Encoding failed: {e}")
            return None
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def is_recommendation_like(self, text: str) -> Tuple[bool, float]:
        """
        Determine if text is semantically similar to recommendation language.
        Returns (is_recommendation, confidence).
        """
        if not self.model or not text:
            return self._keyword_fallback_is_recommendation(text)
        
        try:
            text_embedding = self.model.encode([text], convert_to_numpy=True)[0]
            
            # Average similarity to recommendation exemplars
            rec_similarities = [
                self.cosine_similarity(text_embedding, emb) 
                for emb in self.rec_embeddings
            ]
            avg_rec_sim = np.mean(rec_similarities)
            max_rec_sim = np.max(rec_similarities)
            
            # Average similarity to non-recommendation exemplars
            non_rec_similarities = [
                self.cosine_similarity(text_embedding, emb)
                for emb in self.non_rec_embeddings
            ]
            avg_non_rec_sim = np.mean(non_rec_similarities)
            
            # Decision based on relative similarity
            is_rec = max_rec_sim > 0.5 and avg_rec_sim > avg_non_rec_sim
            confidence = max_rec_sim if is_rec else 1.0 - avg_non_rec_sim
            
            return is_rec, float(confidence)
            
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
            return self._keyword_fallback_is_recommendation(text)
    
    def _keyword_fallback_is_recommendation(self, text: str) -> Tuple[bool, float]:
        """Keyword-based fallback when transformers unavailable."""
        if not text:
            return False, 0.0
        
        text_lower = text.lower()
        
        # Strong recommendation indicators
        strong_patterns = [
            'should', 'must', 'shall', 'recommend', 'require', 
            'needs to', 'ought to', 'is required to'
        ]
        
        # Weak/context indicators
        weak_patterns = [
            'consider', 'review', 'ensure', 'develop', 'implement'
        ]
        
        strong_count = sum(1 for p in strong_patterns if p in text_lower)
        weak_count = sum(1 for p in weak_patterns if p in text_lower)
        
        if strong_count >= 1:
            return True, min(0.7 + (strong_count * 0.1), 0.95)
        elif weak_count >= 2:
            return True, 0.6
        
        return False, 0.3
    
    def are_semantically_similar(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """
        Check if two texts are semantically similar (for deduplication).
        Uses higher threshold (0.85) to catch duplicates with minor variations.
        """
        if not self.model:
            return self._keyword_fallback_similarity(text1, text2) > threshold
        
        try:
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
            similarity = self.cosine_similarity(embeddings[0], embeddings[1])
            return similarity >= threshold
        except Exception:
            return self._keyword_fallback_similarity(text1, text2) > threshold
    
    def _keyword_fallback_similarity(self, text1: str, text2: str) -> float:
        """Keyword-based similarity fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def find_recommendation_start(self, text: str) -> int:
        """
        Find where the actual recommendation content starts.
        Looks for "That [Entity]" pattern or first directive sentence.
        """
        if not text:
            return 0
        
        # Look for "That [Entity]" pattern which typically starts HSIB recommendations
        that_match = re.search(
            r'\bThat\s+(?:the\s+)?(?:NHS|CQC|Care|Department|Trust|Government|DHSC|NIHR)',
            text,
            re.IGNORECASE
        )
        if that_match:
            return that_match.start()
        
        # Look for entity + "should/must/requires" pattern
        entity_directive = re.search(
            r'(?:NHS\s+England|CQC|Department|Trust|Government|DHSC)\s+(?:should|must|requires?)',
            text,
            re.IGNORECASE
        )
        if entity_directive:
            return entity_directive.start()
        
        # If text starts with recommendation header, find end of header
        header_end = re.search(
            r'^(?:Safety\s+)?[Rr]ecommendation\s+(?:R/)?[\d/]+[:\s]+',
            text
        )
        if header_end:
            return header_end.end()
        
        return 0
    
    def find_recommendation_end(self, text: str, start_pos: int = 0) -> int:
        """
        Find where the recommendation content ends using semantic shift detection.
        """
        if not text or start_pos >= len(text):
            return len(text)
        
        working_text = text[start_pos:]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', working_text)
        
        if len(sentences) <= 1:
            return len(text)
        
        # Check each sentence for semantic shift away from recommendation language
        cumulative_length = start_pos
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                cumulative_length += 1
                continue
            
            # Check if this sentence looks like a section header or non-recommendation content
            if self._is_section_boundary(sentence):
                return cumulative_length
            
            # Use semantic analysis if available
            is_rec, confidence = self.is_recommendation_like(sentence)
            
            # If we're past the first sentence and hit non-recommendation content
            if i > 0 and not is_rec and confidence < 0.4:
                return cumulative_length
            
            cumulative_length += len(sentence) + 1  # +1 for space
        
        return len(text)
    
    def _is_section_boundary(self, text: str) -> bool:
        """Check if text represents a section boundary."""
        if not text:
            return False
        
        text_stripped = text.strip()
        
        # Numbered section headers (e.g., "6.3. Safety Observations")
        if re.match(r'^\d+\.[\d.]*\s+[A-Z]', text_stripped):
            return True
        
        # Title case headers on their own
        words = text_stripped.split()
        if len(words) <= 6:
            title_case_words = sum(1 for w in words if w and w[0].isupper())
            if title_case_words >= len(words) * 0.8:
                # Check for boundary keywords
                boundary_keywords = [
                    'observation', 'appendix', 'reference', 'endnote', 
                    'acknowledgement', 'glossary', 'background', 'context',
                    'methodology', 'finding', 'conclusion', 'summary'
                ]
                if any(kw in text_stripped.lower() for kw in boundary_keywords):
                    return True
        
        return False


class StrictRecommendationExtractor:
    """
    Extract recommendations using semantic analysis.
    Handles government-style numbered recommendations and general documents.
    """
    
    MAX_SENTENCE_LENGTH = 500
    MAX_RECOMMENDATION_LENGTH = 1500
    MIN_RECOMMENDATION_LENGTH = 50
    SEMANTIC_DEDUP_THRESHOLD = 0.85
    
    def __init__(self):
        """Initialise with semantic analyser and patterns."""
        self.analyser = SemanticAnalyser()
        
        self.action_verbs = {
            'establish', 'implement', 'develop', 'create', 'improve', 'enhance',
            'strengthen', 'expand', 'increase', 'reduce', 'review', 'assess',
            'evaluate', 'consider', 'adopt', 'introduce', 'maintain', 'monitor',
            'provide', 'support', 'enable', 'facilitate', 'promote', 'encourage',
            'prioritise', 'prioritize', 'address', 'tackle', 'resolve', 'prevent',
            'ensure', 'commission', 'consult', 'update', 'clarify', 'publish',
            'engage', 'deliver', 'conduct', 'undertake', 'initiate', 'collaborate',
            'coordinate', 'oversee', 'regulate', 'enforce', 'mandate', 'allocate',
            'fund', 'resource', 'train', 'educate', 'inform', 'report', 'audit',
            'inspect', 'investigate', 'examine', 'reform', 'revise', 'amend',
            'streamline', 'simplify', 'standardise', 'standardize', 'integrate',
            'consolidate', 'extend', 'limit', 'restrict', 'remove', 'set',
            'define', 'specify', 'determine', 'approve', 'authorise', 'authorize',
            'build', 'design', 'plan', 'prepare', 'bring', 'make', 'take', 'meet',
            'come', 'form', 'work', 'learn', 'drive', 'produce', 'require',
        }
    
    def fix_encoding(self, text: str) -> str:
        """Fix common PDF extraction encoding issues."""
        if not text:
            return ""
        
        replacements = {
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '—', 'â€"': '–',
            'Â ': ' ', '\u00a0': ' ', '�': '', '\ufffd': '',
        }
        
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean text of noise and artifacts."""
        if not text:
            return ""
        
        text = self.fix_encoding(text)
        
        # Remove URLs (keeping regex minimal and necessary)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove timestamps
        text = re.sub(
            r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?',
            '', text, flags=re.IGNORECASE
        )
        
        # Remove page numbers (short format only, preserving recommendation IDs)
        text = re.sub(r'\b(\d{1,2})/(\d{1,3})\b(?!\d)', '', text)
        
        # Remove footnote references like [2], [27]
        text = re.sub(r'\[\d+\]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_recommendation_content(self, raw_block: str, rec_id: str) -> str:
        """
        Extract the actual recommendation content from a raw block.
        Uses semantic analysis to find where the recommendation starts and ends.
        """
        if not raw_block:
            return ""
        
        cleaned = self.clean_text(raw_block)
        
        # Find where the actual recommendation starts (after preamble)
        start_pos = self.analyser.find_recommendation_start(cleaned)
        
        # Find where the recommendation ends (before next section)
        end_pos = self.analyser.find_recommendation_end(cleaned, start_pos)
        
        # Apply maximum length constraint
        if end_pos - start_pos > self.MAX_RECOMMENDATION_LENGTH:
            # Find a sensible sentence boundary
            content = cleaned[start_pos:start_pos + self.MAX_RECOMMENDATION_LENGTH]
            last_period = content.rfind('.')
            if last_period > self.MAX_RECOMMENDATION_LENGTH * 0.5:
                end_pos = start_pos + last_period + 1
            else:
                end_pos = start_pos + self.MAX_RECOMMENDATION_LENGTH
        
        extracted = cleaned[start_pos:end_pos].strip()
        
        # Reconstruct with recommendation header if we stripped it
        if not extracted.lower().startswith(('recommendation', 'safety recommendation', 'that ')):
            # Add back the recommendation ID as header
            if '/' in rec_id:  # HSIB format
                if rec_id.startswith('R/'):
                    extracted = f"Safety recommendation {rec_id}: {extracted}"
                else:
                    extracted = f"Recommendation {rec_id}: {extracted}"
        
        return extracted
    
    def is_garbage(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, str]:
        """First-pass filter: reject obvious garbage before analysis."""
        if not text:
            return True, "empty"
        
        cleaned = self.clean_text(text)
        
        if len(cleaned) < self.MIN_RECOMMENDATION_LENGTH:
            return True, "too_short"
        
        if not is_numbered_rec and len(cleaned) > self.MAX_SENTENCE_LENGTH:
            return True, "too_long"
        
        # Check for corrupted text (high special character ratio)
        special_chars = sum(1 for c in cleaned if not c.isalnum() and not c.isspace() and c not in '.,;:!?\'"-()[]')
        if len(cleaned) > 0 and special_chars / len(cleaned) > 0.12:
            return True, "corrupted"
        
        return False, ""
    
    def is_valid_recommendation(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, float, str, str]:
        """
        Determine if text is a valid recommendation using semantic analysis.
        Returns (is_valid, confidence, method, verb).
        """
        if not text:
            return False, 0.0, 'empty', 'unknown'
        
        cleaned = self.clean_text(text)
        if not cleaned:
            return False, 0.0, 'empty', 'unknown'
        
        # Length check for non-numbered
        if not is_numbered_rec and len(cleaned) > self.MAX_SENTENCE_LENGTH:
            return False, 0.0, 'too_long', 'none'
        
        # Use semantic analysis
        is_rec, semantic_confidence = self.analyser.is_recommendation_like(cleaned)
        
        # Extract verb
        verb = self._extract_verb(cleaned)
        
        # Determine method based on content patterns
        method = self._determine_method(cleaned, is_numbered_rec)
        
        # Adjust confidence based on method
        if 'hsib' in method or 'numbered' in method:
            confidence = max(semantic_confidence, 0.95)
        elif method in ('entity_should', 'we_recommend', 'that_entity'):
            confidence = max(semantic_confidence, 0.90)
        else:
            confidence = semantic_confidence
        
        return is_rec, confidence, method, verb
    
    def _determine_method(self, text: str, is_numbered_rec: bool) -> str:
        """Determine the extraction method based on text patterns."""
        text_lower = text.lower()
        
        # Check for HSIB patterns
        if re.match(r'^safety\s+recommendation\s+r/\d{4}/\d{3}', text_lower):
            match = re.search(r'r/\d{4}/\d{3}', text_lower)
            return f"hsib_recommendation_{match.group().upper()}" if match else "hsib_recommendation"
        
        if re.match(r'^recommendation\s+\d{4}/\d{3}', text_lower):
            match = re.search(r'\d{4}/\d{3}', text_lower)
            return f"hsib_recommendation_{match.group()}" if match else "hsib_recommendation"
        
        # Check for standard numbered recommendation
        if re.match(r'^recommendation\s+\d{1,2}\b', text_lower):
            match = re.search(r'\d{1,2}', text_lower)
            return f"numbered_recommendation_{match.group()}" if match else "numbered_recommendation"
        
        # Check for "That [Entity]" pattern
        if re.match(r'^that\s+(?:the\s+)?(?:nhs|cqc|care|department|trust|government)', text_lower):
            return "that_entity"
        
        # Check for entity + should
        if re.search(r'(?:nhs|cqc|trust|department|government)\s+(?:england\s+)?should', text_lower):
            return "entity_should"
        
        # Check for "we recommend"
        if 'we recommend' in text_lower:
            return "we_recommend"
        
        # Check for modal verbs
        if re.search(r'\b(should|must|shall)\s+\w+', text_lower):
            return "modal_verb"
        
        return "semantic_match"
    
    def _extract_verb(self, text: str) -> str:
        """Extract the main action verb from recommendation text."""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        # Look for "should [verb]" pattern
        should_match = re.search(r'\bshould\s+(?:also\s+)?(?:urgently\s+)?(?:be\s+)?(\w+)', text_lower)
        if should_match:
            verb = should_match.group(1)
            if verb in self.action_verbs:
                return verb
        
        # Look for any action verb
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}s?\b', text_lower):
                return verb
        
        return 'unknown'
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.75) -> List[Dict]:
        """Extract recommendations from text using semantic analysis."""
        recommendations = []
        
        if not text or not text.strip():
            logger.warning("Empty text passed to extract_recommendations")
            return []
        
        logger.info(f"Extracting recommendations from text of length {len(text)}")
        
        cleaned_full_text = self.clean_text(text)
        
        # =======================================================================
        # PHASE 1: Try HSSIB 2023+ format (Safety recommendation R/YYYY/NNN)
        # =======================================================================
        hsib_2023_pattern = re.compile(
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        hsib_2023_matches = list(hsib_2023_pattern.finditer(cleaned_full_text))
        
        if hsib_2023_matches:
            logger.info(f"Found {len(hsib_2023_matches)} HSSIB 2023+ recommendations")
            recommendations = self._extract_from_matches(
                cleaned_full_text, hsib_2023_matches, 'hsib_2023', min_confidence
            )
            if recommendations:
                return self._semantic_deduplicate(recommendations)
        
        # =======================================================================
        # PHASE 2: Try HSIB 2018 format (Recommendation YYYY/NNN)
        # =======================================================================
        hsib_2018_pattern = re.compile(
            r'Recommendation\s+(\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        hsib_2018_matches = list(hsib_2018_pattern.finditer(cleaned_full_text))
        
        if hsib_2018_matches:
            logger.info(f"Found {len(hsib_2018_matches)} HSIB 2018 recommendations")
            recommendations = self._extract_from_matches(
                cleaned_full_text, hsib_2018_matches, 'hsib_2018', min_confidence
            )
            if recommendations:
                return self._semantic_deduplicate(recommendations)
        
        # =======================================================================
        # PHASE 3: Standard "Recommendation N" format
        # =======================================================================
        standard_pattern = re.compile(
            r'(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b',
            re.IGNORECASE,
        )
        standard_matches = list(standard_pattern.finditer(cleaned_full_text))
        
        if standard_matches:
            logger.info(f"Found {len(standard_matches)} standard recommendation headings")
            recommendations = self._extract_from_matches(
                cleaned_full_text, standard_matches, 'standard', min_confidence
            )
            if recommendations:
                return self._semantic_deduplicate(recommendations)
        
        # =======================================================================
        # PHASE 4: Fallback to sentence extraction with semantic filtering
        # =======================================================================
        logger.info("No numbered recommendations found, falling back to sentence extraction")
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        for idx, sentence in enumerate(sentences):
            cleaned = self.clean_text(sentence)
            
            is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=False)
            if is_garbage:
                continue
            
            is_rec, confidence, method, verb = self.is_valid_recommendation(cleaned, is_numbered_rec=False)
            
            if is_rec and confidence >= min_confidence:
                recommendations.append({
                    'text': cleaned,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': idx,
                    'in_section': False,
                })
        
        return self._semantic_deduplicate(recommendations)
    
    def _extract_from_matches(
        self,
        text: str,
        matches: List,
        pattern_type: str,
        min_confidence: float
    ) -> List[Dict]:
        """Extract recommendations from regex matches with semantic content extraction."""
        recommendations = []
        
        for idx, match in enumerate(matches):
            start = match.start()
            next_rec_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            
            # Extract raw block
            raw_block = text[start:next_rec_pos]
            
            # Get recommendation ID
            if pattern_type == 'hsib_2023':
                rec_id = match.group(1)  # e.g., "R/2023/220"
            elif pattern_type == 'hsib_2018':
                rec_id = match.group(1)  # e.g., "2018/006"
            elif pattern_type == 'standard':
                heading_num = match.group(1)
                extra_digit = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                rec_id = f"{heading_num}{extra_digit}" if extra_digit and len(heading_num) == 1 else heading_num
            else:
                rec_id = f"rec_{idx + 1}"
            
            # Use semantic extraction to get clean recommendation content
            cleaned = self.extract_recommendation_content(raw_block, rec_id)
            
            if not cleaned:
                continue
            
            is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=True)
            if is_garbage:
                logger.debug(f"Skipping {pattern_type} rec {idx}: {reason}")
                continue
            
            is_rec, confidence, method, verb = self.is_valid_recommendation(cleaned, is_numbered_rec=True)
            
            if is_rec and confidence >= min_confidence:
                recommendations.append({
                    'text': cleaned,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': idx,
                    'in_section': True,
                    'rec_number': rec_id,
                })
        
        return recommendations
    
    def _semantic_deduplicate(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Remove duplicate recommendations using semantic similarity.
        This catches duplicates even when text differs slightly (footnotes, formatting).
        """
        if not recommendations:
            return []
        
        unique = []
        
        for rec in recommendations:
            is_duplicate = False
            
            for existing in unique:
                # Check semantic similarity
                if self.analyser.are_semantically_similar(
                    rec['text'], 
                    existing['text'],
                    threshold=self.SEMANTIC_DEDUP_THRESHOLD
                ):
                    is_duplicate = True
                    logger.debug(f"Duplicate detected: {rec.get('rec_number', 'N/A')} matches {existing.get('rec_number', 'N/A')}")
                    break
            
            if not is_duplicate:
                unique.append(rec)
        
        logger.info(f"Deduplicated: {len(recommendations)} → {len(unique)} recommendations")
        return unique
    
    def get_statistics(self, recommendations: List[Dict]) -> Dict:
        """Get statistics about extracted recommendations."""
        if not recommendations:
            return {
                'total': 0,
                'unique_verbs': 0,
                'avg_confidence': 0,
                'in_section_count': 0,
                'methods': {},
                'top_verbs': {},
                'high_confidence': 0,
                'medium_confidence': 0,
            }
        
        verb_counts = Counter(r['verb'] for r in recommendations)
        method_counts = Counter(r['method'] for r in recommendations)
        in_section = sum(1 for r in recommendations if r.get('in_section', False))
        
        return {
            'total': len(recommendations),
            'unique_verbs': len(verb_counts),
            'methods': dict(method_counts),
            'top_verbs': dict(verb_counts.most_common(10)),
            'avg_confidence': round(sum(r['confidence'] for r in recommendations) / len(recommendations), 3),
            'in_section_count': in_section,
            'high_confidence': sum(1 for r in recommendations if r['confidence'] >= 0.9),
            'medium_confidence': sum(1 for r in recommendations if 0.75 <= r['confidence'] < 0.9),
        }


# Backward compatibility alias
AdvancedRecommendationExtractor = StrictRecommendationExtractor


def extract_recommendations(text: str, min_confidence: float = 0.75) -> List[Dict]:
    """Main function to extract recommendations."""
    extractor = StrictRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)
