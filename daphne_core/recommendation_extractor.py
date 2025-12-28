"""
Recommendation Extractor v5.0 - Semantic-First with First-Occurrence Architecture
Combines semantic intelligence with efficient duplicate prevention.

v5.0 Features:
- SEMANTIC analysis using sentence transformers (from v4.0)
- FIRST-OCCURRENCE extraction - only processes first instance of each rec ID (from v4.2)
- Semantic deduplication as backup (catches near-duplicates)
- Intelligent content boundary detection
- Minimal regex (only for initial heading detection)

Supported Formats:
- HSSIB 2023+ format: "Safety recommendation R/2023/220:"
- HSIB 2018 format: "Recommendation 2018/006:"
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
from typing import List, Dict, Tuple, Optional, Set
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
        "HSIB recommends that the Care Quality Commission evaluates the process.",
        "The Royal College should form a working group.",
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
        "The care of the patient demonstrates the potential benefit.",
        "Another expert told the investigation that.",
    ]
    
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """Initialise with a sentence transformer model."""
        self.model = None
        self.rec_embeddings = None
        self.non_rec_embeddings = None
        self._available = False
        
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
                self._available = True
                logger.info(f"SemanticAnalyser initialised with {model_name}")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}")
                self.model = None
    
    @property
    def available(self) -> bool:
        return self._available and self.model is not None
    
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
        if not self.available or not text:
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
            max_non_rec_sim = np.max(non_rec_similarities)
            
            # Decision based on relative similarity
            # Text is a recommendation if it's more similar to rec exemplars
            is_rec = max_rec_sim > 0.5 and avg_rec_sim > avg_non_rec_sim
            
            # Also reject if very similar to non-recommendation content
            if max_non_rec_sim > 0.7 and max_non_rec_sim > max_rec_sim:
                is_rec = False
            
            confidence = max_rec_sim if is_rec else 1.0 - max_non_rec_sim
            
            return is_rec, float(confidence)
            
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
            return self._keyword_fallback_is_recommendation(text)
    
    def _keyword_fallback_is_recommendation(self, text: str) -> Tuple[bool, float]:
        """Keyword-based fallback when transformers unavailable."""
        if not text:
            return False, 0.0
        
        text_lower = text.lower()
        
        # "That [Entity]" pattern is a very strong indicator (HSIB style)
        if re.match(r'^that\s+(?:the\s+)?(?:nhs|cqc|care|department|trust|government|national|royal)', text_lower):
            return True, 0.95
        
        # "HSIB recommends" pattern
        if 'hsib recommends' in text_lower:
            return True, 0.95
        
        # Strong recommendation indicators
        strong_patterns = [
            'should', 'must', 'shall', 'recommend', 'require', 
            'needs to', 'ought to', 'is required to'
        ]
        
        # Weak/context indicators
        weak_patterns = [
            'consider', 'review', 'ensure', 'develop', 'implement',
            'evaluate', 'assess', 'establish', 'form', 'create'
        ]
        
        # Negative indicators (non-recommendation content)
        negative_patterns = [
            'demonstrates the potential', 'told the investigation',
            'another expert', 'background', 'observation',
            'appendix', 'reference', 'endnote', 'acknowledgement'
        ]
        
        # Check for negative patterns first
        for pattern in negative_patterns:
            if pattern in text_lower:
                return False, 0.3
        
        strong_count = sum(1 for p in strong_patterns if p in text_lower)
        weak_count = sum(1 for p in weak_patterns if p in text_lower)
        
        if strong_count >= 1:
            return True, min(0.7 + (strong_count * 0.1), 0.95)
        elif weak_count >= 2:
            return True, 0.7
        elif weak_count >= 1:
            # Single weak pattern - still likely a recommendation
            return True, 0.6
        
        return False, 0.3
    
    def are_semantically_similar(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """
        Check if two texts are semantically similar (for deduplication).
        Uses higher threshold (0.85) to catch duplicates with minor variations.
        """
        if not self.available:
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
    
    def find_content_boundary(self, text: str, start_pos: int = 0) -> int:
        """
        Find where the recommendation content ends using semantic shift detection.
        Returns position where non-recommendation content begins.
        """
        if not text or start_pos >= len(text):
            return len(text)
        
        working_text = text[start_pos:]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', working_text)
        
        if len(sentences) <= 1:
            return len(text)
        
        cumulative_length = start_pos
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                cumulative_length += 1
                continue
            
            # Check if this sentence looks like a section header
            if self._is_section_boundary(sentence):
                return cumulative_length
            
            # Use semantic analysis to detect content shift
            is_rec, confidence = self.is_recommendation_like(sentence)
            
            # If we're past the first 2 sentences and hit clear non-recommendation content
            if i > 1 and not is_rec and confidence < 0.4:
                return cumulative_length
            
            cumulative_length += len(sentence) + 1
        
        return len(text)
    
    def _is_section_boundary(self, text: str) -> bool:
        """Check if text represents a section boundary."""
        if not text:
            return False
        
        text_stripped = text.strip()
        
        # Numbered section headers (e.g., "6.3 Safety Observations")
        if re.match(r'^\d+\.[\d.]*\s+[A-Z]', text_stripped):
            return True
        
        # Known boundary markers
        boundary_patterns = [
            r'^Safety\s+observations?\b',
            r'^Safety\s+actions?\b',
            r'^Local.level\s+learning\b',
            r'^Background\s+and\s+context\b',
            r'^Appendix',
            r'^References?\b',
            r'^Endnotes?\b',
            r'^Acknowledgements?\b',
            r'^Glossary\b',
            r'^Methodology\b',
        ]
        
        for pattern in boundary_patterns:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return True
        
        return False


class StrictRecommendationExtractor:
    """
    Extract recommendations using semantic analysis and first-occurrence architecture.
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
        
        # Section boundary markers for content extraction
        self.section_markers = [
            r'\bSafety\s+observations?\b',
            r'\bSafety\s+actions?\b',
            r'\bLocal.level\s+learning\b',
            r'\bBackground\s+and\s+context\b',
            r'\bAppendix(?:es)?(?:\s+\d+)?\b',
            r'\bGlossary\b',
            r'\bReferences\b',
            r'\bEndnotes\b',
            r'\bAcknowledgements?\b',
            r'\bMethodology\b',
            r'\bConclusion(?:s)?\b',
            r'\bOur vision for\b',
            r'\bHSIB\s+makes\s+the\s+following\b',  # End of recommendations section
        ]
    
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
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove timestamps
        text = re.sub(
            r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?',
            '', text, flags=re.IGNORECASE
        )
        
        # Remove page numbers (short format only)
        text = re.sub(r'\b(\d{1,2})/(\d{1,3})\b(?!\d)', '', text)
        
        # Remove footnote references
        text = re.sub(r'\[\d+\]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _find_recommendation_end(self, text: str, start_pos: int, next_rec_pos: int) -> int:
        """
        Find the proper end position for a recommendation.
        Uses both structural markers and semantic analysis.
        """
        # If there's a next recommendation, that's our hard boundary
        if next_rec_pos < len(text):
            search_end = next_rec_pos
        else:
            search_end = len(text)
        
        # Look for section markers within the block
        search_text = text[start_pos:search_end]
        earliest_boundary = search_end
        
        for pattern in self.section_markers:
            match = re.search(pattern, search_text[50:], re.IGNORECASE)  # Skip first 50 chars
            if match:
                boundary = start_pos + 50 + match.start()
                if boundary < earliest_boundary:
                    earliest_boundary = boundary
        
        # Also use semantic boundary detection if available
        if self.analyser.available:
            semantic_boundary = self.analyser.find_content_boundary(text, start_pos)
            if semantic_boundary < earliest_boundary:
                earliest_boundary = semantic_boundary
        
        return earliest_boundary
    
    def _extract_recommendation_content(self, raw_block: str, rec_id: str) -> str:
        """
        Extract the actual recommendation content from a raw block.
        Finds "That [Entity]" pattern or directive content.
        """
        if not raw_block:
            return ""
        
        cleaned = self.clean_text(raw_block)
        
        # Look for "That [Entity]" pattern (common in HSIB)
        that_match = re.search(
            r'\bThat\s+(?:the\s+)?(?:NHS|CQC|Care|Department|Trust|Government|DHSC|NIHR|Royal|National)',
            cleaned,
            re.IGNORECASE
        )
        if that_match:
            start_pos = that_match.start()
            return cleaned[start_pos:].strip()
        
        # Look for entity + "should/must" pattern
        entity_match = re.search(
            r'(?:NHS\s+England|CQC|Care\s+Quality|Department|Trust|Government|DHSC)\s+(?:should|must|requires?)',
            cleaned,
            re.IGNORECASE
        )
        if entity_match:
            start_pos = entity_match.start()
            return cleaned[start_pos:].strip()
        
        # Look for "HSIB recommends" pattern
        hsib_match = re.search(r'HSIB\s+recommends\s+that', cleaned, re.IGNORECASE)
        if hsib_match:
            start_pos = hsib_match.start()
            return cleaned[start_pos:].strip()
        
        # If text starts with recommendation header, skip the header
        header_match = re.match(
            r'^(?:Safety\s+)?[Rr]ecommendation\s+(?:R/)?[\d/]+[:\s]+',
            cleaned
        )
        if header_match:
            return cleaned[header_match.end():].strip()
        
        return cleaned
    
    def is_garbage(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, str]:
        """First-pass filter: reject obvious garbage."""
        if not text:
            return True, "empty"
        
        cleaned = self.clean_text(text)
        
        if len(cleaned) < self.MIN_RECOMMENDATION_LENGTH:
            return True, "too_short"
        
        if not is_numbered_rec and len(cleaned) > self.MAX_SENTENCE_LENGTH:
            return True, "too_long"
        
        # Check for corrupted text
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
        
        # Use semantic analysis
        is_rec, semantic_confidence = self.analyser.is_recommendation_like(cleaned)
        
        # If semantic analysis says it's NOT a recommendation, trust it
        if not is_rec and semantic_confidence < 0.4:
            return False, semantic_confidence, 'rejected_semantic', 'none'
        
        # Extract verb
        verb = self._extract_verb(cleaned)
        
        # Determine method
        method = self._determine_method(cleaned)
        
        # Adjust confidence based on method
        if 'hsib' in method or 'numbered' in method:
            confidence = max(semantic_confidence, 0.95)
        elif method in ('entity_should', 'we_recommend', 'that_entity'):
            confidence = max(semantic_confidence, 0.90)
        else:
            confidence = semantic_confidence
        
        return True, confidence, method, verb
    
    def _determine_method(self, text: str) -> str:
        """Determine the extraction method based on text patterns."""
        text_lower = text.lower()
        
        if re.match(r'^safety\s+recommendation\s+r/\d{4}/\d{3}', text_lower):
            match = re.search(r'r/\d{4}/\d{3}', text_lower)
            return f"hsib_2023_{match.group().upper()}" if match else "hsib_2023"
        
        if re.match(r'^recommendation\s+\d{4}/\d{3}', text_lower):
            match = re.search(r'\d{4}/\d{3}', text_lower)
            return f"hsib_2018_{match.group()}" if match else "hsib_2018"
        
        if re.match(r'^recommendation\s+\d{1,2}\b', text_lower):
            match = re.search(r'\d{1,2}', text_lower)
            return f"numbered_recommendation_{match.group()}" if match else "numbered"
        
        if re.match(r'^that\s+(?:the\s+)?(?:nhs|cqc|care|department|trust|government)', text_lower):
            return "that_entity"
        
        if re.search(r'(?:nhs|cqc|trust|department|government)\s+(?:england\s+)?should', text_lower):
            return "entity_should"
        
        if 'we recommend' in text_lower or 'hsib recommends' in text_lower:
            return "we_recommend"
        
        if re.search(r'\b(should|must|shall)\s+\w+', text_lower):
            return "modal_verb"
        
        return "semantic_match"
    
    def _extract_verb(self, text: str) -> str:
        """Extract the main action verb from recommendation text."""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        should_match = re.search(r'\bshould\s+(?:also\s+)?(?:urgently\s+)?(?:be\s+)?(\w+)', text_lower)
        if should_match:
            verb = should_match.group(1)
            if verb in self.action_verbs:
                return verb
        
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}s?\b', text_lower):
                return verb
        
        return 'unknown'
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.75) -> List[Dict]:
        """
        Extract recommendations using semantic analysis and first-occurrence architecture.
        """
        recommendations = []
        
        if not text or not text.strip():
            logger.warning("Empty text passed to extract_recommendations")
            return []
        
        logger.info(f"Extracting recommendations from text of length {len(text)}")
        
        cleaned_full_text = self.clean_text(text)
        
        # =======================================================================
        # PHASE 1: HSSIB 2023+ format (Safety recommendation R/YYYY/NNN)
        # =======================================================================
        hsib_2023_pattern = re.compile(
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        hsib_2023_matches = list(hsib_2023_pattern.finditer(cleaned_full_text))
        
        if hsib_2023_matches:
            logger.info(f"Found {len(hsib_2023_matches)} HSSIB 2023+ recommendation headers")
            recommendations = self._extract_first_occurrences(
                cleaned_full_text, hsib_2023_matches, 'hsib_2023', min_confidence
            )
            if recommendations:
                return self._semantic_deduplicate(recommendations)
        
        # =======================================================================
        # PHASE 2: HSIB 2018 format (Recommendation YYYY/NNN)
        # =======================================================================
        hsib_2018_pattern = re.compile(
            r'Recommendation\s+(\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        hsib_2018_matches = list(hsib_2018_pattern.finditer(cleaned_full_text))
        
        if hsib_2018_matches:
            logger.info(f"Found {len(hsib_2018_matches)} HSIB 2018 recommendation headers")
            recommendations = self._extract_first_occurrences(
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
            logger.info(f"Found {len(standard_matches)} standard recommendation headers")
            recommendations = self._extract_first_occurrences(
                cleaned_full_text, standard_matches, 'standard', min_confidence
            )
            if recommendations:
                return self._semantic_deduplicate(recommendations)
        
        # =======================================================================
        # PHASE 4: Sentence-based fallback with semantic filtering
        # =======================================================================
        logger.info("No numbered recommendations found, using sentence extraction")
        
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
    
    def _extract_first_occurrences(
        self,
        text: str,
        matches: List,
        pattern_type: str,
        min_confidence: float
    ) -> List[Dict]:
        """
        Extract recommendations from FIRST occurrence of each unique ID only.
        This prevents duplicate extraction when recommendations appear multiple times.
        """
        recommendations = []
        seen_ids: Set[str] = set()
        
        for idx, match in enumerate(matches):
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
            
            # FIRST-OCCURRENCE CHECK: Skip if we've already seen this ID
            if rec_id in seen_ids:
                logger.debug(f"Skipping duplicate occurrence of {rec_id}")
                continue
            
            seen_ids.add(rec_id)
            
            # Find boundaries
            start = match.start()
            next_rec_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            end = self._find_recommendation_end(text, start, next_rec_pos)
            
            # Extract raw block
            raw_block = text[start:end]
            
            # Get clean recommendation content
            content = self._extract_recommendation_content(raw_block, rec_id)
            
            if not content:
                continue
            
            is_garbage, reason = self.is_garbage(content, is_numbered_rec=True)
            if is_garbage:
                logger.debug(f"Skipping {rec_id}: {reason}")
                continue
            
            is_rec, confidence, method, verb = self.is_valid_recommendation(content, is_numbered_rec=True)
            
            if is_rec and confidence >= min_confidence:
                # Reconstruct with header if needed
                if not content.lower().startswith(('recommendation', 'safety recommendation', 'that ')):
                    if pattern_type == 'hsib_2023':
                        content = f"Safety recommendation {rec_id}: {content}"
                    else:
                        content = f"Recommendation {rec_id}: {content}"
                
                recommendations.append({
                    'text': content,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': len(recommendations),
                    'in_section': True,
                    'rec_number': rec_id,
                })
        
        return recommendations
    
    def _semantic_deduplicate(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Remove duplicate recommendations using semantic similarity.
        Catches near-duplicates that first-occurrence might miss.
        """
        if not recommendations:
            return []
        
        unique = []
        
        for rec in recommendations:
            is_duplicate = False
            
            for existing in unique:
                if self.analyser.are_semantically_similar(
                    rec['text'], 
                    existing['text'],
                    threshold=self.SEMANTIC_DEDUP_THRESHOLD
                ):
                    is_duplicate = True
                    logger.debug(f"Semantic duplicate: {rec.get('rec_number', 'N/A')}")
                    break
            
            if not is_duplicate:
                unique.append(rec)
        
        if len(recommendations) != len(unique):
            logger.info(f"Semantic dedup: {len(recommendations)} → {len(unique)}")
        
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
                'semantic_available': self.analyser.available,
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
            'semantic_available': self.analyser.available,
        }


# Backward compatibility
AdvancedRecommendationExtractor = StrictRecommendationExtractor


def extract_recommendations(text: str, min_confidence: float = 0.75) -> List[Dict]:
    """Main function to extract recommendations."""
    extractor = StrictRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)
