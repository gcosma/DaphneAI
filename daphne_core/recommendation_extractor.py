"""
Recommendation Extractor v4.1 - Structure-First Architecture
Detects document structure and extracts accordingly.
Uses semantic analysis ONLY for deduplication and unstructured fallback.

v4.1 Changes:
- Structure-first: detect format, extract between headers
- Semantic deduplication (catches duplicates with different footnotes)
- Semantic analysis only for unstructured documents
- Clean, simple extraction for structured documents

Supported Formats:
- HSIB 2018: "Recommendation 2018/006:"
- HSSIB 2023+: "Safety recommendation R/2023/220:"
- Standard government: "Recommendation 1", "Recommendation 12"
- Unstructured: sentence-based with semantic filtering

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


class SemanticDeduplicator:
    """
    Handles semantic similarity for deduplication only.
    Lightweight - doesn't do content classification.
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """Initialise with a sentence transformer model."""
        self.model = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = SentenceTransformer(model_name, device=device)
                logger.info(f"SemanticDeduplicator initialised with {model_name}")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}")
                self.model = None
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def are_similar(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """
        Check if two texts are semantically similar.
        Used for deduplication - catches duplicates with minor variations.
        """
        if not text1 or not text2:
            return False
        
        if not self.model:
            return self._keyword_similarity(text1, text2) > threshold
        
        try:
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
            similarity = self.cosine_similarity(embeddings[0], embeddings[1])
            return similarity >= threshold
        except Exception:
            return self._keyword_similarity(text1, text2) > threshold
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Keyword-based similarity fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def is_recommendation_like(self, text: str) -> Tuple[bool, float]:
        """
        For unstructured fallback: check if text looks like a recommendation.
        """
        if not text:
            return False, 0.0
        
        text_lower = text.lower()
        
        # Strong indicators
        if any(p in text_lower for p in ['should', 'must', 'shall', 'recommend', 'require']):
            return True, 0.85
        
        # Weaker indicators
        if any(p in text_lower for p in ['consider', 'review', 'ensure', 'develop']):
            return True, 0.65
        
        return False, 0.3


class StrictRecommendationExtractor:
    """
    Structure-first recommendation extractor.
    Detects document format, extracts between headers, deduplicates semantically.
    """
    
    MAX_SENTENCE_LENGTH = 500
    MAX_RECOMMENDATION_LENGTH = 2000
    MIN_RECOMMENDATION_LENGTH = 50
    SEMANTIC_DEDUP_THRESHOLD = 0.80  # Lowered from 0.85 to catch near-duplicates
    
    def __init__(self):
        """Initialise with semantic deduplicator."""
        self.deduplicator = SemanticDeduplicator()
        
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
            'come', 'form', 'work', 'learn', 'drive', 'produce', 'require', 'extend',
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text of noise and artifacts."""
        if not text:
            return ""
        
        # Fix encoding
        replacements = {
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '—',
            'Â ': ' ', '\u00a0': ' ', '�': '', '\ufffd': '',
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove timestamps
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?', '', text, flags=re.IGNORECASE)
        
        # Remove page numbers (short format, not recommendation IDs)
        text = re.sub(r'\b(\d{1,2})/(\d{1,3})\b(?!\d)', '', text)
        
        # Remove footnote references [2], [27]
        text = re.sub(r'\[\d+\]', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def detect_structure(self, text: str) -> Tuple[str, List]:
        """
        Detect document structure and return (format_type, header_matches).
        Checks formats in order of specificity.
        """
        # HSSIB 2023+: "Safety recommendation R/2023/220:"
        hsib_2023 = list(re.finditer(
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]',
            text, re.IGNORECASE
        ))
        if hsib_2023:
            return 'hsib_2023', hsib_2023
        
        # HSIB 2018: "Recommendation 2018/006:"
        hsib_2018 = list(re.finditer(
            r'Recommendation\s+(\d{4}/\d{3})[:\s]',
            text, re.IGNORECASE
        ))
        if hsib_2018:
            return 'hsib_2018', hsib_2018
        
        # Standard: "Recommendation 1", "Recommendation 12"
        standard = list(re.finditer(
            r'(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b',
            text, re.IGNORECASE
        ))
        if standard:
            return 'standard', standard
        
        return 'unstructured', []
    
    def extract_between_headers(self, text: str, matches: List, format_type: str) -> List[Dict]:
        """
        Extract recommendation text between headers.
        Simple and direct - no semantic analysis needed for structured docs.
        """
        recommendations = []
        
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            
            # Get the raw block
            raw_block = text[start:end]
            
            # Get recommendation ID
            if format_type == 'hsib_2023':
                rec_id = match.group(1)  # R/2023/220
            elif format_type == 'hsib_2018':
                rec_id = match.group(1)  # 2018/006
            else:  # standard
                num = match.group(1)
                extra = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                rec_id = f"{num}{extra}" if extra and len(num) == 1 else num
            
            # Clean and trim the block
            cleaned = self._clean_recommendation_block(raw_block, rec_id, format_type)
            
            if cleaned and len(cleaned) >= self.MIN_RECOMMENDATION_LENGTH:
                verb = self._extract_verb(cleaned)
                recommendations.append({
                    'text': cleaned,
                    'verb': verb,
                    'method': f'{format_type}_{rec_id}',
                    'confidence': 0.98,  # High confidence for structured extraction
                    'position': idx,
                    'in_section': True,
                    'rec_number': rec_id,
                })
        
        return recommendations
    
    def _clean_recommendation_block(self, raw_block: str, rec_id: str, format_type: str) -> str:
        """
        Clean a recommendation block extracted between headers.
        Removes trailing non-recommendation content.
        """
        cleaned = self.clean_text(raw_block)
        
        # Find where recommendation content actually starts
        # For HSIB: look for "That [Entity]" pattern
        that_match = re.search(
            r'\bThat\s+(?:the\s+)?(?:NHS|CQC|Care|Department|Trust|Government|DHSC|NIHR)',
            cleaned, re.IGNORECASE
        )
        
        if that_match:
            # Content starts at "That..."
            content_start = that_match.start()
        else:
            # Content starts after the header
            header_match = re.match(
                r'^(?:Safety\s+)?[Rr]ecommendation\s+(?:R/)?[\d/]+[:\s]*',
                cleaned
            )
            content_start = header_match.end() if header_match else 0
        
        # Find where recommendation content ends
        content = cleaned[content_start:]
        
        # Boundary patterns - content that indicates end of THIS recommendation
        # These patterns work on the content AFTER the "That..." start
        boundary_patterns = [
            r'\.\s+The\s+investigation\s+makes',  # "The investigation makes two Safety Observations"
            r'\.\s+HSIB\s+has\s+directed',  # "HSIB has directed safety recommendations"
            r'\.\s+These\s+organisations\s+are\s+expected',  # "These organisations are expected to respond"
            r'\.\s*\d+\.[\d.]*\s+[A-Z]',  # "6.3 Safety Observations" (numbered section)
            r'\.\s+Safety\s+[Oo]bservation',  # "Safety Observation" section
            r'\.\s+Local-level\s+learning',  # HSIB local learning section
            r'\.\s+Background\s+and\s+context',
            r'\n\s*Appendix',
            r'\n\s*References\s*$',
            r'\n\s*Endnotes',
        ]
        
        end_pos = len(content)
        for pattern in boundary_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Include the period, exclude the rest
                cut_point = match.start() + 1  # +1 to include the period
                if cut_point < end_pos:
                    end_pos = cut_point
        
        # Trim content
        content = content[:end_pos].strip()
        
        # Apply max length with sentence boundary
        if len(content) > self.MAX_RECOMMENDATION_LENGTH:
            truncated = content[:self.MAX_RECOMMENDATION_LENGTH]
            last_period = truncated.rfind('.')
            if last_period > self.MAX_RECOMMENDATION_LENGTH * 0.5:
                content = truncated[:last_period + 1]
            else:
                content = truncated
        
        # Reconstruct with header
        if format_type == 'hsib_2023':
            return f"Safety recommendation {rec_id}: {content}"
        elif format_type == 'hsib_2018':
            return f"Recommendation {rec_id}: {content}"
        else:
            return f"Recommendation {rec_id}: {content}"
    
    def extract_unstructured(self, text: str, min_confidence: float) -> List[Dict]:
        """
        Extract recommendations from unstructured text using sentence analysis.
        Only used when no structured format is detected.
        """
        recommendations = []
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        for idx, sentence in enumerate(sentences):
            cleaned = self.clean_text(sentence)
            
            if len(cleaned) < self.MIN_RECOMMENDATION_LENGTH:
                continue
            if len(cleaned) > self.MAX_SENTENCE_LENGTH:
                continue
            
            # Use semantic analysis for unstructured
            is_rec, confidence = self.deduplicator.is_recommendation_like(cleaned)
            
            if is_rec and confidence >= min_confidence:
                verb = self._extract_verb(cleaned)
                recommendations.append({
                    'text': cleaned,
                    'verb': verb,
                    'method': 'sentence_extraction',
                    'confidence': round(confidence, 3),
                    'position': idx,
                    'in_section': False,
                })
        
        return recommendations
    
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
        
        # Look for "requires/require" pattern
        if 'require' in text_lower:
            return 'require'
        
        # Look for any action verb
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}s?\b', text_lower):
                return verb
        
        return 'unknown'
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.75) -> List[Dict]:
        """
        Extract recommendations from text.
        Uses structure detection first, falls back to sentence extraction.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to extract_recommendations")
            return []
        
        logger.info(f"Extracting recommendations from text of length {len(text)}")
        
        cleaned_text = self.clean_text(text)
        
        # Detect document structure
        format_type, matches = self.detect_structure(cleaned_text)
        
        if format_type != 'unstructured' and matches:
            logger.info(f"Detected {format_type} format with {len(matches)} recommendations")
            recommendations = self.extract_between_headers(cleaned_text, matches, format_type)
        else:
            logger.info("No structured format detected, using sentence extraction")
            recommendations = self.extract_unstructured(text, min_confidence)
        
        # Semantic deduplication
        return self._deduplicate(recommendations)
    
    def _deduplicate(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Remove duplicates using semantic similarity.
        Catches duplicates with different footnotes, formatting, etc.
        """
        if not recommendations:
            return []
        
        unique = []
        
        for rec in recommendations:
            is_duplicate = False
            
            for existing in unique:
                if self.deduplicator.are_similar(
                    rec['text'],
                    existing['text'],
                    threshold=self.SEMANTIC_DEDUP_THRESHOLD
                ):
                    is_duplicate = True
                    logger.debug(f"Duplicate: {rec.get('rec_number', 'N/A')} ≈ {existing.get('rec_number', 'N/A')}")
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
