"""
Recommendation Extractor v4.2 - First-Occurrence Architecture
Detects document structure and extracts ONLY the first occurrence of each recommendation.

v4.2 Changes:
- First-occurrence: only extract the first instance of each rec_number
- Fixes duplicate extraction issue where later occurrences had garbage content
- Cleaner boundary detection for recommendation text
- All 4 recommendations now extracted correctly from HSSIB reports

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

logger = logging.getLogger(__name__)


class StrictRecommendationExtractor:
    """
    First-occurrence recommendation extractor.
    Detects document format, extracts ONLY the first occurrence of each unique recommendation.
    """
    
    MAX_SENTENCE_LENGTH = 500
    MAX_RECOMMENDATION_LENGTH = 800  # Reduced - recommendations should be concise
    MIN_RECOMMENDATION_LENGTH = 50
    
    def __init__(self):
        """Initialise extractor."""
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
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3}):',
            text, re.IGNORECASE
        ))
        if hsib_2023:
            return 'hsib_2023', hsib_2023
        
        # HSIB 2018: "Recommendation 2018/006:"
        hsib_2018 = list(re.finditer(
            r'Recommendation\s+(\d{4}/\d{3}):',
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
    
    def extract_first_occurrences(self, text: str, matches: List, format_type: str) -> List[Dict]:
        """
        Extract ONLY the first occurrence of each unique recommendation.
        This is the key fix - we skip duplicates entirely rather than extracting them.
        """
        recommendations = []
        seen_ids = set()
        
        for match in matches:
            # Get recommendation ID
            if format_type == 'hsib_2023':
                rec_id = match.group(1)  # R/2023/220
            elif format_type == 'hsib_2018':
                rec_id = match.group(1)  # 2018/006
            else:  # standard
                num = match.group(1)
                extra = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                rec_id = f"{num}{extra}" if extra and len(num) == 1 else num
            
            # SKIP if we've already seen this ID
            if rec_id in seen_ids:
                logger.debug(f"Skipping duplicate occurrence of {rec_id}")
                continue
            seen_ids.add(rec_id)
            
            # Extract content after this header
            start_pos = match.end()
            rest_of_text = text[start_pos:]
            
            # Find where this recommendation ends
            end_pos = self._find_recommendation_end(rest_of_text, format_type)
            
            # Extract and clean content
            content = rest_of_text[:end_pos].strip()
            content = self.clean_text(content)
            
            # Validate content
            if not content or len(content) < self.MIN_RECOMMENDATION_LENGTH:
                logger.warning(f"Recommendation {rec_id} content too short: {len(content)} chars")
                continue
            
            # Apply max length with sentence boundary
            if len(content) > self.MAX_RECOMMENDATION_LENGTH:
                truncated = content[:self.MAX_RECOMMENDATION_LENGTH]
                last_period = truncated.rfind('.')
                if last_period > self.MAX_RECOMMENDATION_LENGTH * 0.5:
                    content = truncated[:last_period + 1]
                else:
                    content = truncated.rstrip() + '...'
            
            # Build full recommendation text
            if format_type == 'hsib_2023':
                full_text = f"Safety recommendation {rec_id}: {content}"
            elif format_type == 'hsib_2018':
                full_text = f"Recommendation {rec_id}: {content}"
            else:
                full_text = f"Recommendation {rec_id}: {content}"
            
            verb = self._extract_verb(content)
            
            recommendations.append({
                'text': full_text,
                'verb': verb,
                'method': f'{format_type}_{rec_id}',
                'confidence': 0.98,
                'position': len(recommendations),
                'in_section': True,
                'rec_number': rec_id,
            })
        
        return recommendations
    
    def _find_recommendation_end(self, text: str, format_type: str) -> int:
        """
        Find where the recommendation content ends.
        Returns position in text where content should be cut.
        """
        # End markers - look for these patterns
        end_patterns = [
            # Next recommendation header
            r'Safety\s+recommendation\s+R/\d{4}/\d{3}:',
            r'Recommendation\s+\d{4}/\d{3}:',
            r'Recommendation\s+\d{1,2}\b',
            # Safety observations section
            r'HSIB\s+makes\s+the\s+following\s+safety\s+observation',
            r'Safety\s+observation\s+O/\d{4}/\d{3}:',
            # Section boundaries
            r'\d+\.\d+(?:\.\d+)?\s+[A-Z][a-z]',  # Section numbers like "4.3.1 The..."
            # Other boundaries
            r'HSIB\s+notes\s+the\s+following',
            r'Safety\s+action\s+A/',
            r'\n\s*Response\b',
            r'\n\s*Appendix',
            r'\n\s*References\s*$',
        ]
        
        min_pos = len(text)
        
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.start() < min_pos:
                min_pos = match.start()
        
        return min_pos
    
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
            
            # Check if it looks like a recommendation
            is_rec, confidence = self._is_recommendation_like(cleaned)
            
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
    
    def _is_recommendation_like(self, text: str) -> Tuple[bool, float]:
        """Check if text looks like a recommendation."""
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
        
        # Look for "recommends that [entity] [verb]" pattern
        recommends_match = re.search(r'recommends\s+that\s+(?:the\s+)?(?:\w+\s+){1,5}(\w+s?)\b', text_lower)
        if recommends_match:
            verb = recommends_match.group(1)
            # Strip trailing 's' for verb matching
            verb_base = verb.rstrip('s')
            if verb_base in self.action_verbs or verb in self.action_verbs:
                return verb_base if verb_base in self.action_verbs else verb
        
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
        
        # Clean text for structure detection
        # Note: We DON'T fully clean here - keep original for position matching
        
        # Detect document structure
        format_type, matches = self.detect_structure(text)
        
        if format_type != 'unstructured' and matches:
            logger.info(f"Detected {format_type} format with {len(matches)} headers (may include duplicates)")
            recommendations = self.extract_first_occurrences(text, matches, format_type)
            logger.info(f"Extracted {len(recommendations)} unique recommendations")
        else:
            logger.info("No structured format detected, using sentence extraction")
            recommendations = self.extract_unstructured(text, min_confidence)
            # Deduplicate unstructured extractions
            recommendations = self._deduplicate_by_text(recommendations)
        
        return recommendations
    
    def _deduplicate_by_text(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Remove duplicates based on text similarity for unstructured extraction.
        """
        if not recommendations:
            return []
        
        unique = []
        seen_texts = set()
        
        for rec in recommendations:
            # Use first 100 chars as dedup key
            key = rec['text'][:100].lower().strip()
            if key not in seen_texts:
                seen_texts.add(key)
                unique.append(rec)
        
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
