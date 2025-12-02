"""
Improved Strict Recommendation Extractor v2.3
Eliminates false positives from PDF artifacts, URLs, timestamps, and meta-commentary.
Designed for government/policy documents with numbered recommendations.

v2.3 Changes:
- Fixed chunking to handle multiple numbering formats (Recommendation N, N. Title, etc.)
- Added maximum length filter to reject entire documents/sections as single recommendations
- Better handling of numbered list formats common in operational documents

Usage:
    from simple_recommendation_extractor import extract_recommendations
    
    recommendations = extract_recommendations(document_text, min_confidence=0.75)
    for rec in recommendations:
        print(f"{rec['method']}: {rec['text'][:100]}...")
"""

import re
import logging
from typing import List, Dict, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class StrictRecommendationExtractor:
    """
    Extract genuine recommendations with aggressive pre-filtering.
    """
    
    # Maximum length for a single recommendation (characters)
    MAX_RECOMMENDATION_LENGTH = 1500
    
    def __init__(self):
        """Initialise with strict patterns"""
        
        # Action verbs for recommendations
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
            'come', 'form', 'work', 'learn', 'drive',
        }
        
        # Entities that receive recommendations in government docs
        self.recommending_entities = [
            r'NHS\s+England',
            r'(?:the\s+)?(?:Home\s+Office|Cabinet\s+Office|Treasury)',
            r'(?:the\s+)?(?:Department|Ministry)\s+(?:of|for)\s+\w+',
            r'(?:the\s+)?(?:Secretary|Minister)\s+of\s+State',
            r'(?:the\s+)?CQC|Care\s+Quality\s+Commission',
            r'(?:the\s+)?IOPC',
            r'(?:the\s+)?College\s+of\s+Policing',
            r'(?:the\s+)?(?:ICS|ICB)s?',
            r'(?:the\s+)?(?:Trust|Provider|Board)s?',
            r'(?:the\s+)?Government',
            r'(?:the\s+)?(?:Review|Committee|Panel|Commission|Inquiry)',
            r'(?:All|Every)\s+(?:providers?|commissioners?|trusts?|boards?)',
            r'Every\s+(?:provider\s+)?board',
            r'(?:Professional\s+)?bodies',
            r'(?:Local\s+)?(?:authorities|councils)',
            r'DHSC',
            r'Regulators?',
            r'Inpatient\s+staff',
            r'Government\s+ministers?',
        ]
    
    def fix_encoding(self, text: str) -> str:
        """Fix common PDF extraction encoding issues"""
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
        """Aggressively clean text of all noise"""
        if not text:
            return ""
        
        text = self.fix_encoding(text)
        
        # Remove URLs
        text = re.sub(r'https?://[^\s<>"\']+', '', text)
        text = re.sub(r'www\.[^\s<>"\']+', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.gov\.uk[^\s]*', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.org\.uk[^\s]*', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.nhs\.uk[^\s]*', '', text)
        
        # Remove timestamps (including the trailing document title)
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?[^A-Z]*(?=[A-Z]|$)', ' ', text, flags=re.IGNORECASE)
        
        # Remove page numbers
        text = re.sub(r'\b\d+/\d+\b', '', text)
        text = re.sub(r'\bPage\s+\d+\s+(?:of\s+\d+)?', '', text, flags=re.IGNORECASE)
        
        # Remove common PDF artifacts
        text = re.sub(r'Rapid review into data on mental health inpatient settings:.*?GOV\.?UK', '', text, flags=re.IGNORECASE)
        text = re.sub(r'final report and recommendations\s*-?\s*GOV\.?UK?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'-\s*GOV\.?\s*UK?\s*', '', text, flags=re.IGNORECASE)
        
        # Remove "UK " prefix
        text = re.sub(r'^UK\s+', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def is_garbage(self, text: str) -> Tuple[bool, str]:
        """FIRST-PASS filter: reject obvious garbage before any analysis."""
        if not text:
            return True, "empty"
        
        # Clean first for this check
        cleaned = self.clean_text(text)
        
        if len(cleaned) < 40:
            return True, "too_short"
        
        # NEW: Reject text that's too long to be a single recommendation
        if len(cleaned) > self.MAX_RECOMMENDATION_LENGTH:
            return True, "too_long"
        
        # Just section headers
        if re.match(r'^(?:Appendix|Section|Chapter|Table|Figure|Footnote)\s+\d+', cleaned, re.IGNORECASE):
            return True, "header"
        
        # Too many special characters
        special_chars = sum(1 for c in cleaned if not c.isalnum() and not c.isspace() and c not in '.,;:!?\'"-()[]')
        if len(cleaned) > 0 and special_chars / len(cleaned) > 0.12:
            return True, "corrupted"
        
        # Excessive numbers
        digits = sum(1 for c in cleaned if c.isdigit())
        if len(cleaned) > 0 and digits / len(cleaned) > 0.15:
            return True, "too_many_numbers"
        
        return False, ""
    
    def is_meta_recommendation(self, text: str) -> bool:
        """Check if text is talking ABOUT recommendations rather than making one."""
        text_lower = text.lower()
        
        meta_patterns = [
            r'^the\s+recommendations?\s+(?:have|has|are|were|will|can|may|should|from)',
            r'^these\s+recommendations?\s+(?:will|can|should|may|have|are)',
            r'^(?:our|their|its|his|her)\s+recommendations?',
            r'recommendations?\s+(?:from|of|in)\s+(?:this|the)\s+(?:review|report)',
            r'(?:implement|implementing|implemented)\s+(?:these|the|all|our)?\s*recommendations?',
            r'once\s+implemented.*recommendations?',
            r'recommendations?\s+(?:will|can|should)\s+help',
            r'i\s+(?:truly\s+)?believe.*recommendations?',
            r'i\s+hope.*recommendations?',
            r'^our\s+objectives\s+were\s+to',
            r'^we\s+(?:were|are)\s+(?:told|informed|advised)',
            r'^when\s+we\s+(?:first\s+)?established',
            r'^we\s+found\s+that',
            r'^they\s+proposed\s+practical',
            # NEW: Detect document introductions/preambles
            r'^this\s+document\s+outlines',
            r'^the\s+insights\s+below',
            r'^the\s+goal\s+is\s+to\s+provide',
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def is_genuine_recommendation(self, text: str) -> Tuple[bool, float, str, str]:
        """Determine if text is a genuine recommendation."""
        cleaned = self.clean_text(text)
        text_lower = cleaned.lower()
        
        # EARLY EXIT: If text is too long, it's not a single recommendation
        if len(cleaned) > self.MAX_RECOMMENDATION_LENGTH:
            return False, 0.0, 'too_long', 'none'
        
        # Method 1: Explicit numbered recommendation (highest confidence)
        numbered_match = re.match(r'^(?:Recommendations?\s+)?Recommendation\s+(\d+)\s+(.+)', cleaned, re.IGNORECASE)
        if numbered_match:
            rec_num = numbered_match.group(1)
            rec_text = numbered_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.98, f'numbered_recommendation_{rec_num}', verb
        
        # Method 2: Entity + should pattern
        for entity_pattern in self.recommending_entities:
            pattern = rf'{entity_pattern}\s+should\s+(?:also\s+)?(?:urgently\s+)?(?:\w+ly\s+)?(\w+)'
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match and match.group(1):
                verb = match.group(1).lower()
                if verb in self.action_verbs or verb.endswith('e') or verb.endswith('ise') or verb.endswith('ize'):
                    return True, 0.95, 'entity_should', verb
        
        # Method 3: "We recommend" pattern
        if re.search(r'\bwe\s+recommend\b', text_lower):
            verb = self._extract_verb(cleaned)
            return True, 0.90, 'we_recommend', verb
        
        # Method 4: "should be" + action (passive)
        should_be_match = re.search(r'(\w+(?:\s+\w+)?)\s+should\s+be\s+(\w+ed)\b', text_lower)
        if should_be_match:
            subject = should_be_match.group(1)
            verb = should_be_match.group(2)
            if subject not in ['this', 'it', 'that', 'recommendation', 'the recommendation']:
                return True, 0.85, 'should_be_passive', verb
        
        # Method 5: Modal + action verb (stricter)
        modal_match = re.search(r'\b(should|must|shall)\s+(\w+)\b', text_lower)
        if modal_match:
            verb = modal_match.group(2)
            if verb in self.action_verbs:
                words = cleaned.split()
                if len(words) >= 10:
                    if not re.search(r'recommendations?\s+(?:should|must|shall)', text_lower):
                        return True, 0.80, 'modal_verb', verb
        
        # Method 6: Imperative starting with action verb
        first_word = cleaned.split()[0].lower() if cleaned.split() else ''
        if first_word in self.action_verbs:
            if len(cleaned.split()) >= 10:
                return True, 0.75, 'imperative', first_word
        
        return False, 0.0, 'none', 'unknown'
    
    def _extract_verb(self, text: str) -> str:
        """Extract the main action verb from recommendation text"""
        text_lower = text.lower()
        
        should_match = re.search(r'\bshould\s+(?:also\s+)?(?:urgently\s+)?(?:be\s+)?(\w+)', text_lower)
        if should_match:
            verb = should_match.group(1)
            if verb in self.action_verbs:
                return verb
        
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}\b', text_lower):
                return verb
        
        return 'unknown'
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.75) -> List[Dict]:
        """Extract genuine recommendations from text."""
        recommendations = []
        
        if not text or not text.strip():
            logger.warning("Empty text passed to extract_recommendations")
            return []
        
        logger.info(f"Extracting recommendations from text of length {len(text)}")
        
        # IMPROVED: Split on multiple recommendation boundary patterns
        # Pattern 1: "Recommendation N" (government doc style)
        # Pattern 2: "N. " at start of line or after paragraph break (numbered list style)
        # Pattern 3: Paragraph breaks
        
        # First try "Recommendation N" boundaries
        chunks = re.split(r'(?=Recommendation\s+\d+\s)', text, flags=re.IGNORECASE)
        
        # If that didn't split much, try numbered list pattern
        if len(chunks) <= 2:
            # Split on "N. Title" patterns (numbered sections)
            chunks = re.split(r'(?=(?:^|\n\s*)\d{1,2}\.\s+[A-Z])', text)
        
        logger.info(f"Split into {len(chunks)} chunks using boundary patterns")
        
        # Also extract sub-recommendations with entity+should patterns
        all_sentences = []
        for chunk in chunks:
            # Skip chunks that are too long (likely entire documents or sections)
            # A genuine recommendation should rarely exceed MAX_RECOMMENDATION_LENGTH
            if len(chunk) > self.MAX_RECOMMENDATION_LENGTH:
                # For long chunks, only extract sentences, don't add the chunk itself
                sub_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', chunk)
                for sub in sub_sentences:
                    if len(sub) > 50 and len(sub) <= self.MAX_RECOMMENDATION_LENGTH:
                        all_sentences.append(sub)
            else:
                # Add the chunk itself if reasonable length
                all_sentences.append(chunk)
                
                # Also split on sentence boundaries for entity+should patterns
                sub_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', chunk)
                for sub in sub_sentences:
                    if sub not in all_sentences and len(sub) > 50:
                        all_sentences.append(sub)
        
        logger.info(f"Total sentences to process: {len(all_sentences)}")
        
        in_rec_section = False
        
        for idx, sentence in enumerate(all_sentences):
            # Check for section markers
            if re.search(r'^#+\s*Recommendations?\s*$|^Recommendations?\s*$', sentence.strip(), re.IGNORECASE):
                in_rec_section = True
                continue
            
            # Check garbage (now includes length check)
            is_garbage, reason = self.is_garbage(sentence)
            if is_garbage:
                continue
            
            cleaned = self.clean_text(sentence)
            
            if len(cleaned) < 40:
                continue
            
            # Double-check length after cleaning
            if len(cleaned) > self.MAX_RECOMMENDATION_LENGTH:
                continue
            
            if self.is_meta_recommendation(cleaned):
                continue
            
            is_rec, confidence, method, verb = self.is_genuine_recommendation(cleaned)
            
            if is_rec and confidence >= min_confidence:
                recommendations.append({
                    'text': cleaned,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': idx,
                    'in_section': in_rec_section,
                })
        
        recommendations = self._deduplicate(recommendations)
        
        logger.info(f"Found {len(recommendations)} recommendations")
        
        return recommendations
    
    def _deduplicate(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations"""
        seen = set()
        unique = []
        
        for rec in recommendations:
            key = re.sub(r'\s+', ' ', rec['text'].lower().strip())[:100]
            
            if key not in seen:
                seen.add(key)
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
