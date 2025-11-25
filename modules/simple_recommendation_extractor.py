"""
Strict Recommendation Extractor - Eliminates false positives
Designed for government/policy documents with numbered recommendations
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import Counter


class StrictRecommendationExtractor:
    """
    Extract genuine recommendations with aggressive filtering.
    Designed for government documents where recommendations are typically:
    - Numbered (Recommendation 1, Recommendation 2, etc.)
    - Start with entity + "should"
    - Are actionable directives
    """
    
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
            'build', 'design', 'plan', 'prepare', 'bring', 'make', 'take',
        }
        
        # Entities that make recommendations in government docs
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
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Aggressively clean text of all noise
        """
        if not text:
            return ""
        
        original = text
        
        # Remove URLs completely
        text = re.sub(r'https?://[^\s<>"\']+', '', text)
        text = re.sub(r'www\.[^\s<>"\']+', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.gov\.uk[^\s]*', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.org\.uk[^\s]*', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.nhs\.uk[^\s]*', '', text)
        
        # Remove timestamps (e.g., "11/24/25, 5:39 PM")
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?', '', text, flags=re.IGNORECASE)
        
        # Remove page numbers (e.g., "1/84", "Page 5 of 20")
        text = re.sub(r'\b\d+/\d+\b', '', text)
        text = re.sub(r'\bPage\s+\d+\s+(?:of\s+\d+)?', '', text, flags=re.IGNORECASE)
        
        # Remove common PDF artifacts
        text = re.sub(r'Rapid review into data on mental health inpatient settings:.*?GOV\.', '', text, flags=re.IGNORECASE)
        text = re.sub(r'final report and recommendations\s*-\s*GOV\.?', '', text, flags=re.IGNORECASE)
        
        # Remove "UK " prefix (common artifact)
        text = re.sub(r'^UK\s+', '', text)
        
        # Remove reference markers
        text = re.sub(r'\(\s*https?://[^)]+\)', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def is_garbage(self, text: str) -> Tuple[bool, str]:
        """
        Check if text is garbage that should be rejected entirely
        Returns (is_garbage, reason)
        """
        if not text:
            return True, "empty"
        
        # Too short
        if len(text) < 30:
            return True, "too_short"
        
        # Mostly URL or contains URL fragments
        if re.search(r'https?://', text, re.IGNORECASE):
            return True, "contains_url"
        if re.search(r'www\.', text, re.IGNORECASE):
            return True, "contains_www"
        if re.search(r'\.gov\.uk', text, re.IGNORECASE):
            return True, "contains_gov_uk"
        if re.search(r'\.org\.uk', text, re.IGNORECASE):
            return True, "contains_org_uk"
        
        # Contains timestamps
        if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text):
            return True, "contains_timestamp"
        
        # Page number pattern (e.g., "1/84")
        if re.match(r'^[A-Z]*\s*\d+/\d+\s*$', text.strip()):
            return True, "page_number"
        
        # Starts with "UK " followed by nothing useful
        if re.match(r'^UK\s+\d', text):
            return True, "uk_page_artifact"
        
        # Just appendix/section headers
        if re.match(r'^(?:Appendix|Section|Chapter|Table|Figure)\s+\d+', text, re.IGNORECASE):
            return True, "header"
        
        # Sentence fragments (starting with lowercase or certain patterns)
        if re.match(r'^[a-z]', text) and not re.match(r'^[a-z]\)', text):
            return True, "fragment"
        
        # Starts with "address" or similar incomplete fragments
        if re.match(r'^(?:address|ensure|and|or|but|however|therefore|including)\s+', text, re.IGNORECASE):
            if not re.search(r'\bshould\b', text, re.IGNORECASE):
                return True, "fragment"
        
        # Copyright notices
        if re.search(r'©|Crown\s+copyright', text, re.IGNORECASE):
            return True, "copyright"
        
        # Too many special characters (corrupted text)
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,;:!?\'"-()[]')
        if len(text) > 0 and special_chars / len(text) > 0.15:
            return True, "corrupted"
        
        # Excessive numbers (likely data, not recommendations)
        digits = sum(1 for c in text if c.isdigit())
        if len(text) > 0 and digits / len(text) > 0.2:
            return True, "too_many_numbers"
        
        return False, ""
    
    def is_meta_recommendation(self, text: str) -> bool:
        """
        Check if text is talking ABOUT recommendations rather than making one
        """
        text_lower = text.lower()
        
        meta_patterns = [
            r'^the\s+recommendations?\s+(?:have|has|are|were|will|can|may|should)',
            r'^these\s+recommendations?\s+(?:will|can|should|may|have|are)',
            r'^(?:our|their|its|his|her)\s+recommendations?',
            r'recommendations?\s+(?:from|of|in)\s+(?:this|the)\s+review',
            r'(?:implement|implementing|implemented)\s+(?:these|the|all)?\s*recommendations?',
            r'once\s+implemented.*recommendations?',
            r'recommendations?\s+(?:will|can|should)\s+help',
            r'response\s+to\s+(?:the\s+)?recommendations?',
            r'following\s+(?:the\s+)?recommendations?',
            r'based\s+on\s+(?:the\s+)?recommendations?',
            r'make\s+recommendations?\s+about',
            r'i\s+(?:truly\s+)?believe.*recommendations?',
            r'i\s+hope.*recommendations?',
            # Additional patterns for non-recommendations
            r'^our\s+objectives\s+were\s+to',
            r'^we\s+(?:were|are)\s+(?:told|informed|advised)',
            r'^this\s+work\s+should\s+build\s+on',
            r'^data\s+should\s+be\s+entered',
            r'^as\s+far\s+as\s+possible',
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def is_genuine_recommendation(self, text: str) -> Tuple[bool, float, str, str]:
        """
        Determine if text is a genuine recommendation
        Returns (is_recommendation, confidence, method, verb)
        """
        text_lower = text.lower()
        
        # Method 1: Explicit numbered recommendation (highest confidence)
        # e.g., "Recommendation 1 NHS England should..."
        numbered_match = re.match(r'^Recommendation\s+(\d+)\s+(.+)', text, re.IGNORECASE)
        if numbered_match:
            rec_text = numbered_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.98, 'numbered_recommendation', verb
        
        # Method 2: Entity + should pattern
        # e.g., "NHS England should establish..." or "Every board should urgently review..."
        for entity_pattern in self.recommending_entities:
            # Pattern allows for optional adverb (e.g., "should urgently review")
            pattern = rf'{entity_pattern}\s+should\s+(?:also\s+)?(?:\w+ly\s+)?(\w+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.group(1):
                verb = match.group(1).lower()
                if verb in self.action_verbs or verb.endswith('e') or verb.endswith('ise') or verb.endswith('ize'):
                    return True, 0.95, 'entity_should', verb
        
        # Method 3: "We recommend" pattern
        if re.search(r'\bwe\s+recommend\b', text_lower):
            verb = self._extract_verb(text)
            return True, 0.95, 'we_recommend', verb
        
        # Method 4: "should be" + action (passive recommendation)
        # e.g., "Data should be collected..."
        should_be_match = re.search(r'\bshould\s+be\s+(\w+ed)\b', text_lower)
        if should_be_match:
            verb = should_be_match.group(1)
            return True, 0.85, 'should_be_passive', verb
        
        # Method 5: Modal + action verb (but stricter)
        # Must have clear subject and actionable content
        modal_match = re.search(r'\b(should|must|shall)\s+(\w+)\b', text_lower)
        if modal_match:
            modal = modal_match.group(1)
            verb = modal_match.group(2)
            # Verify it's actionable
            if verb in self.action_verbs:
                # Check sentence has reasonable structure
                words = text.split()
                if len(words) >= 8:  # Substantial enough
                    return True, 0.80, 'modal_verb', verb
        
        # Method 6: Imperative starting with action verb
        # e.g., "Ensure that all providers..."
        first_word = text.split()[0].lower() if text.split() else ''
        if first_word in self.action_verbs:
            if len(text.split()) >= 8:
                return True, 0.75, 'imperative', first_word
        
        return False, 0.0, 'none', 'unknown'
    
    def _extract_verb(self, text: str) -> str:
        """Extract the main action verb from recommendation text"""
        text_lower = text.lower()
        
        # Look for "should + verb" pattern
        should_match = re.search(r'\bshould\s+(?:also\s+)?(?:be\s+)?(\w+)', text_lower)
        if should_match:
            verb = should_match.group(1)
            if verb in self.action_verbs:
                return verb
        
        # Look for any action verb
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}\b', text_lower):
                return verb
        
        return 'unknown'
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.7) -> List[Dict]:
        """
        Extract genuine recommendations from text
        
        Args:
            text: Document text or pre-split sentences
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Handle if text is already sentences (from CSV)
        if '\n' in text:
            sentences = text.split('\n')
        else:
            # Split into sentences
            sentences = self._split_sentences(text)
        
        for idx, sentence in enumerate(sentences):
            # Clean the sentence
            cleaned = self.clean_text(sentence)
            
            # Check if it's garbage
            is_garbage, reason = self.is_garbage(cleaned)
            if is_garbage:
                continue
            
            # Check if it's meta-recommendation
            if self.is_meta_recommendation(cleaned):
                continue
            
            # Check if it's a genuine recommendation
            is_rec, confidence, method, verb = self.is_genuine_recommendation(cleaned)
            
            if is_rec and confidence >= min_confidence:
                recommendations.append({
                    'text': cleaned,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': idx,
                })
        
        # Remove duplicates
        recommendations = self._deduplicate(recommendations)
        
        return recommendations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence boundaries and recommendation markers
        text = re.sub(r'(Recommendation\s+\d+)\s+', r'\n\1 ', text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        result = []
        for s in sentences:
            s = s.strip()
            if len(s) > 20:
                # Further split on newlines
                for sub in s.split('\n'):
                    sub = sub.strip()
                    if len(sub) > 20:
                        result.append(sub)
        
        return result
    
    def _deduplicate(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations"""
        seen = set()
        unique = []
        
        for rec in recommendations:
            # Create a normalised key
            key = re.sub(r'\s+', ' ', rec['text'].lower().strip())[:100]
            
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        
        return unique
    
    def get_statistics(self, recommendations: List[Dict]) -> Dict:
        """Get statistics about extracted recommendations"""
        if not recommendations:
            return {'total': 0, 'methods': {}, 'verbs': {}, 'avg_confidence': 0}
        
        verb_counts = Counter(r['verb'] for r in recommendations)
        method_counts = Counter(r['method'] for r in recommendations)
        
        return {
            'total': len(recommendations),
            'methods': dict(method_counts),
            'top_verbs': dict(verb_counts.most_common(10)),
            'avg_confidence': round(sum(r['confidence'] for r in recommendations) / len(recommendations), 3),
            'high_confidence': sum(1 for r in recommendations if r['confidence'] >= 0.9),
            'medium_confidence': sum(1 for r in recommendations if 0.75 <= r['confidence'] < 0.9),
        }


def extract_recommendations(text: str, min_confidence: float = 0.7) -> List[Dict]:
    """
    Main function to extract recommendations
    
    Args:
        text: Document text
        min_confidence: Minimum confidence (0-1)
        
    Returns:
        List of genuine recommendations
    """
    extractor = StrictRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)





if __name__ == "__main__":
    test_with_real_data()
