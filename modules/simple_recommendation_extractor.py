"""
Improved Strict Recommendation Extractor v2.0
Eliminates false positives from PDF artifacts, URLs, timestamps, and meta-commentary.
Designed for government/policy documents with numbered recommendations.

Key improvements over original:
1. Pre-filters garbage BEFORE any analysis (URLs, timestamps, page numbers)
2. Stricter confidence assignment - not everything gets 0.9
3. Better detection of meta-recommendations (text ABOUT recommendations)
4. Focuses on numbered "Recommendation N" patterns as primary signal
5. Character encoding cleanup for common PDF extraction issues

Usage:
    from simple_recommendation_extractor import extract_recommendations
    
    recommendations = extract_recommendations(document_text, min_confidence=0.75)
    for rec in recommendations:
        print(f"{rec['method']}: {rec['text'][:100]}...")
"""

import re
from typing import List, Dict, Tuple
from collections import Counter


class StrictRecommendationExtractor:
    """
    Extract genuine recommendations with aggressive pre-filtering.
    
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
        
        # Patterns that indicate garbage text (to reject early)
        self.garbage_patterns = [
            r'https?://',                              # URLs
            r'www\.',                                  # URLs
            r'\.gov\.uk',                              # UK gov domains
            r'\.org\.uk',                              # UK org domains  
            r'\.nhs\.uk',                              # NHS domains
            r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}',  # Timestamps
            r'\b\d+/\d+\b(?!\s*(?:of|percent))',       # Page numbers like 1/84
            r'^\s*UK\s+\d',                            # "UK 1/84" artifacts
            r'^\s*\d+/\d+\s*$',                        # Just page numbers
            r'©|Crown\s+copyright',                    # Copyright notices
            r'GOV\.UK',                                # GOV.UK references
            r'\d{1,2}/\d{1,2}/\d{2}.*(?:AM|PM)',       # Timestamp patterns
        ]
        
        # Compile garbage patterns for efficiency
        self.garbage_regex = re.compile('|'.join(self.garbage_patterns), re.IGNORECASE)
    
    def fix_encoding(self, text: str) -> str:
        """Fix common PDF extraction encoding issues"""
        if not text:
            return ""
        
        # Common encoding corruptions
        replacements = {
            'â€™': "'",      # Smart apostrophe
            'â€œ': '"',      # Smart quote open
            'â€': '"',       # Smart quote close
            'â€"': '—',      # Em dash
            'â€"': '–',      # En dash
            'Â ': ' ',       # Non-breaking space corruption
            '\u00a0': ' ',   # Actual non-breaking space
            '�': '',         # Replacement character
            '\ufffd': '',    # Unicode replacement
        }
        
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Aggressively clean text of all noise"""
        if not text:
            return ""
        
        text = self.fix_encoding(text)
        
        # Remove URLs completely
        text = re.sub(r'https?://[^\s<>"\']+', '', text)
        text = re.sub(r'www\.[^\s<>"\']+', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.gov\.uk[^\s]*', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.org\.uk[^\s]*', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.nhs\.uk[^\s]*', '', text)
        
        # Remove timestamps
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?', '', text, flags=re.IGNORECASE)
        
        # Remove page numbers
        text = re.sub(r'\b\d+/\d+\b', '', text)
        text = re.sub(r'\bPage\s+\d+\s+(?:of\s+\d+)?', '', text, flags=re.IGNORECASE)
        
        # Remove common PDF artifacts
        text = re.sub(r'Rapid review into data on mental health inpatient settings:.*?GOV\.?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'final report and recommendations\s*-\s*GOV\.?', '', text, flags=re.IGNORECASE)
        
        # Remove "UK " prefix (common artifact)
        text = re.sub(r'^UK\s+', '', text)
        
        # Remove GOV.UK references at end
        text = re.sub(r'\s*-?\s*GOV\.?\s*UK?\s*$', '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def is_garbage(self, text: str) -> Tuple[bool, str]:
        """
        FIRST-PASS filter: reject obvious garbage before any analysis.
        Returns (is_garbage, reason)
        """
        if not text:
            return True, "empty"
        
        original_text = text
        
        # Check for garbage patterns in ORIGINAL text (before cleaning)
        if self.garbage_regex.search(original_text):
            return True, "contains_garbage_pattern"
        
        # Too short after cleaning
        cleaned = self.clean_text(text)
        if len(cleaned) < 40:
            return True, "too_short"
        
        # Just section headers
        if re.match(r'^(?:Appendix|Section|Chapter|Table|Figure|Footnote)\s+\d+', cleaned, re.IGNORECASE):
            return True, "header"
        
        # Starts with lowercase (sentence fragment) - unless it's a list item
        if re.match(r'^[a-z]', cleaned) and not re.match(r'^[a-z]\)', cleaned):
            # Allow if it contains "should" (continuation of recommendation)
            if not re.search(r'\bshould\b', cleaned, re.IGNORECASE):
                return True, "fragment"
        
        # Too many special characters (corrupted text)
        special_chars = sum(1 for c in cleaned if not c.isalnum() and not c.isspace() and c not in '.,;:!?\'"-()[]')
        if len(cleaned) > 0 and special_chars / len(cleaned) > 0.12:
            return True, "corrupted"
        
        # Excessive numbers (likely data tables, not recommendations)
        digits = sum(1 for c in cleaned if c.isdigit())
        if len(cleaned) > 0 and digits / len(cleaned) > 0.15:
            return True, "too_many_numbers"
        
        return False, ""
    
    def is_meta_recommendation(self, text: str) -> bool:
        """
        Check if text is talking ABOUT recommendations rather than making one.
        These should be rejected.
        """
        text_lower = text.lower()
        
        meta_patterns = [
            # Talking about recommendations in general
            r'^the\s+recommendations?\s+(?:have|has|are|were|will|can|may|should|from)',
            r'^these\s+recommendations?\s+(?:will|can|should|may|have|are)',
            r'^(?:our|their|its|his|her)\s+recommendations?',
            r'^this\s+recommendation\s+is\s+for',
            
            # References to recommendations
            r'recommendations?\s+(?:from|of|in)\s+(?:this|the)\s+(?:review|report)',
            r'(?:implement|implementing|implemented)\s+(?:these|the|all|our)?\s*recommendations?',
            r'once\s+implemented.*recommendations?',
            r'when\s+implemented.*recommendations?',
            r'recommendations?\s+(?:will|can|should)\s+help',
            r'response\s+to\s+(?:the\s+)?recommendations?',
            r'following\s+(?:the\s+)?recommendations?',
            r'based\s+on\s+(?:the\s+)?recommendations?',
            r'make\s+recommendations?\s+about',
            r'progress\s+against\s+(?:these\s+)?recommendations?',
            
            # Opinion/belief statements
            r'i\s+(?:truly\s+)?believe.*recommendations?',
            r'i\s+hope.*recommendations?',
            r'we\s+believe\s+that.*recommendations?',
            
            # Review methodology
            r'^our\s+objectives\s+were\s+to',
            r'^we\s+(?:were|are)\s+(?:told|informed|advised)',
            r'^when\s+we\s+(?:first\s+)?established',
            r'^we\s+found\s+that',
            r'^we\s+were\s+(?:especially\s+)?impressed',
            r'^we\s+spoke\s+to',
            r'^the\s+review\s+(?:did|does|has|was)',
            
            # Context/background
            r'^this\s+work\s+should\s+build\s+on',
            r'^data\s+should\s+be\s+entered',
            r'^as\s+far\s+as\s+possible',
            r'^for\s+a\s+person\s+to\s+feel',
            r'^more\s+generally,?\s+we\s+found',
            r'^they\s+proposed\s+practical',
            
            # Descriptions (not directives)
            r'identifies?\s+national\s+trends',
            r'summarises?\s+the\s+lives',
            r'provides?\s+(?:a\s+)?(?:tertiary|secondary)',
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def is_genuine_recommendation(self, text: str) -> Tuple[bool, float, str, str]:
        """
        Determine if text is a genuine recommendation.
        Returns (is_recommendation, confidence, method, verb)
        
        Confidence levels:
        - 0.98: Explicit numbered "Recommendation N" pattern
        - 0.95: Entity + should pattern
        - 0.90: "We recommend" pattern
        - 0.85: Should be + past participle (passive)
        - 0.80: Modal + action verb
        - 0.75: Imperative with action verb
        """
        cleaned = self.clean_text(text)
        text_lower = cleaned.lower()
        
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
            if verb == 'be':
                be_match = re.search(r'\bshould\s+be\s+(\w+)', text_lower)
                if be_match:
                    return be_match.group(1)
        
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}\b', text_lower):
                return verb
        
        return 'unknown'
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.75) -> List[Dict]:
        """
        Extract genuine recommendations from text.
        
        Args:
            text: Document text or pre-split sentences
            min_confidence: Minimum confidence threshold (default 0.75)
            
        Returns:
            List of recommendation dictionaries with keys:
            - text: cleaned recommendation text
            - verb: action verb identified
            - method: extraction method used
            - confidence: confidence score (0-1)
            - position: position in source
        """
        recommendations = []
        
        if '\n' in text:
            sentences = text.split('\n')
        else:
            sentences = self._split_sentences(text)
        
        for idx, sentence in enumerate(sentences):
            # FIRST: Check garbage (before cleaning)
            is_garbage, reason = self.is_garbage(sentence)
            if is_garbage:
                continue
            
            cleaned = self.clean_text(sentence)
            
            if len(cleaned) < 40:
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
                })
        
        recommendations = self._deduplicate(recommendations)
        
        return recommendations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(Recommendation\s+\d+)\s+', r'\n\1 ', text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        result = []
        for s in sentences:
            s = s.strip()
            if len(s) > 30:
                for sub in s.split('\n'):
                    sub = sub.strip()
                    if len(sub) > 30:
                        result.append(sub)
        
        return result
    
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


def extract_recommendations(text: str, min_confidence: float = 0.75) -> List[Dict]:
    """
    Main function to extract recommendations.
    
    Args:
        text: Document text
        min_confidence: Minimum confidence (0-1)
        
    Returns:
        List of genuine recommendations
    """
    extractor = StrictRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)


def test_with_real_data():
    """Test with actual problematic examples"""
    
    test_cases = [
        # Should REJECT
        ("UK https://www.gov.uk/government/publications/rapid-review... 1/84", False),
        ("11/24/25, 5:39 PM Rapid review into data on mental health inpatient settings", False),
        ("The recommendations have identified ways in which the system can improve...", False),
        ("I truly believe that the recommendations from this review can improve...", False),
        ("Once implemented, these recommendations will help contribute towards saving lives.", False),
        # Should ACCEPT
        ("Recommendation 1 NHS England should establish a programme of work.", True),
        ("Recommendation 3 ICSs and provider collaboratives should bring together trusts.", True),
        ("All providers of NHS-funded care should review the information they provide.", True),
        ("Every provider board should urgently review its membership and skillset.", True),
        ("Boards should consider annual mandatory training for their members on data literacy.", True),
    ]
    
    extractor = StrictRecommendationExtractor()
    
    print("=" * 80)
    print("STRICT RECOMMENDATION EXTRACTOR - TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for text, should_extract in test_cases:
        cleaned = extractor.clean_text(text)
        is_garbage, reason = extractor.is_garbage(text)
        is_meta = extractor.is_meta_recommendation(cleaned) if not is_garbage else False
        is_rec, conf, method, verb = extractor.is_genuine_recommendation(cleaned) if not is_garbage and not is_meta else (False, 0, 'n/a', 'n/a')
        
        extracted = is_rec and conf >= 0.75
        correct = extracted == should_extract
        
        status = "✓ PASS" if correct else "✗ FAIL"
        if correct:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status}")
        print(f"  Input: {text[:70]}...")
        print(f"  Expected: {'EXTRACT' if should_extract else 'REJECT'}")
        print(f"  Got: {'EXTRACT' if extracted else 'REJECT'}", end="")
        if is_garbage:
            print(f" (garbage: {reason})")
        elif is_meta:
            print(" (meta-recommendation)")
        elif extracted:
            print(f" (conf: {conf:.2f}, method: {method}, verb: {verb})")
        else:
            print(" (not a recommendation)")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)


if __name__ == "__main__":
    test_with_real_data()
