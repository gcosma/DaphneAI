"""
Improved Recommendation Extractor - Enhanced version with hyperlink filtering
Fixes false positives, removes hyperlinks, and improves extraction quality
"""

import re
from typing import List, Dict, Set, Optional
from collections import Counter

# Try to import NLTK
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


class ImprovedRecommendationExtractor:
    """Extract recommendations with better filtering, validation, and hyperlink removal"""
    
    def __init__(self):
        """Initialise the extractor with improved patterns"""
        
        # Comprehensive action verbs including those from government documents
        self.action_verbs = {
            # Core recommendation verbs
            'should', 'must', 'shall', 'need', 'require', 'recommend',
            'suggest', 'propose', 'advise', 'urge', 'ensure', 'establish',
            'implement', 'develop', 'create', 'improve', 'enhance',
            'strengthen', 'expand', 'increase', 'reduce', 'review',
            'assess', 'evaluate', 'consider', 'adopt', 'introduce',
            'maintain', 'monitor', 'provide', 'support', 'enable',
            'facilitate', 'promote', 'encourage', 'prioritise', 'prioritize',
            'focus', 'address', 'tackle', 'resolve', 'prevent', 'avoid',
            
            # Government/policy specific verbs
            'commission', 'consult', 'update', 'clarify', 'publish',
            'engage', 'deliver', 'conduct', 'undertake', 'initiate',
            'collaborate', 'coordinate', 'oversee', 'regulate', 'enforce',
            'mandate', 'allocate', 'fund', 'resource', 'train',
            'educate', 'inform', 'notify', 'report', 'audit',
            'inspect', 'investigate', 'examine', 'scrutinise', 'scrutinize',
            'reform', 'revise', 'amend', 'modify', 'change',
            'streamline', 'simplify', 'standardise', 'standardize',
            'centralise', 'centralize', 'decentralise', 'decentralize',
            'integrate', 'consolidate', 'merge', 'separate', 'divide',
            'extend', 'limit', 'restrict', 'relax', 'remove',
            'abolish', 'terminate', 'discontinue', 'suspend', 'resume',
            'accelerate', 'expedite', 'delay', 'postpone', 'schedule',
            'plan', 'design', 'build', 'construct', 'install',
            'upgrade', 'modernise', 'modernize', 'digitise', 'digitize',
            'automate', 'process', 'handle', 'manage', 'administer',
            'govern', 'lead', 'direct', 'guide', 'steer',
            'set', 'define', 'specify', 'determine', 'decide',
            'approve', 'authorise', 'authorize', 'permit', 'allow',
            'prohibit', 'ban', 'forbid', 'disallow', 'reject',
            'accept', 'agree', 'confirm', 'verify', 'validate',
            'certify', 'accredit', 'license', 'register', 'record',
            'document', 'log', 'track', 'trace', 'measure',
            'quantify', 'analyse', 'analyze', 'study', 'research',
            'survey', 'poll', 'canvass', 'gather', 'collect',
            'compile', 'aggregate', 'synthesise', 'synthesize', 'summarise',
            'summarize', 'outline', 'detail', 'explain', 'describe',
            'communicate', 'disseminate', 'distribute', 'share', 'release',
            'disclose', 'reveal', 'expose', 'highlight', 'emphasise',
            'emphasize', 'stress', 'underline', 'underscore', 'reinforce',
            'safeguard', 'protect', 'secure', 'defend', 'preserve',
            'conserve', 'sustain', 'uphold', 'maintain', 'retain',
            'keep', 'hold', 'store', 'archive', 'backup',
            'restore', 'recover', 'retrieve', 'return', 'repay',
            'reimburse', 'compensate', 'reward', 'incentivise', 'incentivize',
            'motivate', 'inspire', 'empower', 'equip', 'prepare',
            'ready', 'mobilise', 'mobilize', 'deploy', 'assign',
            'appoint', 'recruit', 'hire', 'employ', 'engage',
            'dismiss', 'fire', 'release', 'transfer', 'relocate',
            'move', 'shift', 'transition', 'transform', 'convert',
            'adapt', 'adjust', 'calibrate', 'tune', 'optimise',
            'optimise', 'maximise', 'maximize', 'minimise', 'minimize',
        }
        
        # Phrases that strongly indicate recommendations
        self.recommendation_phrases = [
            r'\b(?:we|I|the\s+(?:review|committee|board|panel|group|commission|inquiry|report))\s+recommend',
            r'\bit\s+is\s+recommended\s+that\b',
            r'\bshould\s+(?:be\s+)?(?:required|obliged|mandated)\s+to\b',
            r'\b(?:should|must|shall)\s+\w+',
            r'\bneed(?:s)?\s+to\s+\w+',
            r'\bit\s+is\s+(?:essential|important|critical|vital|necessary|imperative)\s+(?:to|that)\b',
            r'\b(?:ensure|establish|implement|develop)\s+that\b',
            r'\baction\s+(?:should|must)\s+be\s+taken\b',
            r'\bsteps\s+(?:should|must)\s+be\s+taken\b',
            r'\b(?:government|department|ministry|agency|authority)\s+should\b',
            r'\brecommendation\s*(?:\d+)?[:.]\s*',
        ]
        
        # Comprehensive URL and hyperlink patterns
        self.url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',  # Standard URLs
            r'www\.[^\s<>"{}|\\^`\[\]]+',  # www URLs
            r'[a-zA-Z0-9.-]+\.(?:gov|org|com|co|net|edu|uk|eu|io|info|biz|ac)\b(?:/[^\s]*)?',  # Domain patterns
            r'mailto:[^\s<>"]+',  # Email links
            r'ftp://[^\s<>"]+',  # FTP links
            r'\[(?:link|url|http|www)[^\]]*\]',  # Markdown-style links
            r'<a\s+href=[^>]+>[^<]*</a>',  # HTML anchor tags
            r'HYPERLINK\s*"[^"]*"',  # Excel hyperlink formula
            r'=HYPERLINK\([^)]+\)',  # Excel HYPERLINK function
        ]
        
        # Patterns that indicate false positives
        self.exclusion_patterns = [
            r'^UK\s+https?://',  # URLs with UK prefix
            r'^\d+/\d+',  # Page numbers
            r'^https?://',  # URLs at start
            r'\.gov\.uk',  # Government URLs
            r'\.org\.uk',  # Organisation URLs
            r'^\s*\d+\s*$',  # Just numbers
            r'^Appendix\s+\d+',  # Appendix headers
            r'^\s*$',  # Empty lines
            r'©\s*Crown\s+copyright',  # Copyright notices
            r'^ISBN\s',  # ISBN numbers
            r'^ISSN\s',  # ISSN numbers
            r'^\d{4}-\d{4}',  # Year ranges or ISSNs
            r'^Table\s+\d+',  # Table references
            r'^Figure\s+\d+',  # Figure references
            r'^Source:',  # Source citations
            r'^Note:',  # Notes
            r'^\[\d+\]',  # Reference markers
            r'^See\s+(?:also|above|below|section|page|chapter|appendix)',  # Cross-references
            r'^(?:Tel|Fax|Email|Phone|Address):\s',  # Contact details
            r'^[A-Z]{2,}\d+',  # Document codes
        ]
        
        # Phrases that indicate talking ABOUT recommendations (not making them)
        self.meta_recommendation_phrases = [
            r'the\s+recommendations\s+(?:have|are|were|will)',
            r'these\s+recommendations\s+(?:will|can|should|may)',
            r'recommendations\s+(?:from|of|in)\s+(?:this|the)\s+review',
            r'(?:proposed|made|provided)\s+(?:\w+\s+)?recommendations',
            r'recommendations\s+(?:identified|highlighted|outlined)',
            r'(?:implement|implementing|implemented)\s+(?:these|the)\s+recommendations',
            r'recommendations\s+(?:for|to)\s+(?:improve|address)',
            r'response\s+to\s+(?:the\s+)?recommendations?',
            r'recommendations?\s+(?:has|have)\s+been\s+(?:accepted|rejected|noted)',
            r'following\s+(?:the\s+)?recommendations?',
            r'in\s+(?:response|relation)\s+to\s+(?:the\s+)?recommendations?',
        ]
        
        # Government entities commonly found in UK documents
        self.government_entities = {
            'home office', 'iopc', 'college of policing', 'met police',
            'metropolitan police', 'government', 'department', 'ministry',
            'cabinet office', 'treasury', 'nhs', 'cps', 'crown prosecution',
            'hmrc', 'dwp', 'moj', 'ministry of justice', 'home secretary',
            'secretary of state', 'minister', 'commissioner', 'chief constable',
            'police force', 'authority', 'council', 'board', 'agency',
            'service', 'trust', 'foundation', 'inspectorate', 'regulator',
        }
    
    def clean_hyperlinks(self, text: str) -> str:
        """
        Remove all hyperlinks and URLs from text
        
        Args:
            text: Input text potentially containing hyperlinks
            
        Returns:
            Cleaned text with hyperlinks removed
        """
        cleaned = text
        
        # Remove all URL patterns
        for pattern in self.url_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove orphaned link text indicators
        cleaned = re.sub(r'\[\s*\]', '', cleaned)  # Empty brackets
        cleaned = re.sub(r'\(\s*\)', '', cleaned)  # Empty parentheses
        cleaned = re.sub(r'<\s*>', '', cleaned)  # Empty angle brackets
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Clean up punctuation issues from URL removal
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        cleaned = re.sub(r'([.,;:!?])\s*\1+', r'\1', cleaned)  # Duplicate punctuation
        
        return cleaned.strip()
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.5) -> List[Dict]:
        """
        Extract recommendations with improved filtering and hyperlink removal
        
        Args:
            text: Document text
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            List of recommendation dictionaries
        """
        # First, clean hyperlinks from the entire text
        text = self.clean_hyperlinks(text)
        
        recommendations = []
        sentences = self._split_sentences(text)
        
        for idx, sentence in enumerate(sentences):
            # Clean any remaining hyperlinks in the sentence
            sentence = self.clean_hyperlinks(sentence)
            
            # Skip if too short after cleaning
            if len(sentence.strip()) < 20:
                continue
            
            # Skip if it's a false positive
            if self._is_false_positive(sentence):
                continue
            
            # Skip if it's talking ABOUT recommendations
            if self._is_meta_recommendation(sentence):
                continue
            
            # Skip if it still contains URLs
            if self._contains_url(sentence):
                continue
            
            # Check if it's an actual recommendation
            confidence, method, verb = self._assess_recommendation(sentence)
            
            if confidence >= min_confidence:
                recommendations.append({
                    'text': sentence.strip(),
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': idx
                })
        
        # Remove duplicates and near-duplicates
        recommendations = self._remove_duplicates(recommendations)
        
        return recommendations
    
    def _contains_url(self, text: str) -> bool:
        """Check if text still contains any URL patterns"""
        for pattern in self.url_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Additional check for common URL indicators
        url_indicators = [
            r'\b(?:http|https|www|ftp|mailto)\b',
            r'\.[a-z]{2,4}/',  # Domain with path
            r'://\w',  # Protocol indicator
        ]
        
        for pattern in url_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling"""
        # Remove multiple spaces and clean up
        text = re.sub(r'\s+', ' ', text)
        
        # Handle numbered recommendations (e.g., "1. The government should...")
        text = re.sub(r'(\d+)\.\s+([A-Z])', r'\1|||SPLIT|||\2', text)
        
        # Handle lettered recommendations (e.g., "a) The department should...")
        text = re.sub(r'([a-z])\)\s+([A-Z])', r'\1)|||SPLIT|||\2', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?:\|\|\|SPLIT\|\|\|)', text)
        
        # Filter and clean
        cleaned = []
        for s in sentences:
            s = s.strip()
            # Must be substantial (not just fragments)
            if len(s) > 30 and len(s.split()) > 5:
                # Clean any leading numbers/letters
                s = re.sub(r'^[\d]+[.\)]\s*', '', s)
                s = re.sub(r'^[a-z]\)\s*', '', s)
                cleaned.append(s.strip())
        
        return cleaned
    
    def _is_false_positive(self, sentence: str) -> bool:
        """Check if sentence is a false positive"""
        for pattern in self.exclusion_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Check for excessive numbers/symbols
        non_alpha = sum(1 for c in sentence if not c.isalpha() and not c.isspace())
        if len(sentence) > 0 and non_alpha > len(sentence) * 0.4:
            return True
        
        # Check for excessive capitalisation (likely headers)
        words = sentence.split()
        if words:
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
            if caps_words > len(words) * 0.5:
                return True
        
        return False
    
    def _is_meta_recommendation(self, sentence: str) -> bool:
        """Check if sentence is talking ABOUT recommendations rather than making them"""
        sentence_lower = sentence.lower()
        
        for pattern in self.meta_recommendation_phrases:
            if re.search(pattern, sentence_lower):
                return True
        
        # Check for past tense discussion of recommendations
        if re.search(r'\b(?:was|were|has been|have been)\s+recommended\b', sentence_lower):
            return True
        
        # Check for third-person discussion
        if re.search(r'\b(?:their|his|her|its)\s+recommendations?\b', sentence_lower):
            return True
        
        return False
    
    def _assess_recommendation(self, sentence: str) -> tuple:
        """
        Assess if sentence is a recommendation and return confidence
        
        Returns:
            (confidence, method, verb)
        """
        sentence_lower = sentence.lower()
        confidence = 0.0
        method = 'none'
        verb = 'unknown'
        
        # Method 1: Direct recommendation phrases (highest confidence)
        for pattern in self.recommendation_phrases:
            if re.search(pattern, sentence_lower):
                confidence = 0.9
                method = 'direct_recommendation'
                verb = self._extract_action_verb(sentence)
                break
        
        # Method 2: Entity + should pattern (very common in government docs)
        if confidence == 0.0:
            entity_pattern = r'\b(?:the\s+)?(' + '|'.join(self.government_entities) + r')\s+should\b'
            if re.search(entity_pattern, sentence_lower):
                confidence = 0.95
                method = 'entity_should'
                verb = self._extract_action_verb(sentence)
        
        # Method 3: Starts with action verb (gerund)
        if confidence == 0.0:
            first_word = sentence.split()[0].lower() if sentence.split() else ''
            if first_word.endswith('ing') and len(first_word) > 4:
                base = first_word[:-3]
                if base in self.action_verbs:
                    confidence = 0.85
                    method = 'action_gerund'
                    verb = base
        
        # Method 4: Contains modal + action verb
        if confidence == 0.0:
            if self._contains_modal_recommendation(sentence):
                confidence = 0.75
                method = 'modal_verb'
                verb = self._extract_action_verb(sentence)
        
        # Method 5: Imperative mood (starts with verb)
        if confidence == 0.0:
            if self._is_imperative(sentence):
                confidence = 0.7
                method = 'imperative'
                verb = self._extract_first_verb(sentence)
        
        # Adjust confidence based on sentence structure
        confidence = self._adjust_confidence(sentence, confidence)
        
        return confidence, method, verb
    
    def _contains_modal_recommendation(self, sentence: str) -> bool:
        """Check if sentence contains modal verb + action"""
        sentence_lower = sentence.lower()
        
        # Look for modal + action verb pattern
        modal_patterns = [
            r'\b(?:should|must|shall|need to|ought to|needs to)\s+(?:be\s+)?(\w+)',
            r'\b(?:should|must|shall)\s+(?:also\s+)?(?:be\s+)?(\w+)',
            r'\b(?:is|are)\s+required\s+to\s+(\w+)',
        ]
        
        for pattern in modal_patterns:
            match = re.search(pattern, sentence_lower)
            if match:
                potential_verb = match.group(1)
                # Check if it's an action verb or ends with action suffix
                if (potential_verb in self.action_verbs or 
                    potential_verb.endswith('ed') or 
                    potential_verb.endswith('ing') or
                    potential_verb.endswith('ise') or
                    potential_verb.endswith('ize')):
                    return True
        
        return False
    
    def _is_imperative(self, sentence: str) -> bool:
        """Check if sentence is in imperative mood"""
        # Clean sentence
        sentence = re.sub(r'^\W+', '', sentence)
        sentence = re.sub(r'^\d+[.\)]\s*', '', sentence)  # Remove numbering
        
        if not sentence:
            return False
        
        first_word = sentence.split()[0].lower()
        
        # Check if starts with action verb
        return first_word in self.action_verbs
    
    def _extract_action_verb(self, sentence: str) -> str:
        """Extract the main action verb from sentence"""
        if NLP_AVAILABLE:
            try:
                tokens = word_tokenize(sentence)
                tagged = pos_tag(tokens)
                
                # Find verbs, prioritising action verbs
                for word, pos in tagged:
                    word_lower = word.lower()
                    if pos.startswith('VB') and word_lower in self.action_verbs:
                        return word_lower
                
                # Fallback to any verb
                for word, pos in tagged:
                    if pos.startswith('VB'):
                        word_lower = word.lower()
                        if word_lower not in {'be', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}:
                            return word_lower
            except Exception:
                pass
        
        # Fallback: find action verbs in sentence
        sentence_lower = sentence.lower()
        
        # First, try to find verb after "should"
        should_match = re.search(r'\bshould\s+(?:also\s+)?(?:be\s+)?(\w+)', sentence_lower)
        if should_match:
            potential_verb = should_match.group(1)
            if potential_verb in self.action_verbs:
                return potential_verb
        
        # Then try any action verb
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if f' {verb} ' in f' {sentence_lower} ':
                return verb
        
        return 'recommend'  # Default fallback
    
    def _extract_first_verb(self, sentence: str) -> str:
        """Extract first verb from sentence"""
        words = sentence.split()
        if not words:
            return 'unknown'
        
        first_word = words[0].lower().strip('.,;:!?"\'')
        
        # Check if it's an action verb
        if first_word in self.action_verbs:
            return first_word
        
        # If gerund, get base form
        if first_word.endswith('ing') and len(first_word) > 4:
            base = first_word[:-3]
            if base in self.action_verbs:
                return base
            # Try adding 'e' (e.g., improving -> improve)
            base_e = first_word[:-3] + 'e'
            if base_e in self.action_verbs:
                return base_e
        
        return first_word
    
    def _adjust_confidence(self, sentence: str, base_confidence: float) -> float:
        """Adjust confidence based on sentence quality indicators"""
        if base_confidence == 0.0:
            return 0.0
        
        confidence = base_confidence
        
        # Boost for formal language
        if re.search(r'\b(?:therefore|furthermore|moreover|consequently|accordingly)\b', sentence.lower()):
            confidence = min(1.0, confidence + 0.05)
        
        # Boost for numbered items
        if re.match(r'^(?:\d+[\.\)]\s+|\([a-z]\)\s+|[a-z]\)\s+)', sentence):
            confidence = min(1.0, confidence + 0.1)
        
        # Boost for known government entities
        sentence_lower = sentence.lower()
        for entity in self.government_entities:
            if entity in sentence_lower:
                confidence = min(1.0, confidence + 0.05)
                break
        
        # Reduce for questions
        if sentence.strip().endswith('?'):
            confidence *= 0.5
        
        # Reduce for very short sentences
        word_count = len(sentence.split())
        if word_count < 8:
            confidence *= 0.8
        elif word_count < 5:
            confidence *= 0.5
        
        # Reduce for sentences that are too long (might be descriptions)
        if word_count > 80:
            confidence *= 0.9
        
        return confidence
    
    def _remove_duplicates(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate and near-duplicate recommendations"""
        if not recommendations:
            return []
        
        unique = []
        seen_texts = set()
        seen_stems = set()
        
        for rec in sorted(recommendations, key=lambda x: -x['confidence']):
            text = rec['text'].lower().strip()
            
            # Create a stem (first 50 chars + last 30 chars)
            stem = text[:50] + text[-30:] if len(text) > 80 else text
            
            # Check for exact duplicates
            if text in seen_texts:
                continue
            
            # Check for near duplicates using stem
            if stem in seen_stems:
                continue
            
            # Check similarity with existing
            is_duplicate = False
            for seen in seen_texts:
                if self._similarity(text, seen) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(rec)
                seen_texts.add(text)
                seen_stems.add(stem)
        
        # Re-sort by position
        return sorted(unique, key=lambda x: x['position'])
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_statistics(self, recommendations: List[Dict]) -> Dict:
        """Get comprehensive statistics about extracted recommendations"""
        if not recommendations:
            return {
                'total': 0,
                'methods': {},
                'verbs': {},
                'avg_confidence': 0,
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }
        
        verb_counts = Counter(r['verb'] for r in recommendations)
        method_counts = Counter(r['method'] for r in recommendations)
        
        return {
            'total': len(recommendations),
            'unique_verbs': len(verb_counts),
            'top_verbs': dict(verb_counts.most_common(10)),
            'methods': dict(method_counts),
            'avg_confidence': round(sum(r['confidence'] for r in recommendations) / len(recommendations), 3),
            'high_confidence': sum(1 for r in recommendations if r['confidence'] > 0.8),
            'medium_confidence': sum(1 for r in recommendations if 0.6 <= r['confidence'] <= 0.8),
            'low_confidence': sum(1 for r in recommendations if r['confidence'] < 0.6)
        }


def extract_recommendations_improved(text: str, min_confidence: float = 0.6) -> List[Dict]:
    """
    Improved function to extract recommendations
    
    Args:
        text: Document text
        min_confidence: Minimum confidence (0-1)
        
    Returns:
        List of high-quality recommendations
    """
    extractor = ImprovedRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)


# Test function
def test_extraction():
    """Test the extractor with sample recommendations"""
    test_texts = [
        "The Home Office should provide sufficient resourcing to ensure proper oversight.",
        "The IOPC should conduct audits of complaint handling procedures.",
        "The College of Policing should review its work on training materials.",
        "Visit https://www.gov.uk/guidance for more information.",
        "The recommendations have been implemented across all departments.",
        "1. Establish a clear framework for accountability.",
        "Government should prioritise community engagement initiatives.",
        "See www.police.uk for the full report.",
        "These recommendations will improve public trust.",
    ]
    
    extractor = ImprovedRecommendationExtractor()
    
    print("=" * 70)
    print("RECOMMENDATION EXTRACTION TEST")
    print("=" * 70)
    
    for text in test_texts:
        cleaned = extractor.clean_hyperlinks(text)
        results = extractor.extract_recommendations(text, min_confidence=0.5)
        
        print(f"\nInput: {text[:60]}...")
        print(f"Cleaned: {cleaned[:60]}..." if cleaned else "Cleaned: [empty]")
        
        if results:
            for r in results:
                print(f"  ✓ EXTRACTED (confidence: {r['confidence']:.2f}, method: {r['method']}, verb: {r['verb']})")
        else:
            print("  ✗ Not extracted (filtered out)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_extraction()
