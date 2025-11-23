"""
Improved recommendation extractor for government documents.
Fixes issues with numbered recommendations and false positives.
"""

import re
from typing import Dict, List, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

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
    logger.warning("NLTK not available - using pattern-based extraction only")


class ImprovedRecommendationExtractor:
    """
    Extract recommendations from government documents with improved accuracy.
    Key improvements:
    - Better detection of numbered recommendations
    - Stronger filtering of false positives
    - Improved sentence splitting for numbered lists
    """
    
    def __init__(self):
        # Explicit recommendation indicators
        self.recommendation_indicators = {
            'high': [
                r'\bthe (?:inquiry|committee|panel|review) recommends?\b',
                r'\bit is recommended that\b',
                r'\bwe recommend that\b',
                r'\bour recommendation is\b',
            ],
            'medium': [
                r'\bshould\s+(?:be\s+)?(?:implemented?|established?|introduced?|created?|developed?|considered?)\b',
                r'\bmust\s+(?:be\s+)?(?:implemented?|established?|introduced?|created?|developed?)\b',
                r'\boughts?\s+to\s+(?:be\s+)?(?:implemented?|established?|introduced?)\b',
                r'\bneed(?:s)?\s+to\s+(?:be\s+)?(?:implemented?|established?|introduced?|created?|developed?)\b',
            ],
            'low': [
                r'\badvis(?:e|ing)\b',
                r'\bsuggest(?:s|ing)?\b',
                r'\bpropos(?:e|ing|al)s?\b',
                r'\burge(?:s|d)?\b',
            ]
        }
        
        # Action verbs commonly used in recommendations
        self.action_verbs = {
            'implement', 'establish', 'create', 'develop', 'introduce',
            'ensure', 'enable', 'improve', 'enhance', 'strengthen',
            'broaden', 'expand', 'increase', 'reduce', 'minimise',
            'provide', 'support', 'maintain', 'review', 'update',
            'adopt', 'require', 'mandate', 'enforce', 'monitor',
            'assess', 'evaluate', 'consider', 'explore', 'investigate',
            'reform', 'modernise', 'streamline', 'simplify', 'clarify',
            'promote', 'encourage', 'facilitate', 'coordinate', 'integrate',
            'optimize', 'optimise', 'modernize', 'consolidate', 'standardize',
            'refine', 'conduct', 'transition', 'track', 'publish'
        }
        
        # Gerunds (verb+ing forms) that start recommendations
        self.recommendation_gerunds = {
            'improving', 'ensuring', 'establishing', 'enabling', 'broadening',
            'reforming', 'implementing', 'developing', 'creating', 'enhancing',
            'introducing', 'reviewing', 'updating', 'providing', 'supporting',
            'maintaining', 'expanding', 'reducing', 'addressing', 'promoting',
            'strengthening', 'facilitating', 'encouraging', 'adopting',
            'requiring', 'mandating', 'enforcing', 'monitoring', 'assessing',
            'evaluating', 'considering', 'exploring', 'investigating',
            'modernising', 'streamlining', 'simplifying', 'consolidating',
            'clarifying', 'coordinating', 'integrating', 'optimizing',
            'optimising', 'modernizing', 'standardizing', 'tracking',
            'conducting', 'refining', 'publishing'
        }
        
        # Section headers that indicate recommendations
        self.recommendation_section_patterns = [
            r'^recommendations?:?\s*$',
            r'^key recommendations?:?\s*$',
            r'^main recommendations?:?\s*$',
            r'^summary of recommendations?:?\s*$',
            r'^\d+\.?\s*recommendations?:?\s*$',
            r'^specific recommendations?:?\s*$',
        ]
        
        # Patterns that indicate text ABOUT recommendations, not actual recommendations
        self.exclusion_patterns = [
            r'\breports? contain.*recommendations?\b',
            r'\bfindings.*and.*recommendations?\b',
            r'\brecommendations? (?:are|can be|will be) (?:found|developed|published|available)\b',
            r'\brecommendations? include\b',
            r'\bthe recommendations?\b.*\b(?:report|document|section)\b',
            r'\bmonitoring.*recommendations?\b',
            r'\bimplementation of.*recommendations?\b',
            r'\bacting on.*recommendations?\b',
            r'\bfollowing.*recommendations?\b',
            r'\bthese recommendations?\b',
            r'^this document outlines\b',
            r'^this document contains\b',
            r'\bdocument outlines a set of recommendations\b',
            r'^operational improvement recommendations report\b',
            r'\bwhile some recommendations\b',
            r'\bset of recommendations intended\b',
            r'\boutlines.*recommendations?\b',
            r'\bcontains.*recommendations?\b',
            r'\ba recommendation\b',
            r'\bthe goal is to provide\b',
            r'\bthe insights below\b',
            r'\bbased on observations\b',
        ]
    
    def extract_recommendations(
        self, 
        text: str, 
        min_confidence: float = 0.5,
        context_window: int = 2
    ) -> List[Dict]:
        """
        Extract recommendations with improved accuracy.
        
        Args:
            text: Document text
            min_confidence: Minimum confidence threshold (0-1)
            context_window: Number of sentences to check for context
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Split text into lines for better numbered list handling
        lines = text.split('\n')
        
        # Track if we're in a recommendation section
        in_rec_section = False
        section_confidence_boost = 0.0
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            # Check if this is a section header
            if self._is_recommendation_section_header(line):
                in_rec_section = True
                section_confidence_boost = 0.15
                continue
            
            # Check if we've left the recommendation section
            if in_rec_section and self._is_new_section_header(line):
                in_rec_section = False
                section_confidence_boost = 0.0
            
            # PRIORITY 1: Check for numbered recommendations (e.g., "1. Strengthen...")
            is_numbered, verb = self._check_numbered_recommendation_line(line)
            if is_numbered:
                # This is definitely a recommendation header
                recommendations.append({
                    'text': line,
                    'verb': verb,
                    'method': 'numbered_header',
                    'confidence': 0.95,
                    'position': line_idx,
                    'in_section': in_rec_section
                })
                continue
            
            # Split line into sentences for other checks
            sentences = self._split_line_into_sentences(line)
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence or len(sentence) < 10:
                    continue
                
                # Skip if this is text ABOUT recommendations
                if self._is_about_recommendations(sentence):
                    continue
                
                confidence = 0.0
                method = 'none'
                
                # Method 2: Explicit recommendation phrases
                explicit_conf, explicit_method = self._check_explicit_recommendations(sentence)
                if explicit_conf > confidence:
                    confidence = explicit_conf
                    method = explicit_method
                
                # Method 3: Starts with gerund
                if confidence < 0.7:
                    gerund_conf = self._check_gerund_opening(sentence)
                    if gerund_conf > confidence:
                        confidence = gerund_conf
                        method = 'gerund'
                
                # Method 4: Imperative sentence (command form)
                if confidence < 0.7:
                    imperative_conf = self._check_imperative_form(sentence)
                    if imperative_conf > confidence:
                        confidence = imperative_conf
                        method = 'imperative'
                
                # Method 5: Modal verbs with action
                if confidence < 0.7:
                    modal_conf = self._check_modal_action(sentence)
                    if modal_conf > confidence:
                        confidence = modal_conf
                        method = 'modal'
                
                # Boost confidence if in recommendation section
                confidence += section_confidence_boost
                confidence = min(confidence, 1.0)
                
                # Extract main verb
                verb = self._extract_main_verb(sentence)
                
                # Only add if confidence is sufficient
                if confidence >= min_confidence:
                    recommendations.append({
                        'text': sentence,
                        'verb': verb,
                        'method': method,
                        'confidence': round(confidence, 2),
                        'position': line_idx * 1000 + sent_idx,  # Preserve order
                        'in_section': in_rec_section
                    })
        
        # Remove duplicates
        recommendations = self._remove_duplicates(recommendations)
        
        # Sort by position (order in document)
        recommendations.sort(key=lambda x: x['position'])
        
        return recommendations
    
    def _check_numbered_recommendation_line(self, line: str) -> Tuple[bool, str]:
        """
        Check if a line is a numbered recommendation header.
        Returns (is_numbered_rec, verb)
        """
        line = line.strip()
        
        # Pattern for numbered recommendations: "1. Action Verb..."
        match = re.match(r'^(\d+)\.?\s+([A-Z][a-z]+)\s+(.+)', line)
        if match:
            first_word = match.group(2).lower()
            
            # Check if first word after number is an action verb
            if first_word in self.action_verbs:
                return True, first_word
            
            # Check second word if first is an article/adjective
            words = line.split()
            if len(words) > 2:
                second_word = words[2].lower().strip('.,;:')
                if second_word in self.action_verbs:
                    return True, second_word
        
        return False, 'unknown'
    
    def _is_about_recommendations(self, text: str) -> bool:
        """Check if text is ABOUT recommendations rather than making them."""
        text_lower = text.lower()
        
        for pattern in self.exclusion_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Additional checks for common false positives
        if text_lower.startswith('operational improvement recommendations'):
            return True
        if 'this document' in text_lower and 'recommendations' in text_lower:
            return True
        if 'set of recommendations' in text_lower:
            return True
            
        return False
    
    def _split_line_into_sentences(self, text: str) -> List[str]:
        """Split a line into sentences."""
        # Protect abbreviations
        text = re.sub(r'(\d+)\.(\d+)', r'\1<<DOT>>\2', text)
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp)\.', r'\1<<DOT>>', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<<DOT>>', '.').strip() for s in sentences]
        
        return [s for s in sentences if len(s.strip()) > 10]
    
    def _is_recommendation_section_header(self, text: str) -> bool:
        """Check if text is a recommendation section header."""
        text_lower = text.lower().strip()
        
        for pattern in self.recommendation_section_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def _is_new_section_header(self, text: str) -> bool:
        """Check if text is a new section header (leaving recommendations)."""
        non_rec_headers = [
            r'^introduction:?\s*$',
            r'^background:?\s*$',
            r'^methodology:?\s*$',
            r'^findings?:?\s*$',
            r'^conclusion:?\s*$',
            r'^response:?\s*$',
            r'^implementation:?\s*$',
            r'^appendix:?\s*$',
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in non_rec_headers:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def _check_explicit_recommendations(self, sentence: str) -> Tuple[float, str]:
        """Check for explicit recommendation phrases."""
        sentence_lower = sentence.lower()
        
        # First check if this is ABOUT recommendations
        if self._is_about_recommendations(sentence):
            return 0.0, 'none'
        
        # High confidence patterns
        for pattern in self.recommendation_indicators['high']:
            if re.search(pattern, sentence_lower):
                return 0.9, 'explicit_high'
        
        # Medium confidence patterns
        for pattern in self.recommendation_indicators['medium']:
            if re.search(pattern, sentence_lower):
                return 0.75, 'explicit_medium'
        
        # Low confidence patterns
        for pattern in self.recommendation_indicators['low']:
            if re.search(pattern, sentence_lower):
                return 0.6, 'explicit_low'
        
        return 0.0, 'none'
    
    def _check_gerund_opening(self, sentence: str) -> float:
        """Check if sentence starts with a recommendation gerund."""
        words = sentence.split()
        if not words:
            return 0.0
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        if first_word in self.recommendation_gerunds:
            return 0.85
        
        return 0.0
    
    def _check_imperative_form(self, sentence: str) -> float:
        """Check if sentence is in imperative form (command)."""
        words = sentence.split()
        if not words:
            return 0.0
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        # Check if first word is an action verb (base form)
        if first_word in self.action_verbs:
            # Make sure it's not a gerund
            if not first_word.endswith('ing'):
                return 0.7
        
        return 0.0
    
    def _check_modal_action(self, sentence: str) -> float:
        """Check for modal verbs followed by action verbs."""
        sentence_lower = sentence.lower()
        
        # Strong modals
        strong_modals = [
            (r'\bshould\s+(?:be\s+)?(\w+)', 0.7),
            (r'\bmust\s+(?:be\s+)?(\w+)', 0.8),
            (r'\boughts?\s+to\s+(?:be\s+)?(\w+)', 0.75),
            (r'\bneed(?:s)?\s+to\s+(?:be\s+)?(\w+)', 0.7),
        ]
        
        for pattern, confidence in strong_modals:
            match = re.search(pattern, sentence_lower)
            if match:
                verb = match.group(1).strip('.,;:')
                # Clean verb extraction
                verb = verb.lower()
                if verb in self.action_verbs or verb.endswith('ed') or verb.endswith('ing'):
                    return confidence
        
        return 0.0
    
    def _extract_main_verb(self, sentence: str) -> str:
        """Extract the main action verb from a sentence."""
        words = sentence.split()
        if not words:
            return 'unknown'
        
        sentence_lower = sentence.lower()
        
        # Check first word (for imperatives and gerunds)
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        # Handle numbered lists
        if first_word.isdigit() and len(words) > 1:
            first_word = words[1].strip('.,;:!?"\'').lower()
        
        if first_word in self.action_verbs:
            return first_word
        
        if first_word.endswith('ing'):
            # Try to get base form
            base = first_word[:-3]
            if base in self.action_verbs:
                return base
            # Still return the gerund form if it's in our list
            if first_word in self.recommendation_gerunds:
                return first_word
        
        # Use NLTK if available
        if NLP_AVAILABLE:
            try:
                tokens = word_tokenize(sentence[:200])
                tagged = pos_tag(tokens)
                
                # Find action verbs
                verbs = [word.lower() for word, pos in tagged if pos.startswith('VB')]
                
                # Filter out auxiliaries
                auxiliaries = {'be', 'is', 'are', 'was', 'were', 'been', 'being', 
                              'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                              'can', 'could', 'may', 'might', 'shall', 'should', 'ought'}
                
                for verb in verbs:
                    if verb in self.action_verbs:
                        return verb
                    if verb not in auxiliaries:
                        return verb
                
                if verbs:
                    return verbs[0]
            except:
                pass
        
        # Fallback: search for any action verb in sentence
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(r'\b' + verb + r'\b', sentence_lower):
                return verb
        
        return 'unknown'
    
    def _remove_duplicates(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate or very similar recommendations."""
        if not recommendations:
            return []
        
        unique = []
        seen_texts = []
        
        for rec in recommendations:
            text = rec['text'].lower().strip()
            
            # Normalize for comparison
            text = re.sub(r'^\d+\.\s*', '', text)  # Remove numbering
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            is_duplicate = False
            for seen in seen_texts:
                if self._similarity(text, seen) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(rec)
                seen_texts.append(text)
        
        return unique
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_statistics(self, recommendations: List[Dict]) -> Dict:
        """Get detailed statistics about extracted recommendations."""
        if not recommendations:
            return {
                'total': 0,
                'unique_verbs': 0,
                'verb_frequency': {},
                'method_distribution': {},
                'avg_confidence': 0.0,
                'in_section_count': 0
            }
        
        verb_counts = Counter(r['verb'] for r in recommendations)
        method_counts = Counter(r['method'] for r in recommendations)
        in_section_count = sum(1 for r in recommendations if r.get('in_section', False))
        
        return {
            'total': len(recommendations),
            'unique_verbs': len(verb_counts),
            'verb_frequency': dict(verb_counts.most_common(10)),
            'method_distribution': dict(method_counts),
            'avg_confidence': sum(r['confidence'] for r in recommendations) / len(recommendations),
            'in_section_count': in_section_count,
            'confidence_range': {
                'min': min(r['confidence'] for r in recommendations),
                'max': max(r['confidence'] for r in recommendations)
            }
        }


def extract_recommendations(text: str, min_confidence: float = 0.7) -> List[Dict]:
    """
    Convenience function to extract recommendations.
    
    Args:
        text: Document text
        min_confidence: Minimum confidence threshold (0-1)
        
    Returns:
        List of recommendation dictionaries
    """
    extractor = ImprovedRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)


# Test the extractor with the provided document
if __name__ == "__main__":
    # Read the test document
    with open("Test.txt", "r", encoding="utf-8") as f:
        test_text = f.read()
    
    # Extract recommendations
    extractor = ImprovedRecommendationExtractor()
    recommendations = extractor.extract_recommendations(test_text, min_confidence=0.7)
    
    # Print results
    print(f"\nFound {len(recommendations)} recommendations:\n")
    print("-" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['method']}] (confidence: {rec['confidence']})")
        print(f"   Verb: {rec['verb']}")
        print(f"   Text: {rec['text'][:100]}...")
        print()
    
    # Print statistics
    stats = extractor.get_statistics(recommendations)
    print("\nStatistics:")
    print("-" * 40)
    print(f"Total recommendations: {stats['total']}")
    print(f"Unique verbs: {stats['unique_verbs']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print(f"Method distribution: {stats['method_distribution']}")
    print(f"Top verbs: {stats['verb_frequency']}")
