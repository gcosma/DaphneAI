"""
Simple Recommendation Extractor - Use spaCy to find all verbs automatically
"""

import re
from typing import List, Dict
from collections import Counter

# Try to import spaCy
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


class SimpleRecommendationExtractor:
    """Extract recommendations by finding sentences with verbs that suggest actions"""
    
    def __init__(self):
        """Initialise the extractor"""
        # Common recommendation phrase patterns
        self.recommendation_patterns = [
            r'\brecommend',
            r'\bsuggest',
            r'\badvise',
            r'\bpropose',
            r'\burge',
            r'\bshould\b',
            r'\bmust\b',
            r'\bneed to\b',
            r'\brequire',
        ]
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.5) -> List[Dict]:
        """
        Extract recommendations from text using verb analysis
        
        Args:
            text: Document text
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        sentences = self._split_sentences(text)
        
        for idx, sentence in enumerate(sentences):
            # Method 1: Look for recommendation keywords
            if self._contains_recommendation_phrase(sentence):
                verb = self._extract_main_verb(sentence)
                recommendations.append({
                    'text': sentence,
                    'verb': verb,
                    'method': 'keyword',
                    'confidence': 0.85,
                    'position': idx
                })
            
            # Method 2: Look for sentences starting with verbs in -ing form (gerunds)
            elif self._starts_with_gerund(sentence):
                verb = self._extract_first_verb(sentence)
                recommendations.append({
                    'text': sentence,
                    'verb': verb,
                    'method': 'gerund',
                    'confidence': 0.9,
                    'position': idx
                })
            
            # Method 3: Look for modal verbs (should, must, etc.)
            elif self._contains_modal(sentence):
                verb = self._extract_main_verb(sentence)
                recommendations.append({
                    'text': sentence,
                    'verb': verb,
                    'method': 'modal',
                    'confidence': 0.7,
                    'position': idx
                })
        
        # Filter by confidence
        recommendations = [r for r in recommendations if r['confidence'] >= min_confidence]
        
        # Remove duplicates
        recommendations = self._remove_duplicates(recommendations)
        
        return recommendations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    def _contains_recommendation_phrase(self, sentence: str) -> bool:
        """Check if sentence contains recommendation keywords"""
        sentence_lower = sentence.lower()
        return any(re.search(pattern, sentence_lower) for pattern in self.recommendation_patterns)
    
    def _starts_with_gerund(self, sentence: str) -> bool:
        """Check if sentence starts with a gerund (verb-ing)"""
        # Get first word
        words = sentence.split()
        if not words:
            return False
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        # Check if it ends with -ing and looks like a verb
        if first_word.endswith('ing') and len(first_word) > 4:
            # Common gerunds that indicate recommendations
            common_gerunds = [
                'improving', 'ensuring', 'establishing', 'enabling', 'broadening',
                'reforming', 'implementing', 'developing', 'creating', 'enhancing',
                'introducing', 'reviewing', 'updating', 'providing', 'supporting',
                'maintaining', 'expanding', 'reducing', 'addressing', 'promoting'
            ]
            return first_word in common_gerunds or self._is_verb_nltk(first_word[:-3])  # Remove 'ing'
        
        return False
    
    def _contains_modal(self, sentence: str) -> bool:
        """Check if sentence contains modal verbs"""
        modals = ['should', 'must', 'ought to', 'need to', 'shall']
        sentence_lower = sentence.lower()
        return any(f' {modal} ' in f' {sentence_lower} ' for modal in modals)
    
    def _extract_first_verb(self, sentence: str) -> str:
        """Extract the first verb from a sentence"""
        words = sentence.split()
        if not words:
            return 'unknown'
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        # If it ends with -ing, remove the -ing to get base form
        if first_word.endswith('ing'):
            return first_word[:-3] if len(first_word) > 4 else first_word
        
        return first_word
    
    def _extract_main_verb(self, sentence: str) -> str:
        """Extract the main verb from a sentence using NLP"""
        if NLP_AVAILABLE:
            try:
                # Tokenize and tag
                tokens = word_tokenize(sentence)
                tagged = pos_tag(tokens)
                
                # Find verbs (VB = verb, VBG = gerund, VBN = past participle, etc.)
                verbs = [word.lower() for word, pos in tagged if pos.startswith('VB')]
                
                if verbs:
                    # Return first meaningful verb (skip auxiliaries)
                    auxiliaries = {'be', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}
                    for verb in verbs:
                        if verb not in auxiliaries:
                            return verb
                    return verbs[0]  # Return first verb if all are auxiliaries
            except:
                pass
        
        # Fallback: look for common recommendation verbs
        sentence_lower = sentence.lower()
        common_verbs = [
            'recommend', 'suggest', 'advise', 'propose', 'improve', 'ensure',
            'establish', 'enable', 'broaden', 'reform', 'implement', 'develop'
        ]
        
        for verb in common_verbs:
            if verb in sentence_lower:
                return verb
        
        return 'unknown'
    
    def _is_verb_nltk(self, word: str) -> bool:
        """Check if a word is a verb using NLTK"""
        if not NLP_AVAILABLE:
            return True  # Assume it's a verb if we can't check
        
        try:
            tagged = pos_tag([word])
            return tagged[0][1].startswith('VB')
        except:
            return True
    
    def _remove_duplicates(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations based on text similarity"""
        if not recommendations:
            return []
        
        unique = []
        seen_texts = []
        
        for rec in recommendations:
            text = rec['text'].lower().strip()
            
            # Check if this is similar to any seen text
            is_duplicate = False
            for seen in seen_texts:
                if self._similarity(text, seen) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(rec)
                seen_texts.append(text)
        
        return unique
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_verb_statistics(self, recommendations: List[Dict]) -> Dict:
        """Get statistics about the verbs used"""
        verb_counts = Counter(r['verb'] for r in recommendations)
        method_counts = Counter(r['method'] for r in recommendations)
        
        return {
            'total': len(recommendations),
            'unique_verbs': len(verb_counts),
            'verb_frequency': dict(verb_counts.most_common()),
            'method_distribution': dict(method_counts),
            'avg_confidence': sum(r['confidence'] for r in recommendations) / len(recommendations) if recommendations else 0
        }


def extract_recommendations_simple(text: str, min_confidence: float = 0.7) -> List[Dict]:
    """
    Simple function to extract recommendations from text
    
    Args:
        text: Document text
        min_confidence: Minimum confidence (0-1)
        
    Returns:
        List of recommendations
    """
    extractor = SimpleRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)
