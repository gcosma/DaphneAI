"""
Advanced recommendation extractor for government documents.
Uses multiple detection methods for high accuracy.
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


class AdvancedRecommendationExtractor:
    """
    Extract recommendations from government documents using multiple methods:
    1. Numbered recommendation headers (1. Strengthen...)
    2. Explicit recommendation phrases
    3. Imperative sentences (command form)
    4. Modal verbs (should, must, ought)
    5. Gerund openings (Improving, Ensuring, etc.)
    6. Numbered/bulleted recommendation sections
    """
    
    def __init__(self):
        # Explicit recommendation indicators
        self.recommendation_indicators = {
            'high': [
                r'\brecommend(?:ation)?s?\b',
                r'\bthe (?:inquiry|committee|panel|review) recommends?\b',
                r'\bit is recommended\b',
                r'\bwe recommend\b',
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
        
        # Action verbs commonly used in recommendations - EXPANDED LIST
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
            'refine', 'conduct', 'transition', 'track', 'publish',
            'address', 'allocate', 'analyze', 'appoint', 'build',
            'collaborate', 'define', 'deliver', 'design', 'document',
            'educate', 'eliminate', 'engage', 'extend', 'identify',
            'incorporate', 'initiate', 'launch', 'measure', 'organize',
            'prepare', 'prioritize', 'produce', 'revise', 'secure',
            'train', 'transform', 'undertake', 'validate', 'verify'
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
            'reforming', 'modernising', 'streamlining', 'simplifying',
            'clarifying', 'coordinating', 'integrating', 'optimizing',
            'optimising', 'modernizing', 'consolidating', 'standardizing',
            'tracking', 'publishing', 'conducting', 'training', 'building'
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
        
        # Bullet point patterns
        self.bullet_patterns = [
            r'^\s*[•●■▪▸►]+\s+',
            r'^\s*[-–—]\s+',
            r'^\s*[*]\s+',
            r'^\s*[○◦]\s+',
            r'^\s*[✓✔]\s+',
            r'^\s*\d+[\.)]\s+',  # Numbered lists
            r'^\s*[a-z][\.)]\s+',  # Lettered lists
        ]
    
    def extract_recommendations(
        self, 
        text: str, 
        min_confidence: float = 0.5,
        context_window: int = 2
    ) -> List[Dict]:
        """
        Extract recommendations with improved context awareness.
        
        Args:
            text: Document text
            min_confidence: Minimum confidence threshold (0-1)
            context_window: Number of sentences to check for context
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # First, process line by line to catch numbered recommendations
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
            
            # PRIORITY CHECK: Is this a numbered recommendation header?
            is_numbered, verb = self._check_numbered_recommendation_improved(line)
            if is_numbered:
                recommendations.append({
                    'text': line,
                    'verb': verb,
                    'method': 'numbered_header',
                    'confidence': 0.95,
                    'position': line_idx,
                    'in_section': in_rec_section
                })
                continue
            
            # Now process as regular sentences
            sentences = self._split_into_sentences(line)
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence or len(sentence) < 10:
                    continue
                
                # Skip sentences that are ABOUT recommendations
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
                
                # Method 6: Bullet point in recommendation section
                if in_rec_section and self._is_bullet_point(sentence):
                    confidence = max(confidence, 0.75)
                    method = 'bullet_in_section' if method == 'none' else method
                
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
                        'position': line_idx * 1000 + sent_idx,
                        'in_section': in_rec_section
                    })
        
        # Remove duplicates and very similar recommendations
        recommendations = self._remove_duplicates(recommendations)
        
        # Sort by position (order in document)
        recommendations.sort(key=lambda x: x['position'])
        
        return recommendations
    
    def _check_numbered_recommendation_improved(self, line: str) -> Tuple[bool, str]:
        """
        Improved check for numbered recommendation headers.
        Returns (is_numbered_recommendation, verb)
        """
        line = line.strip()
        
        # Check various numbered patterns
        patterns = [
            r'^(\d{1,2})\.?\s+([A-Z][a-z]+)\s+(.+)',  # "1. Strengthen..."
            r'^(\d{1,2})\)\s+([A-Z][a-z]+)\s+(.+)',    # "1) Strengthen..."
            r'^([a-z])\.\s+([A-Z][a-z]+)\s+(.+)',      # "a. Strengthen..."
            r'^([a-z])\)\s+([A-Z][a-z]+)\s+(.+)',      # "a) Strengthen..."
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                first_word_after_number = match.group(2).lower()
                
                # Check if it's an action verb
                if first_word_after_number in self.action_verbs:
                    return True, first_word_after_number
                
                # Check second word if first is not a verb
                words = line.split()
                if len(words) > 2:
                    # Get the word after the number and first word
                    second_word = words[2].lower().strip('.,;:')
                    if second_word in self.action_verbs:
                        return True, second_word
        
        return False, 'unknown'
    
    def _is_about_recommendations(self, text: str) -> bool:
        """Check if text is ABOUT recommendations rather than making them."""
        text_lower = text.lower()
        
        # Patterns that indicate meta-text about recommendations
        exclusion_patterns = [
            r'\breports? contain.*recommendations?\b',
            r'\bfindings.*and.*recommendations?\b',
            r'\brecommendations? (?:are|can be|will be) (?:found|developed|published|available)\b',
            r'\brecommendations? include\b',
            r'\bthe recommendations?\b.*\b(?:report|document|section)\b',
            r'\bmonitoring.*recommendations?\b',
            r'\bimplementation of.*recommendations?\b',
            r'\bacting on.*recommendations?\b',
            r'\bfollowing.*recommendations?\b',
            r'^this document (?:outlines|contains)',
            r'^operational improvement recommendations report\b',
            r'\bwhile some recommendations\b',
            r'\bset of recommendations intended\b',
            r'\boutlines.*recommendations?\b',
            r'\bcontains.*recommendations?\b',
            r'\bthis document.*recommendations?\b',
            r'\ba recommendation\b',
            r'\bthe goal is to provide\b',
            r'\bthe insights below\b',
            r'\bbased on observations\b',
        ]
        
        for pattern in exclusion_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling for section headers."""
        # First, protect abbreviations and numbers
        text = re.sub(r'(\d+)\.(\d+)', r'\1<<DOT>>\2', text)
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof)\.', r'\1<<DOT>>', text)
        
        # Fix cases where period is directly followed by capital letter (no space)
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<<DOT>>', '.').strip() for s in sentences]
        
        return [s for s in sentences if s and len(s.strip()) > 10]
    
    def _is_recommendation_section_header(self, text: str) -> bool:
        """Check if text is a recommendation section header."""
        text_lower = text.lower().strip()
        
        for pattern in self.recommendation_section_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def _is_new_section_header(self, text: str) -> bool:
        """Check if text is a new section header (leaving recommendations)."""
        # Common section headers that aren't recommendations
        non_rec_headers = [
            r'^introduction:?\s*$',
            r'^background:?\s*$',
            r'^methodology:?\s*$',
            r'^findings?:?\s*$',
            r'^conclusion:?\s*$',
            r'^response:?\s*$',
            r'^implementation:?\s*$',
            r'^appendix:?\s*$',
            r'^\d+\.?\s+\w+:?\s*$',  # Numbered headers
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in non_rec_headers:
            if re.match(pattern, text_lower):
                return True
        
        return False

    def _check_numbered_recommendation(self, sentence: str) -> Tuple[float, str]:
        """Check for numbered recommendation headers like '1. Strengthen...'"""
        # This method is kept for backward compatibility but improved version is used
        return self._check_numbered_recommendation_improved(sentence)

    def _check_explicit_recommendations(self, sentence: str) -> Tuple[float, str]:
        """Check for explicit recommendation phrases."""
        sentence_lower = sentence.lower()
        
        # First check if this is ABOUT recommendations (exclude)
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
        """Stricter gerund rule: only accept gerunds that indicate actions."""
        words = sentence.split()
        if not words:
            return 0.0
    
        first = words[0].strip('.,;:!?"\'').lower()
    
        # Only accept gerunds in approved set
        if first in self.recommendation_gerunds:
            return 0.85
    
        return 0.0

    def _check_gerund_openingold(self, sentence: str) -> float:
        """Check if sentence starts with a recommendation gerund."""
        words = sentence.split()
        if not words:
            return 0.0
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        if first_word in self.recommendation_gerunds:
            return 0.85
        
        # Check if it ends with 'ing' and might be a verb
        if first_word.endswith('ing') and len(first_word) > 5:
            if NLP_AVAILABLE:
                try:
                    base = first_word[:-3]
                    if base in self.action_verbs:
                        return 0.8
                except:
                    pass
            return 0.5
        
        return 0.0
    
    def _check_imperative_form(self, sentence: str) -> float:
        """Check if sentence is in imperative form (command)."""
        words = sentence.split()
        if not words:
            return 0.0
        
        # Handle numbered lists: "1. Strengthen..." 
        first_word = words[0].strip('.,;:!?"\'').lower()
        if first_word.isdigit() and len(words) > 1:
            first_word = words[1].strip('.,;:!?"\'').lower()
        
        # Check if first word is an action verb (base form)
        if first_word in self.action_verbs:
            # Make sure it's not a gerund
            if not first_word.endswith('ing'):
                return 0.7
        
        return 0.0

    def _check_modal_action(self, sentence: str) -> float:
        """
        Improved modal detection:
        Extracts the FIRST verb immediately following a modal verb.
        """
        text = sentence.lower()
    
        # Order matters: strongest modals first
        modal_patterns = [
            (r'\bshould\s+([a-z]+)', 0.7),
            (r'\bmust\s+([a-z]+)', 0.8),
            (r'\bneed(?:s)?\s+to\s+([a-z]+)', 0.7),
            (r'\bought\s+to\s+([a-z]+)', 0.75)
        ]
    
        for pattern, conf in modal_patterns:
            m = re.search(pattern, text)
            if not m:
                continue
    
            verb = m.group(1)
    
            # Only accept if it's an action verb
            if verb in self.action_verbs:
                return conf
    
            # If verb ends in 'ing' or 'ed', skip (not the main action)
            if verb.endswith(("ing", "ed")):
                continue
    
        return 0.0

    
    def _check_modal_actionold(self, sentence: str) -> float:
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
                verb = match.group(1)
                if verb in self.action_verbs:
                    return confidence
        
        return 0.0
    
    def _is_bullet_point(self, sentence: str) -> bool:
        """Check if sentence starts with a bullet point."""
        for pattern in self.bullet_patterns:
            if re.match(pattern, sentence):
                return True
        return False
    
    def _extract_main_verb(self, sentence: str) -> str:
        """Extract the main action verb from a sentence."""
        words = sentence.split()
        if not words:
            return 'unknown'
        
        sentence_lower = sentence.lower()
        
        # Check first word (for gerunds and imperatives)
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        # Handle numbered lists
        if first_word.isdigit() and len(words) > 1:
            first_word = words[1].strip('.,;:!?"\'').lower()
        
        if first_word in self.action_verbs:
            return first_word
        
        if first_word.endswith('ing') and len(first_word) > 5:
            base = first_word[:-3]
            if base in self.action_verbs:
                return base
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
                              'have', 'has', 'had', 'do', 'does', 'did'}
                
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
            
            # Remove bullet points for comparison
            for pattern in self.bullet_patterns:
                text = re.sub(pattern, '', text)
            
            # Remove numbering for comparison
            text = re.sub(r'^\d+\.\s*', '', text)
            
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
    extractor = AdvancedRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)
