"""
Production-ready recommendation extractor for Streamlit app.
Optimized for government documents with robust error handling.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import NLTK but don't fail if unavailable
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    # Download required NLTK data quietly
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    NLP_AVAILABLE = True
    logger.info("NLTK loaded successfully")
except ImportError:
    NLP_AVAILABLE = False
    logger.info("NLTK not available - using pattern-based extraction only")


class RecommendationExtractor:
    """
    Production-ready extractor for government document recommendations.
    Optimized for Streamlit integration with progress tracking support.
    """
    
    def __init__(self, progress_callback=None):
        """
        Initialize the extractor.
        
        Args:
            progress_callback: Optional callback for progress updates in Streamlit
        """
        self.progress_callback = progress_callback
        
        # Action verbs commonly used in recommendations
        self.action_verbs = {
            'implement', 'establish', 'create', 'develop', 'introduce',
            'ensure', 'enable', 'improve', 'enhance', 'strengthen',
            'broaden', 'expand', 'increase', 'reduce', 'minimize',
            'provide', 'support', 'maintain', 'review', 'update',
            'adopt', 'require', 'mandate', 'enforce', 'monitor',
            'assess', 'evaluate', 'consider', 'explore', 'investigate',
            'reform', 'modernise', 'modernize', 'streamline', 'simplify',
            'clarify', 'promote', 'encourage', 'facilitate', 'coordinate',
            'integrate', 'optimize', 'optimise', 'consolidate', 'standardize',
            'standardise', 'refine', 'conduct', 'transition', 'track', 'publish',
            'address', 'allocate', 'analyze', 'analyse', 'appoint', 'approve',
            'build', 'collaborate', 'communicate', 'compile', 'define',
            'deliver', 'demonstrate', 'design', 'determine', 'document',
            'educate', 'eliminate', 'engage', 'examine', 'extend', 'focus',
            'formulate', 'fund', 'identify', 'incorporate', 'initiate',
            'issue', 'launch', 'lead', 'measure', 'notify', 'obtain',
            'offer', 'organize', 'organise', 'outline', 'oversee', 'prepare',
            'prioritize', 'prioritise', 'produce', 'propose', 'recruit',
            'redesign', 'reinforce', 'remove', 'replace', 'report', 'restructure',
            'retain', 'revise', 'secure', 'seek', 'set', 'share', 'specify',
            'submit', 'train', 'transform', 'undertake', 'validate', 'verify'
        }
        
        # Gerunds that often start recommendations
        self.recommendation_gerunds = {word + 'ing' for word in self.action_verbs 
                                      if not word.endswith('e')}
        self.recommendation_gerunds.update({
            word[:-1] + 'ing' for word in self.action_verbs 
            if word.endswith('e') and word != 'be'
        })
        
        # Patterns that indicate text ABOUT recommendations, not actual recommendations
        self.exclusion_patterns = [
            r'\breports? contain.*recommendations?\b',
            r'\bfindings.*and.*recommendations?\b',
            r'\brecommendations? (?:are|can be|will be|were|have been)',
            r'\brecommendations? include\b',
            r'\bthe recommendations?\b.*\b(?:report|document|section)\b',
            r'\bmonitoring.*recommendations?\b',
            r'\bimplementation of.*recommendations?\b',
            r'^this document (?:outlines|contains|presents)',
            r'^operational improvement recommendations report\b',
            r'\bwhile some recommendations\b',
            r'\bset of recommendations (?:intended|designed|aimed)',
            r'\b(?:outlines|contains|presents) (?:a set of )?recommendations?\b',
            r'\bthe (?:goal|purpose|aim) is to provide\b',
            r'\bthe insights below\b',
            r'\bbased on observations\b',
            r'\bdocument.*recommendations\b',
        ]
        
        # Modal verb patterns
        self.modal_patterns = [
            (r'\bshould\s+(?:be\s+)?(\w+)', 0.75),
            (r'\bmust\s+(?:be\s+)?(\w+)', 0.85),
            (r'\bwill\s+need\s+to\s+(?:be\s+)?(\w+)', 0.8),
            (r'\bneed(?:s)?\s+to\s+(?:be\s+)?(\w+)', 0.75),
            (r'\bough?t\s+to\s+(?:be\s+)?(\w+)', 0.7),
            (r'\bis\s+(?:critical|essential|important)\s+(?:to|that)\s+(\w+)', 0.8),
            (r'\bit\s+is\s+(?:recommended|advised)\s+(?:to|that)\s+(\w+)', 0.9),
        ]
    
    def extract_recommendations(
        self, 
        text: str, 
        min_confidence: float = 0.7,
        return_stats: bool = False
    ) -> List[Dict]:
        """
        Extract recommendations from document text.
        
        Args:
            text: Document text to analyze
            min_confidence: Minimum confidence threshold (0-1)
            return_stats: Whether to return statistics with results
            
        Returns:
            List of recommendation dictionaries, or tuple with stats if requested
        """
        if not text:
            return [] if not return_stats else ([], {})
        
        try:
            recommendations = self._extract_recommendations_internal(text, min_confidence)
            
            if return_stats:
                stats = self.get_statistics(recommendations)
                return recommendations, stats
            return recommendations
            
        except Exception as e:
            logger.error(f"Error extracting recommendations: {e}")
            return [] if not return_stats else ([], {})
    
    def _extract_recommendations_internal(
        self, 
        text: str, 
        min_confidence: float
    ) -> List[Dict]:
        """Internal method for extracting recommendations."""
        recommendations = []
        
        # Update progress if callback provided
        if self.progress_callback:
            self.progress_callback(0.1, "Preprocessing text...")
        
        # Normalize text
        text = self._normalize_text(text)
        lines = text.split('\n')
        
        # Track context
        in_rec_section = False
        section_confidence_boost = 0.0
        
        total_lines = len(lines)
        
        for line_idx, line in enumerate(lines):
            # Update progress
            if self.progress_callback and line_idx % 10 == 0:
                progress = 0.1 + (0.7 * line_idx / total_lines)
                self.progress_callback(progress, f"Analyzing line {line_idx}/{total_lines}")
            
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            # Check for section headers
            if self._is_recommendation_section_header(line):
                in_rec_section = True
                section_confidence_boost = 0.15
                continue
            
            if in_rec_section and self._is_new_section_header(line):
                in_rec_section = False
                section_confidence_boost = 0.0
            
            # Check for numbered recommendation
            is_numbered, verb = self._check_numbered_recommendation(line)
            if is_numbered:
                recommendations.append({
                    'text': line,
                    'verb': verb,
                    'method': 'numbered_header',
                    'confidence': min(0.95 + section_confidence_boost, 1.0),
                    'position': line_idx,
                    'in_section': in_rec_section,
                    'line_number': line_idx + 1
                })
                continue
            
            # Process sentences within the line
            sentences = self._split_into_sentences(line)
            for sent_idx, sentence in enumerate(sentences):
                if not sentence or len(sentence) < 15:
                    continue
                
                # Skip meta-text about recommendations
                if self._is_about_recommendations(sentence):
                    continue
                
                # Evaluate recommendation confidence
                confidence, method, verb = self._evaluate_sentence(sentence)
                
                # Apply section boost
                if in_rec_section:
                    confidence = min(confidence + section_confidence_boost, 1.0)
                
                # Add if meets threshold
                if confidence >= min_confidence:
                    recommendations.append({
                        'text': sentence,
                        'verb': verb,
                        'method': method,
                        'confidence': round(confidence, 2),
                        'position': line_idx * 1000 + sent_idx,
                        'in_section': in_rec_section,
                        'line_number': line_idx + 1
                    })
        
        if self.progress_callback:
            self.progress_callback(0.9, "Removing duplicates...")
        
        # Remove duplicates and sort
        recommendations = self._remove_duplicates(recommendations)
        recommendations.sort(key=lambda x: x['position'])
        
        if self.progress_callback:
            self.progress_callback(1.0, "Complete!")
        
        return recommendations
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        # Fix common encoding issues
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Ensure space after periods
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        return text
    
    def _check_numbered_recommendation(self, line: str) -> Tuple[bool, str]:
        """Check if line is a numbered recommendation."""
        # Pattern: "1. Verb..." or "1) Verb..." or "1 Verb..."
        patterns = [
            r'^(\d{1,2})\.?\s+([A-Z][a-z]+)\s+',
            r'^(\d{1,2})\)\s+([A-Z][a-z]+)\s+',
            r'^([a-z])\.\s+([A-Z][a-z]+)\s+',  # Lettered lists
            r'^([a-z])\)\s+([A-Z][a-z]+)\s+',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                first_word = match.group(2).lower()
                if first_word in self.action_verbs:
                    return True, first_word
                
                # Check second word
                words = line.split()
                if len(words) > 2:
                    second_word = words[2].lower().strip('.,;:')
                    if second_word in self.action_verbs:
                        return True, second_word
        
        return False, 'unknown'
    
    def _evaluate_sentence(self, sentence: str) -> Tuple[float, str, str]:
        """
        Evaluate a sentence for recommendation patterns.
        Returns (confidence, method, verb)
        """
        sentence_lower = sentence.lower()
        words = sentence.split()
        
        if not words:
            return 0.0, 'none', 'unknown'
        
        # Check for explicit recommendation phrases
        if 'recommend' in sentence_lower:
            verb = self._extract_verb(sentence)
            return 0.9, 'explicit', verb
        
        # Check for imperative (starts with verb)
        first_word = words[0].lower().strip('.,;:!?"\'')
        if first_word in self.action_verbs:
            return 0.75, 'imperative', first_word
        
        # Check for gerund opening
        if first_word in self.recommendation_gerunds:
            base_verb = first_word[:-3] if first_word.endswith('ing') else first_word
            if base_verb in self.action_verbs:
                return 0.8, 'gerund', base_verb
            return 0.75, 'gerund', first_word
        
        # Check modal patterns
        for pattern, confidence in self.modal_patterns:
            match = re.search(pattern, sentence_lower)
            if match:
                verb = match.group(1) if match.lastindex else 'unknown'
                if verb in self.action_verbs or verb.endswith(('ed', 'ing')):
                    return confidence, 'modal', verb
        
        return 0.0, 'none', 'unknown'
    
    def _extract_verb(self, sentence: str) -> str:
        """Extract the main action verb from a sentence."""
        sentence_lower = sentence.lower()
        
        # Find the first action verb in the sentence
        for verb in self.action_verbs:
            if re.search(r'\b' + verb + r'\b', sentence_lower):
                return verb
        
        # Try NLTK if available
        if NLP_AVAILABLE:
            try:
                tokens = word_tokenize(sentence)
                tagged = pos_tag(tokens)
                verbs = [word.lower() for word, pos in tagged 
                        if pos.startswith('VB') and word.lower() in self.action_verbs]
                if verbs:
                    return verbs[0]
            except:
                pass
        
        return 'unknown'
    
    def _is_about_recommendations(self, text: str) -> bool:
        """Check if text is ABOUT recommendations rather than making them."""
        text_lower = text.lower()
        for pattern in self.exclusion_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _is_recommendation_section_header(self, text: str) -> bool:
        """Check if text is a recommendation section header."""
        text_lower = text.lower().strip()
        patterns = [
            r'^recommendations?:?\s*$',
            r'^key recommendations?:?\s*$',
            r'^main recommendations?:?\s*$',
            r'^summary of recommendations?:?\s*$',
            r'^\d+\.?\s*recommendations?:?\s*$',
        ]
        return any(re.match(p, text_lower) for p in patterns)
    
    def _is_new_section_header(self, text: str) -> bool:
        """Check if text is a new non-recommendation section."""
        text_lower = text.lower().strip()
        patterns = [
            r'^introduction:?\s*$',
            r'^background:?\s*$',
            r'^methodology:?\s*$',
            r'^findings?:?\s*$',
            r'^conclusions?:?\s*$',
            r'^appendix',
            r'^references?:?\s*$',
        ]
        return any(re.match(p, text_lower) for p in patterns)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Protect abbreviations
        protected = text
        for abbr in ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Inc', 'Ltd', 'Corp', 'vs', 'etc', 'i.e', 'e.g']:
            protected = protected.replace(f'{abbr}.', f'{abbr}<<DOT>>')
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
        
        # Restore dots and clean
        sentences = [s.replace('<<DOT>>', '.').strip() for s in sentences]
        
        return [s for s in sentences if s and len(s.split()) >= 3]
    
    def _remove_duplicates(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations."""
        if not recommendations:
            return []
        
        unique = []
        seen_normalized = set()
        
        for rec in recommendations:
            # Normalize for comparison
            normalized = re.sub(r'[^\w\s]', '', rec['text'].lower())
            normalized = ' '.join(normalized.split())
            
            if normalized not in seen_normalized:
                unique.append(rec)
                seen_normalized.add(normalized)
        
        return unique
    
    def get_statistics(self, recommendations: List[Dict]) -> Dict:
        """Calculate statistics about extracted recommendations."""
        if not recommendations:
            return {
                'total': 0,
                'by_method': {},
                'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
                'avg_confidence': 0.0,
                'verbs': {},
                'in_sections': 0
            }
        
        verb_counts = Counter(r['verb'] for r in recommendations if r['verb'] != 'unknown')
        method_counts = Counter(r['method'] for r in recommendations)
        
        confidence_ranges = {
            'high': sum(1 for r in recommendations if r['confidence'] >= 0.85),
            'medium': sum(1 for r in recommendations if 0.7 <= r['confidence'] < 0.85),
            'low': sum(1 for r in recommendations if r['confidence'] < 0.7)
        }
        
        return {
            'total': len(recommendations),
            'by_method': dict(method_counts),
            'by_confidence': confidence_ranges,
            'avg_confidence': round(sum(r['confidence'] for r in recommendations) / len(recommendations), 2),
            'verbs': dict(verb_counts.most_common(10)),
            'in_sections': sum(1 for r in recommendations if r.get('in_section', False)),
            'unique_verbs': len(verb_counts)
        }


# Convenience function for backward compatibility
def extract_recommendations(
    text: str, 
    min_confidence: float = 0.7,
    return_stats: bool = False,
    progress_callback=None
) -> List[Dict]:
    """
    Extract recommendations from text.
    
    Args:
        text: Document text
        min_confidence: Minimum confidence threshold (0-1)
        return_stats: Whether to return statistics
        progress_callback: Optional Streamlit progress callback
        
    Returns:
        List of recommendations or tuple with stats if requested
    """
    extractor = RecommendationExtractor(progress_callback=progress_callback)
    return extractor.extract_recommendations(text, min_confidence, return_stats)


# For testing only
if __name__ == "__main__":
    print("Recommendation Extractor Module v2.0")
    print("Ready for import into Streamlit app")
    print("\nUsage:")
    print("  from recommendation_extractor_final import extract_recommendations")
    print("  recommendations = extract_recommendations(text, min_confidence=0.7)")
