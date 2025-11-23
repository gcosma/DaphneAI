"""
AdvancedRecommendationExtractor (improved)

Improvements over the original:
- Retains existing functionality (gerund, modal, imperative, explicit phrases, bullets, sections)
- Better sentence splitting that preserves short header-like lines (e.g. "1. Strengthen ...")
- Recognizes and extracts numbered/lettered section headers as recommendations
- Stronger exclusion patterns to avoid meta-descriptive text being classified as recommendations
- Imperative detection robust to numeric prefixes ("1. Strengthen...")
- Section-aware confidence boosting and bullet-in-section handling
- Clearer verb extraction with NLTK fallback if available
- Preserves public functions and original API

Note: NLTK remains optional; behavior gracefully degrades without it.
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
    # quiet downloads if first time - wrapped in try/except to avoid noise
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    NLP_AVAILABLE = True
except Exception:
    NLP_AVAILABLE = False
    logger.warning("NLTK not available - using pattern-based extraction only")


class AdvancedRecommendationExtractor:
    """
    Extract recommendations from government documents using multiple methods:
    1. Explicit recommendation phrases
    2. Imperative sentences (command form / heading verbs)
    3. Modal verbs (should, must, ought, need)
    4. Gerund openings (Improving, Ensuring, etc.)
    5. Numbered/bulleted recommendation sections
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

        # Action verbs commonly used in recommendations
        self.action_verbs = {
            'implement', 'establish', 'create', 'develop', 'introduce',
            'ensure', 'enable', 'improve', 'enhance', 'strengthen',
            'broaden', 'expand', 'increase', 'reduce', 'minimise',
            'provide', 'support', 'maintain', 'review', 'update',
            'adopt', 'require', 'mandate', 'enforce', 'monitor',
            'assess', 'evaluate', 'consider', 'explore', 'investigate',
            'reform', 'modernise', 'streamline', 'simplify', 'clarify',
            'promote', 'encourage', 'facilitate', 'coordinate', 'integrate'
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
            'modernising', 'streamlining', 'simplifying', 'clarifying',
            'coordinating', 'integrating'
        }

        # Section headers that indicate recommendations
        self.recommendation_section_patterns = [
            r'^recommendations?:?\s*$',
            r'^key recommendations?:?\s*$',
            r'^main recommendations?:?\s*$',
            r'^summary of recommendations?:?\s*$',
            r'^\d+\.?\s*recommendations?:?\s*$',
        ]

        # Bullet point patterns
        self.bullet_patterns = [
            r'^\s*[•●■▪▸►]+\s+',
            r'^\s*[-–—]\s+',
            r'^\s[*]\s+',
            r'^\s*[○◦]\s+',
            r'^\s*[✓✔]\s+',
            r'^\s*\d+[\.)]\s+',  # Numbered lists
            r'^\s*[a-zA-Z][\.)]\s+',  # Lettered lists (allow uppercase)
        ]

        # Extended exclusion patterns to avoid meta descriptions being classified
        self.exclusion_patterns = [
            r'\bset of recommendations\b',
            r'\bthis document (?:outlines|provides|contains|presents)\b',
            r'\bcontains\s+recommendations\b',
            r'\bfindings?.*and.*recommendations\b',
            r'\brecommendations? (?:are|can be|will be) (?:found|developed|published|available)\b',
            r'\brecommendations? include\b',
            r'\bthe recommendations?\b.*\b(?:report|document|section)\b',
            r'\bimplementation of.*recommendations?\b',
            r'\bacting on.*recommendations?\b',
            r'\bfollowing.*recommendations?\b',
            r'\bthese recommendations?\b',
            r'\bthe chair\'s? recommendations?\b',
            r'\boutlines.*recommendations\b',
            r'^module \d',
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
        sentences = self._split_into_sentences(text)

        # Track if we're in a recommendation section
        in_rec_section = False
        section_confidence_boost = 0.0

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if this is a section header signaling the start of recs
            if self._is_recommendation_section_header(sentence):
                in_rec_section = True
                section_confidence_boost = 0.15
                continue

            # If line looks like a top-level section header (e.g. "1. Strengthen...")
            header_conf, header_method = self._check_header_recommendation(sentence)
            if header_conf > 0:
                confidence = min(1.0, header_conf + (0.15 if in_rec_section else 0.0))
                verb = self._extract_main_verb(sentence)
                recommendations.append({
                    'text': sentence,
                    'verb': verb,
                    'method': header_method,
                    'confidence': round(confidence, 2),
                    'position': idx,
                    'in_section': in_rec_section
                })
                # continue to next sentence (we've already classified it)
                continue

            # Check if we've left the recommendation section
            if in_rec_section and self._is_new_section_header(sentence):
                in_rec_section = False
                section_confidence_boost = 0.0

            # Method 1: Explicit recommendation phrases
            confidence, method = self._check_explicit_recommendations(sentence)

            # Method 2: Starts with gerund
            if confidence < 0.7:
                gerund_conf = self._check_gerund_opening(sentence)
                if gerund_conf > confidence:
                    confidence = gerund_conf
                    method = 'gerund'

            # Method 3: Imperative sentence (command form)
            if confidence < 0.7:
                imperative_conf = self._check_imperative_form(sentence)
                if imperative_conf > confidence:
                    confidence = imperative_conf
                    method = 'imperative'

            # Method 4: Modal verbs with action
            if confidence < 0.7:
                modal_conf = self._check_modal_action(sentence)
                if modal_conf > confidence:
                    confidence = modal_conf
                    method = 'modal'

            # Method 5: Bullet point in recommendation section
            if in_rec_section and self._is_bullet_point(sentence):
                confidence = max(confidence, 0.75)
                method = 'bullet_in_section'

            # Boost confidence if in recommendation section
            confidence += section_confidence_boost
            confidence = min(confidence, 1.0)

            # Filter out sentences that are explicitly about recommendations (meta)
            if self._is_meta_recommendation(sentence):
                continue

            # Extract main verb
            verb = self._extract_main_verb(sentence)

            # Only add if confidence is sufficient
            if confidence >= min_confidence:
                recommendations.append({
                    'text': sentence,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 2),
                    'position': idx,
                    'in_section': in_rec_section
                })

        # Remove duplicates and very similar recommendations
        recommendations = self._remove_duplicates(recommendations)

        # Sort by position (order in document)
        recommendations.sort(key=lambda x: x['position'])

        return recommendations

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling and preserve headers.

        This splitter aims to keep header-like lines (e.g. "1. Strengthen...")
        and short imperative headings which are often important recommendations.
        """
        # Protect common numeric decimals and abbreviations
        text = re.sub(r'(\d+)\.(\d+)', r'\1<<DOT>>\2', text)
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof)\.', r'\1<<DOT>>', text)

        # Ensure periods stuck to capitals are spaced
        text = re.sub(r'\.([A-Z])', r'. \1', text)

        # Split on sentence boundaries: period/!? followed by space + capital
        candidate_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'"'(\[]))', text)

        final_sentences = []
        for sent in candidate_sentences:
            # Also split on newlines to capture header lines
            parts = re.split(r'\n+', sent)
            for p in parts:
                p = p.replace('<<DOT>>', '.').strip()
                if not p:
                    continue
                # Keep lines that are reasonably long OR look like headers/list items
                if len(p.split()) >= 3 or re.match(r'^\s*\d+[\.)]\s+\w+', p) or re.match(r'^[A-Z][a-z]+:\s*$', p):
                    final_sentences.append(p)
                else:
                    # small lines may be part of nearby sentence - attempt to attach
                    if final_sentences:
                        final_sentences[-1] = final_sentences[-1] + ' ' + p
                    else:
                        final_sentences.append(p)

        # Final cleanup: strip and return
        final_sentences = [s.strip() for s in final_sentences if len(s.strip()) > 0]
        return final_sentences

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
            r'^\d+\.?\s+\w+:?\s*$',  # Numbered headers
        ]

        text_lower = text.lower().strip()

        for pattern in non_rec_headers:
            if re.match(pattern, text_lower):
                return True

        return False

    def _check_explicit_recommendations(self, sentence: str) -> Tuple[float, str]:
        """Check for explicit recommendation phrases."""
        sentence_lower = sentence.lower()

        # Check exclusion (meta) patterns first
        for pattern in self.exclusion_patterns:
            if re.search(pattern, sentence_lower):
                return 0.0, 'none'

        # High confidence patterns - actually MAKING recommendations
        for pattern in self.recommendation_indicators['high']:
            if re.search(pattern, sentence_lower):
                # Avoid meta-mentions that include 'recommend' but are describing the report
                if self._is_meta_recommendation(sentence):
                    return 0.0, 'none'
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

    def _is_meta_recommendation(self, sentence: str) -> bool:
        """Detect sentences that talk ABOUT recommendations rather than giving them."""
        s = sentence.lower()
        meta_indicators = [
            'this document', 'this report', 'set of recommendations', 'contains recommendations',
            'findings and recommendations', 'recommendations include', 'the recommendations section',
            'the recommendations are', 'recommendations for the future', 'these recommendations'
        ]
        for token in meta_indicators:
            if token in s:
                return True
        return False

    def _check_gerund_opening(self, sentence: str) -> float:
        """Check if sentence starts with a recommendation gerund."""
        words = sentence.split()
        if not words:
            return 0.0

        # Skip numeric or bullet prefixes
        i = 0
        if re.match(r'^\d+[\.)]$', words[0].strip()):
            i = 1
        if i < len(words):
            first_word = words[i].strip('.,;:!?"\'"').lower()
        else:
            return 0.0

        if first_word in self.recommendation_gerunds:
            return 0.85

        # Check if it ends with 'ing' and might be a verb
        if first_word.endswith('ing') and len(first_word) > 4:
            if NLP_AVAILABLE:
                try:
                    base = first_word[:-3]
                    if base in self.action_verbs:
                        return 0.8
                except Exception:
                    pass
            return 0.5

        return 0.0

    def _check_imperative_form(self, sentence: str) -> float:
        """Check if sentence is in imperative form (command).

        This handles cases like "Strengthen cross-team..." and "1. Strengthen...".
        """
        words = sentence.split()
        if not words:
            return 0.0

        # Normalize: skip leading numbers/letters used as list markers
        i = 0
        if re.match(r'^\d+[\.)]$', words[0].strip()):
            i = 1
        elif re.match(r'^[a-zA-Z][\.)]$', words[0].strip()):
            i = 1

        if i >= len(words):
            return 0.0

        first_word = words[i].strip('.,;:!?"\'"').lower()

        # Check if first word is an action verb (base form)
        if first_word in self.action_verbs and not first_word.endswith('ing'):
            return 0.9

        # If NLP available, check POS tags for imperative-like structure
        if NLP_AVAILABLE:
            try:
                tokens = word_tokenize(' '.join(words[i:i+8]))
                tagged = pos_tag(tokens)
                # If first real token is a verb in base form (VB) or VB without subject, boost
                if tagged and tagged[0][1] == 'VB':
                    return 0.85
            except Exception:
                pass

        return 0.0

    def _check_modal_action(self, sentence: str) -> float:
        """Check for modal verbs followed by action verbs."""
        sentence_lower = sentence.lower()

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

    def _check_header_recommendation(self, sentence: str) -> Tuple[float, str]:
        """Detect header-like recommendations such as '1. Strengthen ...' or 'A. Improve...'.

        Returns a (confidence, method) tuple.
        """
        s = sentence.strip()
        # Numeric header: e.g. "1. Strengthen..."
        m = re.match(r'^\s*(\d+)[\.)]\s*(.+)$', s)
        if m:
            rest = m.group(2).strip()
            # Check if rest begins with an action verb
            first = rest.split()[0].lower() if rest.split() else ''
            if first in self.action_verbs:
                return 0.95, 'section_header'
            # If starts with TitleCase verb (e.g. Strengthen), still likely a rec
            if re.match(r'^[A-Z][a-z]+', rest):
                return 0.85, 'section_header'
        # Letter header: 'a) Improve...' or 'A. Improve...'
        m2 = re.match(r'^\s*[A-Za-z][\.)]\s*(.+)$', s)
        if m2:
            rest = m2.group(1).strip()
            first = rest.split()[0].lower() if rest.split() else ''
            if first in self.action_verbs:
                return 0.9, 'section_header'
            if re.match(r'^[A-Z][a-z]+', rest):
                return 0.8, 'section_header'

        return 0.0, 'none'

    def _extract_main_verb(self, sentence: str) -> str:
        """Extract the main action verb from a sentence."""
        words = sentence.split()
        if not words:
            return 'unknown'

        sentence_lower = sentence.lower()

        # Skip leading bullets/number markers
        i = 0
        if re.match(r'^\s*[-–—•*✓✔]+$', words[0].strip()):
            i = 1
        if re.match(r'^\d+[\.)]$', words[0].strip()):
            i = 1
        if i < len(words):
            first_word = words[i].strip('.,;:!?'"'"').lower()
        else:
            first_word = words[0].lower()

        # Direct matches
        if first_word in self.action_verbs:
            return first_word

        if first_word.endswith('ing') and len(first_word) > 4:
            base = first_word[:-3]
            if base in self.action_verbs:
                return base
            return first_word

        # Use NLTK if available for better extraction
        if NLP_AVAILABLE:
            try:
                tokens = word_tokenize(sentence[:300])
                tagged = pos_tag(tokens)
                verbs = [word.lower() for word, pos in tagged if pos.startswith('VB')]
                auxiliaries = {'be', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}
                for verb in verbs:
                    if verb in self.action_verbs:
                        return verb
                    if verb not in auxiliaries:
                        return verb
                if verbs:
                    return verbs[0]
            except Exception:
                pass

        # Fallback: search for any action verb in the sentence
        s_sorted = sorted(self.action_verbs, key=len, reverse=True)
        for verb in s_sorted:
            if re.search(r'\b' + re.escape(verb) + r'\b', sentence_lower):
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

            # Remove bullet points and numeric markers for comparison
            for pattern in self.bullet_patterns:
                text = re.sub(pattern, '', text)
            text = re.sub(r'^\s*\d+[\.)]\s*', '', text)

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
        words1 = set(re.findall(r"\w+", text1))
        words2 = set(re.findall(r"\w+", text2))

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
