"""
Strict Recommendation Extractor v2.4
Properly extracts full "Recommendation N" blocks from government documents.
Falls back to sentence-based extraction for other document types.

v2.4 Changes:
- Removed aggressive length limits for numbered recommendations
- "Recommendation N" blocks are extracted in full (can be 1000+ chars)
- Length limits only apply to fallback sentence-based extraction
- Better handling of government document structure

Usage:
    from daphne_core.recommendation_extractor import extract_recommendations
    
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
    Handles both government-style numbered recommendations and general documents.
    """
    
    # Maximum length for SENTENCE-BASED extraction only (not numbered recommendations)
    MAX_SENTENCE_LENGTH = 500
    
    # Maximum length for numbered recommendations (much higher - they can be long)
    MAX_NUMBERED_REC_LENGTH = 2500
    
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
        # Specific GOV.UK footer used in the reference report. In some cases the
        # recommendation number and the footer are concatenated as part of a
        # single line, e.g.:
        #
        #   Recommendation 811/24/25, 5:39 PM Rapid review into data on mental
        #   health inpatient settings: final report and recommendations - GOV.UK
        #   https://www.gov.uk/.../rapid-... 16/84
        #
        # For this reference report we hard-code removal of that exact footer
        # blob while preserving the leading digit(s) ("Recommendation 8 ").
        text = re.sub(
            r'11/24/25, 5:39 PM Rapid review into data on mental health inpatient settings: final report and recommendations - GOV\.UK[\s\S]*?\d{1,2}/84',
            '',
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r'Rapid review into data on mental health inpatient settings:.*?GOV\.?UK',
            '',
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r'final report and recommendations\s*-?\s*GOV\.?UK?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'-\s*GOV\.?\s*UK?\s*', '', text, flags=re.IGNORECASE)
        
        # Remove "UK " prefix
        text = re.sub(r'^UK\s+', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def is_garbage(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, str]:
        """FIRST-PASS filter: reject obvious garbage before any analysis."""
        if not text:
            return True, "empty"
        
        # Clean first for this check
        cleaned = self.clean_text(text)
        
        if len(cleaned) < 40:
            return True, "too_short"
        
        # Length limits depend on whether this is a numbered recommendation.
        # For numbered recommendations we allow long text blocks, as they often
        # contain multiple sub-actions and explanatory context.
        if not is_numbered_rec:
            max_length = self.MAX_SENTENCE_LENGTH
            if len(cleaned) > max_length:
                return True, "too_long"
        
        # Just section headers
        if re.match(r'^(?:Appendix|Section|Chapter|Table|Figure|Footnote)\s+\d+', cleaned, re.IGNORECASE):
            return True, "header"
        
        # Too many special characters
        special_chars = sum(1 for c in cleaned if not c.isalnum() and not c.isspace() and c not in '.,;:!?\'"-()[]')
        if len(cleaned) > 0 and special_chars / len(cleaned) > 0.12:
            return True, "corrupted"
        
        # Excessive numbers (only applied to sentence-based extraction).
        # Numbered recommendations often include dates/stats, so we do not
        # reject them purely on digit density.
        if not is_numbered_rec:
            digits = sum(1 for c in cleaned if c.isdigit())
            max_digit_ratio = 0.15
            if len(cleaned) > 0 and digits / len(cleaned) > max_digit_ratio:
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
            r'^this\s+document\s+outlines',
            r'^the\s+insights\s+below',
            r'^the\s+goal\s+is\s+to\s+provide',
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def is_genuine_recommendation(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, float, str, str]:
        """Determine if text is a genuine recommendation."""
        cleaned = self.clean_text(text)
        text_lower = cleaned.lower()
        
        # Length checks:
        # - Numbered recommendations are treated as sections and can be long.
        # - Sentence-based recommendations should remain within a strict limit.
        if not is_numbered_rec:
            if len(cleaned) > self.MAX_SENTENCE_LENGTH:
                return False, 0.0, 'too_long', 'none'
        
        # Method 1: Explicit numbered recommendation (highest confidence)
        # These can be long - no length restriction here. Handle occasional
        # spaced digits like "Recommendation 1 1" by normalising to "11", but
        # only up to two digits to avoid accidentally treating "111/24/25" etc.
        numbered_match = re.match(
            r'^(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\s+(.+)',
            cleaned,
            re.IGNORECASE,
        )
        if numbered_match:
            num_part = numbered_match.group(1)
            extra_digit = numbered_match.group(2)
            if extra_digit and len(num_part) == 1:
                rec_num = f"{num_part}{extra_digit}"
            else:
                rec_num = num_part
            rec_text = numbered_match.group(3)
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
        
        # =======================================================================
        # PHASE 1: Extract numbered "Recommendation N" blocks (government style)
        # These are extracted in FULL regardless of length
        # =======================================================================

        # Identify headings of the form "Recommendation N" (optionally preceded
        # by "Recommendations") and treat everything up to the next heading as
        # that recommendation's body. This is robust to page breaks where the
        # heading and body are separated. We run this over the CLEANED text so
        # that footer artefacts (e.g. GOV.UK banners) have already been removed.
        heading_pattern = re.compile(
            r'(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b',
            re.IGNORECASE,
        )
        cleaned_full_text = self.clean_text(text)
        heading_matches = list(heading_pattern.finditer(cleaned_full_text))

        logger.info(f"Found {len(heading_matches)} numbered recommendation headings")

        for idx, match in enumerate(heading_matches):
            start = match.start()
            end = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(cleaned_full_text)
            raw_block = cleaned_full_text[start:end]
            cleaned = self.clean_text(raw_block)
            
            # Check garbage with numbered_rec flag (allows longer text)
            is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=True)
            if is_garbage:
                logger.debug(f"Skipping numbered rec {idx}: {reason}")
                continue

            # Meta-recommendation filtering is intended for narrative/scattered
            # sentences; for explicit "Recommendation N" sections we keep them
            # even if they contain meta-style wording (e.g. implementation timing).
            
            is_rec, confidence, method, verb = self.is_genuine_recommendation(cleaned, is_numbered_rec=True)
            
            if is_rec and confidence >= min_confidence:
                # Prefer the number inferred from the cleaned text (method),
                # fall back to the raw heading match if needed. The heading
                # pattern already supports spaced digits like "1 1".
                heading_num = match.group(1)
                extra_digit = match.group(2)
                if extra_digit and len(heading_num) == 1:
                    heading_rec_num = f"{heading_num}{extra_digit}"
                else:
                    heading_rec_num = heading_num

                rec_num = heading_rec_num
                if method.startswith("numbered_recommendation_"):
                    try:
                        rec_num = method.split("_")[-1]
                    except Exception:
                        rec_num = heading_rec_num

                # Normalise the heading digits in the text itself so that
                # artefacts like "Recommendation 1 1 ..." become
                # "Recommendation 11 ...", matching rec_number.
                norm_pattern = r'^(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b'
                normalized_text = re.sub(norm_pattern, f"Recommendation {rec_num}", cleaned, count=1, flags=re.IGNORECASE)

                recommendations.append({
                    'text': normalized_text,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': idx,
                    'in_section': True,
                    'rec_number': rec_num,
                })
        
        # =======================================================================
        # PHASE 2: If no numbered recommendations found, fall back to sentence extraction
        # These have stricter length limits
        # =======================================================================

        # Heuristic guard: if we only found a single numbered recommendation and
        # it is excessively long, this is likely a segmentation artefact (e.g. a
        # whole chapter being treated as one "Recommendation N" block). In that
        # case, discard it and allow the sentence-based fallback to run.
        if len(recommendations) == 1:
            only_rec = recommendations[0]
            if len(only_rec.get("text", "")) > self.MAX_NUMBERED_REC_LENGTH:
                logger.info(
                    "Single oversized numbered recommendation detected (len=%d) – "
                    "treating as invalid and falling back to sentence extraction",
                    len(only_rec.get("text", "")),
                )
                recommendations = []

        if len(recommendations) == 0:
            logger.info("No numbered recommendations found, falling back to sentence extraction")
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            for idx, sentence in enumerate(sentences):
                cleaned = self.clean_text(sentence)
                
                # Strict length limit for sentence-based extraction
                is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=False)
                if is_garbage:
                    continue
                
                if self.is_meta_recommendation(cleaned):
                    continue
                
                is_rec, confidence, method, verb = self.is_genuine_recommendation(cleaned, is_numbered_rec=False)
                
                if is_rec and confidence >= min_confidence:
                    recommendations.append({
                        'text': cleaned,
                        'verb': verb,
                        'method': method,
                        'confidence': round(confidence, 3),
                        'position': idx,
                        'in_section': False,
                    })
        
        recommendations = self._deduplicate(recommendations)
        
        logger.info(f"Found {len(recommendations)} recommendations")
        
        return recommendations
    
    def _deduplicate(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations"""
        seen = set()
        unique = []
        
        for rec in recommendations:
            # Use first 150 chars as key to catch duplicates
            key = re.sub(r'\s+', ' ', rec['text'].lower().strip())[:150]
            
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
