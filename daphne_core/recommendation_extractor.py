"""
Recommendation Extractor v3.3
Extracts recommendation blocks from government and health sector documents.

v3.3 Changes:
- FIXED: Improved section boundary detection for title-case and sentence-case headers
- FIXED: Added explicit markers for "Our vision" and similar post-recommendations sections
- FIXED: HSIB deduplication by rec_number (keeps longest version)
- FIXED: MAX_RECOMMENDATION_LENGTH increased to 2500 for longer government recs
- FIXED: Removed aggressive boundary detection from v3.2 that was truncating recs

v3.1 Changes (preserved):
- Boundary detection catches inline "Recommendation N" patterns
- Handles "Recommendation 1 1" (spaced digits) correctly as boundaries
- Prevents Recommendation N from bleeding into Recommendation N-1

Supported Formats:
- HSIB 2018 format: "Recommendation 2018/006:", "Recommendation 2018/007:"
- HSSIB 2023+ format: "Safety recommendation R/2023/220:"
- Standard government: "Recommendation 1", "Recommendation 12"
- Sentence-based fallback for unstructured documents
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class StrictRecommendationExtractor:
    """
    Extract recommendations with aggressive pre-filtering.
    Handles government-style numbered recommendations and general documents.
    """
    
    # Length limits
    MAX_SENTENCE_LENGTH = 500
    MAX_RECOMMENDATION_LENGTH = 2500  # v3.3: Increased from 1500 for longer gov recs
    MIN_RECOMMENDATION_LENGTH = 50
    
    def __init__(self):
        """Initialise with patterns and action verbs"""
        
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
            'come', 'form', 'work', 'learn', 'drive', 'produce', 'require',
        }
        
        self.recommending_entities = [
            r'NHS\s+England',
            r'NHS\s+Improvement',
            r'HSSIB',
            r'HSIB',
            r'(?:the\s+)?(?:Home\s+Office|Cabinet\s+Office|Treasury)',
            r'(?:the\s+)?(?:Department|Ministry)\s+(?:of|for)\s+[\w\s]+',
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
            r'NIHR',
            r'Regulators?',
            r'Inpatient\s+staff',
            r'Government\s+ministers?',
            r'Clinical\s+Commissioning\s+Groups?',
            r'CCGs?',
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
        """Clean text of noise and artifacts"""
        if not text:
            return ""
        
        text = self.fix_encoding(text)
        
        # Remove URLs
        text = re.sub(r'https?://[^\s<>"\']+', '', text)
        text = re.sub(r'www\.[^\s<>"\']+', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.gov\.uk[^\s]*', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.org\.uk[^\s]*', '', text)
        text = re.sub(r'[a-zA-Z0-9.-]+\.nhs\.uk[^\s]*', '', text)
        
        # Remove timestamps
        text = re.sub(
            r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?[^A-Z]*(?=[A-Z]|$)',
            ' ', text, flags=re.IGNORECASE
        )
        
        # Remove page numbers (but NOT recommendation IDs like 2018/006 or R/2023/220)
        text = re.sub(r'\b(\d{1,2})/(\d{1,3})\b', '', text)
        text = re.sub(r'\bPage\s+\d+\s+(?:of\s+\d+)?', '', text, flags=re.IGNORECASE)
        
        # Remove GOV.UK footer artifacts
        text = re.sub(
            r'Rapid review into data on mental health inpatient settings:.*?GOV\.?UK',
            '', text, flags=re.IGNORECASE
        )
        text = re.sub(r'final report and recommendations\s*-?\s*GOV\.?UK?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'-\s*GOV\.?\s*UK?\s*', '', text, flags=re.IGNORECASE)
        
        # Normalise duplicated section headings
        text = re.sub(
            r'\bRecommendations\s+Recommendation\s+(\d+)\b',
            r'Recommendation \1',
            text,
            flags=re.IGNORECASE,
        )
        
        # Remove "UK " prefix
        text = re.sub(r'^UK\s+', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def is_garbage(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, str]:
        """First-pass filter: reject obvious garbage before analysis."""
        if not text:
            return True, "empty"
        
        cleaned = self.clean_text(text)
        
        if len(cleaned) < self.MIN_RECOMMENDATION_LENGTH:
            return True, "too_short"
        
        if not is_numbered_rec:
            if len(cleaned) > self.MAX_SENTENCE_LENGTH:
                return True, "too_long"
        
        if re.match(r'^(?:Appendix|Section|Chapter|Table|Figure|Footnote)\s+\d+', cleaned, re.IGNORECASE):
            return True, "header"
        
        special_chars = sum(1 for c in cleaned if not c.isalnum() and not c.isspace() and c not in '.,;:!?\'"-()[]')
        if len(cleaned) > 0 and special_chars / len(cleaned) > 0.12:
            return True, "corrupted"
        
        if not is_numbered_rec:
            digits = sum(1 for c in cleaned if c.isdigit())
            if len(cleaned) > 0 and digits / len(cleaned) > 0.15:
                return True, "too_many_numbers"
        
        return False, ""
    
    def is_meta_recommendation(self, text: str) -> bool:
        """Check if text is talking ABOUT recommendations rather than being one."""
        if not text:
            return False
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
    
    def _extract_verb(self, text: str) -> str:
        """Extract the main action verb from recommendation text"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        should_match = re.search(r'\bshould\s+(?:also\s+)?(?:urgently\s+)?(?:be\s+)?(\w+)', text_lower)
        if should_match:
            verb = should_match.group(1)
            if verb in self.action_verbs:
                return verb
        
        recommends_match = re.search(r'\brecommends?\s+that\s+[\w\s]+\s+(\w+)', text_lower)
        if recommends_match:
            verb = recommends_match.group(1)
            if verb in self.action_verbs:
                return verb
        
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}s?\b', text_lower):
                return verb
        
        return 'unknown'
    
    def _is_section_boundary(self, text: str, position: int) -> bool:
        """
        GENERIC section boundary detection.
        Returns True if position appears to be at a major section break.
        
        v3.3: Improved to catch sentence-case headers like "Our vision for a better future"
        """
        if position >= len(text):
            return True
        
        # Look at text from this position
        remaining = text[position:position + 200]
        
        # Check for numbered section headers: "1 Background", "2. Analysis", "1.1 Introduction"
        if re.match(r'^\s*\d+\.?\d*\.?\s+[A-Z][a-z]+', remaining):
            return True
        
        # Check for title case headers on their own line
        lines = remaining.split('\n')
        if lines:
            first_line = lines[0].strip()
            
            # Skip if too short or too long
            if len(first_line) < 5 or len(first_line) > 80:
                return False
            
            # Skip if ends with period (it's a sentence, not a header)
            if first_line.endswith('.'):
                return False
            
            # Skip if starts with common sentence starters
            if first_line.startswith(('The ', 'A ', 'An ', 'This ', 'That ', 'It ', 'We ', 'They ')):
                return False
            
            # Must start with uppercase
            if not first_line[0].isupper():
                return False
            
            words = first_line.split()
            if len(words) < 2:
                return False
            
            # v3.3: More lenient check - first word capitalised + short phrase = likely header
            # Headers like "Our vision for a better future" have first word caps + lowercase rest
            if len(words) <= 8 and words[0][0].isupper():
                # Check if it looks like a header vs. a sentence fragment
                # Headers rarely contain certain words
                sentence_indicators = ['is', 'are', 'was', 'were', 'have', 'has', 'had', 
                                       'will', 'would', 'could', 'should', 'may', 'might',
                                       'because', 'although', 'however', 'therefore']
                has_sentence_indicator = any(w.lower() in sentence_indicators for w in words)
                
                if not has_sentence_indicator:
                    return True
            
            # Original title case check (multiple capitalised words)
            caps = sum(1 for w in words if w[0].isupper() or w.lower() in ['and', 'the', 'of', 'for', 'to', 'in', 'a', 'an'])
            if caps >= len(words) * 0.7:
                return True
        
        return False
    
    def _find_recommendation_end_generic(self, text: str, start_pos: int, next_rec_pos: int) -> int:
        """
        GENERIC end-of-recommendation detection.
        Uses structural heuristics rather than document-specific patterns.
        
        v3.1 FIX: Now checks for inline "Recommendation N" patterns that might
        have been missed, especially when digits are spaced (e.g., "Recommendation 1 1")
        
        v3.3: Added explicit markers for post-recommendations sections
        """
        # If there's a next recommendation, that's our hard boundary
        if next_rec_pos < len(text):
            search_limit = next_rec_pos
        else:
            search_limit = len(text)
        
        # Start searching after the recommendation header
        search_start = start_pos + 50
        
        # =====================================================================
        # v3.1 FIX: Check for inline "Recommendation N" patterns that might have
        # been missed. This catches cases where "Recommendation 1 1" (for 11) or
        # similar patterns appear inline and weren't detected as a heading boundary.
        # =====================================================================
        inline_rec_pattern = re.compile(
            r'\bRecommendation\s+\d{1,2}(?:\s+\d)?\s+(?:All|The|NHS|Every|Provider|Trust|Board|ICS|CQC|DHSC|Ward|More|Professional|Except)',
            re.IGNORECASE
        )
        inline_match = inline_rec_pattern.search(text[search_start:search_limit])
        if inline_match:
            # Found another recommendation start inline - use that as boundary
            potential_end = search_start + inline_match.start()
            if potential_end < search_limit:
                logger.debug(f"Found inline recommendation boundary at position {potential_end}")
                search_limit = potential_end
        
        # Universal end markers (these work across document types)
        # v3.3: Added more explicit section markers
        universal_markers = [
            # Document structure markers
            r'\bAppendix(?:es)?(?:\s+[A-Z0-9]+)?\b',
            r'\bGlossary\b',
            r'\bReferences\b',
            r'\bEndnotes?\b',
            r'\bAcknowledgements?\b',
            r'\bBibliography\b',
            r'\bFurther\s+(?:reading|information)\b',
            r'\bAbout\s+(?:this|the)\s+(?:report|review|investigation|document)\b',
            r'©\s*(?:Crown\s+)?[Cc]opyright',
            r'\bAll\s+content\s+is\s+available\s+under\b',
            r'\bISBN\b',
            # Section transitions for HSIB/HSSIB
            r'\bSafety\s+[Oo]bservation(?:s)?(?:\s+[A-Z0-9/]+)?:',
            r'\bLocal-level\s+learning\b',
            r'\bBackground\s+and\s+context\b',
            r'\bFindings\s+and\s+analysis\b',
            r'\bSummary\s+of\s+(?:findings|recommendations)\b',
            r'\bThe\s+investigation\s+(?:found|makes|identified)\b',
            # v3.3: Explicit markers for common post-recommendations sections
            r'\bOur\s+vision\s+for\s+(?:a\s+)?(?:better\s+)?future\b',
            r'\bKey\s+facts\b',
            r'\bMethodology\b',
            r'\bFindings\b',
            r'\bCase\s+studies\b',
            r'\bMeasuring\s+what\s+matters\b',
            r'\bWhat\s+a\s+better\s+system\s+would\s+look\s+like\b',
            r'\bPrinciples\s+for\s+the\s+collection\b',
            r'\bThe\s+safety\s+issues\s+framework\b',
            r'\bData\s+on\s+deaths\b',
            r'\bPoor\s+safety\s+outcomes\b',
            # Numbered sections (generic)
            r'\n\s*\d+\.?\s+[A-Z][a-z]+(?:\s+[a-z]+){0,3}\s*\n',
        ]
        
        earliest_boundary = search_limit
        
        for pattern in universal_markers:
            match = re.search(pattern, text[search_start:search_limit], re.IGNORECASE)
            if match:
                boundary = search_start + match.start()
                if boundary < earliest_boundary:
                    earliest_boundary = boundary
        
        # Also check for structural boundaries (title case headers, etc.)
        # Scan through looking for section breaks
        check_positions = range(search_start, min(earliest_boundary, search_start + 3000), 100)
        for pos in check_positions:
            if self._is_section_boundary(text, pos):
                # Found a structural boundary
                if pos < earliest_boundary:
                    earliest_boundary = pos
                break
        
        # Apply maximum length safeguard
        max_end = start_pos + self.MAX_RECOMMENDATION_LENGTH
        if earliest_boundary > max_end:
            # Try to find a sentence boundary near the max length
            text_chunk = text[start_pos:max_end]
            last_period = text_chunk.rfind('. ')
            if last_period > self.MIN_RECOMMENDATION_LENGTH:
                earliest_boundary = start_pos + last_period + 1
            else:
                earliest_boundary = max_end
        
        return earliest_boundary
    
    def is_valid_recommendation(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, float, str, str]:
        """Determine if text is a valid recommendation."""
        if not text:
            return False, 0.0, 'empty', 'unknown'
        
        cleaned = self.clean_text(text)
        if not cleaned:
            return False, 0.0, 'empty', 'unknown'
        
        text_lower = cleaned.lower()
        
        if not is_numbered_rec:
            if len(cleaned) > self.MAX_SENTENCE_LENGTH:
                return False, 0.0, 'too_long', 'none'
        
        # HSSIB 2023+ format: "Safety recommendation R/2023/220:"
        hsib_2023_match = re.match(
            r'^Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]+(.+)',
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if hsib_2023_match:
            rec_num = hsib_2023_match.group(1)
            rec_text = hsib_2023_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.98, f'hsib_recommendation_{rec_num}', verb
        
        # HSIB 2018 format: "Recommendation 2018/006:"
        hsib_2018_match = re.match(
            r'^Recommendation\s+(\d{4}/\d{3})[:\s]+(.+)',
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if hsib_2018_match:
            rec_num = hsib_2018_match.group(1)
            rec_text = hsib_2018_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.98, f'hsib_recommendation_{rec_num}', verb
        
        # Standard numbered recommendation: "Recommendation 1"
        numbered_match = re.match(
            r'^(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\s+(.+)',
            cleaned,
            re.IGNORECASE | re.DOTALL,
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
        
        # "HSSIB recommends that" pattern
        hssib_recommends = re.match(
            r'^HSSIB\s+recommends\s+that\s+(.+)',
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if hssib_recommends:
            rec_text = hssib_recommends.group(1)
            verb = self._extract_verb(rec_text)
            return True, 0.96, 'hssib_recommends', verb
        
        # Entity + should pattern
        for entity_pattern in self.recommending_entities:
            pattern = rf'{entity_pattern}\s+should\s+(?:also\s+)?(?:urgently\s+)?(?:\w+ly\s+)?(\w+)'
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match and match.group(1):
                verb = match.group(1).lower()
                if verb in self.action_verbs or verb.endswith('e') or verb.endswith('ise') or verb.endswith('ize'):
                    return True, 0.95, 'entity_should', verb
        
        # "We recommend" pattern
        if re.search(r'\bwe\s+recommend\b', text_lower):
            verb = self._extract_verb(cleaned)
            return True, 0.90, 'we_recommend', verb
        
        # "It is recommended that" pattern
        if re.search(r'\bit\s+is\s+recommended\s+that\b', text_lower):
            verb = self._extract_verb(cleaned)
            return True, 0.90, 'it_is_recommended', verb
        
        # "should be" + action (passive)
        should_be_match = re.search(r'(\w+(?:\s+\w+)?)\s+should\s+be\s+(\w+ed)\b', text_lower)
        if should_be_match:
            subject = should_be_match.group(1)
            verb = should_be_match.group(2)
            if subject not in ['this', 'it', 'that', 'recommendation', 'the recommendation']:
                return True, 0.85, 'should_be_passive', verb
        
        # Modal + action verb
        modal_match = re.search(r'\b(should|must|shall)\s+(\w+)\b', text_lower)
        if modal_match:
            verb = modal_match.group(2)
            if verb in self.action_verbs:
                words = cleaned.split()
                if len(words) >= 10:
                    if not re.search(r'recommendations?\s+(?:should|must|shall)', text_lower):
                        return True, 0.80, 'modal_verb', verb
        
        # Imperative starting with action verb
        first_word = cleaned.split()[0].lower() if cleaned.split() else ''
        if first_word in self.action_verbs:
            if len(cleaned.split()) >= 10:
                return True, 0.75, 'imperative', first_word
        
        return False, 0.0, 'none', 'unknown'
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.75) -> List[Dict]:
        """Extract recommendations from text."""
        recommendations = []
        
        if not text or not text.strip():
            logger.warning("Empty text passed to extract_recommendations")
            return []
        
        logger.info(f"Extracting recommendations from text of length {len(text)}")
        
        cleaned_full_text = self.clean_text(text)
        
        # =======================================================================
        # PHASE 1: Try HSSIB 2023+ format first (Safety recommendation R/YYYY/NNN)
        # =======================================================================
        hsib_2023_pattern = re.compile(
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        hsib_2023_matches = list(hsib_2023_pattern.finditer(cleaned_full_text))
        
        if hsib_2023_matches:
            logger.info(f"Found {len(hsib_2023_matches)} HSSIB 2023+ recommendations")
            recommendations = self._extract_from_matches(
                cleaned_full_text, hsib_2023_matches, 'hsib_2023', min_confidence
            )
            if recommendations:
                # v3.3: Use rec_number deduplication for HSIB
                return self._deduplicate(recommendations, by_rec_number=True)
        
        # =======================================================================
        # PHASE 2: Try HSIB 2018 format (Recommendation YYYY/NNN)
        # =======================================================================
        hsib_2018_pattern = re.compile(
            r'Recommendation\s+(\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        hsib_2018_matches = list(hsib_2018_pattern.finditer(cleaned_full_text))
        
        if hsib_2018_matches:
            logger.info(f"Found {len(hsib_2018_matches)} HSIB 2018 recommendations")
            recommendations = self._extract_from_matches(
                cleaned_full_text, hsib_2018_matches, 'hsib_2018', min_confidence
            )
            if recommendations:
                # v3.3: Use rec_number deduplication for HSIB
                return self._deduplicate(recommendations, by_rec_number=True)
        
        # =======================================================================
        # PHASE 3: Standard "Recommendation N" format
        # =======================================================================
        standard_pattern = re.compile(
            r'(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b',
            re.IGNORECASE,
        )
        standard_matches = list(standard_pattern.finditer(cleaned_full_text))
        
        if standard_matches:
            logger.info(f"Found {len(standard_matches)} standard recommendation headings")
            recommendations = self._extract_from_matches(
                cleaned_full_text, standard_matches, 'standard', min_confidence
            )
            if recommendations:
                # v3.3: Also use rec_number deduplication for standard format
                return self._deduplicate(recommendations, by_rec_number=True)
        
        # =======================================================================
        # PHASE 4: Fallback to sentence extraction
        # =======================================================================
        logger.info("No numbered recommendations found, falling back to sentence extraction")
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        for idx, sentence in enumerate(sentences):
            cleaned = self.clean_text(sentence)
            
            is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=False)
            if is_garbage:
                continue
            
            if self.is_meta_recommendation(cleaned):
                continue
            
            is_rec, confidence, method, verb = self.is_valid_recommendation(cleaned, is_numbered_rec=False)
            
            if is_rec and confidence >= min_confidence:
                recommendations.append({
                    'text': cleaned,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': idx,
                    'in_section': False,
                })
        
        return self._deduplicate(recommendations, by_rec_number=False)
    
    def _extract_from_matches(
        self, 
        text: str, 
        matches: List, 
        pattern_type: str, 
        min_confidence: float
    ) -> List[Dict]:
        """Extract recommendations from regex matches with generic boundary detection."""
        recommendations = []
        
        for idx, match in enumerate(matches):
            start = match.start()
            next_rec_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            
            # Use generic boundary detection
            end = self._find_recommendation_end_generic(text, start, next_rec_pos)
            
            raw_block = text[start:end]
            cleaned = self.clean_text(raw_block)
            
            is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=True)
            if is_garbage:
                logger.debug(f"Skipping {pattern_type} rec {idx}: {reason}")
                continue
            
            is_rec, confidence, method, verb = self.is_valid_recommendation(cleaned, is_numbered_rec=True)
            
            if is_rec and confidence >= min_confidence:
                # Extract recommendation number based on pattern type
                if pattern_type == 'hsib_2023':
                    rec_num = match.group(1)  # e.g., "R/2023/220"
                elif pattern_type == 'hsib_2018':
                    rec_num = match.group(1)  # e.g., "2018/006"
                elif pattern_type == 'standard':
                    heading_num = match.group(1)
                    extra_digit = match.group(2) if match.lastindex >= 2 else None
                    if extra_digit and len(heading_num) == 1:
                        rec_num = f"{heading_num}{extra_digit}"
                    else:
                        rec_num = heading_num
                    # Normalise text
                    norm_pattern = r'^(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b'
                    cleaned = re.sub(norm_pattern, f"Recommendation {rec_num}", cleaned, count=1, flags=re.IGNORECASE)
                else:
                    rec_num = f"rec_{idx + 1}"
                
                recommendations.append({
                    'text': cleaned,
                    'verb': verb,
                    'method': method,
                    'confidence': round(confidence, 3),
                    'position': idx,
                    'in_section': True,
                    'rec_number': rec_num,
                })
        
        return recommendations
    
    def _deduplicate(self, recommendations: List[Dict], by_rec_number: bool = False) -> List[Dict]:
        """
        Remove duplicate recommendations.
        
        v3.3: Added by_rec_number parameter for HSIB documents where the same
        recommendation may appear in summary and detail sections.
        """
        if not recommendations:
            return []
        
        unique = []
        
        if by_rec_number:
            # v3.3: Deduplicate by rec_number, keeping the LONGEST version
            seen_numbers = {}
            for rec in recommendations:
                rec_num = rec.get('rec_number')
                if rec_num:
                    if rec_num not in seen_numbers:
                        seen_numbers[rec_num] = rec
                    else:
                        # Keep the longer version (more complete)
                        if len(rec.get('text', '')) > len(seen_numbers[rec_num].get('text', '')):
                            seen_numbers[rec_num] = rec
                else:
                    # No rec_number, fall back to text-based dedup
                    unique.append(rec)
            
            # Combine the deduplicated numbered recs with any non-numbered ones
            unique.extend(seen_numbers.values())
            
            # Sort by rec_number for consistent ordering
            def sort_key(r):
                num = r.get('rec_number', '')
                # Handle different formats: "1", "2018/006", "R/2023/220"
                if '/' in str(num):
                    # HSIB format - extract numeric part
                    parts = str(num).replace('R/', '').split('/')
                    try:
                        return (1, int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
                    except ValueError:
                        return (2, 0, 0)
                else:
                    try:
                        return (0, int(num), 0)
                    except (ValueError, TypeError):
                        return (2, 0, 0)
            
            unique.sort(key=sort_key)
        else:
            # Original text-based deduplication
            seen = set()
            for rec in recommendations:
                key = re.sub(r'\s+', ' ', rec['text'].lower().strip())[:150]
                if key not in seen:
                    seen.add(key)
                    unique.append(rec)
        
        logger.info(f"Found {len(unique)} recommendations")
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
