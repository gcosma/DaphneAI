"""
Recommendation Extractor v3.2
Extracts recommendation blocks from government and health sector documents.

v3.2 Changes:
- FIXED: Deduplication now uses rec_number for HSIB documents (prevents duplicates from summary/detail sections)
- FIXED: Section marker boundary detection now catches "N.N." patterns (e.g., "6.3. Safety Observations")
- FIXED: Strips preamble text before "That" in HSIB recommendations
- IMPROVED: Better handling of footnote markers in recommendation text

v3.1 Changes:
- Boundary detection catches inline "Recommendation N" patterns
- Handles "Recommendation 1 1" (spaced digits) correctly as boundaries
- Prevents Recommendation N from bleeding into Recommendation N-1

Supported Formats:
- HSIB 2018 format: "Recommendation 2018/006:", "Recommendation 2018/007:"
- HSSIB 2023+ format: "Safety recommendation R/2023/220:"
- Standard government: "Recommendation 1", "Recommendation 12"
- Sentence-based fallback for unstructured documents

Usage:
    from recommendation_extractor import extract_recommendations
    
    recommendations = extract_recommendations(document_text, min_confidence=0.75)
    for rec in recommendations:
        print(f"{rec['rec_number']}: {rec['text'][:100]}...")
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
    MAX_RECOMMENDATION_LENGTH = 1500  # Single recommendation should not exceed this
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
            'extend', 'extends',
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
            ''': "'", ''': "'", '"': '"', '"': '"',
            '–': '-', '—': '-',
        }
        
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean text while preserving important content"""
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
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?[^A-Z]*(?=[A-Z]|$)', ' ', text, flags=re.IGNORECASE)
        
        # Remove page numbers (but NOT HSIB recommendation numbers like 2018/006)
        # Page numbers are typically short like 1/84, 12/84
        text = re.sub(r'\b(\d{1,2})/(\d{1,3})\b(?!\d)', '', text)
        text = re.sub(r'\bPage\s+\d+\s+(?:of\s+\d+)?', '', text, flags=re.IGNORECASE)
        
        # Remove GOV.UK footer artifacts
        text = re.sub(
            r'Rapid review into data on mental health inpatient settings:.*?GOV\.?UK',
            '',
            text,
            flags=re.IGNORECASE,
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
        
        # Remove "UK " prefix at start
        text = re.sub(r'^UK\s+', '', text)
        
        # v3.2: Remove footnote markers like [2], [27] from within text
        # But preserve HSIB numbers in brackets
        text = re.sub(r'\[(\d{1,2})\](?!\d{3})', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _clean_hsib_recommendation_text(self, text: str) -> str:
        """
        v3.2: Additional cleaning specifically for HSIB recommendations.
        Strips preamble text and section markers from the recommendation body.
        """
        if not text:
            return ""
        
        # First, do standard cleaning
        cleaned = self.clean_text(text)
        
        # v3.2: For HSIB recommendations, the actual recommendation usually starts with "That"
        # Strip any preamble text that comes before the recommendation header
        # Pattern: "Recommendation YYYY/NNN: <preamble> That NHS England..."
        # We want: "Recommendation YYYY/NNN: That NHS England..."
        
        # Check if this is an HSIB recommendation
        hsib_match = re.match(r'^(Recommendation\s+\d{4}/\d{3}[:\s]+)', cleaned, re.IGNORECASE)
        if hsib_match:
            header = hsib_match.group(1)
            body = cleaned[len(header):].strip()
            
            # Look for "That" which typically starts the actual recommendation
            that_match = re.search(r'\bThat\s+(?:NHS|the\s+Care|CQC)', body, re.IGNORECASE)
            if that_match and that_match.start() > 0 and that_match.start() < 200:
                # There's preamble text before "That" - strip it
                body = body[that_match.start():]
                cleaned = header + body
        
        # v3.2: Remove section markers that got captured at the end
        # Pattern: "6.3. Safety Observations" or "6.3 Safety Observations"
        cleaned = re.sub(r'\s+\d+\.\d+\.?\s+[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\s*$', '', cleaned)
        
        # Remove trailing section headers
        section_end_patterns = [
            r'\s+Safety\s+Observations?\s*$',
            r'\s+Local-level\s+learning\s*$',
            r'\s+Background\s+and\s+context\s*$',
            r'\s+Summary\s*$',
        ]
        for pattern in section_end_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def is_garbage(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, str]:
        """First-pass filter to reject obvious garbage"""
        if not text:
            return True, "empty"
        
        cleaned = self.clean_text(text)
        
        if len(cleaned) < 40:
            return True, "too_short"
        
        if not is_numbered_rec:
            if len(cleaned) > self.MAX_SENTENCE_LENGTH:
                return True, "too_long"
        
        # Section headers
        if re.match(r'^(?:Appendix|Section|Chapter|Table|Figure|Footnote)\s+\d+', cleaned, re.IGNORECASE):
            return True, "header"
        
        # Too many special characters
        special_chars = sum(1 for c in cleaned if not c.isalnum() and not c.isspace() and c not in '.,;:!?\'"-()[]/')
        if len(cleaned) > 0 and special_chars / len(cleaned) > 0.12:
            return True, "corrupted"
        
        # Excessive numbers (but be more lenient for numbered recs which may have dates)
        if not is_numbered_rec:
            digits = sum(1 for c in cleaned if c.isdigit())
            if len(cleaned) > 0 and digits / len(cleaned) > 0.15:
                return True, "too_many_numbers"
        
        return False, ""
    
    def is_meta_recommendation(self, text: str) -> bool:
        """Check if text is talking ABOUT recommendations rather than making one"""
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
            r'^our\s+objectives\s+were\s+to',
            r'^we\s+(?:were|are)\s+(?:told|informed|advised)',
            r'^this\s+document\s+outlines',
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _is_section_boundary(self, remaining: str) -> bool:
        """Check if remaining text starts with a section boundary"""
        if not remaining:
            return False
        
        remaining_stripped = remaining.lstrip()
        
        # Universal section markers
        section_markers = [
            r'^(?:Appendix|Annex|Schedule)\s+[A-Z0-9]',
            r'^(?:Chapter|Section|Part)\s+\d',
            r'^References\b',
            r'^Glossary\b',
            r'^Acknowledgements?\b',
            r'^Bibliography\b',
            r'^Endnotes?\b',
            r'^About\s+(?:this|the)\s+(?:report|review)',
            r'^Safety\s+[Oo]bservation(?:s)?',
            r'^Local-level\s+learning',
            r'^Background\s+and\s+context',
            # v3.2: Section numbers like "6.3. Safety Observations"
            r'^\d+\.\d+\.?\s+[A-Z]',
        ]
        
        for pattern in section_markers:
            if re.match(pattern, remaining_stripped, re.IGNORECASE):
                return True
        
        # Title case headers on their own line
        lines = remaining.split('\n')
        if lines:
            first_line = lines[0].strip()
            if (len(first_line) > 5 and len(first_line) < 80 and 
                first_line[0].isupper() and
                not first_line.endswith('.') and
                not first_line.startswith(('The ', 'A ', 'An ', 'This ', 'That ', 'It '))):
                words = first_line.split()
                if len(words) >= 2:
                    caps = sum(1 for w in words if w[0].isupper() or w.lower() in ['and', 'the', 'of', 'for', 'to', 'in', 'a', 'an'])
                    if caps >= len(words) * 0.7:
                        return True
        
        return False
    
    def _find_recommendation_end_generic(self, text: str, start_pos: int, next_rec_pos: int) -> int:
        """
        GENERIC end-of-recommendation detection.
        Uses structural heuristics rather than document-specific patterns.
        """
        if next_rec_pos < len(text):
            search_limit = next_rec_pos
        else:
            search_limit = len(text)
        
        search_start = start_pos + 50
        
        # Check for inline "Recommendation N" patterns
        inline_rec_pattern = re.compile(
            r'\bRecommendation\s+\d{1,2}(?:\s+\d)?\s+(?:All|The|NHS|Every|Provider|Trust|Board|ICS|CQC|DHSC|Ward|More|Professional|Except)',
            re.IGNORECASE
        )
        inline_match = inline_rec_pattern.search(text[search_start:search_limit])
        if inline_match:
            potential_end = search_start + inline_match.start()
            if potential_end < search_limit:
                logger.debug(f"Found inline recommendation boundary at position {potential_end}")
                search_limit = potential_end
        
        # Universal end markers
        universal_markers = [
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
            # Section transitions
            r'\bSafety\s+[Oo]bservation(?:s)?(?:\s+[A-Z0-9/]+)?[:\s]',
            r'\bLocal-level\s+learning\b',
            r'\bBackground\s+and\s+context\b',
            r'\bFindings\s+and\s+analysis\b',
            r'\bSummary\s+of\s+(?:findings|recommendations)\b',
            r'\bThe\s+investigation\s+(?:found|makes|identified)\b',
            # v3.2: Section numbers like "6.3. Safety Observations"
            r'\b\d+\.\d+\.?\s+Safety\s+Observation',
            # Numbered sections
            r'\n\s*\d+\.?\s+[A-Z][a-z]+(?:\s+[a-z]+){0,3}\s*\n',
        ]
        
        earliest_boundary = search_limit
        
        for pattern in universal_markers:
            try:
                match = re.search(pattern, text[search_start:search_limit], re.IGNORECASE)
                if match:
                    boundary = search_start + match.start()
                    if boundary < earliest_boundary:
                        earliest_boundary = boundary
            except re.error:
                continue
        
        # Check for section boundaries using heuristics
        chunk_size = 200
        pos = search_start
        while pos < min(earliest_boundary, search_limit):
            chunk = text[pos:pos + chunk_size]
            if self._is_section_boundary(chunk):
                if pos < earliest_boundary:
                    earliest_boundary = pos
                break
            pos += chunk_size // 2
        
        # Length sanity check
        if earliest_boundary - start_pos > self.MAX_RECOMMENDATION_LENGTH:
            sentence_end = text.find('.', start_pos + self.MIN_RECOMMENDATION_LENGTH, start_pos + self.MAX_RECOMMENDATION_LENGTH)
            if sentence_end != -1:
                earliest_boundary = sentence_end + 1
        
        return earliest_boundary
    
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
        
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}s?\b', text_lower):
                return verb
        
        return 'unknown'
    
    def is_valid_recommendation(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, float, str, str]:
        """Determine if text is a genuine recommendation"""
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
                # v3.2: Use rec_number deduplication for HSIB
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
            
            # v3.2: Use HSIB-specific cleaning for HSIB documents
            if pattern_type in ('hsib_2018', 'hsib_2023'):
                cleaned = self._clean_hsib_recommendation_text(raw_block)
            else:
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
        
        v3.2: Added by_rec_number parameter for HSIB documents where the same
        recommendation may appear in summary and detail sections.
        """
        if not recommendations:
            return []
        
        unique = []
        
        if by_rec_number:
            # v3.2: Deduplicate by rec_number, keeping the LONGEST version
            # (the detailed version is usually longer than the summary)
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
            
            # Sort by position to maintain order
            unique.sort(key=lambda x: x.get('position', 0))
        else:
            # Original text-based deduplication
            seen = set()
            for rec in recommendations:
                key = re.sub(r'\s+', ' ', rec['text'].lower().strip())[:150]
                if key not in seen:
                    seen.add(key)
                    unique.append(rec)
        
        logger.info(f"After deduplication: {len(unique)} recommendations (from {len(recommendations)})")
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
