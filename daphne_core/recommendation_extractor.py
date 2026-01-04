"""
Recommendation Extractor v3.10
Extracts recommendation blocks from government and health sector documents.

v3.10 Changes:
- FIXED: "HSIB makes the following" contamination in Report 4 recs 1 & 4
- ADDED: Hard boundary at "HSIB makes the following" pattern
- IMPROVED: Better sentence-end detection before safety observation markers

v3.9 Changes (preserved):
- Report 6 contamination markers (In some locations, Inequalities continued, etc.)
- HSSIB makes the following safety observation detection

Supported Formats:
- HSIB 2018 format: "Recommendation 2018/006:", "Recommendation 2018/007:"
- HSSIB 2023+ format: "Safety recommendation R/2023/220:"
- Standard government: "Recommendation 1", "Recommendation 12"
- Niche/Independent investigation: "Recommendation 1:" with colon
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
    MAX_RECOMMENDATION_LENGTH = 2500
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
            r'(?:the\s+)?[Gg]overnment',
            r'(?:the\s+)?[Dd]epartment(?:\s+of\s+Health)?',
            r'DHSC',
            r'CQC',
            r'Care\s+Quality\s+Commission',
            r'NICE',
            r'(?:the\s+)?[Rr]eview(?:\s+team)?',
            r'(?:the\s+)?[Ii]nvestigation(?:\s+team)?',
            r'(?:the\s+)?[Ii]nquiry',
            r'(?:the\s+)?[Pp]anel',
            r'(?:the\s+)?[Cc]ommission',
            r'(?:the\s+)?[Cc]ommittee',
        ]
        
        self.modal_verbs = ['should', 'must', 'shall', 'need to', 'needs to', 'ought to']

    def extract_recommendations(self, text: str) -> List[Dict]:
        """
        Main entry point for recommendation extraction.
        Tries structured patterns first, falls back to sentence-based extraction.
        """
        if not text or len(text) < 100:
            return []
        
        logger.info(f"Extracting recommendations from text of length {len(text)}")
        
        # Try HSSIB 2023+ format first (R/2023/220 style)
        hssib_recs = self._extract_hssib_2023_recommendations(text)
        if hssib_recs:
            logger.info(f"Found {len(hssib_recs)} HSSIB 2023+ recommendations")
            return self._deduplicate_recommendations(hssib_recs)
        
        # Try HSIB 2018 format (2018/006 style)
        hsib_recs = self._extract_hsib_2018_recommendations(text)
        if hsib_recs:
            logger.info(f"Found {len(hsib_recs)} HSIB 2018 recommendations")
            return self._deduplicate_recommendations(hsib_recs)
        
        # Try standard numbered format
        numbered_recs = self._extract_numbered_recommendations(text)
        if numbered_recs:
            logger.info(f"Found {len(numbered_recs)} standard recommendation headings")
            return self._deduplicate_recommendations(numbered_recs)
        
        # Fallback to sentence-based extraction
        sentence_recs = self._extract_sentence_recommendations(text)
        logger.info(f"Found {len(sentence_recs)} sentence-based recommendations")
        return self._deduplicate_recommendations(sentence_recs)

    def _extract_hssib_2023_recommendations(self, text: str) -> List[Dict]:
        """Extract HSSIB 2023+ format recommendations (R/2023/220 style)."""
        pattern = re.compile(
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3})\s*[:\s]',
            re.IGNORECASE
        )
        
        matches = list(pattern.finditer(text))
        if not matches:
            return []
        
        recommendations = []
        for i, match in enumerate(matches):
            rec_id = match.group(1)
            start_pos = match.start()
            
            # Find end position
            if i + 1 < len(matches):
                next_pos = matches[i + 1].start()
            else:
                next_pos = len(text)
            
            # Find actual end using boundary detection
            end_pos = self._find_recommendation_end(text, start_pos, next_pos)
            
            rec_text = text[start_pos:end_pos].strip()
            rec_text = self._clean_recommendation_text(rec_text)
            
            if len(rec_text) >= self.MIN_RECOMMENDATION_LENGTH:
                recommendations.append({
                    'rec_number': rec_id,
                    'text': rec_text,
                    'confidence': 0.98,
                    'extraction_method': 'hssib_2023'
                })
        
        return recommendations

    def _extract_hsib_2018_recommendations(self, text: str) -> List[Dict]:
        """Extract HSIB 2018 format recommendations (2018/006 style)."""
        pattern = re.compile(
            r'Recommendation\s+(\d{4}/\d{3})\s*[:\s]',
            re.IGNORECASE
        )
        
        matches = list(pattern.finditer(text))
        if not matches:
            return []
        
        recommendations = []
        for i, match in enumerate(matches):
            rec_id = match.group(1)
            start_pos = match.start()
            
            if i + 1 < len(matches):
                next_pos = matches[i + 1].start()
            else:
                next_pos = len(text)
            
            end_pos = self._find_recommendation_end(text, start_pos, next_pos)
            
            rec_text = text[start_pos:end_pos].strip()
            rec_text = self._clean_recommendation_text(rec_text)
            
            if len(rec_text) >= self.MIN_RECOMMENDATION_LENGTH:
                recommendations.append({
                    'rec_number': rec_id,
                    'text': rec_text,
                    'confidence': 0.98,
                    'extraction_method': 'hsib_2018'
                })
        
        return recommendations

    def _extract_numbered_recommendations(self, text: str) -> List[Dict]:
        """Extract standard numbered recommendations."""
        # Pattern for "Recommendation N" or "Recommendation N:"
        pattern = re.compile(
            r'Recommendation\s+(\d{1,2})(?:\s+\d)?[:\s]+',
            re.IGNORECASE
        )
        
        matches = list(pattern.finditer(text))
        if not matches:
            return []
        
        recommendations = []
        for i, match in enumerate(matches):
            rec_num = match.group(1)
            start_pos = match.start()
            
            if i + 1 < len(matches):
                next_pos = matches[i + 1].start()
            else:
                next_pos = len(text)
            
            end_pos = self._find_recommendation_end(text, start_pos, next_pos)
            
            rec_text = text[start_pos:end_pos].strip()
            rec_text = self._clean_recommendation_text(rec_text)
            
            if len(rec_text) >= self.MIN_RECOMMENDATION_LENGTH:
                recommendations.append({
                    'rec_number': rec_num,
                    'text': rec_text,
                    'confidence': 0.98,
                    'extraction_method': 'numbered'
                })
        
        return recommendations

    def _find_recommendation_end(self, text: str, start_pos: int, next_rec_pos: int) -> int:
        """
        Find the end of a recommendation using boundary detection.
        Uses structural heuristics rather than document-specific patterns.
        
        v3.10: Added "HSIB makes the following" as hard stop
        """
        if next_rec_pos < len(text):
            search_limit = next_rec_pos
        else:
            search_limit = len(text)
        
        search_start = start_pos + 50
        
        # v3.1 FIX: Check for inline "Recommendation N" patterns
        inline_rec_pattern = re.compile(
            r'\bRecommendation\s+\d{1,2}(?:\s+\d)?[:\s]+(?:All|The|NHS|Every|Provider|Trust|Board|ICS|CQC|DHSC|Ward|More|Professional|Except|TEWV|Middlesbrough)',
            re.IGNORECASE
        )
        inline_match = inline_rec_pattern.search(text[search_start:search_limit])
        if inline_match:
            potential_end = search_start + inline_match.start()
            if potential_end < search_limit:
                logger.debug(f"Found inline recommendation boundary at position {potential_end}")
                search_limit = potential_end
        
        # =====================================================================
        # v3.10: HARD STOP PATTERNS - These ALWAYS end recommendations
        # =====================================================================
        hard_stop_patterns = [
            # v3.10 FIX: "HSIB makes the following" - causes contamination in Report 4
            r'\bHSIB\s+makes\s+the\s+following\b',
            r'\bHSSIB\s+makes\s+the\s+following\b',
            
            # Safety observations (different from recommendations)
            r'\bSafety\s+observation\s+[A-Z]/\d{4}/\d{3}\b',
            r'\bHSSIB\s+makes\s+the\s+following\s+safety\s+observation\b',
            
            # Document structure
            r'\bWhy\'?s?\s+diagram\b',
            r'\bFigure\s+\d+[:\s]',
            r'\bTable\s+\d+[:\s]',
            
            # v3.7 FIX: PDF merge issues
            r'(?:to|\.)\s*While\s+(?:national|there\s+is)',
            r'(?:to|\.)\s*Current\s+research\s+only',
            
            # v3.9 FIX: HSIB finding paragraph starters
            r'\b\d+\.\d+\.\d+\s+(?:The|A|In|Staff|Subject|Mental|While|Current)',
            r'\b\d+\.\d+\s+(?:What|The|How|Why|Where|When|Impact|Capability|Awareness)',
            r'(?<=[.!?])\s+While\s+national\s+guidance',
            r'(?<=[.!?])\s+Current\s+research\s+only',
            r'(?<=[.!?])\s+The\s+investigation\s+(?:was|visited|observed|found|\'s)',
            r'(?<=[.!?])\s+Staff\s+(?:told|said|reported|also)',
            r'(?<=[.!?])\s+Subject\s+matter\s+advisors',
            r'(?<=[.!?])\s+Mental\s+health\s+practitioners\s+told',
            r'\bHSSIB\s+proposes\s+the\s+following\b',
            r'(?<=[.!?])\s+In\s+the\s+following\s+sections',
            r'(?<=[.!?])\s+New\s+builds\s+and',
            r'(?<=[.!?])\s+Infrastructure\s*[-–]',
            
            # v3.9 FIX: Report 6 contamination markers
            r'(?<=[.!?])\s+In\s+some\s+locations',
            r'(?<=[.!?])\s+Inequalities\s+continued',
            r'(?<=[.!?])\s+Some\s+organisational\s+cultures',
            r'(?<=[.!?])\s+Availability\s+and\s+access',
            r'\bBuilt\s+mental\s+health\s+inpatient\s+environments\b',
            r'\bSocial\s+and\s+organisational\s+factors\b',
            r'(?<=[.!?])\s+This\s+reduced\s+continuity',
        ]
        
        # Universal section markers
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
            r'\bSafety\s+[Oo]bservation(?:s)?(?:\s+[A-Z0-9/]+)?[:\s]',
            r'\bLocal-level\s+learning\b',
            r'\bBackground\s+and\s+context\b',
            r'\bFindings\s+and\s+analysis\b',
            r'\bSummary\s+of\s+(?:findings|recommendations)\b',
            r'\bThe\s+investigation\s+(?:found|makes|identified)\b',
            r'\bProposed\s+(?:ICB|safety)\s+(?:response|action)\b',
            r'(?<=[.!?])\s+\d+\.\d+\.\d+\s+[A-Z]',
            r'(?<=[.!?])\s+\d+\.\d+\s+[A-Z][a-z]',
            r'\n\s*\d+\.\d+\.\d+\s+[A-Z]',
            r'\n\s*\d+\.\d+\s+[A-Z][a-z]+',
            r'\bLessons\s+(?:learned|learnt)\b',
            r'\bGood\s+practice\b',
            r'\bAreas?\s+(?:of|for)\s+(?:good\s+practice|improvement)\b',
        ]
        
        # Header-only markers
        header_only_markers = [
            r'(?:^|[.!?]\s+)Our\s+vision\s+for\s+(?:a\s+)?(?:better\s+)?future\b',
            r'(?:^|[.!?]\s+)Key\s+facts\b',
            r'(?:^|[.!?]\s+)Methodology\b',
            r'(?:^|[.!?]\s+)Findings\b(?!\s+and\s+analysis)',
            r'(?:^|[.!?]\s+)Case\s+studies\b',
            r'(?:^|[.!?]\s+)Conclusion(?:s)?(?:\s+and\s+next\s+steps)?\b',
            r'(?:^|[.!?]\s+)This\s+section\s+(?:considers|explores|examines|discusses|looks\s+at)\b',
            r'(?:^|[.!?]\s+)The\s+investigation(?:\'s)?\s+(?:found|visited|observed|focus)\b',
            r'(?:^|[.!?]\s+)(?:Staff|Interviewees|Those\s+interviewed)\s+(?:told|said|reported)\b',
            r'(?:^|[.!?]\s+)Hospital\s+environment\b',
            r'(?:^|[.!?]\s+)The\s+reference\s+investigation\b',
            r'(?:^|[.!?]\s+)New\s+builds\s+and\s+the\s+New\s+Hospital\s+Programme\b',
            r'(?:^|[.!?]\s+)Infrastructure\s+-\s+digital\b',
            r'(?:^|[.!?]\s+)Awareness\s+of\s+women\'?s?\s+health\b',
            r'(?:^|[.!?]\s+)Capability\s+to\s+provide\b',
            r'(?:^|[.!?]\s+)The\s+first\s+episode\s+of\s+psychosis\s+pathway\b',
            r'(?:^|[.!?]\s+)Mental\s+health\s+practitioners\s+told\b',
            r'(?:^|[.!?]\s+)HSSIB\s+proposes\s+the\s+following\s+safety\b',
            r'(?:^|[.!?]\s+)Impact\s+of\s+built\s+environments\b',
        ]
        
        earliest_boundary = search_limit
        
        # Check hard stop patterns first (highest priority)
        for pattern in hard_stop_patterns:
            match = re.search(pattern, text[search_start:search_limit], re.IGNORECASE)
            if match:
                boundary = search_start + match.start()
                if boundary < earliest_boundary:
                    earliest_boundary = boundary
                    logger.debug(f"Hard stop boundary found: {pattern[:30]}...")
        
        # Check universal markers
        for pattern in universal_markers:
            match = re.search(pattern, text[search_start:search_limit], re.IGNORECASE)
            if match:
                boundary = search_start + match.start()
                if boundary < earliest_boundary:
                    earliest_boundary = boundary
        
        # Check header-only markers
        for pattern in header_only_markers:
            match = re.search(pattern, text[search_start:search_limit], re.IGNORECASE)
            if match:
                matched_text = match.group()
                header_offset = 0
                for i, c in enumerate(matched_text):
                    if c.isalpha():
                        header_offset = i
                        break
                boundary = search_start + match.start() + header_offset
                if boundary < earliest_boundary:
                    earliest_boundary = boundary
        
        # Check for structural boundaries
        check_positions = range(search_start, min(earliest_boundary, search_start + 3000), 100)
        for pos in check_positions:
            if self._is_section_boundary(text, pos):
                if pos < earliest_boundary:
                    earliest_boundary = pos
                break
        
        # Apply maximum length safeguard
        max_end = start_pos + self.MAX_RECOMMENDATION_LENGTH
        if earliest_boundary > max_end:
            text_chunk = text[start_pos:max_end]
            last_period = text_chunk.rfind('. ')
            if last_period > self.MIN_RECOMMENDATION_LENGTH:
                earliest_boundary = start_pos + last_period + 1
            else:
                earliest_boundary = max_end
        
        return earliest_boundary

    def _is_section_boundary(self, text: str, position: int) -> bool:
        """
        Check if a position appears to be at a major section break.
        
        v3.10: Added HSIB/HSSIB-specific boundary patterns
        """
        if position >= len(text):
            return True
        
        remaining = text[position:position + 300]
        remaining_stripped = remaining.lstrip()
        
        # Pattern: Numbered paragraphs like "4.1.33 The latest"
        if re.match(r'^\d+\.\d+\.\d+\s', remaining_stripped):
            return True
        
        # Pattern: Section headers like "4.2 What competencies"
        if re.match(r'^\d+\.\d+\s+[A-Z]', remaining_stripped):
            return True
        
        # Pattern: HSSIB safety response separator
        if remaining_stripped.startswith('HSSIB proposes'):
            return True
        
        # v3.10: HSIB makes the following
        if re.match(r'^HSIB\s+makes\s+the\s+following', remaining_stripped, re.IGNORECASE):
            return True
        if re.match(r'^HSSIB\s+makes\s+the\s+following', remaining_stripped, re.IGNORECASE):
            return True
        
        # Common HSIB finding starters
        hsib_finding_starters = [
            'While national guidance',
            'Current research only',
            'The investigation',
            'During the investigation',
            'In the following sections',
            'This section considers',
            'This section explores',
            'Staff told the investigation',
            'Subject matter advisors',
            'Mental health practitioners told',
        ]
        for starter in hsib_finding_starters:
            if remaining_stripped.startswith(starter):
                return True
        
        # Check for numbered subsection headers
        if re.match(r'^\s*\d+\.\d+(?:\.\d+)?\s+[A-Z]', remaining):
            return True
        
        # Check for numbered section headers
        if re.match(r'^\s*\d+\.?\d*\s+[A-Z][a-z]+(?:\s+[a-z]+){0,5}\s*(?:\n|$)', remaining):
            return True
        
        return False

    def _clean_recommendation_text(self, text: str) -> str:
        """Clean extracted recommendation text."""
        # Remove multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # v3.10: Remove "HSIB makes the following" if it appears at the end
        text = re.sub(r'\s*HSIB\s+makes\s+the\s+following\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*HSSIB\s+makes\s+the\s+following\s*$', '', text, flags=re.IGNORECASE)
        
        # Remove trailing partial sentences (no ending punctuation)
        text = text.strip()
        if text and text[-1] not in '.!?':
            last_sentence_end = max(text.rfind('. '), text.rfind('! '), text.rfind('? '))
            if last_sentence_end > len(text) * 0.5:
                text = text[:last_sentence_end + 1]
        
        return text.strip()

    def _extract_sentence_recommendations(self, text: str) -> List[Dict]:
        """Fallback: extract recommendations from sentences with modal verbs."""
        recommendations = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 50 or len(sentence) > self.MAX_SENTENCE_LENGTH:
                continue
            
            # Check for modal verbs
            sentence_lower = sentence.lower()
            has_modal = any(modal in sentence_lower for modal in self.modal_verbs)
            
            if not has_modal:
                continue
            
            # Check for recommending entity
            has_entity = any(
                re.search(entity, sentence, re.IGNORECASE)
                for entity in self.recommending_entities
            )
            
            # Check for action verb
            words = sentence_lower.split()
            has_action = any(word in self.action_verbs for word in words)
            
            if has_modal and (has_entity or has_action):
                recommendations.append({
                    'rec_number': str(len(recommendations) + 1),
                    'text': sentence,
                    'confidence': 0.75,
                    'extraction_method': 'sentence'
                })
        
        return recommendations

    def _deduplicate_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations, keeping the longest version."""
        if not recommendations:
            return []
        
        # Group by rec_number
        by_number = {}
        for rec in recommendations:
            num = rec.get('rec_number', '')
            if num not in by_number:
                by_number[num] = []
            by_number[num].append(rec)
        
        # Keep longest for each number
        result = []
        for num, recs in by_number.items():
            longest = max(recs, key=lambda r: len(r.get('text', '')))
            result.append(longest)
        
        # Sort by rec_number
        def sort_key(r):
            num = r.get('rec_number', '0')
            # Handle R/2023/220 format
            if '/' in str(num):
                parts = str(num).replace('R/', '').split('/')
                return (int(parts[0]) if parts[0].isdigit() else 0,
                        int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0)
            return (0, int(num) if str(num).isdigit() else 0)
        
        result.sort(key=sort_key)
        return result


# Module-level function for backwards compatibility
def extract_recommendations(text: str, min_confidence: float = 0.0) -> List[Dict]:
    """Extract recommendations from text."""
    extractor = StrictRecommendationExtractor()
    recs = extractor.extract_recommendations(text)
    return [r for r in recs if r.get('confidence', 0) >= min_confidence]


# For backwards compatibility
logger.info("Using StrictRecommendationExtractor (improved filtering)")
