"""
Strict Recommendation Extractor v3.1
Properly extracts full recommendation blocks from government documents.
Supports multiple report formats including standard government and HSIB.

v3.1 Changes:
- Added HSIB format support (Recommendation YYYY/NNN)
- Added Safety Recommendation pattern support
- Pattern priority system (specific → general)
- Improved section boundary detection
- Falls back to sentence-based extraction for other document types

Supported Formats:
- Standard government: "Recommendation 1", "Recommendation 12"
- HSIB format: "Recommendation 2018/006", "Recommendation 2018/007"
- Safety Recommendations: "Safety Recommendation to NHS England:"
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
    Extract genuine recommendations with aggressive pre-filtering.
    Handles government-style, HSIB-style, and general documents.
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
            'come', 'form', 'work', 'learn', 'drive', 'require', 'requires',
        }
        
        # Entities that receive recommendations in government docs
        self.recommending_entities = [
            r'NHS\s+England',
            r'NHS\s+Improvement',
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
            r'Clinical\s+Commissioning\s+Groups?',
            r'CCGs?',
        ]
        
        # =====================================================================
        # PATTERN DEFINITIONS - Order matters! More specific patterns first.
        # =====================================================================
        
        # HSIB 2018 format: "Recommendation 2018/006:" or "Recommendation 2018/006"
        self.hsib_2018_pattern = re.compile(
            r'Recommendation\s+(\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        
        # HSIB 2023+ format: "Safety recommendation R/2023/220:" 
        self.hsib_2023_pattern = re.compile(
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        
        # HSSIB Safety Observation format: "Safety observation O/2024/037:"
        self.safety_observation_pattern = re.compile(
            r'Safety\s+observation\s+(O/\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        
        # Generic Safety Recommendation (without specific ID)
        self.safety_rec_generic_pattern = re.compile(
            r'HSIB\s+(?:makes\s+the\s+following\s+)?[Ss]afety\s+[Rr]ecommendation[:\s]',
            re.IGNORECASE
        )
        
        # Standard government format: "Recommendation 1", "Recommendation 12"
        self.standard_heading_pattern = re.compile(
            r'(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b',
            re.IGNORECASE
        )
    
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
        
        # Remove timestamps
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?[^A-Z]*(?=[A-Z]|$)', ' ', text, flags=re.IGNORECASE)
        
        # Remove page numbers (but NOT HSIB recommendation numbers like 2018/006)
        # Page numbers are typically short like 1/84, 12/84, not 2018/006
        text = re.sub(r'\b(\d{1,2})/(\d{1,3})\b', '', text)  # Only match short page numbers
        text = re.sub(r'\bPage\s+\d+\s+(?:of\s+\d+)?', '', text, flags=re.IGNORECASE)
        
        # Remove GOV.UK footer artifacts
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
        """FIRST-PASS filter: reject obvious garbage before any analysis."""
        if not text:
            return True, "empty"
        
        cleaned = self.clean_text(text)
        
        if len(cleaned) < 40:
            return True, "too_short"
        
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
        
        # Excessive numbers (only for sentence-based)
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
    
    def _extract_verb(self, text: str) -> str:
        """Extract the main action verb from recommendation text"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        # Check for "should [verb]" pattern
        should_match = re.search(r'\bshould\s+(?:also\s+)?(?:urgently\s+)?(?:be\s+)?(\w+)', text_lower)
        if should_match:
            verb = should_match.group(1)
            if verb in self.action_verbs:
                return verb
        
        # Check for entity patterns like "NHS England requires..."
        for entity in self.recommending_entities:
            entity_verb_match = re.search(rf'{entity}\s+(\w+)', text_lower, re.IGNORECASE)
            if entity_verb_match:
                verb = entity_verb_match.group(1).lower()
                if verb in self.action_verbs:
                    return verb
        
        # Look for any action verb
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}\b', text_lower):
                return verb
        
        return 'unknown'
    
    def _find_recommendation_end(self, text: str, start_pos: int, next_rec_pos: int, pattern_type: str) -> int:
        """
        Find the proper end position for a recommendation.
        Handles different document structures.
        """
        # If there's a next recommendation, that's our boundary
        if next_rec_pos < len(text):
            return next_rec_pos
        
        # Section markers that indicate end of recommendations
        section_markers = [
            r'\bOur vision for (?:a |the )?(?:better )?future\b',
            r'\bKey facts\b',
            r'\bMethodology\b',
            r'\bFindings\b',
            r'\bAppendix(?:es)?(?:\s+\d+)?\b',
            r'\bGlossary\b',
            r'\bReferences\b',
            r'\bCase studies\b',
            r'\bConclusion(?:s)?(?:\s+and\s+next\s+steps)?\b',
            r'\bAcknowledgements?\b',
            r'\bAbout (?:this|the) (?:report|review)\b',
            r'\bData mapping\b',
            r'\bSafety issues framework\b',
            r'\bMeasuring what matters:\b',
            r'\bPoor safety outcomes\b',
            r'\bAll content is available under\b',
            r'©\s*Crown copyright',
            r'\bThroughout the review\b',
            # HSIB-specific markers
            r'\bSafety\s+Observation(?:s)?\b',  # Safety Observations come after Safety Recommendations
            r'\b\d+\s+Background\s+and\s+context\b',
            r'\bEndnotes\b',
            r'\bProviding feedback\b',
        ]
        
        earliest_boundary = next_rec_pos
        search_start = start_pos + 30  # Skip past recommendation header
        
        for pattern in section_markers:
            match = re.search(pattern, text[search_start:], re.IGNORECASE)
            if match:
                boundary = search_start + match.start()
                if boundary < earliest_boundary:
                    earliest_boundary = boundary
        
        return earliest_boundary
    
    def is_genuine_recommendation(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, float, str, str]:
        """Determine if text is a genuine recommendation."""
        if not text:
            return False, 0.0, 'empty', 'unknown'
        
        cleaned = self.clean_text(text)
        if not cleaned:
            return False, 0.0, 'empty', 'unknown'
        
        text_lower = cleaned.lower()
        
        # Length checks for sentence-based extraction
        if not is_numbered_rec:
            if len(cleaned) > self.MAX_SENTENCE_LENGTH:
                return False, 0.0, 'too_long', 'none'
        
        # Method 1a: HSIB 2023+ format (e.g., "Safety recommendation R/2023/220")
        hsib_2023_match = re.match(
            r'^Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]+(.+)',
            cleaned,
            re.IGNORECASE,
        )
        if hsib_2023_match:
            rec_num = hsib_2023_match.group(1)
            rec_text = hsib_2023_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.98, f'hsib_2023_{rec_num}', verb
        
        # Method 1b: HSIB 2018 format (e.g., "Recommendation 2018/006")
        hsib_2018_match = re.match(
            r'^Recommendation\s+(\d{4}/\d{3})[:\s]+(.+)',
            cleaned,
            re.IGNORECASE,
        )
        if hsib_2018_match:
            rec_num = hsib_2018_match.group(1)
            rec_text = hsib_2018_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.98, f'hsib_2018_{rec_num}', verb
        
        # Method 1c: Safety Observation format (e.g., "Safety observation O/2024/037")
        safety_obs_match = re.match(
            r'^Safety\s+observation\s+(O/\d{4}/\d{3})[:\s]+(.+)',
            cleaned,
            re.IGNORECASE,
        )
        if safety_obs_match:
            rec_num = safety_obs_match.group(1)
            rec_text = safety_obs_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.95, f'safety_observation_{rec_num}', verb
        
        # Method 1d: Standard numbered recommendation (e.g., "Recommendation 1")
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
        
        # Method 1c: Safety Recommendation (HSIB format)
        safety_match = re.match(
            r'^Safety\s+Recommendation(?:\s+to\s+[^:]+)?[:\s]+(.+)',
            cleaned,
            re.IGNORECASE,
        )
        if safety_match:
            rec_text = safety_match.group(1)
            verb = self._extract_verb(rec_text)
            return True, 0.96, 'safety_recommendation', verb
        
        # Method 2: "That [Entity]" pattern (common in HSIB)
        that_entity_match = re.match(
            r'^That\s+(NHS\s+England|the\s+Care\s+Quality\s+Commission|CQC|NHS\s+Improvement)',
            cleaned,
            re.IGNORECASE,
        )
        if that_entity_match:
            verb = self._extract_verb(cleaned)
            return True, 0.95, 'that_entity', verb
        
        # Method 3: Entity + should pattern
        for entity_pattern in self.recommending_entities:
            pattern = rf'{entity_pattern}\s+should\s+(?:also\s+)?(?:urgently\s+)?(?:\w+ly\s+)?(\w+)'
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match and match.group(1):
                verb = match.group(1).lower()
                if verb in self.action_verbs or verb.endswith('e') or verb.endswith('ise') or verb.endswith('ize'):
                    return True, 0.95, 'entity_should', verb
        
        # Method 4: "It is recommended that" pattern
        if re.search(r'\bit\s+is\s+recommended\s+that\b', text_lower):
            verb = self._extract_verb(cleaned)
            return True, 0.92, 'it_is_recommended', verb
        
        # Method 5: "We recommend" pattern
        if re.search(r'\bwe\s+recommend\b', text_lower):
            verb = self._extract_verb(cleaned)
            return True, 0.90, 'we_recommend', verb
        
        # Method 6: "should be" + action (passive)
        should_be_match = re.search(r'(\w+(?:\s+\w+)?)\s+should\s+be\s+(\w+ed)\b', text_lower)
        if should_be_match:
            subject = should_be_match.group(1)
            verb = should_be_match.group(2)
            if subject not in ['this', 'it', 'that', 'recommendation', 'the recommendation']:
                return True, 0.85, 'should_be_passive', verb
        
        # Method 7: Modal + action verb (stricter)
        modal_match = re.search(r'\b(should|must|shall)\s+(\w+)\b', text_lower)
        if modal_match:
            verb = modal_match.group(2)
            if verb in self.action_verbs:
                words = cleaned.split()
                if len(words) >= 10:
                    if not re.search(r'recommendations?\s+(?:should|must|shall)', text_lower):
                        return True, 0.80, 'modal_verb', verb
        
        # Method 8: Imperative starting with action verb
        first_word = cleaned.split()[0].lower() if cleaned.split() else ''
        if first_word in self.action_verbs:
            if len(cleaned.split()) >= 10:
                return True, 0.75, 'imperative', first_word
        
        return False, 0.0, 'none', 'unknown'
    
    def _detect_document_type(self, cleaned_text: str) -> Tuple[str, List]:
        """
        Detect document type and return appropriate pattern matches.
        Returns (pattern_type, matches) tuple.
        
        NOTE: cleaned_text should already be cleaned by the caller to ensure
        match positions are consistent with the text used for extraction.
        """
        # Try HSIB 2023+ format first (most specific): R/2023/220
        hsib_2023_matches = list(self.hsib_2023_pattern.finditer(cleaned_text))
        if hsib_2023_matches:
            logger.info(f"Detected HSIB 2023+ format: found {len(hsib_2023_matches)} safety recommendations")
            return 'hsib_2023', hsib_2023_matches
        
        # Try HSIB 2018 format: 2018/006
        hsib_2018_matches = list(self.hsib_2018_pattern.finditer(cleaned_text))
        if hsib_2018_matches:
            logger.info(f"Detected HSIB 2018 format: found {len(hsib_2018_matches)} recommendations")
            return 'hsib_2018', hsib_2018_matches
        
        # Try Safety Observation format: O/2024/037
        safety_obs_matches = list(self.safety_observation_pattern.finditer(cleaned_text))
        if safety_obs_matches:
            logger.info(f"Detected Safety Observation format: found {len(safety_obs_matches)} observations")
            return 'safety_observation', safety_obs_matches
        
        # Try standard government format: Recommendation 1, Recommendation 2
        standard_matches = list(self.standard_heading_pattern.finditer(cleaned_text))
        if standard_matches:
            logger.info(f"Detected standard government format: found {len(standard_matches)} recommendations")
            return 'standard', standard_matches
        
        # No structured format found
        logger.info("No structured recommendation format detected, will use sentence fallback")
        return 'none', []
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.75) -> List[Dict]:
        """Extract genuine recommendations from text."""
        recommendations = []
        
        if not text or not text.strip():
            logger.warning("Empty text passed to extract_recommendations")
            return []
        
        logger.info(f"Extracting recommendations from text of length {len(text)}")
        
        # =======================================================================
        # PHASE 1: Detect document type and extract structured recommendations
        # =======================================================================
        
        # Clean text ONCE and use it for both detection and extraction
        cleaned_full_text = self.clean_text(text)
        
        pattern_type, heading_matches = self._detect_document_type(cleaned_full_text)
        
        if pattern_type != 'none' and heading_matches:
            logger.info(f"Processing {len(heading_matches)} {pattern_type} recommendations")
            
            for idx, match in enumerate(heading_matches):
                start = match.start()
                next_rec_pos = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(cleaned_full_text)
                end = self._find_recommendation_end(cleaned_full_text, start, next_rec_pos, pattern_type)
                
                raw_block = cleaned_full_text[start:end]
                cleaned = self.clean_text(raw_block)
                
                # Check garbage with numbered_rec flag
                is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=True)
                if is_garbage:
                    logger.debug(f"Skipping {pattern_type} rec {idx}: {reason}")
                    continue
                
                is_rec, confidence, method, verb = self.is_genuine_recommendation(cleaned, is_numbered_rec=True)
                
                if is_rec and confidence >= min_confidence:
                    # Extract recommendation number based on pattern type
                    if pattern_type == 'hsib_2023':
                        rec_num = match.group(1)  # e.g., "R/2023/220"
                    elif pattern_type == 'hsib_2018':
                        rec_num = match.group(1)  # e.g., "2018/006"
                    elif pattern_type == 'safety_observation':
                        rec_num = match.group(1)  # e.g., "O/2024/037"
                    elif pattern_type == 'standard':
                        heading_num = match.group(1)
                        extra_digit = match.group(2) if match.lastindex >= 2 else None
                        if extra_digit and len(heading_num) == 1:
                            rec_num = f"{heading_num}{extra_digit}"
                        else:
                            rec_num = heading_num
                    else:
                        rec_num = f"rec_{idx + 1}"
                    
                    # Normalise text if needed
                    if pattern_type == 'standard':
                        norm_pattern = r'^(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b'
                        cleaned = re.sub(norm_pattern, f"Recommendation {rec_num}", cleaned, count=1, flags=re.IGNORECASE)
                    
                    recommendations.append({
                        'text': cleaned,
                        'verb': verb,
                        'method': method,
                        'confidence': round(confidence, 3),
                        'position': idx,
                        'in_section': True,
                        'rec_number': rec_num,
                        'pattern_type': pattern_type,
                    })
        
        # =======================================================================
        # PHASE 2: Validate extracted recommendations
        # =======================================================================
        
        # If we only found one oversized recommendation, it's likely a segmentation error
        if len(recommendations) == 1:
            only_rec = recommendations[0]
            if len(only_rec.get("text", "")) > self.MAX_NUMBERED_REC_LENGTH:
                logger.info(
                    f"Single oversized recommendation detected (len={len(only_rec.get('text', ''))}) – "
                    "treating as invalid and falling back to sentence extraction"
                )
                recommendations = []
        
        # =======================================================================
        # PHASE 3: Sentence-based fallback (only if no structured recs found)
        # =======================================================================
        
        if len(recommendations) == 0:
            logger.info("No structured recommendations found, falling back to sentence extraction")
            
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
                        'pattern_type': 'sentence',
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
                'pattern_types': {},
            }
        
        verb_counts = Counter(r['verb'] for r in recommendations)
        method_counts = Counter(r['method'] for r in recommendations)
        pattern_counts = Counter(r.get('pattern_type', 'unknown') for r in recommendations)
        in_section = sum(1 for r in recommendations if r.get('in_section', False))
        
        return {
            'total': len(recommendations),
            'unique_verbs': len(verb_counts),
            'methods': dict(method_counts),
            'pattern_types': dict(pattern_counts),
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


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_extractor():
    """Test the extractor with sample texts"""
    
    # Test HSIB format
    hsib_sample = """
    Recommendation 2018/006:
    That NHS England, within the 'Long-Term Plan', works with partners to identify 
    and meet the needs of young adults who have mental health problems that require 
    support but do not meet the current criteria for access to adult mental health services.
    
    Recommendation 2018/007:
    That NHS England requires Clinical Commissioning Groups to demonstrate that the 
    budget identified for current children and young people's services – those delivering 
    care up until the age of 18 - is spent only on this group.
    
    Recommendation 2018/008:
    That NHS England and NHS Improvement ensure that transition guidance, pathways 
    or performance measures require structured conversations to take place with the 
    young person transitioning to assess their readiness.
    """
    
    print("=" * 60)
    print("Testing HSIB Format")
    print("=" * 60)
    
    extractor = StrictRecommendationExtractor()
    recs = extractor.extract_recommendations(hsib_sample)
    
    for rec in recs:
        print(f"\nRec {rec.get('rec_number')}:")
        print(f"  Method: {rec['method']}")
        print(f"  Confidence: {rec['confidence']}")
        print(f"  Pattern: {rec.get('pattern_type')}")
        print(f"  Text: {rec['text'][:100]}...")
    
    print(f"\nTotal: {len(recs)} recommendations found")
    
    # Test standard format
    standard_sample = """
    Recommendation 1
    NHS England should establish a national framework for mental health data collection.
    
    Recommendation 2
    The Department of Health and Social Care should review current guidance on 
    patient safety reporting within 12 months.
    
    Recommendation 3
    All provider boards should ensure robust governance structures are in place.
    """
    
    print("\n" + "=" * 60)
    print("Testing Standard Government Format")
    print("=" * 60)
    
    recs = extractor.extract_recommendations(standard_sample)
    
    for rec in recs:
        print(f"\nRec {rec.get('rec_number')}:")
        print(f"  Method: {rec['method']}")
        print(f"  Confidence: {rec['confidence']}")
        print(f"  Pattern: {rec.get('pattern_type')}")
        print(f"  Text: {rec['text'][:100]}...")
    
    print(f"\nTotal: {len(recs)} recommendations found")
    
    stats = extractor.get_statistics(recs)
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_extractor()
