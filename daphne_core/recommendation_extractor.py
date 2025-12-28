"""
Strict Recommendation Extractor v3.0
With Semantic Coherence-Based Boundary Detection

v3.0 Changes:
- NEW: Semantic coherence detector to identify when content shifts away from recommendation
- NEW: Sentence-level analysis to find natural recommendation boundaries
- NEW: "That [Entity]" pattern detection for HSIB-style recommendations
- FIXED: Rec 2018/011 now correctly extracts only the CQC recommendation
- Maintains all previous format support (HSIB 2018, 2023, standard government)

The key insight: HSIB recommendations often start with "That [Entity] should/extends/requires..."
and end when we hit content that's clearly NOT a recommendation (statistics, background, etc.)

Usage:
    from recommendation_extractor import extract_recommendations
    
    recommendations = extract_recommendations(document_text, min_confidence=0.75)
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter

logger = logging.getLogger(__name__)


class StrictRecommendationExtractor:
    """
    Extract genuine recommendations with semantic coherence-based boundary detection.
    """
    
    MAX_SENTENCE_LENGTH = 500
    MAX_NUMBERED_REC_LENGTH = 2500
    
    # Maximum reasonable length for a single recommendation (characters)
    # Most recommendations are 100-500 chars; very long ones might be 1000
    MAX_SINGLE_REC_LENGTH = 1200
    
    def __init__(self):
        """Initialise with strict patterns"""
        
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
            'extends', 'extend',
        }
        
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
            r'Clinical\s+Commissioning\s+Groups?',
            r'CCGs?',
        ]
        
        # Patterns that indicate we've LEFT the recommendation text
        self.non_recommendation_indicators = [
            # Statistics and data
            r'\b\d+\s*(?:per\s*cent|%)\s+(?:of|had|were|have)',
            r'\bsuicide\s+(?:rate|is|was|accounts)',
            r'\b(?:an\s+)?average\s+of\s+\d+',
            r'\bstatistics\s+show',
            r'\bdata\s+(?:from|suggests?|shows?)',
            
            # Background/context language
            r'\b\d+\.\d+\.?\d*\s+[A-Z]',  # Numbered sections like "1.1.1. The..."
            r'\bprevalence\s+of',
            r'\bhistory\s+of',
            r'\bresearch\s+suggests',
            r'\bstudies?\s+(?:show|found|indicate)',
            
            # Meta-discussion about recommendations
            r'\bthe\s+investigation\s+makes',
            r'\bsafety\s+observation',
            r'\bhsib\s+has\s+directed',
            r'\bthese\s+organisations\s+are\s+expected',
            r'\bwe\s+will\s+publish',
            r'\bbackground\s+and\s+context',
            
            # Report structure (HSIB-specific)
            r'\bfindings\s+and\s+analysis',
            r'\bsummary\s+of\s+(?:hsib\s+)?findings',
            r'\bmethodology\b',
            r'\breference\s+event',
            r'\binvolvement\s+of\s+hsib',
            
            # Report structure (Standard government reports)
            r'\bour\s+vision\s+for',
            r'\bkey\s+facts\b',
            r'\bappendix(?:es)?\b',
            r'\bglossary\b',
            r'\breferences\b',
            r'\bcase\s+studies\b',
            r'\bconclusion(?:s)?\b',
            r'\backnowledgements?\b',
            r'\babout\s+(?:this|the)\s+(?:report|review)',
            r'\bdata\s+mapping\b',
            r'\ball\s+content\s+is\s+available\s+under',
            r'\bcrown\s+copyright',
            r'\bthroughout\s+the\s+review',
        ]
        
        # Compile for efficiency
        self.non_rec_pattern = re.compile(
            '|'.join(self.non_recommendation_indicators),
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
        
        # Remove page numbers (but NOT HSIB numbers like 2018/006)
        text = re.sub(r'\b(\d{1,2})/(\d{1,3})\b', '', text)
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
        
        text = re.sub(r'^UK\s+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with their start/end positions.
        Returns list of (sentence_text, start_pos, end_pos)
        """
        if not text:
            return []
        
        sentences = []
        # Match sentence-ending punctuation followed by space and capital letter
        # or end of string
        pattern = r'[.!?](?:\s+(?=[A-Z])|$)'
        
        last_end = 0
        for match in re.finditer(pattern, text):
            end = match.end()
            sentence = text[last_end:end].strip()
            if sentence:
                sentences.append((sentence, last_end, end))
            last_end = end
        
        # Capture any remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append((remaining, last_end, len(text)))
        
        return sentences
    
    def _find_semantic_boundary(self, text: str, rec_header_end: int) -> int:
        """
        Find where the recommendation content semantically ends.
        
        Strategy:
        1. Look for "That [Entity]" pattern - this is the actual recommendation
        2. Find where this recommendation sentence ends
        3. Check if following content is still recommendation-like or has shifted
        
        Returns the position where we should cut the text.
        """
        search_text = text[rec_header_end:]
        
        # HSIB recommendations typically start with "That [Entity]..."
        that_entity_match = re.search(
            r'\bThat\s+(?:the\s+)?(?:NHS\s+England|Care\s+Quality\s+Commission|CQC|'
            r'NHS\s+Improvement|DHSC|Clinical\s+Commissioning\s+Groups?)',
            search_text,
            re.IGNORECASE
        )
        
        if that_entity_match:
            # Found "That [Entity]" - now find where this sentence ends
            from_that = search_text[that_entity_match.start():]
            
            # The recommendation sentence typically ends at the first period
            # followed by something that's NOT part of the recommendation
            sentences = self._split_into_sentences(from_that)
            
            if sentences:
                # First sentence is the recommendation
                first_sentence = sentences[0][0]
                first_sentence_end = sentences[0][2]
                
                # Check if there are more sentences that are ALSO recommendations
                # (some recs have multiple sentences)
                cumulative_end = first_sentence_end
                
                for i in range(1, len(sentences)):
                    sent_text = sentences[i][0]
                    
                    # Check if this sentence looks like continuation of recommendation
                    # vs. background/context
                    if self._is_non_recommendation_content(sent_text):
                        # Stop here - this is no longer recommendation content
                        break
                    
                    # Check if it's a new recommendation (e.g., Safety Observation)
                    if re.search(r'\bSafety\s+Observation', sent_text, re.IGNORECASE):
                        break
                    if re.search(r'\bThe\s+investigation\s+makes', sent_text, re.IGNORECASE):
                        break
                    if re.search(r'\bHSIB\s+has\s+directed', sent_text, re.IGNORECASE):
                        break
                    
                    # Check for numbered sections (background content)
                    if re.match(r'\d+\s+[A-Z]', sent_text):
                        break
                    if re.match(r'\d+\.\d+', sent_text):
                        break
                    
                    # This sentence might still be part of the recommendation
                    # but let's be conservative - if it's getting long, stop
                    if cumulative_end > self.MAX_SINGLE_REC_LENGTH:
                        break
                    
                    cumulative_end = sentences[i][2]
                
                return rec_header_end + that_entity_match.start() + cumulative_end
        
        # No "That [Entity]" found - fall back to finding first non-rec content
        # or use a maximum length
        
        # Look for indicators that we've left the recommendation
        non_rec_match = self.non_rec_pattern.search(search_text)
        if non_rec_match:
            boundary = rec_header_end + non_rec_match.start()
            # Make sure we have at least some content
            if boundary - rec_header_end > 50:
                return boundary
        
        # Fall back to maximum length
        return rec_header_end + min(len(search_text), self.MAX_SINGLE_REC_LENGTH)
    
    def _is_non_recommendation_content(self, text: str) -> bool:
        """Check if text is clearly NOT recommendation content."""
        if not text:
            return True
        
        text_lower = text.lower()
        
        # Statistics patterns
        if re.search(r'\b\d+\s*(?:per\s*cent|%)', text_lower):
            # But allow "100 per cent" in context of commitment
            if not re.search(r'commit|ensure|require', text_lower):
                return True
        
        # Historical/background language
        if re.search(r'\b(?:in\s+)?(?:19|20)\d{2}[,.]?\s+(?:the|a|an)\b', text_lower):
            return True
        
        # Study/research language
        if re.search(r'\b(?:study|research|report|review)\s+(?:found|showed|indicates)', text_lower):
            return True
        
        # Numbered sections (like "1.1.1 Prevalence...")
        if re.match(r'\d+\.\d+\.?\d*\s+[A-Z]', text):
            return True
        
        # Death/suicide statistics
        if re.search(r'\b(?:suicide|death|died)\s+(?:rate|is|was|by|accounting)', text_lower):
            return True
        
        return False
    
    def _extract_that_entity_recommendation(self, text: str, header_match: re.Match) -> Optional[str]:
        """
        Extract just the "That [Entity]..." recommendation from HSIB format.
        
        HSIB recommendations have format:
        Recommendation 2018/011: [optional noise/previous content] That the Care Quality Commission extends...
        
        We want to extract just the "That..." part and find its natural end.
        
        Key insight: Sometimes PDF extraction merges the end of a previous paragraph
        with the recommendation header, e.g.:
        "Recommendation 2018/011:that early intervention reduces..." 
        where "that early intervention..." is actually from the previous section,
        and the real rec is "That the Care Quality Commission extends..."
        """
        # Get text after the header
        after_header = text[header_match.end():]
        
        # Look for "That [Entity]" pattern - the TRUE start of the recommendation
        # This handles cases where there's noise before the actual recommendation
        that_entity_patterns = [
            # Standard patterns
            r'(That\s+(?:the\s+)?(?:NHS\s+England|Care\s+Quality\s+Commission|CQC|'
            r'NHS\s+Improvement|DHSC|CCGs?|Clinical\s+Commissioning\s+Groups?)'
            r'[^.]*\.)',
            # Also try with just "That the" followed by a capital entity
            r'(That\s+the\s+[A-Z][a-zA-Z\s]+(?:Commission|England|Improvement|Department|Office)'
            r'[^.]*\.)',
        ]
        
        for pattern in that_entity_patterns:
            that_match = re.search(pattern, after_header, re.IGNORECASE | re.DOTALL)
            if that_match:
                rec_text = that_match.group(1).strip()
                # Clean it up
                rec_text = re.sub(r'\s+', ' ', rec_text)
                
                # Verify this looks like a real recommendation (has a verb)
                if re.search(r'\b(?:extends?|requires?|ensures?|works?|should|must)\b', rec_text, re.IGNORECASE):
                    return rec_text
        
        # If no "That [Entity]" found, check if there's lowercase "that" at the start
        # which might indicate PDF extraction noise - skip past it
        noise_match = re.match(r'^that\s+[a-z].*?\.', after_header, re.IGNORECASE | re.DOTALL)
        if noise_match:
            # There's likely noise at the start - look for "That [Entity]" after it
            remaining = after_header[noise_match.end():]
            for pattern in that_entity_patterns:
                that_match = re.search(pattern, remaining, re.IGNORECASE | re.DOTALL)
                if that_match:
                    rec_text = that_match.group(1).strip()
                    rec_text = re.sub(r'\s+', ' ', rec_text)
                    if re.search(r'\b(?:extends?|requires?|ensures?|works?|should|must)\b', rec_text, re.IGNORECASE):
                        return rec_text
        
        return None
    
    def is_garbage(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, str]:
        """FIRST-PASS filter: reject obvious garbage before any analysis."""
        if not text:
            return True, "empty"
        
        cleaned = self.clean_text(text)
        
        if len(cleaned) < 40:
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
        """Check if text is talking ABOUT recommendations rather than making one."""
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
        
        # Check for "extends", "requires" etc. directly after entity
        entity_verb_match = re.search(r'(?:commission|cqc|england|improvement)\s+(\w+)', text_lower)
        if entity_verb_match:
            verb = entity_verb_match.group(1)
            if verb in self.action_verbs:
                return verb
        
        for verb in sorted(self.action_verbs, key=len, reverse=True):
            if re.search(rf'\b{verb}\b', text_lower):
                return verb
        
        return 'unknown'
    
    def is_genuine_recommendation(self, text: str, is_numbered_rec: bool = False) -> Tuple[bool, float, str, str]:
        """Determine if text is a genuine recommendation."""
        if not text:
            return False, 0.0, 'empty', 'unknown'
        
        cleaned = self.clean_text(text)
        if not cleaned:
            return False, 0.0, 'empty', 'unknown'
        
        text_lower = cleaned.lower()
        
        if not is_numbered_rec:
            if len(cleaned) > self.MAX_SENTENCE_LENGTH:
                return False, 0.0, 'too_long', 'none'
        
        # HSIB 2023+ format
        hsib_2023_match = re.match(
            r'^Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]+(.+)',
            cleaned,
            re.IGNORECASE,
        )
        if hsib_2023_match:
            rec_num = hsib_2023_match.group(1)
            rec_text = hsib_2023_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.98, f'hsib_recommendation_{rec_num}', verb
        
        # HSIB 2018 format
        hsib_2018_match = re.match(
            r'^Recommendation\s+(\d{4}/\d{3})[:\s]+(.+)',
            cleaned,
            re.IGNORECASE,
        )
        if hsib_2018_match:
            rec_num = hsib_2018_match.group(1)
            rec_text = hsib_2018_match.group(2)
            verb = self._extract_verb(rec_text)
            return True, 0.98, f'hsib_recommendation_{rec_num}', verb
        
        # Standard numbered recommendation
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
        
        # "That [Entity]" pattern (common in HSIB)
        that_entity_match = re.match(
            r'^That\s+(?:the\s+)?(NHS\s+England|Care\s+Quality\s+Commission|CQC|'
            r'NHS\s+Improvement|DHSC)',
            cleaned,
            re.IGNORECASE,
        )
        if that_entity_match:
            verb = self._extract_verb(cleaned)
            return True, 0.96, 'that_entity', verb
        
        # Entity + should pattern
        for entity_pattern in self.recommending_entities:
            pattern = rf'{entity_pattern}\s+should\s+(?:also\s+)?(?:urgently\s+)?(?:\w+ly\s+)?(\w+)'
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match and match.group(1):
                verb = match.group(1).lower()
                if verb in self.action_verbs:
                    return True, 0.95, 'entity_should', verb
        
        # "We recommend" pattern
        if re.search(r'\bwe\s+recommend\b', text_lower):
            verb = self._extract_verb(cleaned)
            return True, 0.90, 'we_recommend', verb
        
        # "should be" + action (passive)
        should_be_match = re.search(r'(\w+(?:\s+\w+)?)\s+should\s+be\s+(\w+ed)\b', text_lower)
        if should_be_match:
            subject = should_be_match.group(1)
            verb = should_be_match.group(2)
            if subject not in ['this', 'it', 'that', 'recommendation']:
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
        
        return False, 0.0, 'none', 'unknown'
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.75) -> List[Dict]:
        """Extract genuine recommendations from text with semantic boundary detection."""
        recommendations = []
        seen_rec_numbers: Set[str] = set()  # Track to prevent duplicates
        
        if not text or not text.strip():
            logger.warning("Empty text passed to extract_recommendations")
            return []
        
        logger.info(f"Extracting recommendations from text of length {len(text)}")
        
        cleaned_full_text = self.clean_text(text)
        
        # =======================================================================
        # PHASE 1: HSIB 2023+ format (Safety recommendation R/YYYY/NNN)
        # =======================================================================
        hsib_2023_pattern = re.compile(
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        hsib_2023_matches = list(hsib_2023_pattern.finditer(cleaned_full_text))
        
        if hsib_2023_matches:
            logger.info(f"Found {len(hsib_2023_matches)} HSIB 2023+ recommendations")
            for idx, match in enumerate(hsib_2023_matches):
                rec_num = match.group(1)
                
                if rec_num in seen_rec_numbers:
                    logger.debug(f"Skipping duplicate: {rec_num}")
                    continue
                
                # Use semantic boundary detection
                boundary = self._find_semantic_boundary(cleaned_full_text, match.end())
                raw_block = cleaned_full_text[match.start():boundary]
                cleaned = self.clean_text(raw_block)
                
                is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=True)
                if is_garbage:
                    continue
                
                is_rec, confidence, method, verb = self.is_genuine_recommendation(cleaned, is_numbered_rec=True)
                
                if is_rec and confidence >= min_confidence:
                    seen_rec_numbers.add(rec_num)
                    recommendations.append({
                        'text': cleaned,
                        'verb': verb,
                        'method': method,
                        'confidence': round(confidence, 3),
                        'position': idx,
                        'in_section': True,
                        'rec_number': rec_num,
                    })
            
            if recommendations:
                recommendations = self._deduplicate(recommendations)
                logger.info(f"Found {len(recommendations)} HSIB 2023+ recommendations")
                return recommendations
        
        # =======================================================================
        # PHASE 2: HSIB 2018 format (Recommendation YYYY/NNN)
        # =======================================================================
        hsib_2018_pattern = re.compile(
            r'Recommendation\s+(\d{4}/\d{3})[:\s]',
            re.IGNORECASE
        )
        hsib_2018_matches = list(hsib_2018_pattern.finditer(cleaned_full_text))
        
        if hsib_2018_matches:
            logger.info(f"Found {len(hsib_2018_matches)} HSIB 2018 recommendations")
            for idx, match in enumerate(hsib_2018_matches):
                rec_num = match.group(1)
                
                if rec_num in seen_rec_numbers:
                    logger.debug(f"Skipping duplicate: {rec_num}")
                    continue
                
                # Try to extract just the "That [Entity]..." part
                that_rec = self._extract_that_entity_recommendation(cleaned_full_text, match)
                
                if that_rec:
                    # Successfully extracted the core recommendation
                    cleaned = f"Recommendation {rec_num}: {that_rec}"
                else:
                    # Fall back to semantic boundary detection
                    boundary = self._find_semantic_boundary(cleaned_full_text, match.end())
                    raw_block = cleaned_full_text[match.start():boundary]
                    cleaned = self.clean_text(raw_block)
                
                is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=True)
                if is_garbage:
                    continue
                
                is_rec, confidence, method, verb = self.is_genuine_recommendation(cleaned, is_numbered_rec=True)
                
                if is_rec and confidence >= min_confidence:
                    seen_rec_numbers.add(rec_num)
                    recommendations.append({
                        'text': cleaned,
                        'verb': verb,
                        'method': method,
                        'confidence': round(confidence, 3),
                        'position': idx,
                        'in_section': True,
                        'rec_number': rec_num,
                    })
            
            if recommendations:
                recommendations = self._deduplicate(recommendations)
                logger.info(f"Found {len(recommendations)} HSIB 2018 recommendations")
                return recommendations
        
        # =======================================================================
        # PHASE 3: Standard "Recommendation N" format
        # =======================================================================
        heading_pattern = re.compile(
            r'(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b',
            re.IGNORECASE,
        )
        heading_matches = list(heading_pattern.finditer(cleaned_full_text))
        
        if heading_matches:
            logger.info(f"Found {len(heading_matches)} standard recommendation headings")
            
            for idx, match in enumerate(heading_matches):
                heading_num = match.group(1)
                extra_digit = match.group(2)
                if extra_digit and len(heading_num) == 1:
                    rec_num = f"{heading_num}{extra_digit}"
                else:
                    rec_num = heading_num
                
                if rec_num in seen_rec_numbers:
                    logger.debug(f"Skipping duplicate: {rec_num}")
                    continue
                
                # Find boundary
                next_rec_pos = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(cleaned_full_text)
                boundary = min(next_rec_pos, self._find_semantic_boundary(cleaned_full_text, match.end()))
                
                raw_block = cleaned_full_text[match.start():boundary]
                cleaned = self.clean_text(raw_block)
                
                is_garbage, reason = self.is_garbage(cleaned, is_numbered_rec=True)
                if is_garbage:
                    continue
                
                is_rec, confidence, method, verb = self.is_genuine_recommendation(cleaned, is_numbered_rec=True)
                
                if is_rec and confidence >= min_confidence:
                    # Normalise the heading
                    norm_pattern = r'^(?:Recommendations?\s+)?Recommendation\s+(\d{1,2})(?:\s+(\d))?\b'
                    cleaned = re.sub(norm_pattern, f"Recommendation {rec_num}", cleaned, count=1, flags=re.IGNORECASE)
                    
                    seen_rec_numbers.add(rec_num)
                    recommendations.append({
                        'text': cleaned,
                        'verb': verb,
                        'method': method,
                        'confidence': round(confidence, 3),
                        'position': idx,
                        'in_section': True,
                        'rec_number': rec_num,
                    })
        
        # =======================================================================
        # PHASE 4: Validate and fall back to sentence extraction if needed
        # =======================================================================
        if len(recommendations) == 1:
            only_rec = recommendations[0]
            if len(only_rec.get("text", "")) > self.MAX_NUMBERED_REC_LENGTH:
                logger.info("Single oversized recommendation - falling back to sentence extraction")
                recommendations = []
        
        if len(recommendations) == 0:
            logger.info("No structured recommendations found, falling back to sentence extraction")
            
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            for idx, sentence in enumerate(sentences):
                cleaned = self.clean_text(sentence)
                
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


# Backward compatibility
AdvancedRecommendationExtractor = StrictRecommendationExtractor


def extract_recommendations(text: str, min_confidence: float = 0.75) -> List[Dict]:
    """Main function to extract recommendations."""
    extractor = StrictRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)
