"""
Alignment Engine for matching recommendations with responses.

This module handles:
1. Matching recommendations to their corresponding responses
2. Classifying acceptance status (Accepted, Partial, Rejected)
3. Confidence scoring for matches
4. Content type classification
5. Pattern matching for document analysis

v4.1 Changes:
- RESTORED: calculate_simple_similarity, classify_content_type, 
  determine_alignment_status, find_pattern_matches functions
- FIXED: Status classification now handles partial acceptance language properly
- FIXED: "will give careful consideration" now classified as Partial, not Accepted
- ADDED: More comprehensive partial acceptance patterns
- ADDED: Decimal ID matching support (8.1, 8.2, 8.3-8.4)
- IMPROVED: Better rejection detection patterns
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stop Words for text processing
# --------------------------------------------------------------------------- #

STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was',
    'were', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
    'just', 'should', 'now', 'i', 'you', 'we', 'our', 'your', 'their', 'would',
    'could', 'may', 'might', 'must', 'shall', 'into', 'if', 'then', 'because',
    'while', 'although', 'though', 'after', 'before', 'above', 'below', 'between',
    'under', 'again', 'further', 'once', 'here', 'there', 'these', 'those', 'am',
    'been', 'being', 'do', 'does', 'did', 'doing', 'having', 'her', 'him', 'his',
    'herself', 'himself', 'itself', 'me', 'my', 'myself', 'ourselves', 'she',
    'them', 'themselves', 'us', 'whom', 'yours', 'yourself', 'yourselves',
}


# --------------------------------------------------------------------------- #
# Utility Functions (RESTORED)
# --------------------------------------------------------------------------- #

def calculate_simple_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity between two strings.
    
    Uses word overlap (Jaccard similarity) for efficiency.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    # Tokenise and normalise
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Remove stop words
    words1 = words1 - STOP_WORDS
    words2 = words2 - STOP_WORDS
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def classify_content_type(text: str) -> str:
    """
    Classify the type of content in the text.
    
    Args:
        text: Text to classify
        
    Returns:
        Content type string: 'Recommendation', 'Response', 'Policy', 
        'Procedural', 'Financial', or 'General'
    """
    if not text:
        return 'General'
    
    text_lower = text.lower()
    
    # Check for recommendation indicators
    rec_patterns = [
        r'\brecommend(?:s|ed|ation|ations)?\b',
        r'\bshould\b',
        r'\bmust\b',
        r'\badvise[sd]?\b',
        r'\bsuggest(?:s|ed|ion|ions)?\b',
        r'\burge[sd]?\b',
        r'\bpropose[sd]?\b',
    ]
    
    for pattern in rec_patterns:
        if re.search(pattern, text_lower):
            return 'Recommendation'
    
    # Check for response indicators
    resp_patterns = [
        r'\baccept(?:s|ed)?\b',
        r'\breject(?:s|ed)?\b',
        r'\bagree[sd]?\b',
        r'\bdisagree[sd]?\b',
        r'\bgovernment\s+(?:will|has|is)\b',
        r'\bwe\s+(?:will|have|are)\b',
        r'\bresponse\b',
        r'\bimplement(?:s|ed|ing|ation)?\b',
    ]
    
    for pattern in resp_patterns:
        if re.search(pattern, text_lower):
            return 'Response'
    
    # Check for policy indicators
    policy_patterns = [
        r'\bpolic(?:y|ies)\b',
        r'\bframework\b',
        r'\bguideline[s]?\b',
        r'\bprotocol[s]?\b',
        r'\bstrateg(?:y|ies)\b',
        r'\blegislat(?:ion|ive)\b',
    ]
    
    for pattern in policy_patterns:
        if re.search(pattern, text_lower):
            return 'Policy'
    
    # Check for procedural indicators
    proc_patterns = [
        r'\bprocedure[s]?\b',
        r'\bprocess(?:es)?\b',
        r'\bstep[s]?\b',
        r'\bmethod(?:s|ology)?\b',
        r'\bapproach(?:es)?\b',
    ]
    
    for pattern in proc_patterns:
        if re.search(pattern, text_lower):
            return 'Procedural'
    
    # Check for financial indicators
    fin_patterns = [
        r'\bbudget[s]?\b',
        r'\bfund(?:s|ing|ed)?\b',
        r'\bcost[s]?\b',
        r'\bexpenditure[s]?\b',
        r'\bfinancial\b',
        r'\b£\d',
        r'\bmillion\b',
        r'\bbillion\b',
    ]
    
    for pattern in fin_patterns:
        if re.search(pattern, text_lower):
            return 'Financial'
    
    return 'General'


def determine_alignment_status(
    recommendation: Dict[str, Any],
    response: Optional[Dict[str, Any]]
) -> Tuple[str, float]:
    """
    Determine the alignment status between a recommendation and response.
    
    Args:
        recommendation: Recommendation dictionary
        response: Response dictionary (or None if no response)
        
    Returns:
        Tuple of (status, confidence)
    """
    if not response:
        return 'No Response', 0.0
    
    response_text = response.get('text', '') or response.get('sentence', '')
    
    classifier = StatusClassifier()
    status, confidence, _ = classifier.classify(response_text)
    
    return status, confidence


def find_pattern_matches(
    documents: List[Dict[str, Any]],
    patterns: List[str],
    content_type: str = 'general'
) -> List[Dict[str, Any]]:
    """
    Find sentences matching given patterns in documents.
    
    Args:
        documents: List of document dictionaries with 'text' field
        patterns: List of pattern strings to search for
        content_type: Type label for matched content
        
    Returns:
        List of match dictionaries with sentence, document, position info
    """
    matches = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            
            # Check each pattern
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in sentence_lower:
                    # Calculate position
                    pos = text.find(sentence)
                    if pos == -1:
                        pos = i * 100
                    
                    # Get context (surrounding sentences)
                    context_start = max(0, i - 1)
                    context_end = min(len(sentences), i + 2)
                    context = ' '.join(
                        s.strip() for s in sentences[context_start:context_end]
                        if s.strip()
                    )
                    
                    matches.append({
                        'sentence': sentence,
                        'context': context,
                        'document': doc,
                        'position': pos,
                        'page_number': max(1, pos // 2000 + 1),
                        'pattern': pattern,
                        'content_type': content_type,
                        'recommendation_type': classify_content_type(sentence),
                    })
                    break  # Only match once per sentence
    
    return matches


def extract_response_sentences(text: str) -> List[Dict[str, Any]]:
    """
    Extract response sentences from text.
    
    Looks for government response patterns and extracts relevant sentences.
    
    Args:
        text: Document text to search
        
    Returns:
        List of response dictionaries
    """
    responses = []
    
    if not text:
        return responses
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Response indicator patterns
    response_patterns = [
        r'\bgovernment\s+(?:accepts?|agrees?|supports?|rejects?|notes?)\b',
        r'\bwe\s+(?:accept|agree|support|reject|note)\b',
        r'\bthis\s+recommendation\s+(?:is|has\s+been)\b',
        r'\b(?:accept|reject|agree|implement)(?:s|ed|ing)?\s+(?:this|the)\s+recommendation\b',
        r'\bresponse\s+to\s+recommendation\b',
    ]
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        
        sentence_lower = sentence.lower()
        
        for pattern in response_patterns:
            if re.search(pattern, sentence_lower):
                # Get context
                context_start = max(0, i - 1)
                context_end = min(len(sentences), i + 2)
                context = ' '.join(
                    s.strip() for s in sentences[context_start:context_end]
                    if s.strip()
                )
                
                # Classify status
                classifier = StatusClassifier()
                status, confidence, patterns_matched = classifier.classify(sentence)
                
                responses.append({
                    'text': sentence,
                    'sentence': sentence,
                    'context': context,
                    'status': status,
                    'confidence': confidence,
                    'patterns_matched': patterns_matched,
                })
                break
    
    return responses


# --------------------------------------------------------------------------- #
# Status Classification
# --------------------------------------------------------------------------- #

class StatusClassifier:
    """
    Classify government response acceptance status.
    
    Categories:
    - Accepted: Full acceptance, commitment to implement
    - Partial: Partial acceptance, consideration, or conditional acceptance
    - Rejected: Explicit rejection or non-acceptance
    """
    
    def __init__(self):
        """Initialise classification patterns."""
        
        # v4.0: Expanded and refined patterns
        
        # Strong acceptance indicators (high confidence)
        self.accepted_patterns = [
            # Direct acceptance
            r'\b(?:the\s+)?government\s+(?:fully\s+)?accepts?\b',
            r'\b(?:the\s+)?government\s+(?:fully\s+)?agrees?\b',
            r'\b(?:the\s+)?government\s+(?:fully\s+)?supports?\b',
            r'\b(?:the\s+)?government\s+welcomes?\b',
            r'\bwe\s+(?:fully\s+)?accept\b',
            r'\bwe\s+(?:fully\s+)?agree\b',
            r'\bwe\s+(?:fully\s+)?support\b',
            r'\bwe\s+welcome\b',
            r'\bwe\s+are\s+happy\s+to\s+(?:accept|confirm)\b',
            r'\bwe\s+are\s+pleased\s+to\s+(?:accept|confirm)\b',
            
            # Commitment to action
            r'\b(?:the\s+)?government\s+(?:has\s+)?committed\b',
            r'\b(?:the\s+)?government\s+will\s+(?:implement|deliver|take\s+forward)\b',
            r'\bwe\s+(?:have\s+)?committed\b',
            r'\bwe\s+will\s+(?:implement|deliver|take\s+forward)\b',
            r'\bthis\s+(?:recommendation\s+)?(?:has\s+been|is\s+being)\s+implemented\b',
            r'\bwe\s+have\s+already\s+(?:implemented|actioned|delivered)\b',
            r'\bthis\s+is\s+already\s+(?:in\s+place|implemented|underway)\b',
            
            # Organisation-specific acceptance
            r'\bnhs\s+england\s+(?:fully\s+)?(?:accepts?|agrees?|supports?)\b',
            r'\bdhsc\s+(?:fully\s+)?(?:accepts?|agrees?|supports?)\b',
            r'\bcqc\s+(?:fully\s+)?(?:accepts?|agrees?|supports?)\b',
            r'\bnice\s+(?:fully\s+)?(?:accepts?|agrees?|supports?)\b',
        ]
        
        # v4.0: Expanded partial acceptance indicators
        self.partial_patterns = [
            # Explicit partial acceptance
            r'\b(?:the\s+)?government\s+(?:partially\s+)?accepts?\s+(?:this\s+)?(?:recommendation\s+)?in\s+(?:part|principle)\b',
            r'\bwe\s+(?:partially\s+)?accept\s+(?:this\s+)?(?:recommendation\s+)?in\s+(?:part|principle)\b',
            r'\bpartially\s+(?:accept|agree|support)\b',
            r'\baccepts?\s+in\s+part\b',
            r'\baccepts?\s+in\s+principle\b',
            r'\bagrees?\s+in\s+(?:part|principle)\b',
            
            # Consideration language (NOT acceptance)
            r'\bwill\s+(?:give\s+)?(?:careful\s+)?consideration\b',
            r'\bwill\s+consider\b',
            r'\bwill\s+(?:continue\s+to\s+)?review\b',
            r'\bwill\s+(?:further\s+)?explore\b',
            r'\bwill\s+(?:further\s+)?examine\b',
            r'\bwill\s+look\s+(?:further\s+)?(?:at|into)\b',
            r'\b(?:is|are)\s+(?:being\s+)?considered\b',
            r'\b(?:is|are)\s+under\s+(?:consideration|review)\b',
            r'\brequires?\s+(?:further\s+)?(?:consideration|review|analysis)\b',
            
            # Conditional acceptance
            r'\bsubject\s+to\b',
            r'\bdepending\s+on\b',
            r'\bif\s+(?:and\s+when|appropriate|feasible)\b',
            r'\bwhere\s+(?:possible|appropriate|feasible)\b',
            r'\bprovided\s+that\b',
            r'\bon\s+condition\s+that\b',
            
            # Notes/acknowledges without commitment
            r'\b(?:the\s+)?government\s+notes?\b(?!\s+(?:that\s+)?(?:it|we)\s+(?:has|have|will|is|are))',
            r'\bwe\s+note\b(?!\s+(?:that\s+)?(?:it|we)\s+(?:has|have|will|is|are))',
            r'\b(?:the\s+)?government\s+recognises?\b',
            r'\bwe\s+recognise\b',
            r'\b(?:the\s+)?government\s+acknowledges?\b',
            r'\bwe\s+acknowledge\b',
            
            # Concerns or reservations
            r'\bhowever\s*,?\s*(?:the\s+)?(?:government|we)\b',
            r'\bwhile\s+(?:the\s+)?(?:government|we)\s+(?:accept|agree|support)\b',
            r'\balthough\s+(?:the\s+)?(?:government|we)\s+(?:accept|agree|support)\b',
            r'\bwith\s+(?:some\s+)?reservations?\b',
            r'\bwith\s+(?:some\s+)?caveats?\b',
            
            # v4.0: Additional partial patterns from parliamentary documents
            r'\bwill\s+take\s+(?:this\s+)?into\s+account\b',
            r'\bwill\s+bear\s+(?:this\s+)?in\s+mind\b',
            r'\bwill\s+keep\s+(?:this\s+)?under\s+review\b',
            r'\bopen\s+to\s+(?:considering|exploring)\b',
            r'\bmay\s+(?:consider|explore|review)\b',
            r'\bsympathetic\s+to\b',
            r'\bunderstands?\s+the\s+(?:concern|point|argument)\b',
        ]
        
        # v4.0: Expanded rejection indicators
        self.rejected_patterns = [
            # Direct rejection
            r'\b(?:the\s+)?government\s+(?:does\s+not|doesn\'t)\s+(?:accept|agree|support)\b',
            r'\b(?:the\s+)?government\s+rejects?\b',
            r'\b(?:the\s+)?government\s+declines?\b',
            r'\b(?:the\s+)?government\s+(?:is\s+)?unable\s+to\s+(?:accept|agree|support)\b',
            r'\bwe\s+(?:do\s+not|don\'t)\s+(?:accept|agree|support)\b',
            r'\bwe\s+reject\b',
            r'\bwe\s+decline\b',
            r'\bwe\s+(?:are\s+)?unable\s+to\s+(?:accept|agree|support)\b',
            r'\bcannot\s+(?:accept|agree|support)\b',
            
            # Indirect rejection
            r'\b(?:the\s+)?government\s+(?:does\s+not|doesn\'t)\s+(?:believe|consider|think)\b',
            r'\bwe\s+(?:do\s+not|don\'t)\s+(?:believe|consider|think)\b',
            r'\bnot\s+(?:in\s+a\s+position|able)\s+to\s+(?:accept|agree|support|implement)\b',
            r'\b(?:is|are)\s+not\s+(?:appropriate|feasible|possible|practical)\b',
            r'\bwould\s+not\s+be\s+(?:appropriate|feasible|possible|practical)\b',
            
            # Not addressed to this body
            r'\bnot\s+(?:a\s+matter\s+)?for\s+(?:the\s+)?(?:government|us)\b',
            r'\boutside\s+(?:the\s+)?(?:government\'?s?|our)\s+(?:remit|scope)\b',
            r'\bfor\s+(?:other\s+)?organi[sz]ations?\s+to\s+(?:consider|address)\b',
            
            # v4.0: Additional rejection patterns
            r'\brespectfully\s+(?:disagrees?|declines?)\b',
            r'\bregrettably\s+(?:cannot|unable)\b',
            r'\bunfortunately\s+(?:cannot|unable)\b',
            r'\b(?:does\s+not|doesn\'t)\s+share\s+(?:this|the)\s+view\b',
        ]
        
        # Compile patterns for efficiency
        self.accepted_compiled = [re.compile(p, re.IGNORECASE) for p in self.accepted_patterns]
        self.partial_compiled = [re.compile(p, re.IGNORECASE) for p in self.partial_patterns]
        self.rejected_compiled = [re.compile(p, re.IGNORECASE) for p in self.rejected_patterns]
    
    def classify(self, response_text: str) -> Tuple[str, float, List[str]]:
        """
        Classify response acceptance status.
        
        Args:
            response_text: The response text to classify
            
        Returns:
            Tuple of (status, confidence, matched_patterns)
        """
        if not response_text:
            return 'Unknown', 0.0, []
        
        text = response_text.lower()
        
        # Count matches for each category
        accepted_matches = []
        partial_matches = []
        rejected_matches = []
        
        for pattern in self.accepted_compiled:
            if pattern.search(text):
                accepted_matches.append(pattern.pattern)
        
        for pattern in self.partial_compiled:
            if pattern.search(text):
                partial_matches.append(pattern.pattern)
        
        for pattern in self.rejected_compiled:
            if pattern.search(text):
                rejected_matches.append(pattern.pattern)
        
        # v4.0: Priority-based classification
        # Rejection takes priority (if explicit rejection found, it's rejected)
        if rejected_matches and not accepted_matches:
            confidence = min(0.95, 0.7 + len(rejected_matches) * 0.1)
            return 'Rejected', confidence, rejected_matches
        
        # v4.0: Partial takes priority over acceptance if partial language found
        # This is the key fix: "will give careful consideration" should be Partial
        if partial_matches:
            # Check if there's also strong acceptance language
            if accepted_matches:
                # Has both - need to determine which is dominant
                # If partial language like "consideration" or "notes" is present,
                # it modifies the acceptance
                consideration_patterns = [
                    r'consideration', r'consider\b', r'review\b', r'explore',
                    r'examine', r'notes?\b', r'recognise', r'acknowledge'
                ]
                has_consideration = any(
                    re.search(p, text, re.IGNORECASE) 
                    for p in consideration_patterns
                )
                
                if has_consideration:
                    # Partial wins - they're considering, not committing
                    confidence = min(0.90, 0.6 + len(partial_matches) * 0.1)
                    return 'Partial', confidence, partial_matches
                else:
                    # Acceptance with caveats - still Partial
                    confidence = min(0.85, 0.6 + len(partial_matches) * 0.1)
                    return 'Partial', confidence, partial_matches
            else:
                # Only partial patterns found
                confidence = min(0.90, 0.6 + len(partial_matches) * 0.1)
                return 'Partial', confidence, partial_matches
        
        # Pure acceptance
        if accepted_matches:
            confidence = min(0.95, 0.7 + len(accepted_matches) * 0.1)
            return 'Accepted', confidence, accepted_matches
        
        # No clear patterns found
        return 'Unknown', 0.3, []
    
    def get_status_summary(self, response_text: str) -> Dict:
        """
        Get detailed status classification summary.
        
        Returns dict with status, confidence, matched patterns, and reasoning.
        """
        status, confidence, patterns = self.classify(response_text)
        
        return {
            'status': status,
            'confidence': confidence,
            'matched_patterns': patterns,
            'pattern_count': len(patterns),
        }


# --------------------------------------------------------------------------- #
# Recommendation-Response Alignment
# --------------------------------------------------------------------------- #

class AlignmentEngine:
    """
    Match recommendations with their corresponding responses.
    
    Supports multiple matching strategies:
    1. ID-based matching (for structured documents)
    2. Position-based matching (for ordered documents)
    3. Content similarity matching (for unstructured documents)
    """
    
    def __init__(self):
        """Initialise alignment engine."""
        self.status_classifier = StatusClassifier()
    
    def normalise_rec_id(self, rec_id: str) -> str:
        """
        Normalise recommendation ID for matching.
        
        Handles various formats:
        - "1", "01" -> "1"
        - "R/2023/220" -> "R/2023/220"
        - "2018/006" -> "2018/006"
        - "8.1", "8.3-8.4" -> "8.1", "8.3-8.4" (v4.0: preserve decimal IDs)
        """
        if not rec_id:
            return ''
        
        rec_id = str(rec_id).strip()
        
        # Preserve HSIB format IDs
        if '/' in rec_id:
            return rec_id
        
        # v4.0: Preserve decimal format IDs
        if '.' in rec_id:
            return rec_id
        
        # v4.0: Preserve range format IDs
        if '-' in rec_id and not rec_id.startswith('-'):
            return rec_id
        
        # Try to convert to integer for simple numbers
        try:
            return str(int(rec_id))
        except ValueError:
            return rec_id
    
    def match_by_id(
        self,
        recommendations: List[Dict],
        responses: List[Dict]
    ) -> List[Dict]:
        """
        Match recommendations and responses by ID.
        
        v4.0: Now handles decimal IDs (8.1, 8.2, 8.3-8.4) properly.
        """
        aligned = []
        
        # Build response lookup by normalised ID
        response_lookup = {}
        for resp in responses:
            rec_id = resp.get('rec_id') or resp.get('rec_number')
            if rec_id:
                normalised = self.normalise_rec_id(rec_id)
                response_lookup[normalised] = resp
                logger.debug(f"Response indexed: {rec_id} -> {normalised}")
        
        logger.info(f"Response lookup has {len(response_lookup)} entries: {list(response_lookup.keys())}")
        
        for rec in recommendations:
            rec_id = rec.get('rec_number') or rec.get('rec_id')
            normalised_id = self.normalise_rec_id(rec_id) if rec_id else None
            
            logger.debug(f"Looking for response to recommendation: {rec_id} (normalised: {normalised_id})")
            
            matched_response = None
            if normalised_id and normalised_id in response_lookup:
                matched_response = response_lookup[normalised_id]
                logger.debug(f"Found match for {normalised_id}")
            
            # Classify status
            if matched_response:
                status, confidence, patterns = self.status_classifier.classify(
                    matched_response.get('text', '')
                )
            else:
                status, confidence, patterns = 'No Response', 0.0, []
            
            aligned.append({
                'recommendation': rec,
                'response': matched_response,
                'rec_id': rec_id,
                'match_method': 'id_match' if matched_response else 'no_match',
                'match_confidence': 0.95 if matched_response else 0.0,
                'status': status,
                'status_confidence': confidence,
                'status_patterns': patterns,
            })
        
        return aligned
    
    def match_by_position(
        self,
        recommendations: List[Dict],
        responses: List[Dict]
    ) -> List[Dict]:
        """
        Match recommendations and responses by position/order.
        
        Assumes recommendations and responses appear in the same order.
        """
        aligned = []
        
        for i, rec in enumerate(recommendations):
            matched_response = responses[i] if i < len(responses) else None
            
            if matched_response:
                status, confidence, patterns = self.status_classifier.classify(
                    matched_response.get('text', '')
                )
            else:
                status, confidence, patterns = 'No Response', 0.0, []
            
            rec_id = rec.get('rec_number') or rec.get('rec_id') or str(i + 1)
            
            aligned.append({
                'recommendation': rec,
                'response': matched_response,
                'rec_id': rec_id,
                'match_method': 'position_match' if matched_response else 'no_match',
                'match_confidence': 0.7 if matched_response else 0.0,
                'status': status,
                'status_confidence': confidence,
                'status_patterns': patterns,
            })
        
        return aligned
    
    def match_by_similarity(
        self,
        recommendations: List[Dict],
        responses: List[Dict],
        threshold: float = 0.4
    ) -> List[Dict]:
        """
        Match recommendations and responses by content similarity.
        
        Uses text similarity to find best matches.
        """
        aligned = []
        used_responses = set()
        
        for rec in recommendations:
            rec_text = rec.get('text', '')
            best_match = None
            best_score = threshold
            best_idx = -1
            
            for idx, resp in enumerate(responses):
                if idx in used_responses:
                    continue
                
                resp_text = resp.get('text', '')
                
                # Calculate similarity
                similarity = self._calculate_similarity(rec_text, resp_text)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = resp
                    best_idx = idx
            
            if best_match and best_idx >= 0:
                used_responses.add(best_idx)
                status, confidence, patterns = self.status_classifier.classify(
                    best_match.get('text', '')
                )
            else:
                status, confidence, patterns = 'No Response', 0.0, []
            
            rec_id = rec.get('rec_number') or rec.get('rec_id')
            
            aligned.append({
                'recommendation': rec,
                'response': best_match,
                'rec_id': rec_id,
                'match_method': 'similarity_match' if best_match else 'no_match',
                'match_confidence': best_score if best_match else 0.0,
                'status': status,
                'status_confidence': confidence,
                'status_patterns': patterns,
            })
        
        return aligned
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        
        # Normalise texts
        t1 = ' '.join(text1.lower().split())
        t2 = ' '.join(text2.lower().split())
        
        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, t1, t2).ratio()
    
    def align(
        self,
        recommendations: List[Dict],
        responses: List[Dict],
        method: str = 'auto'
    ) -> List[Dict]:
        """
        Main alignment function.
        
        Args:
            recommendations: List of recommendation dicts
            responses: List of response dicts
            method: Alignment method ('id', 'position', 'similarity', 'auto')
            
        Returns:
            List of aligned recommendation-response pairs with status
        """
        if not recommendations:
            return []
        
        if not responses:
            # No responses - return recommendations with "No Response" status
            return [
                {
                    'recommendation': rec,
                    'response': None,
                    'rec_id': rec.get('rec_number') or rec.get('rec_id'),
                    'match_method': 'no_responses',
                    'match_confidence': 0.0,
                    'status': 'No Response',
                    'status_confidence': 0.0,
                    'status_patterns': [],
                }
                for rec in recommendations
            ]
        
        # Auto-detect best method
        if method == 'auto':
            # Check if we have IDs to match on
            rec_has_ids = any(
                rec.get('rec_number') or rec.get('rec_id')
                for rec in recommendations
            )
            resp_has_ids = any(
                resp.get('rec_id') or resp.get('rec_number')
                for resp in responses
            )
            
            if rec_has_ids and resp_has_ids:
                method = 'id'
            elif len(recommendations) == len(responses):
                method = 'position'
            else:
                method = 'similarity'
            
            logger.info(f"Auto-selected alignment method: {method}")
        
        # Execute alignment
        if method == 'id':
            return self.match_by_id(recommendations, responses)
        elif method == 'position':
            return self.match_by_position(recommendations, responses)
        elif method == 'similarity':
            return self.match_by_similarity(recommendations, responses)
        else:
            raise ValueError(f"Unknown alignment method: {method}")
    
    def get_alignment_statistics(self, aligned: List[Dict]) -> Dict:
        """
        Calculate statistics for alignment results.
        """
        if not aligned:
            return {
                'total': 0,
                'matched': 0,
                'unmatched': 0,
                'match_rate': 0.0,
                'status_counts': {},
                'avg_match_confidence': 0.0,
                'avg_status_confidence': 0.0,
            }
        
        matched = sum(1 for a in aligned if a.get('response'))
        unmatched = len(aligned) - matched
        
        status_counts = {}
        for a in aligned:
            status = a.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        match_confidences = [
            a.get('match_confidence', 0) for a in aligned if a.get('response')
        ]
        status_confidences = [
            a.get('status_confidence', 0) for a in aligned if a.get('response')
        ]
        
        return {
            'total': len(aligned),
            'matched': matched,
            'unmatched': unmatched,
            'match_rate': matched / len(aligned) if aligned else 0.0,
            'status_counts': status_counts,
            'avg_match_confidence': (
                sum(match_confidences) / len(match_confidences)
                if match_confidences else 0.0
            ),
            'avg_status_confidence': (
                sum(status_confidences) / len(status_confidences)
                if status_confidences else 0.0
            ),
        }


# --------------------------------------------------------------------------- #
# Semantic Matching Support
# --------------------------------------------------------------------------- #

class RecommendationResponseMatcher:
    """
    Matcher class for UI integration.
    
    Provides semantic matching when sentence-transformers is available,
    falls back to keyword matching otherwise.
    """
    
    def __init__(self):
        """Initialise matcher with optional transformer support."""
        self.use_transformer = False
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_transformer = True
            logger.info("Initialised with sentence-transformers")
        except ImportError:
            logger.info("sentence-transformers not available, using keyword matching")
    
    def find_best_matches(
        self,
        recommendations: List[Dict],
        responses: List[Dict],
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        Find best response matches for each recommendation.
        
        Returns list of alignment results with recommendation, response,
        similarity score, and status classification.
        """
        results = []
        classifier = StatusClassifier()
        
        for rec in recommendations:
            rec_text = rec.get('text', '') or rec.get('sentence', '')
            
            best_match = None
            best_score = threshold
            
            for resp in responses:
                resp_text = resp.get('text', '') or resp.get('sentence', '')
                
                if self.use_transformer and self.model:
                    # Use semantic similarity
                    score = self._semantic_similarity(rec_text, resp_text)
                else:
                    # Use keyword similarity
                    score = calculate_simple_similarity(rec_text, resp_text)
                
                if score > best_score:
                    best_score = score
                    best_match = resp
            
            if best_match:
                resp_text = best_match.get('text', '') or best_match.get('sentence', '')
                status, confidence, patterns = classifier.classify(resp_text)
                
                results.append({
                    'recommendation': rec,
                    'response': {
                        **best_match,
                        'similarity': best_score,
                        'status': status,
                        'status_confidence': confidence,
                        'match_method': 'semantic' if self.use_transformer else 'keyword',
                    },
                    'has_response': True,
                    'score': best_score,
                })
            else:
                results.append({
                    'recommendation': rec,
                    'response': None,
                    'has_response': False,
                    'score': 0.0,
                })
        
        return results
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if not self.model or not text1 or not text2:
            return 0.0
        
        try:
            embeddings = self.model.encode([text1, text2])
            # Cosine similarity
            from numpy import dot
            from numpy.linalg import norm
            return float(dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1])))
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return calculate_simple_similarity(text1, text2)


# --------------------------------------------------------------------------- #
# Convenience functions
# --------------------------------------------------------------------------- #

def align_recommendations_responses(
    recommendations: List[Dict],
    responses: List[Dict],
    method: str = 'auto'
) -> List[Dict]:
    """
    Convenience function to align recommendations with responses.
    
    This version returns data in format expected by UI components.
    """
    engine = AlignmentEngine()
    aligned = engine.align(recommendations, responses, method)
    
    # Transform to UI-expected format
    results = []
    for item in aligned:
        rec = item.get('recommendation', {})
        resp = item.get('response')
        
        # Build alignment result in UI format
        result = {
            'recommendation': rec,
            'responses': [],
            'action_verb': rec.get('verb', rec.get('pattern', '')),
            'detection_method': rec.get('method', 'unknown'),
            'detection_confidence': rec.get('confidence', 0.5),
            'alignment_status': item.get('status', 'Unknown'),
            'alignment_confidence': item.get('match_confidence', 0.0),
        }
        
        if resp:
            result['responses'].append({
                'response': resp,
                'combined_score': item.get('match_confidence', 0.0),
                'status': item.get('status', 'Unknown'),
            })
        
        results.append(result)
    
    return results


def classify_response_status(response_text: str) -> Tuple[str, float]:
    """
    Convenience function to classify a single response.
    
    Returns (status, confidence) tuple.
    """
    classifier = StatusClassifier()
    status, confidence, _ = classifier.classify(response_text)
    return status, confidence


def get_status_classification_details(response_text: str) -> Dict:
    """
    Get detailed classification for a response.
    """
    classifier = StatusClassifier()
    return classifier.get_status_summary(response_text)


# --------------------------------------------------------------------------- #
# Backward compatibility aliases
# --------------------------------------------------------------------------- #

# Alias for different naming convention
align_recommendations_with_responses = align_recommendations_responses


# --------------------------------------------------------------------------- #
# Module exports
# --------------------------------------------------------------------------- #

__all__ = [
    # Classes
    'StatusClassifier',
    'AlignmentEngine',
    'RecommendationResponseMatcher',
    
    # Main functions
    'align_recommendations_responses',
    'align_recommendations_with_responses',  # Alias
    'classify_response_status',
    'get_status_classification_details',
    
    # Utility functions (RESTORED)
    'calculate_simple_similarity',
    'classify_content_type',
    'determine_alignment_status',
    'find_pattern_matches',
    'extract_response_sentences',
    
    # Constants
    'STOP_WORDS',
]
