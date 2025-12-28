"""
Alignment Engine v2.0
Core alignment engine: recommendation/response matching + response extraction.
Moved out of the Streamlit UI to keep logic reusable and testable.

v2.0 Changes:
- Added HSIB response format support (Safety recommendation R/YYYY/NNN)
- Auto-detects document format and uses appropriate extraction method
- Improved response boundary detection
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .search_utils import STOP_WORDS, get_meaningful_words
except ImportError:
    # Fallback for standalone testing
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
    def get_meaningful_words(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in STOP_WORDS and len(w) > 2]

logger = logging.getLogger(__name__)


# =============================================================================
# HSIB RESPONSE EXTRACTION
# =============================================================================

def is_hsib_document(text: str) -> bool:
    """
    Detect if a document is an HSIB/HSSIB-style document.
    """
    hsib_indicators = [
        r'\bHSIB\s+recommends\b',
        r'\bHSSIB\s+recommends\b', 
        r'\bSafety\s+recommendation\s+R/\d{4}/\d{3}\b',
        r'\bSafety\s+observation\s+O/\d{4}/\d{3}\b',
        r'\bRecommendation\s+\d{4}/\d{3}\b',  # HSIB 2018 format
        r'(?:^|\n)\s*Response\s*\n',  # Standalone "Response" header
    ]
    
    matches = sum(1 for pattern in hsib_indicators 
                  if re.search(pattern, text, re.IGNORECASE | re.MULTILINE))
    
    return matches >= 2


def extract_hsib_responses(text: str) -> List[Dict]:
    """
    Extract responses from HSIB-format response documents.
    
    HSIB format:
    - "Safety recommendation R/YYYY/NNN: [rec text]" OR "Recommendation YYYY/NNN:"
    - "Response" header followed by response text
    - Responses are positionally linked to the preceding recommendation
    """
    if not text:
        return []
    
    responses = []
    
    # Pattern to find HSIB recommendation markers
    hsib_rec_patterns = [
        re.compile(r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]', re.IGNORECASE),
        re.compile(r'Recommendation\s+(\d{4}/\d{3})[:\s]', re.IGNORECASE),
    ]
    
    # Find all recommendation positions
    rec_matches = []
    for pattern in hsib_rec_patterns:
        rec_matches.extend(list(pattern.finditer(text)))
    rec_matches.sort(key=lambda m: m.start())
    
    # Pattern to find "Response" headers
    response_header_pattern = re.compile(
        r'(?:^|\n)\s*Response\s*(?:\n|$)',
        re.IGNORECASE | re.MULTILINE
    )
    
    response_matches = list(response_header_pattern.finditer(text))
    
    logger.info(f"HSIB: Found {len(rec_matches)} recommendations and {len(response_matches)} response headers")
    
    # Organisation names that act as section headers (boundaries between rec/response pairs)
    # These appear on their own line before each recommendation
    org_header_pattern = re.compile(
        r'\n\s*(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|Royal\s+College\s+of\s+Psychiatrists)\s*\n',
        re.IGNORECASE
    )
    org_headers = list(org_header_pattern.finditer(text))
    
    # Other boundary markers
    boundary_markers = [
        r'\n\s*Actions\s+planned\s+to\s+deliver',
        r'\n\s*Safety\s+(?:observation|action)',
        r'\n\s*Response\s+received\s+on',
        r'\n\s*Original\s+response\s+received',
    ]
    
    # Extract response text for each "Response" header
    for idx, resp_match in enumerate(response_matches):
        start = resp_match.end()
        
        # End is either the next "Response" header, boundary marker, or end of text
        # NOTE: We do NOT use next recommendation as boundary because the Response
        # section follows the recommendation it addresses
        end = len(text)
        
        # Check for next "Response" header (primary boundary)
        if idx + 1 < len(response_matches):
            end = min(end, response_matches[idx + 1].start())
        
        # Check for organisation headers (they indicate new recommendation/response pairs)
        for org_match in org_headers:
            if org_match.start() > start and org_match.start() < end:
                end = org_match.start()
                break
        
        # Check for other boundary markers
        for boundary in boundary_markers:
            boundary_match = re.search(boundary, text[start:end], re.IGNORECASE)
            if boundary_match:
                potential_end = start + boundary_match.start()
                if potential_end > start + 50:  # Ensure we have meaningful content
                    end = min(end, potential_end)
        
        response_text = text[start:end].strip()
        
        # Clean up
        response_text = re.sub(r'\s+', ' ', response_text)
        
        if len(response_text) < 50:
            continue
        
        # Determine which recommendation this response is for
        # Find the recommendation that appears in the same section (between prev org header and this response)
        rec_id = None
        
        # Find the org header that precedes this response
        prev_org_pos = 0
        for org_match in org_headers:
            if org_match.start() < resp_match.start():
                prev_org_pos = org_match.start()
            else:
                break
        
        # Find recommendation between prev_org_pos and this response
        for rec_match in rec_matches:
            if prev_org_pos <= rec_match.start() < resp_match.start():
                rec_id = rec_match.group(1)
        
        responses.append({
            'text': response_text,
            'position': start,
            'response_type': 'hsib_structured',
            'confidence': 0.95,
            'rec_number': rec_id,
            'rec_id': rec_id,
        })
    
    logger.info(f"Extracted {len(responses)} HSIB responses")
    return responses


# =============================================================================
# STANDARD GOVERNMENT RESPONSE EXTRACTION  
# =============================================================================

def is_recommendation_text(text: str) -> bool:
    """Check if text appears to be a recommendation rather than a response."""
    text_lower = text.lower().strip()
    recommendation_starters = [
        r"^nhs\s+england\s+should",
        r"^providers?\s+should",
        r"^trusts?\s+should",
        r"^boards?\s+should",
        r"^icss?\s+should",
        r"^cqc\s+should",
        r"^dhsc\s+should",
        r"^every\s+provider",
        r"^all\s+providers?\s+should",
        r"^recommendation\s+\d+",
        r"^we\s+recommend",
        r"^hsib\s+recommends",
    ]
    return any(re.search(pattern, text_lower) for pattern in recommendation_starters)


def is_genuine_response(text: str) -> bool:
    """Check if text appears to be a genuine government response."""
    text_lower = text.lower().strip()
    strong_starters = [
        r"^government\s+response",
        r"^the\s+government\s+supports?",
        r"^the\s+government\s+accepts?",
        r"^the\s+government\s+agrees?",
        r"^the\s+government\s+notes?",
        r"^the\s+government\s+rejects?",
        r"^the\s+government\s+recognises?",
        r"^the\s+government\s+is\s+committed",
        r"^the\s+government\s+will",
        r"^we\s+support",
        r"^we\s+accept",
        r"^we\s+agree",
        r"^we\s+welcome",
        r"^nhs\s+england\s+(?:has|will|is|supports?|accepts?)",
        r"^care\s+quality\s+commission\s+(?:is|has|will|welcomes?)",
        r"^nice\s+(?:is|has|will)",
    ]
    return any(re.search(pattern, text_lower) for pattern in strong_starters)


def clean_pdf_artifacts(text: str) -> str:
    """Remove common PDF extraction artifacts."""
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?\s*", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"GOV\.UK", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_government_responses(text: str) -> List[Dict]:
    """
    Extract responses from standard government response documents.
    Format: "Government response to recommendation N"
    """
    if not text:
        return []

    text = clean_pdf_artifacts(text)
    responses: List[Dict] = []
    seen_responses = set()

    # Find "Government response to recommendation N" headers
    gov_resp_pattern = re.compile(
        r"Government\s+response\s+to\s+recommendation\s+(\d+)",
        re.IGNORECASE
    )
    gov_resp_starts = list(gov_resp_pattern.finditer(text))
    
    # Find recommendation markers for boundary detection
    rec_pattern = re.compile(r"Recommendation\s+(\d+)\s+([A-Z])", re.IGNORECASE)
    rec_positions = list(rec_pattern.finditer(text))

    logger.info(f"Standard: Found {len(gov_resp_starts)} government response headers")

    for i, gov_resp in enumerate(gov_resp_starts):
        start_pos = gov_resp.end()
        rec_num = gov_resp.group(1)
        end_pos = len(text)

        # End at next government response
        if i + 1 < len(gov_resp_starts):
            end_pos = min(end_pos, gov_resp_starts[i + 1].start())

        # End at next recommendation
        for rec_pos in rec_positions:
            if rec_pos.start() > start_pos and rec_pos.start() < end_pos:
                end_pos = rec_pos.start()
                break

        resp_content = text[start_pos:end_pos].strip()
        resp_content = clean_pdf_artifacts(resp_content)
        
        if not resp_content or len(resp_content) < 50:
            continue

        responses.append({
            "text": resp_content,
            "position": start_pos,
            "response_type": "structured",
            "confidence": 0.95,
            "rec_number": rec_num,
        })

    # Fallback: sentence-level response detection
    if not responses:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for idx, sentence in enumerate(sentences):
            sentence_clean = clean_pdf_artifacts(sentence.strip())
            if len(sentence_clean) < 50 or len(sentence_clean) > 500:
                continue
            if is_recommendation_text(sentence_clean):
                continue
            if not is_genuine_response(sentence_clean):
                continue
            
            resp_key = sentence_clean[:100].lower()
            if resp_key in seen_responses:
                continue
            seen_responses.add(resp_key)
            
            rec_ref = re.search(r"recommendation\s+(\d+)", sentence_clean.lower())
            responses.append({
                "text": sentence_clean,
                "position": idx,
                "response_type": "sentence",
                "confidence": 0.85,
                "rec_number": rec_ref.group(1) if rec_ref else None,
            })

    logger.info(f"Extracted {len(responses)} government responses")
    return responses


# =============================================================================
# UNIFIED RESPONSE EXTRACTION
# =============================================================================

def extract_response_sentences(text: str) -> List[Dict]:
    """
    Extract responses from any document format.
    Auto-detects HSIB vs standard government format.
    """
    if not text:
        return []
    
    # Detect document type and use appropriate extraction
    if is_hsib_document(text):
        logger.info("Detected HSIB document format")
        return extract_hsib_responses(text)
    else:
        logger.info("Using standard government response extraction")
        return extract_government_responses(text)


# =============================================================================
# MATCHER CLASS
# =============================================================================

class RecommendationResponseMatcher:
    """
    Semantic/keyword matcher for recommendation-response alignment.
    Falls back to keyword matching when sentence-transformers are unavailable.
    """

    def __init__(self):
        self.model = None
        self.use_transformer = False
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cpu"
            torch.set_default_device("cpu")
            self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
            self.use_transformer = True
            logger.info("✅ Sentence transformer model loaded (BAAI/bge-small-en-v1.5)")
        except ImportError:
            logger.warning("sentence-transformers not installed, using keyword matching")
            self.use_transformer = False
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            self.use_transformer = False

    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.use_transformer or not self.model:
            return None
        try:
            import torch

            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_tensor=False, batch_size=16)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return None

    def calculate_similarity(self, text1: str, text2: str) -> float:
        if self.use_transformer and self.model:
            try:
                embeddings = self.encode_texts([text1, text2])
                if embeddings is not None:
                    similarity = np.dot(embeddings[0], embeddings[1]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                    )
                    return float(similarity)
            except Exception as e:
                logger.warning(f"Transformer similarity failed: {e}")
        return self._keyword_similarity(text1, text2)

    def find_best_matches(self, recommendations: List[Dict], responses: List[Dict], top_k: int = 3) -> List[Dict]:
        if not recommendations or not responses:
            return []

        # Build response index by rec_number/rec_id
        responses_by_id: Dict[str, List[Dict]] = {}
        for resp in responses:
            rec_id = resp.get("rec_number") or resp.get("rec_id")
            if rec_id:
                # Normalize ID (handle both "R/2023/220" and "2023/220" and "1")
                normalized_id = str(rec_id).replace("R/", "").strip()
                responses_by_id.setdefault(normalized_id, []).append(resp)
                responses_by_id.setdefault(rec_id, []).append(resp)

        alignments = []
        
        for rec in recommendations:
            best_match = None
            best_score = 0.0
            
            # Get rec_number from recommendation
            rec_num = rec.get("rec_number")
            if rec_num:
                # Normalize for matching
                normalized_rec_num = str(rec_num).replace("R/", "").strip()
                
                # Try to find matching response by ID
                candidate_responses = (
                    responses_by_id.get(rec_num, []) + 
                    responses_by_id.get(normalized_rec_num, [])
                )
                
                for resp in candidate_responses:
                    score = 0.9  # High base score for ID match
                    # Boost with semantic similarity
                    sim = self._keyword_similarity(rec.get("text", ""), resp.get("text", ""))
                    score = min(score + sim * 0.1, 1.0)
                    
                    if score > best_score:
                        best_score = score
                        status, status_conf = self._classify_response_status(resp.get("text", ""))
                        best_match = {
                            "response_text": resp.get("text", ""),
                            "similarity": score,
                            "status": status,
                            "status_confidence": status_conf,
                            "source_document": resp.get("source_document", "unknown"),
                            "match_method": "id_match",
                        }
            
            # Fallback to semantic/keyword matching if no ID match
            if best_match is None:
                for resp in responses:
                    resp_text = resp.get("text", "")
                    if is_recommendation_text(resp_text):
                        continue
                    
                    score = self._keyword_similarity(rec.get("text", ""), resp_text)
                    if is_genuine_response(resp_text):
                        score += 0.15
                    
                    if score > best_score and score >= 0.3:
                        best_score = score
                        status, status_conf = self._classify_response_status(resp_text)
                        best_match = {
                            "response_text": resp_text,
                            "similarity": min(score, 1.0),
                            "status": status,
                            "status_confidence": status_conf,
                            "source_document": resp.get("source_document", "unknown"),
                            "match_method": "keyword",
                        }

            alignments.append({
                "recommendation": rec,
                "response": best_match,
                "has_response": best_match is not None,
            })

        return alignments

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r"\b\w+\b", text1.lower())) - STOP_WORDS
        words2 = set(re.findall(r"\b\w+\b", text2.lower())) - STOP_WORDS
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _classify_response_status(self, text: str) -> Tuple[str, float]:
        """Classify government/organisation response status."""
        text_lower = text.lower()
        
        # Check for rejection
        rejection_patterns = [
            r"\breject(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bdoes?\s+not\s+accept\s+(?:this\s+|the\s+)?recommendation",
        ]
        for pattern in rejection_patterns:
            if re.search(pattern, text_lower):
                return "Rejected", 0.9
        
        # Check for partial acceptance
        partial_patterns = [
            r"\baccept(?:s|ed)?\s+in\s+(?:part|principle)",
            r"\bpartially\s+accept",
            r"\bbroadly\s+support",
            r"\bwill\s+consider",
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partial", 0.85
        
        # Check for explicit acceptance
        accepted_patterns = [
            r"\baccept(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bsupport(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bagree(?:s|d)?\s+(?:with\s+)?(?:this\s+|the\s+)?recommendation",
            r"\bwelcome(?:s|d)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bwill\s+implement",
            r"\bhas\s+(?:begun|started|commenced)",
        ]
        for pattern in accepted_patterns:
            if re.search(pattern, text_lower):
                return "Accepted", 0.9
        
        # Check for implicit acceptance (taking action)
        action_patterns = [
            r"\bcommitted\s+to\s+(?:working|improving|delivering)",
            r"\bconvened\s+(?:a\s+)?(?:working|expert)\s+group",
            r"\bwill\s+(?:provide|publish|deliver|work)",
            r"\bhas\s+begun\s+work",
            r"\bwe\s+(?:are|have)\s+(?:developing|establishing|reviewing)",
        ]
        for pattern in action_patterns:
            if re.search(pattern, text_lower):
                return "Accepted", 0.8
        
        # Check for "noted"
        if re.search(r"\bnote(?:s|d)?\s+(?:this\s+|the\s+)?recommendation", text_lower):
            return "Noted", 0.75
        
        # Default - check for positive language
        if any(word in text_lower for word in ["support", "accept", "welcome", "agree", "committed"]):
            return "Accepted", 0.6
        
        return "Unclear", 0.5


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RecommendationResponseMatcher",
    "extract_response_sentences",
    "extract_hsib_responses",
    "extract_government_responses",
    "is_hsib_document",
    "is_recommendation_text",
    "is_genuine_response",
    "clean_pdf_artifacts",
]
