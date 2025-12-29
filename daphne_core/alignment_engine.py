"""
Core alignment engine: recommendation/response matching + response extraction.
Moved out of the Streamlit UI to keep logic reusable and testable.

v2.1 - Added organisation-based matching for HSIB documents
v2.0 - Added HSIB/HSSIB response format support
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .search_utils import STOP_WORDS, get_meaningful_words

logger = logging.getLogger(__name__)


# =============================================================================
# ORGANISATION EXTRACTION FOR HSIB MATCHING
# =============================================================================

# Canonical organisation names and their variants
ORGANISATION_MAPPINGS = {
    'nhs_england': [
        r'NHS\s+England',
        r'NHS\s+England\s+and\s+NHS\s+Improvement',
        r'NHSE',
        r'NHS\s+Improvement',
    ],
    'cqc': [
        r'Care\s+Quality\s+Commission',
        r'CQC',
    ],
    'nice': [
        r'National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence',
        r'NICE',
        r'National\s+Institute',
    ],
    'royal_college': [
        r'Royal\s+College\s+of\s+Psychiatrists',
        r'Royal\s+College',
        r'RCPsych',
    ],
    'dhsc': [
        r'Department\s+of\s+Health\s+and\s+Social\s+Care',
        r'DHSC',
        r'Department\s+of\s+Health',
    ],
    'nihr': [
        r'National\s+Institute\s+for\s+Health(?:\s+and\s+Care)?\s+Research',
        r'NIHR',
    ],
}


def extract_target_organisation(rec_text: str) -> Optional[str]:
    """
    Extract the target organisation from a recommendation.
    
    HSIB recommendations typically use patterns like:
    - "HSIB recommends that NHS England..."
    - "HSSIB recommends that the Care Quality Commission..."
    - "It is recommended that NHS England..."
    """
    if not rec_text:
        return None
    
    # Patterns for extracting target org from recommendations
    rec_patterns = [
        r'(?:HSIB|HSSIB)\s+recommends\s+that\s+(?:the\s+)?([^,\.]+?)(?:\s+(?:should|works?|produces?|evaluates?|reviews?|assesses?))',
        r'It\s+is\s+recommended\s+that\s+(?:the\s+)?([^,\.]+?)(?:\s+(?:should|works?|requires?|ensures?))',
        r'(?:Safety\s+)?[Rr]ecommendation[^:]*:\s*(?:HSIB|HSSIB)\s+recommends\s+that\s+(?:the\s+)?([^,\.]+?)(?:\s+(?:should|works?|working))',
        r'recommends\s+that\s+(?:the\s+)?([^,\.]+?)(?:,|\s+(?:should|works?|produces?|evaluates?|reviews?|assesses?|working))',
    ]
    
    for pattern in rec_patterns:
        match = re.search(pattern, rec_text, re.IGNORECASE)
        if match:
            org_text = match.group(1).strip()
            # Map to canonical name
            return _map_to_canonical_org(org_text)
    
    # Fallback: look for organisation names anywhere in text
    for canonical, variants in ORGANISATION_MAPPINGS.items():
        for variant in variants:
            if re.search(variant, rec_text, re.IGNORECASE):
                return canonical
    
    return None


def extract_source_organisation(resp_text: str) -> Optional[str]:
    """
    Extract the source organisation from a response.
    
    Responses typically start with org-specific language like:
    - "NHS England has begun work..."
    - "CQC welcomes the recommendation..."
    - "NICE is currently reviewing..."
    - "We are happy to confirm..." + mentions RCPsych/Expert Working Group
    - "Our current approach to inspection..." (CQC)
    """
    if not resp_text:
        return None
    
    # Check for explicit org mentions at start of response
    resp_starters = [
        (r'^(?:The\s+)?NHS\s+England', 'nhs_england'),
        (r'^(?:The\s+)?Care\s+Quality\s+Commission', 'cqc'),
        (r'^(?:The\s+)?CQC', 'cqc'),
        (r'^(?:The\s+)?National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence', 'nice'),
        (r'^(?:The\s+)?NICE\s+is', 'nice'),
        (r'^(?:The\s+)?Royal\s+College', 'royal_college'),
        (r'^(?:The\s+)?Department\s+of\s+Health', 'dhsc'),
        (r'^(?:The\s+)?DHSC', 'dhsc'),
        (r'^(?:The\s+)?NIHR', 'nihr'),
        (r'^(?:The\s+)?National\s+Institute\s+for\s+Health(?:\s+and\s+Care)?\s+Research', 'nihr'),
        (r'^NHSE\b', 'nhs_england'),
        (r'^Our\s+current\s+approach\s+to\s+inspection', 'cqc'),  # CQC inspection response
        (r'^Funding\s+for\s+Children\s+and\s+Young\s+People', 'nhs_england'),  # NHS context
    ]
    
    for pattern, org in resp_starters:
        if re.match(pattern, resp_text, re.IGNORECASE):
            return org
    
    # Check for characteristic response language
    response_indicators = [
        (r'\bNHS\s+England\s+(?:has|will|is|continues)\b', 'nhs_england'),
        (r'\bNHSE\s+(?:gathered|will|has)\b', 'nhs_england'),
        (r'\bCQC\s+(?:welcomes?|is|has|will)\b', 'cqc'),
        (r'\bCare\s+Quality\s+Commission\s+is\b', 'cqc'),
        (r'\bNICE\s+is\s+currently\b', 'nice'),
        (r'\bNational\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence\s+\(NICE\)\s+is', 'nice'),
        # Royal College indicators - check for Expert Working Group or RCPsych
        (r'\b(?:Expert\s+Working\s+Group|RCPsych)\b', 'royal_college'),
        (r'\bwe\s+have\s+convened\b', 'royal_college'),  # Royal College style
        (r'\bwe\s+are\s+happy\s+to\s+confirm\b.*?(?:menopause|mental\s+health)', 'royal_college'),
        (r'\bNIHR\s+(?:funds?|will)\b', 'nihr'),
        (r'\bfunding\s+for\s+Children\s+and\s+Young\s+People', 'nhs_england'),
        (r'\binspection\s+is\s+determined\s+by\s+the\s+Health\s+and\s+Social\s+Care\s+Act', 'cqc'),  # CQC
    ]
    
    for pattern, org in response_indicators:
        if re.search(pattern, resp_text, re.IGNORECASE):
            return org
    
    return None


def _map_to_canonical_org(org_text: str) -> Optional[str]:
    """Map organisation text to canonical name."""
    if not org_text:
        return None
    
    org_lower = org_text.lower().strip()
    
    for canonical, variants in ORGANISATION_MAPPINGS.items():
        for variant in variants:
            if re.search(variant, org_text, re.IGNORECASE):
                return canonical
    
    # Partial matches
    if 'nhs england' in org_lower or 'nhse' in org_lower:
        return 'nhs_england'
    if 'care quality' in org_lower or 'cqc' in org_lower:
        return 'cqc'
    if 'nice' in org_lower or 'national institute for health and care excellence' in org_lower:
        return 'nice'
    if 'royal college' in org_lower or 'rcpsych' in org_lower:
        return 'royal_college'
    if 'dhsc' in org_lower or 'department of health' in org_lower:
        return 'dhsc'
    if 'nihr' in org_lower or 'national institute for health' in org_lower:
        return 'nihr'
    
    return None


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
        except Exception as e:  # pragma: no cover - defensive
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
        except Exception as e:  # pragma: no cover - defensive
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

        cleaned_responses = []
        for resp in responses:
            cleaned_text = self._clean_response_text(resp["text"])
            cleaned_responses.append({**resp, "cleaned_text": cleaned_text})

        if self.use_transformer and self.model:
            return self._semantic_matching(recommendations, cleaned_responses, top_k)
        return self._keyword_matching(recommendations, cleaned_responses, top_k)

    def _semantic_matching(self, recommendations: List[Dict], responses: List[Dict], top_k: int) -> List[Dict]:
        # Build indexes for different matching strategies
        responses_by_number: Dict[str, List[Dict]] = {}
        responses_by_org: Dict[str, List[Dict]] = {}
        
        for resp in responses:
            # Index by rec_number/rec_id
            rec_num = resp.get("rec_number") or resp.get("rec_id")
            if rec_num:
                normalized = str(rec_num).replace("R/", "").strip()
                responses_by_number.setdefault(rec_num, []).append(resp)
                responses_by_number.setdefault(normalized, []).append(resp)
            
            # Index by source organisation (for HSIB matching)
            source_org = extract_source_organisation(resp.get("cleaned_text", resp.get("text", "")))
            if source_org:
                responses_by_org.setdefault(source_org, []).append(resp)
                resp["source_org"] = source_org  # Store for later use
        
        # Check if this looks like an HSIB document (has org-indexed responses)
        is_hsib_style = len(responses_by_org) > 0 and len(responses_by_number) == 0
        
        if is_hsib_style:
            logger.info(f"Using organisation-based matching. Orgs found: {list(responses_by_org.keys())}")

        rec_texts = [r["text"] for r in recommendations]
        resp_texts = [r["cleaned_text"] for r in responses]

        try:
            rec_embeddings = self.encode_texts(rec_texts)
            resp_embeddings = self.encode_texts(resp_texts)
            if rec_embeddings is None or resp_embeddings is None:
                return self._keyword_matching(recommendations, responses, top_k)

            rec_norms = np.linalg.norm(rec_embeddings, axis=1, keepdims=True)
            resp_norms = np.linalg.norm(resp_embeddings, axis=1, keepdims=True)
            rec_embeddings_norm = rec_embeddings / rec_norms
            resp_embeddings_norm = resp_embeddings / resp_norms
            similarity_matrix = np.dot(rec_embeddings_norm, resp_embeddings_norm.T)

            alignments = []
            for rec_idx, rec in enumerate(recommendations):
                best_match = None
                best_score = 0.0

                rec_num = self._extract_rec_number(rec["text"]) or rec.get("rec_number")
                
                # PRIORITY 1: Try ID-based matching (exact rec number)
                if rec_num:
                    normalized_rec_num = str(rec_num).replace("R/", "").strip()
                    candidates = responses_by_number.get(rec_num, []) + responses_by_number.get(normalized_rec_num, [])
                    
                    for resp in candidates:
                        resp_text = resp["cleaned_text"]
                        if self._is_self_match(rec["text"], resp_text):
                            continue
                        resp_idx = next((i for i, r in enumerate(responses) if r is resp), None)
                        score = similarity_matrix[rec_idx][resp_idx] if resp_idx is not None else 0.7
                        score = min(score + 0.4, 1.0)  # Boost for ID match
                        if score > best_score:
                            best_score = score
                            status, status_conf = self._classify_response_status(resp_text)
                            best_match = {
                                "response_text": resp_text,
                                "similarity": score,
                                "status": status,
                                "status_confidence": status_conf,
                                "source_document": resp.get("source_document", "unknown"),
                                "match_method": "number_match",
                            }

                # PRIORITY 2: Try organisation-based matching (for HSIB documents)
                if best_match is None and is_hsib_style:
                    target_org = extract_target_organisation(rec["text"])
                    if target_org and target_org in responses_by_org:
                        candidates = responses_by_org[target_org]
                        logger.debug(f"Rec targets '{target_org}', found {len(candidates)} candidate responses")
                        
                        for resp in candidates:
                            resp_text = resp["cleaned_text"]
                            if self._is_self_match(rec["text"], resp_text):
                                continue
                            resp_idx = next((i for i, r in enumerate(responses) if r is resp), None)
                            score = similarity_matrix[rec_idx][resp_idx] if resp_idx is not None else 0.6
                            score = min(score + 0.3, 1.0)  # Boost for org match
                            if score > best_score:
                                best_score = score
                                status, status_conf = self._classify_response_status(resp_text)
                                best_match = {
                                    "response_text": resp_text,
                                    "similarity": score,
                                    "status": status,
                                    "status_confidence": status_conf,
                                    "source_document": resp.get("source_document", "unknown"),
                                    "match_method": "org_match",
                                    "matched_org": target_org,
                                }

                # PRIORITY 3: Fallback to pure semantic matching
                if best_match is None:
                    similarities = similarity_matrix[rec_idx]
                    top_indices = np.argsort(similarities)[::-1][: top_k * 2]
                    for resp_idx in top_indices:
                        resp = responses[resp_idx]
                        score = similarities[resp_idx]
                        if score < 0.4:
                            continue
                        if self._is_self_match(rec["text"], resp["cleaned_text"]):
                            continue
                        if is_recommendation_text(resp["cleaned_text"]):
                            continue
                        if not self._has_government_response_language(resp["cleaned_text"]):
                            score *= 0.7
                        if score > best_score:
                            best_score = min(score, 1.0)
                            status, status_conf = self._classify_response_status(resp["cleaned_text"])
                            best_match = {
                                "response_text": resp["cleaned_text"],
                                "similarity": best_score,
                                "status": status,
                                "status_confidence": status_conf,
                                "source_document": resp.get("source_document", "unknown"),
                                "match_method": "semantic",
                            }

                alignments.append(
                    {
                        "recommendation": rec,
                        "response": best_match,
                        "has_response": best_match is not None,
                    }
                )

            return alignments
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Semantic matching failed: {e}")
            return self._keyword_matching(recommendations, responses, top_k)

    def _keyword_matching(self, recommendations: List[Dict], responses: List[Dict], top_k: int) -> List[Dict]:
        # Build org index for HSIB matching
        responses_by_org: Dict[str, List[Dict]] = {}
        for resp in responses:
            source_org = extract_source_organisation(resp.get("cleaned_text", resp.get("text", "")))
            if source_org:
                responses_by_org.setdefault(source_org, []).append(resp)
        
        is_hsib_style = len(responses_by_org) > 0
        
        alignments = []
        for rec in recommendations:
            best_match = None
            best_score = 0.0
            
            # Try organisation-based matching first for HSIB
            if is_hsib_style:
                target_org = extract_target_organisation(rec["text"])
                if target_org and target_org in responses_by_org:
                    for resp in responses_by_org[target_org]:
                        resp_text = resp.get("cleaned_text", resp["text"])
                        if self._is_self_match(rec["text"], resp_text):
                            continue
                        score = self._keyword_similarity(rec["text"], resp_text)
                        score += 0.3  # Boost for org match
                        if self._has_government_response_language(resp_text):
                            score += 0.1
                        if score > best_score and score >= 0.25:
                            best_score = min(score, 1.0)
                            status, status_conf = self._classify_response_status(resp_text)
                            best_match = {
                                "response_text": resp_text,
                                "similarity": best_score,
                                "status": status,
                                "status_confidence": status_conf,
                                "source_document": resp.get("source_document", "unknown"),
                                "match_method": "org_keyword",
                            }
            
            # Fallback to all responses
            if best_match is None:
                for resp in responses:
                    resp_text = resp.get("cleaned_text", resp["text"])
                    if self._is_self_match(rec["text"], resp_text):
                        continue
                    score = self._keyword_similarity(rec["text"], resp_text)
                    rec_num = self._extract_rec_number(rec["text"]) or rec.get("rec_number")
                    resp_num = resp.get("rec_number") or resp.get("rec_id")
                    if rec_num and resp_num:
                        rec_norm = str(rec_num).replace("R/", "").strip()
                        resp_norm = str(resp_num).replace("R/", "").strip()
                        if rec_norm == resp_norm:
                            score += 0.4
                    if self._has_government_response_language(resp_text):
                        score += 0.2
                    if score > best_score and score >= 0.25:
                        best_score = min(score, 1.0)
                        status, status_conf = self._classify_response_status(resp_text)
                        best_match = {
                            "response_text": resp_text,
                            "similarity": best_score,
                            "status": status,
                            "status_confidence": status_conf,
                            "source_document": resp.get("source_document", "unknown"),
                            "match_method": "keyword",
                        }
            
            alignments.append(
                {
                    "recommendation": rec,
                    "response": best_match,
                    "has_response": best_match is not None,
                }
            )
        return alignments

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r"\b\w+\b", text1.lower())) - STOP_WORDS
        words2 = set(re.findall(r"\b\w+\b", text2.lower())) - STOP_WORDS
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        base_score = intersection / union if union > 0 else 0.0

        gov_terms = {
            "government",
            "recommendation",
            "accept",
            "support",
            "implement",
            "policy",
            "department",
            "nhs",
            "england",
            "health",
            "care",
            "provider",
            "commissioner",
            "trust",
            "board",
            "cqc",
        }
        gov_matches = len((words1 & words2) & gov_terms)
        gov_boost = min(gov_matches * 0.05, 0.15)
        return min(base_score + gov_boost, 1.0)

    def _clean_response_text(self, text: str) -> str:
        markers = [
            "Government response to recommendation",
            "government response to recommendation",
            "The government supports",
            "The government accepts",
            "The government agrees",
            "The government notes",
            "The government rejects",
            "The Government supports",
            "The Government accepts",
        ]
        text_lower = text.lower()
        for marker in markers:
            marker_lower = marker.lower()
            pos = text_lower.find(marker_lower)
            if pos > 0:
                return text[pos:]
        return text

    def _is_self_match(self, rec_text: str, resp_text: str) -> bool:
        rec_clean = re.sub(r"\s+", " ", rec_text.lower().strip())
        resp_clean = re.sub(r"\s+", " ", resp_text.lower().strip())
        if rec_clean == resp_clean:
            return True
        if len(rec_clean) > 100 and len(resp_clean) > 100:
            if rec_clean in resp_clean or resp_clean in rec_clean:
                if "government response" in resp_clean[:100] or "the government" in resp_clean[:100]:
                    return False
                return True
        if len(rec_clean) > 150 and len(resp_clean) > 150:
            if rec_clean[:150] == resp_clean[:150]:
                return True
        return False

    def _has_government_response_language(self, text: str) -> bool:
        text_lower = text.lower()
        indicators = [
            "government response",
            "the government supports",
            "the government accepts",
            "the government agrees",
            "the government notes",
            "the government rejects",
            "we accept",
            "we support",
            "we agree",
            "we welcome",
            "accept the recommendation",
            "support the recommendation",
            "accept this recommendation",
            "nhs england has",
            "nhs england will",
            "nhs england is",
            "care quality commission",
            "nice is",
            "we have convened",
            "we are happy to confirm",
        ]
        return any(ind in text_lower for ind in indicators)

    def _extract_rec_number(self, text: str) -> Optional[str]:
        # Try HSIB 2023 format: R/2023/220
        match = re.search(r"R/\d{4}/\d{3}", text)
        if match:
            return match.group(0)
        # Try HSIB 2018 format: 2018/006
        match = re.search(r"\b(\d{4}/\d{3})\b", text)
        if match:
            return match.group(1)
        # Try standard format: recommendation 1
        match = re.search(r"recommendation\s+(\d+)", text.lower())
        if match:
            return match.group(1)
        return None

    def _classify_response_status(self, text: str) -> Tuple[str, float]:
        """
        Classify government response status with context-aware detection.
        """
        text_lower = text.lower()
        
        # PRIORITY 1: Check for full rejection
        full_rejection_patterns = [
            r"\breject(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bdoes?\s+not\s+accept\s+(?:this\s+|the\s+)?recommendation",
            r"\bcannot\s+accept\s+(?:this\s+|the\s+)?recommendation",
        ]
        for pattern in full_rejection_patterns:
            if re.search(pattern, text_lower):
                return "Rejected", 0.9
        
        # PRIORITY 2: Check for partial acceptance
        partial_patterns = [
            r"\baccept(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation\s+)?in\s+(?:part|principle)",
            r"\bpartially\s+accept",
            r"\bsupport(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation\s+)?in\s+principle",
            r"\bbroadly\s+support",
            r"\bwill\s+consider\s+(?:the\s+|this\s+)?(?:recommendation|proposal)",
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partial", 0.85
        
        # PRIORITY 3: Check for explicit acceptance
        accepted_patterns = [
            r"\baccept(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bsupport(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bagree(?:s|d)?\s+(?:with\s+)?(?:this\s+|the\s+)?recommendation",
            r"\bwelcome(?:s|d)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bthe\s+government\s+support(?:s|ed)?\b",
            r"\bthe\s+government\s+accept(?:s|ed)?\b",
            r"\bwill\s+implement\b",
            r"\bfully\s+accept",
            r"\bhas\s+(?:begun|started|commenced)",
        ]
        for pattern in accepted_patterns:
            if re.search(pattern, text_lower):
                return "Accepted", 0.9
        
        # PRIORITY 4: Check for implicit acceptance (taking action)
        implicit_acceptance_patterns = [
            r"\bcommitted\s+to\s+(?:working|improving|delivering|ensuring)",
            r"\bconvened\s+(?:a\s+)?(?:working|expert)\s+group",
            r"\bwe\s+have\s+convened",
            r"\bintends?\s+to\s+(?:provide|publish|deliver|implement)",
            r"\bwill\s+(?:provide|publish|deliver|work\s+with)",
            r"\bhas\s+begun\s+work",
            r"\bwe\s+(?:are|have)\s+(?:developing|establishing|reviewing|currently)",
            r"\bnhs\s+england\s+has\s+(?:begun|written|started)",
            r"\bwe\s+are\s+happy\s+to\s+confirm",
        ]
        for pattern in implicit_acceptance_patterns:
            if re.search(pattern, text_lower):
                return "Accepted", 0.8
        
        # PRIORITY 5: Check for "noted" patterns
        noted_patterns = [
            r"\bnote(?:s|d)?\s+(?:this\s+|the\s+)?recommendation",
            r"\backnowledge(?:s|d)?\s+(?:this\s+|the\s+)?recommendation",
        ]
        for pattern in noted_patterns:
            if re.search(pattern, text_lower):
                return "Noted", 0.75
        
        # Fallback - check for positive language
        if any(word in text_lower for word in ["support", "accept", "welcome", "agree", "committed", "happy to confirm"]):
            return "Accepted", 0.6
        
        return "Unclear", 0.5


# --------------------------------------------------------------------------- #
# Response extraction helpers
# --------------------------------------------------------------------------- #


def is_recommendation_text(text: str) -> bool:
    text_lower = text.lower().strip()
    recommendation_starters = [
        r"^nhs\s+england\s+should",
        r"^providers?\s+should",
        r"^trusts?\s+should",
        r"^boards?\s+should",
        r"^icss?\s+should",
        r"^ics\s+and\s+provider",
        r"^cqc\s+should",
        r"^dhsc\s+should",
        r"^dhsc,?\s+in\s+partnership",
        r"^every\s+provider",
        r"^all\s+providers?\s+should",
        r"^provider\s+boards?\s+should",
        r"^this\s+multi-professional\s+alliance\s+should",
        r"^the\s+review\s+should",
        r"^commissioners?\s+should",
        r"^regulators?\s+should",
        r"^recommendation\s+\d+\s+[a-z]",
        r"^we\s+recommend",
        r"^hsib\s+recommends",
        r"^it\s+should\s+also",
        r"^they\s+should",
        r"^this\s+forum\s+should",
        r"^this\s+programme\s+should",
        r"^these\s+systems\s+should",
        r"^the\s+digital\s+platforms",
        r"^the\s+output\s+of\s+the",
        r"^including,?\s+where\s+appropriate",
        r"^to\s+facilitate\s+this",
    ]
    return any(re.search(pattern, text_lower) for pattern in recommendation_starters)


def is_genuine_response(text: str) -> bool:
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
        r"^we\s+are\s+happy\s+to\s+confirm",
        r"^dhsc\s+and\s+nhs\s+england\s+support",
        r"^nhs\s+england\s+(?:has|will|is)",
        r"^care\s+quality\s+commission\s+(?:is|has|will)",
        r"^the\s+national\s+institute\s+for\s+health",
        r"^nice\s+(?:is|has|will)",
    ]
    return any(re.search(pattern, text_lower) for pattern in strong_starters)


def has_pdf_artifacts(text: str) -> bool:
    artifacts = [
        r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?",
        r"GOV\.UK",
        r"https?://www\.gov\.uk",
        r"https?://www\.england\.nhs\.uk",
        r"\d+/\d+\s*$",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in artifacts)


def clean_pdf_artifacts(text: str) -> str:
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?\s*", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"Government response to the rapid review.*?GOV\.UK[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# HSIB Response Extraction (NEW)
# --------------------------------------------------------------------------- #

def is_hsib_response_document(text: str) -> bool:
    """
    Detect if a document is an HSIB/HSSIB-style response document.
    
    HSIB response documents use standalone "Response" headers rather than
    "Government response to recommendation N" format.
    """
    # Check for standard government format first - if present, not HSIB
    gov_response_pattern = re.compile(
        r'Government\s+response\s+to\s+recommendation\s+\d+',
        re.IGNORECASE
    )
    if len(list(gov_response_pattern.finditer(text))) >= 2:
        return False
    
    # Count standalone "Response" headers
    # NOTE: PyPDF2 sometimes merges "Response" with following text (no newline),
    # so we also match Response followed by uppercase letter
    response_header_pattern = re.compile(
        r'(?:^|\n)\s*Response\s*(?:\n|(?=[A-Z]))(?! ?received)',
        re.IGNORECASE | re.MULTILINE
    )
    response_count = len(list(response_header_pattern.finditer(text)))
    
    logger.info(f"HSIB detection: found {response_count} Response headers")
    
    # If 3+ standalone Response headers, it's likely HSIB format
    if response_count >= 3:
        logger.info("Detected HSIB format: 3+ Response headers")
        return True
    
    # Check for other HSIB indicators
    hsib_indicators = [
        r'\bHSIB\s+recommends\b',
        r'\bHSSIB\s+recommends\b',
        r'\bSafety\s+recommendation\s+R/\d{4}/\d{3}\b',
        r'\bRecommendation\s+\d{4}/\d{3}\b',
        r'\bIt\s+is\s+recommended\s+that\b',  # HSIB style recommendation text
        r'\bSafety\s+recommendations\b',  # Section header in HSIB docs
        r'\bResponse\s+received\s+on\b',  # HSIB response metadata
        r'\bTimescale:\s*(?:completed|already|Spring|January|December)\b',  # HSIB timescale format
    ]
    
    indicator_matches = sum(1 for pattern in hsib_indicators 
                           if re.search(pattern, text, re.IGNORECASE | re.MULTILINE))
    
    logger.info(f"HSIB detection: found {indicator_matches} HSIB indicators")
    
    # If we have ANY Response headers + HSIB indicators, it's HSIB
    # Also trigger if we have multiple strong HSIB indicators even without Response headers
    if response_count >= 1 and indicator_matches >= 1:
        logger.info("Detected HSIB format: Response headers + indicators")
        return True
    
    if indicator_matches >= 3:
        logger.info("Detected HSIB format: 3+ indicators")
        return True
    
    # Final check: count simple "Response" followed by newline anywhere
    # This catches cases where the regex is too strict
    simple_response_count = len(re.findall(r'\nResponse\n', text, re.IGNORECASE))
    if simple_response_count >= 3:
        logger.info(f"Detected HSIB format: {simple_response_count} simple Response headers")
        return True
    
    return False


def extract_hsib_responses(text: str) -> List[Dict]:
    """
    Extract responses from HSIB-format response documents.
    
    HSIB format uses:
    - Organisation name as section header
    - "Response" as standalone header
    - Responses follow immediately after the recommendation they address
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
    # NOTE: PyPDF2 sometimes merges "Response" with following text (no newline),
    # e.g. "ResponseNHS England..." or "Responsesafe transition..."
    # So we match: Response followed by newline OR followed by uppercase letter
    # But exclude "Response received on..." which is metadata, not a response header
    response_header_pattern = re.compile(
        r'(?:^|\n)\s*Response\s*(?:\n|(?=[A-Z]))(?! ?received)',
        re.IGNORECASE | re.MULTILINE
    )
    response_matches = list(response_header_pattern.finditer(text))
    
    # Organisation headers that act as section boundaries
    org_header_pattern = re.compile(
        r'(?:^|\n)\s*(Department\s+of\s+Health\s+and\s+Social\s+Care|NHS\s+England(?:\s+and\s+NHS\s+Improvement)?|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|National\s+Institute|Royal\s+College\s+of\s+Psychiatrists|Royal\s+College|DHSC|NICE|CQC)\s*\n',
        re.IGNORECASE | re.MULTILINE
    )
    org_headers = list(org_header_pattern.finditer(text))
    
    # Other boundary markers
    boundary_markers = [
        r'\n\s*Actions\s+planned\s+to\s+deliver',
        r'\n\s*Safety\s+(?:observation|action)',
        r'\n\s*Response\s+received\s+on',
    ]
    
    logger.info(f"HSIB: Found {len(rec_matches)} recommendations and {len(response_matches)} response headers")
    
    # Extract response text for each "Response" header
    for idx, resp_match in enumerate(response_matches):
        start = resp_match.end()
        end = len(text)
        
        # Check for next "Response" header
        if idx + 1 < len(response_matches):
            end = min(end, response_matches[idx + 1].start())
        
        # Check for organisation headers
        for org_match in org_headers:
            if org_match.start() > start and org_match.start() < end:
                end = org_match.start()
                break
        
        # Check for other boundary markers
        for boundary in boundary_markers:
            boundary_match = re.search(boundary, text[start:end], re.IGNORECASE)
            if boundary_match:
                potential_end = start + boundary_match.start()
                if potential_end > start + 50:
                    end = min(end, potential_end)
        
        response_text = text[start:end].strip()
        response_text = re.sub(r'\s+', ' ', response_text)
        
        if len(response_text) < 30:
            continue
        
        # Find the org header that precedes this response
        prev_org_pos = 0
        for org_match in org_headers:
            if org_match.start() < resp_match.start():
                prev_org_pos = org_match.start()
            else:
                break
        
        # Find recommendation between prev_org_pos and this response
        rec_id = None
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


# --------------------------------------------------------------------------- #
# Main Response Extraction Function
# --------------------------------------------------------------------------- #

def extract_response_sentences(text: str) -> List[Dict]:
    """
    Extract sentences that are government responses.
    Auto-detects HSIB vs standard government format.
    """
    if not text:
        return []

    # Check for HSIB format first
    if is_hsib_response_document(text):
        logger.info("Detected HSIB response document format")
        return extract_hsib_responses(text)

    # Standard government format extraction
    logger.info("Using standard government response extraction")
    
    text = clean_pdf_artifacts(text)
    responses: List[Dict] = []
    seen_responses = set()

    gov_resp_starts = []
    for match in re.finditer(r"Government\s+response\s+to\s+recommendation\s+(\d+)", text, re.IGNORECASE):
        gov_resp_starts.append({"pos": match.start(), "end": match.end(), "rec_num": match.group(1)})

    rec_positions = []
    for match in re.finditer(r"Recommendation\s+(\d+)\s+([A-Z])", text):
        rec_positions.append({"pos": match.start(), "rec_num": match.group(1)})

    logger.info(f"Found {len(gov_resp_starts)} government response headers")
    logger.info(f"Found {len(rec_positions)} recommendation markers")

    structured_spans: List[Tuple[int, int]] = []
    structured_texts: List[str] = []

    for i, gov_resp in enumerate(gov_resp_starts):
        start_pos = gov_resp["end"]
        rec_num = gov_resp["rec_num"]
        end_pos = len(text)

        if i + 1 < len(gov_resp_starts):
            next_gov = gov_resp_starts[i + 1]["pos"]
            if next_gov < end_pos:
                end_pos = next_gov

        for rec_pos in rec_positions:
            if rec_pos["pos"] > start_pos and rec_pos["pos"] < end_pos:
                end_pos = rec_pos["pos"]
                break

        resp_content = text[start_pos:end_pos].strip()
        if not resp_content:
            continue
        resp_content = clean_pdf_artifacts(resp_content)

        responses.append(
            {
                "text": resp_content,
                "position": start_pos,
                "response_type": "structured",
                "confidence": 0.95,
                "rec_number": rec_num,
            }
        )
        structured_spans.append((start_pos, end_pos))
        structured_texts.append(resp_content)

    # Sentence-level responses
    sentence_spans = []
    start_idx = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end_idx = match.end()
        sentence_spans.append((start_idx, end_idx))
        start_idx = end_idx
    if start_idx < len(text):
        sentence_spans.append((start_idx, len(text)))

    for idx, (start_idx, end_idx) in enumerate(sentence_spans):
        sentence_raw = text[start_idx:end_idx]
        sentence_clean = clean_pdf_artifacts(sentence_raw.strip())
        if len(sentence_clean) < 40 or len(sentence_clean) > 500:
            continue
        if is_recommendation_text(sentence_clean):
            continue
        if not is_genuine_response(sentence_clean):
            continue
        if any(start <= start_idx < end for start, end in structured_spans):
            continue

        sent_norm = re.sub(r"\s+", " ", sentence_clean.lower()).strip()
        is_struct_dup = False
        for struct_text in structured_texts:
            struct_norm = re.sub(r"\s+", " ", struct_text.lower()).strip()
            if not struct_norm:
                continue
            if sent_norm and sent_norm in struct_norm:
                is_struct_dup = True
                break
            sent_tokens = set(sent_norm.split())
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & set(struct_norm.split())) / len(sent_tokens)
            if overlap >= 0.8:
                is_struct_dup = True
                break
        if is_struct_dup:
            continue

        resp_key = sentence_clean[:100].lower()
        if resp_key in seen_responses:
            continue
        seen_responses.add(resp_key)
        rec_ref = re.search(r"recommendation\s+(\d+)", sentence_clean.lower())
        responses.append(
            {
                "text": sentence_clean,
                "position": idx,
                "response_type": "sentence",
                "confidence": 0.85,
                "rec_number": rec_ref.group(1) if rec_ref else None,
            }
        )

    logger.info(f"Extracted {len(responses)} genuine responses")
    return responses


__all__ = [
    "RecommendationResponseMatcher",
    "is_recommendation_text",
    "is_genuine_response",
    "has_pdf_artifacts",
    "clean_pdf_artifacts",
    "extract_response_sentences",
    "is_hsib_response_document",
    "extract_hsib_responses",
    "extract_target_organisation",
    "extract_source_organisation",
    "find_pattern_matches",
    "align_recommendations_with_responses",
    "determine_alignment_status",
    "calculate_simple_similarity",
    "classify_content_type",
]

# --------------------------------------------------------------------------- #
# Lightweight keyword-based response finding and alignment (exposed for UI)
# --------------------------------------------------------------------------- #


def classify_content_type(sentence: str) -> str:
    """Classify sentence content type based on common policy categories."""
    sentence_lower = sentence.lower()
    if any(word in sentence_lower for word in ["implementation", "deploy", "roll out", "deliver"]):
        return "Implementation"
    if any(word in sentence_lower for word in ["monitor", "measure", "evaluate", "review"]):
        return "Monitoring"
    if any(word in sentence_lower for word in ["training", "education", "workforce"]):
        return "Workforce"
    if any(word in sentence_lower for word in ["technology", "digital", "data", "system"]):
        return "Technology"
    if any(word in sentence_lower for word in ["policy", "framework", "guideline"]):
        return "Policy"
    if any(word in sentence_lower for word in ["financial", "budget", "cost"]):
        return "Financial"
    return "General"


def find_pattern_matches(documents: List[Dict[str, Any]], patterns: List[str], match_type: str) -> List[Dict[str, Any]]:
    """Find sentences containing any of the provided patterns."""
    matches: List[Dict[str, Any]] = []
    normalized = [p.lower() for p in patterns]

    for doc in documents:
        text = doc.get("text", "")
        if not text:
            continue

        for sentence in re.split(r"[.!?]+", text):
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            sentence_lower = sentence_clean.lower()
            if any(pat in sentence_lower for pat in normalized):
                pos = text.find(sentence_clean)
                matches.append(
                    {
                        "sentence": sentence_clean,
                        "document": doc,
                        "pattern": match_type,
                        "position": pos,
                        "page_number": max(1, pos // 2000 + 1) if pos >= 0 else 1,
                        "content_type": classify_content_type(sentence_clean),
                    }
                )

    return matches


def determine_alignment_status(responses: List[Dict[str, Any]]) -> str:
    """Heuristic alignment status for a recommendation."""
    if not responses:
        return "No response found"
    top_response = responses[0]
    if top_response.get("same_document"):
        return "Direct response"
    if top_response.get("combined_score", 0) >= 0.5:
        return "Possible response"
    return "Uncertain"


def align_recommendations_with_responses(
    recommendations: List[Dict[str, Any]], responses: List[Dict[str, Any]], similarity_threshold: float
) -> List[Dict[str, Any]]:
    """Align recommendations with responses using meaningful-word similarity."""
    alignments = []
    for rec in recommendations:
        rec_doc = rec["document"]["filename"]
        best_responses = []
        for resp in responses:
            similarity = calculate_simple_similarity(rec["sentence"], resp["sentence"])
            if resp["document"]["filename"] == rec_doc:
                similarity *= 1.2
            if resp.get("document", {}).get("filename") == rec_doc and resp.get("position", 0) > rec.get("position", 0):
                similarity *= 1.1
            if similarity >= similarity_threshold:
                best_responses.append(
                    {
                        "response": resp,
                        "combined_score": min(similarity, 1.0),
                        "similarity_score": similarity,
                        "same_document": resp["document"]["filename"] == rec_doc,
                    }
                )
        best_responses.sort(key=lambda x: x["combined_score"], reverse=True)
        alignment = {
            "recommendation": rec,
            "responses": best_responses[:3],
            "alignment_confidence": best_responses[0]["combined_score"] if best_responses else 0,
            "alignment_status": determine_alignment_status(best_responses),
            "detection_method": rec.get("method", "unknown"),
            "detection_confidence": rec.get("confidence", 0),
            "action_verb": rec.get("verb", "unknown"),
        }
        alignments.append(alignment)
    alignments.sort(key=lambda x: x["detection_confidence"], reverse=True)
    return alignments


def calculate_simple_similarity(text1: str, text2: str) -> float:
    """Calculate similarity using meaningful words only."""
    words1 = set(get_meaningful_words(text1))
    words2 = set(get_meaningful_words(text2))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0
