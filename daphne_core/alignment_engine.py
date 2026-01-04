"""
Core alignment engine: recommendation/response matching.

This module handles:
- Semantic and keyword-based matching of recommendations to responses
- Response status classification (Accepted/Rejected/Partial)
- Alignment scoring and ranking

v2.6 Changes:
- ADDED: Organisation mismatch detection - recommendations directed to other orgs marked as "No response from this organisation"
- ADDED: extract_recommendation_target_org() to determine who rec is directed to
- ADDED: get_response_document_org() to determine who authored response document
- FIXED: Semantic fallback no longer matches inappropriate responses for other-org recs

v2.5 Changes (preserved):
- org_match requires BOTH org match AND semantic relevance >= 0.55
- number_match properly validates rec_id format matching
- Response extraction moved to response_extractor.py

Response extraction has been moved to response_extractor.py but is
re-exported here for backwards compatibility.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .search_utils import STOP_WORDS, get_meaningful_words

# Import from response_extractor and re-export for backwards compatibility
from .response_extractor import (
    is_recommendation_text,
    is_genuine_response,
    has_pdf_artifacts,
    clean_pdf_artifacts,
    is_hsib_response_document,
    is_trust_response_document,
    is_org_based_hsib_response,
    extract_target_org_from_text,
    extract_recommendation_target_org,
    get_response_document_org,
    extract_hsib_responses,
    extract_trust_responses,
    extract_org_based_hsib_responses,
    extract_response_sentences,
)

logger = logging.getLogger(__name__)


class RecommendationResponseMatcher:
    """
    Semantic/keyword matcher for recommendation-response alignment.
    Falls back to keyword matching when sentence-transformers are unavailable.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialise matcher with optional transformer model."""
        self.model = None
        self.use_transformer = False
        self.response_doc_org = None  # Track which org authored responses
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_transformer = True
            logger.info(f"✅ Sentence transformer model loaded ({model_name})")
        except Exception as e:
            logger.warning(f"Sentence transformers not available: {e}")
            self.use_transformer = False

    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode texts to embeddings."""
        if not self.use_transformer or not texts:
            return None
        try:
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return None

    def find_best_matches(
        self,
        recommendations: List[Dict],
        responses: List[Dict],
        top_k: int = 1,
        response_doc_text: str = None,
    ) -> List[Dict]:
        """
        Find best response matches for each recommendation.
        
        v2.6: Added response_doc_text parameter to detect responding organisation
        """
        if not recommendations:
            return []
        if not responses:
            # Return all recs with no match
            return [
                {
                    "recommendation": rec,
                    "response_text": None,
                    "similarity": 0.0,
                    "status": "No response found",
                    "status_confidence": 1.0,
                    "match_method": "none",
                }
                for rec in recommendations
            ]

        # v2.6: Detect which organisation authored the response document
        if response_doc_text:
            self.response_doc_org = get_response_document_org(response_doc_text)
            logger.info(f"Response document organisation: {self.response_doc_org}")

        # Clean response texts
        for resp in responses:
            resp["cleaned_text"] = clean_pdf_artifacts(resp.get("text", ""))

        if self.use_transformer:
            return self._semantic_matching(recommendations, responses, top_k)
        else:
            return self._keyword_matching(recommendations, responses, top_k)

    def _normalize_rec_id(self, rec_id: str) -> Optional[str]:
        """
        Normalize recommendation ID for matching.
        
        Examples:
        - "R/2024/025" -> "2024/025"
        - "2018/006" -> "2018/006"
        - "1" -> "1"
        """
        if not rec_id:
            return None
        
        rec_id_str = str(rec_id).strip()
        
        # Remove R/ prefix if present
        if rec_id_str.startswith("R/"):
            rec_id_str = rec_id_str[2:]
        
        return rec_id_str

    def _check_org_mismatch(self, rec: Dict) -> Tuple[bool, str]:
        """
        Check if a recommendation is directed to a different organisation
        than the one that authored the response document.
        
        Returns (is_mismatch, target_org)
        """
        rec_text = rec.get("text", "")
        target_org = extract_recommendation_target_org(rec_text)
        
        if not target_org:
            # Can't determine target org - allow matching
            return False, None
        
        if not self.response_doc_org:
            # Don't know who authored responses - allow matching
            return False, target_org
        
        # Special case: multi-org HSIB responses can match any org
        if self.response_doc_org == 'multi_org_hsib':
            return False, target_org
        
        # Check if target matches response doc org
        if target_org == self.response_doc_org:
            return False, target_org
        
        # Mismatch detected
        logger.info(f"Org mismatch: rec targets '{target_org}' but responses are from '{self.response_doc_org}'")
        return True, target_org

    def _semantic_matching(self, recommendations: List[Dict], responses: List[Dict], top_k: int) -> List[Dict]:
        """Semantic matching using embeddings."""
        responses_by_number: Dict[str, List[Dict]] = {}
        responses_by_org: Dict[str, List[Dict]] = {}
        
        for resp in responses:
            rec_num = resp.get("rec_number") or resp.get("rec_id")
            if rec_num:
                normalized = self._normalize_rec_id(rec_num)
                responses_by_number.setdefault(rec_num, []).append(resp)
                if normalized and normalized != rec_num:
                    responses_by_number.setdefault(normalized, []).append(resp)
            
            source_org = resp.get("source_org") or extract_target_org_from_text(resp.get("text", ""))
            if source_org:
                responses_by_org.setdefault(source_org, []).append(resp)

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

                # v2.6: Check for organisation mismatch FIRST
                is_mismatch, target_org = self._check_org_mismatch(rec)
                if is_mismatch:
                    # This rec is for a different org - don't try to match
                    alignments.append({
                        "recommendation": rec,
                        "response_text": None,
                        "similarity": 0.0,
                        "status": f"No response (rec directed to {target_org})",
                        "status_confidence": 0.95,
                        "match_method": "org_mismatch",
                        "target_org": target_org,
                    })
                    continue

                rec_num = self._extract_rec_number(rec["text"]) or rec.get("rec_number")
                normalized_rec_num = self._normalize_rec_id(rec_num) if rec_num else None
                rec_target_org = extract_target_org_from_text(rec["text"])
                
                # =============================================================
                # TIER 1: ID-based matching (highest priority)
                # =============================================================
                if rec_num:
                    candidates = responses_by_number.get(rec_num, [])
                    if normalized_rec_num and normalized_rec_num != rec_num:
                        candidates = candidates + responses_by_number.get(normalized_rec_num, [])
                    
                    seen_texts = set()
                    unique_candidates = []
                    for c in candidates:
                        text_key = c.get("cleaned_text", c.get("text", ""))[:100]
                        if text_key not in seen_texts:
                            seen_texts.add(text_key)
                            unique_candidates.append(c)
                    
                    for resp in unique_candidates:
                        resp_text = resp["cleaned_text"]
                        if self._is_self_match(rec["text"], resp_text):
                            continue
                        resp_idx = next((i for i, r in enumerate(responses) if r is resp), None)
                        semantic_score = similarity_matrix[rec_idx][resp_idx] if resp_idx is not None else 0.7
                        
                        if semantic_score >= 0.3:
                            score = min(semantic_score + 0.5, 1.0)
                            if score > best_score:
                                best_score = score
                                status, status_conf = self._classify_response_status(resp_text)
                                best_match = {
                                    "response_text": resp_text,
                                    "similarity": round(score, 3),
                                    "status": status,
                                    "status_confidence": status_conf,
                                    "source_document": resp.get("source_document", "unknown"),
                                    "match_method": "number_match",
                                }

                # =============================================================
                # TIER 2: Organisation-based matching
                # =============================================================
                if best_match is None and rec_target_org and rec_target_org in responses_by_org:
                    for resp in responses_by_org[rec_target_org]:
                        resp_text = resp["cleaned_text"]
                        if self._is_self_match(rec["text"], resp_text):
                            continue
                        resp_idx = next((i for i, r in enumerate(responses) if r is resp), None)
                        semantic_score = similarity_matrix[rec_idx][resp_idx] if resp_idx is not None else 0.0
                        
                        if semantic_score >= 0.55:
                            score = min(semantic_score + 0.15, 0.95)
                            if score > best_score:
                                best_score = score
                                status, status_conf = self._classify_response_status(resp_text)
                                best_match = {
                                    "response_text": resp_text,
                                    "similarity": round(score, 3),
                                    "status": status,
                                    "status_confidence": status_conf,
                                    "source_document": resp.get("source_document", "unknown"),
                                    "match_method": "org_match",
                                }

                # =============================================================
                # TIER 3: Semantic matching fallback
                # v2.6: Only if no org mismatch detected
                # =============================================================
                if best_match is None:
                    similarities = similarity_matrix[rec_idx]
                    top_indices = np.argsort(similarities)[::-1][: top_k * 2]
                    for resp_idx in top_indices:
                        resp = responses[resp_idx]
                        resp_text = resp["cleaned_text"]
                        if self._is_self_match(rec["text"], resp_text):
                            continue
                        score = similarities[resp_idx]
                        if score > best_score and score >= 0.5:
                            best_score = score
                            status, status_conf = self._classify_response_status(resp_text)
                            best_match = {
                                "response_text": resp_text,
                                "similarity": round(float(score), 3),
                                "status": status,
                                "status_confidence": status_conf,
                                "source_document": resp.get("source_document", "unknown"),
                                "match_method": "semantic",
                            }
                            break

                # Build alignment result
                if best_match:
                    alignments.append({
                        "recommendation": rec,
                        **best_match,
                    })
                else:
                    alignments.append({
                        "recommendation": rec,
                        "response_text": None,
                        "similarity": 0.0,
                        "status": "No response found",
                        "status_confidence": 1.0,
                        "match_method": "none",
                    })

            return alignments

        except Exception as e:
            logger.error(f"Semantic matching error: {e}")
            return self._keyword_matching(recommendations, responses, top_k)

    def _keyword_matching(self, recommendations: List[Dict], responses: List[Dict], top_k: int) -> List[Dict]:
        """Keyword-based matching fallback."""
        alignments = []
        
        for rec in recommendations:
            # v2.6: Check org mismatch
            is_mismatch, target_org = self._check_org_mismatch(rec)
            if is_mismatch:
                alignments.append({
                    "recommendation": rec,
                    "response_text": None,
                    "similarity": 0.0,
                    "status": f"No response (rec directed to {target_org})",
                    "status_confidence": 0.95,
                    "match_method": "org_mismatch",
                    "target_org": target_org,
                })
                continue
            
            rec_words = set(get_meaningful_words(rec["text"]))
            best_match = None
            best_score = 0.0
            
            for resp in responses:
                resp_text = resp["cleaned_text"]
                if self._is_self_match(rec["text"], resp_text):
                    continue
                
                resp_words = set(get_meaningful_words(resp_text))
                if not rec_words or not resp_words:
                    continue
                
                intersection = len(rec_words & resp_words)
                union = len(rec_words | resp_words)
                score = intersection / union if union > 0 else 0.0
                
                if score > best_score:
                    best_score = score
                    status, status_conf = self._classify_response_status(resp_text)
                    best_match = {
                        "response_text": resp_text,
                        "similarity": round(score, 3),
                        "status": status,
                        "status_confidence": status_conf,
                        "match_method": "keyword",
                    }
            
            if best_match:
                alignments.append({"recommendation": rec, **best_match})
            else:
                alignments.append({
                    "recommendation": rec,
                    "response_text": None,
                    "similarity": 0.0,
                    "status": "No response found",
                    "status_confidence": 1.0,
                    "match_method": "none",
                })
        
        return alignments

    def _extract_rec_number(self, text: str) -> Optional[str]:
        """Extract recommendation number from text."""
        patterns = [
            r'Safety\s+recommendation\s+(R/\d{4}/\d{3})',
            r'Recommendation\s+(\d{4}/\d{3})',
            r'Recommendation\s+(\d{1,2})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _is_self_match(self, rec_text: str, resp_text: str) -> bool:
        """Check if response is actually the recommendation itself."""
        rec_clean = re.sub(r'\s+', ' ', rec_text.lower().strip())[:200]
        resp_clean = re.sub(r'\s+', ' ', resp_text.lower().strip())[:200]
        
        if rec_clean == resp_clean:
            return True
        
        rec_words = set(rec_clean.split())
        resp_words = set(resp_clean.split())
        if rec_words and resp_words:
            overlap = len(rec_words & resp_words) / len(rec_words)
            if overlap > 0.9:
                return True
        
        return False

    def _classify_response_status(self, response_text: str) -> Tuple[str, float]:
        """Classify response as Accepted/Rejected/Partial."""
        text_lower = response_text.lower()
        
        # Rejection indicators
        rejection_patterns = [
            r'\b(?:do\s+not|does\s+not|cannot|will\s+not)\s+(?:accept|agree|support)',
            r'\breject(?:s|ed)?\b',
            r'\b(?:not\s+)?(?:able|possible)\s+to\s+(?:accept|implement)',
            r'\bdecline[sd]?\b',
        ]
        
        # Partial indicators
        partial_patterns = [
            r'\bpartial(?:ly)?\b',
            r'\bin\s+part\b',
            r'\bdo\s+not\s+support\s+mandating\b',
            r'\bsome\s+(?:aspects?|elements?|parts?)\b',
            r'\bwith\s+(?:some\s+)?(?:reservations?|caveats?|modifications?)\b',
        ]
        
        # Acceptance indicators
        acceptance_patterns = [
            r'\b(?:accept|accepts|accepted)\b',
            r'\b(?:agree|agrees|agreed)\b',
            r'\b(?:support|supports|supported)\b',
            r'\b(?:welcome|welcomes|welcomed)\b',
            r'\bcommitted\s+to\b',
            r'\bwill\s+(?:implement|undertake|deliver|ensure)\b',
            r'\bhas\s+(?:begun|started|commenced)\b',
            r'\bhappy\s+to\s+confirm\b',
        ]
        
        # Check patterns
        for pattern in rejection_patterns:
            if re.search(pattern, text_lower):
                return "Rejected", 0.9
        
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partial", 0.85
        
        for pattern in acceptance_patterns:
            if re.search(pattern, text_lower):
                return "Accepted", 0.9
        
        return "Unclear", 0.5


def calculate_simple_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    words1 = set(get_meaningful_words(text1))
    words2 = set(get_meaningful_words(text2))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def classify_content_type(sentence: str) -> str:
    """Classify a sentence's content type."""
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
                matches.append({
                    "sentence": sentence_clean,
                    "document": doc,
                    "pattern": match_type,
                    "position": pos,
                    "page_number": max(1, pos // 2000 + 1) if pos >= 0 else 1,
                    "content_type": classify_content_type(sentence_clean),
                })

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
        rec_words = set(get_meaningful_words(rec.get("text", "")))
        best_match = None
        best_score = 0.0

        for resp in responses:
            resp_words = set(get_meaningful_words(resp.get("text", "")))
            if not rec_words or not resp_words:
                continue

            intersection = len(rec_words & resp_words)
            union = len(rec_words | resp_words)
            score = intersection / union if union > 0 else 0.0

            if score > best_score:
                best_score = score
                best_match = resp

        if best_match and best_score >= similarity_threshold:
            alignments.append({
                "recommendation": rec,
                "response": best_match,
                "similarity": best_score,
            })
        else:
            alignments.append({
                "recommendation": rec,
                "response": None,
                "similarity": 0.0,
            })

    return alignments


# Re-export everything for backwards compatibility
__all__ = [
    # Main class
    "RecommendationResponseMatcher",
    # Re-exported from response_extractor
    "is_recommendation_text",
    "is_genuine_response",
    "has_pdf_artifacts",
    "clean_pdf_artifacts",
    "is_hsib_response_document",
    "is_trust_response_document",
    "is_org_based_hsib_response",
    "extract_target_org_from_text",
    "extract_recommendation_target_org",
    "get_response_document_org",
    "extract_hsib_responses",
    "extract_trust_responses",
    "extract_org_based_hsib_responses",
    "extract_response_sentences",
    # Helper functions
    "find_pattern_matches",
    "align_recommendations_with_responses",
    "determine_alignment_status",
    "calculate_simple_similarity",
    "classify_content_type",
]
