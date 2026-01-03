"""
Core alignment engine: recommendation/response matching.

This module handles:
- Semantic and keyword-based matching of recommendations to responses
- Response status classification (Accepted/Rejected/Partial)
- Alignment scoring and ranking

Response extraction has been moved to response_extractor.py but is
re-exported here for backwards compatibility.

v2.5 Changes:
- FIXED: org_match no longer gives false positives - requires BOTH org match AND semantic relevance >= 0.55
- FIXED: number_match now properly validates rec_id format matching (R/2024/025 vs 2018/006)
- ADDED: _normalize_rec_id() for cross-format ID matching
- ADDED: Stricter semantic threshold for org_match
- SPLIT: Response extraction moved to response_extractor.py

v2.3 Changes (preserved):
- Added partial rejection detection ("do not support mandating" = Partial)
- Fixed HSIB/HSSIB response format support with PyPDF2 compatibility
- 3-tier matching: number_match → org_match → semantic
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
    extract_target_org_from_text,
    extract_hsib_responses,
    extract_response_sentences,
)

logger = logging.getLogger(__name__)


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

        cleaned_responses = []
        for resp in responses:
            cleaned_text = self._clean_response_text(resp["text"])
            cleaned_responses.append({**resp, "cleaned_text": cleaned_text})

        if self.use_transformer and self.model:
            return self._semantic_matching(recommendations, cleaned_responses, top_k)
        return self._keyword_matching(recommendations, cleaned_responses, top_k)

    # =========================================================================
    # v2.5: Normalize rec IDs for cross-format matching
    # =========================================================================
    def _normalize_rec_id(self, rec_id: Optional[str]) -> Optional[str]:
        """
        Normalize recommendation IDs for matching across formats.
        
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

    def _semantic_matching(self, recommendations: List[Dict], responses: List[Dict], top_k: int) -> List[Dict]:
        responses_by_number: Dict[str, List[Dict]] = {}
        responses_by_org: Dict[str, List[Dict]] = {}
        
        for resp in responses:
            rec_num = resp.get("rec_number") or resp.get("rec_id")
            if rec_num:
                # v2.5: Normalize ID for matching
                normalized = self._normalize_rec_id(rec_num)
                responses_by_number.setdefault(rec_num, []).append(resp)
                if normalized and normalized != rec_num:
                    responses_by_number.setdefault(normalized, []).append(resp)
            
            # Index by organisation
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

                rec_num = self._extract_rec_number(rec["text"]) or rec.get("rec_number")
                normalized_rec_num = self._normalize_rec_id(rec_num) if rec_num else None
                target_org = extract_target_org_from_text(rec["text"])
                
                # =============================================================
                # TIER 1: ID-based matching (highest priority)
                # =============================================================
                if rec_num:
                    candidates = responses_by_number.get(rec_num, [])
                    if normalized_rec_num and normalized_rec_num != rec_num:
                        candidates = candidates + responses_by_number.get(normalized_rec_num, [])
                    
                    # Deduplicate candidates
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
                        
                        # v2.5: ID match requires minimal semantic relevance
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
                # v2.5 FIX: Requires BOTH org match AND semantic relevance >= 0.55
                # =============================================================
                if best_match is None and target_org and target_org in responses_by_org:
                    for resp in responses_by_org[target_org]:
                        resp_text = resp["cleaned_text"]
                        if self._is_self_match(rec["text"], resp_text):
                            continue
                        resp_idx = next((i for i, r in enumerate(responses) if r is resp), None)
                        semantic_score = similarity_matrix[rec_idx][resp_idx] if resp_idx is not None else 0.0
                        
                        # v2.5 FIX: Require minimum semantic relevance
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
                # =============================================================
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
                                "similarity": round(best_score, 3),
                                "status": status,
                                "status_confidence": status_conf,
                                "source_document": resp.get("source_document", "unknown"),
                                "match_method": "semantic",
                            }

                alignments.append({
                    "recommendation": rec,
                    "response": best_match,
                    "has_response": best_match is not None,
                })

            return alignments
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return self._keyword_matching(recommendations, responses, top_k)

    def _keyword_matching(self, recommendations: List[Dict], responses: List[Dict], top_k: int) -> List[Dict]:
        alignments = []
        for rec in recommendations:
            best_match = None
            best_score = 0.0
            for resp in responses:
                resp_text = resp.get("cleaned_text", resp["text"])
                if self._is_self_match(rec["text"], resp_text):
                    continue
                score = self._keyword_similarity(rec["text"], resp_text)
                rec_num = self._extract_rec_number(rec["text"]) or rec.get("rec_number")
                resp_num = resp.get("rec_number") or resp.get("rec_id")
                if rec_num and resp_num:
                    rec_norm = self._normalize_rec_id(rec_num)
                    resp_norm = self._normalize_rec_id(resp_num)
                    if rec_norm == resp_norm:
                        score += 0.4
                if self._has_government_response_language(resp_text):
                    score += 0.2
                if score > best_score and score >= 0.25:
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
        base_score = intersection / union if union > 0 else 0.0

        gov_terms = {
            "government", "recommendation", "accept", "support", "implement",
            "policy", "department", "nhs", "england", "health", "care",
            "provider", "commissioner", "trust", "board", "cqc",
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
            "government response", "the government supports", "the government accepts",
            "the government agrees", "the government notes", "the government rejects",
            "we accept", "we support", "we agree", "we welcome",
            "accept the recommendation", "support the recommendation",
            "accept this recommendation", "nhs england has", "nhs england will",
            "nhs england is", "care quality commission", "nice is",
            "we have convened", "we are happy to confirm",
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
        v2.3 - Added partial rejection patterns for "do not support mandating"
        """
        text_lower = text.lower()
        
        # PRIORITY 1: Full rejection
        full_rejection_patterns = [
            r"\breject(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bdoes?\s+not\s+accept\s+(?:this\s+|the\s+)?recommendation",
            r"\bcannot\s+accept\s+(?:this\s+|the\s+)?recommendation",
            r"\bwill\s+not\s+(?:be\s+)?implement(?:ing)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bdisagree(?:s|d)?\s+with\s+(?:this\s+|the\s+)?recommendation",
        ]
        for pattern in full_rejection_patterns:
            if re.search(pattern, text_lower):
                return "Rejected", 0.9
        
        # PRIORITY 2: Partial acceptance / rejection
        partial_patterns = [
            r"\baccept(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation\s+)?in\s+(?:part|principle)",
            r"\bpartially\s+accept",
            r"\bsupport(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation\s+)?in\s+principle",
            r"\bin\s+principle[,.]?\s+(?:we\s+)?(?:support|accept|agree)",
            r"\bbroadly\s+support(?:s|ed)?(?!\s+(?:the|this)\s+recommendation)",
            r"\baccept(?:s|ed)?\s+(?:the\s+)?(?:intent|spirit)",
            r"\bunder\s+consideration",
            r"\bwill\s+consider\s+(?:the\s+|this\s+)?(?:recommendation|proposal|request)",
            r"\brequires?\s+further\s+consideration",
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partial", 0.85
        
        # Partial rejection of approach (not full rejection)
        partial_rejection_patterns = [
            r"\bdo(?:es)?\s+not\s+support\s+(?:mandating|requiring|prescribing)\b",
            r"\bwill\s+not\s+(?:be\s+)?mandat(?:ing|e)\b",
            r"\bcannot\s+(?:mandate|require|prescribe)\b",
        ]
        for pattern in partial_rejection_patterns:
            if re.search(pattern, text_lower):
                if re.search(r"\b(?:recognis|support|expect|however|important)", text_lower):
                    return "Partial", 0.8
        
        # PRIORITY 3: Explicit acceptance
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
        
        # PRIORITY 4: Implicit acceptance (taking action)
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
        
        # PRIORITY 5: Noted
        noted_patterns = [
            r"\bnote(?:s|d)?\s+(?:this\s+|the\s+)?recommendation",
            r"\backnowledge(?:s|d)?\s+(?:this\s+|the\s+)?recommendation",
        ]
        for pattern in noted_patterns:
            if re.search(pattern, text_lower):
                return "Noted", 0.75
        
        # Fallback
        if any(word in text_lower for word in ["support", "accept", "welcome", "agree", "committed", "happy to confirm"]):
            return "Accepted", 0.6
        
        return "Unclear", 0.5


# --------------------------------------------------------------------------- #
# Lightweight keyword-based helpers (exposed for UI)
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
        rec_doc = rec["document"]["filename"]
        best_responses = []
        for resp in responses:
            similarity = calculate_simple_similarity(rec["sentence"], resp["sentence"])
            if resp["document"]["filename"] == rec_doc:
                similarity *= 1.2
            if resp.get("document", {}).get("filename") == rec_doc and resp.get("position", 0) > rec.get("position", 0):
                similarity *= 1.1
            if similarity >= similarity_threshold:
                best_responses.append({
                    "response": resp,
                    "combined_score": min(similarity, 1.0),
                    "similarity_score": similarity,
                    "same_document": resp["document"]["filename"] == rec_doc,
                })
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
    "extract_target_org_from_text",
    "extract_hsib_responses",
    "extract_response_sentences",
    # Helper functions
    "find_pattern_matches",
    "align_recommendations_with_responses",
    "determine_alignment_status",
    "calculate_simple_similarity",
    "classify_content_type",
]
