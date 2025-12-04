"""
Core alignment engine: recommendation/response matching + response extraction.
Moved out of the Streamlit UI to keep logic reusable and testable.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .search_utils import STOP_WORDS

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
        responses_by_number: Dict[str, List[Dict]] = {}
        for resp in responses:
            if resp.get("rec_number"):
                num = resp["rec_number"]
                responses_by_number.setdefault(num, []).append(resp)

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

                rec_num = self._extract_rec_number(rec["text"])
                if rec_num and rec_num in responses_by_number:
                    for resp in responses_by_number[rec_num]:
                        resp_text = resp["cleaned_text"]
                        if self._is_self_match(rec["text"], resp_text):
                            continue
                        resp_idx = next((i for i, r in enumerate(responses) if r is resp), None)
                        score = similarity_matrix[rec_idx][resp_idx] if resp_idx is not None else 0.7
                        score = min(score + 0.4, 1.0)
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
        alignments = []
        for rec in recommendations:
            best_match = None
            best_score = 0.0
            for resp in responses:
                resp_text = resp.get("cleaned_text", resp["text"])
                if self._is_self_match(rec["text"], resp_text):
                    continue
                score = self._keyword_similarity(rec["text"], resp_text)
                rec_num = self._extract_rec_number(rec["text"])
                resp_num = self._extract_rec_number(resp_text)
                if rec_num and resp_num and rec_num == resp_num:
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
            "accept the recommendation",
            "support the recommendation",
            "accept this recommendation",
        ]
        return any(ind in text_lower for ind in indicators)

    def _extract_rec_number(self, text: str) -> Optional[str]:
        match = re.search(r"recommendation\s+(\d+)", text.lower())
        return match.group(1) if match else None

    def _classify_response_status(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        accepted_patterns = [
            r"\baccept(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)",
            r"\bsupport(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)",
            r"\bagree(?:s|d)?\s+(?:with\s+)?(?:this\s+|the\s+)?(?:recommendation|rec)",
            r"\bthe\s+government\s+support(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bwill\s+implement",
            r"\bfully\s+accept",
            r"\baccept(?:s|ed)?\s+in\s+full",
        ]
        rejected_patterns = [
            r"\breject(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)",
            r"\bdoes?\s+not\s+accept",
            r"\bcannot\s+accept",
            r"\bdisagree(?:s|d)?",
        ]
        partial_patterns = [
            r"\baccept(?:s|ed)?\s+in\s+(?:part|principle)",
            r"\bpartially\s+accept",
            r"\bsupport(?:s|ed)?\s+in\s+principle",
            r"\bwill\s+consider",
            r"\bunder\s+consideration",
        ]
        noted_patterns = [
            r"\bnote(?:s|d)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)",
            r"\backnowledge(?:s|d)?",
            r"\btake(?:s|n)?\s+note",
        ]

        for pattern in accepted_patterns:
            if re.search(pattern, text_lower):
                return "Accepted", 0.9
        for pattern in rejected_patterns:
            if re.search(pattern, text_lower):
                return "Rejected", 0.9
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partial", 0.8
        for pattern in noted_patterns:
            if re.search(pattern, text_lower):
                return "Noted", 0.75
        if any(kw in text_lower for kw in ["supports", "support", "accepts", "accept"]):
            if "recommendation" in text_lower:
                return "Accepted", 0.7
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
        r"^dhsc\s+and\s+nhs\s+england\s+support",
        r"^nhs\s+england\s+will",
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


def extract_response_sentences(text: str) -> List[Dict]:
    """
    Extract sentences that are government responses.
    Uses strict filtering to exclude recommendation text.
    """
    if not text:
        return []

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
        rec_ref = re.search(r"recommendation\s+(\d+)", resp_content.lower())
        responses.append(
            {
                "text": resp_content,
                "position": start_pos,
                "response_type": "structured",
                "confidence": 0.95,
                "rec_number": rec_ref.group(1) if rec_ref else rec_num,
            }
        )

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for idx, sentence in enumerate(sentences):
        sentence = clean_pdf_artifacts(sentence.strip())
        if len(sentence) < 40 or len(sentence) > 500:
            continue
        if is_recommendation_text(sentence):
            continue
        if not is_genuine_response(sentence):
            continue
        resp_key = sentence[:100].lower()
        if resp_key in seen_responses:
            continue
        seen_responses.add(resp_key)
        rec_ref = re.search(r"recommendation\s+(\d+)", sentence.lower())
        responses.append(
            {
                "text": sentence,
                "position": idx,
                "response_type": "sentence",
                "confidence": 0.85,
                "rec_number": rec_ref.group(1) if rec_ref else None,
            }
        )

    logger.info(f"Extracted {len(responses)} genuine responses (no bleeding)")
    return responses


__all__ = [
    "RecommendationResponseMatcher",
    "is_recommendation_text",
    "is_genuine_response",
    "has_pdf_artifacts",
    "clean_pdf_artifacts",
    "extract_response_sentences",
]
