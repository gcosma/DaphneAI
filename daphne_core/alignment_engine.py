"""
Core alignment engine v3.0 - DEFINITIVE VERSION
Handles all HSIB/HSSIB response formats properly

Changes from v2.2:
- Simplified response pattern: just '\nResponse\n' which works on all documents  
- Better HSIB detection
- Cleaner boundary detection
- Added extensive logging for debugging
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .search_utils import STOP_WORDS, get_meaningful_words

logger = logging.getLogger(__name__)


# =============================================================================
# ORGANISATION EXTRACTION
# =============================================================================

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
    """Extract the target organisation from a recommendation."""
    if not rec_text:
        return None
    
    rec_patterns = [
        r'(?:HSIB|HSSIB)\s+recommends\s+that\s+(?:the\s+)?([^,\.]+?)(?:\s+(?:should|works?|produces?|evaluates?|reviews?|assesses?|working))',
        r'It\s+is\s+recommended\s+that\s+(?:the\s+)?([^,\.]+?)(?:\s+(?:should|works?|requires?|ensures?))',
        r'recommends\s+that\s+(?:the\s+)?([^,\.]+?)(?:,|\s+(?:should|works?|produces?|evaluates?|reviews?|assesses?|working|forms))',
    ]
    
    for pattern in rec_patterns:
        match = re.search(pattern, rec_text, re.IGNORECASE)
        if match:
            org_text = match.group(1).strip()
            return _map_to_canonical_org(org_text)
    
    for canonical, variants in ORGANISATION_MAPPINGS.items():
        for variant in variants:
            if re.search(variant, rec_text, re.IGNORECASE):
                return canonical
    
    return None


def extract_source_organisation(resp_text: str) -> Optional[str]:
    """Extract the source organisation from a response."""
    if not resp_text:
        return None
    
    # Check first 300 chars for org indicators
    first_part = resp_text[:300].lower()
    
    org_indicators = {
        'nhs_england': ['nhs england', 'nhse'],
        'cqc': ['care quality commission', 'cqc'],
        'nice': ['national institute for health and care excellence', 'nice'],
        'royal_college': ['royal college of psychiatrists', 'royal college'],
        'dhsc': ['department of health', 'dhsc'],
        'nihr': ['national institute for health', 'nihr'],
    }
    
    for canonical, indicators in org_indicators.items():
        for ind in indicators:
            if ind in first_part:
                return canonical
    
    return None


def _map_to_canonical_org(org_text: str) -> Optional[str]:
    """Map organisation text to canonical name."""
    if not org_text:
        return None
    
    for canonical, variants in ORGANISATION_MAPPINGS.items():
        for variant in variants:
            if re.search(variant, org_text, re.IGNORECASE):
                return canonical
    
    org_lower = org_text.lower()
    if 'nhs' in org_lower and 'england' in org_lower:
        return 'nhs_england'
    if 'cqc' in org_lower or 'care quality' in org_lower:
        return 'cqc'
    if 'nice' in org_lower:
        return 'nice'
    if 'royal college' in org_lower:
        return 'royal_college'
    if 'dhsc' in org_lower or 'department of health' in org_lower:
        return 'dhsc'
    if 'nihr' in org_lower:
        return 'nihr'
    
    return None


# =============================================================================
# RESPONSE EXTRACTION - SIMPLIFIED AND ROBUST
# =============================================================================

def is_hsib_response_document(text: str) -> bool:
    """Detect if a document is an HSIB/HSSIB-style response document."""
    if not text:
        return False
    
    # Check for standard government format first
    gov_pattern = re.compile(r'Government\s+response\s+to\s+recommendation\s+\d+', re.IGNORECASE)
    if len(list(gov_pattern.finditer(text))) >= 2:
        logger.info("Detected standard government format (not HSIB)")
        return False
    
    # Count standalone Response headers using simple pattern
    response_pattern = re.compile(r'\nResponse\s*\n', re.IGNORECASE)
    response_count = len(list(response_pattern.finditer(text)))
    logger.info(f"HSIB detection: found {response_count} '\\nResponse\\n' headers")
    
    if response_count >= 2:
        logger.info("Detected HSIB format: 2+ Response headers")
        return True
    
    # Check for HSIB/HSSIB indicators
    hsib_indicators = [
        (r'\bHSIB\s+recommends\b', 'HSIB recommends'),
        (r'\bHSSIB\s+recommends\b', 'HSSIB recommends'),
        (r'\bSafety\s+recommendation\s+R/\d{4}/\d{3}\b', 'Safety rec R/YYYY/NNN'),
        (r'\bRecommendation\s+\d{4}/\d{3}\b', 'Rec YYYY/NNN'),
        (r'\bResponse\s+received\s+on\b', 'Response received'),
        (r'\bSafety\s+recommendations\b', 'Safety recommendations section'),
    ]
    
    indicator_count = 0
    for pattern, name in hsib_indicators:
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count > 0:
            logger.info(f"HSIB indicator: {name} ({count})")
            indicator_count += count
    
    if indicator_count >= 2:
        logger.info(f"Detected HSIB format: {indicator_count} indicators")
        return True
    
    # Also check for merged Response patterns (PyPDF2 quirk)
    merged_pattern = re.compile(r'Response(?=[A-Z][a-z])', re.IGNORECASE)
    merged_count = len(list(merged_pattern.finditer(text)))
    if merged_count >= 2:
        logger.info(f"Detected HSIB format: {merged_count} merged Response headers")
        return True
    
    return False


def extract_hsib_responses(text: str) -> List[Dict]:
    """
    Extract responses from HSIB-format documents.
    Uses simple, robust pattern matching.
    """
    if not text:
        return []
    
    responses = []
    
    # SIMPLE PATTERN: Response on its own line
    # This works reliably with PyPDF2 output
    response_pattern = re.compile(r'\nResponse\s*\n', re.IGNORECASE)
    response_matches = list(response_pattern.finditer(text))
    
    # Also catch merged patterns like "ResponseNHS England..."
    merged_pattern = re.compile(r'\nResponse(?=[A-Z][a-z])', re.IGNORECASE)
    merged_matches = list(merged_pattern.finditer(text))
    
    # Combine and deduplicate
    all_matches = response_matches + merged_matches
    all_matches.sort(key=lambda m: m.start())
    
    # Deduplicate (within 50 chars)
    unique_matches = []
    last_pos = -100
    for match in all_matches:
        if match.start() - last_pos > 50:
            unique_matches.append(match)
            last_pos = match.start()
    
    logger.info(f"HSIB extraction: found {len(unique_matches)} Response headers")
    
    if not unique_matches:
        logger.warning("No Response headers found in HSIB document")
        return []
    
    # Find recommendation positions for context
    rec_pattern = re.compile(
        r'(?:Safety\s+recommendation\s+R/\d{4}/\d{3}|Recommendation\s+\d{4}/\d{3}|HSIB\s+recommends|HSSIB\s+recommends)',
        re.IGNORECASE
    )
    rec_matches = list(rec_pattern.finditer(text))
    
    # Boundary markers
    boundary_patterns = [
        r'\nResponse\s+received\s+on',
        r'\nActions\s+planned\s+to\s+deliver',
        r'\nSafety\s+(?:observation|action)',
        r'\n(?:HSIB|HSSIB)\s+recommends',
        r'\nSafety\s+recommendation\s+R/',
    ]
    
    # Extract each response
    for idx, match in enumerate(unique_matches):
        start = match.end()
        end = len(text)
        
        # Find end: next Response header
        if idx + 1 < len(unique_matches):
            end = min(end, unique_matches[idx + 1].start())
        
        # Find end: boundary markers
        for boundary in boundary_patterns:
            boundary_match = re.search(boundary, text[start:end], re.IGNORECASE)
            if boundary_match:
                end = min(end, start + boundary_match.start())
        
        # Find associated recommendation number
        rec_number = None
        for rec_match in reversed(rec_matches):
            if rec_match.start() < match.start():
                # Extract number from match
                rec_text = rec_match.group()
                num_match = re.search(r'(\d{4}/\d{3})', rec_text)
                if num_match:
                    rec_number = num_match.group(1)
                break
        
        # Extract and clean response text
        resp_text = text[start:end].strip()
        resp_text = clean_pdf_artifacts(resp_text)
        
        if len(resp_text) < 30:
            continue
        
        # Get source organisation
        source_org = extract_source_organisation(resp_text)
        
        responses.append({
            "text": resp_text,
            "position": start,
            "response_type": "hsib_structured",
            "confidence": 0.95,
            "rec_number": rec_number,
            "source_org": source_org,
        })
        
        logger.info(f"HSIB response {idx+1}: rec={rec_number}, org={source_org}, len={len(resp_text)}")
    
    logger.info(f"Extracted {len(responses)} HSIB responses total")
    return responses


def extract_response_sentences(text: str) -> List[Dict]:
    """
    Main response extraction function.
    Detects document type and uses appropriate method.
    """
    if not text:
        return []

    text = clean_pdf_artifacts(text)
    
    # Check for HSIB format
    if is_hsib_response_document(text):
        logger.info("Using HSIB response extraction")
        return extract_hsib_responses(text)
    
    # Standard government format
    logger.info("Using standard government response extraction")
    return extract_standard_responses(text)


def extract_standard_responses(text: str) -> List[Dict]:
    """Extract responses from standard government format documents."""
    responses = []
    
    # Find "Government response to recommendation N" sections
    gov_pattern = re.compile(
        r"Government\s+response\s+to\s+recommendation\s+(\d+)",
        re.IGNORECASE
    )
    matches = list(gov_pattern.finditer(text))
    
    logger.info(f"Found {len(matches)} government response headers")
    
    for i, match in enumerate(matches):
        start = match.end()
        rec_num = match.group(1)
        end = len(text)
        
        # End at next government response
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        
        # End at next recommendation
        rec_marker = re.search(r"Recommendation\s+\d+\s+[A-Z]", text[start:end])
        if rec_marker:
            end = start + rec_marker.start()
        
        resp_content = text[start:end].strip()
        resp_content = clean_pdf_artifacts(resp_content)
        
        if not resp_content or len(resp_content) < 30:
            continue
        
        responses.append({
            "text": resp_content,
            "position": start,
            "response_type": "structured",
            "confidence": 0.95,
            "rec_number": rec_num,
        })
    
    logger.info(f"Extracted {len(responses)} standard responses")
    return responses


def clean_pdf_artifacts(text: str) -> str:
    """Clean common PDF extraction artifacts."""
    if not text:
        return ""
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?\s*", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"GOV\.UK[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# RECOMMENDATION-RESPONSE MATCHER
# =============================================================================

class RecommendationResponseMatcher:
    """Match recommendations to responses using 3-tier priority."""

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
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.warning(f"Using keyword matching: {e}")
            self.use_transformer = False

    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.use_transformer or not self.model:
            return None
        try:
            import torch
            with torch.no_grad():
                return np.array(self.model.encode(texts, convert_to_tensor=False, batch_size=16))
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return None

    def find_best_matches(self, recommendations: List[Dict], responses: List[Dict], top_k: int = 3) -> List[Dict]:
        """Match using 3-tier priority: number → org → semantic."""
        if not recommendations or not responses:
            return []

        # Preprocess responses
        for resp in responses:
            if "source_org" not in resp:
                resp["source_org"] = extract_source_organisation(resp.get("text", ""))

        # Build indexes
        by_number = {}
        by_org = {}
        for resp in responses:
            if resp.get("rec_number"):
                by_number.setdefault(resp["rec_number"], []).append(resp)
            if resp.get("source_org"):
                by_org.setdefault(resp["source_org"], []).append(resp)

        logger.info(f"Response indexes: {len(by_number)} by number, {len(by_org)} by org")

        # Get embeddings
        similarity_matrix = None
        if self.use_transformer:
            rec_emb = self.encode_texts([r.get("text", "") for r in recommendations])
            resp_emb = self.encode_texts([r.get("text", "") for r in responses])
            if rec_emb is not None and resp_emb is not None:
                rec_norm = rec_emb / np.linalg.norm(rec_emb, axis=1, keepdims=True)
                resp_norm = resp_emb / np.linalg.norm(resp_emb, axis=1, keepdims=True)
                similarity_matrix = np.dot(rec_norm, resp_norm.T)

        alignments = []
        for rec_idx, rec in enumerate(recommendations):
            best_match = None
            best_score = 0.0
            match_method = "none"
            
            rec_text = rec.get("text", "")
            rec_num = self._extract_rec_number(rec_text)
            target_org = extract_target_organisation(rec_text)

            # TIER 1: Number match
            if rec_num and rec_num in by_number:
                for resp in by_number[rec_num]:
                    score = self._get_similarity(rec_idx, responses.index(resp), similarity_matrix, rec_text, resp["text"])
                    score = min(score + 0.4, 1.0)
                    if score > best_score:
                        best_score = score
                        best_match = self._create_match(resp, score, "number_match")
                        match_method = "number_match"

            # TIER 2: Organisation match
            if best_match is None and target_org and target_org in by_org:
                for resp in by_org[target_org]:
                    score = self._get_similarity(rec_idx, responses.index(resp), similarity_matrix, rec_text, resp["text"])
                    score = min(score + 0.3, 1.0)
                    if score > best_score:
                        best_score = score
                        best_match = self._create_match(resp, score, "org_match")
                        match_method = "org_match"

            # TIER 3: Semantic match
            if best_match is None:
                for resp_idx, resp in enumerate(responses):
                    score = self._get_similarity(rec_idx, resp_idx, similarity_matrix, rec_text, resp["text"])
                    if score > best_score and score >= 0.4:
                        best_score = score
                        best_match = self._create_match(resp, score, "semantic")
                        match_method = "semantic"

            logger.info(f"Rec {rec_num or rec_idx}: {match_method} (score={best_score:.2f})")
            alignments.append({
                "recommendation": rec,
                "response": best_match,
                "has_response": best_match is not None,
            })

        return alignments

    def _get_similarity(self, rec_idx, resp_idx, matrix, rec_text, resp_text):
        if matrix is not None:
            return float(matrix[rec_idx][resp_idx])
        return self._keyword_similarity(rec_text, resp_text)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        words1 = set(re.findall(r"\b\w+\b", text1.lower())) - STOP_WORDS
        words2 = set(re.findall(r"\b\w+\b", text2.lower())) - STOP_WORDS
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _create_match(self, resp, score, method):
        status, conf = self._classify_status(resp.get("text", ""))
        return {
            "response_text": resp.get("text", ""),
            "similarity": score,
            "status": status,
            "status_confidence": conf,
            "source_document": resp.get("source_document", "unknown"),
            "match_method": method,
        }

    def _extract_rec_number(self, text: str) -> Optional[str]:
        # HSIB: 2018/006, R/2023/220
        match = re.search(r'(?:R/)?(\d{4}/\d{3})', text)
        if match:
            return match.group(1)
        # Standard: Recommendation 1
        match = re.search(r"recommendation\s+(\d+)", text.lower())
        return match.group(1) if match else None

    def _classify_status(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        
        if re.search(r"\breject", text_lower):
            return "Rejected", 0.9
        if re.search(r"\b(partial|in\s+principle|under\s+consideration)", text_lower):
            return "Partial", 0.85
        if re.search(r"\b(accept|support|agree|will\s+implement)", text_lower):
            return "Accepted", 0.9
        if re.search(r"\b(begun|committed|working|will\s+work)", text_lower):
            return "Accepted", 0.8
        if re.search(r"\bnote", text_lower):
            return "Noted", 0.75
        
        return "Unclear", 0.5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_recommendation_text(text: str) -> bool:
    text_lower = text.lower().strip()
    patterns = [
        r"^nhs\s+england\s+should",
        r"^hsib\s+recommends",
        r"^hssib\s+recommends",
        r"^we\s+recommend",
    ]
    return any(re.search(p, text_lower) for p in patterns)


def is_genuine_response(text: str) -> bool:
    text_lower = text.lower().strip()
    patterns = [
        r"^government\s+response",
        r"^the\s+government\s+supports?",
        r"^nhs\s+england\s+(?:has|will|is)",
        r"^cqc\s+(?:has|will|is)",
        r"^we\s+(?:support|accept|welcome)",
    ]
    return any(re.search(p, text_lower) for p in patterns)


def has_pdf_artifacts(text: str) -> bool:
    return bool(re.search(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}", text))


# Lightweight UI helpers
def classify_content_type(sentence: str) -> str:
    s = sentence.lower()
    if any(w in s for w in ["implementation", "deploy", "deliver"]):
        return "Implementation"
    if any(w in s for w in ["monitor", "evaluate", "review"]):
        return "Monitoring"
    return "General"


def find_pattern_matches(documents, patterns, match_type):
    matches = []
    for doc in documents:
        text = doc.get("text", "")
        for sentence in re.split(r"[.!?]+", text):
            s = sentence.strip()
            if any(p.lower() in s.lower() for p in patterns):
                matches.append({"sentence": s, "document": doc, "pattern": match_type})
    return matches


def determine_alignment_status(responses):
    if not responses:
        return "No response found"
    return "Possible response" if responses[0].get("combined_score", 0) >= 0.5 else "Uncertain"


def align_recommendations_with_responses(recs, resps, threshold):
    alignments = []
    for rec in recs:
        best = []
        for resp in resps:
            sim = calculate_simple_similarity(rec["sentence"], resp["sentence"])
            if sim >= threshold:
                best.append({"response": resp, "combined_score": sim})
        best.sort(key=lambda x: x["combined_score"], reverse=True)
        alignments.append({
            "recommendation": rec,
            "responses": best[:3],
            "alignment_status": determine_alignment_status(best),
        })
    return alignments


def calculate_simple_similarity(t1: str, t2: str) -> float:
    w1 = set(get_meaningful_words(t1))
    w2 = set(get_meaningful_words(t2))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


__all__ = [
    "RecommendationResponseMatcher",
    "extract_response_sentences",
    "extract_hsib_responses",
    "is_hsib_response_document",
    "extract_target_organisation",
    "extract_source_organisation",
    "is_recommendation_text",
    "is_genuine_response",
    "has_pdf_artifacts",
    "clean_pdf_artifacts",
    "find_pattern_matches",
    "align_recommendations_with_responses",
    "determine_alignment_status",
    "calculate_simple_similarity",
    "classify_content_type",
]
