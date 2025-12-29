"""
Core alignment engine: recommendation/response matching + response extraction.
v2.2 - BEST VERSION - Clean HSIB extraction without text bleeding

Key features:
- Organisation-based matching for HSIB documents (3-tier priority)
- Multi-pattern response detection for PyPDF2 compatibility
- Clean boundary detection to prevent text bleeding
- Deduplication of responses
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
        r'(?:HSIB|HSSIB)\s+recommends\s+that\s+(?:the\s+)?([^,\.]+?)(?:\s+(?:should|works?|produces?|evaluates?|reviews?|assesses?))',
        r'It\s+is\s+recommended\s+that\s+(?:the\s+)?([^,\.]+?)(?:\s+(?:should|works?|requires?|ensures?))',
        r'(?:Safety\s+)?[Rr]ecommendation[^:]*:\s*(?:HSIB|HSSIB)\s+recommends\s+that\s+(?:the\s+)?([^,\.]+?)(?:\s+(?:should|works?|working))',
        r'recommends\s+that\s+(?:the\s+)?([^,\.]+?)(?:,|\s+(?:should|works?|produces?|evaluates?|reviews?|assesses?|working))',
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
    
    resp_patterns = [
        r'^(NHS\s+England(?:\s+and\s+NHS\s+Improvement)?)',
        r'^(Care\s+Quality\s+Commission|CQC)',
        r'^(The\s+(?:CQC|Care\s+Quality\s+Commission))',
        r'^(NICE|National\s+Institute)',
        r'^(Royal\s+College)',
        r'^(DHSC|Department\s+of\s+Health)',
        r'^(NIHR|National\s+Institute\s+for\s+Health\s+Research)',
    ]
    
    clean_text = resp_text.strip()
    for pattern in resp_patterns:
        match = re.match(pattern, clean_text, re.IGNORECASE)
        if match:
            org_text = match.group(1).strip()
            return _map_to_canonical_org(org_text)
    
    first_200 = clean_text[:200].lower()
    for canonical, variants in ORGANISATION_MAPPINGS.items():
        for variant in variants:
            if re.search(variant, first_200, re.IGNORECASE):
                return canonical
    
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
    
    if 'nhs' in org_lower and 'england' in org_lower:
        return 'nhs_england'
    if 'cqc' in org_lower or 'care quality' in org_lower:
        return 'cqc'
    if 'nice' in org_lower or 'national institute' in org_lower:
        return 'nice'
    if 'royal college' in org_lower:
        return 'royal_college'
    if 'dhsc' in org_lower or 'department of health' in org_lower:
        return 'dhsc'
    
    return None


# =============================================================================
# RECOMMENDATION-RESPONSE MATCHER
# =============================================================================

class RecommendationResponseMatcher:
    """
    Semantic/keyword matcher for recommendation-response alignment.
    Uses 3-tier priority: number_match → org_match → semantic
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
            logger.info("✅ Sentence transformer model loaded")
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
        """
        Match recommendations to responses using 3-tier priority:
        1. number_match: Explicit rec number in response
        2. org_match: Organisation alignment (HSIB)
        3. semantic: Embedding/keyword similarity
        """
        if not recommendations or not responses:
            return []

        # Preprocess responses
        cleaned_responses = []
        for resp in responses:
            cleaned_text = self._clean_response_text(resp.get("text", ""))
            source_org = extract_source_organisation(cleaned_text)
            cleaned_responses.append({
                **resp, 
                "cleaned_text": cleaned_text,
                "source_org": source_org
            })

        # Build indexes
        responses_by_number: Dict[str, List[Dict]] = {}
        responses_by_org: Dict[str, List[Dict]] = {}
        
        for resp in cleaned_responses:
            if resp.get("rec_number"):
                num = resp["rec_number"]
                responses_by_number.setdefault(num, []).append(resp)
            if resp.get("source_org"):
                org = resp["source_org"]
                responses_by_org.setdefault(org, []).append(resp)

        logger.info(f"Response indexes: {len(responses_by_number)} by number, {len(responses_by_org)} by org")

        # Get embeddings if available
        rec_embeddings = None
        resp_embeddings = None
        similarity_matrix = None
        
        if self.use_transformer and self.model:
            rec_texts = [r.get("text", "") for r in recommendations]
            resp_texts = [r["cleaned_text"] for r in cleaned_responses]
            
            rec_embeddings = self.encode_texts(rec_texts)
            resp_embeddings = self.encode_texts(resp_texts)
            
            if rec_embeddings is not None and resp_embeddings is not None:
                rec_norms = np.linalg.norm(rec_embeddings, axis=1, keepdims=True)
                resp_norms = np.linalg.norm(resp_embeddings, axis=1, keepdims=True)
                rec_embeddings_norm = rec_embeddings / rec_norms
                resp_embeddings_norm = resp_embeddings / resp_norms
                similarity_matrix = np.dot(rec_embeddings_norm, resp_embeddings_norm.T)

        alignments = []
        
        for rec_idx, rec in enumerate(recommendations):
            best_match = None
            best_score = 0.0
            match_method = "none"
            
            rec_text = rec.get("text", "")
            rec_num = self._extract_rec_number(rec_text)
            target_org = extract_target_organisation(rec_text)
            
            # TIER 1: Number match (highest priority)
            if rec_num and rec_num in responses_by_number:
                for resp in responses_by_number[rec_num]:
                    if self._is_self_match(rec_text, resp["cleaned_text"]):
                        continue
                    
                    if similarity_matrix is not None:
                        resp_idx = cleaned_responses.index(resp)
                        score = float(similarity_matrix[rec_idx][resp_idx])
                    else:
                        score = self._keyword_similarity(rec_text, resp["cleaned_text"])
                    
                    score = min(score + 0.4, 1.0)  # Boost for number match
                    
                    if score > best_score:
                        best_score = score
                        status, status_conf = self._classify_response_status(resp["cleaned_text"])
                        best_match = {
                            "response_text": resp["cleaned_text"],
                            "similarity": score,
                            "status": status,
                            "status_confidence": status_conf,
                            "source_document": resp.get("source_document", "unknown"),
                            "match_method": "number_match",
                        }
                        match_method = "number_match"
            
            # TIER 2: Organisation match (HSIB documents)
            if best_match is None and target_org and target_org in responses_by_org:
                for resp in responses_by_org[target_org]:
                    if self._is_self_match(rec_text, resp["cleaned_text"]):
                        continue
                    
                    if similarity_matrix is not None:
                        resp_idx = cleaned_responses.index(resp)
                        score = float(similarity_matrix[rec_idx][resp_idx])
                    else:
                        score = self._keyword_similarity(rec_text, resp["cleaned_text"])
                    
                    score = min(score + 0.3, 1.0)  # Boost for org match
                    
                    if score > best_score:
                        best_score = score
                        status, status_conf = self._classify_response_status(resp["cleaned_text"])
                        best_match = {
                            "response_text": resp["cleaned_text"],
                            "similarity": score,
                            "status": status,
                            "status_confidence": status_conf,
                            "source_document": resp.get("source_document", "unknown"),
                            "match_method": "org_match",
                        }
                        match_method = "org_match"
            
            # TIER 3: Semantic/keyword matching
            if best_match is None:
                for resp_idx, resp in enumerate(cleaned_responses):
                    if self._is_self_match(rec_text, resp["cleaned_text"]):
                        continue
                    if is_recommendation_text(resp["cleaned_text"]):
                        continue
                    
                    if similarity_matrix is not None:
                        score = float(similarity_matrix[rec_idx][resp_idx])
                    else:
                        score = self._keyword_similarity(rec_text, resp["cleaned_text"])
                    
                    if score < 0.4:
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
                        match_method = "semantic"
            
            logger.info(f"Rec {rec_num or rec_idx}: matched via {match_method} (score={best_score:.2f})")
            
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

        gov_terms = {"government", "recommendation", "accept", "support", "implement", 
                     "policy", "department", "nhs", "england", "health", "care", 
                     "provider", "commissioner", "trust", "board", "cqc"}
        gov_matches = len((words1 & words2) & gov_terms)
        gov_boost = min(gov_matches * 0.05, 0.15)
        return min(base_score + gov_boost, 1.0)

    def _clean_response_text(self, text: str) -> str:
        markers = [
            "Government response to recommendation",
            "The government supports",
            "The government accepts",
            "NHS England",
            "CQC",
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
        return False

    def _has_government_response_language(self, text: str) -> bool:
        text_lower = text.lower()
        indicators = [
            "government response", "the government supports", "the government accepts",
            "the government agrees", "we accept", "we support", "we agree",
            "accept the recommendation", "nhs england", "cqc",
        ]
        return any(ind in text_lower for ind in indicators)

    def _extract_rec_number(self, text: str) -> Optional[str]:
        # HSIB format: 2018/006, R/2023/220
        hsib_match = re.search(r'(?:R/)?(\d{4}/\d{3})', text)
        if hsib_match:
            return hsib_match.group(1)
        # Standard format
        match = re.search(r"recommendation\s+(\d+)", text.lower())
        return match.group(1) if match else None

    def _classify_response_status(self, text: str) -> Tuple[str, float]:
        """Classify government response status."""
        text_lower = text.lower()
        
        # PRIORITY 1: Full rejection
        rejection_patterns = [
            r"\breject(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bdoes?\s+not\s+accept\s+(?:this\s+|the\s+)?recommendation",
            r"\bcannot\s+accept\s+(?:this\s+|the\s+)?recommendation",
        ]
        for pattern in rejection_patterns:
            if re.search(pattern, text_lower):
                return "Rejected", 0.9
        
        # PRIORITY 2: Partial acceptance
        partial_patterns = [
            r"\baccept(?:s|ed)?\s+(?:this\s+)?(?:recommendation\s+)?in\s+(?:part|principle)",
            r"\bpartially\s+accept",
            r"\bsupport(?:s|ed)?\s+(?:this\s+)?(?:recommendation\s+)?in\s+principle",
            r"\bunder\s+consideration",
            r"\bwill\s+consider\s+(?:the\s+|this\s+)?(?:recommendation|proposal)",
        ]
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return "Partial", 0.85
        
        # PRIORITY 3: Explicit acceptance
        accepted_patterns = [
            r"\baccept(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bsupport(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation",
            r"\bagree(?:s|d)?\s+(?:with\s+)?(?:this\s+|the\s+)?recommendation",
            r"\bthe\s+government\s+support(?:s|ed)?\s+(?:this|the)\b",
            r"\bwill\s+implement\b",
            r"\bfully\s+accept",
        ]
        for pattern in accepted_patterns:
            if re.search(pattern, text_lower):
                return "Accepted", 0.9
        
        # PRIORITY 4: Implicit acceptance
        implicit_patterns = [
            r"\bcommitted\s+to\s+(?:working|improving|delivering)",
            r"\bintends?\s+to\s+(?:provide|publish|deliver)",
            r"\bwill\s+(?:provide|publish|deliver|work\s+with)",
            r"\balready\s+(?:taking|undertaking|progressing)",
            r"\baction\s+(?:is\s+)?underway",
        ]
        for pattern in implicit_patterns:
            if re.search(pattern, text_lower):
                return "Accepted", 0.8
        
        # PRIORITY 5: Noted
        if re.search(r"\bnote(?:s|d)?\s+(?:this\s+|the\s+)?recommendation", text_lower):
            return "Noted", 0.75
        
        # Fallback
        if re.search(r"^(?:nhs\s+england|the\s+government)\s+support", text_lower):
            return "Accepted", 0.85
        
        return "Unclear", 0.5


# =============================================================================
# RESPONSE EXTRACTION HELPERS
# =============================================================================

def is_recommendation_text(text: str) -> bool:
    """Check if text is a recommendation (not a response)."""
    text_lower = text.lower().strip()
    recommendation_starters = [
        r"^nhs\s+england\s+should",
        r"^providers?\s+should",
        r"^trusts?\s+should",
        r"^boards?\s+should",
        r"^cqc\s+should",
        r"^dhsc\s+should",
        r"^we\s+recommend",
        r"^it\s+is\s+recommended\s+that",
        r"^hsib\s+recommends",
        r"^hssib\s+recommends",
    ]
    return any(re.search(pattern, text_lower) for pattern in recommendation_starters)


def is_genuine_response(text: str) -> bool:
    """Check if text is a genuine response."""
    text_lower = text.lower().strip()
    strong_starters = [
        r"^government\s+response",
        r"^the\s+government\s+supports?",
        r"^the\s+government\s+accepts?",
        r"^the\s+government\s+agrees?",
        r"^nhs\s+england\s+(?:has|will|is|continues)",
        r"^cqc\s+(?:has|will|is|welcomes)",
        r"^we\s+support",
        r"^we\s+accept",
    ]
    return any(re.search(pattern, text_lower) for pattern in strong_starters)


def has_pdf_artifacts(text: str) -> bool:
    artifacts = [
        r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?",
        r"GOV\.UK",
        r"https?://www\.gov\.uk",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in artifacts)


def clean_pdf_artifacts(text: str) -> str:
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?\s*", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"Government response to the rapid review.*?GOV\.UK[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# HSIB RESPONSE EXTRACTION - CLEAN VERSION
# =============================================================================

def is_hsib_response_document(text: str) -> bool:
    """Detect if a document is an HSIB/HSSIB-style response document."""
    # Check for standard government format first
    gov_response_pattern = re.compile(
        r'Government\s+response\s+to\s+recommendation\s+\d+',
        re.IGNORECASE
    )
    if len(list(gov_response_pattern.finditer(text))) >= 2:
        return False
    
    # Count standalone "Response" headers (PyPDF2 compatible)
    response_header_pattern = re.compile(
        r'Response\s*(?:\n|(?=[A-Z][a-z]))(?!.*?received)',
        re.IGNORECASE
    )
    response_count = len(list(response_header_pattern.finditer(text)))
    
    logger.info(f"HSIB detection: found {response_count} Response headers")
    
    if response_count >= 3:
        return True
    
    # Check for HSIB indicators
    hsib_indicators = [
        r'\bHSIB\s+recommends\b',
        r'\bHSSIB\s+recommends\b',
        r'\bSafety\s+recommendation\s+R/\d{4}/\d{3}\b',
        r'\bRecommendation\s+\d{4}/\d{3}\b',
        r'\bResponse\s+received\s+on\b',
        r'\bTimescale:\s*(?:completed|already|Spring|January|December)\b',
    ]
    
    indicator_matches = sum(1 for pattern in hsib_indicators 
                           if re.search(pattern, text, re.IGNORECASE))
    
    logger.info(f"HSIB detection: found {indicator_matches} HSIB indicators")
    
    if response_count >= 1 and indicator_matches >= 1:
        return True
    if indicator_matches >= 3:
        return True
    
    return False


def extract_hsib_responses(text: str) -> List[Dict]:
    """
    Extract responses from HSIB-format documents.
    Uses multi-pattern approach for PyPDF2 compatibility.
    Clean boundary detection prevents text bleeding.
    """
    if not text:
        return []
    
    responses = []
    
    # Find recommendation positions
    hsib_rec_patterns = [
        re.compile(r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]', re.IGNORECASE),
        re.compile(r'Recommendation\s+(\d{4}/\d{3})[:\s]', re.IGNORECASE),
    ]
    
    rec_matches = []
    for pattern in hsib_rec_patterns:
        rec_matches.extend(list(pattern.finditer(text)))
    rec_matches.sort(key=lambda m: m.start())
    
    # Multi-pattern response detection (PyPDF2 compatible)
    # NO strict line-start requirement - PyPDF2 doesn't preserve newlines consistently
    response_patterns = [
        re.compile(r'Response\s*\n', re.IGNORECASE),
        re.compile(r'Response(?=[A-Z][a-z])', re.IGNORECASE),
        re.compile(r'Response\s*\n?\s*Timescale', re.IGNORECASE),
        re.compile(r'Response\s*\n?\s*(?:NHS|The|Our|We|This|As|Funding|safe)', re.IGNORECASE),
    ]
    
    response_matches = []
    for pattern in response_patterns:
        response_matches.extend(list(pattern.finditer(text)))
    
    # Deduplicate by position (20-char window)
    seen_positions = set()
    unique_matches = []
    for match in sorted(response_matches, key=lambda m: m.start()):
        pos_key = match.start() // 20
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            unique_matches.append(match)
    response_matches = unique_matches
    
    logger.info(f"HSIB: Found {len(response_matches)} unique Response headers")
    
    # Organisation headers as boundaries
    org_header_pattern = re.compile(
        r'(?:^|\n)\s*(Department\s+of\s+Health|NHS\s+England|Care\s+Quality\s+Commission|DHSC|NICE|CQC|Royal\s+College)\s*\n',
        re.IGNORECASE | re.MULTILINE
    )
    org_headers = list(org_header_pattern.finditer(text))
    
    # Boundary markers to prevent text bleeding
    boundary_patterns = [
        re.compile(r'\nSafety\s+(?:observation|action)', re.IGNORECASE),
        re.compile(r'\nResponse\s+received\s+on', re.IGNORECASE),
        re.compile(r'\nActions\s+planned\s+to\s+deliver', re.IGNORECASE),
        re.compile(r'\n(?:Recommendation|Safety\s+recommendation)\s+(?:R/)?\d', re.IGNORECASE),
    ]
    
    # Associate responses with recommendations
    for idx, resp_match in enumerate(response_matches):
        start = resp_match.end()
        end = len(text)
        
        # Find nearest preceding recommendation
        rec_number = None
        for rec_match in reversed(rec_matches):
            if rec_match.start() < resp_match.start():
                rec_number = rec_match.group(1)
                break
        
        # Find end boundary
        # 1. Next response header
        if idx + 1 < len(response_matches):
            end = min(end, response_matches[idx + 1].start())
        
        # 2. Next recommendation header
        for rec_match in rec_matches:
            if rec_match.start() > start and rec_match.start() < end:
                end = rec_match.start()
                break
        
        # 3. Organisation headers
        for org_match in org_headers:
            if org_match.start() > start and org_match.start() < end:
                end = org_match.start()
                break
        
        # 4. Boundary markers
        for boundary_pattern in boundary_patterns:
            boundary_match = boundary_pattern.search(text, start)
            if boundary_match and boundary_match.start() < end:
                end = boundary_match.start()
        
        # Extract and clean response text
        resp_text = text[start:end].strip()
        resp_text = clean_pdf_artifacts(resp_text)
        
        # Skip if too short or is actually a recommendation
        if len(resp_text) < 50:
            continue
        if is_recommendation_text(resp_text):
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
    
    logger.info(f"Extracted {len(responses)} HSIB responses")
    return responses


def extract_response_sentences(text: str) -> List[Dict]:
    """
    Extract government responses from document text.
    Detects HSIB format and uses appropriate extraction method.
    """
    if not text:
        return []

    text = clean_pdf_artifacts(text)
    
    # Check for HSIB format
    if is_hsib_response_document(text):
        logger.info("Detected HSIB response document format")
        return extract_hsib_responses(text)
    
    # Standard government format
    responses: List[Dict] = []
    seen_responses = set()

    # Find "Government response to recommendation N" sections
    gov_resp_pattern = re.compile(
        r"Government\s+response\s+to\s+recommendation\s+(\d+)",
        re.IGNORECASE
    )
    gov_resp_starts = list(gov_resp_pattern.finditer(text))
    
    logger.info(f"Found {len(gov_resp_starts)} government response headers")
    
    structured_spans: List[Tuple[int, int]] = []
    
    for i, match in enumerate(gov_resp_starts):
        start_pos = match.end()
        rec_num = match.group(1)
        end_pos = len(text)
        
        # Find end boundary
        if i + 1 < len(gov_resp_starts):
            end_pos = gov_resp_starts[i + 1].start()
        
        # Check for next recommendation marker
        rec_marker = re.search(r"Recommendation\s+\d+\s+[A-Z]", text[start_pos:end_pos])
        if rec_marker:
            end_pos = start_pos + rec_marker.start()
        
        resp_content = text[start_pos:end_pos].strip()
        if not resp_content:
            continue
        
        resp_content = clean_pdf_artifacts(resp_content)
        
        responses.append({
            "text": resp_content,
            "position": start_pos,
            "response_type": "structured",
            "confidence": 0.95,
            "rec_number": rec_num,
        })
        structured_spans.append((start_pos, end_pos))
    
    # Fallback: sentence-level extraction for scattered responses
    if not responses:
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text)
        
        for idx, sentence in enumerate(sentences):
            sentence_clean = clean_pdf_artifacts(sentence.strip())
            
            if len(sentence_clean) < 40 or len(sentence_clean) > 500:
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
    
    logger.info(f"Extracted {len(responses)} responses")
    return responses


# =============================================================================
# LIGHTWEIGHT ALIGNMENT HELPERS (for UI)
# =============================================================================

def classify_content_type(sentence: str) -> str:
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
    if not responses:
        return "No response found"
    top_response = responses[0]
    if top_response.get("same_document"):
        return "Direct response"
    if top_response.get("combined_score", 0) >= 0.5:
        return "Possible response"
    return "Uncertain"


def align_recommendations_with_responses(
    recommendations: List[Dict[str, Any]], 
    responses: List[Dict[str, Any]], 
    similarity_threshold: float
) -> List[Dict[str, Any]]:
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
    words1 = set(get_meaningful_words(text1))
    words2 = set(get_meaningful_words(text2))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


__all__ = [
    "RecommendationResponseMatcher",
    "is_recommendation_text",
    "is_genuine_response",
    "has_pdf_artifacts",
    "clean_pdf_artifacts",
    "extract_response_sentences",
    "extract_hsib_responses",
    "is_hsib_response_document",
    "extract_target_organisation",
    "extract_source_organisation",
    "find_pattern_matches",
    "align_recommendations_with_responses",
    "determine_alignment_status",
    "calculate_simple_similarity",
    "classify_content_type",
]
