"""
Response extraction from government and HSIB/HSSIB response documents.

This module handles:
- Detection of document format (standard government vs HSIB)
- Extraction of responses from PDF text
- Text cleaning and artifact removal

v2.5 Changes:
- FIXED: HSIB response extraction now segments by recommendation ID
- FIXED: Response boundaries properly detect org headers and rec IDs
- ADDED: Response-to-rec ID linking in HSIB format extraction

Split from alignment_engine.py for cleaner separation of concerns.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Text classification helpers
# --------------------------------------------------------------------------- #

def is_recommendation_text(text: str) -> bool:
    """Check if text appears to be a recommendation rather than a response."""
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
        r"^hssib\s+recommends",
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
        r"^we\s+are\s+happy\s+to\s+confirm",
        r"^dhsc\s+and\s+nhs\s+england\s+support",
        r"^nhs\s+england\s+(?:has|will|is)",
        r"^care\s+quality\s+commission\s+(?:is|has|will)",
        r"^the\s+national\s+institute\s+for\s+health",
        r"^nice\s+(?:is|has|will)",
    ]
    return any(re.search(pattern, text_lower) for pattern in strong_starters)


# --------------------------------------------------------------------------- #
# PDF artifact handling
# --------------------------------------------------------------------------- #

def has_pdf_artifacts(text: str) -> bool:
    """Check if text contains common PDF extraction artifacts."""
    artifacts = [
        r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?",
        r"GOV\.UK",
        r"https?://www\.gov\.uk",
        r"https?://www\.england\.nhs\.uk",
        r"\d+/\d+\s*$",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in artifacts)


def clean_pdf_artifacts(text: str) -> str:
    """Remove common PDF extraction artifacts from text."""
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?\s*", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"Government response to the rapid review.*?GOV\.UK[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# HSIB format detection and extraction
# --------------------------------------------------------------------------- #

def is_hsib_response_document(text: str) -> bool:
    """
    Detect if a document is an HSIB/HSSIB-style response document.
    
    HSIB documents use recommendation IDs like R/2024/025 or 2018/006
    and have "Response" headers after each recommendation.
    
    v2.5: Ultra-forgiving detection for PyPDF2 mangled text.
    """
    if not text:
        return False
    
    # Very forgiving HSIB recommendation patterns
    # Handles: "R/2024/025", "2018/006", "R / 2024 / 025", "2018 / 006"
    has_hsib_rec = bool(re.search(r'[Rr]ecommendation\s*[R/]?\s*\d{4}\s*/\s*\d{3}', text))
    
    # Very forgiving Response header pattern
    has_response_header = bool(re.search(r'\bResponse\b', text, re.IGNORECASE))
    
    # HSIB/HSSIB mentions
    has_hsib_mention = bool(re.search(r'\bH[SS]I?B\b', text, re.IGNORECASE))
    
    # If we have HSIB-style recommendation IDs, it's definitely HSIB format
    if has_hsib_rec:
        logger.info("✅ HSIB format detected: Found HSIB-style recommendation IDs")
        return True
    
    # Or if we have HSIB mentions + Response headers
    if has_hsib_mention and has_response_header:
        logger.info("✅ HSIB format detected: Found HSIB mentions + Response headers")
        return True
    
    logger.info(f"❌ HSIB format NOT detected (has_rec: {has_hsib_rec}, has_resp: {has_response_header}, has_hsib: {has_hsib_mention})")
    return False


def extract_target_org_from_text(text: str) -> Optional[str]:
    """
    Extract target organisation from recommendation or response text.
    
    Returns a normalised org key for matching purposes.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Check for specific organisations (order matters - more specific first)
    if 'national institute for health and care research' in text_lower or 'nihr' in text_lower:
        return 'nihr'
    if 'department of health and social care' in text_lower or 'dhsc' in text_lower:
        return 'dhsc'
    if 'nhs england' in text_lower:
        return 'nhs_england'
    if 'care quality commission' in text_lower or 'cqc' in text_lower:
        return 'cqc'
    if 'national institute for health and care excellence' in text_lower or 'nice' in text_lower:
        return 'nice'
    if 'royal college' in text_lower:
        return 'royal_college'
    if 'the shelford group' in text_lower or 'shelford group' in text_lower:
        return 'shelford_group'
    if 'tewv' in text_lower:
        return 'tewv'
    
    return None


def extract_hsib_responses(text: str) -> List[Dict]:
    """
    Extract responses from HSIB-format response documents.
    
    HSIB format uses:
    - Organisation name as section header
    - "Response" as standalone header
    - Responses follow immediately after the recommendation they address
    
    v2.5 FIX: Now properly links responses to rec_ids by looking for the 
    recommendation ID that precedes each Response header.
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
    
    # Multiple patterns to catch all Response headers (PyPDF2 compatibility)
    response_patterns = [
        re.compile(r'(?:^|\n)\s*Response\s*\n', re.IGNORECASE | re.MULTILINE),
        re.compile(r'Response(?=[A-Z][a-z])', re.IGNORECASE),  # Merged: "ResponseNHS England..."
    ]
    
    response_matches = []
    for pattern in response_patterns:
        response_matches.extend(list(pattern.finditer(text)))
    
    # Deduplicate by position (within 20 chars)
    unique_matches = []
    last_pos = -100
    for m in sorted(response_matches, key=lambda x: x.start()):
        if m.start() - last_pos > 20:
            unique_matches.append(m)
            last_pos = m.start()
    response_matches = unique_matches
    
    # Organisation headers that act as section boundaries
    org_header_pattern = re.compile(
        r'(?:^|\n)\s*(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|National\s+Institute|Royal\s+College\s+of\s+Psychiatrists|Royal\s+College|The\s+Shelford\s+Group|DHSC)\s*\n',
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
        
        # v2.5 FIX: Find the rec_id that PRECEDES this response header
        rec_id = None
        for rec_match in reversed(rec_matches):
            if rec_match.start() < resp_match.start():
                rec_id = rec_match.group(1)
                break
        
        # Extract organisation from response text
        source_org = extract_target_org_from_text(response_text)
        
        responses.append({
            'text': response_text,
            'position': start,
            'response_type': 'hsib_structured',
            'confidence': 0.95,
            'rec_number': rec_id,
            'rec_id': rec_id,
            'source_org': source_org,
        })
    
    logger.info(f"Extracted {len(responses)} HSIB responses")
    return responses


# --------------------------------------------------------------------------- #
# Main response extraction function
# --------------------------------------------------------------------------- #

def extract_response_sentences(text: str) -> List[Dict]:
    """
    Extract sentences that are government responses.
    Auto-detects HSIB vs standard government format.
    
    Returns a list of response dictionaries with:
    - text: The response text
    - position: Position in document
    - response_type: 'structured', 'hsib_structured', or 'sentence'
    - confidence: Extraction confidence score
    - rec_number: Associated recommendation number (if detected)
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

    # Find structured "Government response to recommendation N" sections
    gov_resp_starts = []
    for match in re.finditer(r"Government\s+response\s+to\s+recommendation\s+(\d+)", text, re.IGNORECASE):
        gov_resp_starts.append({"pos": match.start(), "end": match.end(), "rec_num": match.group(1)})

    # Find recommendation markers (to detect boundaries)
    rec_positions = []
    for match in re.finditer(r"Recommendation\s+(\d+)\s+([A-Z])", text):
        rec_positions.append({"pos": match.start(), "rec_num": match.group(1)})

    logger.info(f"Found {len(gov_resp_starts)} government response headers")
    logger.info(f"Found {len(rec_positions)} recommendation markers")

    structured_spans: List[Tuple[int, int]] = []
    structured_texts: List[str] = []

    # Extract structured responses
    for i, gov_resp in enumerate(gov_resp_starts):
        start_pos = gov_resp["end"]
        rec_num = gov_resp["rec_num"]
        end_pos = len(text)

        # Check for next government response header
        if i + 1 < len(gov_resp_starts):
            next_gov = gov_resp_starts[i + 1]["pos"]
            if next_gov < end_pos:
                end_pos = next_gov

        # Check for recommendation markers (indicates new section)
        for rec_pos in rec_positions:
            if rec_pos["pos"] > start_pos and rec_pos["pos"] < end_pos:
                end_pos = rec_pos["pos"]
                break

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
        structured_texts.append(resp_content)

    # Sentence-level response extraction (fallback for unstructured documents)
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
        
        # Skip very short or very long sentences
        if len(sentence_clean) < 40 or len(sentence_clean) > 500:
            continue
        
        # Skip if it looks like a recommendation
        if is_recommendation_text(sentence_clean):
            continue
        
        # Skip if it doesn't look like a response
        if not is_genuine_response(sentence_clean):
            continue
        
        # Skip if already covered by structured extraction
        if any(start <= start_idx < end for start, end in structured_spans):
            continue

        # Check for duplicates with structured responses
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

        # Check for exact duplicates
        resp_key = sentence_clean[:100].lower()
        if resp_key in seen_responses:
            continue
        seen_responses.add(resp_key)
        
        # Try to extract recommendation reference
        rec_ref = re.search(r"recommendation\s+(\d+)", sentence_clean.lower())
        
        responses.append({
            "text": sentence_clean,
            "position": idx,
            "response_type": "sentence",
            "confidence": 0.85,
            "rec_number": rec_ref.group(1) if rec_ref else None,
        })

    logger.info(f"Extracted {len(responses)} genuine responses")
    return responses


__all__ = [
    "is_recommendation_text",
    "is_genuine_response",
    "has_pdf_artifacts",
    "clean_pdf_artifacts",
    "is_hsib_response_document",
    "extract_target_org_from_text",
    "extract_hsib_responses",
    "extract_response_sentences",
]
