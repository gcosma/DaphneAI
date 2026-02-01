"""
Document format detection and text classification helpers.

This module handles:
- Detection of document format (standard government vs HSIB vs Trust vs HSSIB)
- Text classification (recommendation vs response)
- PDF artifact handling
- Organisation extraction from text

v3.1 Changes:
- FIXED: is_trust_response_document() pattern now matches "Recommendation 9 is..." (lowercase after number)

v3.0 Changes:
- ADDED: detect_excluded_recommendations() - detects recs addressed to other orgs

v2.8 Changes:
- ADDED: is_hssib_org_structured_response() function for Report 6 style

v2.7 Changes:
- FIXED: Line ending normalisation for all detection functions
- ADDED: Comprehensive debug logging for detection failures

Split from response_extractor.py for cleaner separation of concerns.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Line ending normalisation
# --------------------------------------------------------------------------- #

def normalise_line_endings(text: str) -> str:
    """
    Normalise all line endings to \\n for consistent regex matching.
    
    Handles:
    - Windows: \\r\\n -> \\n
    - Old Mac: \\r -> \\n
    - Unix: \\n (unchanged)
    """
    if not text:
        return text
    # First convert \r\n to \n, then convert remaining \r to \n
    return text.replace('\r\n', '\n').replace('\r', '\n')


# --------------------------------------------------------------------------- #
# Excluded recommendations detection
# --------------------------------------------------------------------------- #

def detect_excluded_recommendations(text: str) -> List[int]:
    """
    Detect recommendations that are explicitly stated as not included.
    
    Report 7 example: "Recommendations 4, 5, 10 and 12 relate to other 
    organisations and are therefore not included in this assurance statement."
    
    v3.0: NEW function
    
    Returns list of recommendation numbers that should be marked as excluded/N/A.
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    excluded = []
    
    # Patterns for detecting excluded recommendations
    # Note: handles "5,10" (no space) as well as "5, 10" (with space)
    patterns = [
        # "Recommendations 4, 5, 10 and 12 relate to other organisations"
        r'[Rr]ecommendations?\s+([\d,\s]+(?:and\s+\d+)?)\s+relate\s+to\s+other\s+organi[sz]ations',
        # "Recommendations 4, 5, 10 and 12 are therefore not included"
        r'[Rr]ecommendations?\s+([\d,\s]+(?:and\s+\d+)?)\s+(?:are|is)\s+(?:therefore\s+)?not\s+included',
        # "Recommendations 4, 5, 10 and 12 are for other organisations"
        r'[Rr]ecommendations?\s+([\d,\s]+(?:and\s+\d+)?)\s+(?:are\s+)?for\s+other\s+organi[sz]ations',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            nums_text = match.group(1)
            nums = re.findall(r'\d+', nums_text)
            excluded.extend([int(n) for n in nums])
            logger.info(f"Found excluded recommendations from pattern: {excluded}")
    
    return list(set(excluded))


# --------------------------------------------------------------------------- #
# Text classification helpers
# --------------------------------------------------------------------------- #

def is_recommendation_text(text: str) -> bool:
    """
    Check if text appears to be a recommendation rather than a response.
    
    Used to filter out recommendation text that might be mistakenly
    extracted as a response.
    """
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
    """
    Check if text appears to be a genuine government/organisation response.
    
    Used to identify response text during sentence-level extraction.
    """
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
# Document format detection functions
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
    
    text = normalise_line_endings(text)
    
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


def is_trust_response_document(text: str) -> bool:
    """
    Detect if a document is a Trust response document (Report 7 style).
    
    Format: "Recommendation N" followed by "TEWV response:" or "[Trust] response:"
    
    v3.1: FIXED - Pattern now matches "Recommendation 9 is..." (lowercase after number)
    v2.7: Updated to handle PDFs with no line breaks
    """
    if not text:
        return False
    
    text = normalise_line_endings(text)
    
    # Check for Trust response patterns - more flexible (no newline required)
    has_trust_response = bool(re.search(r'(?:TEWV|Trust)\s+response[:\s]', text, re.IGNORECASE))
    
    # v3.1 FIX: More flexible pattern - just needs "Recommendation N" anywhere
    # Old pattern required uppercase letter or newline after number, missing "Recommendation 9 is..."
    has_recommendation_headers = bool(re.search(r'Recommendation\s+\d+', text, re.IGNORECASE))
    
    logger.info(f"Trust detection: has_trust_response={has_trust_response}, has_recommendation_headers={has_recommendation_headers}")
    
    if has_trust_response and has_recommendation_headers:
        logger.info("✅ Trust response format detected")
        return True
    
    return False


def is_hssib_org_structured_response(text: str) -> bool:
    """
    Detect if a document is an HSSIB org-structured response (Report 6 style).
    
    Format: Organisation headers (Shelford Group, NHS England, DHSC) with
    "HSSIB recommends" + rec IDs (R/2024/XXX) + "Response" headers +
    "Actions planned" + "Response received on" markers.
    
    This differs from standard HSIB format because responses are grouped by ORG,
    not just by rec ID sequence.
    
    v2.8: NEW - handles Report 6 format
    """
    if not text:
        return False
    
    text = normalise_line_endings(text)
    
    # Must have HSSIB recommendation pattern with rec IDs
    has_hssib_rec = bool(re.search(r'HSSIB\s+recommends', text, re.IGNORECASE))
    has_rec_ids = bool(re.search(r'R/\d{4}/\d{3}', text))
    
    # Must have standalone "Response" headers (not just "Response received")
    has_response_headers = bool(re.search(r'(?:^|\n)Response\n', text, re.MULTILINE))
    
    # Must have org headers (including The Shelford Group)
    org_pattern = r'(?:^|\n)(The\s+Shelford\s+Group|NHS\s+England|Department\s+of\s+Health\s+and\s+Social\s+Care|DHSC|Care\s+Quality\s+Commission|NICE)\s*\n'
    has_org_headers = bool(re.search(org_pattern, text, re.IGNORECASE | re.MULTILINE))
    
    # Must have "Actions planned" or "Response received on" markers (HSSIB specific)
    has_hssib_markers = bool(re.search(r'Actions\s+planned\s+to\s+deliver|Response\s+received\s+on', text, re.IGNORECASE))
    
    logger.info(f"HSSIB org-structured detection: hssib_rec={has_hssib_rec}, rec_ids={has_rec_ids}, "
                f"response_headers={has_response_headers}, org_headers={has_org_headers}, hssib_markers={has_hssib_markers}")
    
    if has_hssib_rec and has_rec_ids and has_response_headers and has_org_headers and has_hssib_markers:
        logger.info("✅ HSSIB org-structured response format detected (Report 6 style)")
        return True
    
    return False


def is_org_based_hsib_response(text: str) -> bool:
    """
    Detect if a document is an org-based HSIB response (Report 4 style).
    
    Format: Organisation headers (NHS England, CQC, etc.) followed by 
    "HSIB recommends..." then "Response" header, but NO rec IDs like R/2023/220
    
    v2.7: Updated to handle PDFs with no line breaks - uses context patterns
    """
    if not text:
        return False
    
    text = normalise_line_endings(text)
    
    # Must have "HSIB recommends" pattern
    has_hsib_recommends = bool(re.search(r'HSIB\s+recommends', text, re.IGNORECASE))
    
    # Check for "Response" followed by org starting their response
    response_followed_by_content = bool(re.search(
        r'\bResponse\b\s*(?:NHS\s+England|Care\s+Quality\s+Commission|The\s+National|We\s+(?:are|have|welcome)|NICE\s+is)',
        text, re.IGNORECASE
    ))
    
    # Also check for standalone Response on its own line (if newlines exist)
    has_response_on_line = bool(re.search(r'(?:^|\n)\s*Response\s*(?:\n|$)', text, re.MULTILINE))
    
    has_response_headers = response_followed_by_content or has_response_on_line
    
    # Check for org headers
    org_on_line = bool(re.search(
        r'(?:^|\n)\s*(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute|Royal\s+College|NICE|DHSC|The\s+Shelford\s+Group)\s*(?:\n|$)',
        text, re.IGNORECASE | re.MULTILINE
    ))
    
    org_after_section = bool(re.search(
        r'(?:process|planning|guidance|services|management)[.\s]+(?:Response\s+)?(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute|Royal\s+College)',
        text, re.IGNORECASE
    ))
    
    org_names = re.findall(
        r'\b(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|Royal\s+College\s+of\s+Psychiatrists)\b',
        text, re.IGNORECASE
    )
    has_multiple_orgs = len(set(org.lower() for org in org_names)) >= 2
    
    has_org_headers = org_on_line or org_after_section or has_multiple_orgs
    
    # Should NOT have HSIB rec IDs (otherwise use HSSIB org-structured or standard HSIB)
    has_hsib_rec_ids = bool(re.search(r'R/\d{4}/\d{3}', text))
    
    logger.info(f"Org-based HSIB detection: hsib_recommends={has_hsib_recommends}, "
                f"response_headers={has_response_headers}, org_headers={has_org_headers}, "
                f"has_rec_ids={has_hsib_rec_ids}")
    
    if has_hsib_recommends and has_response_headers and has_org_headers and not has_hsib_rec_ids:
        logger.info("✅ Org-based HSIB response format detected (Report 4 style)")
        return True
    
    # Log why it failed
    if not has_hsib_recommends:
        logger.info("❌ Org-based HSIB: Missing 'HSIB recommends' pattern")
    if not has_response_headers:
        logger.info("❌ Org-based HSIB: Missing 'Response' headers")
    if not has_org_headers:
        logger.info("❌ Org-based HSIB: Missing organisation headers")
    if has_hsib_rec_ids:
        logger.info("❌ Org-based HSIB: Has R/YYYY/NNN rec IDs (use HSSIB org-structured or standard HSIB)")
    
    return False


# --------------------------------------------------------------------------- #
# Organisation extraction helpers
# --------------------------------------------------------------------------- #

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
    if 'tewv' in text_lower or 'tees, esk and wear' in text_lower:
        return 'tewv'
    if 'middlesbrough council' in text_lower:
        return 'middlesbrough_council'
    if 'south tees' in text_lower:
        return 'south_tees'
    if 'provider collaborative' in text_lower:
        return 'provider_collaborative'
    
    return None


def extract_recommendation_target_org(rec_text: str) -> Optional[str]:
    """
    Extract the TARGET organisation that a recommendation is directed TO.
    
    This looks for patterns like:
    - "TEWV must ensure..." -> tewv
    - "NHS England should..." -> nhs_england
    - "Middlesbrough Council must respond..." -> middlesbrough_council
    
    Returns normalised org key or None.
    """
    if not rec_text:
        return None
    
    target_patterns = [
        (r'\bTEWV\s+(?:must|should|needs?\s+to)', 'tewv'),
        (r'\bTees,?\s+Esk\s+and\s+Wear\s+Valleys?\s+(?:NHS\s+)?(?:Foundation\s+)?Trust\s+(?:must|should)', 'tewv'),
        (r'\bNHS\s+England\s+(?:must|should|needs?\s+to|and\s+provider|works)', 'nhs_england'),
        (r'\bMiddlesbrough\s+Council\s+(?:must|should|needs?\s+to|and\s+Health)', 'middlesbrough_council'),
        (r'\bSouth\s+Tees\s+Safeguarding\s+Children', 'south_tees'),
        (r'\bCare\s+Quality\s+Commission\s+(?:must|should|evaluates)', 'cqc'),
        (r'\bNICE\s+(?:must|should|evaluates)', 'nice'),
        (r'\bNational\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence\s+(?:must|should|evaluates)', 'nice'),
        (r'\bRoyal\s+College\s+(?:of\s+Psychiatrists\s+)?(?:must|should|forms)', 'royal_college'),
        (r'\bDHSC\s+(?:must|should)', 'dhsc'),
        (r'\bprovider\s+collaborative[s]?\s+(?:must|should)', 'provider_collaborative'),
    ]
    
    for pattern, org_key in target_patterns:
        if re.search(pattern, rec_text, re.IGNORECASE):
            return org_key
    
    return None


def get_response_document_org(text: str) -> Optional[str]:
    """
    Determine which organisation authored a response document.
    
    Looks at document headers and common patterns to identify the responding org.
    """
    if not text:
        return None
    
    text_lower = text[:2000].lower()  # Check first 2000 chars
    
    # Check for Trust response documents
    if 'tewv response' in text_lower or 'tees, esk and wear' in text_lower:
        return 'tewv'
    
    # Check for org-based HSIB responses (multiple orgs)
    if 'hsib recommends that nhs england' in text_lower:
        return 'multi_org_hsib'
    
    # Check for government response
    if 'government response' in text_lower:
        return 'government'
    
    return None


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #

__all__ = [
    # Line ending normalisation
    "normalise_line_endings",
    # Excluded recommendations
    "detect_excluded_recommendations",
    # Text classification
    "is_recommendation_text",
    "is_genuine_response",
    # PDF artifacts
    "has_pdf_artifacts",
    "clean_pdf_artifacts",
    # Document format detection
    "is_hsib_response_document",
    "is_trust_response_document",
    "is_hssib_org_structured_response",
    "is_org_based_hsib_response",
    # Organisation extraction
    "extract_target_org_from_text",
    "extract_recommendation_target_org",
    "get_response_document_org",
]
