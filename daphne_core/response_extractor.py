"""
Response extraction from government and HSIB/HSSIB response documents.

This module handles:
- Detection of document format (standard government vs HSIB vs Trust)
- Extraction of responses from PDF text
- Text cleaning and artifact removal

v2.8 Changes:
- ADDED: HSSIB org-structured format detection (Report 6 style)
- ADDED: is_hssib_org_structured_response() function
- ADDED: extract_hssib_org_structured_responses() function
- FIXED: Report 6 now correctly extracts responses grouped by organisation

v2.7 Changes:
- FIXED: Line ending normalisation at start of extract_response_sentences()
- FIXED: All detection functions now handle \r, \n, and \r\n consistently
- ADDED: Comprehensive debug logging for detection failures
- IMPROVED: is_org_based_hsib_response() now logs why it fails

v2.6 Changes:
- ADDED: Trust response format detection (Report 7 style - "TEWV response:")
- ADDED: Org-based HSIB response format (Report 4 style - no rec IDs)
- ADDED: extract_trust_responses() function
- ADDED: extract_org_based_hsib_responses() function

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
# Line ending normalisation
# --------------------------------------------------------------------------- #

def normalise_line_endings(text: str) -> str:
    """
    Normalise all line endings to \n for consistent regex matching.
    
    Handles:
    - Windows: \r\n -> \n
    - Old Mac: \r -> \n
    - Unix: \n (unchanged)
    """
    if not text:
        return text
    # First convert \r\n to \n, then convert remaining \r to \n
    return text.replace('\r\n', '\n').replace('\r', '\n')


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
# Document format detection
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
    
    # Normalise line endings
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
    
    v2.7: Updated to handle PDFs with no line breaks
    """
    if not text:
        return False
    
    # Normalise line endings
    text = normalise_line_endings(text)
    
    # Check for Trust response patterns - more flexible (no newline required)
    has_trust_response = bool(re.search(r'(?:TEWV|Trust)\s+response[:\s]', text, re.IGNORECASE))
    
    # Check for recommendation headers - flexible pattern (newline OR followed by text)
    has_recommendation_headers = bool(re.search(r'Recommendation\s+\d+\s*(?:\n|[A-Z])', text))
    
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
    
    # Normalise line endings
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
    
    # Normalise line endings
    text = normalise_line_endings(text)
    
    # Must have "HSIB recommends" pattern
    has_hsib_recommends = bool(re.search(r'HSIB\s+recommends', text, re.IGNORECASE))
    
    # Check for "Response" followed by org starting their response
    # Pattern: "Response" followed by org name or response-starting words
    # e.g., "Response NHS England has begun..." or "Response We are happy to confirm..."
    response_followed_by_content = bool(re.search(
        r'\bResponse\b\s*(?:NHS\s+England|Care\s+Quality\s+Commission|The\s+National|We\s+(?:are|have|welcome)|NICE\s+is)',
        text, re.IGNORECASE
    ))
    
    # Also check for standalone Response on its own line (if newlines exist)
    has_response_on_line = bool(re.search(r'(?:^|\n)\s*Response\s*(?:\n|$)', text, re.MULTILINE))
    
    has_response_headers = response_followed_by_content or has_response_on_line
    
    # Check for org headers - either on own line OR as section starters after recommendations text
    # Pattern 1: Org on its own line
    org_on_line = bool(re.search(
        r'(?:^|\n)\s*(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute|Royal\s+College|NICE|DHSC|The\s+Shelford\s+Group)\s*(?:\n|$)',
        text, re.IGNORECASE | re.MULTILINE
    ))
    # Pattern 2: Org appearing after recommendation section markers (for no-newline PDFs)
    org_after_section = bool(re.search(
        r'(?:process|planning|guidance|services|management)[.\s]+(?:Response\s+)?(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute|Royal\s+College)',
        text, re.IGNORECASE
    ))
    # Pattern 3: Check if document has multiple distinct org names (suggests org-based structure)
    org_names = re.findall(
        r'\b(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|Royal\s+College\s+of\s+Psychiatrists)\b',
        text, re.IGNORECASE
    )
    has_multiple_orgs = len(set(org.lower() for org in org_names)) >= 2
    
    has_org_headers = org_on_line or org_after_section or has_multiple_orgs
    
    # Should NOT have HSIB rec IDs (otherwise use HSSIB org-structured or standard HSIB)
    has_hsib_rec_ids = bool(re.search(r'R/\d{4}/\d{3}', text))
    
    # Log all conditions for debugging
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
# Response extraction functions
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
    
    # Patterns that indicate who the recommendation is directed to
    # Format: "Org must/should/needs to..."
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


def extract_trust_responses(text: str) -> List[Dict]:
    """
    Extract responses from Trust response documents (Report 7 style).
    
    Format: "Recommendation N" followed by "[Trust] response:" then response text
    
    v2.7: Updated to handle PDFs with no line breaks
    """
    if not text:
        return []
    
    # Normalise line endings
    text = normalise_line_endings(text)
    
    responses = []
    
    # Find all "Recommendation N" headers - flexible (with or without newline)
    rec_pattern = re.compile(r'Recommendation\s+(\d+)\s*(?:\n|(?=[A-Z]))', re.IGNORECASE)
    rec_matches = list(rec_pattern.finditer(text))
    
    # Find all "[Trust] response:" markers
    trust_resp_pattern = re.compile(r'(?:TEWV|Trust)\s+response[:\s]*', re.IGNORECASE)
    trust_resp_matches = list(trust_resp_pattern.finditer(text))
    
    logger.info(f"Trust format: Found {len(rec_matches)} recommendations and {len(trust_resp_matches)} response markers")
    
    # For each recommendation, find its corresponding response
    for i, rec_match in enumerate(rec_matches):
        rec_num = rec_match.group(1)
        rec_start = rec_match.start()
        
        # Find the end boundary (next recommendation or end of text)
        if i + 1 < len(rec_matches):
            rec_end = rec_matches[i + 1].start()
        else:
            rec_end = len(text)
        
        # Find the Trust response marker within this recommendation's section
        for resp_match in trust_resp_matches:
            if rec_start < resp_match.start() < rec_end:
                # Response starts after the "TEWV response:" marker
                resp_start = resp_match.end()
                resp_text = text[resp_start:rec_end].strip()
                
                # Clean up the response text
                resp_text = re.sub(r'\s+', ' ', resp_text)
                
                if len(resp_text) > 30:
                    responses.append({
                        'text': resp_text[:2000],  # Limit length
                        'position': resp_start,
                        'response_type': 'trust_response',
                        'confidence': 0.95,
                        'rec_number': rec_num,
                        'rec_id': rec_num,
                        'source_org': 'tewv',
                    })
                break
    
    logger.info(f"Extracted {len(responses)} Trust responses")
    return responses


def extract_hssib_org_structured_responses(text: str) -> List[Dict]:
    """
    Extract responses from HSSIB org-structured response documents (Report 6 style).
    
    Format: [Org Header] -> "HSSIB recommends..." -> "Safety recommendation R/YYYY/NNN"
            -> "Response" -> [Response text] -> "Actions planned..." -> "Response received on..."
    
    Key difference from standard HSIB: responses are grouped by ORGANISATION,
    and we need to extract the rec_id from each section.
    
    v2.8: NEW - handles Report 6 format
    """
    if not text:
        return []
    
    # Normalise line endings
    text = normalise_line_endings(text)
    
    responses = []
    
    # Find organisation headers (including The Shelford Group)
    org_pattern = re.compile(
        r'(?:^|\n)(The\s+Shelford\s+Group|NHS\s+England|Department\s+of\s+Health\s+and\s+Social\s+Care|DHSC|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|NICE|Royal\s+College[^\n]*)\s*\n',
        re.IGNORECASE | re.MULTILINE
    )
    org_matches = list(org_pattern.finditer(text))
    
    logger.info(f"HSSIB org-structured: Found {len(org_matches)} org headers")
    
    for i, org_match in enumerate(org_matches):
        org_name = org_match.group(1).strip()
        org_start = org_match.end()
        
        # Find end of this org's section (next org header or end)
        if i + 1 < len(org_matches):
            section_end = org_matches[i + 1].start()
        else:
            section_end = len(text)
        
        section_text = text[org_start:section_end]
        
        # Find the rec_id from "Safety recommendation R/YYYY/NNN" in this section
        rec_id_match = re.search(r'Safety\s+recommendation\s+(R/\d{4}/\d{3})', section_text, re.IGNORECASE)
        if not rec_id_match:
            # Try alternate pattern
            rec_id_match = re.search(r'recommendation\s+(R/\d{4}/\d{3})', section_text, re.IGNORECASE)
        
        rec_id = rec_id_match.group(1) if rec_id_match else str(i + 1)
        
        # Find "Response" header in this section (standalone on its own line)
        resp_header_match = re.search(r'\nResponse\n', section_text, re.IGNORECASE)
        if not resp_header_match:
            # Try alternate: Response followed by text starting with org acceptance
            resp_header_match = re.search(r'\nResponse\s*(?=The|NHS|We|DHSC|NICE|Care)', section_text, re.IGNORECASE)
        
        if not resp_header_match:
            logger.warning(f"No Response header found for org section: {org_name}")
            continue
        
        resp_start = resp_header_match.end()
        
        # Find end boundary (Actions planned, Response received, or end of section)
        boundary_match = re.search(
            r'Actions\s+planned\s+to\s+deliver|Response\s+received\s+on',
            section_text[resp_start:],
            re.IGNORECASE
        )
        if boundary_match:
            resp_end = resp_start + boundary_match.start()
        else:
            resp_end = len(section_text)
        
        resp_text = section_text[resp_start:resp_end].strip()
        resp_text = re.sub(r'\s+', ' ', resp_text)
        
        if len(resp_text) < 30:
            logger.warning(f"Response too short for {org_name}: {len(resp_text)} chars")
            continue
        
        # Determine source org key
        source_org = extract_target_org_from_text(org_name)
        if 'shelford' in org_name.lower():
            source_org = 'shelford_group'
        elif 'department of health' in org_name.lower():
            source_org = 'dhsc'
        
        responses.append({
            'text': resp_text[:2000],
            'position': org_start + resp_start,
            'response_type': 'hssib_org_structured',
            'confidence': 0.95,
            'rec_number': rec_id,
            'rec_id': rec_id,
            'source_org': source_org,
            'org_name': org_name,
        })
        
        logger.info(f"Extracted response for {rec_id} from {org_name}")
    
    logger.info(f"Extracted {len(responses)} HSSIB org-structured responses")
    return responses


def extract_org_based_hsib_responses(text: str) -> List[Dict]:
    """
    Extract responses from org-based HSIB response documents (Report 4 style).
    
    Format: [Org Header] -> [HSIB recommends...] -> "Response" -> [Response text]
    
    v2.7: Updated to handle PDFs with no line breaks - uses "Response [Org]" pattern
    """
    if not text:
        return []
    
    # Normalise line endings
    text = normalise_line_endings(text)
    
    responses = []
    
    # Strategy: Find "HSIB recommends" blocks, then find "Response" after each
    # Each "HSIB recommends" is a recommendation, "Response" starts the response
    
    # Find all "HSIB recommends" positions (these mark recommendations)
    hsib_rec_pattern = re.compile(r'HSIB\s+recommends\s+that\s+(?:the\s+)?(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute[^.]*|Royal\s+College[^.]*|NICE|DHSC)', re.IGNORECASE)
    hsib_matches = list(hsib_rec_pattern.finditer(text))
    
    # Find all "Response" positions
    resp_pattern = re.compile(r'\bResponse\b', re.IGNORECASE)
    resp_matches = list(resp_pattern.finditer(text))
    
    logger.info(f"Org-based HSIB: Found {len(hsib_matches)} 'HSIB recommends' and {len(resp_matches)} 'Response' markers")
    
    # For each HSIB recommendation, find the Response that follows it
    for idx, hsib_match in enumerate(hsib_matches):
        org_name = hsib_match.group(1).strip()
        rec_start = hsib_match.start()
        
        # Find the end of this recommendation section (next HSIB recommends or end)
        if idx + 1 < len(hsib_matches):
            section_end = hsib_matches[idx + 1].start()
        else:
            section_end = len(text)
        
        # Find the "Response" within this section
        resp_start = None
        for resp_match in resp_matches:
            if rec_start < resp_match.start() < section_end:
                resp_start = resp_match.end()
                break
        
        if resp_start is None:
            logger.warning(f"No Response found for recommendation {idx + 1} ({org_name})")
            continue
        
        # Extract response text (from after "Response" to next HSIB recommends or boundary)
        resp_end = section_end
        
        # Check for boundary markers within the response
        boundary_patterns = [
            r'\bActions\s+planned\s+to\s+deliver',
            r'\bResponse\s+received\s+on',
            r'\bSafety\s+observation',
            r'\bSafety\s+action',
        ]
        for boundary in boundary_patterns:
            match = re.search(boundary, text[resp_start:resp_end], re.IGNORECASE)
            if match:
                resp_end = min(resp_end, resp_start + match.start())
        
        resp_text = text[resp_start:resp_end].strip()
        resp_text = re.sub(r'\s+', ' ', resp_text)
        
        if len(resp_text) < 30:
            continue
        
        # Recommendation number is 1-based index
        rec_num = str(idx + 1)
        source_org = extract_target_org_from_text(org_name)
        
        responses.append({
            'text': resp_text[:2000],
            'position': resp_start,
            'response_type': 'org_based_hsib',
            'confidence': 0.95,
            'rec_number': rec_num,
            'rec_id': rec_num,
            'source_org': source_org,
            'org_name': org_name,
        })
    
    logger.info(f"Extracted {len(responses)} org-based HSIB responses")
    return responses


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
    
    # Normalise line endings
    text = normalise_line_endings(text)
    
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
    Auto-detects document format and uses appropriate extraction method.
    
    Supported formats:
    - Standard government: "Government response to recommendation N"
    - HSSIB org-structured (Report 6 style): Org headers + rec IDs + "HSSIB recommends"
    - HSIB with rec IDs: Documents with R/YYYY/NNN or YYYY/NNN patterns
    - Org-based HSIB (Report 4 style): Org headers + "HSIB recommends" + "Response" (no rec IDs)
    - Trust response (Report 7 style): "Recommendation N" + "[Trust] response:"
    
    Returns a list of response dictionaries with:
    - text: The response text
    - position: Position in document
    - response_type: 'structured', 'hsib_structured', 'hssib_org_structured', 'org_based_hsib', 'trust_response', or 'sentence'
    - confidence: Extraction confidence score
    - rec_number: Associated recommendation number (if detected)
    """
    if not text:
        logger.warning("extract_response_sentences called with empty text")
        return []
    
    # =========================================================================
    # CRITICAL: Normalise line endings FIRST before any detection
    # =========================================================================
    text = normalise_line_endings(text)
    
    logger.info(f"extract_response_sentences: Processing {len(text)} chars")
    logger.info(f"First 200 chars: {text[:200]}")

    # Check for Trust response format first (Report 7 style)
    logger.info("Checking for Trust response format...")
    if is_trust_response_document(text):
        logger.info("✅ Detected Trust response document format")
        return extract_trust_responses(text)
    
    # v2.8: Check for HSSIB org-structured format (Report 6 style) BEFORE standard HSIB
    # This format has BOTH org headers AND rec IDs
    logger.info("Checking for HSSIB org-structured format...")
    if is_hssib_org_structured_response(text):
        logger.info("✅ Detected HSSIB org-structured response format")
        return extract_hssib_org_structured_responses(text)
    
    # Check for org-based HSIB format (Report 4 style) - no rec IDs
    logger.info("Checking for org-based HSIB format...")
    if is_org_based_hsib_response(text):
        logger.info("✅ Detected org-based HSIB response format")
        return extract_org_based_hsib_responses(text)

    # Check for standard HSIB format (with rec IDs, not org-grouped)
    logger.info("Checking for standard HSIB format...")
    if is_hsib_response_document(text):
        logger.info("✅ Detected HSIB response document format")
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
    "normalise_line_endings",
    "is_recommendation_text",
    "is_genuine_response",
    "has_pdf_artifacts",
    "clean_pdf_artifacts",
    "is_hsib_response_document",
    "is_trust_response_document",
    "is_hssib_org_structured_response",
    "is_org_based_hsib_response",
    "extract_target_org_from_text",
    "extract_recommendation_target_org",
    "get_response_document_org",
    "extract_hsib_responses",
    "extract_trust_responses",
    "extract_hssib_org_structured_responses",
    "extract_org_based_hsib_responses",
    "extract_response_sentences",
]
