"""
Response extraction from government and HSIB/HSSIB response documents.

This module handles extraction of responses from various document formats.
Format detection logic is in format_detection.py.

v4.0 Changes:
- FIXED: Decimal recommendation IDs (8.1, 8.2, 8.3-8.4) now properly extracted
- FIXED: "Government response to recommendation 8.1" patterns now matched
- ADDED: Parliamentary/Select Committee response format support
- IMPROVED: Better boundary detection for multi-format documents

Supported formats:
- Standard government: "Government response to recommendation N"
- Decimal government: "Government response to recommendation 8.1"
- HSSIB org-structured (Report 6 style): Org headers + rec IDs + "HSSIB recommends"
- HSIB with rec IDs: Documents with R/YYYY/NNN or YYYY/NNN patterns
- Org-based HSIB (Report 4 style): Org headers + "HSIB recommends" + "Response" (no rec IDs)
- Trust response (Report 7 style): "Recommendation N" + "[Trust] response:"
"""

import logging
import re
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Line ending normalisation
# --------------------------------------------------------------------------- #

def normalise_line_endings(text: str) -> str:
    """
    Normalise all line endings to \\n for consistent regex matching.
    """
    if not text:
        return text
    return text.replace('\r\n', '\n').replace('\r', '\n')


# --------------------------------------------------------------------------- #
# Excluded recommendations detection
# --------------------------------------------------------------------------- #

def detect_excluded_recommendations(text: str) -> List[int]:
    """
    Detect recommendations that are explicitly stated as not included.
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    excluded = []
    
    patterns = [
        r'[Rr]ecommendations?\s+([\d,\s]+(?:and\s+\d+)?)\s+relate\s+to\s+other\s+organi[sz]ations',
        r'[Rr]ecommendations?\s+([\d,\s]+(?:and\s+\d+)?)\s+(?:are|is)\s+(?:therefore\s+)?not\s+included',
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
        r"^recommendation\s+\d+(?:\.\d+)?\s+[a-z]",
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
        # v4.0: Parliamentary recommendation patterns
        r"^the\s+idea\s+of\s+the\s+government",
        r"^ministers?\s+(?:are|should)",
        r"^we\s+(?:urge|call\s+on|ask)",
    ]
    return any(re.search(pattern, text_lower) for pattern in recommendation_starters)


def is_genuine_response(text: str) -> bool:
    """Check if text appears to be a genuine government/organisation response."""
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
        r"^the\s+government\s+does\s+not",
        r"^the\s+government\s+has",
        r"^we\s+support",
        r"^we\s+accept",
        r"^we\s+agree",
        r"^we\s+welcome",
        r"^we\s+are\s+happy\s+to\s+confirm",
        r"^we\s+note",
        r"^we\s+recognise",
        r"^dhsc\s+and\s+nhs\s+england\s+support",
        r"^nhs\s+england\s+(?:has|will|is)",
        r"^care\s+quality\s+commission\s+(?:is|has|will)",
        r"^the\s+national\s+institute\s+for\s+health",
        r"^nice\s+(?:is|has|will)",
        # v4.0: Additional response indicators
        r"^(?:the\s+)?government\s+(?:agrees?|accepts?|supports?|notes?|rejects?|welcomes?)",
        r"^this\s+recommendation\s+(?:is|has\s+been)",
        r"^(?:we|the\s+government)\s+will\s+give\s+(?:careful\s+)?consideration",
        r"^(?:we|the\s+government)\s+(?:have|has)\s+(?:already|now)",
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
    """Detect if a document is an HSIB/HSSIB-style response document."""
    if not text:
        return False
    
    text = normalise_line_endings(text)
    
    has_hsib_rec = bool(re.search(r'[Rr]ecommendation\s*[R/]?\s*\d{4}\s*/\s*\d{3}', text))
    has_response_header = bool(re.search(r'\bResponse\b', text, re.IGNORECASE))
    has_hsib_mention = bool(re.search(r'\bH[SS]I?B\b', text, re.IGNORECASE))
    
    if has_hsib_rec:
        logger.info("✅ HSIB format detected: Found HSIB-style recommendation IDs")
        return True
    
    if has_hsib_mention and has_response_header:
        logger.info("✅ HSIB format detected: Found HSIB mentions + Response headers")
        return True
    
    return False


def is_trust_response_document(text: str) -> bool:
    """Detect if a document is a Trust response document (Report 7 style)."""
    if not text:
        return False
    
    text = normalise_line_endings(text)
    
    has_trust_response = bool(re.search(r'(?:TEWV|Trust)\s+response[:\s]', text, re.IGNORECASE))
    has_recommendation_headers = bool(re.search(r'Recommendation\s+\d+', text, re.IGNORECASE))
    
    if has_trust_response and has_recommendation_headers:
        logger.info("✅ Trust response format detected")
        return True
    
    return False


def is_hssib_org_structured_response(text: str) -> bool:
    """Detect if a document is an HSSIB org-structured response (Report 6 style)."""
    if not text:
        return False
    
    text = normalise_line_endings(text)
    
    has_hssib_rec = bool(re.search(r'HSSIB\s+recommends', text, re.IGNORECASE))
    has_rec_ids = bool(re.search(r'R/\d{4}/\d{3}', text))
    has_response_headers = bool(re.search(r'(?:^|\n)Response\n', text, re.MULTILINE))
    
    org_pattern = r'(?:^|\n)(The\s+Shelford\s+Group|NHS\s+England|Department\s+of\s+Health\s+and\s+Social\s+Care|DHSC|Care\s+Quality\s+Commission|NICE)\s*\n'
    has_org_headers = bool(re.search(org_pattern, text, re.IGNORECASE | re.MULTILINE))
    
    has_hssib_markers = bool(re.search(r'Actions\s+planned\s+to\s+deliver|Response\s+received\s+on', text, re.IGNORECASE))
    
    if has_hssib_rec and has_rec_ids and has_response_headers and has_org_headers and has_hssib_markers:
        logger.info("✅ HSSIB org-structured response format detected (Report 6 style)")
        return True
    
    return False


def is_org_based_hsib_response(text: str) -> bool:
    """Detect if a document is an org-based HSIB response (Report 4 style)."""
    if not text:
        return False
    
    text = normalise_line_endings(text)
    
    has_hsib_recommends = bool(re.search(r'HSIB\s+recommends', text, re.IGNORECASE))
    
    response_followed_by_content = bool(re.search(
        r'\bResponse\b\s*(?:NHS\s+England|Care\s+Quality\s+Commission|The\s+National|We\s+(?:are|have|welcome)|NICE\s+is)',
        text, re.IGNORECASE
    ))
    
    has_response_on_line = bool(re.search(r'(?:^|\n)\s*Response\s*(?:\n|$)', text, re.MULTILINE))
    has_response_headers = response_followed_by_content or has_response_on_line
    
    org_on_line = bool(re.search(
        r'(?:^|\n)\s*(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute|Royal\s+College|NICE|DHSC|The\s+Shelford\s+Group)\s*(?:\n|$)',
        text, re.IGNORECASE | re.MULTILINE
    ))
    
    org_names = re.findall(
        r'\b(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|Royal\s+College\s+of\s+Psychiatrists)\b',
        text, re.IGNORECASE
    )
    has_multiple_orgs = len(set(org.lower() for org in org_names)) >= 2
    
    has_org_headers = org_on_line or has_multiple_orgs
    has_hsib_rec_ids = bool(re.search(r'R/\d{4}/\d{3}', text))
    
    if has_hsib_recommends and has_response_headers and has_org_headers and not has_hsib_rec_ids:
        logger.info("✅ Org-based HSIB response format detected (Report 4 style)")
        return True
    
    return False


def is_decimal_government_response(text: str) -> bool:
    """
    v4.0: Detect if a document uses decimal recommendation IDs (8.1, 8.2, 8.3-8.4).
    Common in House of Lords/Commons Select Committee reports.
    """
    if not text:
        return False
    
    text = normalise_line_endings(text)
    
    # Check for "Government response to recommendation X.Y" pattern
    has_decimal_response_headers = bool(re.search(
        r'Government\s+response\s+to\s+recommendation\s+\d+\.\d+',
        text, re.IGNORECASE
    ))
    
    # Check for decimal recommendation pattern
    has_decimal_recommendations = bool(re.search(
        r'Recommendation\s+\d+\.\d+',
        text, re.IGNORECASE
    ))
    
    # Check for range patterns like "8.3-8.4"
    has_range_patterns = bool(re.search(
        r'\d+\.\d+-\d+\.\d+',
        text
    ))
    
    if has_decimal_response_headers:
        logger.info("✅ Decimal government response format detected")
        return True
    
    if has_decimal_recommendations and (has_range_patterns or has_decimal_response_headers):
        logger.info("✅ Decimal recommendation format detected")
        return True
    
    return False


# --------------------------------------------------------------------------- #
# Organisation extraction helpers
# --------------------------------------------------------------------------- #

def extract_target_org_from_text(text: str) -> Optional[str]:
    """Extract target organisation from recommendation or response text."""
    if not text:
        return None
    
    text_lower = text.lower()
    
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
    if 'dwp' in text_lower or 'department for work' in text_lower:
        return 'dwp'
    if 'cabinet office' in text_lower:
        return 'cabinet_office'
    
    return None


# --------------------------------------------------------------------------- #
# Response extraction functions
# --------------------------------------------------------------------------- #

def extract_decimal_government_responses(text: str) -> List[Dict]:
    """
    v4.0: Extract responses from documents with decimal recommendation IDs.
    
    Format: "Government response to recommendation 8.1" followed by response text
    Also handles range formats like "8.3-8.4"
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    text = clean_pdf_artifacts(text)
    
    responses = []
    
    # v4.0: Pattern for decimal recommendation IDs including ranges
    # Matches: "Government response to recommendation 8.1", "8.3-8.4", "10.1"
    gov_resp_pattern = re.compile(
        r'Government\s+response\s+to\s+recommendation[s]?\s+(\d+\.\d+(?:-\d+(?:\.\d+)?)?)',
        re.IGNORECASE
    )
    
    gov_resp_matches = list(gov_resp_pattern.finditer(text))
    
    logger.info(f"Found {len(gov_resp_matches)} decimal government response headers")
    
    # Also find recommendation headers to detect boundaries
    rec_pattern = re.compile(
        r'(?:Recommendation\s+)?(\d+\.\d+(?:-\d+(?:\.\d+)?)?)[:\s]+(?=[A-Z])',
        re.IGNORECASE
    )
    rec_matches = list(rec_pattern.finditer(text))
    
    for i, match in enumerate(gov_resp_matches):
        rec_id = match.group(1)  # e.g., "8.1" or "8.3-8.4"
        start_pos = match.end()
        
        # Find end boundary
        end_pos = len(text)
        
        # Check for next government response header
        if i + 1 < len(gov_resp_matches):
            end_pos = min(end_pos, gov_resp_matches[i + 1].start())
        
        # Check for next recommendation header (indicates new section)
        for rec_match in rec_matches:
            if rec_match.start() > start_pos and rec_match.start() < end_pos:
                # Only use as boundary if it looks like a new recommendation, not part of response
                preceding_text = text[max(0, rec_match.start() - 50):rec_match.start()].lower()
                if 'recommendation' in preceding_text or rec_match.start() - start_pos > 100:
                    end_pos = rec_match.start()
                    break
        
        resp_content = text[start_pos:end_pos].strip()
        
        if not resp_content or len(resp_content) < 30:
            logger.warning(f"Response too short for recommendation {rec_id}")
            continue
        
        # Clean up the response
        resp_content = clean_pdf_artifacts(resp_content)
        resp_content = re.sub(r'\s+', ' ', resp_content).strip()
        
        responses.append({
            'text': resp_content,
            'position': start_pos,
            'response_type': 'decimal_structured',
            'confidence': 0.95,
            'rec_number': rec_id,
            'rec_id': rec_id,
            'source_org': extract_target_org_from_text(resp_content),
        })
        
        logger.info(f"Extracted response for recommendation {rec_id}: {len(resp_content)} chars")
    
    logger.info(f"Extracted {len(responses)} decimal government responses")
    return responses


def extract_trust_responses(text: str) -> List[Dict]:
    """Extract responses from Trust response documents (Report 7 style)."""
    if not text:
        return []
    
    text = normalise_line_endings(text)
    responses = []
    
    excluded_recs = detect_excluded_recommendations(text)
    if excluded_recs:
        logger.info(f"Detected excluded recommendations (other orgs): {excluded_recs}")
    
    rec_pattern = re.compile(
        r'Recommendation\s+(\d+)\s*(\d)?\s*\*?\s*(?:\n|(?=[A-Za-z]))',
        re.IGNORECASE
    )
    rec_matches = list(rec_pattern.finditer(text))
    
    trust_resp_pattern = re.compile(r'(?:TEWV|Trust)\s+response[:\s]*', re.IGNORECASE)
    
    logger.info(f"Trust format: Found {len(rec_matches)} recommendations")
    
    rec_sections = {}
    
    for i, rec_match in enumerate(rec_matches):
        rec_num = rec_match.group(1)
        if rec_match.group(2):
            rec_num = rec_num + rec_match.group(2)
        
        rec_num_int = int(rec_num)
        rec_start = rec_match.start()
        
        if i + 1 < len(rec_matches):
            rec_end = rec_matches[i + 1].start()
        else:
            rec_end = len(text)
        
        section_text = text[rec_start:rec_end]
        has_response_marker = bool(trust_resp_pattern.search(section_text))
        
        if rec_num_int not in rec_sections:
            rec_sections[rec_num_int] = {
                'start': rec_start,
                'end': rec_end,
                'has_response': has_response_marker,
                'section_text': section_text
            }
        elif has_response_marker and not rec_sections[rec_num_int]['has_response']:
            rec_sections[rec_num_int] = {
                'start': rec_start,
                'end': rec_end,
                'has_response': has_response_marker,
                'section_text': section_text
            }
    
    # Add N/A responses for excluded recs
    for rec_num in excluded_recs:
        responses.append({
            'text': f"N/A - Recommendation {rec_num} is addressed to another organisation",
            'position': 0,
            'response_type': 'trust_excluded',
            'confidence': 1.0,
            'rec_number': str(rec_num),
            'rec_id': str(rec_num),
            'source_org': None,
            'excluded': True,
        })
    
    for rec_num, section_info in sorted(rec_sections.items()):
        if rec_num in excluded_recs:
            continue
        
        section_text = section_info['section_text']
        
        resp_marker = trust_resp_pattern.search(section_text)
        if not resp_marker:
            continue
        
        resp_start_local = resp_marker.end()
        resp_text = section_text[resp_start_local:].strip()
        resp_text = re.sub(r'\s+', ' ', resp_text)
        
        if len(resp_text) < 30:
            continue
        
        responses.append({
            'text': resp_text[:2000],
            'position': section_info['start'] + resp_start_local,
            'response_type': 'trust_response',
            'confidence': 0.95,
            'rec_number': str(rec_num),
            'rec_id': str(rec_num),
            'source_org': 'tewv',
            'excluded': False,
        })
    
    logger.info(f"Extracted {len(responses)} Trust responses")
    return responses


def extract_hssib_org_structured_responses(text: str) -> List[Dict]:
    """Extract responses from HSSIB org-structured response documents (Report 6 style)."""
    if not text:
        return []
    
    text = normalise_line_endings(text)
    
    # Exclude Safety observations/actions sections
    safety_obs_match = re.search(r'\nSafety\s+observations?\s*\n', text, re.IGNORECASE)
    safety_act_match = re.search(r'\nSafety\s+actions?\s*\n', text, re.IGNORECASE)
    
    doc_end = len(text)
    if safety_obs_match:
        doc_end = min(doc_end, safety_obs_match.start())
    if safety_act_match:
        doc_end = min(doc_end, safety_act_match.start())
    
    if doc_end < len(text):
        text = text[:doc_end]
    
    responses = []
    
    org_pattern = re.compile(
        r'(?:^|\n)(The\s+Shelford\s+Group|NHS\s+England|Department\s+of\s+Health\s+and\s+Social\s+Care|DHSC|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|NICE|Royal\s+College[^\n]*)\s*\n',
        re.IGNORECASE | re.MULTILINE
    )
    org_matches = list(org_pattern.finditer(text))
    
    for i, org_match in enumerate(org_matches):
        org_name = org_match.group(1).strip()
        org_start = org_match.end()
        
        if i + 1 < len(org_matches):
            section_end = org_matches[i + 1].start()
        else:
            section_end = len(text)
        
        section_text = text[org_start:section_end]
        
        rec_id_match = re.search(r'Safety\s+recommendation\s+(R/\d{4}/\d{3})', section_text, re.IGNORECASE)
        if not rec_id_match:
            rec_id_match = re.search(r'recommendation\s+(R/\d{4}/\d{3})', section_text, re.IGNORECASE)
        
        rec_id = rec_id_match.group(1) if rec_id_match else str(i + 1)
        
        resp_header_match = re.search(r'\nResponse\n', section_text, re.IGNORECASE)
        if not resp_header_match:
            resp_header_match = re.search(r'\nResponse\s*(?=The|NHS|We|DHSC|NICE|Care)', section_text, re.IGNORECASE)
        
        if not resp_header_match:
            continue
        
        resp_start = resp_header_match.end()
        
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
            continue
        
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
    
    logger.info(f"Extracted {len(responses)} HSSIB org-structured responses")
    return responses


def extract_org_based_hsib_responses(text: str) -> List[Dict]:
    """Extract responses from org-based HSIB response documents (Report 4 style)."""
    if not text:
        return []
    
    text = normalise_line_endings(text)
    responses = []
    
    hsib_rec_pattern = re.compile(
        r'HSIB\s+recommends\s+that\s+(?:the\s+)?(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute[^.]*|Royal\s+College[^.]*|NICE|DHSC)',
        re.IGNORECASE
    )
    hsib_matches = list(hsib_rec_pattern.finditer(text))
    
    resp_pattern = re.compile(r'\bResponse\b', re.IGNORECASE)
    resp_matches = list(resp_pattern.finditer(text))
    
    for idx, hsib_match in enumerate(hsib_matches):
        org_name = hsib_match.group(1).strip()
        rec_start = hsib_match.start()
        
        if idx + 1 < len(hsib_matches):
            section_end = hsib_matches[idx + 1].start()
        else:
            section_end = len(text)
        
        resp_start = None
        for resp_match in resp_matches:
            if rec_start < resp_match.start() < section_end:
                resp_start = resp_match.end()
                break
        
        if resp_start is None:
            continue
        
        resp_end = section_end
        
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
    """Extract responses from HSIB-format response documents."""
    if not text:
        return []
    
    text = normalise_line_endings(text)
    
    # Exclude Safety observations/actions
    safety_obs_match = re.search(r'\nSafety\s+observations?\s*\n', text, re.IGNORECASE)
    safety_act_match = re.search(r'\nSafety\s+actions?\s*\n', text, re.IGNORECASE)
    
    doc_end = len(text)
    if safety_obs_match:
        doc_end = min(doc_end, safety_obs_match.start())
    if safety_act_match:
        doc_end = min(doc_end, safety_act_match.start())
    
    if doc_end < len(text):
        text = text[:doc_end]
    
    responses = []
    
    hsib_rec_patterns = [
        re.compile(r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]', re.IGNORECASE),
        re.compile(r'Recommendation\s+(\d{4}/\d{3})[:\s]', re.IGNORECASE),
    ]
    
    rec_matches = []
    for pattern in hsib_rec_patterns:
        rec_matches.extend(list(pattern.finditer(text)))
    rec_matches.sort(key=lambda m: m.start())
    
    response_patterns = [
        re.compile(r'(?:^|\n)\s*Response\s*\n', re.IGNORECASE | re.MULTILINE),
        re.compile(r'Response(?=[A-Z][a-z])', re.IGNORECASE),
    ]
    
    response_matches = []
    for pattern in response_patterns:
        response_matches.extend(list(pattern.finditer(text)))
    
    # Deduplicate
    unique_matches = []
    last_pos = -100
    for m in sorted(response_matches, key=lambda x: x.start()):
        if m.start() - last_pos > 20:
            unique_matches.append(m)
            last_pos = m.start()
    response_matches = unique_matches
    
    org_header_pattern = re.compile(
        r'(?:^|\n)\s*(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|National\s+Institute|Royal\s+College\s+of\s+Psychiatrists|Royal\s+College|The\s+Shelford\s+Group|DHSC)\s*\n',
        re.IGNORECASE | re.MULTILINE
    )
    org_headers = list(org_header_pattern.finditer(text))
    
    boundary_markers = [
        r'\n\s*Actions\s+planned\s+to\s+deliver',
        r'\n\s*Safety\s+(?:observation|action)',
        r'\n\s*Response\s+received\s+on',
    ]
    
    for idx, resp_match in enumerate(response_matches):
        start = resp_match.end()
        end = len(text)
        
        if idx + 1 < len(response_matches):
            end = min(end, response_matches[idx + 1].start())
        
        for org_match in org_headers:
            if org_match.start() > start and org_match.start() < end:
                end = org_match.start()
                break
        
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
        
        rec_id = None
        for rec_match in reversed(rec_matches):
            if rec_match.start() < resp_match.start():
                rec_id = rec_match.group(1)
                break
        
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


def extract_government_responses(text: str) -> List[Dict]:
    """
    Extract responses from standard government response documents.
    
    Format: "Government response to recommendation N" followed by response text.
    v4.0: Now also handles decimal IDs via extract_decimal_government_responses().
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    
    # v4.0: Check if this is a decimal format document first
    if is_decimal_government_response(text):
        logger.info("Detected decimal government response format, using decimal extractor")
        return extract_decimal_government_responses(text)
    
    responses = []
    
    # Standard pattern: "Government response to recommendation N"
    gov_resp_pattern = re.compile(
        r'Government\s+response\s+to\s+recommendation[s]?\s+(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)',
        re.IGNORECASE
    )
    
    gov_resp_matches = list(gov_resp_pattern.finditer(text))
    
    logger.info(f"Found {len(gov_resp_matches)} government response headers")
    
    for i, match in enumerate(gov_resp_matches):
        rec_id = match.group(1)
        start_pos = match.end()
        
        # Find end boundary
        if i + 1 < len(gov_resp_matches):
            end_pos = gov_resp_matches[i + 1].start()
        else:
            end_pos = len(text)
        
        # Also check for next "Recommendation N" header
        next_rec_match = re.search(
            r'\n\s*Recommendation\s+\d+(?:\.\d+)?[:\s]',
            text[start_pos:end_pos],
            re.IGNORECASE
        )
        if next_rec_match:
            end_pos = start_pos + next_rec_match.start()
        
        resp_content = text[start_pos:end_pos].strip()
        
        if not resp_content or len(resp_content) < 30:
            continue
        
        # Clean up
        resp_content = clean_pdf_artifacts(resp_content)
        resp_content = re.sub(r'\s+', ' ', resp_content).strip()
        
        responses.append({
            'text': resp_content,
            'position': start_pos,
            'response_type': 'government_structured',
            'confidence': 0.95,
            'rec_number': rec_id,
            'rec_id': rec_id,
            'source_org': extract_target_org_from_text(resp_content),
        })
    
    logger.info(f"Extracted {len(responses)} government responses")
    return responses


def extract_responses(text: str) -> List[Dict]:
    """
    Main entry point for response extraction.
    Automatically detects document format and uses appropriate extractor.
    
    v4.0: Added decimal format detection and extraction.
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    
    # Try format detection in order of specificity
    
    # 1. v4.0: Check for decimal government response format first
    if is_decimal_government_response(text):
        logger.info("Using decimal government response extractor")
        return extract_decimal_government_responses(text)
    
    # 2. HSSIB org-structured (Report 6 style)
    if is_hssib_org_structured_response(text):
        logger.info("Using HSSIB org-structured extractor")
        return extract_hssib_org_structured_responses(text)
    
    # 3. Trust response (Report 7 style)
    if is_trust_response_document(text):
        logger.info("Using Trust response extractor")
        return extract_trust_responses(text)
    
    # 4. Org-based HSIB (Report 4 style)
    if is_org_based_hsib_response(text):
        logger.info("Using org-based HSIB extractor")
        return extract_org_based_hsib_responses(text)
    
    # 5. HSIB with rec IDs
    if is_hsib_response_document(text):
        logger.info("Using HSIB response extractor")
        return extract_hsib_responses(text)
    
    # 6. Standard government response (fallback)
    logger.info("Using standard government response extractor")
    return extract_government_responses(text)


# --------------------------------------------------------------------------- #
# Convenience function for external use
# --------------------------------------------------------------------------- #

def get_responses_for_document(text: str) -> Tuple[List[Dict], str]:
    """
    Extract responses and return detected format type.
    
    Returns:
        Tuple of (responses list, format type string)
    """
    if not text:
        return [], 'unknown'
    
    text = normalise_line_endings(text)
    
    if is_decimal_government_response(text):
        return extract_decimal_government_responses(text), 'decimal_government'
    
    if is_hssib_org_structured_response(text):
        return extract_hssib_org_structured_responses(text), 'hssib_org_structured'
    
    if is_trust_response_document(text):
        return extract_trust_responses(text), 'trust_response'
    
    if is_org_based_hsib_response(text):
        return extract_org_based_hsib_responses(text), 'org_based_hsib'
    
    if is_hsib_response_document(text):
        return extract_hsib_responses(text), 'hsib_response'
    
    return extract_government_responses(text), 'government_response'
