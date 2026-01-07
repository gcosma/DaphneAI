"""
Response extraction from government and HSIB/HSSIB response documents.

This module handles extraction of responses from various document formats.
Format detection logic is in format_detection.py.

Supported formats:
- Standard government: "Government response to recommendation N"
- HSSIB org-structured (Report 6 style): Org headers + rec IDs + "HSSIB recommends"
- HSIB with rec IDs: Documents with R/YYYY/NNN or YYYY/NNN patterns
- Org-based HSIB (Report 4 style): Org headers + "HSIB recommends" + "Response" (no rec IDs)
- Trust response (Report 7 style): "Recommendation N" + "[Trust] response:"

v3.1 Changes:
- FIXED: extract_trust_responses() now creates placeholder responses for excluded recs even when no section exists
- FIXED: Excluded recs (4,5,10,12) now properly marked as N/A in output
- SPLIT: Detection functions moved to format_detection.py

v3.0 Changes:
- FIXED: extract_trust_responses() handles asterisks in headers (Report 7 Rec 7)
- FIXED: extract_trust_responses() handles duplicate rec headers - prefers one with "TEWV response:"
- FIXED: extract_trust_responses() handles spaced digits ("Recommendation 1 1" -> "11")

v2.9 Changes:
- FIXED: Excludes "Safety observations" and "Safety actions" sections from extraction

v2.8 Changes:
- ADDED: extract_hssib_org_structured_responses() function for Report 6 style

v2.5 Changes:
- FIXED: HSIB response extraction now segments by recommendation ID
- FIXED: Response boundaries properly detect org headers and rec IDs
"""

import logging
import re
from typing import Dict, List, Tuple

# Import detection functions from format_detection module
from .format_detection import (
    normalise_line_endings,
    detect_excluded_recommendations,
    is_recommendation_text,
    is_genuine_response,
    clean_pdf_artifacts,
    is_hsib_response_document,
    is_trust_response_document,
    is_hssib_org_structured_response,
    is_org_based_hsib_response,
    extract_target_org_from_text,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Trust response extraction (Report 7 style)
# --------------------------------------------------------------------------- #

def extract_trust_responses(text: str) -> List[Dict]:
    """
    Extract responses from Trust response documents (Report 7 style).
    
    Format: "Recommendation N" followed by "[Trust] response:" then response text
    
    v3.1 Changes:
    - FIXED: Now creates N/A responses for excluded recs even when no section exists in document
    - FIXED: Pattern for finding rec headers is more flexible
    
    v3.0 Changes:
    - Handle asterisks in recommendation headers (e.g., "Recommendation 7 *")
    - Handle duplicate recommendation headers - prefer ones with "TEWV response:"
    - Handle spaced digits ("Recommendation 1 1" -> "11")
    - Detect excluded recommendations (addressed to other orgs)
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    responses = []
    
    # Detect excluded recommendations
    excluded_recs = detect_excluded_recommendations(text)
    if excluded_recs:
        logger.info(f"Detected excluded recommendations (other orgs): {excluded_recs}")
    
    # Pattern to handle asterisks and spaced digits
    # Matches: "Recommendation 1", "Recommendation 7 *", "Recommendation 1 1", "Recommendation 9 is..."
    rec_pattern = re.compile(
        r'Recommendation\s+(\d+)\s*(\d)?\s*\*?\s*(?:\n|(?=[A-Za-z]))',
        re.IGNORECASE
    )
    rec_matches = list(rec_pattern.finditer(text))
    
    # Find all "[Trust] response:" markers
    trust_resp_pattern = re.compile(r'(?:TEWV|Trust)\s+response[:\s]*', re.IGNORECASE)
    trust_resp_matches = list(trust_resp_pattern.finditer(text))
    
    logger.info(f"Trust format: Found {len(rec_matches)} recommendations and {len(trust_resp_matches)} response markers")
    
    # Build a map of rec_num -> best section (prefer one with TEWV response:)
    rec_sections = {}
    
    for i, rec_match in enumerate(rec_matches):
        # Handle spaced digits: "Recommendation 1 1" -> "11"
        rec_num = rec_match.group(1)
        if rec_match.group(2):
            rec_num = rec_num + rec_match.group(2)
        
        rec_num_int = int(rec_num)
        rec_start = rec_match.start()
        
        # Find end boundary (next recommendation or end)
        if i + 1 < len(rec_matches):
            rec_end = rec_matches[i + 1].start()
        else:
            rec_end = len(text)
        
        section_text = text[rec_start:rec_end]
        
        # Check if this section has a "TEWV response:" marker
        has_response_marker = bool(trust_resp_pattern.search(section_text))
        
        # Only keep the section that has the response marker (for duplicates)
        if rec_num_int not in rec_sections:
            rec_sections[rec_num_int] = {
                'start': rec_start,
                'end': rec_end,
                'has_response': has_response_marker,
                'section_text': section_text
            }
        elif has_response_marker and not rec_sections[rec_num_int]['has_response']:
            logger.info(f"Rec {rec_num_int}: Found better section with TEWV response marker")
            rec_sections[rec_num_int] = {
                'start': rec_start,
                'end': rec_end,
                'has_response': has_response_marker,
                'section_text': section_text
            }
    
    logger.info(f"Trust format: {len(rec_sections)} unique recommendation sections")
    
    # v3.1 FIX: First, add N/A responses for ALL excluded recs (even if no section exists)
    for rec_num in excluded_recs:
        responses.append({
            'text': f"N/A - Recommendation {rec_num} is addressed to another organisation and not included in this response document",
            'position': 0,
            'response_type': 'trust_excluded',
            'confidence': 1.0,
            'rec_number': str(rec_num),
            'rec_id': str(rec_num),
            'source_org': None,
            'excluded': True,
        })
        logger.info(f"Rec {rec_num}: Marked as excluded (other organisation)")
    
    # Extract responses from each section (skip excluded ones since we already added them)
    for rec_num, section_info in sorted(rec_sections.items()):
        # Skip excluded recommendations - already handled above
        if rec_num in excluded_recs:
            continue
        
        section_text = section_info['section_text']
        
        # Find "TEWV response:" marker in this section
        resp_marker = trust_resp_pattern.search(section_text)
        if not resp_marker:
            logger.warning(f"Rec {rec_num}: No 'TEWV response:' marker found in section")
            continue
        
        # Response starts after the marker
        resp_start_local = resp_marker.end()
        resp_text = section_text[resp_start_local:].strip()
        
        # Clean up whitespace
        resp_text = re.sub(r'\s+', ' ', resp_text)
        
        if len(resp_text) < 30:
            logger.warning(f"Rec {rec_num}: Response too short ({len(resp_text)} chars)")
            continue
        
        responses.append({
            'text': resp_text[:2000],  # Limit length
            'position': section_info['start'] + resp_start_local,
            'response_type': 'trust_response',
            'confidence': 0.95,
            'rec_number': str(rec_num),
            'rec_id': str(rec_num),
            'source_org': 'tewv',
            'excluded': False,
        })
        logger.info(f"Rec {rec_num}: Extracted {len(resp_text)} chars")
    
    logger.info(f"Extracted {len(responses)} Trust responses")
    return responses


# --------------------------------------------------------------------------- #
# HSSIB org-structured extraction (Report 6 style)
# --------------------------------------------------------------------------- #

def extract_hssib_org_structured_responses(text: str) -> List[Dict]:
    """
    Extract responses from HSSIB org-structured response documents (Report 6 style).
    
    Format: [Org Header] -> "HSSIB recommends..." -> "Safety recommendation R/YYYY/NNN"
            -> "Response" -> [Response text] -> "Actions planned..." -> "Response received on..."
    
    Key difference from standard HSIB: responses are grouped by ORGANISATION.
    
    v2.9: FIXED - excludes Safety observations and Safety actions sections
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    
    # Find and exclude "Safety observations" and "Safety actions" sections
    safety_obs_match = re.search(r'\nSafety\s+observations?\s*\n', text, re.IGNORECASE)
    safety_act_match = re.search(r'\nSafety\s+actions?\s*\n', text, re.IGNORECASE)
    
    doc_end = len(text)
    if safety_obs_match:
        doc_end = min(doc_end, safety_obs_match.start())
        logger.info(f"Found 'Safety observations' at {safety_obs_match.start()}, truncating")
    if safety_act_match:
        doc_end = min(doc_end, safety_act_match.start())
        logger.info(f"Found 'Safety actions' at {safety_act_match.start()}, truncating")
    
    if doc_end < len(text):
        logger.info(f"Truncating document from {len(text)} to {doc_end} chars to exclude Safety observations/actions")
        text = text[:doc_end]
    
    responses = []
    
    # Find organisation headers
    org_pattern = re.compile(
        r'(?:^|\n)(The\s+Shelford\s+Group|NHS\s+England|Department\s+of\s+Health\s+and\s+Social\s+Care|DHSC|Care\s+Quality\s+Commission|National\s+Institute\s+for\s+Health\s+and\s+Care\s+Excellence|NICE|Royal\s+College[^\n]*)\s*\n',
        re.IGNORECASE | re.MULTILINE
    )
    org_matches = list(org_pattern.finditer(text))
    
    logger.info(f"HSSIB org-structured: Found {len(org_matches)} org headers")
    
    for i, org_match in enumerate(org_matches):
        org_name = org_match.group(1).strip()
        org_start = org_match.end()
        
        # Find end of this org's section
        if i + 1 < len(org_matches):
            section_end = org_matches[i + 1].start()
        else:
            section_end = len(text)
        
        section_text = text[org_start:section_end]
        
        # Find the rec_id from "Safety recommendation R/YYYY/NNN"
        rec_id_match = re.search(r'Safety\s+recommendation\s+(R/\d{4}/\d{3})', section_text, re.IGNORECASE)
        if not rec_id_match:
            rec_id_match = re.search(r'recommendation\s+(R/\d{4}/\d{3})', section_text, re.IGNORECASE)
        
        rec_id = rec_id_match.group(1) if rec_id_match else str(i + 1)
        
        # Find "Response" header in this section
        resp_header_match = re.search(r'\nResponse\n', section_text, re.IGNORECASE)
        if not resp_header_match:
            resp_header_match = re.search(r'\nResponse\s*(?=The|NHS|We|DHSC|NICE|Care)', section_text, re.IGNORECASE)
        
        if not resp_header_match:
            logger.warning(f"No Response header found for org section: {org_name}")
            continue
        
        resp_start = resp_header_match.end()
        
        # Find end boundary
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


# --------------------------------------------------------------------------- #
# Org-based HSIB extraction (Report 4 style)
# --------------------------------------------------------------------------- #

def extract_org_based_hsib_responses(text: str) -> List[Dict]:
    """
    Extract responses from org-based HSIB response documents (Report 4 style).
    
    Format: [Org Header] -> [HSIB recommends...] -> "Response" -> [Response text]
    
    No rec IDs like R/2023/220 - uses position-based numbering.
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    responses = []
    
    # Find all "HSIB recommends" positions
    hsib_rec_pattern = re.compile(
        r'HSIB\s+recommends\s+that\s+(?:the\s+)?(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute[^.]*|Royal\s+College[^.]*|NICE|DHSC)',
        re.IGNORECASE
    )
    hsib_matches = list(hsib_rec_pattern.finditer(text))
    
    # Find all "Response" positions
    resp_pattern = re.compile(r'\bResponse\b', re.IGNORECASE)
    resp_matches = list(resp_pattern.finditer(text))
    
    logger.info(f"Org-based HSIB: Found {len(hsib_matches)} 'HSIB recommends' and {len(resp_matches)} 'Response' markers")
    
    for idx, hsib_match in enumerate(hsib_matches):
        org_name = hsib_match.group(1).strip()
        rec_start = hsib_match.start()
        
        # Find section end
        if idx + 1 < len(hsib_matches):
            section_end = hsib_matches[idx + 1].start()
        else:
            section_end = len(text)
        
        # Find "Response" within this section
        resp_start = None
        for resp_match in resp_matches:
            if rec_start < resp_match.start() < section_end:
                resp_start = resp_match.end()
                break
        
        if resp_start is None:
            logger.warning(f"No Response found for recommendation {idx + 1} ({org_name})")
            continue
        
        resp_end = section_end
        
        # Check for boundary markers
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


# --------------------------------------------------------------------------- #
# Standard HSIB extraction
# --------------------------------------------------------------------------- #

def extract_hsib_responses(text: str) -> List[Dict]:
    """
    Extract responses from HSIB-format response documents.
    
    HSIB format uses:
    - Organisation name as section header
    - "Response" as standalone header
    - Responses follow immediately after the recommendation they address
    
    v2.9 FIX: Excludes Safety observations and Safety actions sections.
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    
    # Exclude "Safety observations" and "Safety actions" sections
    safety_obs_match = re.search(r'\nSafety\s+observations?\s*\n', text, re.IGNORECASE)
    safety_act_match = re.search(r'\nSafety\s+actions?\s*\n', text, re.IGNORECASE)
    
    doc_end = len(text)
    if safety_obs_match:
        doc_end = min(doc_end, safety_obs_match.start())
    if safety_act_match:
        doc_end = min(doc_end, safety_act_match.start())
    
    if doc_end < len(text):
        logger.info(f"HSIB: Truncating from {len(text)} to {doc_end} chars to exclude Safety observations/actions")
        text = text[:doc_end]
    
    responses = []
    
    # Pattern to find HSIB recommendation markers
    hsib_rec_patterns = [
        re.compile(r'Safety\s+recommendation\s+(R/\d{4}/\d{3})[:\s]', re.IGNORECASE),
        re.compile(r'Recommendation\s+(\d{4}/\d{3})[:\s]', re.IGNORECASE),
    ]
    
    rec_matches = []
    for pattern in hsib_rec_patterns:
        rec_matches.extend(list(pattern.finditer(text)))
    rec_matches.sort(key=lambda m: m.start())
    
    # Response header patterns
    response_patterns = [
        re.compile(r'(?:^|\n)\s*Response\s*\n', re.IGNORECASE | re.MULTILINE),
        re.compile(r'Response(?=[A-Z][a-z])', re.IGNORECASE),
    ]
    
    response_matches = []
    for pattern in response_patterns:
        response_matches.extend(list(pattern.finditer(text)))
    
    # Deduplicate by position
    unique_matches = []
    last_pos = -100
    for m in sorted(response_matches, key=lambda x: x.start()):
        if m.start() - last_pos > 20:
            unique_matches.append(m)
            last_pos = m.start()
    response_matches = unique_matches
    
    # Organisation headers as boundaries
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
    
    logger.info(f"HSIB: Found {len(rec_matches)} recommendations and {len(response_matches)} response headers")
    
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
        
        # Check for boundary markers
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
        
        # Find the rec_id that PRECEDES this response header
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


# --------------------------------------------------------------------------- #
# Standard government format extraction
# --------------------------------------------------------------------------- #

def extract_government_responses(text: str) -> List[Dict]:
    """
    Extract responses from standard government response documents.
    
    Format: "Government response to recommendation N" followed by response text.
    """
    if not text:
        return []
    
    text = normalise_line_endings(text)
    text = clean_pdf_artifacts(text)
    
    responses = []
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
        
        # Check for recommendation markers
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
    
    # Sentence-level fallback extraction
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
    
    logger.info(f"Extracted {len(responses)} government responses")
    return responses


# --------------------------------------------------------------------------- #
# Main response extraction function
# --------------------------------------------------------------------------- #

def extract_response_sentences(text: str) -> List[Dict]:
    """
    Extract sentences that are government responses.
    Auto-detects document format and uses appropriate extraction method.
    
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
    
    # CRITICAL: Normalise line endings FIRST before any detection
    text = normalise_line_endings(text)
    
    logger.info(f"extract_response_sentences: Processing {len(text)} chars")
    logger.info(f"First 200 chars: {text[:200]}")
    
    # Check for Trust response format first (Report 7 style)
    logger.info("Checking for Trust response format...")
    if is_trust_response_document(text):
        logger.info("✅ Detected Trust response document format")
        return extract_trust_responses(text)
    
    # Check for HSSIB org-structured format (Report 6 style)
    logger.info("Checking for HSSIB org-structured format...")
    if is_hssib_org_structured_response(text):
        logger.info("✅ Detected HSSIB org-structured response format")
        return extract_hssib_org_structured_responses(text)
    
    # Check for org-based HSIB format (Report 4 style)
    logger.info("Checking for org-based HSIB format...")
    if is_org_based_hsib_response(text):
        logger.info("✅ Detected org-based HSIB response format")
        return extract_org_based_hsib_responses(text)
    
    # Check for standard HSIB format
    logger.info("Checking for standard HSIB format...")
    if is_hsib_response_document(text):
        logger.info("✅ Detected HSIB response document format")
        return extract_hsib_responses(text)
    
    # Standard government format extraction
    logger.info("Using standard government response extraction")
    return extract_government_responses(text)


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #

__all__ = [
    # Main entry point
    "extract_response_sentences",
    # Format-specific extractors
    "extract_trust_responses",
    "extract_hssib_org_structured_responses",
    "extract_org_based_hsib_responses",
    "extract_hsib_responses",
    "extract_government_responses",
]
