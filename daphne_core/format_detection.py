"""
Format Detection Module

Detects document formats to determine which extraction strategy to use.

v4.0 Changes:
- ADDED: Decimal format detection (8.1, 8.2, 8.3-8.4)
- ADDED: Parliamentary/Select Committee report detection
- IMPROVED: Better disambiguation between similar formats

Supported formats:
- HSIB 2018: Recommendation YYYY/NNN format
- HSSIB 2023+: Safety recommendation R/YYYY/NNN format
- Standard government: Recommendation N format
- Decimal government: Recommendation X.Y format (House of Lords/Commons)
- Trust response: TEWV/Trust response format
- HSSIB org-structured: Organisation-based with rec IDs
- Org-based HSIB: Organisation-based without rec IDs
"""

import logging
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FormatDetectionResult:
    """Result of format detection."""
    format_type: str
    confidence: float
    indicators: Dict[str, bool]
    recommended_extractor: str
    description: str


def normalise_line_endings(text: str) -> str:
    """Normalise all line endings to \\n."""
    if not text:
        return text
    return text.replace('\r\n', '\n').replace('\r', '\n')


# --------------------------------------------------------------------------- #
# Individual format detection functions
# --------------------------------------------------------------------------- #

def detect_hsib_2018_format(text: str) -> Tuple[bool, float]:
    """
    Detect HSIB 2018 format: Recommendation YYYY/NNN
    """
    if not text:
        return False, 0.0
    
    text = normalise_line_endings(text)
    
    # Pattern: "Recommendation 2018/006" or similar
    pattern = r'Recommendation\s+(\d{4}/\d{3})'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if len(matches) >= 2:
        return True, 0.95
    elif len(matches) == 1:
        return True, 0.80
    
    return False, 0.0


def detect_hssib_2023_format(text: str) -> Tuple[bool, float]:
    """
    Detect HSSIB 2023+ format: Safety recommendation R/YYYY/NNN
    """
    if not text:
        return False, 0.0
    
    text = normalise_line_endings(text)
    
    # Pattern: "Safety recommendation R/2023/220"
    pattern = r'Safety\s+recommendation\s+(R/\d{4}/\d{3})'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if len(matches) >= 2:
        return True, 0.95
    elif len(matches) == 1:
        return True, 0.80
    
    return False, 0.0


def detect_standard_numbered_format(text: str) -> Tuple[bool, float]:
    """
    Detect standard government format: Recommendation N
    """
    if not text:
        return False, 0.0
    
    text = normalise_line_endings(text)
    
    # Exclude other formats first
    if re.search(r'Recommendation\s+\d{4}/\d{3}', text, re.IGNORECASE):
        return False, 0.0  # HSIB 2018 format
    if re.search(r'Safety\s+recommendation\s+R/\d{4}/\d{3}', text, re.IGNORECASE):
        return False, 0.0  # HSSIB 2023 format
    if re.search(r'Recommendation\s+\d+\.\d+', text, re.IGNORECASE):
        return False, 0.0  # Decimal format
    
    # Pattern: "Recommendation 1", "Recommendation 12"
    pattern = r'Recommendation\s+(\d{1,2})(?:\s|:|$)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if len(matches) >= 3:
        return True, 0.90
    elif len(matches) >= 1:
        return True, 0.70
    
    return False, 0.0


def detect_decimal_format(text: str) -> Tuple[bool, float]:
    """
    v4.0: Detect decimal format: Recommendation X.Y or X.Y-X.Z
    Common in House of Lords/Commons Select Committee reports.
    """
    if not text:
        return False, 0.0
    
    text = normalise_line_endings(text)
    
    # Pattern 1: "Recommendation 8.1" explicit
    explicit_pattern = r'Recommendation\s+(\d+\.\d+)'
    explicit_matches = re.findall(explicit_pattern, text, re.IGNORECASE)
    
    # Pattern 2: "Government response to recommendation 8.1"
    response_pattern = r'Government\s+response\s+to\s+recommendation\s+(\d+\.\d+)'
    response_matches = re.findall(response_pattern, text, re.IGNORECASE)
    
    # Pattern 3: Range format "8.3-8.4"
    range_pattern = r'(\d+\.\d+)-(\d+\.\d+)'
    range_matches = re.findall(range_pattern, text)
    
    total_matches = len(explicit_matches) + len(response_matches)
    
    if total_matches >= 5 or len(range_matches) >= 2:
        return True, 0.95
    elif total_matches >= 2:
        return True, 0.85
    elif total_matches >= 1:
        return True, 0.70
    
    return False, 0.0


def detect_trust_response_format(text: str) -> Tuple[bool, float]:
    """
    Detect Trust response format: "TEWV response:" or "Trust response:"
    """
    if not text:
        return False, 0.0
    
    text = normalise_line_endings(text)
    
    # Pattern: "TEWV response:" or "Trust response:"
    pattern = r'(?:TEWV|Trust)\s+response[:\s]'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if len(matches) >= 3:
        return True, 0.95
    elif len(matches) >= 1:
        return True, 0.80
    
    return False, 0.0


def detect_hssib_org_structured_format(text: str) -> Tuple[bool, float]:
    """
    Detect HSSIB org-structured format (Report 6 style).
    Has: org headers + rec IDs + "HSSIB recommends" + "Actions planned"
    """
    if not text:
        return False, 0.0
    
    text = normalise_line_endings(text)
    
    indicators = {
        'hssib_recommends': bool(re.search(r'HSSIB\s+recommends', text, re.IGNORECASE)),
        'rec_ids': bool(re.search(r'R/\d{4}/\d{3}', text)),
        'response_headers': bool(re.search(r'(?:^|\n)Response\n', text, re.MULTILINE)),
        'org_headers': bool(re.search(
            r'(?:^|\n)(The\s+Shelford\s+Group|NHS\s+England|DHSC|Care\s+Quality\s+Commission|NICE)\s*\n',
            text, re.IGNORECASE | re.MULTILINE
        )),
        'actions_planned': bool(re.search(r'Actions\s+planned\s+to\s+deliver', text, re.IGNORECASE)),
    }
    
    # Need all key indicators
    if all([
        indicators['hssib_recommends'],
        indicators['rec_ids'],
        indicators['response_headers'],
        indicators['org_headers'],
        indicators['actions_planned'],
    ]):
        return True, 0.95
    
    # Partial match
    true_count = sum(indicators.values())
    if true_count >= 4:
        return True, 0.80
    
    return False, 0.0


def detect_org_based_hsib_format(text: str) -> Tuple[bool, float]:
    """
    Detect org-based HSIB format (Report 4 style).
    Has: org headers + "HSIB recommends" + "Response" but NO rec IDs
    """
    if not text:
        return False, 0.0
    
    text = normalise_line_endings(text)
    
    indicators = {
        'hsib_recommends': bool(re.search(r'HSIB\s+recommends', text, re.IGNORECASE)),
        'no_rec_ids': not bool(re.search(r'R/\d{4}/\d{3}', text)),
        'response_headers': bool(re.search(r'\bResponse\b', text, re.IGNORECASE)),
        'multiple_orgs': len(set(re.findall(
            r'\b(NHS\s+England|Care\s+Quality\s+Commission|National\s+Institute|Royal\s+College|NICE)\b',
            text, re.IGNORECASE
        ))) >= 2,
    }
    
    if all(indicators.values()):
        return True, 0.90
    
    return False, 0.0


def detect_government_response_format(text: str) -> Tuple[bool, float]:
    """
    Detect standard government response format.
    """
    if not text:
        return False, 0.0
    
    text = normalise_line_endings(text)
    
    # Check for decimal format first (v4.0)
    is_decimal, decimal_conf = detect_decimal_format(text)
    if is_decimal and decimal_conf >= 0.80:
        return False, 0.0  # Use decimal extractor instead
    
    # Pattern: "Government response to recommendation N"
    pattern = r'Government\s+response\s+to\s+recommendation\s+(\d+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if len(matches) >= 3:
        return True, 0.95
    elif len(matches) >= 1:
        return True, 0.80
    
    return False, 0.0


# --------------------------------------------------------------------------- #
# Main detection function
# --------------------------------------------------------------------------- #

def detect_document_format(text: str) -> FormatDetectionResult:
    """
    Detect document format and return detection result.
    
    Checks formats in order of specificity:
    1. HSSIB 2023+ (R/YYYY/NNN)
    2. HSIB 2018 (YYYY/NNN)
    3. Decimal (X.Y)
    4. HSSIB org-structured
    5. Trust response
    6. Org-based HSIB
    7. Standard numbered
    8. Government response
    """
    if not text:
        return FormatDetectionResult(
            format_type='unknown',
            confidence=0.0,
            indicators={},
            recommended_extractor='fallback',
            description='No text provided'
        )
    
    text = normalise_line_endings(text)
    
    # Test each format
    results = []
    
    # 1. HSSIB 2023+
    is_match, conf = detect_hssib_2023_format(text)
    if is_match:
        results.append(('hssib_2023', conf, 'hssib_2023_extractor', 
                       'HSSIB 2023+ format with R/YYYY/NNN IDs'))
    
    # 2. HSIB 2018
    is_match, conf = detect_hsib_2018_format(text)
    if is_match:
        results.append(('hsib_2018', conf, 'hsib_2018_extractor',
                       'HSIB 2018 format with YYYY/NNN IDs'))
    
    # 3. Decimal format (v4.0)
    is_match, conf = detect_decimal_format(text)
    if is_match:
        results.append(('decimal', conf, 'decimal_extractor',
                       'Decimal format with X.Y recommendation IDs'))
    
    # 4. HSSIB org-structured
    is_match, conf = detect_hssib_org_structured_format(text)
    if is_match:
        results.append(('hssib_org_structured', conf, 'hssib_org_extractor',
                       'HSSIB org-structured format (Report 6 style)'))
    
    # 5. Trust response
    is_match, conf = detect_trust_response_format(text)
    if is_match:
        results.append(('trust_response', conf, 'trust_extractor',
                       'Trust response format with TEWV/Trust markers'))
    
    # 6. Org-based HSIB
    is_match, conf = detect_org_based_hsib_format(text)
    if is_match:
        results.append(('org_based_hsib', conf, 'org_based_extractor',
                       'Org-based HSIB format (Report 4 style)'))
    
    # 7. Standard numbered
    is_match, conf = detect_standard_numbered_format(text)
    if is_match:
        results.append(('standard_numbered', conf, 'standard_extractor',
                       'Standard numbered format (Recommendation N)'))
    
    # 8. Government response
    is_match, conf = detect_government_response_format(text)
    if is_match:
        results.append(('government_response', conf, 'government_extractor',
                       'Standard government response format'))
    
    # Return best match
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        best = results[0]
        return FormatDetectionResult(
            format_type=best[0],
            confidence=best[1],
            indicators={r[0]: r[1] for r in results},
            recommended_extractor=best[2],
            description=best[3]
        )
    
    # No match found
    return FormatDetectionResult(
        format_type='unstructured',
        confidence=0.5,
        indicators={},
        recommended_extractor='sentence_fallback',
        description='No structured format detected, using sentence-based extraction'
    )


def detect_response_document_format(text: str) -> FormatDetectionResult:
    """
    Detect format of a response document specifically.
    """
    if not text:
        return FormatDetectionResult(
            format_type='unknown',
            confidence=0.0,
            indicators={},
            recommended_extractor='fallback',
            description='No text provided'
        )
    
    text = normalise_line_endings(text)
    
    # Check specific response formats
    results = []
    
    # Decimal government response (v4.0)
    is_match, conf = detect_decimal_format(text)
    if is_match:
        results.append(('decimal_government', conf, 'decimal_response_extractor',
                       'Decimal government response format'))
    
    # HSSIB org-structured response
    is_match, conf = detect_hssib_org_structured_format(text)
    if is_match:
        results.append(('hssib_org_structured', conf, 'hssib_org_response_extractor',
                       'HSSIB org-structured response format'))
    
    # Trust response
    is_match, conf = detect_trust_response_format(text)
    if is_match:
        results.append(('trust_response', conf, 'trust_response_extractor',
                       'Trust response format'))
    
    # Org-based HSIB response
    is_match, conf = detect_org_based_hsib_format(text)
    if is_match:
        results.append(('org_based_hsib', conf, 'org_based_response_extractor',
                       'Org-based HSIB response format'))
    
    # Standard government response
    is_match, conf = detect_government_response_format(text)
    if is_match:
        results.append(('government_response', conf, 'government_response_extractor',
                       'Standard government response format'))
    
    # HSIB response (generic)
    has_hsib = bool(re.search(r'\bH[SS]I?B\b', text, re.IGNORECASE))
    has_response = bool(re.search(r'\bResponse\b', text, re.IGNORECASE))
    if has_hsib and has_response:
        results.append(('hsib_response', 0.70, 'hsib_response_extractor',
                       'Generic HSIB response format'))
    
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        best = results[0]
        return FormatDetectionResult(
            format_type=best[0],
            confidence=best[1],
            indicators={r[0]: r[1] for r in results},
            recommended_extractor=best[2],
            description=best[3]
        )
    
    return FormatDetectionResult(
        format_type='unknown',
        confidence=0.3,
        indicators={},
        recommended_extractor='generic_response_extractor',
        description='No specific response format detected'
    )


# --------------------------------------------------------------------------- #
# Convenience functions
# --------------------------------------------------------------------------- #

def get_format_type(text: str) -> str:
    """Simple function to get format type string."""
    result = detect_document_format(text)
    return result.format_type


def get_recommended_extractor(text: str) -> str:
    """Get recommended extractor for document."""
    result = detect_document_format(text)
    return result.recommended_extractor


def is_structured_document(text: str) -> bool:
    """Check if document has structured recommendation format."""
    result = detect_document_format(text)
    return result.confidence >= 0.70 and result.format_type != 'unstructured'
