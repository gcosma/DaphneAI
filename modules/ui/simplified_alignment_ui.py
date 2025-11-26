# modules/ui/simplified_alignment_ui.py
"""
🔗 Simplified Recommendation-Response Alignment Interface
Uses recommendations extracted in the Recommendations tab and finds responses in separate documents.

UPDATED: Fixed response classification patterns to properly detect "supports the recommendation"
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging
import re
from typing import Dict, List, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# =============================================================================
# DOCX REPORT GENERATION
# =============================================================================

def generate_docx_report(alignments: List[Dict], status_counts: Dict, total: int, with_response: int) -> bytes:
    """Generate a Word document report of the alignment results"""
    import subprocess
    import tempfile
    import os
    
    # Create JavaScript file for docx generation
    js_content = create_docx_js(alignments, status_counts, total, with_response)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        js_path = os.path.join(tmpdir, 'generate_report.js')
        docx_path = os.path.join(tmpdir, 'report.docx')
        
        with open(js_path, 'w') as f:
            f.write(js_content)
        
        # Run the JavaScript to generate the docx
        result = subprocess.run(
            ['node', js_path],
            cwd=tmpdir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to generate docx: {result.stderr}")
        
        with open(docx_path, 'rb') as f:
            return f.read()


def create_docx_js(alignments: List[Dict], status_counts: Dict, total: int, with_response: int) -> str:
    """Create JavaScript code to generate the Word document"""
    import json
    
    # Prepare data for JavaScript
    details = []
    for idx, a in enumerate(alignments, 1):
        rec = a['recommendation']
        resp = a['response']
        details.append({
            'number': idx,
            'status': resp['status'] if resp else 'No Response',
            'recommendation': rec['text'][:2000],  # Limit length
            'response': resp['response_text'][:2000] if resp else 'No response found',
            'match_confidence': f"{resp['similarity']:.0%}" if resp else 'N/A'
        })
    
    # Escape for JavaScript
    details_json = json.dumps(details)
    
    js_code = f'''
const {{ Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        Header, Footer, AlignmentType, BorderStyle, WidthType, HeadingLevel,
        ShadingType, PageNumber }} = require('docx');
const fs = require('fs');

const statusCounts = {json.dumps(dict(status_counts))};
const total = {total};
const withResponse = {with_response};
const details = {details_json};

// Status colors
const statusColors = {{
    'Accepted': '92D050',
    'Partial': 'FFC000', 
    'Rejected': 'FF6B6B',
    'Noted': '87CEEB',
    'No Response': 'D3D3D3',
    'Unclear': 'D3D3D3'
}};

const tableBorder = {{ style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" }};
const cellBorders = {{ top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder }};

// Build document sections
const children = [];

// Title
children.push(new Paragraph({{
    heading: HeadingLevel.TITLE,
    alignment: AlignmentType.CENTER,
    spacing: {{ after: 400 }},
    children: [new TextRun({{ text: "Recommendation-Response Alignment Report", bold: true, size: 48 }})]
}}));

// Generated date
children.push(new Paragraph({{
    alignment: AlignmentType.CENTER,
    spacing: {{ after: 400 }},
    children: [new TextRun({{ text: "Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", size: 22, color: "666666" }})]
}}));

// Summary section
children.push(new Paragraph({{
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 400, after: 200 }},
    children: [new TextRun({{ text: "Summary", bold: true, size: 32 }})]
}}));

// Summary table
children.push(new Table({{
    columnWidths: [4680, 4680],
    rows: [
        new TableRow({{
            children: [
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    children: [new Paragraph({{ children: [new TextRun({{ text: "Total Recommendations", bold: true }})] }})]
                }}),
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    children: [new Paragraph({{ children: [new TextRun(String(total))] }})]
                }})
            ]
        }}),
        new TableRow({{
            children: [
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    children: [new Paragraph({{ children: [new TextRun({{ text: "Responses Found", bold: true }})] }})]
                }}),
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    children: [new Paragraph({{ children: [new TextRun(String(withResponse))] }})]
                }})
            ]
        }}),
        new TableRow({{
            children: [
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    shading: {{ fill: "92D050", type: ShadingType.CLEAR }},
                    children: [new Paragraph({{ children: [new TextRun({{ text: "Accepted", bold: true }})] }})]
                }}),
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    children: [new Paragraph({{ children: [new TextRun(String(statusCounts['Accepted'] || 0))] }})]
                }})
            ]
        }}),
        new TableRow({{
            children: [
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    shading: {{ fill: "FFC000", type: ShadingType.CLEAR }},
                    children: [new Paragraph({{ children: [new TextRun({{ text: "Partial", bold: true }})] }})]
                }}),
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    children: [new Paragraph({{ children: [new TextRun(String(statusCounts['Partial'] || 0))] }})]
                }})
            ]
        }}),
        new TableRow({{
            children: [
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    shading: {{ fill: "FF6B6B", type: ShadingType.CLEAR }},
                    children: [new Paragraph({{ children: [new TextRun({{ text: "Rejected", bold: true }})] }})]
                }}),
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    children: [new Paragraph({{ children: [new TextRun(String(statusCounts['Rejected'] || 0))] }})]
                }})
            ]
        }}),
        new TableRow({{
            children: [
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    shading: {{ fill: "D3D3D3", type: ShadingType.CLEAR }},
                    children: [new Paragraph({{ children: [new TextRun({{ text: "No Response", bold: true }})] }})]
                }}),
                new TableCell({{
                    borders: cellBorders,
                    width: {{ size: 4680, type: WidthType.DXA }},
                    children: [new Paragraph({{ children: [new TextRun(String(statusCounts['No Response'] || 0))] }})]
                }})
            ]
        }})
    ]
}}));

// Details section
children.push(new Paragraph({{
    heading: HeadingLevel.HEADING_1,
    spacing: {{ before: 600, after: 200 }},
    children: [new TextRun({{ text: "Detailed Results", bold: true, size: 32 }})]
}}));

// Add each recommendation
details.forEach((item, index) => {{
    const statusColor = statusColors[item.status] || 'D3D3D3';
    
    // Recommendation heading
    children.push(new Paragraph({{
        heading: HeadingLevel.HEADING_2,
        spacing: {{ before: 400, after: 100 }},
        children: [
            new TextRun({{ text: `Recommendation ${{item.number}}`, bold: true, size: 26 }}),
            new TextRun({{ text: "  |  ", size: 26, color: "999999" }}),
            new TextRun({{ text: item.status, bold: true, size: 26, color: statusColor.replace('#', '') }})
        ]
    }}));
    
    // Recommendation text
    children.push(new Paragraph({{
        spacing: {{ before: 100, after: 100 }},
        children: [new TextRun({{ text: "Recommendation: ", bold: true }})]
    }}));
    children.push(new Paragraph({{
        spacing: {{ after: 200 }},
        shading: {{ fill: "F0F8FF", type: ShadingType.CLEAR }},
        children: [new TextRun({{ text: item.recommendation, size: 22 }})]
    }}));
    
    // Response text
    children.push(new Paragraph({{
        spacing: {{ before: 100, after: 100 }},
        children: [new TextRun({{ text: "Government Response: ", bold: true }})]
    }}));
    children.push(new Paragraph({{
        spacing: {{ after: 100 }},
        shading: {{ fill: "F0FFF0", type: ShadingType.CLEAR }},
        children: [new TextRun({{ text: item.response, size: 22 }})]
    }}));
    
    // Match confidence
    children.push(new Paragraph({{
        spacing: {{ after: 300 }},
        children: [new TextRun({{ text: `Match Confidence: ${{item.match_confidence}}`, size: 20, color: "666666", italics: true }})]
    }}));
}});

const doc = new Document({{
    styles: {{
        default: {{
            document: {{
                run: {{ font: "Arial", size: 24 }}
            }}
        }},
        paragraphStyles: [
            {{ id: "Title", name: "Title", basedOn: "Normal",
                run: {{ size: 48, bold: true, color: "000000", font: "Arial" }},
                paragraph: {{ spacing: {{ before: 240, after: 120 }}, alignment: AlignmentType.CENTER }} }},
            {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: {{ size: 32, bold: true, color: "2E74B5", font: "Arial" }},
                paragraph: {{ spacing: {{ before: 240, after: 120 }} }} }},
            {{ id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: {{ size: 26, bold: true, color: "404040", font: "Arial" }},
                paragraph: {{ spacing: {{ before: 200, after: 100 }} }} }}
        ]
    }},
    sections: [{{
        properties: {{
            page: {{
                margin: {{ top: 1440, right: 1440, bottom: 1440, left: 1440 }}
            }}
        }},
        headers: {{
            default: new Header({{
                children: [new Paragraph({{
                    alignment: AlignmentType.RIGHT,
                    children: [new TextRun({{ text: "Recommendation-Response Alignment Report", size: 20, color: "999999" }})]
                }})]
            }})
        }},
        footers: {{
            default: new Footer({{
                children: [new Paragraph({{
                    alignment: AlignmentType.CENTER,
                    children: [
                        new TextRun({{ text: "Page ", size: 20 }}),
                        new TextRun({{ children: [PageNumber.CURRENT], size: 20 }}),
                        new TextRun({{ text: " of ", size: 20 }}),
                        new TextRun({{ children: [PageNumber.TOTAL_PAGES], size: 20 }})
                    ]
                }})]
            }})
        }},
        children: children
    }}]
}});

Packer.toBuffer(doc).then(buffer => {{
    fs.writeFileSync("report.docx", buffer);
    console.log("Report generated successfully");
}}).catch(err => {{
    console.error("Error generating report:", err);
    process.exit(1);
}});
'''
    return js_code


def generate_markdown_report(alignments: List[Dict], status_counts: Dict, total: int, with_response: int) -> str:
    """Generate markdown report as fallback"""
    report = f"""# Recommendation-Response Alignment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Recommendations: {total}
- Responses Found: {with_response}
- Accepted: {status_counts.get('Accepted', 0)}
- Partial: {status_counts.get('Partial', 0)}
- Rejected: {status_counts.get('Rejected', 0)}
- Noted: {status_counts.get('Noted', 0)}
- No Response: {status_counts.get('No Response', 0)}

## Details
"""
    for idx, a in enumerate(alignments, 1):
        rec = a['recommendation']
        resp = a['response']
        status = resp['status'] if resp else 'No Response'
        report += f"""
### Recommendation {idx}
**Status:** {status}

**Recommendation:**
> {rec['text']}

**Response:**
> {resp['response_text'] if resp else 'No response found'}

---
"""
    return report


# =============================================================================
# RESPONSE STATUS CLASSIFICATION - UPDATED PATTERNS
# =============================================================================

RESPONSE_PATTERNS = {
    'accepted': [
        # Original patterns
        r'\baccept(?:s|ed|ing)?\s+(?:this\s+)?(?:recommendation|rec)\b',
        r'\baccept(?:s|ed|ing)?\s+in\s+full\b',
        r'\bfully\s+accept(?:s|ed)?\b',
        r'\bagree(?:s|d)?\s+(?:with\s+)?(?:this\s+)?(?:recommendation|rec)\b',
        r'\bwill\s+implement\b',
        r'\bcommit(?:s|ted)?\s+to\s+implement\b',
        r'\bendorse(?:s|d)?\b',
        # FIXED: Handle "supports the recommendation" pattern
        r'\bsupport(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)\b',
        # NEW: Additional acceptance patterns found in government docs
        r'\bthe\s+government\s+support(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation\b',
        r'\bgovernment\s+support(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation\b',
        r'\bwe\s+support\s+(?:this\s+|the\s+)?recommendation\b',
        r'\baccept(?:s|ed)?\s+(?:the\s+)?recommendation\b',
        r'\bagree(?:s|d)?\s+(?:with\s+)?(?:the\s+)?recommendation\b',
    ],
    'rejected': [
        r'\breject(?:s|ed|ing)?\s+(?:this\s+)?(?:recommendation|rec)\b',
        r'\bdoes?\s+not\s+accept\b',
        r'\bcannot\s+accept\b',
        r'\bdecline(?:s|d)?\s+(?:to\s+accept)?\b',
        r'\bdo(?:es)?\s+not\s+agree\b',
        r'\bdisagree(?:s|d)?\b',
        r'\bnot\s+(?:be\s+)?implement(?:ed|ing)?\b',
        # NEW: Additional rejection patterns
        r'\bthe\s+government\s+(?:does\s+not|cannot)\s+(?:accept|support)\b',
        r'\breject(?:s|ed)?\s+(?:the\s+)?recommendation\b',
    ],
    'partial': [
        r'\baccept(?:s|ed)?\s+in\s+(?:part|principle)\b',
        r'\bpartially\s+accept(?:s|ed)?\b',
        r'\baccept(?:s|ed)?\s+(?:with\s+)?(?:some\s+)?(?:reservations?|modifications?|amendments?)\b',
        r'\baccept(?:s|ed)?\s+(?:the\s+)?(?:spirit|intent)\b',
        r'\bagree(?:s|d)?\s+in\s+principle\b',
        r'\bunder\s+consideration\b',
        r'\bwill\s+consider\b',
        r'\bfurther\s+(?:consideration|review|work)\s+(?:is\s+)?(?:needed|required)\b',
        # NEW: Additional partial patterns
        r'\bsupport(?:s|ed)?\s+in\s+(?:part|principle)\b',
    ],
    'noted': [
        r'\bnote(?:s|d)?\s+(?:this\s+)?(?:recommendation|rec)\b',
        r'\backnowledge(?:s|d)?\b',
        r'\btake(?:s|n)?\s+note\b',
        r'\bwill\s+(?:review|examine|look\s+at)\b',
        # NEW: Additional noted patterns
        r'\bnote(?:s|d)?\s+(?:the\s+)?recommendation\b',
    ]
}

# Keywords that indicate a response to a recommendation
RESPONSE_INDICATORS = [
    'government response', 'response to', 'in response', 'responding to',
    'the government', 'we accept', 'we reject', 'we agree', 'we note',
    'this recommendation', 'recommendation is', 'recommendation will',
    'accept', 'reject', 'implement', 'agree', 'noted', 'consider',
    'support', 'supports'  # NEW: Added support keywords
]


# =============================================================================
# RESPONSE EXTRACTION
# =============================================================================

def extract_response_sentences(text: str) -> List[Dict]:
    """Extract sentences that look like responses to recommendations"""
    
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    responses = []
    
    # Track if we're in a "Government response to recommendation X" section
    current_rec_number = None
    
    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 30:
            continue
        
        sentence_lower = sentence.lower()
        
        # =================================================================
        # SKIP sentences that are just quoting recommendations
        # =================================================================
        
        # Skip if it starts with "Recommendation N" and doesn't have response language
        if re.match(r'^recommendation\s+\d+\s+', sentence_lower):
            # Check if this also contains response language
            has_response_language = any(term in sentence_lower for term in 
                ['government response', 'accept', 'reject', 'agree', 'the government', 
                 'we will', 'we accept', 'support', 'noted'])
            if not has_response_language:
                continue  # Skip - this is just quoting the recommendation
        
        # =================================================================
        # CRITICAL: Skip if sentence looks like a recommendation, not a response
        # =================================================================
        
        # If sentence starts with recommendation-style language and doesn't start with gov response
        starts_like_recommendation = any(sentence_lower.startswith(pattern) for pattern in [
            'nhs england should', 'providers should', 'trusts should', 'boards should',
            'icss should', 'ics should', 'cqc should', 'dhsc should', 'every provider',
            'all providers', 'provider boards', 'this multi-professional alliance should',
            'the review should', 'commissioners should', 'regulators should'
        ])
        
        starts_with_gov_response = sentence_lower.startswith('government response') or \
                                   sentence_lower.startswith('the government')
        
        if starts_like_recommendation and not starts_with_gov_response:
            # Check if "Government response" appears later in the sentence
            gov_response_pos = sentence_lower.find('government response to recommendation')
            if gov_response_pos > 0:
                # Extract only the government response part
                sentence = sentence[gov_response_pos:]
                sentence_lower = sentence.lower()
            else:
                continue  # Skip - this looks like a recommendation
        
        # =================================================================
        # Detect "Government response to recommendation N" headers
        # =================================================================
        gov_resp_match = re.search(r'government\s+response\s+to\s+recommendation\s+(\d+)', sentence_lower)
        if gov_resp_match:
            current_rec_number = gov_resp_match.group(1)
        
        # =================================================================
        # Check if this looks like an actual response
        # =================================================================
        is_response = False
        response_type = 'unknown'
        confidence = 0.0
        
        # PRIORITY: Sentences starting with "Government response" are definitely responses
        if starts_with_gov_response:
            is_response = True
            response_type = 'government_response'
            confidence = 0.95
        
        # Check for response patterns
        if not is_response:
            for status, patterns in RESPONSE_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, sentence_lower):
                        is_response = True
                        response_type = status
                        confidence = 0.9
                        break
                if is_response:
                    break
        
        # Check for general response indicators (must have multiple)
        if not is_response:
            indicator_count = sum(1 for ind in RESPONSE_INDICATORS if ind in sentence_lower)
            if indicator_count >= 2:
                is_response = True
                response_type = 'general_response'
                confidence = 0.6 + (indicator_count * 0.1)
        
        # Check for "The government" statements
        if not is_response and re.search(r'\bthe\s+government\s+(will|has|is|supports?|agrees?|accepts?)\b', sentence_lower):
            is_response = True
            response_type = 'government_statement'
            confidence = 0.85
        
        # Check for recommendation number references in response context
        rec_ref = re.search(r'recommendation\s+(\d+)', sentence_lower)
        if rec_ref:
            # Only count as response if it has response language
            if any(term in sentence_lower for term in ['accept', 'reject', 'agree', 'support', 'response', 'government']):
                is_response = True
                confidence = max(confidence, 0.8)
        
        # Use the tracked recommendation number if we're in a response section
        detected_rec_number = rec_ref.group(1) if rec_ref else current_rec_number
        
        if is_response:
            responses.append({
                'text': sentence,
                'position': idx,
                'response_type': response_type,
                'confidence': min(confidence, 1.0),
                'rec_number': detected_rec_number
            })
    
    return responses


def classify_response_status(response_text: str) -> Tuple[str, float]:
    """
    Classify a response as Accepted, Rejected, Partial, or Noted
    
    UPDATED: Now properly handles "the government supports the recommendation" patterns
    """
    
    text_lower = response_text.lower()
    
    # Check patterns in order of specificity
    for status, patterns in RESPONSE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                confidence = 0.9 if status in ['accepted', 'rejected'] else 0.75
                return status.title(), confidence
    
    # NEW: Additional keyword-based classification as fallback
    # Check for strong acceptance indicators even without exact pattern match
    acceptance_keywords = ['supports', 'support', 'accepts', 'accept', 'agrees', 'agree', 'endorsed']
    rejection_keywords = ['rejects', 'reject', 'decline', 'refuses', 'oppose']
    
    has_acceptance = any(kw in text_lower for kw in acceptance_keywords)
    has_rejection = any(kw in text_lower for kw in rejection_keywords)
    has_recommendation_ref = 'recommendation' in text_lower
    has_government = 'government' in text_lower or 'we ' in text_lower
    
    # If it mentions government + recommendation + acceptance word = likely accepted
    if has_government and has_recommendation_ref and has_acceptance and not has_rejection:
        return 'Accepted', 0.8
    
    if has_government and has_recommendation_ref and has_rejection:
        return 'Rejected', 0.8
    
    return 'Unclear', 0.5


# =============================================================================
# SEMANTIC MATCHING
# =============================================================================

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate word-based similarity between two texts"""
    
    # Tokenise and clean
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'must', 'shall', 'this', 'that',
                  'these', 'those', 'it', 'its', 'they', 'their', 'we', 'our'}
    
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def is_self_match(rec_text: str, resp_text: str) -> bool:
    """Check if the response is actually just quoting the recommendation"""
    
    # Clean texts for comparison
    rec_clean = re.sub(r'\s+', ' ', rec_text.lower().strip())
    resp_clean = re.sub(r'\s+', ' ', resp_text.lower().strip())
    
    # Check 1: Exact match
    if rec_clean == resp_clean:
        return True
    
    # Check 2: One contains the other almost entirely
    if len(rec_clean) > 50 and len(resp_clean) > 50:
        if rec_clean in resp_clean or resp_clean in rec_clean:
            return True
    
    # Check 3: Very high similarity (>90%) without response keywords
    if len(rec_clean) >= 100 and len(resp_clean) >= 100:
        if rec_clean[:100] == resp_clean[:100]:
            response_words = ['accept', 'reject', 'agree', 'support', 'government response', 
                             'we will', 'the government', 'noted', 'implement']
            if not any(word in resp_clean for word in response_words):
                return True
    
    # Check 4: Calculate actual text overlap
    rec_words = set(rec_clean.split())
    resp_words = set(resp_clean.split())
    
    if len(rec_words) > 10 and len(resp_words) > 10:
        overlap = len(rec_words & resp_words) / min(len(rec_words), len(resp_words))
        if overlap > 0.85:
            response_words = ['accept', 'reject', 'agree', 'support', 'government response', 
                             'we will', 'the government', 'noted', 'implement']
            if not any(word in resp_clean for word in response_words):
                return True
    
    return False


def is_response_identical_to_recommendation(rec_text: str, resp_text: str) -> bool:
    """
    Check if the response text is identical or nearly identical to the recommendation.
    This catches cases where recommendation text is being matched as a response.
    """
    # Clean and normalise both texts
    rec_clean = re.sub(r'\s+', ' ', rec_text.lower().strip())
    resp_clean = re.sub(r'\s+', ' ', resp_text.lower().strip())
    
    # Check 1: Exact match
    if rec_clean == resp_clean:
        return True
    
    # Check 2: Response is contained within recommendation
    if resp_clean in rec_clean:
        return True
    
    # Check 3: Recommendation is contained within response (response is just expanded rec)
    if rec_clean in resp_clean:
        # But allow if response has genuine government response language BEFORE the rec text
        resp_before_rec = resp_clean.split(rec_clean)[0]
        if 'government response' in resp_before_rec or 'the government supports' in resp_before_rec:
            return False  # This is a genuine response that quotes the recommendation
        return True
    
    # Check 4: Response starts with the same text as recommendation (first 150 chars)
    if len(rec_clean) > 150 and len(resp_clean) > 150:
        if rec_clean[:150] == resp_clean[:150]:
            return True
    
    # Check 5: Very high word overlap without government response markers at the start
    rec_words = set(rec_clean.split())
    resp_words = set(resp_clean.split())
    
    if len(rec_words) > 20 and len(resp_words) > 20:
        overlap = len(rec_words & resp_words) / min(len(rec_words), len(resp_words))
        if overlap > 0.8:
            # Check if response starts with government language
            resp_start = resp_clean[:100]
            if not any(marker in resp_start for marker in ['government response', 'the government supports', 'the government accepts', 'we accept', 'we support']):
                return True
    
    return False


def find_best_response(recommendation: Dict, responses: List[Dict], 
                       min_similarity: float = 0.15) -> Dict:
    """Find the best matching response for a recommendation"""
    
    rec_text = recommendation['text']
    best_match = None
    best_score = 0.0
    
    for response in responses:
        resp_text = response['text']
        
        # Skip self-matches
        if is_self_match(rec_text, resp_text):
            continue
        
        # Skip if response is identical/near-identical to recommendation
        if is_response_identical_to_recommendation(rec_text, resp_text):
            continue
        
        # Calculate base similarity
        similarity = calculate_text_similarity(rec_text, resp_text)
        
        # Boost for recommendation number matches
        rec_num_match = re.search(r'recommendation\s+(\d+)', rec_text.lower())
        resp_num_match = re.search(r'recommendation\s+(\d+)', resp_text.lower())
        
        if rec_num_match and resp_num_match:
            if rec_num_match.group(1) == resp_num_match.group(1):
                similarity += 0.4  # Strong boost for matching recommendation numbers
        
        # Boost for response language
        response_boost = 0.0
        for term in ['government response', 'accept', 'reject', 'agree', 'support', 
                     'the government will', 'we will', 'implement', 'supports']:
            if term in resp_text.lower():
                response_boost += 0.1
        similarity += min(response_boost, 0.3)
        
        # Penalise if response looks like it's just quoting the recommendation
        if similarity > 0.8 and not any(term in resp_text.lower() for term in 
                                        ['accept', 'reject', 'agree', 'support', 'government']):
            similarity *= 0.5  # Heavy penalty for high similarity without response language
        
        if similarity > best_score and similarity >= min_similarity:
            best_score = similarity
            status, status_conf = classify_response_status(resp_text)
            best_match = {
                'response_text': resp_text,
                'similarity': min(similarity, 1.0),
                'status': status,
                'status_confidence': status_conf,
                'response_type': response.get('response_type', 'unknown'),
                'source_document': response.get('source_document', 'unknown')
            }
    
    return best_match


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def render_simple_alignment_interface(documents: List[Dict]):
    """Render the simplified alignment interface"""
    
    st.markdown("### 🔗 Find Government Responses")
    
    # Check if we have extracted recommendations
    if 'extracted_recommendations' not in st.session_state or not st.session_state.extracted_recommendations:
        st.warning("⚠️ No recommendations extracted yet!")
        st.info("""
        **How to use this feature:**
        1. Go to the **🎯 Recommendations** tab
        2. Upload and process your **inquiry/report document**
        3. Click **Extract Recommendations**
        4. Then return here to find government responses
        """)
        
        # Option to extract recommendations here
        st.markdown("---")
        st.markdown("**Or extract recommendations directly:**")
        
        doc_names = [doc['filename'] for doc in documents]
        rec_doc = st.selectbox("Select recommendation document:", doc_names, key="align_rec_doc")
        
        if st.button("🔍 Extract Recommendations Now"):
            doc = next((d for d in documents if d['filename'] == rec_doc), None)
            if doc and 'text' in doc:
                try:
                    from modules.simple_recommendation_extractor import extract_recommendations
                    recs = extract_recommendations(doc['text'], min_confidence=0.75)
                    if recs:
                        st.session_state.extracted_recommendations = recs
                        st.success(f"✅ Extracted {len(recs)} recommendations!")
                        st.rerun()
                    else:
                        st.warning("No recommendations found in this document.")
                except Exception as e:
                    st.error(f"Error: {e}")
        return
    
    recommendations = st.session_state.extracted_recommendations
    
    # Show summary of recommendations
    st.success(f"✅ Using **{len(recommendations)}** recommendations from previous extraction")
    
    with st.expander("📋 View Extracted Recommendations", expanded=False):
        for i, rec in enumerate(recommendations[:10], 1):
            st.markdown(f"**{i}.** {rec['text'][:150]}...")
        if len(recommendations) > 10:
            st.caption(f"... and {len(recommendations) - 10} more")
    
    st.markdown("---")
    
    # Select response document(s)
    st.markdown("#### 📄 Select Government Response Document(s)")
    
    doc_names = [doc['filename'] for doc in documents]
    
    # Try to auto-detect response documents
    suggested_resp_docs = []
    for name in doc_names:
        name_lower = name.lower()
        if any(term in name_lower for term in ['response', 'reply', 'government', 'answer']):
            suggested_resp_docs.append(name)
    
    resp_docs = st.multiselect(
        "Select response documents:",
        options=doc_names,
        default=suggested_resp_docs,
        help="Select documents containing government responses to the recommendations"
    )
    
    if not resp_docs:
        st.info("👆 Select at least one response document to continue")
        return
    
    # Run alignment
    if st.button("🔗 Find Responses", type="primary"):
        
        with st.spinner("Analysing documents for responses..."):
            
            # Extract responses from selected documents
            all_responses = []
            for doc_name in resp_docs:
                doc = next((d for d in documents if d['filename'] == doc_name), None)
                if doc and 'text' in doc:
                    doc_responses = extract_response_sentences(doc['text'])
                    for resp in doc_responses:
                        resp['source_document'] = doc_name
                    all_responses.extend(doc_responses)
            
            if not all_responses:
                st.warning("⚠️ No response patterns found in selected documents.")
                st.info("This might mean the document format is different than expected, or it's not a government response document.")
                return
            
            st.info(f"Found **{len(all_responses)}** potential response sentences")
            
            # Match recommendations to responses
            alignments = []
            
            progress = st.progress(0)
            
            for idx, rec in enumerate(recommendations):
                progress.progress((idx + 1) / len(recommendations))
                
                best_response = find_best_response(rec, all_responses)
                
                alignments.append({
                    'recommendation': rec,
                    'response': best_response,
                    'has_response': best_response is not None
                })
            
            progress.empty()
            
            # Store results
            st.session_state.alignment_results = alignments
    
    # Display results
    if 'alignment_results' in st.session_state:
        display_alignment_results(st.session_state.alignment_results)


def display_alignment_results(alignments: List[Dict]):
    """Display the alignment results with status indicators"""
    
    st.markdown("---")
    st.markdown("### 📊 Alignment Results")
    
    # Calculate statistics
    total = len(alignments)
    with_response = sum(1 for a in alignments if a['has_response'])
    
    # Count by status
    status_counts = Counter()
    for a in alignments:
        if a['has_response'] and a['response']:
            status_counts[a['response']['status']] += 1
        else:
            status_counts['No Response'] += 1
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Recommendations", total)
    with col2:
        st.metric("✅ Accepted", status_counts.get('Accepted', 0))
    with col3:
        st.metric("⚠️ Partial", status_counts.get('Partial', 0))
    with col4:
        st.metric("❌ Rejected", status_counts.get('Rejected', 0))
    with col5:
        st.metric("❓ No Response", status_counts.get('No Response', 0) + status_counts.get('Unclear', 0))
    
    # Status legend
    st.markdown("""
    ---
    #### 🎨 Status Guide
    | Status | Meaning |
    |--------|---------|
    | ✅ **Accepted** | Government fully accepts the recommendation |
    | ⚠️ **Partial** | Accepted in principle or with modifications |
    | ❌ **Rejected** | Government does not accept the recommendation |
    | 📝 **Noted** | Acknowledged but no clear commitment |
    | ❓ **No Response** | No matching response found |
    """)
    
    st.markdown("---")
    
    # Sort options
    sort_option = st.selectbox(
        "Sort by:",
        ["Original Order", "Status (Accepted first)", "Status (No Response first)", "Match Confidence"]
    )
    
    # Sort alignments
    sorted_alignments = alignments.copy()
    
    if sort_option == "Status (Accepted first)":
        status_order = {'Accepted': 0, 'Partial': 1, 'Noted': 2, 'Rejected': 3, 'Unclear': 4, 'No Response': 5}
        sorted_alignments.sort(key=lambda x: status_order.get(
            x['response']['status'] if x['has_response'] else 'No Response', 5
        ))
    elif sort_option == "Status (No Response first)":
        sorted_alignments.sort(key=lambda x: 0 if not x['has_response'] else 1)
    elif sort_option == "Match Confidence":
        sorted_alignments.sort(
            key=lambda x: x['response']['similarity'] if x['has_response'] else 0,
            reverse=True
        )
    
    # Display each alignment
    st.markdown("#### 📋 Detailed Results")
    
    for idx, alignment in enumerate(sorted_alignments, 1):
        rec = alignment['recommendation']
        resp = alignment['response']
        
        # Determine status icon
        if not resp:
            status_icon = "❓"
            status_text = "No Response Found"
        else:
            status = resp['status']
            if status == 'Accepted':
                status_icon = "✅"
            elif status == 'Partial':
                status_icon = "⚠️"
            elif status == 'Rejected':
                status_icon = "❌"
            elif status == 'Noted':
                status_icon = "📝"
            else:
                status_icon = "❓"
            status_text = status
        
        # Create expander title
        rec_preview = rec['text'][:80] + "..." if len(rec['text']) > 80 else rec['text']
        title = f"{status_icon} **{idx}.** {rec_preview}"
        
        with st.expander(title, expanded=(idx <= 3)):
            # Recommendation
            st.markdown("**📝 Recommendation:**")
            st.info(rec['text'])
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.caption(f"Confidence: {rec.get('confidence', 0):.0%}")
            with col2:
                st.caption(f"Method: {rec.get('method', 'unknown')}")
            
            st.markdown("---")
            
            # Response
            st.markdown(f"**📢 Government Response:** {status_icon} **{status_text}**")
            
            if resp:
                st.success(resp['response_text'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Match confidence: {resp['similarity']:.0%}")
                with col2:
                    st.caption(f"Status confidence: {resp['status_confidence']:.0%}")
                with col3:
                    st.caption(f"Source: {resp.get('source_document', 'unknown')}")
            else:
                st.warning("No matching response found in the selected documents.")
    
    # Export options
    st.markdown("---")
    st.markdown("#### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create CSV export
        export_data = []
        for idx, a in enumerate(alignments, 1):
            rec = a['recommendation']
            resp = a['response']
            export_data.append({
                'Number': idx,
                'Recommendation': rec['text'],
                'Recommendation_Confidence': rec.get('confidence', 0),
                'Response_Status': resp['status'] if resp else 'No Response',
                'Response_Text': resp['response_text'] if resp else '',
                'Match_Confidence': resp['similarity'] if resp else 0,
            })
        
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            "📥 Download CSV",
            csv,
            f"recommendation_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    with col2:
        # Generate Word document
        if st.button("📥 Generate Word Report"):
            try:
                docx_buffer = generate_docx_report(alignments, status_counts, total, with_response)
                st.download_button(
                    "📥 Download Word Report",
                    docx_buffer,
                    f"alignment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"Error generating Word document: {e}")
                # Fallback to markdown
                report = generate_markdown_report(alignments, status_counts, total, with_response)
                st.download_button(
                    "📥 Download Report (MD)",
                    report,
                    f"alignment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown"
                )


# Export
__all__ = ['render_simple_alignment_interface']
