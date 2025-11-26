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
        
        # Check for response patterns
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
        # Check if response is contained in recommendation or vice versa
        if rec_clean in resp_clean or resp_clean in rec_clean:
            return True
    
    # Check 3: Very high word overlap without response language
    words_rec = set(re.findall(r'\b\w+\b', rec_clean))
    words_resp = set(re.findall(r'\b\w+\b', resp_clean))
    
    if len(words_rec) > 10 and len(words_resp) > 10:
        overlap = len(words_rec & words_resp)
        overlap_ratio = overlap / min(len(words_rec), len(words_resp))
        
        # If >80% overlap and no response indicators, it's a self-match
        if overlap_ratio > 0.8:
            response_indicators = ['accept', 'reject', 'agree', 'support', 'government', 
                                   'response', 'noted', 'implemented', 'will']
            has_response_lang = any(ind in resp_clean for ind in response_indicators)
            if not has_response_lang:
                return True
    
    return False


def find_best_response_match(recommendation: Dict, responses: List[Dict], 
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
                'response_type': response.get('response_type', 'unknown')
            }
    
    return best_match


# =============================================================================
# MAIN ALIGNMENT FUNCTION
# =============================================================================

def perform_alignment(recommendations: List[Dict], response_documents: List[Dict],
                      min_similarity: float = 0.15) -> List[Dict]:
    """
    Align recommendations with responses from separate documents
    
    Args:
        recommendations: List of extracted recommendations
        response_documents: List of document dicts with 'text' key
        min_similarity: Minimum similarity threshold for matching
        
    Returns:
        List of alignments with recommendation-response pairs
    """
    
    # Extract all response sentences from response documents
    all_responses = []
    for doc in response_documents:
        doc_text = doc.get('text', '')
        responses = extract_response_sentences(doc_text)
        all_responses.extend(responses)
    
    logger.info(f"Extracted {len(all_responses)} response sentences from {len(response_documents)} documents")
    
    # Match each recommendation with best response
    alignments = []
    
    for rec in recommendations:
        best_match = find_best_response_match(rec, all_responses, min_similarity)
        
        alignments.append({
            'recommendation': rec,
            'response': best_match,
            'has_response': best_match is not None
        })
    
    return alignments


# =============================================================================
# UI RENDERING
# =============================================================================

def render_simple_alignment_interface():
    """Render the simplified alignment interface"""
    
    st.header("🔗 Recommendation-Response Alignment")
    
    st.markdown("""
    This tool matches recommendations with government responses from separate documents.
    
    **How it works:**
    1. Extract recommendations in the **Recommendations** tab first
    2. Upload government response documents here
    3. The system will automatically match and classify responses
    """)
    
    # Check for extracted recommendations
    if 'extracted_recommendations' not in st.session_state or not st.session_state.extracted_recommendations:
        st.warning("⚠️ No recommendations found. Please extract recommendations in the **Recommendations** tab first.")
        return
    
    recommendations = st.session_state.extracted_recommendations
    st.success(f"✅ Found {len(recommendations)} recommendations ready for alignment")
    
    st.markdown("---")
    
    # Response document upload
    st.subheader("📄 Upload Response Documents")
    
    response_files = st.file_uploader(
        "Upload government response documents (PDF, TXT, DOCX)",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        key="response_docs"
    )
    
    if not response_files:
        st.info("Please upload one or more government response documents to begin alignment.")
        return
    
    # Process response documents
    response_documents = []
    for f in response_files:
        # Simple text extraction - in production, use proper PDF/DOCX parsing
        try:
            if f.name.endswith('.txt'):
                text = f.read().decode('utf-8')
            else:
                # For PDF/DOCX, you'd use appropriate libraries
                text = f.read().decode('utf-8', errors='ignore')
            
            response_documents.append({
                'filename': f.name,
                'text': text
            })
        except Exception as e:
            st.error(f"Error reading {f.name}: {e}")
    
    st.success(f"✅ Loaded {len(response_documents)} response documents")
    
    # Alignment settings
    st.subheader("⚙️ Alignment Settings")
    
    min_similarity = st.slider(
        "Minimum match confidence",
        min_value=0.1,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="Lower = more matches but potentially less accurate"
    )
    
    # Run alignment
    if st.button("🔍 Run Alignment", type="primary"):
        with st.spinner("Matching recommendations with responses..."):
            alignments = perform_alignment(
                recommendations,
                response_documents,
                min_similarity
            )
            st.session_state.alignments = alignments
    
    # Display results
    if 'alignments' in st.session_state and st.session_state.alignments:
        display_alignment_results(st.session_state.alignments)


def display_alignment_results(alignments: List[Dict]):
    """Display alignment results with statistics and details"""
    
    st.markdown("---")
    st.subheader("📊 Alignment Results")
    
    # Calculate statistics
    total = len(alignments)
    with_response = sum(1 for a in alignments if a['has_response'])
    
    status_counts = {}
    for a in alignments:
        if a['has_response']:
            status = a['response']['status']
        else:
            status = 'No Response'
        status_counts[status] = status_counts.get(status, 0) + 1
    
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
            status_color = "gray"
        else:
            status = resp['status']
            if status == 'Accepted':
                status_icon = "✅"
                status_color = "green"
            elif status == 'Partial':
                status_icon = "⚠️"
                status_color = "orange"
            elif status == 'Rejected':
                status_icon = "❌"
                status_color = "red"
            elif status == 'Noted':
                status_icon = "📝"
                status_color = "blue"
            else:
                status_icon = "❓"
                status_color = "gray"
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
                    st.caption(f"Response type: {resp['response_type']}")
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
        # Summary report
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
        
        st.download_button(
            "📥 Download Report",
            report,
            f"alignment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            "text/markdown"
        )


# Export
__all__ = ['render_simple_alignment_interface']
