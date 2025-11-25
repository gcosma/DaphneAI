# modules/ui/simplified_alignment_ui.py - FIXED VERSION
"""
🔗 Simplified Recommendation-Response Alignment Interface
FIXED: Prevents recommendations from being classified as responses
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
# RECOMMENDATION EXCLUSION PATTERNS (NEW - CRITICAL FIX)
# =============================================================================

# Patterns that indicate this is a RECOMMENDATION, not a response
RECOMMENDATION_EXCLUSION_PATTERNS = [
    r'^recommendation\s+\d+',  # Starts with "Recommendation N"
    r'we\s+recommend',
    r'committee\s+recommends?',
    r'inquiry\s+recommends?',
    r'report\s+recommends?',
    r'it\s+is\s+recommended',
    r'our\s+recommendation',
    r'should\s+(?:be\s+)?(?:consider|implement|establish|develop)',  # Recommendation language
    r'must\s+(?:be\s+)?(?:consider|implement|establish|develop)',
    r'ought\s+to\s+(?:consider|implement|establish)',
]

# Strong indicators that this IS a response (government speaking)
STRONG_RESPONSE_INDICATORS = [
    r'government\s+(?:accepts?|rejects?|agrees?|notes?)',
    r'the\s+government\s+(?:will|has|is)',
    r'we\s+(?:accept|reject|agree|note)\s+(?:this\s+)?recommendation',
    r'(?:accept|reject)(?:s|ed)?\s+in\s+(?:full|part|principle)',
    r'government\s+response\s+to\s+recommendation',
    r'in\s+response\s+to\s+(?:this\s+)?recommendation',
    r'department\s+(?:accepts?|rejects?|agrees?)',
]


# =============================================================================
# RESPONSE EXTRACTION (FIXED VERSION)
# =============================================================================

def extract_response_sentences(text: str, extracted_recommendations: List[Dict] = None) -> List[Dict]:
    """
    Extract sentences that look like responses to recommendations
    FIXED: Now filters out sentences that are actually recommendations
    """
    
    if not text:
        return []
    
    # Create a set of recommendation texts to exclude
    known_recommendation_texts = set()
    if extracted_recommendations:
        for rec in extracted_recommendations:
            rec_text = rec.get('text', '').strip().lower()
            if rec_text:
                known_recommendation_texts.add(rec_text)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    responses = []
    current_rec_number = None
    
    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 30:
            continue
        
        sentence_lower = sentence.lower()
        
        # =================================================================
        # CRITICAL FIX 1: Skip if this is an extracted recommendation
        # =================================================================
        if sentence_lower in known_recommendation_texts:
            logger.info(f"Skipping extracted recommendation: {sentence[:50]}...")
            continue
        
        # Check for partial match with extracted recommendations (95% similarity)
        for known_rec in known_recommendation_texts:
            if len(known_rec) > 50 and len(sentence_lower) > 50:
                # Compare first 100 chars
                if known_rec[:100] == sentence_lower[:100]:
                    logger.info(f"Skipping similar to extracted recommendation: {sentence[:50]}...")
                    continue
        
        # =================================================================
        # CRITICAL FIX 2: Exclude sentences that are RECOMMENDATIONS
        # =================================================================
        is_recommendation = False
        for pattern in RECOMMENDATION_EXCLUSION_PATTERNS:
            if re.search(pattern, sentence_lower):
                is_recommendation = True
                logger.debug(f"Excluded as recommendation (pattern: {pattern}): {sentence[:50]}...")
                break
        
        if is_recommendation:
            continue
        
        # =================================================================
        # CRITICAL FIX 3: Only include if it has STRONG response indicators
        # =================================================================
        has_strong_response_indicator = False
        for pattern in STRONG_RESPONSE_INDICATORS:
            if re.search(pattern, sentence_lower):
                has_strong_response_indicator = True
                break
        
        # If it doesn't have strong indicators, skip it
        if not has_strong_response_indicator:
            # Check for weaker indicators but ONLY if it also has response context
            weak_indicators = ['accept', 'reject', 'agree', 'note', 'implement']
            has_weak = any(word in sentence_lower for word in weak_indicators)
            has_context = any(phrase in sentence_lower for phrase in 
                            ['government', 'department', 'ministry', 'in response'])
            
            if not (has_weak and has_context):
                continue
        
        # =================================================================
        # Detect "Government response to recommendation N" headers
        # =================================================================
        gov_resp_match = re.search(r'government\s+response\s+to\s+recommendation\s+(\d+)', sentence_lower)
        if gov_resp_match:
            current_rec_number = gov_resp_match.group(1)
        
        # =================================================================
        # Classify the response
        # =================================================================
        is_response = True
        response_type = 'government_response'
        confidence = 0.9  # High confidence since we passed strict filters
        
        # Check for specific response patterns
        for status, patterns in RESPONSE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    response_type = status
                    break
        
        # Get recommendation number reference
        rec_ref = re.search(r'recommendation\s+(\d+)', sentence_lower)
        detected_rec_number = rec_ref.group(1) if rec_ref else current_rec_number
        
        responses.append({
            'text': sentence,
            'position': idx,
            'response_type': response_type,
            'confidence': confidence,
            'rec_number': detected_rec_number
        })
    
    logger.info(f"Extracted {len(responses)} responses after filtering out recommendations")
    return responses


# =============================================================================
# RESPONSE PATTERNS (Keep existing)
# =============================================================================

RESPONSE_PATTERNS = {
    'accepted': [
        r'\baccept(?:s|ed|ing)?\s+(?:this\s+)?(?:recommendation|rec)\b',
        r'\baccept(?:s|ed|ing)?\s+in\s+full\b',
        r'\bfully\s+accept(?:s|ed)?\b',
        r'\bagree(?:s|d)?\s+(?:with\s+)?(?:this\s+)?(?:recommendation|rec)\b',
        r'\bwill\s+implement\b',
        r'\bcommit(?:s|ted)?\s+to\s+implement\b',
        r'\bendorse(?:s|d)?\b',
        r'\bsupport(?:s|ed)?\s+(?:this\s+)?(?:recommendation|rec)\b',
    ],
    'rejected': [
        r'\breject(?:s|ed|ing)?\s+(?:this\s+)?(?:recommendation|rec)\b',
        r'\bdoes?\s+not\s+accept\b',
        r'\bcannot\s+accept\b',
        r'\bdecline(?:s|d)?\s+(?:to\s+accept)?\b',
        r'\bdo(?:es)?\s+not\s+agree\b',
        r'\bdisagree(?:s|d)?\b',
        r'\bnot\s+(?:be\s+)?implement(?:ed|ing)?\b',
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
    ],
    'noted': [
        r'\bnote(?:s|d)?\s+(?:this\s+)?(?:recommendation|rec)\b',
        r'\backnowledge(?:s|d)?\b',
        r'\btake(?:s|n)?\s+note\b',
        r'\bwill\s+(?:review|examine|look\s+at)\b',
    ]
}


def classify_response_status(response_text: str) -> Tuple[str, float]:
    """Classify a response as Accepted, Rejected, Partial, or Noted"""
    
    text_lower = response_text.lower()
    
    # Check patterns in order of specificity
    for status, patterns in RESPONSE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                confidence = 0.9 if status in ['accepted', 'rejected'] else 0.75
                return status.title(), confidence
    
    return 'Unclear', 0.5


# =============================================================================
# SEMANTIC MATCHING (Keep existing but improve)
# =============================================================================

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using keyword overlap"""
    
    if not text1 or not text2:
        return 0.0
    
    # Clean and tokenize
    def get_keywords(text):
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'then', 'once',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                     'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                     'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
                     'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her'}
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return set(w for w in words if w not in stopwords)
    
    keywords1 = get_keywords(text1)
    keywords2 = get_keywords(text2)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    
    return intersection / union if union > 0 else 0.0


def find_best_response(recommendation: Dict, responses: List[Dict], 
                       min_similarity: float = 0.15) -> Dict:
    """Find the best matching response for a recommendation"""
    
    rec_text = recommendation.get('text', '')
    rec_number = None
    
    # Check if recommendation has a number
    num_match = re.search(r'Recommendation\s+(\d+)', rec_text, re.IGNORECASE)
    if num_match:
        rec_number = num_match.group(1)
    
    best_match = None
    best_score = 0.0
    
    for response in responses:
        resp_text = response.get('text', '')
        
        # =================================================================
        # ENHANCED SELF-MATCH PREVENTION
        # =================================================================
        if is_self_match(rec_text, resp_text):
            logger.debug(f"Prevented self-match between recommendation and response")
            continue
        
        # Calculate similarity
        similarity = calculate_similarity(rec_text, resp_text)
        
        # Boost if response references the same recommendation number
        if rec_number and response.get('rec_number') == rec_number:
            similarity += 0.3
        
        # Boost if response contains actual response language
        response_boost = 0
        for term in ['government response', 'accept', 'reject', 'agree', 'support', 
                     'the government will', 'we will', 'implement']:
            if term in resp_text.lower():
                response_boost += 0.1
        similarity += min(response_boost, 0.3)
        
        # Penalize if response looks like it's just quoting the recommendation
        if similarity > 0.8 and not any(term in resp_text.lower() for term in 
                                        ['accept', 'reject', 'agree', 'support', 'government']):
            similarity *= 0.3  # Heavy penalty for high similarity without response language
        
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


# =============================================================================
# MAIN INTERFACE (UPDATED)
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
    
    # Show summary
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
    
    # Auto-detect response documents
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
        
        with st.spinner("Analyzing documents for responses (filtering out recommendations)..."):
            
            # Extract responses from selected documents - NOW WITH FILTERING
            all_responses = []
            for doc_name in resp_docs:
                doc = next((d for d in documents if d['filename'] == doc_name), None)
                if doc and 'text' in doc:
                    # FIXED: Pass extracted recommendations to filter them out
                    doc_responses = extract_response_sentences(
                        doc['text'], 
                        extracted_recommendations=recommendations
                    )
                    for resp in doc_responses:
                        resp['source_document'] = doc_name
                    all_responses.extend(doc_responses)
            
            if not all_responses:
                st.warning("⚠️ No response patterns found in selected documents.")
                st.info("""
                This might mean:
                - The document format is different than expected
                - It's not a government response document
                - All responses were filtered out as recommendations
                
                Try selecting a different document.
                """)
                return
            
            st.info(f"Found **{len(all_responses)}** genuine response sentences (after filtering out {len(recommendations)} recommendations)")
            
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
            
            # Show summary
            with_response = sum(1 for a in alignments if a['has_response'])
            st.success(f"✅ Alignment complete! Found responses for {with_response} out of {len(alignments)} recommendations")
    
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
