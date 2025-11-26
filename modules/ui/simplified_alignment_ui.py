# modules/ui/simplified_alignment_ui.py
"""
🔗 Enhanced Recommendation-Response Alignment Interface
Uses sentence transformers for semantic matching between recommendations and responses.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC MATCHER WITH SENTENCE TRANSFORMERS
# =============================================================================

class RecommendationResponseMatcher:
    """
    Advanced semantic matcher using sentence transformers.
    Falls back to enhanced keyword matching if transformers unavailable.
    """
    
    def __init__(self):
        self.model = None
        self.use_transformer = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Try to initialise sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Use CPU to avoid CUDA issues on Streamlit Cloud
            device = 'cpu'
            torch.set_default_device('cpu')
            
            # Use BGE model - small (33MB) but high quality for semantic search
            # This matches the model specified in requirements.txt
            self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
            self.use_transformer = True
            logger.info("✅ Sentence transformer model loaded (BAAI/bge-small-en-v1.5)")
        except ImportError:
            logger.warning("sentence-transformers not installed, using keyword matching")
            self.use_transformer = False
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            self.use_transformer = False
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
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
        """Calculate semantic similarity between two texts"""
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
        
        # Fallback to keyword matching
        return self._keyword_similarity(text1, text2)
    
    def find_best_matches(self, recommendations: List[Dict], responses: List[Dict], 
                          top_k: int = 3) -> List[Dict]:
        """
        Find best response matches for each recommendation using semantic similarity.
        
        Returns list of alignments with top_k best matches per recommendation.
        """
        if not recommendations or not responses:
            return []
        
        # Clean response texts (extract only government response portion)
        cleaned_responses = []
        for resp in responses:
            cleaned_text = self._clean_response_text(resp['text'])
            cleaned_responses.append({
                **resp,
                'cleaned_text': cleaned_text
            })
        
        # If transformer available, use batch encoding for efficiency
        if self.use_transformer and self.model:
            return self._semantic_matching(recommendations, cleaned_responses, top_k)
        else:
            return self._keyword_matching(recommendations, cleaned_responses, top_k)
    
    def _semantic_matching(self, recommendations: List[Dict], responses: List[Dict],
                           top_k: int) -> List[Dict]:
        """Use sentence transformer for semantic matching"""
        
        # Encode all texts
        rec_texts = [r['text'] for r in recommendations]
        resp_texts = [r['cleaned_text'] for r in responses]
        
        try:
            rec_embeddings = self.encode_texts(rec_texts)
            resp_embeddings = self.encode_texts(resp_texts)
            
            if rec_embeddings is None or resp_embeddings is None:
                return self._keyword_matching(recommendations, responses, top_k)
            
            # Calculate similarity matrix
            # Normalise embeddings
            rec_norms = np.linalg.norm(rec_embeddings, axis=1, keepdims=True)
            resp_norms = np.linalg.norm(resp_embeddings, axis=1, keepdims=True)
            
            rec_embeddings_norm = rec_embeddings / rec_norms
            resp_embeddings_norm = resp_embeddings / resp_norms
            
            # Cosine similarity matrix
            similarity_matrix = np.dot(rec_embeddings_norm, resp_embeddings_norm.T)
            
            # Find best matches for each recommendation
            alignments = []
            
            for rec_idx, rec in enumerate(recommendations):
                similarities = similarity_matrix[rec_idx]
                
                # Get top_k indices
                top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra to filter
                
                best_match = None
                best_score = 0.0
                
                for resp_idx in top_indices:
                    resp = responses[resp_idx]
                    score = similarities[resp_idx]
                    
                    # Skip if too similar (self-match)
                    if self._is_self_match(rec['text'], resp['cleaned_text']):
                        continue
                    
                    # Skip if response doesn't have government language
                    if not self._has_government_response_language(resp['cleaned_text']):
                        score *= 0.5  # Penalise
                    
                    # Boost for matching recommendation numbers
                    rec_num = self._extract_rec_number(rec['text'])
                    resp_num = self._extract_rec_number(resp['cleaned_text'])
                    if rec_num and resp_num and rec_num == resp_num:
                        score += 0.3
                    
                    if score > best_score and score >= 0.3:
                        best_score = score
                        status, status_conf = self._classify_response_status(resp['cleaned_text'])
                        best_match = {
                            'response_text': resp['cleaned_text'],
                            'similarity': min(score, 1.0),
                            'status': status,
                            'status_confidence': status_conf,
                            'source_document': resp.get('source_document', 'unknown'),
                            'match_method': 'semantic'
                        }
                
                alignments.append({
                    'recommendation': rec,
                    'response': best_match,
                    'has_response': best_match is not None
                })
            
            return alignments
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return self._keyword_matching(recommendations, responses, top_k)
    
    def _keyword_matching(self, recommendations: List[Dict], responses: List[Dict],
                          top_k: int) -> List[Dict]:
        """Fallback keyword-based matching"""
        
        alignments = []
        
        for rec in recommendations:
            best_match = None
            best_score = 0.0
            
            for resp in responses:
                resp_text = resp.get('cleaned_text', resp['text'])
                
                # Skip self-matches
                if self._is_self_match(rec['text'], resp_text):
                    continue
                
                # Calculate similarity
                score = self._keyword_similarity(rec['text'], resp_text)
                
                # Boost for matching recommendation numbers
                rec_num = self._extract_rec_number(rec['text'])
                resp_num = self._extract_rec_number(resp_text)
                if rec_num and resp_num and rec_num == resp_num:
                    score += 0.4
                
                # Boost for government response language
                if self._has_government_response_language(resp_text):
                    score += 0.2
                
                if score > best_score and score >= 0.25:
                    best_score = score
                    status, status_conf = self._classify_response_status(resp_text)
                    best_match = {
                        'response_text': resp_text,
                        'similarity': min(score, 1.0),
                        'status': status,
                        'status_confidence': status_conf,
                        'source_document': resp.get('source_document', 'unknown'),
                        'match_method': 'keyword'
                    }
            
            alignments.append({
                'recommendation': rec,
                'response': best_match,
                'has_response': best_match is not None
            })
        
        return alignments
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Enhanced keyword-based similarity"""
        
        # Extract meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                      'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'may', 'might', 'must', 'shall', 'this', 'that',
                      'these', 'those', 'it', 'its', 'they', 'their', 'we', 'our'}
        
        words1 = set(re.findall(r'\b\w+\b', text1.lower())) - stop_words
        words2 = set(re.findall(r'\b\w+\b', text2.lower())) - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        base_score = intersection / union if union > 0 else 0.0
        
        # Boost for government/policy terminology
        gov_terms = {'government', 'recommendation', 'accept', 'support', 'implement',
                     'policy', 'department', 'nhs', 'england', 'health', 'care',
                     'provider', 'commissioner', 'trust', 'board', 'cqc'}
        
        gov_matches = len((words1 & words2) & gov_terms)
        gov_boost = min(gov_matches * 0.05, 0.15)
        
        return min(base_score + gov_boost, 1.0)
    
    def _clean_response_text(self, text: str) -> str:
        """Extract only the government response portion from text"""
        
        markers = [
            'Government response to recommendation',
            'government response to recommendation',
            'The government supports',
            'The government accepts',
            'The government agrees',
            'The government notes',
            'The government rejects',
            'The Government supports',
            'The Government accepts'
        ]
        
        text_lower = text.lower()
        
        for marker in markers:
            marker_lower = marker.lower()
            pos = text_lower.find(marker_lower)
            if pos > 0:
                return text[pos:]
        
        return text
    
    def _is_self_match(self, rec_text: str, resp_text: str) -> bool:
        """Check if response is identical to recommendation"""
        
        rec_clean = re.sub(r'\s+', ' ', rec_text.lower().strip())
        resp_clean = re.sub(r'\s+', ' ', resp_text.lower().strip())
        
        # Exact match
        if rec_clean == resp_clean:
            return True
        
        # One contains the other
        if len(rec_clean) > 100 and len(resp_clean) > 100:
            if rec_clean in resp_clean or resp_clean in rec_clean:
                # Allow if response has gov language before rec text
                if 'government response' in resp_clean[:100] or 'the government' in resp_clean[:100]:
                    return False
                return True
        
        # Same start
        if len(rec_clean) > 150 and len(resp_clean) > 150:
            if rec_clean[:150] == resp_clean[:150]:
                return True
        
        return False
    
    def _has_government_response_language(self, text: str) -> bool:
        """Check if text has government response indicators"""
        text_lower = text.lower()
        indicators = [
            'government response', 'the government supports', 'the government accepts',
            'the government agrees', 'the government notes', 'the government rejects',
            'we accept', 'we support', 'we agree', 'accept the recommendation',
            'support the recommendation', 'accept this recommendation'
        ]
        return any(ind in text_lower for ind in indicators)
    
    def _extract_rec_number(self, text: str) -> Optional[str]:
        """Extract recommendation number from text"""
        match = re.search(r'recommendation\s+(\d+)', text.lower())
        return match.group(1) if match else None
    
    def _classify_response_status(self, text: str) -> Tuple[str, float]:
        """Classify response as Accepted, Rejected, Partial, or Noted"""
        
        text_lower = text.lower()
        
        # Check patterns
        accepted_patterns = [
            r'\baccept(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)',
            r'\bsupport(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)',
            r'\bagree(?:s|d)?\s+(?:with\s+)?(?:this\s+|the\s+)?(?:recommendation|rec)',
            r'\bthe\s+government\s+support(?:s|ed)?\s+(?:this\s+|the\s+)?recommendation',
            r'\bwill\s+implement',
            r'\bfully\s+accept',
            r'\baccept(?:s|ed)?\s+in\s+full'
        ]
        
        rejected_patterns = [
            r'\breject(?:s|ed)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)',
            r'\bdoes?\s+not\s+accept',
            r'\bcannot\s+accept',
            r'\bdisagree(?:s|d)?'
        ]
        
        partial_patterns = [
            r'\baccept(?:s|ed)?\s+in\s+(?:part|principle)',
            r'\bpartially\s+accept',
            r'\bsupport(?:s|ed)?\s+in\s+principle',
            r'\bwill\s+consider',
            r'\bunder\s+consideration'
        ]
        
        noted_patterns = [
            r'\bnote(?:s|d)?\s+(?:this\s+|the\s+)?(?:recommendation|rec)',
            r'\backnowledge(?:s|d)?',
            r'\btake(?:s|n)?\s+note'
        ]
        
        for pattern in accepted_patterns:
            if re.search(pattern, text_lower):
                return 'Accepted', 0.9
        
        for pattern in rejected_patterns:
            if re.search(pattern, text_lower):
                return 'Rejected', 0.9
        
        for pattern in partial_patterns:
            if re.search(pattern, text_lower):
                return 'Partial', 0.8
        
        for pattern in noted_patterns:
            if re.search(pattern, text_lower):
                return 'Noted', 0.75
        
        # Fallback: check for keywords
        if any(kw in text_lower for kw in ['supports', 'support', 'accepts', 'accept']):
            if 'recommendation' in text_lower:
                return 'Accepted', 0.7
        
        return 'Unclear', 0.5


# =============================================================================
# RESPONSE EXTRACTION
# =============================================================================

def extract_response_sentences(text: str) -> List[Dict]:
    """Extract sentences that are government responses"""
    
    if not text:
        return []
    
    responses = []
    
    # Method 1: Find "Government response to recommendation N" blocks
    gov_resp_pattern = r'Government response to recommendation\s+\d+[^G]*?(?=Government response to recommendation|\Z)'
    matches = re.finditer(gov_resp_pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        resp_text = match.group().strip()
        if len(resp_text) > 50:
            # Extract recommendation number
            num_match = re.search(r'recommendation\s+(\d+)', resp_text.lower())
            rec_num = num_match.group(1) if num_match else None
            
            responses.append({
                'text': resp_text,
                'position': match.start(),
                'response_type': 'government_response',
                'confidence': 0.95,
                'rec_number': rec_num
            })
    
    # Method 2: If no structured responses found, try sentence-level extraction
    if not responses:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            
            sentence_lower = sentence.lower()
            
            # Skip if looks like recommendation
            if any(sentence_lower.startswith(p) for p in [
                'nhs england should', 'providers should', 'trusts should',
                'recommendation', 'we recommend', 'all providers should'
            ]):
                continue
            
            # Check for response language
            is_response = False
            confidence = 0.0
            
            if sentence_lower.startswith('government response') or sentence_lower.startswith('the government'):
                is_response = True
                confidence = 0.9
            elif any(term in sentence_lower for term in ['accept', 'support', 'agree', 'reject', 'note']):
                if 'recommendation' in sentence_lower or 'government' in sentence_lower:
                    is_response = True
                    confidence = 0.7
            
            if is_response:
                rec_ref = re.search(r'recommendation\s+(\d+)', sentence_lower)
                
                responses.append({
                    'text': sentence,
                    'position': idx,
                    'response_type': 'extracted',
                    'confidence': confidence,
                    'rec_number': rec_ref.group(1) if rec_ref else None
                })
    
    return responses


# =============================================================================
# MAIN INTERFACE
# =============================================================================

# Global matcher instance (cached)
@st.cache_resource
def get_matcher():
    """Get or create cached matcher instance"""
    return RecommendationResponseMatcher()


def render_simple_alignment_interface(documents: List[Dict]):
    """Render the alignment interface with semantic matching"""
    
    st.markdown("### 🔗 Find Government Responses")
    
    # Check matcher status
    matcher = get_matcher()
    if matcher.use_transformer:
        st.success("🧠 Using semantic matching (sentence-transformers)")
    else:
        st.info("📊 Using keyword matching (install sentence-transformers for better results)")
    
    # Check for recommendations
    if 'extracted_recommendations' not in st.session_state or not st.session_state.extracted_recommendations:
        st.warning("⚠️ No recommendations extracted yet!")
        st.info("""
        **How to use:**
        1. Go to the **🎯 Recommendations** tab
        2. Extract recommendations from your document
        3. Return here to find government responses
        """)
        
        # Quick extract option
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
                        st.warning("No recommendations found.")
                except Exception as e:
                    st.error(f"Error: {e}")
        return
    
    recommendations = st.session_state.extracted_recommendations
    st.success(f"✅ Using **{len(recommendations)}** recommendations")
    
    with st.expander("📋 View Recommendations", expanded=False):
        for i, rec in enumerate(recommendations[:10], 1):
            st.markdown(f"**{i}.** {rec['text'][:150]}...")
        if len(recommendations) > 10:
            st.caption(f"... and {len(recommendations) - 10} more")
    
    st.markdown("---")
    
    # Select response documents
    st.markdown("#### 📄 Select Government Response Document(s)")
    
    doc_names = [doc['filename'] for doc in documents]
    
    # Auto-detect response docs
    suggested = [n for n in doc_names if any(t in n.lower() for t in ['response', 'government', 'reply'])]
    
    resp_docs = st.multiselect(
        "Select response documents:",
        options=doc_names,
        default=suggested,
        help="Select documents containing government responses"
    )
    
    if not resp_docs:
        st.info("👆 Select at least one response document")
        return
    
    # Run alignment
    if st.button("🔗 Find Responses", type="primary"):
        
        with st.spinner("Analysing documents..."):
            
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
                return
            
            st.info(f"Found **{len(all_responses)}** potential responses")
            
            # Use semantic matcher
            progress = st.progress(0)
            progress.progress(50)
            
            alignments = matcher.find_best_matches(recommendations, all_responses)
            
            progress.progress(100)
            progress.empty()
            
            # Store results
            st.session_state.alignment_results = alignments
    
    # Display results
    if 'alignment_results' in st.session_state:
        display_alignment_results(st.session_state.alignment_results)


def display_alignment_results(alignments: List[Dict]):
    """Display alignment results"""
    
    st.markdown("---")
    st.markdown("### 📊 Alignment Results")
    
    # Statistics
    total = len(alignments)
    with_response = sum(1 for a in alignments if a['has_response'])
    
    status_counts = Counter()
    for a in alignments:
        if a['has_response'] and a['response']:
            status_counts[a['response']['status']] += 1
        else:
            status_counts['No Response'] += 1
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total)
    col2.metric("✅ Accepted", status_counts.get('Accepted', 0))
    col3.metric("⚠️ Partial", status_counts.get('Partial', 0))
    col4.metric("❌ Rejected", status_counts.get('Rejected', 0))
    col5.metric("❓ No Response", status_counts.get('No Response', 0) + status_counts.get('Unclear', 0))
    
    st.markdown("---")
    
    # Sort options
    sort_option = st.selectbox(
        "Sort by:",
        ["Original Order", "Status (Accepted first)", "Status (No Response first)", "Match Confidence"]
    )
    
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
    
    # Display details
    st.markdown("#### 📋 Detailed Results")
    
    for idx, alignment in enumerate(sorted_alignments, 1):
        rec = alignment['recommendation']
        resp = alignment['response']
        
        if not resp:
            status_icon = "❓"
            status_text = "No Response Found"
        else:
            status = resp['status']
            status_icon = {'Accepted': '✅', 'Partial': '⚠️', 'Rejected': '❌', 'Noted': '📝'}.get(status, '❓')
            status_text = status
        
        rec_preview = rec['text'][:80] + "..." if len(rec['text']) > 80 else rec['text']
        
        with st.expander(f"{status_icon} **{idx}.** {rec_preview}", expanded=(idx <= 3)):
            st.markdown("**📝 Recommendation:**")
            st.info(rec['text'])
            
            col1, col2 = st.columns(2)
            col1.caption(f"Confidence: {rec.get('confidence', 0):.0%}")
            col2.caption(f"Method: {rec.get('method', 'unknown')}")
            
            st.markdown("---")
            st.markdown(f"**📢 Government Response:** {status_icon} **{status_text}**")
            
            if resp:
                st.success(resp['response_text'])
                
                col1, col2, col3 = st.columns(3)
                col1.caption(f"Match: {resp['similarity']:.0%}")
                col2.caption(f"Method: {resp.get('match_method', 'unknown')}")
                col3.caption(f"Source: {resp.get('source_document', 'unknown')}")
            else:
                st.warning("No matching response found.")
    
    # Export
    st.markdown("---")
    st.markdown("#### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
                'Match_Method': resp.get('match_method', '') if resp else ''
            })
        
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            "📥 Download CSV",
            csv,
            f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    with col2:
        # Generate report
        report = f"""# Recommendation-Response Alignment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Recommendations: {total}
- Responses Found: {with_response}
- Accepted: {status_counts.get('Accepted', 0)}
- Partial: {status_counts.get('Partial', 0)}
- Rejected: {status_counts.get('Rejected', 0)}
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
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            "text/markdown"
        )


# Export
__all__ = ['render_simple_alignment_interface']
