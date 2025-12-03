# modules/ui/alignment_ui.py
"""
🔗 Enhanced Recommendation-Response Alignment Interface
Uses sentence transformers for semantic matching between recommendations and responses.

v2.1 - Fixed response extraction bleeding issue
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
        """Use sentence transformer for semantic matching with number-based priority"""
        
        # First, create a map of responses by recommendation number
        responses_by_number = {}
        for resp in responses:
            if resp.get('rec_number'):
                num = resp['rec_number']
                if num not in responses_by_number:
                    responses_by_number[num] = []
                responses_by_number[num].append(resp)
        
        # Encode all texts for semantic matching
        rec_texts = [r['text'] for r in recommendations]
        resp_texts = [r['cleaned_text'] for r in responses]
        
        try:
            rec_embeddings = self.encode_texts(rec_texts)
            resp_embeddings = self.encode_texts(resp_texts)
            
            if rec_embeddings is None or resp_embeddings is None:
                return self._keyword_matching(recommendations, responses, top_k)
            
            # Normalise embeddings for cosine similarity
            rec_norms = np.linalg.norm(rec_embeddings, axis=1, keepdims=True)
            resp_norms = np.linalg.norm(resp_embeddings, axis=1, keepdims=True)
            
            rec_embeddings_norm = rec_embeddings / rec_norms
            resp_embeddings_norm = resp_embeddings / resp_norms
            
            similarity_matrix = np.dot(rec_embeddings_norm, resp_embeddings_norm.T)
            
            alignments = []
            
            for rec_idx, rec in enumerate(recommendations):
                best_match = None
                best_score = 0.0
                
                # PRIORITY 1: Try to match by recommendation number
                rec_num = self._extract_rec_number(rec['text'])
                
                if rec_num and rec_num in responses_by_number:
                    # Found responses explicitly for this recommendation number
                    for resp in responses_by_number[rec_num]:
                        resp_text = resp['cleaned_text']
                        
                        # Skip self-matches
                        if self._is_self_match(rec['text'], resp_text):
                            continue
                        
                        # Get semantic score for ranking multiple responses
                        resp_idx = next((i for i, r in enumerate(responses) if r is resp), None)
                        score = similarity_matrix[rec_idx][resp_idx] if resp_idx is not None else 0.7
                        
                        # Strong boost for number match
                        score = min(score + 0.4, 1.0)
                        
                        if score > best_score:
                            best_score = score
                            status, status_conf = self._classify_response_status(resp_text)
                            best_match = {
                                'response_text': resp_text,
                                'similarity': score,
                                'status': status,
                                'status_confidence': status_conf,
                                'source_document': resp.get('source_document', 'unknown'),
                                'match_method': 'number_match'
                            }
                
                # PRIORITY 2: If no number match, use semantic similarity
                if best_match is None:
                    similarities = similarity_matrix[rec_idx]
                    top_indices = np.argsort(similarities)[::-1][:top_k * 2]
                    
                    for resp_idx in top_indices:
                        resp = responses[resp_idx]
                        score = similarities[resp_idx]
                        
                        # Skip if too low
                        if score < 0.4:
                            continue
                        
                        # Skip self-matches
                        if self._is_self_match(rec['text'], resp['cleaned_text']):
                            continue
                        
                        # Skip if it's recommendation text
                        if is_recommendation_text(resp['cleaned_text']):
                            continue
                        
                        # Must have government response language
                        if not self._has_government_response_language(resp['cleaned_text']):
                            score *= 0.7  # Penalise
                        
                        if score > best_score:
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
# RESPONSE EXTRACTION - STRICT FILTERING (v2.1 FIX)
# =============================================================================

def is_recommendation_text(text: str) -> bool:
    """
    Check if text looks like a recommendation (should be excluded from responses).
    Recommendations use imperative language with 'should'.
    """
    text_lower = text.lower().strip()
    
    # Patterns that indicate this is recommendation text, NOT a response
    recommendation_starters = [
        r'^nhs\s+england\s+should',
        r'^providers?\s+should',
        r'^trusts?\s+should',
        r'^boards?\s+should',
        r'^icss?\s+should',
        r'^ics\s+and\s+provider',
        r'^cqc\s+should',
        r'^dhsc\s+should',
        r'^dhsc,?\s+in\s+partnership',
        r'^every\s+provider',
        r'^all\s+providers?\s+should',
        r'^provider\s+boards?\s+should',
        r'^this\s+multi-professional\s+alliance\s+should',
        r'^the\s+review\s+should',
        r'^commissioners?\s+should',
        r'^regulators?\s+should',
        r'^recommendation\s+\d+\s+[a-z]',
        r'^we\s+recommend',
        r'^it\s+should\s+also',
        r'^they\s+should',
        r'^this\s+forum\s+should',
        r'^this\s+programme\s+should',
        r'^these\s+systems\s+should',
        r'^the\s+digital\s+platforms',
        r'^the\s+output\s+of\s+the',
        r'^including,?\s+where\s+appropriate',
        r'^to\s+facilitate\s+this',
    ]
    
    for pattern in recommendation_starters:
        if re.search(pattern, text_lower):
            return True
    
    return False


def is_genuine_response(text: str) -> bool:
    """
    Check if text looks like a genuine government response.
    Must START with response language, not just contain it.
    """
    text_lower = text.lower().strip()
    
    # Strong indicators - text STARTS with these
    strong_starters = [
        r'^government\s+response',
        r'^the\s+government\s+supports?',
        r'^the\s+government\s+accepts?',
        r'^the\s+government\s+agrees?',
        r'^the\s+government\s+notes?',
        r'^the\s+government\s+rejects?',
        r'^the\s+government\s+recognises?',
        r'^the\s+government\s+is\s+committed',
        r'^the\s+government\s+will',
        r'^we\s+support',
        r'^we\s+accept',
        r'^we\s+agree',
        r'^dhsc\s+and\s+nhs\s+england\s+support',
        r'^nhs\s+england\s+will',
    ]
    
    for pattern in strong_starters:
        if re.search(pattern, text_lower):
            return True
    
    return False


def has_pdf_artifacts(text: str) -> bool:
    """Check if text contains PDF extraction artifacts"""
    artifacts = [
        r'\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?',
        r'GOV\.UK',
        r'https?://www\.gov\.uk',
        r'https?://www\.england\.nhs\.uk',
        r'\d+/\d+\s*$',
    ]
    for pattern in artifacts:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def clean_pdf_artifacts(text: str) -> str:
    """Remove PDF artifacts from text"""
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)?\s*', '', text)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'Government response to the rapid review.*?GOV\.UK[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_response_sentences(text: str) -> List[Dict]:
    """
    Extract sentences that are government responses.
    Uses strict filtering to exclude recommendation text.
    
    v2.1 FIX: Properly stop extraction at next recommendation boundary.
    """
    
    if not text:
        return []
    
    # Clean PDF artifacts first
    text = clean_pdf_artifacts(text)
    
    responses = []
    seen_responses = set()
    
    # ==========================================================================
    # Method 1: Find structured "Government response to recommendation N" blocks
    # ==========================================================================
    # 
    # CRITICAL FIX: The document structure is:
    #   "Government response to recommendation 1 [response text] Recommendation 2 [rec text] 
    #    Government response to recommendation 2 [response text] Recommendation 3 ..."
    #
    # We need to stop at EITHER:
    #   - Next "Government response to recommendation"
    #   - Next "Recommendation N " followed by entity (NHS, Provider, etc.) or action verb
    #
    # The key insight is that quoted recommendations in responses start with 
    # "Recommendation N [Entity] should" while section headers are different
    
    # First, find all "Government response to recommendation N" positions
    gov_resp_starts = []
    for match in re.finditer(r'Government\s+response\s+to\s+recommendation\s+(\d+)', text, re.IGNORECASE):
        gov_resp_starts.append({
            'pos': match.start(),
            'end': match.end(),
            'rec_num': match.group(1),
            'full_match': match.group(0)
        })
    
    # Find all "Recommendation N" positions (quoted recommendations in response doc)
    rec_positions = []
    for match in re.finditer(r'Recommendation\s+(\d+)\s+([A-Z])', text):
        rec_positions.append({
            'pos': match.start(),
            'rec_num': match.group(1)
        })
    
    logger.info(f"Found {len(gov_resp_starts)} government response headers")
    logger.info(f"Found {len(rec_positions)} recommendation markers")
    
    # For each government response, find where it ends
    for i, gov_resp in enumerate(gov_resp_starts):
        start_pos = gov_resp['end']  # Start after the header
        rec_num = gov_resp['rec_num']
        
        # Find the end position: next gov response OR next recommendation marker
        end_pos = len(text)  # Default to end of text
        
        # Check for next government response
        if i + 1 < len(gov_resp_starts):
            next_gov = gov_resp_starts[i + 1]['pos']
            if next_gov < end_pos:
                end_pos = next_gov
        
        # Check for recommendation markers that come AFTER this response starts
        # but BEFORE the next government response
        for rec_pos in rec_positions:
            if rec_pos['pos'] > start_pos and rec_pos['pos'] < end_pos:
                # This is a quoted recommendation - stop here
                end_pos = rec_pos['pos']
                break
        
        # Extract the response content
        resp_content = text[start_pos:end_pos].strip()
        
        # Clean up
        resp_content = clean_pdf_artifacts(resp_content)
        
        # Skip if too short
        if len(resp_content) < 30:
            continue
        
        # Build the full response text
        resp_text = f"Government response to recommendation {rec_num} {resp_content}"
        
        # Skip duplicates
        resp_key = resp_text[:100].lower()
        if resp_key in seen_responses:
            continue
        seen_responses.add(resp_key)
        
        # Final validation: ensure no recommendation text leaked through
        # Check if response contains "Recommendation N [A-Z]" pattern
        leaked_rec = re.search(r'Recommendation\s+\d+\s+[A-Z][a-z]', resp_text)
        if leaked_rec:
            # Cut off at that point
            resp_text = resp_text[:leaked_rec.start()].strip()
        
        if len(resp_text) > 50:
            responses.append({
                'text': resp_text,
                'position': gov_resp['pos'],
                'response_type': 'structured',
                'confidence': 0.95,
                'rec_number': rec_num
            })
            logger.debug(f"Extracted response for rec {rec_num}: {len(resp_text)} chars")
    
    # ==========================================================================
    # Method 2: Fallback - Find standalone response sentences
    # ==========================================================================
    if len(responses) < 5:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            sentence = clean_pdf_artifacts(sentence)
            
            if len(sentence) < 40 or len(sentence) > 500:
                continue
            
            if is_recommendation_text(sentence):
                continue
            
            if not is_genuine_response(sentence):
                continue
            
            resp_key = sentence[:100].lower()
            if resp_key in seen_responses:
                continue
            seen_responses.add(resp_key)
            
            rec_ref = re.search(r'recommendation\s+(\d+)', sentence.lower())
            
            responses.append({
                'text': sentence,
                'position': idx,
                'response_type': 'sentence',
                'confidence': 0.85,
                'rec_number': rec_ref.group(1) if rec_ref else None
            })
    
    logger.info(f"Extracted {len(responses)} genuine responses (no bleeding)")
    return responses


# =============================================================================
# MAIN INTERFACE
# =============================================================================

@st.cache_resource
def get_matcher():
    """Get or create cached matcher instance"""
    return RecommendationResponseMatcher()


def render_simple_alignment_interface(documents: List[Dict]):
    """Render the alignment interface with semantic matching"""
    
    st.markdown("### 🔗 Match Recommendations to Responses")
    
    matcher = get_matcher()
    if matcher.use_transformer:
        st.success("🧠 Using semantic matching (sentence-transformers)")
    else:
        st.info("📊 Using keyword matching (install sentence-transformers for better results)")
    
    if not documents:
        st.warning("📁 Please upload documents first in the Upload tab.")
        return
    
    doc_names = [doc['filename'] for doc in documents]
    
    st.markdown("---")
    
    # Step 1: Select document(s)
    st.markdown("#### 📄 Step 1: Select Documents")
    
    doc_mode = st.radio(
        "Are recommendations and responses in the same document?",
        ["Same document", "Different documents"],
        horizontal=True,
        help="Select 'Same document' if both recommendations and responses are in one file"
    )
    
    if doc_mode == "Same document":
        selected_doc = st.selectbox(
            "Select document containing both recommendations and responses:",
            doc_names,
            key="single_doc_select"
        )
        rec_docs = [selected_doc]
        resp_docs = [selected_doc]
    else:
        col1, col2 = st.columns(2)
        with col1:
            rec_doc = st.selectbox(
                "📋 Recommendation document:",
                doc_names,
                key="rec_doc_select"
            )
            rec_docs = [rec_doc]
        with col2:
            suggested = [n for n in doc_names if any(t in n.lower() for t in ['response', 'reply', 'answer'])]
            default_resp = suggested[0] if suggested else doc_names[0]
            
            resp_doc = st.selectbox(
                "📢 Response document:",
                doc_names,
                index=doc_names.index(default_resp) if default_resp in doc_names else 0,
                key="resp_doc_select"
            )
            resp_docs = [resp_doc]
    
    st.markdown("---")
    
    # Step 2: Extract and Match
    st.markdown("#### 🔍 Step 2: Extract & Match")
    
    if st.button("🚀 Extract Recommendations & Find Responses", type="primary"):
        
        rec_text = ""
        for doc_name in rec_docs:
            doc = next((d for d in documents if d['filename'] == doc_name), None)
            if doc and 'text' in doc:
                rec_text += doc['text'] + "\n\n"
        
        resp_text = ""
        for doc_name in resp_docs:
            doc = next((d for d in documents if d['filename'] == doc_name), None)
            if doc and 'text' in doc:
                resp_text += doc['text'] + "\n\n"
        
        if not rec_text:
            st.error("Could not read recommendation document")
            return
        
        if not resp_text:
            st.error("Could not read response document")
            return
        
        progress = st.progress(0, text="Extracting recommendations...")
        
        try:
            from modules.recommendation_extractor import extract_recommendations
            recommendations = extract_recommendations(rec_text, min_confidence=0.75)
            
            if not recommendations:
                st.warning("⚠️ No recommendations found in the document.")
                progress.empty()
                return
            
            progress.progress(33, text=f"Found {len(recommendations)} recommendations. Extracting responses...")
            
            responses = extract_response_sentences(resp_text)
            
            if not responses:
                st.warning("⚠️ No response patterns found in the document.")
                progress.empty()
                return
            
            progress.progress(66, text=f"Found {len(responses)} responses. Matching...")
            
            alignments = matcher.find_best_matches(recommendations, responses)
            
            progress.progress(100, text="Complete!")
            progress.empty()
            
            st.session_state.alignment_results = alignments
            st.session_state.extracted_recommendations = recommendations
            
            st.success(f"✅ Matched {len(recommendations)} recommendations with {len(responses)} responses")
            
        except Exception as e:
            progress.empty()
            st.error(f"Error: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            return
    
    if 'alignment_results' in st.session_state and st.session_state.alignment_results:
        display_alignment_results(st.session_state.alignment_results)


def display_alignment_results(alignments: List[Dict]):
    """Display alignment results"""
    
    st.markdown("---")
    st.markdown("### 📊 Alignment Results")
    
    total = len(alignments)
    with_response = sum(1 for a in alignments if a['has_response'])
    
    status_counts = Counter()
    for a in alignments:
        if a['has_response'] and a['response']:
            status_counts[a['response']['status']] += 1
        else:
            status_counts['No Response'] += 1
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total)
    col2.metric("✅ Accepted", status_counts.get('Accepted', 0))
    col3.metric("⚠️ Partial", status_counts.get('Partial', 0))
    col4.metric("❌ Rejected", status_counts.get('Rejected', 0))
    col5.metric("❓ No Response", status_counts.get('No Response', 0) + status_counts.get('Unclear', 0))
    
    st.markdown("---")
    
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


__all__ = ['render_simple_alignment_interface']
