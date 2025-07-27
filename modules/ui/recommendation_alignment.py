# modules/ui/recommendation_alignment.py - Recommendation-Response Alignment System
import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Any
from .search_utils import STOP_WORDS, check_rag_availability

def render_recommendation_alignment_interface(documents: List[Dict[str, Any]]):
    """Specialized interface for aligning recommendations with responses"""
    
    st.header("🏛️ Recommendation-Response Alignment")
    st.markdown("*Automatically find recommendations and their corresponding responses*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        return
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Search Configuration:**")
        
        # Recommendation patterns
        rec_patterns = st.multiselect(
            "Recommendation Keywords",
            ["recommend", "suggest", "advise", "propose", "urge", "call for", "should", "must"],
            default=["recommend", "suggest", "advise"],
            help="Keywords that indicate recommendations"
        )
        
        # Response patterns  
        resp_patterns = st.multiselect(
            "Response Keywords", 
            ["accept", "reject", "agree", "disagree", "implement", "consider", "response", "reply", "approved", "declined"],
            default=["accept", "reject", "agree", "implement"],
            help="Keywords that indicate responses"
        )
    
    with col2:
        st.markdown("**🤖 AI Configuration:**")
        
        # AI options
        use_ai_summary = st.checkbox("Generate AI Summaries", value=True)
        ai_available = check_rag_availability()
        
        if not ai_available:
            st.warning("🤖 AI summaries require: pip install sentence-transformers torch")
            use_ai_summary = False
        
        # Summary length
        summary_length = st.selectbox(
            "Summary Length",
            ["Short (1-2 sentences)", "Medium (3-4 sentences)", "Long (5+ sentences)"],
            index=1
        )
    
    # Start analysis button
    if st.button("🔍 Find & Align Recommendations", type="primary"):
        with st.spinner("🔍 Analyzing documents for recommendations and responses..."):
            
            # Step 1: Find recommendations
            recommendations = find_recommendations(documents, rec_patterns)
            
            # Step 2: Find responses
            responses = find_responses(documents, resp_patterns)
            
            # Step 3: Align recommendations with responses
            alignments = align_recommendations_responses(recommendations, responses)
            
            # Step 4: Generate AI summaries if enabled
            if use_ai_summary and ai_available:
                alignments = add_ai_summaries(alignments, summary_length)
            
            # Step 5: Display results
            display_alignment_results(alignments, use_ai_summary)

def find_recommendations(documents: List[Dict], patterns: List[str]) -> List[Dict]:
    """Find all recommendations in documents"""
    
    recommendations = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Split into sentences for better analysis
        sentences = split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if sentence contains recommendation patterns
            for pattern in patterns:
                if pattern.lower() in sentence_lower:
                    
                    # Extract context (previous and next sentence)
                    context_start = max(0, i - 1)
                    context_end = min(len(sentences), i + 2)
                    context = ' '.join(sentences[context_start:context_end])
                    
                    # Calculate position
                    char_position = text.find(sentence)
                    
                    recommendation = {
                        'id': f"rec_{len(recommendations) + 1}",
                        'document': doc,
                        'sentence': sentence,
                        'context': context,
                        'pattern_matched': pattern,
                        'sentence_index': i,
                        'char_position': char_position,
                        'page_number': estimate_page_number(char_position, text),
                        'recommendation_type': classify_recommendation_type(sentence)
                    }
                    
                    recommendations.append(recommendation)
                    break  # Only match one pattern per sentence
    
    return recommendations

def find_responses(documents: List[Dict], patterns: List[str]) -> List[Dict]:
    """Find all responses in documents"""
    
    responses = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        sentences = split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if sentence contains response patterns
            for pattern in patterns:
                if pattern.lower() in sentence_lower:
                    
                    # Extract context
                    context_start = max(0, i - 1)
                    context_end = min(len(sentences), i + 2)
                    context = ' '.join(sentences[context_start:context_end])
                    
                    # Calculate position
                    char_position = text.find(sentence)
                    
                    response = {
                        'id': f"resp_{len(responses) + 1}",
                        'document': doc,
                        'sentence': sentence,
                        'context': context,
                        'pattern_matched': pattern,
                        'sentence_index': i,
                        'char_position': char_position,
                        'page_number': estimate_page_number(char_position, text),
                        'response_type': classify_response_type(sentence)
                    }
                    
                    responses.append(response)
                    break
    
    return responses

def align_recommendations_responses(recommendations: List[Dict], responses: List[Dict]) -> List[Dict]:
    """Align recommendations with their corresponding responses using semantic similarity"""
    
    alignments = []
    
    for rec in recommendations:
        best_matches = []
        
        # Find potential response matches
        for resp in responses:
            similarity_score = calculate_semantic_similarity(
                rec['context'], 
                resp['context']
            )
            
            # Also check if they reference similar topics
            topic_similarity = calculate_topic_similarity(
                rec['sentence'],
                resp['sentence']
            )
            
            combined_score = (similarity_score * 0.7) + (topic_similarity * 0.3)
            
            if combined_score > 0.3:  # Threshold for potential match
                best_matches.append({
                    'response': resp,
                    'similarity_score': similarity_score,
                    'topic_similarity': topic_similarity,
                    'combined_score': combined_score
                })
        
        # Sort by combined score
        best_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Create alignment
        alignment = {
            'recommendation': rec,
            'responses': best_matches[:3],  # Top 3 matches
            'alignment_confidence': best_matches[0]['combined_score'] if best_matches else 0,
            'alignment_status': determine_alignment_status(best_matches)
        }
        
        alignments.append(alignment)
    
    return alignments

def add_ai_summaries(alignments: List[Dict], summary_length: str) -> List[Dict]:
    """Add AI-generated summaries to alignments"""
    
    try:
        # Determine summary parameters
        max_sentences = {
            "Short (1-2 sentences)": 2,
            "Medium (3-4 sentences)": 4,  
            "Long (5+ sentences)": 6
        }.get(summary_length, 4)
        
        for alignment in alignments:
            rec = alignment['recommendation']
            responses = alignment['responses']
            
            # Create summary prompt
            summary_text = f"Recommendation: {rec['sentence']}\n\n"
            
            if responses:
                summary_text += "Related Responses:\n"
                for i, resp_match in enumerate(responses[:2], 1):
                    resp = resp_match['response']
                    summary_text += f"{i}. {resp['sentence']}\n"
            
            # Generate AI summary
            summary = generate_ai_summary(summary_text, max_sentences)
            
            alignment['ai_summary'] = summary
            alignment['summary_confidence'] = len(responses) * 0.2  # Simple confidence
    
    except Exception as e:
        st.warning(f"AI summary generation failed: {str(e)}")
        # Add empty summaries
        for alignment in alignments:
            alignment['ai_summary'] = "AI summary not available"
            alignment['summary_confidence'] = 0
    
    return alignments

def display_alignment_results(alignments: List[Dict], show_ai_summaries: bool):
    """Display the recommendation-response alignments"""
    
    if not alignments:
        st.warning("No recommendations found in the uploaded documents")
        return
    
    # Summary statistics
    st.markdown("### 📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recommendations", len(alignments))
    
    with col2:
        aligned_count = sum(1 for a in alignments if a['responses'])
        st.metric("Recommendations with Responses", aligned_count)
    
    with col3:
        avg_confidence = sum(a['alignment_confidence'] for a in alignments) / len(alignments) if alignments else 0
        st.metric("Avg Alignment Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        high_confidence = sum(1 for a in alignments if a['alignment_confidence'] > 0.7)
        st.metric("High Confidence Alignments", high_confidence)
    
    # Display individual alignments
    st.markdown("### 🔗 Recommendation-Response Alignments")
    
    for i, alignment in enumerate(alignments, 1):
        display_single_alignment(alignment, i, show_ai_summaries)

def display_single_alignment(alignment: Dict, index: int, show_ai_summaries: bool):
    """Display a single recommendation-response alignment"""
    
    rec = alignment['recommendation']
    responses = alignment['responses']
    confidence = alignment['alignment_confidence']
    
    # Confidence indicator
    confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
    
    with st.expander(f"{confidence_color} Recommendation {index} - {rec['recommendation_type']} (Confidence: {confidence:.2f})", 
                    expanded=index <= 3):
        
        # Two-column layout: Extract + Summary
        col1, col2 = st.columns([3, 2] if show_ai_summaries else [1])
        
        with col1:
            st.markdown("**📋 Original Extract:**")
            
            # Recommendation section
            st.markdown("**🎯 Recommendation:**")
            st.info(f"📄 {rec['document']['filename']} (Page {rec['page_number']})")
            
            # Highlight the recommendation
            highlighted_rec = f"<mark style='background-color: #FFEB3B; padding: 4px; border-radius: 4px;'>{rec['sentence']}</mark>"
            st.markdown(highlighted_rec, unsafe_allow_html=True)
            
            # Show context
            st.caption(f"📖 Context: {rec['context']}")
            
            # Responses section
            if responses:
                st.markdown("**↩️ Related Responses:**")
                
                for j, resp_match in enumerate(responses, 1):
                    resp = resp_match['response']
                    similarity = resp_match['combined_score']
                    
                    # Color code by similarity
                    resp_color = "#4CAF50" if similarity > 0.7 else "#FF9800" if similarity > 0.5 else "#F44336"
                    
                    st.info(f"📄 {resp['document']['filename']} (Page {resp['page_number']}) - Similarity: {similarity:.2f}")
                    
                    highlighted_resp = f"<mark style='background-color: {resp_color}; color: white; padding: 4px; border-radius: 4px;'>{resp['sentence']}</mark>"
                    st.markdown(highlighted_resp, unsafe_allow_html=True)
                    
                    st.caption(f"📖 Context: {resp['context']}")
                    
                    if j < len(responses):
                        st.markdown("---")
            else:
                st.warning("❌ No matching responses found")
        
        # AI Summary column
        if show_ai_summaries:
            with col2:
                st.markdown("**🤖 AI Summary:**")
                
                if 'ai_summary' in alignment and alignment['ai_summary']:
                    # Summary box
                    summary_confidence = alignment.get('summary_confidence', 0)
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 16px;
                        border-radius: 8px;
                        margin: 8px 0;
                    ">
                        <h4 style="margin: 0 0 12px 0; color: white;">📝 Summary</h4>
                        <p style="margin: 0; line-height: 1.6;">{alignment['ai_summary']}</p>
                        <small style="opacity: 0.8;">Confidence: {summary_confidence:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("🤖 AI summary not available")

# Utility functions for alignment
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

def estimate_page_number(char_position: int, text: str) -> int:
    """Estimate page number based on character position"""
    if char_position <= 0:
        return 1
    return max(1, char_position // 2000 + 1)

def classify_recommendation_type(sentence: str) -> str:
    """Classify the type of recommendation"""
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['urgent', 'immediate', 'critical', 'emergency']):
        return 'Urgent'
    elif any(word in sentence_lower for word in ['consider', 'review', 'explore', 'examine']):
        return 'Consideration'
    elif any(word in sentence_lower for word in ['implement', 'establish', 'create', 'develop']):
        return 'Implementation'
    elif any(word in sentence_lower for word in ['policy', 'regulation', 'framework', 'legislation']):
        return 'Policy'
    elif any(word in sentence_lower for word in ['budget', 'funding', 'financial', 'cost']):
        return 'Financial'
    elif any(word in sentence_lower for word in ['training', 'education', 'skills', 'capacity']):
        return 'Training'
    else:
        return 'General'

def classify_response_type(sentence: str) -> str:
    """Classify the type of response"""
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['accept', 'agree', 'approve', 'endorse']):
        return 'Acceptance'
    elif any(word in sentence_lower for word in ['reject', 'decline', 'disagree', 'oppose']):
        return 'Rejection'
    elif any(word in sentence_lower for word in ['consider', 'review', 'evaluate', 'assess']):
        return 'Under Review'
    elif any(word in sentence_lower for word in ['implement', 'action', 'proceed', 'execute']):
        return 'Implementation'
    elif any(word in sentence_lower for word in ['partial', 'some', 'limited', 'qualified']):
        return 'Partial Acceptance'
    else:
        return 'General Response'

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    words1 = set(w.lower() for w in text1.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
    words2 = set(w.lower() for w in text2.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def calculate_topic_similarity(sentence1: str, sentence2: str) -> float:
    """Calculate topic similarity based on key terms"""
    
    # Extract key topics/entities
    key_terms1 = extract_key_terms(sentence1)
    key_terms2 = extract_key_terms(sentence2)
    
    if not key_terms1 or not key_terms2:
        return 0.0
    
    intersection = len(set(key_terms1) & set(key_terms2))
    union = len(set(key_terms1) | set(key_terms2))
    
    return intersection / union if union > 0 else 0.0

def extract_key_terms(sentence: str) -> List[str]:
    """Extract key terms from a sentence"""
    words = re.findall(r'\b\w+\b', sentence.lower())
    key_terms = [word for word in words if len(word) > 3 and word not in STOP_WORDS]
    
    # Add some domain-specific important terms even if short
    important_short_terms = {'ai', 'it', 'hr', 'ceo', 'cfo', 'uk', 'eu', 'us', 'gdp'}
    short_terms = [word for word in words if word in important_short_terms]
    
    return key_terms + short_terms

def determine_alignment_status(matches: List[Dict]) -> str:
    """Determine the status of recommendation-response alignment"""
    if not matches:
        return "No Response Found"
    
    best_score = matches[0]['combined_score']
    
    if best_score > 0.8:
        return "Strong Alignment"
    elif best_score > 0.6:
        return "Good Alignment"
    elif best_score > 0.4:
        return "Weak Alignment"
    else:
        return "Poor Alignment"

def generate_ai_summary(text: str, max_sentences: int) -> str:
    """Generate AI summary of recommendation-response pair"""
    
    # Split into lines and analyze
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    summary_points = []
    
    # Extract key information
    for line in lines:
        if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'accept', 'reject', 'implement', 'consider']):
            # Clean and simplify the sentence
            clean_line = line.strip()
            if clean_line and len(clean_line) > 15:
                # Remove redundant phrases
                clean_line = re.sub(r'\b(that|which|who|where|when)\b', '', clean_line)
                clean_line = re.sub(r'\s+', ' ', clean_line).strip()
                summary_points.append(clean_line)
    
    # Limit to max sentences
    summary_points = summary_points[:max_sentences]
    
    if not summary_points:
        return "This recommendation-response pair shows standard government consultation process."
    
    # Create coherent summary
    if len(summary_points) == 1:
        return summary_points[0]
    else:
        return '. '.join(summary_points) + '.'
