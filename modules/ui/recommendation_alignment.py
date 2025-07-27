# Fixed recommendation_alignment.py - Handles unpacking errors
import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

def render_recommendation_alignment_interface(documents: List[Dict[str, Any]]):
    """Fixed recommendation-response alignment interface with proper error handling"""
    
    st.header("🏛️ Recommendation-Response Alignment")
    st.markdown("*Automatically find recommendations and their corresponding responses*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        return
    
    # Add tabs for different modes - FIXED: Handle tab creation safely
    try:
        tab1, tab2 = st.tabs(["🔄 Auto Alignment", "🔍 Manual Search"])
        
        with tab1:
            render_auto_alignment_tab_fixed(documents)
        
        with tab2:
            render_manual_search_tab_fixed(documents)
            
    except Exception as e:
        logger.error(f"Tab creation error: {e}")
        # Fallback: render without tabs
        st.markdown("### 🔄 Automatic Alignment")
        render_auto_alignment_tab_fixed(documents)
        
        st.markdown("---")
        st.markdown("### 🔍 Manual Search")
        render_manual_search_tab_fixed(documents)

def render_auto_alignment_tab_fixed(documents: List[Dict[str, Any]]):
    """Fixed automatic alignment with proper error handling"""
    
    st.markdown("### 🔄 Automatic Recommendation-Response Alignment")
    st.markdown("*Automatically find and align all recommendations with their responses*")
    
    # Configuration options - FIXED: Handle missing columns gracefully
    try:
        col1, col2 = st.columns(2)
    except Exception:
        # Fallback: use single column layout
        col1 = st.container()
        col2 = st.container()
    
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
        use_ai_summary = st.checkbox("Generate AI Summaries", value=False)  # Default to False
        
        # Check AI availability safely
        ai_available = False
        try:
            import sentence_transformers
            import torch
            ai_available = True
        except ImportError:
            st.warning("🤖 AI summaries require: pip install sentence-transformers torch")
            use_ai_summary = False
        
        # Summary length
        summary_length = st.selectbox(
            "Summary Length",
            ["Short (1-2 sentences)", "Medium (3-4 sentences)", "Long (5+ sentences)"],
            index=1
        )
    
    # Start analysis button
    if st.button("🔍 Find & Align All Recommendations", type="primary"):
        with st.spinner("🔍 Analyzing documents for recommendations and responses..."):
            
            try:
                # Step 1: Find recommendations - FIXED: Handle return values safely
                recommendations = find_recommendations_safe(documents, rec_patterns)
                
                # Step 2: Find responses - FIXED: Handle return values safely
                responses = find_responses_safe(documents, resp_patterns)
                
                # Step 3: Align recommendations with responses - FIXED: Handle alignment safely
                alignments = align_recommendations_responses_safe(recommendations, responses)
                
                # Step 4: Generate AI summaries if enabled - FIXED: Handle AI safely
                if use_ai_summary and ai_available:
                    alignments = add_ai_summaries_safe(alignments, summary_length)
                
                # Step 5: Display results - FIXED: Handle display safely
                display_alignment_results_safe(alignments, use_ai_summary)
                
            except Exception as e:
                logger.error(f"Alignment process error: {e}")
                st.error(f"Alignment process error: {str(e)}")
                
                # Fallback: Show basic analysis
                st.markdown("### 📊 Basic Analysis (Fallback)")
                show_basic_analysis_fallback(documents, rec_patterns, resp_patterns)

def find_recommendations_safe(documents: List[Dict], patterns: List[str]) -> List[Dict]:
    """Safely find recommendations with proper error handling"""
    
    recommendations = []
    
    try:
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue
            
            # Split into sentences safely
            sentences = split_into_sentences_safe(text)
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                
                # Check if sentence contains recommendation patterns
                for pattern in patterns:
                    if pattern.lower() in sentence_lower:
                        
                        # Extract context safely
                        context = get_context_safe(sentences, i)
                        
                        # Calculate position safely
                        char_position = text.find(sentence) if sentence in text else i * 100
                        
                        recommendation = {
                            'id': f"rec_{len(recommendations) + 1}",
                            'document': doc,
                            'sentence': sentence,
                            'context': context,
                            'pattern_matched': pattern,
                            'sentence_index': i,
                            'char_position': char_position,
                            'page_number': estimate_page_number_safe(char_position, text),
                            'recommendation_type': classify_recommendation_type_safe(sentence)
                        }
                        
                        recommendations.append(recommendation)
                        break  # Only match one pattern per sentence
                        
    except Exception as e:
        logger.error(f"Error finding recommendations: {e}")
        # Return what we have so far
        
    return recommendations

def find_responses_safe(documents: List[Dict], patterns: List[str]) -> List[Dict]:
    """Safely find responses with proper error handling"""
    
    responses = []
    
    try:
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue
            
            sentences = split_into_sentences_safe(text)
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                
                # Check if sentence contains response patterns
                for pattern in patterns:
                    if pattern.lower() in sentence_lower:
                        
                        # Extract context safely
                        context = get_context_safe(sentences, i)
                        
                        # Calculate position safely
                        char_position = text.find(sentence) if sentence in text else i * 100
                        
                        response = {
                            'id': f"resp_{len(responses) + 1}",
                            'document': doc,
                            'sentence': sentence,
                            'context': context,
                            'pattern_matched': pattern,
                            'sentence_index': i,
                            'char_position': char_position,
                            'page_number': estimate_page_number_safe(char_position, text),
                            'response_type': classify_response_type_safe(sentence)
                        }
                        
                        responses.append(response)
                        break
                        
    except Exception as e:
        logger.error(f"Error finding responses: {e}")
        
    return responses

def align_recommendations_responses_safe(recommendations: List[Dict], responses: List[Dict]) -> List[Dict]:
    """Safely align recommendations with responses"""
    
    alignments = []
    
    try:
        for rec in recommendations:
            best_matches = []
            
            # Find potential response matches safely
            for resp in responses:
                try:
                    # Safe similarity calculation
                    similarity_score = calculate_semantic_similarity_safe(
                        rec.get('context', ''), 
                        resp.get('context', '')
                    )
                    
                    # Safe topic similarity
                    topic_similarity = calculate_topic_similarity_safe(
                        rec.get('sentence', ''),
                        resp.get('sentence', '')
                    )
                    
                    combined_score = (similarity_score * 0.7) + (topic_similarity * 0.3)
                    
                    if combined_score > 0.3:  # Threshold for potential match
                        best_matches.append({
                            'response': resp,
                            'similarity_score': similarity_score,
                            'topic_similarity': topic_similarity,
                            'combined_score': combined_score
                        })
                        
                except Exception as e:
                    logger.error(f"Error calculating similarity: {e}")
                    continue
            
            # Sort by combined score safely
            try:
                best_matches.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            except Exception:
                pass
            
            # Create alignment safely
            alignment = {
                'recommendation': rec,
                'responses': best_matches[:3],  # Top 3 matches
                'alignment_confidence': best_matches[0].get('combined_score', 0) if best_matches else 0,
                'alignment_status': determine_alignment_status_safe(best_matches)
            }
            
            alignments.append(alignment)
            
    except Exception as e:
        logger.error(f"Error in alignment process: {e}")
        
    return alignments

def display_alignment_results_safe(alignments: List[Dict], show_ai_summaries: bool):
    """Safely display alignment results with proper error handling"""
    
    if not alignments:
        st.warning("No recommendations found in the uploaded documents")
        return
    
    try:
        # Summary statistics - FIXED: Handle missing data gracefully
        st.markdown("### 📊 Analysis Summary")
        
        # Use container layout as fallback
        try:
            col1, col2, col3, col4 = st.columns(4)
        except Exception:
            # Fallback: use metrics without columns
            st.metric("Total Recommendations", len(alignments))
            
            aligned_count = sum(1 for a in alignments if a.get('responses', []))
            st.metric("Recommendations with Responses", aligned_count)
            
            avg_confidence = sum(a.get('alignment_confidence', 0) for a in alignments) / len(alignments) if alignments else 0
            st.metric("Avg Alignment Confidence", f"{avg_confidence:.2f}")
            
            high_confidence = sum(1 for a in alignments if a.get('alignment_confidence', 0) > 0.7)
            st.metric("High Confidence Alignments", high_confidence)
            return
        
        with col1:
            st.metric("Total Recommendations", len(alignments))
        
        with col2:
            aligned_count = sum(1 for a in alignments if a.get('responses', []))
            st.metric("Recommendations with Responses", aligned_count)
        
        with col3:
            avg_confidence = sum(a.get('alignment_confidence', 0) for a in alignments) / len(alignments) if alignments else 0
            st.metric("Avg Alignment Confidence", f"{avg_confidence:.2f}")
        
        with col4:
            high_confidence = sum(1 for a in alignments if a.get('alignment_confidence', 0) > 0.7)
            st.metric("High Confidence Alignments", high_confidence)
        
        # Display individual alignments
        st.markdown("### 🔗 Recommendation-Response Alignments")
        
        for i, alignment in enumerate(alignments, 1):
            display_single_alignment_safe(alignment, i, show_ai_summaries)
            
    except Exception as e:
        logger.error(f"Error displaying alignment results: {e}")
        st.error(f"Display error: {str(e)}")
        
        # Fallback: Simple display
        st.markdown("### 📊 Simple Results")
        for i, alignment in enumerate(alignments[:5], 1):  # First 5 only
            rec = alignment.get('recommendation', {})
            st.write(f"{i}. {rec.get('sentence', 'Unknown')[:100]}...")

def display_single_alignment_safe(alignment: Dict, index: int, show_ai_summaries: bool):
    """Safely display a single alignment with error handling"""
    
    try:
        rec = alignment.get('recommendation', {})
        responses = alignment.get('responses', [])
        confidence = alignment.get('alignment_confidence', 0)
        
        # Confidence indicator
        confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
        rec_type = rec.get('recommendation_type', 'General')
        
        with st.expander(f"{confidence_color} Recommendation {index} - {rec_type} (Confidence: {confidence:.2f})", 
                        expanded=index <= 3):
            
            # Layout handling - FIXED: Use safe column creation
            try:
                col1, col2 = st.columns([3, 2] if show_ai_summaries else [1])
            except Exception:
                # Fallback: single column
                col1 = st.container()
                col2 = st.container() if show_ai_summaries else None
            
            with col1:
                st.markdown("**📋 Original Extract:**")
                
                # Recommendation section
                st.markdown("**🎯 Recommendation:**")
                doc_name = rec.get('document', {}).get('filename', 'Unknown Document')
                page_num = rec.get('page_number', 1)
                st.info(f"📄 {doc_name} (Page {page_num})")
                
                # Highlight the recommendation
                sentence = rec.get('sentence', 'No sentence available')
                highlighted_rec = f"<mark style='background-color: #FFEB3B; padding: 4px; border-radius: 4px;'>{sentence}</mark>"
                st.markdown(highlighted_rec, unsafe_allow_html=True)
                
                # Show context
                context = rec.get('context', 'No context available')
                st.caption(f"📖 Context: {context}")
                
                # Responses section
                if responses:
                    st.markdown("**↩️ Related Responses:**")
                    
                    for j, resp_match in enumerate(responses, 1):
                        try:
                            resp = resp_match.get('response', {})
                            similarity = resp_match.get('combined_score', 0)
                            
                            # Color code by similarity
                            resp_color = "#4CAF50" if similarity > 0.7 else "#FF9800" if similarity > 0.5 else "#F44336"
                            
                            resp_doc_name = resp.get('document', {}).get('filename', 'Unknown Document')
                            resp_page_num = resp.get('page_number', 1)
                            st.info(f"📄 {resp_doc_name} (Page {resp_page_num}) - Similarity: {similarity:.2f}")
                            
                            resp_sentence = resp.get('sentence', 'No sentence available')
                            highlighted_resp = f"<mark style='background-color: {resp_color}; color: white; padding: 4px; border-radius: 4px;'>{resp_sentence}</mark>"
                            st.markdown(highlighted_resp, unsafe_allow_html=True)
                            
                            resp_context = resp.get('context', 'No context available')
                            st.caption(f"📖 Context: {resp_context}")
                            
                            if j < len(responses):
                                st.markdown("---")
                        except Exception as e:
                            logger.error(f"Error displaying response {j}: {e}")
                            st.error(f"Error displaying response {j}")
                else:
                    st.warning("❌ No matching responses found")
            
            # AI Summary column (if enabled)
            if show_ai_summaries and col2:
                with col2:
                    st.markdown("**🤖 AI Summary:**")
                    
                    if 'ai_summary' in alignment and alignment['ai_summary']:
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
                        
    except Exception as e:
        logger.error(f"Error displaying single alignment: {e}")
        st.error(f"Error displaying alignment {index}: {str(e)}")

def render_manual_search_tab_fixed(documents: List[Dict[str, Any]]):
    """Fixed manual search with proper error handling"""
    
    st.markdown("### 🔍 Manual Sentence Search")
    st.markdown("*Paste a sentence to find matching recommendations or responses*")
    
    # Text input for manual search
    search_sentence = st.text_area(
        "📝 Paste your sentence here:",
        placeholder="e.g., 'We recommend implementing new security protocols immediately'",
        help="Paste any sentence and we'll find similar recommendations or responses",
        height=100
    )
    
    # Search configuration - FIXED: Handle column creation safely
    try:
        col1, col2 = st.columns(2)
    except Exception:
        col1 = st.container()
        col2 = st.container()
    
    with col1:
        search_type = st.radio(
            "What are you looking for?",
            ["🔍 Find Similar (Any Type)", "🎯 Find Recommendations", "↩️ Find Responses"],
            help="Choose what type of content to search for"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower values find more matches but may be less relevant"
        )
    
    with col2:
        max_matches = st.selectbox(
            "Max matches to show",
            [5, 10, 20, 50, "All"],
            index=1,
            help="Maximum number of matches to display"
        )
        
        show_scores = st.checkbox("Show similarity scores", value=True)
    
    # Search execution
    if st.button("🔎 Find Matches", type="primary") and search_sentence.strip():
        
        start_time = time.time()
        
        with st.spinner("🔍 Searching for similar content..."):
            try:
                # Determine search patterns based on type
                if search_type == "🎯 Find Recommendations":
                    patterns = ["recommend", "suggest", "advise", "propose", "urge", "should", "must"]
                    search_mode = "recommendations"
                elif search_type == "↩️ Find Responses":
                    patterns = ["accept", "reject", "agree", "disagree", "implement", "consider", "response", "reply", "approved", "declined"]
                    search_mode = "responses"
                else:
                    patterns = []  # Search all content
                    search_mode = "all"
                
                # Find matches safely
                matches = find_similar_sentences_safe(
                    documents=documents,
                    target_sentence=search_sentence,
                    search_mode=search_mode,
                    patterns=patterns,
                    similarity_threshold=similarity_threshold,
                    max_matches=max_matches
                )
                
                search_time = time.time() - start_time
                
                # Display results safely
                display_manual_search_results_safe(
                    matches=matches,
                    target_sentence=search_sentence,
                    search_time=search_time,
                    show_scores=show_scores,
                    search_mode=search_mode
                )
                
            except Exception as e:
                logger.error(f"Manual search error: {e}")
                st.error(f"Search error: {str(e)}")

# Helper functions with safe implementations

def split_into_sentences_safe(text: str) -> List[str]:
    """Safely split text into sentences"""
    try:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    except Exception:
        # Fallback: split by periods only
        return [s.strip() for s in text.split('.') if s.strip()]

def get_context_safe(sentences: List[str], sentence_idx: int, context_window: int = 2) -> str:
    """Safely get context around a sentence"""
    try:
        start = max(0, sentence_idx - context_window)
        end = min(len(sentences), sentence_idx + context_window + 1)
        context_sentences = sentences[start:end]
        return ' '.join(context_sentences)
    except Exception:
        # Fallback: return the sentence itself
        return sentences[sentence_idx] if sentence_idx < len(sentences) else ""

def estimate_page_number_safe(char_position: int, text: str) -> int:
    """Safely estimate page number"""
    try:
        if char_position <= 0:
            return 1
        return max(1, char_position // 2000 + 1)
    except Exception:
        return 1

def classify_recommendation_type_safe(sentence: str) -> str:
    """Safely classify recommendation type"""
    try:
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['urgent', 'immediate', 'critical', 'emergency']):
            return 'Urgent'
        elif any(word in sentence_lower for word in ['consider', 'review', 'explore', 'examine']):
            return 'Consideration'
        elif any(word in sentence_lower for word in ['implement', 'establish', 'create', 'develop']):
            return 'Implementation'
        elif any(word in sentence_lower for word in ['policy', 'regulation', 'framework', 'legislation']):
            return 'Policy'
        else:
            return 'General'
    except Exception:
        return 'General'

def classify_response_type_safe(sentence: str) -> str:
    """Safely classify response type"""
    try:
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['accept', 'agree', 'approve', 'endorse']):
            return 'Acceptance'
        elif any(word in sentence_lower for word in ['reject', 'decline', 'disagree', 'oppose']):
            return 'Rejection'
        elif any(word in sentence_lower for word in ['consider', 'review', 'evaluate', 'assess']):
            return 'Under Review'
        else:
            return 'General Response'
    except Exception:
        return 'General Response'

def calculate_semantic_similarity_safe(text1: str, text2: str) -> float:
    """Safely calculate semantic similarity"""
    try:
        words1 = set(w.lower() for w in text1.split() if len(w) > 2)
        words2 = set(w.lower() for w in text2.split() if len(w) > 2)
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0

def calculate_topic_similarity_safe(sentence1: str, sentence2: str) -> float:
    """Safely calculate topic similarity"""
    try:
        # Simple word overlap
        words1 = set(sentence1.lower().split())
        words2 = set(sentence2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0

def determine_alignment_status_safe(matches: List[Dict]) -> str:
    """Safely determine alignment status"""
    try:
        if not matches:
            return "No Response Found"
        
        best_score = matches[0].get('combined_score', 0)
        
        if best_score > 0.8:
            return "Strong Alignment"
        elif best_score > 0.6:
            return "Good Alignment"
        elif best_score > 0.4:
            return "Weak Alignment"
        else:
            return "Poor Alignment"
    except Exception:
        return "Unknown"

def add_ai_summaries_safe(alignments: List[Dict], summary_length: str) -> List[Dict]:
    """Safely add AI summaries with fallback"""
    try:
        # Simple fallback summaries
        for alignment in alignments:
            rec = alignment.get('recommendation', {})
            responses = alignment.get('responses', [])
            
            if responses:
                summary = f"Recommendation about {rec.get('sentence', 'policy matter')[:50]}... has {len(responses)} related response(s)."
            else:
                summary = f"Recommendation about {rec.get('sentence', 'policy matter')[:50]}... has no matching responses found."
            
            alignment['ai_summary'] = summary
            alignment['summary_confidence'] = 0.5
            
    except Exception as e:
        logger.error(f"Error adding AI summaries: {e}")
        
    return alignments

def find_similar_sentences_safe(documents: List[Dict], target_sentence: str, search_mode: str, 
                               patterns: List[str], similarity_threshold: float, max_matches) -> List[Dict]:
    """Safely find similar sentences"""
    matches = []
    
    try:
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue
            
            sentences = split_into_sentences_safe(text)
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                
                # Filter by search mode if specified
                if search_mode == "recommendations" and not any(pattern in sentence_lower for pattern in patterns):
                    continue
                elif search_mode == "responses" and not any(pattern in sentence_lower for pattern in patterns):
                    continue
                
                # Calculate similarity
                similarity_score = calculate_semantic_similarity_safe(target_sentence, sentence)
                
                if similarity_score >= similarity_threshold:
                    
                    context = get_context_safe(sentences, i)
                    char_position = text.find(sentence) if sentence in text else i * 100
                    
                    match = {
                        'sentence': sentence,
                        'context': context,
                        'similarity_score': similarity_score,
                        'document': doc,
                        'sentence_index': i,
                        'char_position': char_position,
                        'page_number': estimate_page_number_safe(char_position, text),
                        'content_type': classify_recommendation_type_safe(sentence),
                        'matched_patterns': [p for p in patterns if p in sentence_lower] if patterns else []
                    }
                    
                    matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Limit results
        if max_matches != "All":
            matches = matches[:max_matches]
            
    except Exception as e:
        logger.error(f"Error finding similar sentences: {e}")
        
    return matches

def display_manual_search_results_safe(matches: List[Dict], target_sentence: str, search_time: float, 
                                     show_scores: bool, search_mode: str):
    """Safely display manual search results"""
    
    try:
        if not matches:
            st.warning(f"No matches found for your sentence.")
            st.info("💡 Try lowering the similarity threshold or changing the search type.")
            return
        
        # Summary
        mode_text = {
            "recommendations": "recommendations",
            "responses": "responses", 
            "all": "sentences"
        }.get(search_mode, "items")
        
        st.success(f"🎯 Found **{len(matches)}** similar {mode_text} in {search_time:.3f} seconds")
        
        # Show original sentence
        st.markdown("### 📝 Your Original Sentence:")
        st.info(f"*{target_sentence}*")
        
        # Display matches
        st.markdown("### 🔍 Similar Matches Found:")
        
        for i, match in enumerate(matches, 1):
            
            # Confidence indicator
            similarity = match.get('similarity_score', 0)
            if similarity > 0.8:
                confidence_color = "🟢"
                confidence_text = "Very High"
            elif similarity > 0.6:
                confidence_color = "🟡"
                confidence_text = "High"
            elif similarity > 0.4:
                confidence_color = "🟠"
                confidence_text = "Medium"
            else:
                confidence_color = "🔴"
                confidence_text = "Low"
            
            # Score display
            score_text = f" (Score: {similarity:.3f})" if show_scores else ""
            content_type = match.get('content_type', 'General')
            
            with st.expander(f"{confidence_color} Match {i} - {content_type} - {confidence_text} Similarity{score_text}", 
                            expanded=i <= 3):
                
                # Document info
                doc_name = match.get('document', {}).get('filename', 'Unknown')
                page_num = match.get('page_number', 1)
                
                st.markdown(f"**📄 Document:** {doc_name} (Page {page_num})")
                
                # Found sentence
                st.markdown("**📄 Found Sentence:**")
                sentence = match.get('sentence', 'No sentence available')
                st.markdown(f"*{sentence}*")
                
                # Context
                context = match.get('context', 'No context available')
                st.markdown("**📖 Context:**")
                st.caption(context)
                
                # Matched patterns if any
                matched_patterns = match.get('matched_patterns', [])
                if matched_patterns:
                    st.markdown(f"**🎯 Matched Keywords:** {', '.join(matched_patterns)}")
                
                if show_scores:
                    st.markdown(f"**Similarity:** {similarity:.3f}")
                    
    except Exception as e:
        logger.error(f"Error displaying manual search results: {e}")
        st.error(f"Display error: {str(e)}")

def show_basic_analysis_fallback(documents: List[Dict], rec_patterns: List[str], resp_patterns: List[str]):
    """Show basic analysis when full alignment fails"""
    
    st.markdown("### 📊 Basic Pattern Analysis")
    
    rec_count = 0
    resp_count = 0
    doc_analysis = []
    
    for doc in documents:
        text = doc.get('text', '').lower()
        filename = doc.get('filename', 'Unknown')
        
        doc_rec_count = sum(text.count(pattern.lower()) for pattern in rec_patterns)
        doc_resp_count = sum(text.count(pattern.lower()) for pattern in resp_patterns)
        
        rec_count += doc_rec_count
        resp_count += doc_resp_count
        
        if doc_rec_count > 0 or doc_resp_count > 0:
            doc_analysis.append({
                'Document': filename,
                'Recommendations': doc_rec_count,
                'Responses': doc_resp_count
            })
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Recommendation Mentions", rec_count)
    with col2:
        st.metric("Total Response Mentions", resp_count)
    with col3:
        st.metric("Documents with Content", len(doc_analysis))
    
    # Document breakdown
    if doc_analysis:
        st.markdown("### 📄 Document Breakdown")
        df = pd.DataFrame(doc_analysis)
        st.dataframe(df, use_container_width=True)
    
    # Pattern breakdown
    st.markdown("### 🔍 Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Recommendation Patterns Found:**")
        for pattern in rec_patterns:
            total_count = sum(doc.get('text', '').lower().count(pattern.lower()) for doc in documents)
            if total_count > 0:
                st.write(f"• '{pattern}': {total_count} occurrences")
    
    with col2:
        st.markdown("**↩️ Response Patterns Found:**")
        for pattern in resp_patterns:
            total_count = sum(doc.get('text', '').lower().count(pattern.lower()) for doc in documents)
            if total_count > 0:
                st.write(f"• '{pattern}': {total_count} occurrences")

# Export the main function
__all__ = ['render_recommendation_alignment_interface']
