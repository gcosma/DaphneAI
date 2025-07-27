# modules/ui/recommendation_alignment.py - Recommendation-Response Alignment System
import streamlit as st
import pandas as pd
import re
import time  # ADDED: Missing import
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
    
    # Add tabs for different modes
    tab1, tab2 = st.tabs(["🔄 Auto Alignment", "🔍 Manual Search"])
    
    with tab1:
        render_auto_alignment_tab(documents)
    
    with tab2:
        render_manual_search_tab(documents)

def render_auto_alignment_tab(documents: List[Dict[str, Any]]):
    """Render the automatic alignment functionality"""
    
    st.markdown("### 🔄 Automatic Recommendation-Response Alignment")
    st.markdown("*Automatically find and align all recommendations with their responses*")
    
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
    if st.button("🔍 Find & Align All Recommendations", type="primary"):
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

def render_manual_search_tab(documents: List[Dict[str, Any]]):
    """Render the manual search functionality"""
    
    st.markdown("### 🔍 Manual Sentence Search")
    st.markdown("*Paste a sentence to find matching recommendations or responses*")
    
    # Text input for manual search
    search_sentence = st.text_area(
        "📝 Paste your sentence here:",
        placeholder="e.g., 'We recommend implementing new security protocols immediately' or 'The department agrees to consider this proposal'",
        help="Paste any sentence and we'll find similar recommendations or responses in your documents",
        height=100
    )
    
    # Search configuration
    col1, col2 = st.columns(2)
    
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
            
            # Find matches
            matches = find_similar_sentences(
                documents=documents,
                target_sentence=search_sentence,
                search_mode=search_mode,
                patterns=patterns,
                similarity_threshold=similarity_threshold,
                max_matches=max_matches
            )
            
            search_time = time.time() - start_time
            
            # Display results
            display_manual_search_results(
                matches=matches,
                target_sentence=search_sentence,
                search_time=search_time,
                show_scores=show_scores,
                search_mode=search_mode
            )

def find_similar_sentences(documents: List[Dict], target_sentence: str, search_mode: str, 
                          patterns: List[str], similarity_threshold: float, max_matches) -> List[Dict]:
    """Find sentences similar to the target sentence"""
    
    matches = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Filter by search mode if specified
            if search_mode == "recommendations" and not any(pattern in sentence_lower for pattern in patterns):
                continue
            elif search_mode == "responses" and not any(pattern in sentence_lower for pattern in patterns):
                continue
            
            # Calculate similarity
            similarity_score = calculate_sentence_similarity(target_sentence, sentence)
            
            if similarity_score >= similarity_threshold:
                
                # Extract context
                context_start = max(0, i - 1)
                context_end = min(len(sentences), i + 2)
                context = ' '.join(sentences[context_start:context_end])
                
                # Calculate position
                char_position = text.find(sentence)
                
                match = {
                    'sentence': sentence,
                    'context': context,
                    'similarity_score': similarity_score,
                    'document': doc,
                    'sentence_index': i,
                    'char_position': char_position,
                    'page_number': estimate_page_number(char_position, text),
                    'content_type': classify_sentence_type(sentence),
                    'matched_patterns': [p for p in patterns if p in sentence_lower] if patterns else []
                }
                
                matches.append(match)
    
    # Sort by similarity score
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Limit results
    if max_matches != "All":
        matches = matches[:max_matches]
    
    return matches

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """Calculate similarity between two sentences using multiple methods"""
    
    # Method 1: Word overlap similarity (enhanced)
    words1 = set(w.lower() for w in sentence1.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
    words2 = set(w.lower() for w in sentence2.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    word_overlap = intersection / union if union > 0 else 0.0
    
    # Method 2: Semantic similarity (using key terms)
    semantic_score = calculate_semantic_sentence_similarity(sentence1, sentence2)
    
    # Method 3: Structural similarity (length and punctuation)
    len1, len2 = len(sentence1), len(sentence2)
    length_similarity = 1 - abs(len1 - len2) / max(len1, len2, 1)
    
    # Combine scores with weights
    combined_score = (
        word_overlap * 0.5 +
        semantic_score * 0.4 +
        length_similarity * 0.1
    )
    
    return combined_score

def calculate_semantic_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """Calculate semantic similarity using domain-specific terms"""
    
    # Enhanced semantic groups for government documents
    semantic_groups = {
        'recommendation': ['recommend', 'suggest', 'advise', 'propose', 'recommendation', 'proposal'],
        'response': ['respond', 'response', 'reply', 'answer', 'feedback', 'reaction'],
        'agreement': ['agree', 'accept', 'approve', 'endorse', 'support', 'adopt'],
        'disagreement': ['reject', 'decline', 'oppose', 'disagree', 'refuse', 'deny'],
        'action': ['implement', 'execute', 'carry out', 'perform', 'conduct', 'undertake'],
        'review': ['review', 'examine', 'assess', 'evaluate', 'analyze', 'consider'],
        'policy': ['policy', 'procedure', 'guideline', 'framework', 'protocol', 'strategy'],
        'urgent': ['urgent', 'immediate', 'critical', 'priority', 'emergency', 'pressing'],
        'financial': ['budget', 'cost', 'funding', 'financial', 'money', 'expense'],
        'government': ['government', 'department', 'ministry', 'authority', 'agency', 'administration']
    }
    
    # Extract semantic categories from both sentences
    categories1 = extract_semantic_categories(sentence1, semantic_groups)
    categories2 = extract_semantic_categories(sentence2, semantic_groups)
    
    if not categories1 or not categories2:
        return 0.0
    
    # Calculate category overlap
    common_categories = len(set(categories1) & set(categories2))
    total_categories = len(set(categories1) | set(categories2))
    
    return common_categories / total_categories if total_categories > 0 else 0.0

def extract_semantic_categories(sentence: str, semantic_groups: Dict) -> List[str]:
    """Extract semantic categories from a sentence"""
    
    sentence_lower = sentence.lower()
    categories = []
    
    for category, terms in semantic_groups.items():
        if any(term in sentence_lower for term in terms):
            categories.append(category)
    
    return categories

def classify_sentence_type(sentence: str) -> str:
    """Classify whether a sentence is a recommendation, response, or other"""
    
    sentence_lower = sentence.lower()
    
    # Check for recommendation indicators
    if any(word in sentence_lower for word in ['recommend', 'suggest', 'advise', 'propose', 'should', 'must', 'ought']):
        return 'Recommendation'
    
    # Check for response indicators
    elif any(word in sentence_lower for word in ['accept', 'reject', 'agree', 'disagree', 'implement', 'consider', 'response', 'reply']):
        return 'Response'
    
    # Check for policy/procedural content
    elif any(word in sentence_lower for word in ['policy', 'procedure', 'guideline', 'framework', 'protocol']):
        return 'Policy'
    
    # Check for analysis/review content
    elif any(word in sentence_lower for word in ['review', 'analyze', 'assess', 'evaluate', 'examine']):
        return 'Analysis'
    
    else:
        return 'General'

def display_manual_search_results(matches: List[Dict], target_sentence: str, search_time: float, 
                                show_scores: bool, search_mode: str):
    """Display manual search results"""
    
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
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy Results"):
            copy_manual_search_results(matches, target_sentence)
    with col2:
        if st.button("📊 Export to CSV"):
            export_manual_search_csv(matches, target_sentence)
    
    # Display matches
    st.markdown("### 🔍 Similar Matches Found:")
    
    for i, match in enumerate(matches, 1):
        
        # Confidence indicator
        similarity = match['similarity_score']
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
        
        with st.expander(f"{confidence_color} Match {i} - {match['content_type']} - {confidence_text} Similarity{score_text}", 
                        expanded=i <= 3):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Main content
                st.markdown("**📄 Found Sentence:**")
                
                # Highlight similar words
                highlighted_sentence = highlight_similar_words(match['sentence'], target_sentence)
                st.markdown(highlighted_sentence, unsafe_allow_html=True)
                
                # Context
                st.markdown("**📖 Context:**")
                st.caption(match['context'])
                
                # Matched patterns if any
                if match['matched_patterns']:
                    st.markdown(f"**🎯 Matched Keywords:** {', '.join(match['matched_patterns'])}")
            
            with col2:
                # Document info
                st.markdown("**📋 Document Info:**")
                st.markdown(f"**File:** {match['document']['filename']}")
                st.markdown(f"**Page:** {match['page_number']}")
                st.markdown(f"**Type:** {match['content_type']}")
                
                if show_scores:
                    st.markdown(f"**Similarity:** {similarity:.3f}")
                
                # Quick action button
                if st.button(f"🔗 Find Related Content", key=f"related_{i}"):
                    # Find content related to this match
                    find_related_content(match, target_sentence)

def highlight_similar_words(sentence: str, target_sentence: str) -> str:
    """Highlight words in sentence that are similar to target sentence"""
    
    target_words = set(w.lower() for w in target_sentence.split() if len(w) > 2 and w.lower() not in STOP_WORDS)
    
    highlighted = sentence
    words = sentence.split()
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in target_words:
            highlighted = highlighted.replace(word, f"<mark style='background-color: #FFEB3B; padding: 2px; border-radius: 2px;'>{word}</mark>")
    
    return highlighted

def find_related_content(match: Dict, target_sentence: str):
    """Find content related to the selected match"""
    
    # This could trigger a new search for content related to this match
    st.info(f"🔍 Searching for content related to: '{match['sentence'][:100]}...'")
    
    # You could implement additional logic here to find related recommendations/responses
    # For now, show basic info
    st.markdown(f"**Document:** {match['document']['filename']}")
    st.markdown(f"**Page:** {match['page_number']}")
    st.markdown(f"**Content Type:** {match['content_type']}")

def copy_manual_search_results(matches: List[Dict], target_sentence: str):
    """Copy manual search results"""
    
    output = f"Manual Search Results\n"
    output += f"Original Sentence: {target_sentence}\n"
    output += "=" * 50 + "\n\n"
    
    for i, match in enumerate(matches, 1):
        output += f"Match {i} ({match['content_type']}):\n"
        output += f"Similarity: {match['similarity_score']:.3f}\n"
        output += f"Document: {match['document']['filename']} (Page {match['page_number']})\n"
        output += f"Sentence: {match['sentence']}\n"
        output += f"Context: {match['context']}\n\n"
        output += "-" * 30 + "\n\n"
    
    st.code(output)
    st.success("Results copied to display! Use Ctrl+A, Ctrl+C to copy to clipboard")

def export_manual_search_csv(matches: List[Dict], target_sentence: str):
    """Export manual search results to CSV"""
    
    csv_data = []
    
    for i, match in enumerate(matches, 1):
        row = {
            'Match_Number': i,
            'Original_Sentence': target_sentence,
            'Found_Sentence': match['sentence'],
            'Similarity_Score': match['similarity_score'],
            'Content_Type': match['content_type'],
            'Document': match['document']['filename'],
            'Page_Number': match['page_number'],
            'Context': match['context'],
            'Matched_Patterns': ', '.join(match['matched_patterns']) if match['matched_patterns'] else ''
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"manual_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

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
