# modules/ui/beautiful_display.py - COMPLETE REVISED VERSION WITH FIXED HIGHLIGHTING
"""
Beautiful display functions for DaphneAI search results and alignments.
FULLY REVISED - All highlighting syntax fixed for Streamlit compatibility.
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List

# STOP WORDS for highlighting
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 
    'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 
    'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 
    'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 
    'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 
    'who', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 
    'come', 'made', 'may', 'part', 'or', 'also', 'back', 'any', 'good', 
    'new', 'where', 'much', 'take', 'know', 'just', 'see', 'after', 
    'very', 'well', 'here', 'should', 'old', 'still'
}

# =============================================================================
# FIXED UTILITY FUNCTIONS FOR HIGHLIGHTING
# =============================================================================

def get_meaningful_words(text: str) -> List[str]:
    """Extract meaningful words (non-stop words) from text"""
    words = re.findall(r'\b\w+\b', text.lower())
    meaningful = [word for word in words 
                 if word not in STOP_WORDS and len(word) > 1]
    return meaningful

def clean_html_artifacts(text: str) -> str:
    """Remove HTML artifacts and clean up text"""
    if not text:
        return "No content available"
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def format_text_as_clean_paragraphs(text: str) -> str:
    """Format text as clean paragraphs"""
    
    if not text:
        return "*No content available*"
    
    # Clean the text first
    clean_text = clean_html_artifacts(text)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return clean_text
    
    # Group into paragraphs
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        
        # Break on natural indicators or when we have enough sentences
        if len(current_paragraph) >= 3:
            if any(indicator in sentence.lower() for indicator in 
                   ['however', 'furthermore', 'additionally', 'therefore', 'moreover', 'meanwhile']):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        if len(current_paragraph) >= 4:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    # Add remaining sentences
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Format as clean paragraphs
    formatted_text = '\n\n'.join(f"> {paragraph}" for paragraph in paragraphs if paragraph.strip())
    
    return formatted_text if formatted_text else f"> {clean_text}"

# =============================================================================
# FIXED HIGHLIGHTING FUNCTIONS
# =============================================================================

def highlight_recommendation_terms_fixed(text: str) -> str:
    """FIXED: Highlight government terms with correct Streamlit syntax"""
    
    highlight_terms = [
        'recommend', 'recommendation', 'recommendations', 'suggest', 'advise', 'propose',
        'accept', 'reject', 'agree', 'disagree', 'implement', 'implementation', 
        'consider', 'approved', 'declined', 'response', 'reply', 'answer',
        'policy', 'framework', 'guideline', 'protocol', 'strategy',
        'committee', 'department', 'ministry', 'government', 'authority',
        'urgent', 'immediate', 'critical', 'priority', 'essential',
        'budget', 'funding', 'financial', 'cost', 'expenditure',
        'review', 'analysis', 'assessment', 'evaluation', 'inquiry'
    ]
    
    highlighted = text
    highlight_terms.sort(key=len, reverse=True)
    
    for term in highlight_terms:
        if len(term) > 3:
            try:
                # Create pattern for word boundaries
                pattern = re.compile(r'\b' + re.escape(term) + r'\w*', re.IGNORECASE)
                
                # Find all matches first
                matches = pattern.findall(highlighted)
                
                # Replace each unique match - FIXED: Use correct Streamlit syntax
                for match in set(matches):
                    if match.lower().startswith(term.lower()):
                        # FIXED: Use :yellow-background[] syntax instead of **:yellow[]**
                        highlighted = highlighted.replace(match, f':yellow-background[{match}]')
                        
            except re.error:
                # Skip invalid patterns
                continue
    
    return highlighted

def highlight_meaningful_words_fixed(text: str, query: str) -> str:
    """FIXED: Highlight meaningful words from query with correct Streamlit syntax"""
    
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return text
    
    highlighted = text
    meaningful_words.sort(key=len, reverse=True)
    
    for word in meaningful_words:
        if len(word) > 2:  # Only words longer than 2 characters
            try:
                # Create pattern for word boundaries
                pattern = re.compile(r'\b' + re.escape(word) + r'\w*', re.IGNORECASE)
                
                # Find all matches first
                matches = pattern.findall(highlighted)
                
                # Replace each unique match - FIXED: Use correct Streamlit syntax
                for match in set(matches):
                    if match.lower().startswith(word.lower()):
                        # FIXED: Use :yellow-background[] syntax instead of **:yellow[]**
                        highlighted = highlighted.replace(match, f':yellow-background[{match}]')
                        
            except re.error:
                # Skip invalid patterns
                continue
    
    return highlighted

# =============================================================================
# MAIN DISPLAY FUNCTIONS
# =============================================================================

def display_search_results_beautiful(results: List[Dict], query: str, search_time: float, 
                                   show_context: bool, highlight_matches: bool):
    """Display search results with beautiful paragraph formatting"""
    
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    # Group by document
    doc_groups = {}
    for result in results:
        doc_name = result['document']['filename']
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(result)
    
    # Summary with filtering info
    meaningful_words = get_meaningful_words(query)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0 0 10px 0;">🎯 Search Results</h3>
        <p style="margin: 0; font-size: 18px;">
            Found <strong>{len(results)}</strong> results in <strong>{len(doc_groups)}</strong> documents for <strong>"{query}"</strong>
        </p>
        <small style="opacity: 0.9;">Search completed in {search_time:.3f} seconds</small>
        {f'<br><small style="opacity: 0.8;">Focused on meaningful words: {", ".join(meaningful_words[:5])}{"..." if len(meaningful_words) > 5 else ""}</small>' if meaningful_words else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy All Results"):
            copy_results_beautiful(results, query)
    with col2:
        if st.button("📊 Export to CSV"):
            export_results_csv_beautiful(results, query)
    
    # Display results
    for doc_name, doc_results in doc_groups.items():
        
        best_score = max(r.get('score', 0) for r in doc_results)
        doc = doc_results[0].get('document', {})
        
        with st.expander(f"📄 {doc_name} ({len(doc_results)} matches, best score: {best_score:.1f})", expanded=True):
            
            # Document info
            word_count = len(doc.get('text', '').split()) if doc.get('text') else 0
            char_count = len(doc.get('text', '')) if doc.get('text') else 0
            est_pages = max(1, char_count // 2000)
            
            st.markdown(f"""
            <div style="
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #007bff;
            ">
                <strong>📊 Document Statistics:</strong><br>
                <strong>Words:</strong> {word_count:,} | <strong>Characters:</strong> {char_count:,} | <strong>Est. Pages:</strong> {est_pages}
            </div>
            """, unsafe_allow_html=True)
            
            # Display each match
            for i, result in enumerate(doc_results, 1):
                display_single_search_result_beautiful(result, i, query, show_context, highlight_matches)

def display_single_search_result_beautiful(result: Dict, index: int, query: str, 
                                         show_context: bool, highlight_matches: bool):
    """Display a single search result with FIXED highlighting"""
    
    method = result.get('match_type', 'unknown')
    score = result.get('score', 0)
    page = result.get('page_number', 1)
    position = result.get('position', 0)
    
    # Method info
    method_info = {
        'exact': {'icon': '🎯', 'color': '#28a745', 'name': 'Exact Match'},
        'smart': {'icon': '🧠', 'color': '#007bff', 'name': 'Smart Search'},
        'fuzzy': {'icon': '🌀', 'color': '#ffc107', 'name': 'Fuzzy Match'},
        'semantic': {'icon': '🤖', 'color': '#6f42c1', 'name': 'Semantic Match'},
        'hybrid': {'icon': '🔄', 'color': '#17a2b8', 'name': 'Hybrid Search'}
    }
    
    info = method_info.get(method, {'icon': '🔍', 'color': '#6c757d', 'name': 'Search'})
    
    # Header
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {info['color']} 0%, {info['color']}dd 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    ">
        <h4 style="margin: 0 0 5px 0;">
            {info['icon']} Match {index} - {info['name']}
        </h4>
        <div style="font-size: 14px; opacity: 0.9;">
            <strong>Score:</strong> {score:.1f} | <strong>Page:</strong> {page} | <strong>Position:</strong> {position:,}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show meaningful words found
    if method == 'smart' and 'meaningful_words_found' in result:
        meaningful_found = result['meaningful_words_found']
        total_meaningful = result.get('total_meaningful_words', 0)
        
        if meaningful_found:
            st.markdown(f"""
            <div style="
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 12px;
                border-radius: 6px;
                margin: 10px 0;
                font-size: 14px;
            ">
                <strong>🎯 Meaningful Words Found:</strong> {', '.join(meaningful_found)} 
                ({len(meaningful_found)}/{total_meaningful} words matched)
            </div>
            """, unsafe_allow_html=True)
    
    # Display context with FIXED highlighting
    if show_context:
        full_context = result.get('context', '')
        if full_context:
            
            st.markdown("**📖 Complete Context:**")
            
            # FIXED: Apply highlighting with correct syntax
            if highlight_matches:
                # Use the fixed highlighting function
                highlighted_context = highlight_meaningful_words_fixed(full_context, query)
                clean_content = clean_html_artifacts(highlighted_context)
                formatted_content = format_text_as_clean_paragraphs(clean_content)
            else:
                clean_content = clean_html_artifacts(full_context)
                formatted_content = format_text_as_clean_paragraphs(clean_content)
            
            st.markdown(formatted_content)
            
            # Additional details
            percentage = result.get('percentage_through', 0)
            st.caption(f"💡 This content appears around page {page}, {percentage:.1f}% through the document")

def display_alignment_results_beautiful(alignments: List[Dict], show_ai_summaries: bool):
    """Display alignment results with FIXED highlighting"""
    
    if not alignments:
        st.warning("No recommendations found in the uploaded documents")
        return
    
    # Summary statistics
    st.markdown("### 📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
        display_single_alignment_beautiful(alignment, i, show_ai_summaries)

def display_single_alignment_beautiful(alignment: Dict, index: int, show_ai_summaries: bool):
    """Display a single alignment with FIXED highlighting"""
    
    rec = alignment.get('recommendation', {})
    responses = alignment.get('responses', [])
    confidence = alignment.get('alignment_confidence', 0)
    
    # Confidence indicator
    confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
    rec_type = rec.get('recommendation_type', 'General')
    
    with st.expander(f"{confidence_color} Recommendation {index} - {rec_type} (Confidence: {confidence:.2f})", 
                    expanded=index <= 3):
        
        st.markdown("### 📋 Complete Extract")
        
        # Recommendation section
        st.markdown("#### 🎯 Recommendation")
        
        doc_name = rec.get('document', {}).get('filename', 'Unknown Document')
        page_num = rec.get('page_number', 1)
        st.info(f"📄 **Document:** {doc_name} | **Page:** {page_num}")
        
        # Display recommendation with FIXED highlighting
        full_sentence = rec.get('sentence', 'No sentence available')
        clean_sentence = clean_html_artifacts(full_sentence)
        
        st.markdown("**📝 Full Recommendation:**")
        with st.container():
            if len(clean_sentence.split()) > 5:
                # Use FIXED highlighting function
                highlighted_sentence = highlight_recommendation_terms_fixed(clean_sentence)
                st.markdown(f"> {highlighted_sentence}")
            else:
                st.markdown(f"> {clean_sentence}")
        
        st.markdown("")
        
        # Show context with FIXED highlighting
        full_context = rec.get('context', 'No context available')
        if full_context and full_context != full_sentence:
            st.markdown("#### 📖 Complete Context")
            
            clean_context = clean_html_artifacts(full_context)
            highlighted_context = highlight_recommendation_terms_fixed(clean_context)
            formatted_context = format_text_as_clean_paragraphs(highlighted_context)
            
            st.markdown(formatted_context)
        
        # Responses section with FIXED highlighting
        if responses:
            st.markdown("#### ↩️ Related Responses")
            
            for j, resp_match in enumerate(responses, 1):
                resp = resp_match.get('response', {})
                similarity = resp_match.get('combined_score', 0)
                
                # Color code by similarity
                if similarity > 0.7:
                    confidence_text = "High Confidence"
                elif similarity > 0.5:
                    confidence_text = "Medium Confidence"
                else:
                    confidence_text = "Lower Confidence"
                
                resp_doc_name = resp.get('document', {}).get('filename', 'Unknown Document')
                resp_page_num = resp.get('page_number', 1)
                
                # Display response with FIXED highlighting
                full_resp_sentence = resp.get('sentence', 'No sentence available')
                full_resp_context = resp.get('context', 'No context available')
                
                clean_resp_sentence = clean_html_artifacts(full_resp_sentence)
                clean_resp_context = clean_html_artifacts(full_resp_context)
                
                st.markdown(f"**📄 Response {j} - {confidence_text} ({similarity:.2f})**")
                st.info(f"📄 **Document:** {resp_doc_name} | **Page:** {resp_page_num}")
                
                with st.container():
                    st.markdown("**📝 Full Response:**")
                    # Check if response relates to recommendation terms
                    rec_sentence = str(rec.get('sentence', ''))
                    if any(word.lower() in clean_resp_sentence.lower() for word in get_meaningful_words(rec_sentence)):
                        highlighted_response = highlight_meaningful_words_fixed(clean_resp_sentence, rec_sentence)
                        st.markdown(f"> {highlighted_response}")
                    else:
                        st.markdown(f"> {clean_resp_sentence}")
                    
                    if clean_resp_context and clean_resp_context != clean_resp_sentence:
                        st.markdown("**📖 Complete Context:**")
                        formatted_context = format_text_as_clean_paragraphs(clean_resp_context)
                        st.markdown(formatted_context)
                
                if j < len(responses):
                    st.markdown("---")
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                border-left: 4px solid #dc3545;
                padding: 20px;
                border-radius: 8px;
                margin: 10px 0;
                text-align: center;
            ">
                <strong>❌ No matching responses found for this recommendation</strong><br>
                <small>This recommendation may be awaiting a response or responses may be in separate documents</small>
            </div>
            """, unsafe_allow_html=True)

def display_manual_search_results_beautiful(matches: List[Dict], target_sentence: str, 
                                          search_time: float, show_scores: bool, search_mode: str):
    """Display manual search results with FIXED highlighting"""
    
    if not matches:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left: 4px solid #dc3545;
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
        ">
            <h4 style="margin: 0 0 10px 0; color: #721c24;">No matches found</h4>
            <p style="margin: 0; color: #721c24;">
                No similar content found for your sentence.<br>
                <small>💡 Try lowering the similarity threshold or changing the search type.</small>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get meaningful words for display
    meaningful_words = get_meaningful_words(target_sentence)
    
    # Summary
    mode_text = {
        "recommendations": "recommendations",
        "responses": "responses", 
        "similar content": "sentences"
    }.get(search_mode, "items")
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0 0 10px 0;">🎯 Similar Content Found</h3>
        <p style="margin: 0; font-size: 18px;">
            Found <strong>{len(matches)}</strong> similar {mode_text} in <strong>{search_time:.3f}</strong> seconds
        </p>
        {f'<br><small style="opacity: 0.8;">Based on meaningful words: {", ".join(meaningful_words[:5])}{"..." if len(meaningful_words) > 5 else ""}</small>' if meaningful_words else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Show original sentence
    st.markdown("### 📝 Your Original Sentence")
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        font-size: 16px;
        line-height: 1.8;
        font-style: italic;
    ">
        "{target_sentence}"
    </div>
    """, unsafe_allow_html=True)
    
    # Display matches
    st.markdown("### 🔍 Similar Content Matches")
    
    for i, match in enumerate(matches, 1):
        
        # Confidence indicator
        similarity = match.get('similarity_score', 0)
        if similarity > 0.8:
            confidence_text = "Very High Similarity"
            confidence_icon = "🟢"
        elif similarity > 0.6:
            confidence_text = "High Similarity"
            confidence_icon = "🟡"
        elif similarity > 0.4:
            confidence_text = "Medium Similarity"
            confidence_icon = "🟠"
        else:
            confidence_text = "Lower Similarity"
            confidence_icon = "🔴"
        
        score_text = f" (Score: {similarity:.3f})" if show_scores else ""
        content_type = match.get('content_type', 'General')
        
        with st.expander(f"{confidence_icon} Match {i} - {content_type} - {confidence_text}{score_text}", 
                        expanded=i <= 3):
            
            # Show meaningful words matched
            if 'matched_meaningful_words' in match:
                matched_words = match['matched_meaningful_words']
                total_words = match.get('total_meaningful_words', 0)
                
                if matched_words:
                    st.markdown(f"""
                    <div style="
                        background: #e8f5e8;
                        border-left: 4px solid #4caf50;
                        padding: 12px;
                        border-radius: 6px;
                        margin: 10px 0;
                        font-size: 14px;
                    ">
                        <strong>🎯 Meaningful Words Matched:</strong> {', '.join(matched_words)} 
                        ({len(matched_words)}/{total_words} words)
                    </div>
                    """, unsafe_allow_html=True)
            
            # Document information
            doc_name = match.get('document', {}).get('filename', 'Unknown')
            page_num = match.get('page_number', 1)
            
            st.info(f"📄 **File:** {doc_name} | **Page:** {page_num} | **Type:** {content_type}")
            
            # Display found sentence with FIXED highlighting
            full_sentence = match.get('sentence', 'No sentence available')
            st.markdown("#### 📄 Complete Found Content")
            
            if show_scores:  
                highlighted_sentence = highlight_meaningful_words_fixed(full_sentence, target_sentence)
                st.markdown(highlighted_sentence)
            else:
                st.markdown(f"> {full_sentence}")
            
            # Display context with FIXED highlighting
            full_context = match.get('context', '')
            if full_context and full_context != full_sentence:
                st.markdown("#### 📖 Complete Context")
                
                if show_scores:
                    highlighted_context = highlight_meaningful_words_fixed(full_context, target_sentence)
                    clean_context = clean_html_artifacts(highlighted_context)
                    formatted_context = format_text_as_clean_paragraphs(clean_context)
                else:
                    clean_context = clean_html_artifacts(full_context)
                    formatted_context = format_text_as_clean_paragraphs(clean_context)
                
                st.markdown(formatted_context)

def show_alignment_feature_info_beautiful():
    """Show alignment feature information"""
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    ">
        <h2 style="margin: 0 0 20px 0; text-align: center;">
            🎯 Recommendation-Response Alignment
        </h2>
        <p style="margin: 0; font-size: 18px; text-align: center; opacity: 0.9;">
            Automatically discover and align government recommendations with their corresponding responses
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 10px 0;
            height: 300px;
        ">
            <h3 style="margin: 0 0 15px 0;">🔍 What It Finds</h3>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>All recommendations in your documents</li>
                <li>Corresponding responses to those recommendations</li>
                <li>Alignment confidence scores</li>
                <li>Content relationships and patterns</li>
                <li>Cross-document connections</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 10px 0;
            height: 300px;
        ">
            <h3 style="margin: 0 0 15px 0;">📊 What You Get</h3>
            <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>Side-by-side recommendation + response view</li>
                <li>Pattern analysis and statistics</li>
                <li>Confidence scores for each alignment</li>
                <li>Export options for further analysis</li>
                <li>Document relationship insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# EXPORT AND UTILITY FUNCTIONS
# =============================================================================

def copy_results_beautiful(results: List[Dict], query: str):
    """Copy results with formatting"""
    
    meaningful_words = get_meaningful_words(query)
    
    output = f"""
DAPHNE AI - SEARCH RESULTS REPORT
==================================

Search Query: "{query}"
Meaningful Words: {', '.join(meaningful_words) if meaningful_words else 'None'}
Total Results: {len(results)}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    # Group by document
    doc_groups = {}
    for result in results:
        doc_name = result['document']['filename']
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(result)
    
    for doc_name, doc_results in doc_groups.items():
        output += f"\n📄 DOCUMENT: {doc_name}\n"
        output += f"{'=' * (len(doc_name) + 15)}\n"
        output += f"Total Matches: {len(doc_results)}\n\n"
        
        for i, result in enumerate(doc_results, 1):
            score = result.get('score', 0)
            method = result.get('match_type', 'unknown').title()
            page = result.get('page_number', 1)
            context = result.get('context', '')
            
            output += f"Match {i} - {method} (Score: {score:.1f}) - Page {page}\n"
            output += f"{'-' * 50}\n"
            
            # Include FULL context without truncation
            output += f"Content:\n{context}\n\n"
    
    # Display in a beautiful code block
    st.markdown("### 📋 Complete Results Report")
    st.code(output, language="text")
    st.success("✅ Complete results displayed above! Use Ctrl+A, Ctrl+C to copy to clipboard")
    
    # Also provide download option
    safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_query = safe_query.replace(' ', '_')[:20]
    
    st.download_button(
        label="📥 Download Complete Report",
        data=output,
        file_name=f"daphne_search_results_{safe_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def export_results_csv_beautiful(results: List[Dict], query: str):
    """Export results with complete content"""
    
    csv_data = []
    meaningful_words = get_meaningful_words(query)
    
    for i, result in enumerate(results, 1):
        # Include FULL content without truncation
        row = {
            'Match_Number': i,
            'Query': query,
            'Meaningful_Words': ', '.join(meaningful_words),
            'Document': result['document']['filename'],
            'Match_Type': result.get('match_type', 'Unknown'),
            'Score': result.get('score', 0),
            'Page_Number': result.get('page_number', 1),
            'Position': result.get('position', 0),
            'Percentage_Through': result.get('percentage_through', 0),
            'Complete_Context': result.get('context', ''),  # FULL context
            'Word_Count': len(result.get('context', '').split()),
            'Character_Count': len(result.get('context', ''))
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    
    # Generate filename
    safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_query = safe_query.replace(' ', '_')[:20]
    filename = f"daphne_results_{safe_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    st.download_button(
        label="📥 Download Complete CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )
    
    st.success(f"✅ CSV ready for download with complete content! ({len(results)} results)")
    
    # Show preview
    with st.expander("📊 CSV Preview"):
        st.dataframe(df[['Match_Number', 'Document', 'Score', 'Page_Number', 'Word_Count']].head())

def format_as_beautiful_paragraphs(text: str) -> str:
    """Format text as beautiful, properly spaced paragraphs"""
    
    if not text:
        return "No content available"
    
    # Clean up the text
    text = text.strip()
    
    # Split into sentences and group into paragraphs
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return text
    
    # Group sentences into paragraphs (every 3-4 sentences or at natural breaks)
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        
        # Natural paragraph breaks
        if (len(current_paragraph) >= 3 and 
            any(indicator in sentence.lower() for indicator in 
                ['however', 'furthermore', 'additionally', 'in conclusion', 'therefore', 'moreover'])):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
        elif len(current_paragraph) >= 4:  # Max 4 sentences per paragraph
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    # Add remaining sentences
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Format as paragraphs with beautiful spacing
    formatted_paragraphs = []
    for paragraph in paragraphs:
        if paragraph.strip():
            formatted_paragraphs.append(f'> {paragraph.strip()}')
    
    return '\n\n'.join(formatted_paragraphs) if formatted_paragraphs else text

# =============================================================================
# TESTING AND DEBUGGING FUNCTIONS
# =============================================================================

def test_highlighting_syntax():
    """Test highlighting to ensure it works correctly"""
    
    st.markdown("### 🧪 Highlighting Test")
    
    test_text = "The committee recommends implementing new policies for government agencies and departments."
    test_query = "recommend policy government"
    
    st.markdown("**Original text:**")
    st.code(test_text)
    
    st.markdown("**Query:**")
    st.code(test_query)
    
    st.markdown("**Meaningful words extracted:**")
    meaningful_words = get_meaningful_words(test_query)
    st.write(meaningful_words)
    
    st.markdown("**With Streamlit highlighting:**")
    highlighted = highlight_meaningful_words_fixed(test_text, test_query)
    st.markdown(highlighted)
    
    st.markdown("**With government terms highlighting:**")
    gov_highlighted = highlight_recommendation_terms_fixed(test_text)
    st.markdown(gov_highlighted)
    
    # Test different color options
    st.markdown("### 🎨 Color Options Test")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Yellow Background:**")
        st.markdown(":yellow-background[This text has yellow background]")
    
    with col2:
        st.markdown("**Blue Background:**")
        st.markdown(":blue-background[This text has blue background]")
    
    with col3:
        st.markdown("**Green Background:**")
        st.markdown(":green-background[This text has green background]")

def debug_highlighting_issues(text: str, query: str):
    """Debug highlighting issues for troubleshooting"""
    
    st.markdown("### 🔧 Highlighting Debug Info")
    
    # Show step by step process
    st.markdown("**1. Original Text:**")
    st.code(text)
    
    st.markdown("**2. Query:**")
    st.code(query)
    
    st.markdown("**3. Meaningful Words:**")
    meaningful_words = get_meaningful_words(query)
    st.write(meaningful_words)
    
    st.markdown("**4. Pattern Matching:**")
    for word in meaningful_words[:3]:  # Show first 3 words
        pattern = r'\b' + re.escape(word) + r'\w*'
        matches = re.findall(pattern, text, re.IGNORECASE)
        st.write(f"Word '{word}' pattern '{pattern}' found: {matches}")
    
    st.markdown("**5. Final Highlighted Result:**")
    highlighted = highlight_meaningful_words_fixed(text, query)
    st.markdown(highlighted)
    
    st.markdown("**6. Character-by-character comparison:**")
    if highlighted != text:
        st.success("✅ Highlighting was applied!")
        st.code(f"Original:    {text}")
        st.code(f"Highlighted: {highlighted}")
    else:
        st.warning("⚠️ No highlighting was applied - no matches found")

# =============================================================================
# ALTERNATIVE HIGHLIGHTING METHODS (FALLBACK OPTIONS)
# =============================================================================

def highlight_with_html_fallback(text: str, query: str) -> str:
    """Alternative highlighting using HTML for fallback if Streamlit colors don't work"""
    
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return text
    
    highlighted = text
    meaningful_words.sort(key=len, reverse=True)
    
    for word in meaningful_words:
        if len(word) > 2:
            try:
                pattern = re.compile(r'\b' + re.escape(word) + r'\w*', re.IGNORECASE)
                matches = pattern.findall(highlighted)
                
                for match in set(matches):
                    if match.lower().startswith(word.lower()):
                        # HTML highlighting with inline styles
                        highlighted = highlighted.replace(
                            match, 
                            f'<mark style="background-color: #FFEB3B; padding: 2px; border-radius: 2px; font-weight: bold;">{match}</mark>'
                        )
                        
            except re.error:
                continue
    
    return highlighted

def highlight_with_bold_fallback(text: str, query: str) -> str:
    """Simple bold highlighting as last resort fallback"""
    
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return text
    
    highlighted = text
    meaningful_words.sort(key=len, reverse=True)
    
    for word in meaningful_words:
        if len(word) > 2:
            try:
                pattern = re.compile(r'\b' + re.escape(word) + r'\w*', re.IGNORECASE)
                matches = pattern.findall(highlighted)
                
                for match in set(matches):
                    if match.lower().startswith(word.lower()):
                        # Simple bold highlighting
                        highlighted = highlighted.replace(match, f'**{match}**')
                        
            except re.error:
                continue
    
    return highlighted

def smart_highlight_with_fallbacks(text: str, query: str) -> tuple[str, bool]:
    """
    Smart highlighting that tries multiple methods and returns success status
    Returns: (highlighted_text, uses_html)
    """
    
    # Try Streamlit color highlighting first
    streamlit_highlighted = highlight_meaningful_words_fixed(text, query)
    
    # Check if highlighting worked (text changed)
    if streamlit_highlighted != text:
        return streamlit_highlighted, False
    
    # If no highlighting occurred, try HTML fallback
    html_highlighted = highlight_with_html_fallback(text, query)
    
    if html_highlighted != text:
        return html_highlighted, True
    
    # Last resort: bold highlighting
    bold_highlighted = highlight_with_bold_fallback(text, query)
    return bold_highlighted, False

# =============================================================================
# ENHANCED DISPLAY FUNCTIONS WITH SMART HIGHLIGHTING
# =============================================================================

def display_context_with_smart_highlighting(context: str, query: str, highlight_matches: bool):
    """Display context with smart highlighting that uses fallbacks"""
    
    if not context:
        st.markdown("*No context available*")
        return
    
    if highlight_matches:
        # Use smart highlighting with fallbacks
        highlighted_context, uses_html = smart_highlight_with_fallbacks(context, query)
        clean_content = clean_html_artifacts(highlighted_context)
        formatted_content = format_text_as_clean_paragraphs(clean_content)
        
        # Display with appropriate method
        if uses_html:
            st.markdown(formatted_content, unsafe_allow_html=True)
        else:
            st.markdown(formatted_content)
    else:
        clean_content = clean_html_artifacts(context)
        formatted_content = format_text_as_clean_paragraphs(clean_content)
        st.markdown(formatted_content)

# Export all functions
__all__ = [
    'display_search_results_beautiful',
    'display_single_search_result_beautiful', 
    'display_alignment_results_beautiful',
    'display_single_alignment_beautiful',
    'display_manual_search_results_beautiful',
    'show_alignment_feature_info_beautiful',
    'format_as_beautiful_paragraphs',
    'highlight_recommendation_terms_fixed',
    'highlight_meaningful_words_fixed',
    'copy_results_beautiful',
    'export_results_csv_beautiful',
    'get_meaningful_words',
    'clean_html_artifacts',
    'format_text_as_clean_paragraphs',
    'test_highlighting_syntax',
    'debug_highlighting_issues',
    'smart_highlight_with_fallbacks',
    'display_context_with_smart_highlighting'
]
