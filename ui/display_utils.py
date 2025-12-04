# modules/ui/display_utils.py - COMPLETELY FIXED HIGHLIGHTING SYSTEM
"""
Beautiful display functions for DaphneAI search results and alignments.
COMPLETELY REVISED - Uses multiple highlighting methods with proper fallbacks.
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Tuple

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
# UTILITY FUNCTIONS
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
# COMPLETELY REVISED HIGHLIGHTING SYSTEM
# =============================================================================

def highlight_with_html_method(text: str, words_to_highlight: List[str]) -> str:
    """Method 1: HTML highlighting with inline styles"""
    
    if not words_to_highlight:
        return text
    
    highlighted = text
    words_to_highlight.sort(key=len, reverse=True)  # Longest first to avoid conflicts
    
    for word in words_to_highlight:
        if len(word) > 2:
            try:
                # Create pattern for word boundaries
                pattern = re.compile(r'\b' + re.escape(word) + r'\w*', re.IGNORECASE)
                
                def replace_match(match):
                    matched_word = match.group()
                    return f'<mark style="background-color: #FFEB3B; padding: 2px; border-radius: 2px; font-weight: bold; color: #000;">{matched_word}</mark>'
                
                highlighted = pattern.sub(replace_match, highlighted)
                
            except re.error:
                continue
    
    return highlighted

def highlight_with_bold_method(text: str, words_to_highlight: List[str]) -> str:
    """Method 2: Bold highlighting using Markdown"""
    
    if not words_to_highlight:
        return text
    
    highlighted = text
    words_to_highlight.sort(key=len, reverse=True)
    
    for word in words_to_highlight:
        if len(word) > 2:
            try:
                pattern = re.compile(r'\b' + re.escape(word) + r'\w*', re.IGNORECASE)
                
                def replace_match(match):
                    matched_word = match.group()
                    return f'**{matched_word}**'
                
                highlighted = pattern.sub(replace_match, highlighted)
                
            except re.error:
                continue
    
    return highlighted

def highlight_with_capitalization_method(text: str, words_to_highlight: List[str]) -> str:
    """Method 3: Capitalization highlighting"""
    
    if not words_to_highlight:
        return text
    
    highlighted = text
    words_to_highlight.sort(key=len, reverse=True)
    
    for word in words_to_highlight:
        if len(word) > 2:
            try:
                pattern = re.compile(r'\b' + re.escape(word) + r'\w*', re.IGNORECASE)
                
                def replace_match(match):
                    matched_word = match.group()
                    return f'**{matched_word.upper()}**'
                
                highlighted = pattern.sub(replace_match, highlighted)
                
            except re.error:
                continue
    
    return highlighted

def highlight_with_brackets_method(text: str, words_to_highlight: List[str]) -> str:
    """Method 4: Bracket highlighting (most reliable fallback)"""
    
    if not words_to_highlight:
        return text
    
    highlighted = text
    words_to_highlight.sort(key=len, reverse=True)
    
    for word in words_to_highlight:
        if len(word) > 2:
            try:
                pattern = re.compile(r'\b' + re.escape(word) + r'\w*', re.IGNORECASE)
                
                def replace_match(match):
                    matched_word = match.group()
                    return f'[{matched_word}]'
                
                highlighted = pattern.sub(replace_match, highlighted)
                
            except re.error:
                continue
    
    return highlighted

def smart_highlight_text(text: str, query: str, method_preference: str = "auto") -> Tuple[str, str]:
    """
    Smart highlighting that tries multiple methods and returns the best result.
    Returns: (highlighted_text, method_used)
    """
    
    if not text or not query:
        return text, "none"
    
    # Get meaningful words to highlight
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return text, "none"
    
    # Try different highlighting methods based on preference
    if method_preference == "html" or method_preference == "auto":
        highlighted = highlight_with_html_method(text, meaningful_words)
        if highlighted != text:
            return highlighted, "html"
    
    if method_preference == "bold" or method_preference == "auto":
        highlighted = highlight_with_bold_method(text, meaningful_words)
        if highlighted != text:
            return highlighted, "bold"
    
    if method_preference == "caps" or method_preference == "auto":
        highlighted = highlight_with_capitalization_method(text, meaningful_words)
        if highlighted != text:
            return highlighted, "caps"
    
    # Fallback to brackets (most reliable)
    highlighted = highlight_with_brackets_method(text, meaningful_words)
    return highlighted, "brackets"

def highlight_government_terms(text: str) -> Tuple[str, str]:
    """Highlight government-specific terms"""
    
    gov_terms = [
        'recommend', 'recommendation', 'recommendations', 'suggest', 'advise', 'propose',
        'accept', 'reject', 'agree', 'disagree', 'implement', 'implementation', 
        'consider', 'approved', 'declined', 'response', 'reply', 'answer',
        'policy', 'framework', 'guideline', 'protocol', 'strategy',
        'committee', 'department', 'ministry', 'government', 'authority',
        'urgent', 'immediate', 'critical', 'priority', 'essential',
        'budget', 'funding', 'financial', 'cost', 'expenditure',
        'review', 'analysis', 'assessment', 'evaluation', 'inquiry'
    ]
    
    # Try HTML first, then fallback
    highlighted = highlight_with_html_method(text, gov_terms)
    if highlighted != text:
        return highlighted, "html"
    
    # Fallback to bold
    highlighted = highlight_with_bold_method(text, gov_terms)
    if highlighted != text:
        return highlighted, "bold"
    
    # Final fallback
    highlighted = highlight_with_brackets_method(text, gov_terms)
    return highlighted, "brackets"

# =============================================================================
# DISPLAY FUNCTIONS WITH FIXED HIGHLIGHTING
# =============================================================================

def display_search_results_beautiful(results: List[Dict], query: str, search_time: float, 
                                   show_context: bool, highlight_matches: bool):
    """Display search results with WORKING highlighting"""
    
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
    """Display a single search result with WORKING highlighting"""
    
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
    
    # Display context with WORKING highlighting
    if show_context:
        full_context = result.get('context', '')
        if full_context:
            
            st.markdown("**📖 Complete Context:**")
            
            # Apply highlighting with multiple method support
            if highlight_matches:
                highlighted_context, method_used = smart_highlight_text(full_context, query, "auto")
                
                # Display based on the method that worked
                if method_used == "html":
                    st.markdown(highlighted_context, unsafe_allow_html=True)
                elif method_used == "bold" or method_used == "caps":
                    # Format as paragraphs first
                    formatted_content = format_text_as_clean_paragraphs(highlighted_context)
                    st.markdown(formatted_content)
                elif method_used == "brackets":
                    # Show with brackets and explanation
                    formatted_content = format_text_as_clean_paragraphs(highlighted_context)
                    st.markdown(formatted_content)
                    st.caption("💡 Highlighted words are shown in [brackets]")
                else:
                    # No highlighting worked, just show clean text
                    clean_content = clean_html_artifacts(full_context)
                    formatted_content = format_text_as_clean_paragraphs(clean_content)
                    st.markdown(formatted_content)
            else:
                clean_content = clean_html_artifacts(full_context)
                formatted_content = format_text_as_clean_paragraphs(clean_content)
                st.markdown(formatted_content)
            
            # Additional details
            percentage = result.get('percentage_through', 0)
            st.caption(f"💡 This content appears around page {page}, {percentage:.1f}% through the document")

def display_manual_search_results_beautiful(matches: List[Dict], target_sentence: str, 
                                          search_time: float, show_scores: bool, search_mode: str):
    """Display manual search results with WORKING highlighting"""
    
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
            
            # Display found sentence with WORKING highlighting
            full_sentence = match.get('sentence', 'No sentence available')
            st.markdown("#### 📄 Complete Found Content")
            
            if show_scores:  
                highlighted_sentence, sentence_method = smart_highlight_text(full_sentence, target_sentence, "auto")
                
                if sentence_method == "html":
                    st.markdown(highlighted_sentence, unsafe_allow_html=True)
                elif sentence_method in ["bold", "caps"]:
                    st.markdown(highlighted_sentence)
                elif sentence_method == "brackets":
                    st.markdown(highlighted_sentence)
                    st.caption("💡 Matched terms are shown in [brackets]")
                else:
                    st.markdown(f"> {full_sentence}")
            else:
                st.markdown(f"> {full_sentence}")
            
            # Display context with WORKING highlighting
            full_context = match.get('context', '')
            if full_context and full_context != full_sentence:
                st.markdown("#### 📖 Complete Context")
                
                if show_scores:
                    highlighted_context, context_method = smart_highlight_text(full_context, target_sentence, "auto")
                    
                    if context_method == "html":
                        clean_context = clean_html_artifacts(highlighted_context)
                        formatted_context = format_text_as_clean_paragraphs(clean_context)
                        st.markdown(formatted_context, unsafe_allow_html=True)
                    elif context_method in ["bold", "caps", "brackets"]:
                        clean_context = clean_html_artifacts(highlighted_context)
                        formatted_context = format_text_as_clean_paragraphs(clean_context)
                        st.markdown(formatted_context)
                        if context_method == "brackets":
                            st.caption("💡 Matched terms are shown in [brackets]")
                    else:
                        clean_context = clean_html_artifacts(full_context)
                        formatted_context = format_text_as_clean_paragraphs(clean_context)
                        st.markdown(formatted_context)
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
            # Clean any highlighting artifacts from the context
            clean_context = clean_html_artifacts(context)
            output += f"Content:\n{clean_context}\n\n"
    
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
        context = result.get('context', '')
        clean_context = clean_html_artifacts(context)  # Remove any highlighting artifacts
        
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
            'Complete_Context': clean_context,  # FULL clean context
            'Word_Count': len(clean_context.split()),
            'Character_Count': len(clean_context)
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
    
    # Clean up the text first
    text = clean_html_artifacts(text)
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

def test_highlighting_methods():
    """Test all highlighting methods to see which ones work"""
    
    st.markdown("### 🧪 Highlighting Method Test")
    
    test_text = "The committee recommends implementing new policies for government response."
    test_query = "recommend policy response"
    
    st.markdown("**Original text:**")
    st.code(test_text)
    
    st.markdown("**Query:**")
    st.code(test_query)
    
    meaningful_words = get_meaningful_words(test_query)
    st.markdown(f"**Meaningful words:** {meaningful_words}")
    
    # Test each method
    st.markdown("### 🎨 Testing Different Highlighting Methods")
    
    # Method 1: HTML
    st.markdown("**Method 1: HTML Highlighting**")
    html_result = highlight_with_html_method(test_text, meaningful_words)
    st.markdown(html_result, unsafe_allow_html=True)
    st.code(f"Result: {html_result}")
    
    # Method 2: Bold
    st.markdown("**Method 2: Bold Highlighting**")
    bold_result = highlight_with_bold_method(test_text, meaningful_words)
    st.markdown(bold_result)
    st.code(f"Result: {bold_result}")
    
    # Method 3: Capitalization
    st.markdown("**Method 3: Capitalization Highlighting**")
    caps_result = highlight_with_capitalization_method(test_text, meaningful_words)
    st.markdown(caps_result)
    st.code(f"Result: {caps_result}")
    
    # Method 4: Brackets
    st.markdown("**Method 4: Bracket Highlighting**")
    bracket_result = highlight_with_brackets_method(test_text, meaningful_words)
    st.markdown(bracket_result)
    st.code(f"Result: {bracket_result}")
    
    # Smart method
    st.markdown("**Smart Method (Auto-Select):**")
    smart_result, method_used = smart_highlight_text(test_text, test_query, "auto")
    st.markdown(f"**Method used:** {method_used}")
    
    if method_used == "html":
        st.markdown(smart_result, unsafe_allow_html=True)
    else:
        st.markdown(smart_result)
    
    st.code(f"Result: {smart_result}")

def debug_highlighting_issue(problematic_text: str, query: str):
    """Debug specific highlighting issues"""
    
    st.markdown("### 🔧 Debugging Highlighting Issue")
    
    st.markdown("**Problematic Text:**")
    st.code(problematic_text)
    
    st.markdown("**Query:**")
    st.code(query)
    
    # Check for meaningful words
    meaningful_words = get_meaningful_words(query)
    st.markdown(f"**Meaningful words extracted:** {meaningful_words}")
    
    if not meaningful_words:
        st.error("❌ No meaningful words found in query!")
        return
    
    # Test step by step
    st.markdown("### Step-by-Step Analysis")
    
    for i, word in enumerate(meaningful_words, 1):
        st.markdown(f"**Step {i}: Testing word '{word}'**")
        
        # Check if word exists in text
        if word.lower() in problematic_text.lower():
            st.success(f"✅ Word '{word}' found in text")
            
            # Try each highlighting method
            html_test = highlight_with_html_method(problematic_text, [word])
            bold_test = highlight_with_bold_method(problematic_text, [word])
            bracket_test = highlight_with_brackets_method(problematic_text, [word])
            
            st.code(f"HTML result: {html_test}")
            st.code(f"Bold result: {bold_test}")
            st.code(f"Bracket result: {bracket_test}")
            
            # Show if highlighting worked
            if html_test != problematic_text:
                st.info("✅ HTML highlighting worked")
            if bold_test != problematic_text:
                st.info("✅ Bold highlighting worked")
            if bracket_test != problematic_text:
                st.info("✅ Bracket highlighting worked")
                
        else:
            st.warning(f"⚠️ Word '{word}' not found in text")
    
    # Test smart highlighting
    st.markdown("### Final Smart Highlighting Test")
    final_result, method_used = smart_highlight_text(problematic_text, query, "auto")
    
    st.markdown(f"**Method selected:** {method_used}")
    st.markdown(f"**Final result:**")
    
    if method_used == "html":
        st.markdown(final_result, unsafe_allow_html=True)
    else:
        st.markdown(final_result)
    
    st.code(f"Raw result: {final_result}")
    
    if final_result == problematic_text:
        st.error("❌ No highlighting was applied!")
        st.markdown("**Possible issues:**")
        st.markdown("- No meaningful words found")
        st.markdown("- Words don't match with word boundaries")
        st.markdown("- Text encoding issues")
        st.markdown("- Regular expression issues")
    else:
        st.success("✅ Highlighting was successfully applied!")

# =============================================================================
# ADVANCED HIGHLIGHTING FEATURES
# =============================================================================

def highlight_with_color_codes(text: str, words_to_highlight: List[str], color_scheme: str = "default") -> str:
    """Advanced highlighting with different color schemes"""
    
    if not words_to_highlight:
        return text
    
    # Color schemes
    color_schemes = {
        "default": {"bg": "#FFEB3B", "text": "#000"},
        "blue": {"bg": "#2196F3", "text": "#FFF"},
        "green": {"bg": "#4CAF50", "text": "#FFF"},
        "red": {"bg": "#F44336", "text": "#FFF"},
        "purple": {"bg": "#9C27B0", "text": "#FFF"}
    }
    
    colors = color_schemes.get(color_scheme, color_schemes["default"])
    
    highlighted = text
    words_to_highlight.sort(key=len, reverse=True)
    
    for word in words_to_highlight:
        if len(word) > 2:
            try:
                pattern = re.compile(r'\b' + re.escape(word) + r'\w*', re.IGNORECASE)
                
                def replace_match(match):
                    matched_word = match.group()
                    return f'<mark style="background-color: {colors["bg"]}; color: {colors["text"]}; padding: 2px; border-radius: 2px; font-weight: bold;">{matched_word}</mark>'
                
                highlighted = pattern.sub(replace_match, highlighted)
                
            except re.error:
                continue
    
    return highlighted

def create_highlighting_legend(words_highlighted: List[str], method_used: str) -> str:
    """Create a legend explaining the highlighting"""
    
    if not words_highlighted:
        return ""
    
    legend_text = "💡 **Highlighting Legend:** "
    
    if method_used == "html":
        legend_text += "Highlighted words are shown with colored background"
    elif method_used == "bold":
        legend_text += "**Highlighted words are shown in bold**"
    elif method_used == "caps":
        legend_text += "**HIGHLIGHTED WORDS ARE SHOWN IN CAPITALS**"
    elif method_used == "brackets":
        legend_text += "Highlighted words are shown in [brackets]"
    else:
        legend_text += "Highlighted words are emphasized"
    
    legend_text += f" | Words found: {', '.join(words_highlighted[:5])}{'...' if len(words_highlighted) > 5 else ''}"
    
    return legend_text

def highlight_with_context_awareness(text: str, query: str, context_type: str = "general") -> Tuple[str, str, List[str]]:
    """Context-aware highlighting that adapts based on content type"""
    
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return text, "none", []
    
    # Choose highlighting strategy based on context
    if context_type == "recommendation":
        # Use green for recommendations
        highlighted = highlight_with_color_codes(text, meaningful_words, "green")
        if highlighted != text:
            return highlighted, "html_green", meaningful_words
    elif context_type == "response":
        # Use blue for responses
        highlighted = highlight_with_color_codes(text, meaningful_words, "blue")
        if highlighted != text:
            return highlighted, "html_blue", meaningful_words
    elif context_type == "policy":
        # Use purple for policy content
        highlighted = highlight_with_color_codes(text, meaningful_words, "purple")
        if highlighted != text:
            return highlighted, "html_purple", meaningful_words
    
    # Fallback to standard highlighting
    highlighted, method_used = smart_highlight_text(text, query, "auto")
    return highlighted, method_used, meaningful_words

def render_highlighted_content_with_legend(content: str, query: str, context_type: str = "general", 
                                         show_legend: bool = True) -> None:
    """Render highlighted content with optional legend"""
    
    highlighted_content, method_used, words_found = highlight_with_context_awareness(content, query, context_type)
    
    # Display the content
    if method_used.startswith("html"):
        st.markdown(highlighted_content, unsafe_allow_html=True)
    elif method_used in ["bold", "caps", "brackets"]:
        st.markdown(highlighted_content)
    else:
        st.markdown(content)
    
    # Show legend if requested
    if show_legend and words_found and method_used != "none":
        legend = create_highlighting_legend(words_found, method_used)
        st.caption(legend)

# -----------------------------------------------------------------------------
# Alignment display re-exports (to keep legacy imports working while we refactor)
# -----------------------------------------------------------------------------
from .alignment_display import (  # noqa: E402
    display_alignment_results_beautiful,
    display_single_alignment_beautiful,
    show_alignment_feature_info_beautiful,
    highlight_government_terms,
)

# Export all functions (COMPLETE LIST)
__all__ = [
    # Main display functions
    'display_search_results_beautiful',
    'display_single_search_result_beautiful', 
    'display_alignment_results_beautiful',
    'display_single_alignment_beautiful',
    'display_manual_search_results_beautiful',
    'show_alignment_feature_info_beautiful',
    
    # Core highlighting functions
    'smart_highlight_text',
    'highlight_government_terms',
    'highlight_with_html_method',
    'highlight_with_bold_method',
    'highlight_with_capitalization_method',
    'highlight_with_brackets_method',
    
    # Advanced highlighting
    'highlight_with_color_codes',
    'highlight_with_context_awareness',
    'render_highlighted_content_with_legend',
    'create_highlighting_legend',
    
    # Utility functions
    'get_meaningful_words',
    'clean_html_artifacts',
    'format_text_as_clean_paragraphs',
    'format_as_beautiful_paragraphs',
    
    # Export functions
    'copy_results_beautiful',
    'export_results_csv_beautiful'
]
