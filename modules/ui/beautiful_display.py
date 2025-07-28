# modules/ui/beautiful_display.py - Beautiful Display Functions with Stop Word Filtering
"""
Beautiful display functions for DaphneAI search results and alignments.
This file contains all the formatting and presentation logic with complete paragraphs.
Properly handles stop word filtering for clean, meaningful results.
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List

# STOP WORDS for highlighting (same as search_components)
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 
    'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 
    'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 
    'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 
    'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 
    'come', 'made', 'may', 'part', 'or', 'also', 'back', 'any', 'good', 
    'new', 'where', 'much', 'take', 'know', 'just', 'see', 'after', 
    'very', 'well', 'here', 'should', 'old', 'still'
}

def display_search_results_beautiful(results: List[Dict], query: str, search_time: float, 
                                   show_context: bool, highlight_matches: bool):
    """Display search results with beautiful paragraph formatting and proper filtering"""
    
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
    
    # Beautiful summary with filtering info
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
    
    # Display results with beautiful formatting
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
            
            # Display each match beautifully
            for i, result in enumerate(doc_results, 1):
                display_single_search_result_beautiful(result, i, query, show_context, highlight_matches)

def display_single_search_result_beautiful(result: Dict, index: int, query: str, 
                                         show_context: bool, highlight_matches: bool):
    """Display a single search result with beautiful formatting and filtering info"""
    
    method = result.get('match_type', 'unknown')
    score = result.get('score', 0)
    page = result.get('page_number', 1)
    position = result.get('position', 0)
    
    # Method icons and colors
    method_info = {
        'exact': {'icon': '🎯', 'color': '#28a745', 'name': 'Exact Match'},
        'smart': {'icon': '🧠', 'color': '#007bff', 'name': 'Smart Search'},
        'fuzzy': {'icon': '🌀', 'color': '#ffc107', 'name': 'Fuzzy Match'},
        'semantic': {'icon': '🤖', 'color': '#6f42c1', 'name': 'Semantic Match'}
    }
    
    info = method_info.get(method, {'icon': '🔍', 'color': '#6c757d', 'name': 'Search'})
    
    # Beautiful match header
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
    
    # Show meaningful words found (for smart search)
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
    
    # Display FULL context beautifully
    if show_context:
        full_context = result.get('context', '')
        if full_context:
            
            # Apply highlighting if requested (only meaningful words) - FIXED
            if highlight_matches:
                display_content = highlight_meaningful_words_only(full_context, query)
            else:
                display_content = full_context
            
            # Clean and format the content
            clean_content = clean_html_artifacts(display_content)
            formatted_content = format_text_as_clean_paragraphs(clean_content)
            
            # Display with highlighting preserved
            st.markdown("**📖 Complete Context:**")
            st.markdown(formatted_content)
            
            # Additional match details
            percentage = result.get('percentage_through', 0)
            st.caption(f"💡 This content appears around page {page}, {percentage:.1f}% through the document")

def display_alignment_results_beautiful(alignments: List[Dict], show_ai_summaries: bool):
    """Display alignment results with beautiful paragraph formatting"""
    
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
    
    # Display individual alignments with beautiful formatting
    st.markdown("### 🔗 Recommendation-Response Alignments")
    
    for i, alignment in enumerate(alignments, 1):
        display_single_alignment_beautiful(alignment, i, show_ai_summaries)

def display_single_alignment_beautiful(alignment: Dict, index: int, show_ai_summaries: bool):
    """Display a single alignment with beautiful paragraph formatting"""
    
    rec = alignment.get('recommendation', {})
    responses = alignment.get('responses', [])
    confidence = alignment.get('alignment_confidence', 0)
    
    # Confidence indicator
    confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
    rec_type = rec.get('recommendation_type', 'General')
    
    with st.expander(f"{confidence_color} Recommendation {index} - {rec_type} (Confidence: {confidence:.2f})", 
                    expanded=index <= 3):
        
        # Beautiful layout
        st.markdown("### 📋 Complete Extract")
        
        # Recommendation section with full beautiful content
        st.markdown("#### 🎯 Recommendation")
        
        doc_name = rec.get('document', {}).get('filename', 'Unknown Document')
        page_num = rec.get('page_number', 1)
        st.info(f"📄 **Document:** {doc_name} | **Page:** {page_num}")
        
        # Display the FULL recommendation sentence - WITH HIGHLIGHTING
        full_sentence = rec.get('sentence', 'No sentence available')
        clean_sentence = clean_html_artifacts(full_sentence)
        
        # Use clean Streamlit formatting with highlighting for meaningful words
        st.markdown("**📝 Full Recommendation:**")
        with st.container():
            # Check if this looks like a real recommendation (has meaningful content)
            if len(clean_sentence.split()) > 5:  # More than just metadata
                # Try to highlight meaningful terms commonly found in recommendations
                highlighted_sentence = highlight_recommendation_terms(clean_sentence)
                st.markdown(f"> {highlighted_sentence}")
            else:
                st.markdown(f"> {clean_sentence}")
        
        # Add some spacing
        st.markdown("")
        
        # Show FULL context as beautiful paragraphs - WITH HIGHLIGHTING
        full_context = rec.get('context', 'No context available')
        if full_context and full_context != full_sentence:
            st.markdown("#### 📖 Complete Context")
            
            # Format context with highlighting for key terms
            clean_context = clean_html_artifacts(full_context)
            highlighted_context = highlight_recommendation_terms(clean_context)
            formatted_context = format_text_as_clean_paragraphs(highlighted_context)
            
            st.markdown(formatted_context)
        
        # Responses section with full beautiful content
        if responses:
            st.markdown("#### ↩️ Related Responses")
            
            for j, resp_match in enumerate(responses, 1):
                resp = resp_match.get('response', {})
                similarity = resp_match.get('combined_score', 0)
                
                # Color code by similarity with beautiful gradients
                if similarity > 0.7:
                    bg_gradient = "linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)"
                    border_color = "#28a745"
                    confidence_text = "High Confidence"
                elif similarity > 0.5:
                    bg_gradient = "linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)"
                    border_color = "#ffc107"
                    confidence_text = "Medium Confidence"
                else:
                    bg_gradient = "linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)"
                    border_color = "#dc3545"
                    confidence_text = "Lower Confidence"
                
                resp_doc_name = resp.get('document', {}).get('filename', 'Unknown Document')
                resp_page_num = resp.get('page_number', 1)
                
                # Display FULL response content beautifully - FIXED FORMATTING
                full_resp_sentence = resp.get('sentence', 'No sentence available')
                full_resp_context = resp.get('context', 'No context available')
                
                # Clean the text to remove HTML artifacts
                clean_resp_sentence = clean_html_artifacts(full_resp_sentence)
                clean_resp_context = clean_html_artifacts(full_resp_context)
                
                # Use Streamlit components instead of raw HTML to avoid bleeding
                st.markdown(f"**📄 Response {j} - {confidence_text} ({similarity:.2f})**")
                st.info(f"📄 **Document:** {resp_doc_name} | **Page:** {resp_page_num}")
                
                # Response content in a clean container - WITH HIGHLIGHTING
                with st.container():
                    st.markdown("**📝 Full Response:**")
                    # Apply highlighting to response if it contains meaningful words
                    if any(word.lower() in clean_resp_sentence.lower() for word in get_meaningful_words(str(rec.get('sentence', '')))):
                        highlighted_response = highlight_meaningful_words_only(clean_resp_sentence, str(rec.get('sentence', '')))
                        st.markdown(f"> {highlighted_response}")
                    else:
                        st.markdown(f"> {clean_resp_sentence}")
                    
                    if clean_resp_context and clean_resp_context != clean_resp_sentence:
                        st.markdown("**📖 Complete Context:**")
                        # Format context without HTML but with potential highlighting
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
    """Display manual search results with beautiful formatting and filtering info"""
    
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
    
    # Beautiful summary
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
    
    # Show original sentence beautifully
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
    
    # Display matches beautifully
    st.markdown("### 🔍 Similar Content Matches")
    
    for i, match in enumerate(matches, 1):
        
        # Confidence indicator with beautiful styling
        similarity = match.get('similarity_score', 0)
        if similarity > 0.8:
            confidence_color = "#28a745"
            confidence_bg = "linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)"
            confidence_text = "Very High Similarity"
            confidence_icon = "🟢"
        elif similarity > 0.6:
            confidence_color = "#ffc107"
            confidence_bg = "linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)"
            confidence_text = "High Similarity"
            confidence_icon = "🟡"
        elif similarity > 0.4:
            confidence_color = "#fd7e14"
            confidence_bg = "linear-gradient(135deg, #ffe8d1 0%, #ffd19a 100%)"
            confidence_text = "Medium Similarity"
            confidence_icon = "🟠"
        else:
            confidence_color = "#dc3545"
            confidence_bg = "linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)"
            confidence_text = "Lower Similarity"
            confidence_icon = "🔴"
        
        # Score display
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
            
            # Document information beautifully displayed
            doc_name = match.get('document', {}).get('filename', 'Unknown')
            page_num = match.get('page_number', 1)
            
            st.markdown(f"""
            <div style="
                background: {confidence_bg};
                border-left: 4px solid {confidence_color};
                padding: 20px;
                border-radius: 10px;
                margin: 15px 0;
            ">
                <h5 style="margin: 0 0 15px 0; color: #333;">
                    📄 Document Information
                </h5>
                <p style="margin: 5px 0; font-weight: 500;">
                    <strong>File:</strong> {doc_name}<br>
                    <strong>Page:</strong> {page_num}<br>
                    <strong>Content Type:</strong> {content_type}
                    {f'<br><strong>Similarity Score:</strong> {similarity:.3f}' if show_scores else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display FULL found sentence beautifully
            full_sentence = match.get('sentence', 'No sentence available')
            st.markdown("#### 📄 Complete Found Content")
            
            # Highlight meaningful words only
            highlighted_sentence = highlight_meaningful_words_only(full_sentence, target_sentence)
            
            st.markdown(f"""
            <div style="
                background: white;
                border: 2px solid {confidence_color};
                padding: 25px;
                border-radius: 10px;
                margin: 15px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 16px; line-height: 1.8; color: #333;">
                    {highlighted_sentence}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display FULL context beautifully
            full_context = match.get('context', '')
            if full_context and full_context != full_sentence:
                st.markdown("#### 📖 Complete Context")
                
                formatted_context = format_as_beautiful_paragraphs(full_context)
                
                st.markdown(f"""
                <div style="
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    padding: 25px;
                    border-radius: 10px;
                    margin: 15px 0;
                    font-size: 15px;
                    line-height: 1.7;
                    color: #495057;
                ">
                    {formatted_context}
                </div>
                """, unsafe_allow_html=True)

def show_alignment_feature_info_beautiful():
    """Show beautiful information about the alignment feature"""
    
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
    
    # Feature breakdown in beautiful cards
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
    
    # Perfect for section
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #fd7e14 0%, #dc3545 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
    ">
        <h3 style="margin: 0 0 15px 0; text-align: center;">💡 Perfect For</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
            <div>
                <strong>• Government inquiry reports</strong><br>
                <strong>• Policy documents and responses</strong>
            </div>
            <div>
                <strong>• Committee recommendations and outcomes</strong><br>
                <strong>• Audit findings and management responses</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_html_artifacts(text: str) -> str:
    """Remove HTML artifacts and clean up text display"""
    if not text:
        return "No content available"
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def format_text_as_clean_paragraphs(text: str) -> str:
    """Format text as clean paragraphs without HTML styling"""
    
    if not text:
        return "*No content available*"
    
    # Clean the text first
    clean_text = clean_html_artifacts(text)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return clean_text
    
    # Group into paragraphs (3-4 sentences each)
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        
        # Break on natural indicators or when we have enough sentences
        if (len(current_paragraph) >= 3 and 
            any(indicator in sentence.lower() for indicator in 
                ['however', 'furthermore', 'additionally', 'therefore', 'moreover', 'meanwhile'])):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
        elif len(current_paragraph) >= 4:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    # Add remaining sentences
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Format as clean markdown paragraphs
    formatted_text = '\n\n'.join(f"> {paragraph}" for paragraph in paragraphs if paragraph.strip())
    
    return formatted_text if formatted_text else f"> {clean_text}"

def get_meaningful_words(text: str) -> List[str]:
    """Extract meaningful words (non-stop words) from text"""
    words = re.findall(r'\b\w+\b', text.lower())
    meaningful = [word for word in words 
                 if word not in STOP_WORDS and len(word) > 1]
    return meaningful

def format_as_beautiful_paragraphs(text: str) -> str:
    """Format text as beautiful, properly spaced paragraphs - CLEANED VERSION"""
    
    if not text:
        return "No content available"
    
    # Clean HTML artifacts first
    clean_text = clean_html_artifacts(text)
    
    # Split into sentences and group into paragraphs
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return clean_text
    
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
    
    # Format as clean text paragraphs (no HTML to avoid bleeding)
    formatted_paragraphs = []
    for paragraph in paragraphs:
        if paragraph.strip():
            formatted_paragraphs.append(paragraph.strip())
    
    return '\n\n'.join(formatted_paragraphs) if formatted_paragraphs else clean_text

def highlight_recommendation_terms(text: str) -> str:
    """Highlight common recommendation and response terms"""
    
    # Common terms found in government recommendations and responses
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
    
    # Sort by length (longest first) to avoid partial highlighting conflicts
    highlight_terms.sort(key=len, reverse=True)
    
    for term in highlight_terms:
        if len(term) > 3:  # Only highlight meaningful terms
            # Case-insensitive highlighting using markdown
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            highlighted = pattern.sub(f'**:yellow[{term}]**', highlighted)
    
    return highlighted
    """Highlight only meaningful words from the query in the text using Streamlit markdown"""
    
    # Get meaningful words from query (filter out stop words)
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return text
    
    highlighted = text
    
    # Sort by length (longest first) to avoid partial highlighting conflicts
    meaningful_words.sort(key=len, reverse=True)
    
    for word in meaningful_words:
        if len(word) > 1:  # Skip very short words
            # Case-insensitive highlighting using markdown bold + color
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            highlighted = pattern.sub(f'**:yellow[{word}]**', highlighted)
    
    return highlighted

def highlight_meaningful_words_only(text: str, query: str) -> str:
    """Alternative highlighting using Streamlit color syntax"""
    
    meaningful_words = get_meaningful_words(query)
    
    if not meaningful_words:
        return text
    
    highlighted = text
    
    # Sort by length (longest first)
    meaningful_words.sort(key=len, reverse=True)
    
    for word in meaningful_words:
        if len(word) > 1:
            # Use Streamlit's color syntax for highlighting
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            highlighted = pattern.sub(f':yellow-background[**{word}**]', highlighted)
    
    return highlighted

def copy_results_beautiful(results: List[Dict], query: str):
    """Copy results with beautiful formatting and filtering info"""
    
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
            
            # Show meaningful words found for smart search
            if method == 'Smart' and 'meaningful_words_found' in result:
                words_found = result['meaningful_words_found']
                if words_found:
                    output += f"Meaningful Words Found: {', '.join(words_found)}\n"
            
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
    """Export results with complete content and filtering info"""
    
    meaningful_words = get_meaningful_words(query)
    
    csv_data = []
    
    for i, result in enumerate(results, 1):
        # Include FULL content without truncation
        row = {
            'Match_Number': i,
            'Query': query,
            'Meaningful_Words_Used': ', '.join(meaningful_words),
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
        
        # Add meaningful words found for smart search
        if result.get('match_type') == 'smart' and 'meaningful_words_found' in result:
            row['Meaningful_Words_Found'] = ', '.join(result['meaningful_words_found'])
            row['Words_Matched'] = len(result['meaningful_words_found'])
            row['Total_Meaningful_Words'] = result.get('total_meaningful_words', 0)
        else:
            row['Meaningful_Words_Found'] = ''
            row['Words_Matched'] = 0
            row['Total_Meaningful_Words'] = 0
        
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
    
    st.success(f"✅ CSV ready for download with complete content and filtering info! ({len(results)} results)")
    
    # Show preview
    with st.expander("📊 CSV Preview"):
        preview_cols = ['Match_Number', 'Document', 'Score', 'Page_Number', 'Words_Matched', 'Word_Count']
        available_cols = [col for col in preview_cols if col in df.columns]
        st.dataframe(df[available_cols].head())

# Export all functions - UPDATED
__all__ = [
    'display_search_results_beautiful',
    'display_single_search_result_beautiful', 
    'display_alignment_results_beautiful',
    'display_single_alignment_beautiful',
    'display_manual_search_results_beautiful',
    'show_alignment_feature_info_beautiful',
    'format_as_beautiful_paragraphs',
    'highlight_meaningful_words_only',
    'get_meaningful_words',
    'copy_results_beautiful',
    'export_results_csv_beautiful',
    'clean_html_artifacts',
    'format_text_as_clean_paragraphs'
]
