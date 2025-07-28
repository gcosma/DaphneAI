# modules/ui/beautiful_display.py - Beautiful Display Functions
"""
Beautiful display functions for DaphneAI search results and alignments.
This file contains all the formatting and presentation logic with complete paragraphs.
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List

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
    
    # Beautiful summary
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
    """Display a single search result with beautiful formatting"""
    
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
    
    # Display FULL context beautifully
    if show_context:
        full_context = result.get('context', '')
        if full_context:
            
            # Apply highlighting if requested
            if highlight_matches:
                display_content = highlight_text_beautiful(full_context, query)
            else:
                display_content = full_context
            
            # Format as beautiful paragraphs
            formatted_content = format_as_beautiful_paragraphs(display_content)
            
            st.markdown(f"""
            <div style="
                background: white;
                border: 2px solid #e9ecef;
                padding: 25px;
                border-radius: 10px;
                margin: 15px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            ">
                <h5 style="margin: 0 0 15px 0; color: #495057; border-bottom: 2px solid #e9ecef; padding-bottom: 8px;">
                    📖 Complete Context
                </h5>
                <div style="font-size: 16px; line-height: 1.8; color: #333;">
                    {formatted_content}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
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
        
        # Display the FULL recommendation sentence - no truncation
        full_sentence = rec.get('sentence', 'No sentence available')
        
        # Beautiful highlighted display of the full content
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border-left: 4px solid #f39c12;
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.6;
        ">
            <strong>📝 Full Recommendation:</strong><br><br>
            {full_sentence}
        </div>
        """, unsafe_allow_html=True)
        
        # Show FULL context as beautiful paragraphs
        full_context = rec.get('context', 'No context available')
        if full_context and full_context != full_sentence:
            st.markdown("#### 📖 Complete Context")
            
            # Format context as beautiful paragraphs
            formatted_context = format_as_beautiful_paragraphs(full_context)
            
            st.markdown(f"""
            <div style="
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 20px;
                border-radius: 8px;
                margin: 10px 0;
                font-size: 15px;
                line-height: 1.7;
                color: #495057;
            ">
                {formatted_context}
            </div>
            """, unsafe_allow_html=True)
        
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
                
                # Display FULL response content beautifully
                full_resp_sentence = resp.get('sentence', 'No sentence available')
                full_resp_context = resp.get('context', 'No context available')
                
                st.markdown(f"""
                <div style="
                    background: {bg_gradient};
                    border-left: 4px solid {border_color};
                    padding: 20px;
                    border-radius: 8px;
                    margin: 15px 0;
                ">
                    <h5 style="margin: 0 0 10px 0; color: #333;">
                        📄 Response {j} - {confidence_text} ({similarity:.2f})
                    </h5>
                    <p style="margin: 5px 0; font-weight: 500; color: #666;">
                        <strong>Document:</strong> {resp_doc_name} | <strong>Page:</strong> {resp_page_num}
                    </p>
                    
                    <div style="margin: 15px 0;">
                        <strong>📝 Full Response:</strong><br><br>
                        <span style="font-size: 16px; line-height: 1.6;">{full_resp_sentence}</span>
                    </div>
                    
                    {f'''
                    <div style="margin: 15px 0;">
                        <strong>📖 Complete Context:</strong><br><br>
                        <span style="font-size: 15px; line-height: 1.7; color: #555;">
                            {format_as_beautiful_paragraphs(full_resp_context)}
                        </span>
                    </div>
                    ''' if full_resp_context and full_resp_context != full_resp_sentence else ''}
                </div>
                """, unsafe_allow_html=True)
                
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
    """Display manual search results with beautiful formatting"""
    
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
            
            # Highlight similar words in the found sentence
            highlighted_sentence = highlight_similar_words_beautiful(full_sentence, target_sentence)
            
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
    
    # Format as HTML paragraphs with beautiful spacing
    formatted_paragraphs = []
    for paragraph in paragraphs:
        if paragraph.strip():
            formatted_paragraphs.append(f'<p style="margin-bottom: 15px; text-align: justify;">{paragraph.strip()}</p>')
    
    return ''.join(formatted_paragraphs) if formatted_paragraphs else text

def highlight_text_beautiful(text: str, query: str) -> str:
    """Apply beautiful highlighting to search terms"""
    
    highlighted = text
    query_words = [word for word in query.split() if len(word) > 2]
    
    # Sort by length (longest first) to avoid partial highlighting conflicts
    query_words.sort(key=len, reverse=True)
    
    for word in query_words:
        # Case-insensitive highlighting with beautiful styling
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background: linear-gradient(135deg, #ffeb3b 0%, #ffc107 100%); '
            f'padding: 3px 6px; border-radius: 4px; font-weight: bold; '
            f'box-shadow: 0 1px 3px rgba(0,0,0,0.1);">{word}</span>', 
            highlighted
        )
    
    return highlighted

def highlight_similar_words_beautiful(sentence: str, target_sentence: str) -> str:
    """Highlight words in sentence that are similar to target sentence with beautiful styling"""
    
    # Extract meaningful words from target (excluding common words)
    stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                  'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                  'to', 'was', 'will', 'with', 'but', 'they', 'have', 'had', 'what'}
    
    target_words = set(w.lower() for w in target_sentence.split() 
                      if len(w) > 2 and w.lower() not in stop_words)
    
    highlighted = sentence
    words = sentence.split()
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in target_words and len(clean_word) > 2:
            # Beautiful highlighting with gradient and shadow
            highlighted = highlighted.replace(word, 
                f'<span style="background: linear-gradient(135deg, #ffeb3b 0%, #ffc107 100%); '
                f'padding: 3px 8px; border-radius: 6px; font-weight: bold; '
                f'box-shadow: 0 2px 4px rgba(255,193,7,0.3); '
                f'border: 1px solid #ffc107;">{word}</span>')
    
    return highlighted

def copy_results_beautiful(results: List[Dict], query: str):
    """Copy results with beautiful formatting"""
    
    output = f"""
DAPHNE AI - SEARCH RESULTS REPORT
==================================

Search Query: "{query}"
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
    
    for i, result in enumerate(results, 1):
        # Include FULL content without truncation
        row = {
            'Match_Number': i,
            'Query': query,
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

# Export all functions
__all__ = [
    'display_search_results_beautiful',
    'display_single_search_result_beautiful', 
    'display_alignment_results_beautiful',
    'display_single_alignment_beautiful',
    'display_manual_search_results_beautiful',
    'show_alignment_feature_info_beautiful',
    'format_as_beautiful_paragraphs',
    'highlight_text_beautiful',
    'highlight_similar_words_beautiful',
    'copy_results_beautiful',
    'export_results_csv_beautiful'
]
