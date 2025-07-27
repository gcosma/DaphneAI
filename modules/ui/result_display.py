# modules/ui/result_display.py - Search Results Display
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from .search_utils import highlight_search_terms, filter_stop_words

def display_results_grouped(results: List[Dict], query: str, search_time: float, 
                          show_context: bool = True, highlight_matches: bool = True,
                          search_method: str = "Search"):
    """Display search results grouped by document with enhanced information"""
    
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    # Group results by document
    doc_groups = defaultdict(list)
    for result in results:
        doc_name = result['document']['filename']
        doc_groups[doc_name].append(result)
    
    # Summary
    total_docs = len(doc_groups)
    total_matches = len(results)
    
    st.success(f"🎯 Found **{total_matches}** result(s) in **{total_docs}** document(s) for '**{query}**' in {search_time:.3f} seconds")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy All Results"):
            copy_all_results(results, query)
    with col2:
        if st.button("📊 Export to CSV"):
            export_results_csv(results, query)
    with col3:
        if st.button("📄 Generate Report"):
            generate_search_report(results, query, search_method)
    
    # Display each document group
    for doc_name, doc_results in doc_groups.items():
        
        doc = doc_results[0]['document']
        best_score = max(r['score'] for r in doc_results)
        
        # Document header with enhanced info
        with st.expander(f"📄 {doc_name} ({len(doc_results)} matches, best score: {best_score:.1f})", expanded=True):
            
            # Document statistics
            text = doc.get('text', '')
            word_count = len(text.split())
            char_count = len(text)
            est_pages = max(1, char_count // 2000)
            file_size_mb = char_count / (1024 * 1024)
            
            st.markdown(f"""
            **File Type:** {doc_name.split('.')[-1].upper()}  |  **Words:** {word_count:,}  |  **Size:** {file_size_mb:.1f} MB  |  **Est. Pages:** {est_pages}
            """)
            
            # Display each match in this document
            for i, result in enumerate(doc_results, 1):
                display_single_result(result, i, query, show_context, highlight_matches)

def display_single_result(result: Dict, index: int, query: str, show_context: bool, highlight_matches: bool):
    """Display a single search result with enhanced formatting - ALIGNMENT-QUALITY DISPLAY"""
    
    # Result header with score and method
    method_icons = {
        'exact': '🎯',
        'smart': '🧠', 
        'smart_enhanced': '🧠⭐',  # NEW: Enhanced smart search
        'fuzzy': '🌀',
        'semantic': '🤖',
        'hybrid': '🔄'
    }
    
    method = result.get('match_type', 'unknown')
    icon = method_icons.get(method, '🔍')
    score = result.get('score', 0)
    
    # ENHANCED: Content type display
    content_type = result.get('content_type', 'General')
    content_relevance = result.get('content_relevance', 0)
    
    # Content type icons
    content_icons = {
        'Recommendation': '🎯',
        'Response': '↩️',
        'Policy': '📋',
        'Analysis': '📊',
        'Financial': '💰',
        'Urgent': '⚡',
        'Meeting': '🤝',
        'Government': '🏛️',
        'General': '📄'
    }
    
    content_icon = content_icons.get(content_type, '📄')
    
    # ENHANCED: Relevance indicator
    relevance_color = "🟢" if content_relevance > 0.8 else "🟡" if content_relevance > 0.5 else "🔴"
    
    st.markdown(f"""
    **{icon} {method.replace('_', ' ').title()} - Match {index}** {content_icon} {content_type}  
    Score: {score:.1f} | Relevance: {relevance_color} {content_relevance:.2f}
    """)
    
    # DEBUG: Show enhanced matching info for smart search
    if method == 'smart_enhanced':
        word_sim = result.get('word_similarity', 0)
        semantic_sim = result.get('semantic_similarity', 0)
        coverage = result.get('query_coverage', 'N/A')
        st.caption(f"🔍 **Enhanced Match:** Word: {word_sim:.2f} | Semantic: {semantic_sim:.2f} | Coverage: {coverage}")
    
    # DEBUG: Show what was actually matched for other methods
    matched_text = result.get('matched_text', '')
    query_word = result.get('query_word', query)
    
    if method == 'fuzzy' and 'similarity' in result:
        similarity = result['similarity']
        st.caption(f"🔍 **Fuzzy Match:** '{matched_text}' ↔ '{query_word}' (similarity: {similarity:.2f})")
    elif method == 'semantic' and 'semantic_relation' in result:
        st.caption(f"🤖 **Semantic:** {result['semantic_relation']}")
    elif method == 'smart' and 'pattern_used' in result:
        st.caption(f"🧠 **Smart Match:** Found '{matched_text}' using pattern search")
    
    # Position information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"📄 **Page {result.get('page_number', 1)}**")
    
    with col2:
        pos = result.get('position', 0)
        st.markdown(f"📍 **Position {pos:,}**")
    
    with col3:
        percentage = result.get('percentage_through', 0)
        st.markdown(f"📊 **{percentage:.0f}% through doc**")
    
    with col4:
        word_pos = result.get('word_position', 0)
        st.markdown(f"🔢 **Word {word_pos:,}**")
    
    # ENHANCED: Content analysis for smart enhanced results
    if method == 'smart_enhanced':
        st.markdown(f"**🏷️ Content Analysis:** {content_type} content with {content_relevance:.0%} relevance to your query")
        
        # Show specific insights based on content type
        if content_type == 'Recommendation' and any(word in query.lower() for word in ['recommend', 'suggest']):
            st.info("💡 This appears to be a recommendation that matches your search terms")
        elif content_type == 'Response' and any(word in query.lower() for word in ['respond', 'response', 'reply']):
            st.info("💡 This appears to be a response that matches your search terms")
        elif content_type == 'Policy' and any(word in query.lower() for word in ['policy', 'framework']):
            st.info("💡 This appears to be policy content relevant to your query")
    
    # Match details with better information
    if matched_text:
        # For exact matches, show phrase counting
        if method == 'exact':
            exact_count = result.get('context', '').lower().count(query.lower())
            st.markdown(f"🎯 **Exact phrase '{query}':** Found {exact_count} time(s)")
        else:
            # For other methods, show word matching info (filtered)
            original_query = result.get('original_query', query)
            filtered_query = result.get('processed_query', filter_stop_words(query))
            
            # Show both original and filtered if different
            if filtered_query != original_query:
                st.markdown(f"🔍 **Original query:** '{original_query}' → **Filtered:** '{filtered_query}'")
            
            # Show which meaningful words were found
            if method == 'smart_enhanced':
                # Enhanced display for smart enhanced results
                coverage = result.get('query_coverage', 'N/A')
                st.markdown(f"📝 **Query Coverage:** {coverage} terms matched")
            else:
                query_words = filtered_query.lower().split() if filtered_query else []
                context_lower = result.get('context', '').lower()
                found_words = [word for word in query_words if word in context_lower]
                st.markdown(f"📝 **Meaningful words found:** {', '.join(found_words) if found_words else 'Related terms found'}")
    
    # Context display
    if show_context:
        context = result.get('context', '')
        if context:
            
            # Highlight matches if requested
            if highlight_matches:
                # For fuzzy search, also highlight the actual matched word
                highlight_query = query
                if method == 'fuzzy' and matched_text:
                    highlight_query = f"{query} {matched_text}"
                
                highlighted_context = highlight_search_terms(context, highlight_query)
                st.markdown(f"📖 **Context:** {highlighted_context}", unsafe_allow_html=True)
            else:
                st.markdown(f"📖 **Context:** {context}")
            
            # Position hint
            page_num = result.get('page_number', 1)
            percentage = result.get('percentage_through', 0)
            st.caption(f"💡 This appears around page {page_num}, {percentage:.1f}% through the document")
    
    # Additional info for special search types
    if method == 'fuzzy' and 'similarity' in result:
        similarity = result['similarity']
        st.caption(f"🌀 Fuzzy match similarity: {similarity:.2f} (threshold: 0.6)")
    
    elif method == 'semantic':
        st.caption(f"🤖 AI found this as semantically related to your query")
    
    elif method == 'smart_enhanced':
        st.caption(f"🧠⭐ Enhanced search using government document analysis")
    
    st.markdown("---")

def copy_all_results(results: List[Dict], query: str):
    """Copy all results to clipboard"""
    
    output = f"Search Results for: {query}\n"
    output += "=" * 50 + "\n\n"
    
    doc_groups = defaultdict(list)
    for result in results:
        doc_name = result['document']['filename']
        doc_groups[doc_name].append(result)
    
    for doc_name, doc_results in doc_groups.items():
        output += f"Document: {doc_name}\n"
        output += f"Matches: {len(doc_results)}\n\n"
        
        for i, result in enumerate(doc_results, 1):
            output += f"Match {i}:\n"
            output += f"  Page: {result.get('page_number', 1)}\n"
            output += f"  Score: {result.get('score', 0):.1f}\n"
            output += f"  Context: {result.get('context', '')}\n\n"
        
        output += "-" * 30 + "\n\n"
    
    st.code(output)
    st.success("Results copied to display! Use Ctrl+A, Ctrl+C to copy to clipboard")

def export_results_csv(results: List[Dict], query: str):
    """Export search results to CSV"""
    
    csv_data = []
    
    for result in results:
        row = {
            'Query': query,
            'Document': result['document']['filename'],
            'Match_Type': result.get('match_type', ''),
            'Score': result.get('score', 0),
            'Page_Number': result.get('page_number', 1),
            'Position': result.get('position', 0),
            'Matched_Text': result.get('matched_text', ''),
            'Context': result.get('context', ''),
            'Percentage_Through': result.get('percentage_through', 0)
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"search_results_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def generate_search_report(results: List[Dict], query: str, search_method: str):
    """Generate a comprehensive search report"""
    
    # Group results by document
    doc_groups = defaultdict(list)
    for result in results:
        doc_name = result['document']['filename']
        doc_groups[doc_name].append(result)
    
    report = f"""# Search Report: "{query}"

**Search Method:** {search_method}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Results:** {len(results)} matches in {len(doc_groups)} documents

## Summary Statistics

| Metric | Value |
|--------|-------|
| Documents Searched | {len(doc_groups)} |
| Total Matches | {len(results)} |
| Average Score | {sum(r.get('score', 0) for r in results) / len(results):.2f} |
| Highest Score | {max(r.get('score', 0) for r in results):.2f} |

## Results by Document

"""
    
    for doc_name, doc_results in doc_groups.items():
        best_score = max(r.get('score', 0) for r in doc_results)
        avg_score = sum(r.get('score', 0) for r in doc_results) / len(doc_results)
        
        report += f"""### 📄 {doc_name}
- **Matches:** {len(doc_results)}
- **Best Score:** {best_score:.2f}
- **Average Score:** {avg_score:.2f}

"""
        
        for i, result in enumerate(doc_results, 1):
            report += f"""**Match {i}** (Score: {result.get('score', 0):.1f})  
*Page {result.get('page_number', 1)}, Position {result.get('position', 0):,}*

> {result.get('context', '')[:200]}...

---

"""
    
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"search_report_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )
