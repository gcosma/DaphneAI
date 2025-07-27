# modules/ui/result_display.py - Search Results Display - COMPLETE FIXED VERSION
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
        doc_name = result.get('document', {}).get('filename', 'Unknown Document')
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
        
        doc = doc_results[0].get('document', {})
        best_score = max(r.get('score', 0) for r in doc_results)
        
        # Document header with enhanced info
        with st.expander(f"📄 {doc_name} ({len(doc_results)} matches, best score: {best_score:.1f})", expanded=True):
            
            # Document statistics
            text = doc.get('text', '')
            word_count = len(text.split()) if text else 0
            char_count = len(text) if text else 0
            est_pages = max(1, char_count // 2000)
            file_size_mb = char_count / (1024 * 1024)
            
            st.markdown(f"""
            **File Type:** {doc_name.split('.')[-1].upper() if '.' in doc_name else 'TXT'}  |  **Words:** {word_count:,}  |  **Size:** {file_size_mb:.1f} MB  |  **Est. Pages:** {est_pages}
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
    """Copy all results to clipboard - COMPLETE FIXED VERSION"""
    
    if not results:
        st.warning("No results to copy")
        return
    
    try:
        output = f"Search Results for: {query}\n"
        output += f"Total Results: {len(results)}\n"
        output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += "=" * 60 + "\n\n"
        
        doc_groups = defaultdict(list)
        for result in results:
            doc_name = result.get('document', {}).get('filename', 'Unknown Document')
            doc_groups[doc_name].append(result)
        
        for doc_name, doc_results in doc_groups.items():
            output += f"Document: {doc_name}\n"
            output += f"Matches: {len(doc_results)}\n\n"
            
            for i, result in enumerate(doc_results, 1):
                output += f"Match {i}:\n"
                output += f"  Type: {result.get('content_type', 'General')}\n"
                output += f"  Score: {result.get('score', 0):.1f}\n"
                output += f"  Page: {result.get('page_number', 1)}\n"
                output += f"  Context: {result.get('context', '')[:150]}...\n\n"
            
            output += "-" * 40 + "\n\n"
        
        # Limit output size for display
        if len(output) > 3000:
            display_output = output[:3000] + "\n... [Content truncated for display] ..."
        else:
            display_output = output
        
        st.code(display_output, language="text")
        st.success("✅ Results displayed above! Use Ctrl+A, Ctrl+C to copy to clipboard")
        
        # Also provide as downloadable text file
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_query = safe_query.replace(' ', '_')[:20]
        
        st.download_button(
            label="📋 Download as Text File",
            data=output,
            file_name=f"search_results_{safe_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key=f"text_download_{len(results)}_{hash(query)}"
        )
        
    except Exception as e:
        st.error(f"Error copying results: {str(e)}")
        
        # Fallback: Simple list
        st.markdown("**Simple Results List:**")
        for i, result in enumerate(results[:20], 1):  # First 20 results
            doc_name = result.get('document', {}).get('filename', 'Unknown')
            score = result.get('score', 0)
            st.write(f"{i}. {doc_name} - {score:.1f}")

def export_results_csv(results: List[Dict], query: str):
    """Export search results to CSV - COMPLETE FIXED VERSION"""
    
    if not results:
        st.warning("No results to export")
        return
    
    csv_data = []
    
    for i, result in enumerate(results, 1):
        # Handle missing keys gracefully
        row = {
            'Match_Number': i,
            'Query': query,
            'Document': result.get('document', {}).get('filename', 'Unknown'),
            'Match_Type': result.get('match_type', 'Unknown'),
            'Score': result.get('score', 0),
            'Content_Type': result.get('content_type', 'General'),
            'Content_Relevance': result.get('content_relevance', 0),
            'Page_Number': result.get('page_number', 1),
            'Position': result.get('position', 0),
            'Percentage_Through': result.get('percentage_through', 0),
            'Matched_Text': str(result.get('matched_text', ''))[:500],  # Limit length
            'Context': str(result.get('context', ''))[:1000],  # Limit length
            'Word_Similarity': result.get('word_similarity', 'N/A'),
            'Semantic_Similarity': result.get('semantic_similarity', 'N/A'),
            'Query_Coverage': result.get('query_coverage', 'N/A')
        }
        csv_data.append(row)
    
    # Create DataFrame
    try:
        df = pd.DataFrame(csv_data)
        csv = df.to_csv(index=False)
        
        # Generate safe filename
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_query = safe_query.replace(' ', '_')[:20]  # Limit length
        filename = f"search_results_{safe_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=f"csv_download_{len(results)}_{hash(query)}"  # Unique key
        )
        
        st.success(f"✅ CSV ready for download! ({len(results)} results)")
        
        # Show preview
        with st.expander("📊 CSV Preview (First 5 rows)"):
            st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Error creating CSV: {str(e)}")
        
        # Fallback: Show data as table
        st.markdown("**CSV Data Preview:**")
        try:
            simple_data = []
            for i, result in enumerate(results[:10], 1):  # First 10 results
                simple_data.append({
                    'Match': i,
                    'Document': result.get('document', {}).get('filename', 'Unknown')[:30],
                    'Type': result.get('content_type', 'General'),
                    'Score': result.get('score', 0),
                    'Page': result.get('page_number', 1)
                })
            st.table(simple_data)
        except:
            st.write("Unable to display data preview")

def generate_search_report(results: List[Dict], query: str, search_method: str):
    """Generate a comprehensive search report - COMPLETE FIXED VERSION"""
    
    if not results:
        st.warning("No results to generate report")
        return
    
    try:
        # Group results by document
        doc_groups = defaultdict(list)
        for result in results:
            doc_name = result.get('document', {}).get('filename', 'Unknown Document')
            doc_groups[doc_name].append(result)
        
        # Calculate statistics
        total_results = len(results)
        total_docs = len(doc_groups)
        avg_score = sum(r.get('score', 0) for r in results) / total_results if total_results > 0 else 0
        max_score = max((r.get('score', 0) for r in results), default=0)
        
        # Content type analysis
        content_types = {}
        for result in results:
            content_type = result.get('content_type', 'General')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        # Generate report content
        report = f"""# Search Report: "{query}"

**Search Method:** {search_method}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Results:** {total_results} matches in {total_docs} documents

## Executive Summary

| Metric | Value |
|--------|-------|
| Documents Searched | {total_docs} |
| Total Matches | {total_results} |
| Average Score | {avg_score:.2f} |
| Highest Score | {max_score:.2f} |

## Content Type Analysis

"""
        
        # Add content type breakdown
        if content_types:
            for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_results) * 100
                report += f"- **{content_type}:** {count} results ({percentage:.1f}%)\n"
        else:
            report += "- No content type analysis available\n"
        
        report += "\n## Results by Document\n\n"
        
        # Add document-by-document analysis
        for doc_name, doc_results in doc_groups.items():
            best_score = max((r.get('score', 0) for r in doc_results), default=0)
            avg_score_doc = sum(r.get('score', 0) for r in doc_results) / len(doc_results) if doc_results else 0
            
            report += f"""### 📄 {doc_name}
- **Matches:** {len(doc_results)}
- **Best Score:** {best_score:.2f}
- **Average Score:** {avg_score_doc:.2f}

"""
            
            # Add top matches from this document
            sorted_results = sorted(doc_results, key=lambda x: x.get('score', 0), reverse=True)
            for i, result in enumerate(sorted_results[:3], 1):  # Top 3 matches
                score = result.get('score', 0)
                page = result.get('page_number', 1)
                position = result.get('position', 0)
                content_type = result.get('content_type', 'General')
                context = str(result.get('context', ''))[:200]  # First 200 chars
                
                report += f"""**Match {i}** - {content_type} (Score: {score:.1f})  
*Page {page}, Position {position:,}*

> {context}...

---

"""
        
        # Generate safe filename
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_query = safe_query.replace(' ', '_')[:20]  # Limit length
        filename = f"search_report_{safe_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Provide download button
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=filename,
            mime="text/markdown",
            key=f"report_download_{len(results)}_{hash(query)}"  # Unique key
        )
        
        st.success(f"✅ Report ready for download! ({len(report)} characters)")
        
        # Show preview
        with st.expander("📋 Report Preview"):
            st.markdown("**First 1000 characters:**")
            preview_text = report[:1000] + "..." if len(report) > 1000 else report
            st.markdown(preview_text)
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        
        # Fallback: Simple report
        st.markdown("**Simple Report:**")
        simple_report = f"""# Search Report for: {query}

**Search Method:** {search_method}  
**Total Results:** {len(results)}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Results Summary:
"""
        for i, result in enumerate(results[:10], 1):  # First 10 results
            doc_name = result.get('document', {}).get('filename', 'Unknown')
            score = result.get('score', 0)
            content_type = result.get('content_type', 'General')
            simple_report += f"{i}. **{doc_name}** - {content_type} (Score: {score:.1f})\n"
        
        st.markdown(simple_report)
        
        # Provide fallback download
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_query = safe_query.replace(' ', '_')[:20]
        
        st.download_button(
            label="📄 Download Simple Report",
            data=simple_report,
            file_name=f"simple_report_{safe_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            key=f"simple_report_{len(results)}_{hash(query)}"
        )
