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
        best_score = max(
