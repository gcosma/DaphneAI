# ===============================================
# FILE: modules/ui/extraction_components.py
# ===============================================

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import required modules with error handling
try:
    import sys
    sys.path.append('modules')
    from llm_extractor import LLMRecommendationExtractor
    from core_utils import Recommendation
    from .shared_components import add_error_message, show_progress_indicator
except ImportError as e:
    logging.error(f"Import error in extraction_components: {e}")
    # Create mock classes for development
    class LLMRecommendationExtractor:
        def extract_recommendations_and_concerns(self, text, source): 
            return {'recommendations': [], 'concerns': []}
    class Recommendation:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

def render_extraction_tab():
    """Render the content extraction tab"""
    st.header("🔍 Content Extraction")
    
    if not st.session_state.uploaded_documents:
        st.warning("⚠️ Please upload documents first in the Upload tab.")
        return
    
    st.markdown("""
    Extract recommendations and concerns from uploaded documents using AI-powered analysis 
    or pattern-based methods.
    """)
    
    # Extraction configuration
    render_extraction_configuration()
    
    # Document selection and extraction
    render_extraction_interface()
    
    # Display results
    display_extraction_results()

def render_extraction_configuration():
    """Render extraction configuration options"""
    st.subheader("⚙️ Extraction Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        extraction_method = st.selectbox(
            "Extraction Method",
            ["AI-Powered (GPT)", "Pattern-Based", "Hybrid (Recommended)"],
            index=2,  # Default to Hybrid
            help="Choose how to extract recommendations and concerns",
            key="extraction_method"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.6, 0.05,
            help="Minimum confidence for including extracted items",
            key="confidence_threshold"
        )
    
    with col3:
        max_extractions = st.number_input(
            "Max Extractions per Document",
            1, 100, 50,
            help="Maximum number of items to extract per document",
            key="max_extractions"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_context = st.checkbox(
                "Include Context",
                value=True,
                help="Include surrounding text for better understanding",
                key="include_context"
            )
            
            extract_concerns = st.checkbox(
                "Extract Concerns",
                value=True,
                help="Also extract concerns and issues from documents",
                key="extract_concerns"
            )
        
        with col2:
            min_text_length = st.number_input(
                "Minimum Text Length",
                10, 500, 50,
                help="Minimum character length for extracted items",
                key="min_text_length"
            )
            
            use_smart_filtering = st.checkbox(
                "Smart Filtering",
                value=True,
                help="Use AI to filter out irrelevant extractions",
                key="use_smart_filtering"
            )

def render_extraction_interface():
    """Render the main extraction interface"""
    st.subheader("📄 Document Selection & Extraction")
    
    # Document selection
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_docs = st.multiselect(
            "Select documents to extract from:",
            options=doc_options,
            default=doc_options,  # Select all by default
            help="Choose which documents to process for extraction",
            key="selected_docs_extraction"
        )
    
    with col2:
        st.markdown("**Extraction Info:**")
        st.write(f"• Documents selected: {len(selected_docs)}")
        st.write(f"• Method: {st.session_state.get('extraction_method', 'Hybrid')}")
        st.write(f"• Confidence: {st.session_state.get('confidence_threshold', 0.6):.1f}")
        
        if st.session_state.get('extract_concerns', True):
            st.write("• ✅ Extract concerns")
        else:
            st.write("• ❌ Skip concerns")
    
    # Extraction button
    if selected_docs:
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True):
            extract_content_from_documents(selected_docs)
    else:
        st.info("Please select documents to extract from.")

def extract_content_from_documents(selected_docs: List[str]):
    """Extract recommendations and concerns from selected documents"""
    if not selected_docs:
        st.warning("No documents selected for extraction.")
        return
    
    # Get configuration
    extraction_method = st.session_state.get('extraction_method', 'Hybrid (Recommended)')
    confidence_threshold = st.session_state.get('confidence_threshold', 0.6)
    max_extractions = st.session_state.get('max_extractions', 50)
    extract_concerns = st.session_state.get('extract_concerns', True)
    min_text_length = st.session_state.get('min_text_length', 50)
    
    # Initialize extractor
    extractor = LLMRecommendationExtractor()
    
    # Progress tracking
    total_docs = len(selected_docs)
    progress_container = st.container()
    status_container = st.container()
    results_container = st.container()
    
    all_recommendations = []
    all_concerns = []
    processing_results = []
    
    # Set processing status
    st.session_state.processing_status = "extracting"
    
    try:
        for i, doc_name in enumerate(selected_docs):
            current_step = i + 1
            
            # Update progress
            with progress_container:
                show_progress_indicator(current_step, total_docs, f"Extracting from {doc_name}")
            
            with status_container:
                status_text = st.empty()
                status_text.info(f"🔍 Processing: {doc_name}")
            
            try:
                # Find document
                doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
                if not doc:
                    status_text.error(f"❌ Document not found: {doc_name}")
                    continue
                
                # Extract content
                extraction_result = extractor.extract_recommendations_and_concerns(
                    doc['content'], 
                    doc['filename']
                )
                
                recommendations = extraction_result.get('recommendations', [])
                concerns = extraction_result.get('concerns', [])
                
                # Apply filtering
                filtered_recommendations = []
                for rec in recommendations:
                    if (rec.confidence_score >= confidence_threshold and 
                        len(rec.text) >= min_text_length):
                        filtered_recommendations.append(rec)
                
                filtered_concerns = []
                if extract_concerns:
                    for concern in concerns:
                        if (concern.get('confidence_score', 0) >= confidence_threshold and 
                            len(concern.get('text', '')) >= min_text_length):
                            filtered_concerns.append(concern)
                
                # Limit extractions
                filtered_recommendations = filtered_recommendations[:max_extractions]
                filtered_concerns = filtered_concerns[:max_extractions]
                
                # Store results
                all_recommendations.extend(filtered_recommendations)
                all_concerns.extend(filtered_concerns)
                
                processing_results.append({
                    'document': doc_name,
                    'recommendations_found': len(filtered_recommendations),
                    'concerns_found': len(filtered_concerns),
                    'status': 'success'
                })
                
                status_text.success(f"✅ Extracted {len(filtered_recommendations)} recommendations, {len(filtered_concerns)} concerns from {doc_name}")
                
            except Exception as e:
                error_msg = f"Error extracting from {doc_name}: {str(e)}"
                add_error_message(error_msg)
                processing_results.append({
                    'document': doc_name,
                    'recommendations_found': 0,
                    'concerns_found': 0,
                    'status': 'error',
                    'error': str(e)
                })
                status_text.error(f"❌ Failed: {doc_name}")
                logging.error(f"Extraction error: {e}", exc_info=True)
        
        # Update session state
        st.session_state.extracted_recommendations = all_recommendations
        st.session_state.extracted_concerns = all_concerns
        st.session_state.last_extraction_results = processing_results
        st.session_state.last_processing_time = datetime.now().isoformat()
        
        # Clear progress displays
        progress_container.empty()
        status_container.empty()
        
        # Show final results
        with results_container:
            show_extraction_summary(processing_results, all_recommendations, all_concerns)
    
    finally:
        st.session_state.processing_status = "idle"

def show_extraction_summary(processing_results: List[Dict], recommendations: List, concerns: List):
    """Show summary of extraction results"""
    st.success("🎉 Extraction Complete!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        successful_docs = len([r for r in processing_results if r['status'] == 'success'])
        st.metric("Documents Processed", f"{successful_docs}/{len(processing_results)}")
    
    with col2:
        total_recommendations = len(recommendations)
        st.metric("Recommendations Found", total_recommendations)
    
    with col3:
        total_concerns = len(concerns)
        st.metric("Concerns Found", total_concerns)
    
    with col4:
        avg_per_doc = (total_recommendations + total_concerns) / len(processing_results) if processing_results else 0
        st.metric("Avg per Document", f"{avg_per_doc:.1f}")
    
    # Detailed results table
    if processing_results:
        st.subheader("📊 Processing Details")
        
        results_df = pd.DataFrame(processing_results)
        results_df = results_df.rename(columns={
            'document': 'Document',
            'recommendations_found': 'Recommendations',
            'concerns_found': 'Concerns',
            'status': 'Status'
        })
        
        # Add status icons
        results_df['Status'] = results_df['Status'].map({
            'success': '✅ Success',
            'error': '❌ Error'
        })
        
        st.dataframe(results_df[['Document', 'Recommendations', 'Concerns', 'Status']], use_container_width=True)
    
    # Show any errors
    failed_docs = [r for r in processing_results if r['status'] == 'error']
    if failed_docs:
        with st.expander("⚠️ Processing Errors"):
            for result in failed_docs:
                st.error(f"**{result['document']}:** {result.get('error', 'Unknown error')}")

def display_extraction_results():
    """Display extracted recommendations and concerns"""
    if not st.session_state.extracted_recommendations and not st.session_state.extracted_concerns:
        st.info("💡 No extractions yet. Configure settings above and click 'Start Extraction' to begin.")
        return
    
    # Results tabs
    rec_tab, concern_tab, combined_tab = st.tabs(["📋 Recommendations", "⚠️ Concerns", "📊 Combined View"])
    
    with rec_tab:
        display_extracted_recommendations()
    
    with concern_tab:
        display_extracted_concerns()
    
    with combined_tab:
        display_combined_results()

def display_extracted_recommendations():
    """Display extracted recommendations with interactive features"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    
    if not recommendations:
        st.info("No recommendations extracted yet.")
        return
    
    st.subheader(f"📋 Extracted Recommendations ({len(recommendations)})")
    
    # Filtering and sorting
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_filter = st.selectbox(
            "Filter by Source:",
            options=['All'] + sorted(list(set(rec.document_source for rec in recommendations))),
            key="rec_source_filter"
        )
    
    with col2:
        confidence_filter = st.slider(
            "Min Confidence:",
            0.0, 1.0, 0.0, 0.05,
            key="rec_confidence_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["Confidence (High to Low)", "Confidence (Low to High)", "Document Source", "Text Length"],
            key="rec_sort_by"
        )
    
    # Apply filters
    filtered_recs = recommendations
    
    if source_filter != 'All':
        filtered_recs = [rec for rec in filtered_recs if rec.document_source == source_filter]
    
    filtered_recs = [rec for rec in filtered_recs if rec.confidence_score >= confidence_filter]
    
    # Apply sorting
    if sort_by == "Confidence (High to Low)":
        filtered_recs.sort(key=lambda x: x.confidence_score, reverse=True)
    elif sort_by == "Confidence (Low to High)":
        filtered_recs.sort(key=lambda x: x.confidence_score)
    elif sort_by == "Document Source":
        filtered_recs.sort(key=lambda x: x.document_source)
    elif sort_by == "Text Length":
        filtered_recs.sort(key=lambda x: len(x.text), reverse=True)
    
    if not filtered_recs:
        st.warning("No recommendations match the current filters.")
        return
    
    # Display recommendations
    st.write(f"Showing {len(filtered_recs)} of {len(recommendations)} recommendations")
    
    # Create summary table
    rec_data = []
    for i, rec in enumerate(filtered_recs):
        rec_data.append({
            "Index": i,
            "ID": rec.id,
            "Preview": rec.text[:100] + "..." if len(rec.text) > 100 else rec.text,
            "Source": rec.document_source,
            "Section": rec.section_title,
            "Confidence": f"{rec.confidence_score:.2f}",
            "Length": len(rec.text)
        })
    
    df = pd.DataFrame(rec_data)
    
    # Display with selection capability
    selected_indices = st.dataframe(
        df.drop('Index', axis=1),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row"
    )
    
    # Show detailed view for selected recommendations
    if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
        st.subheader("📖 Detailed View")
        
        for idx in selected_indices.selection.rows:
            if idx < len(filtered_recs):
                rec = filtered_recs[idx]
                display_recommendation_detail(rec)

def display_recommendation_detail(rec):
    """Display detailed view of a single recommendation"""
    with st.expander(f"📋 Recommendation {rec.id}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Full Text:**")
            st.write(rec.text)
        
        with col2:
            st.markdown("**Details:**")
            st.write(f"**ID:** {rec.id}")
            st.write(f"**Source:** {rec.document_source}")
            st.write(f"**Section:** {rec.section_title}")
            st.write(f"**Page:** {rec.page_number or 'N/A'}")
            st.write(f"**Confidence:** {rec.confidence_score:.2f}")
            st.write(f"**Length:** {len(rec.text)} characters")
            
            # Show metadata if available
            if hasattr(rec, 'metadata') and rec.metadata:
                st.markdown("**Metadata:**")
                for key, value in rec.metadata.items():
                    st.write(f"• **{key}:** {value}")

def display_extracted_concerns():
    """Display extracted concerns"""
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not concerns:
        st.info("No concerns extracted yet.")
        return
    
    st.subheader(f"⚠️ Extracted Concerns ({len(concerns)})")
    
    # Convert concerns to DataFrame for display
    concern_data = []
    for i, concern in enumerate(concerns):
        concern_data.append({
            "Index": i + 1,
            "ID": concern.get('id', f"CONCERN-{i+1}"),
            "Preview": concern.get('text', '')[:100] + "..." if len(concern.get('text', '')) > 100 else concern.get('text', ''),
            "Source": concern.get('document_source', 'Unknown'),
            "Section": concern.get('section', 'Unknown'),
            "Confidence": f"{concern.get('confidence_score', 0):.2f}",
            "Category": concern.get('category', 'General')
        })
    
    df = pd.DataFrame(concern_data)
    
    # Filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        category_filter = st.selectbox(
            "Filter by Category:",
            options=['All'] + sorted(list(set(concern.get('category', 'General') for concern in concerns))),
            key="concern_category_filter"
        )
    
    with col2:
        confidence_filter = st.slider(
            "Min Confidence:",
            0.0, 1.0, 0.0, 0.05,
            key="concern_confidence_filter"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if category_filter != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
    
    filtered_df = filtered_df[filtered_df['Confidence'].astype(float) >= confidence_filter]
    
    # Display filtered results
    if not filtered_df.empty:
        st.dataframe(filtered_df.drop('Index', axis=1), use_container_width=True, hide_index=True)
        
        # Detailed view
        selected_concern_id = st.selectbox(
            "View Details:",
            options=['Select a concern...'] + filtered_df['ID'].tolist(),
            key="selected_concern_detail"
        )
        
        if selected_concern_id != 'Select a concern...':
            concern = next((c for c in concerns if c.get('id') == selected_concern_id), None)
            if concern:
                display_concern_detail(concern)
    else:
        st.warning("No concerns match the current filters.")

def display_concern_detail(concern):
    """Display detailed view of a single concern"""
    with st.expander(f"⚠️ Concern {concern.get('id')}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Full Text:**")
            st.write(concern.get('text', 'No text available'))
        
        with col2:
            st.markdown("**Details:**")
            st.write(f"**ID:** {concern.get('id', 'N/A')}")
            st.write(f"**Source:** {concern.get('document_source', 'Unknown')}")
            st.write(f"**Section:** {concern.get('section', 'Unknown')}")
            st.write(f"**Page:** {concern.get('page_number', 'N/A')}")
            st.write(f"**Confidence:** {concern.get('confidence_score', 0):.2f}")
            st.write(f"**Category:** {concern.get('category', 'General')}")

def display_combined_results():
    """Display combined view of recommendations and concerns"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not recommendations and not concerns:
        st.info("No extractions available for combined view.")
        return
    
    st.subheader("📊 Combined Results Analysis")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recommendations", len(recommendations))
    
    with col2:
        st.metric("Total Concerns", len(concerns))
    
    with col3:
        total_items = len(recommendations) + len(concerns)
        st.metric("Total Items", total_items)
    
    with col4:
        if recommendations:
            avg_confidence = sum(rec.confidence_score for rec in recommendations) / len(recommendations)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Source distribution
    if recommendations or concerns:
        st.subheader("📈 Distribution by Source")
        
        source_data = {}
        
        for rec in recommendations:
            source = rec.document_source
            if source not in source_data:
                source_data[source] = {'recommendations': 0, 'concerns': 0}
            source_data[source]['recommendations'] += 1
        
        for concern in concerns:
            source = concern.get('document_source', 'Unknown')
            if source not in source_data:
                source_data[source] = {'recommendations': 0, 'concerns': 0}
            source_data[source]['concerns'] += 1
        
        # Create distribution chart
        chart_data = []
        for source, counts in source_data.items():
            chart_data.append({
                'Source': source,
                'Recommendations': counts['recommendations'],
                'Concerns': counts['concerns']
            })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            st.bar_chart(chart_df.set_index('Source'))
    
    # Export options
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Recommendations", use_container_width=True):
            export_recommendations_csv()
    
    with col2:
        if st.button("⚠️ Export Concerns", use_container_width=True):
            export_concerns_csv()
    
    with col3:
        if st.button("📊 Export Combined", use_container_width=True):
            export_combined_csv()

def export_recommendations_csv():
    """Export recommendations to CSV"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    
    if not recommendations:
        st.warning("No recommendations to export.")
        return
    
    # Prepare data for export
    export_data = []
    for rec in recommendations:
        export_data.append({
            'ID': rec.id,
            'Text': rec.text,
            'Source_Document': rec.document_source,
            'Section': rec.section_title,
            'Page_Number': rec.page_number,
            'Confidence_Score': rec.confidence_score,
            'Text_Length': len(rec.text),
            'Extraction_Method': getattr(rec, 'metadata', {}).get('extraction_method', 'Unknown')
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Recommendations CSV",
        data=csv,
        file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_concerns_csv():
    """Export concerns to CSV"""
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not concerns:
        st.warning("No concerns to export.")
        return
    
    # Prepare data for export
    export_data = []
    for concern in concerns:
        export_data.append({
            'ID': concern.get('id', ''),
            'Text': concern.get('text', ''),
            'Source_Document': concern.get('document_source', ''),
            'Section': concern.get('section', ''),
            'Page_Number': concern.get('page_number', ''),
            'Confidence_Score': concern.get('confidence_score', 0),
            'Category': concern.get('category', ''),
            'Text_Length': len(concern.get('text', '')),
            'Extraction_Method': concern.get('extraction_method', 'Unknown')
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Concerns CSV",
        data=csv,
        file_name=f"concerns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_combined_csv():
    """Export combined recommendations and concerns to CSV"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not recommendations and not concerns:
        st.warning("No data to export.")
        return
    
    # Prepare combined data
    export_data = []
    
    for rec in recommendations:
        export_data.append({
            'Type': 'Recommendation',
            'ID': rec.id,
            'Text': rec.text,
            'Source_Document': rec.document_source,
            'Section': rec.section_title,
            'Page_Number': rec.page_number,
            'Confidence_Score': rec.confidence_score,
            'Category': getattr(rec, 'metadata', {}).get('category', 'General'),
            'Text_Length': len(rec.text)
        })
    
    for concern in concerns:
        export_data.append({
            'Type': 'Concern',
            'ID': concern.get('id', ''),
            'Text': concern.get('text', ''),
            'Source_Document': concern.get('document_source', ''),
            'Section': concern.get('section', ''),
            'Page_Number': concern.get('page_number', ''),
            'Confidence_Score': concern.get('confidence_score', 0),
            'Category': concern.get('category', 'General'),
            'Text_Length': len(concern.get('text', ''))
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Combined CSV",
        data=csv,
        file_name=f"combined_extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
