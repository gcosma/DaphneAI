# ===============================================
# FILE: modules/ui/extraction_components.py (UPDATED VERSION)
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import json

# Import the new UK inquiry extractor
try:
    import sys
    sys.path.append('modules')
    from uk_inquiry_extractor import UKInquiryRecommendationExtractor
except ImportError as e:
    logging.error(f"Import error in extraction_components: {e}")
    # Create mock class for development
    class UKInquiryRecommendationExtractor:
        def extract_recommendations(self, text, source): 
            return {'recommendations': [], 'extraction_info': {'error': 'Extractor not available'}}
        def get_extraction_stats(self, results): 
            return {'total_recommendations': 0}
        def validate_extraction(self, results): 
            return {'quality_score': 0, 'issues': ['Extractor not available']}

def render_extraction_tab():
    """Render the main extraction tab with updated recommendation extraction"""
    st.header("🔍 Recommendation & Response Extraction")
    
    st.markdown("""
    Extract recommendations and responses from your uploaded documents. The system automatically 
    focuses on relevant sections when documents are processed with section extraction.
    """)
    
    # Check if documents are uploaded
    if not st.session_state.uploaded_documents:
        st.info("📁 Please upload documents first in the Upload tab.")
        return
    
    # ✅ NEW: Show section extraction status
    show_document_processing_status()
    
    # Document selection
    render_document_selection()
    
    # Extraction methods
    render_extraction_methods()
    
    # Show results if available
    show_extraction_results()

def show_document_processing_status():
    """Show status of uploaded documents and their sections"""
    docs = st.session_state.uploaded_documents
    
    # Count documents by extraction type
    sections_docs = len([d for d in docs if d.get('extraction_type') == 'sections_only'])
    full_docs = len([d for d in docs if d.get('extraction_type') == 'full_document'])
    total_sections = sum(len(d.get('sections', [])) for d in docs)
    
    # Display status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(docs))
    
    with col2:
        st.metric("Section-Extracted", sections_docs)
    
    with col3:
        st.metric("Full Documents", full_docs)
    
    with col4:
        st.metric("Total Sections", total_sections)
    
    if total_sections > 0:
        st.success(f"✅ {total_sections} recommendation/response sections are ready for extraction!")
    elif sections_docs > 0:
        st.warning("⚠️ Documents were processed for sections but none were found. Consider using different documents or full extraction.")
    else:
        st.info("ℹ️ Documents processed with full extraction - all content will be analyzed.")

def render_document_selection():
    """Render document selection interface with section information"""
    st.subheader("📄 Select Documents")
    
    docs = st.session_state.uploaded_documents
    
    # Create enhanced document list with section info
    doc_options = []
    for doc in docs:
        filename = doc['filename']
        sections = doc.get('sections', [])
        extraction_type = doc.get('extraction_type', 'unknown')
        
        if sections:
            section_info = f" ({len(sections)} sections: {len([s for s in sections if s['type'] == 'recommendations'])} rec, {len([s for s in sections if s['type'] == 'responses'])} resp)"
        else:
            section_info = f" ({extraction_type})"
        
        doc_options.append(f"{filename}{section_info}")
    
    # Selection options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_display = st.multiselect(
            "Choose documents to process:",
            doc_options,
            default=doc_options[:3] if len(doc_options) > 3 else doc_options,  # Select first 3 by default
            key="extraction_doc_selection_display",
            help="Select the documents you want to extract recommendations from"
        )
        
        # Convert back to just filenames
        selected_docs = [opt.split(" (")[0] for opt in selected_display]
    
    with col2:
        # Quick selection buttons
        if st.button("📋 Select All", use_container_width=True):
            st.session_state.extraction_doc_selection_display = doc_options
            st.rerun()
        
        if st.button("🔄 Clear Selection", use_container_width=True):
            st.session_state.extraction_doc_selection_display = []
            st.rerun()
    
    # Store selected documents in session state
    st.session_state.selected_extraction_docs = selected_docs
    
    if not selected_docs:
        st.warning("⚠️ Please select at least one document to process.")
        return False
    
    # ✅ NEW: Show preview of what will be extracted
    with st.expander("📋 Preview: What will be extracted", expanded=False):
        for doc_name in selected_docs:
            doc = next((d for d in docs if d['filename'] == doc_name), None)
            if doc:
                st.write(f"**{doc_name}**")
                sections = doc.get('sections', [])
                if sections:
                    for section in sections:
                        st.write(f"  • {section['type'].title()}: {section['title']} (Pages {section['page_start']}-{section['page_end']})")
                else:
                    content_length = len(doc.get('content', ''))
                    st.write(f"  • Full document content ({content_length:,} characters)")
    
    return True

def render_extraction_methods():
    """Render extraction method selection and execution"""
    if not st.session_state.selected_extraction_docs:
        return
    
    st.subheader("🎯 Choose Extraction Method")
    
    # Method explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Pattern-Based Extraction
        **Fast • Free • Reliable**
        
        - Uses specialized patterns for UK inquiry documents
        - Optimized for coroner reports and government responses
        - Instant results with no API costs
        - Best for structured documents
        """)
        
        if st.button("🚀 Run Pattern Extraction", type="primary", use_container_width=True):
            run_pattern_extraction()
    
    with col2:
        st.markdown("""
        ### 🤖 AI-Powered Extraction
        **Intelligent • Contextual • Flexible**
        
        - Uses OpenAI GPT for context understanding
        - Adapts to various document formats
        - Finds subtle or indirect recommendations
        - Requires API key and internet connection
        """)
        
        api_key_available = bool(os.getenv('OPENAI_API_KEY'))
        
        if api_key_available:
            if st.button("🧠 Run AI Extraction", type="secondary", use_container_width=True):
                run_ai_extraction()
        else:
            st.button("🧠 AI Extraction (API Key Required)", disabled=True, use_container_width=True)
            st.caption("Set OPENAI_API_KEY in environment variables")
    
    # ✅ NEW: Combined extraction option
    st.markdown("---")
    st.markdown("### 🔄 Combined Extraction (Recommended)")
    st.markdown("Run both methods and combine results for maximum coverage and accuracy.")
    
    if st.button("⚡ Run Combined Extraction", type="primary", use_container_width=True):
        run_combined_extraction()

def run_pattern_extraction():
    """Run pattern-based extraction on selected documents"""
    selected_docs = st.session_state.selected_extraction_docs
    
    if not selected_docs:
        st.warning("No documents selected")
        return
    
    st.subheader("🎯 Running Pattern Extraction...")
    
    # Initialize extractor
    extractor = UKInquiryRecommendationExtractor()
    
    # Progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_recommendations = []
        doc_results = []
        
        for i, doc_name in enumerate(selected_docs):
            # Update progress
            progress = (i + 1) / len(selected_docs)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {doc_name}")
            
            # Get document
            doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
            
            if not doc or not doc.get('content'):
                doc_results.append({
                    'document': doc_name,
                    'status': '❌ No content',
                    'recommendations_found': 0,
                    'quality_score': 0
                })
                continue
            
            try:
                # ✅ UPDATED: Use appropriate content based on extraction type
                content = get_document_content_for_extraction(doc)
                
                # Extract recommendations
                result = extractor.extract_recommendations(content, doc_name)
                recommendations = result['recommendations']
                
                # Get statistics and validation
                stats = extractor.get_extraction_stats(result)
                validation = extractor.validate_extraction(result)
                
                all_recommendations.extend(recommendations)
                
                doc_results.append({
                    'document': doc_name,
                    'status': '✅ Success',
                    'recommendations_found': len(recommendations),
                    'quality_score': validation['quality_score'],
                    'stats': stats,
                    'validation': validation,
                    'sections_processed': len(doc.get('sections', [])),
                    'extraction_type': doc.get('extraction_type', 'unknown')
                })
                
            except Exception as e:
                doc_results.append({
                    'document': doc_name,
                    'status': f'❌ Error: {str(e)[:50]}',
                    'recommendations_found': 0,
                    'quality_score': 0
                })
    
    # Clear progress indicators
    progress_container.empty()
    
    # Store results in session state
    st.session_state.extracted_recommendations = all_recommendations
    st.session_state.extraction_results = {
        'method': 'Pattern Extraction',
        'recommendations': all_recommendations,
        'doc_results': doc_results,
        'timestamp': datetime.now(),
        'total_docs_processed': len(selected_docs)
    }
    
    # Display results
    with results_container:
        display_extraction_results(all_recommendations, doc_results, "Pattern Extraction")

def run_ai_extraction():
    """Run AI-powered extraction on selected documents"""
    selected_docs = st.session_state.selected_extraction_docs
    
    if not selected_docs:
        st.warning("No documents selected")
        return
    
    st.subheader("🤖 Running AI Extraction...")
    st.warning("⚠️ This method requires OpenAI API and may incur costs (~$0.001-0.002 per document)")
    
    # Initialize AI extractor (would need to implement)
    # For now, show placeholder
    st.info("🚧 AI extraction is under development. Use Pattern Extraction for now.")
    
    # TODO: Implement AI extraction similar to pattern extraction
    # This would use OpenAI API to extract recommendations

def run_combined_extraction():
    """Run both pattern and AI extraction, then combine results"""
    selected_docs = st.session_state.selected_extraction_docs
    
    if not selected_docs:
        st.warning("No documents selected")
        return
    
    st.subheader("⚡ Running Combined Extraction...")
    
    # Initialize extractor
    extractor = UKInquiryRecommendationExtractor()
    
    # Progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_recommendations = []
        doc_results = []
        
        for i, doc_name in enumerate(selected_docs):
            # Update progress
            progress = (i + 1) / len(selected_docs)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {doc_name}")
            
            # Get document
            doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
            
            if not doc or not doc.get('content'):
                doc_results.append({
                    'document': doc_name,
                    'status': '❌ No content',
                    'recommendations_found': 0,
                    'quality_score': 0
                })
                continue
            
            try:
                # ✅ UPDATED: Use appropriate content based on extraction type
                content = get_document_content_for_extraction(doc)
                
                # Extract recommendations using both methods
                result = extractor.extract_recommendations(content, doc_name)
                recommendations = result['recommendations']
                
                # Get statistics and validation
                stats = extractor.get_extraction_stats(result)
                validation = extractor.validate_extraction(result)
                
                # Count methods used
                ai_count = len([r for r in recommendations if r.get('source') == 'ai_extraction'])
                pattern_count = len([r for r in recommendations if r.get('source') == 'pattern_extraction'])
                
                all_recommendations.extend(recommendations)
                
                doc_results.append({
                    'document': doc_name,
                    'status': '✅ Success',
                    'recommendations_found': len(recommendations),
                    'quality_score': validation['quality_score'],
                    'stats': stats,
                    'validation': validation,
                    'ai_count': ai_count,
                    'pattern_count': pattern_count,
                    'combined': True,
                    'sections_processed': len(doc.get('sections', [])),
                    'extraction_type': doc.get('extraction_type', 'unknown')
                })
                
            except Exception as e:
                doc_results.append({
                    'document': doc_name,
                    'status': f'❌ Error: {str(e)[:50]}',
                    'recommendations_found': 0,
                    'quality_score': 0,
                    'combined': False
                })
    
    # Clear progress indicators
    progress_container.empty()
    
    # Store results in session state
    st.session_state.extracted_recommendations = all_recommendations
    st.session_state.extraction_results = {
        'method': 'Combined Methods',
        'recommendations': all_recommendations,
        'doc_results': doc_results,
        'timestamp': datetime.now(),
        'total_docs_processed': len(selected_docs)
    }
    
    # Display results
    with results_container:
        display_extraction_results(all_recommendations, doc_results, "Combined Methods")

def get_document_content_for_extraction(doc: Dict[str, Any]) -> str:
    """
    ✅ NEW: Get appropriate content for extraction based on document type
    """
    # Always use the full content that was extracted
    # If sections were extracted, the content will already contain only relevant sections
    # If full document was extracted, the content will contain everything
    
    content = doc.get('content', '')
    sections = doc.get('sections', [])
    extraction_type = doc.get('extraction_type', 'unknown')
    
    # Log what we're processing
    if sections:
        logging.info(f"Processing document with {len(sections)} pre-extracted sections")
    else:
        logging.info(f"Processing document with {extraction_type} extraction")
    
    return content

def display_extraction_results(recommendations: List[Dict], doc_results: List[Dict], method_name: str):
    """Display extraction results with comprehensive statistics and section information"""
    
    if recommendations:
        st.success(f"🎉 **{method_name} completed!** Found **{len(recommendations)}** recommendations from {len(doc_results)} documents.")
        
        # ✅ UPDATED: Enhanced results summary with section information
        # Document processing summary
        with st.expander("📊 Processing Summary", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Document Results:**")
                for result in doc_results:
                    status_icon = "✅" if "Success" in result['status'] else "❌"
                    sections_info = ""
                    if 'sections_processed' in result:
                        sections_info = f" ({result['sections_processed']} sections)"
                    
                    st.write(f"{status_icon} {result['document']}: {result['recommendations_found']} found{sections_info}")
            
            with col2:
                st.write("**Extraction Details:**")
                total_sections = sum(r.get('sections_processed', 0) for r in doc_results)
                sections_docs = len([r for r in doc_results if r.get('extraction_type') == 'sections_only'])
                full_docs = len([r for r in doc_results if r.get('extraction_type') == 'full_document'])
                
                st.write(f"• Total sections processed: {total_sections}")
                st.write(f"• Section-extracted docs: {sections_docs}")
                st.write(f"• Full-extracted docs: {full_docs}")
                st.write(f"• Average quality score: {sum(r.get('quality_score', 0) for r in doc_results) / len(doc_results):.2f}")
        
        # Recommendation display options
        display_option = st.radio(
            "Display format:",
            ["Compact View", "Detailed View", "Export Data"],
            horizontal=True
        )
        
        if display_option == "Compact View":
            show_recommendations_compact(recommendations)
        elif display_option == "Detailed View":
            show_recommendations_detailed(recommendations)
        elif display_option == "Export Data":
            show_export_options(recommendations)
        
        # Recommendation type breakdown
        with st.expander("📈 Recommendation Analysis", expanded=False):
            # Create type distribution
            type_counts = {}
            source_counts = {}
            
            for rec in recommendations:
                rec_type = rec.get('type', 'unknown')
                type_counts[rec_type] = type_counts.get(rec_type, 0) + 1
                
                source = rec.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**By Type:**")
                for rec_type, count in type_counts.items():
                    st.write(f"• {rec_type.replace('_', ' ').title()}: {count}")
            
            with col2:
                st.write("**By Source:**")
                for source, count in source_counts.items():
                    st.write(f"• {source.replace('_', ' ').title()}: {count}")
    
    else:
        st.warning(f"⚠️ **No recommendations found** using {method_name}")
        
        # ✅ UPDATED: Better guidance based on extraction type
        sections_processed = any(r.get('sections_processed', 0) > 0 for r in doc_results)
        
        if sections_processed:
            st.markdown("""
            **Analysis**: Documents were processed with section extraction, but no recommendations were found in the extracted sections.
            
            **Possible reasons:**
            • Recommendation sections exist but use different formatting than expected
            • Content might be in tables or non-standard format
            • Text might be unclear due to PDF quality issues
            
            **Suggestions:**
            • Try uploading the same documents with 'Full Document' extraction mode
            • Check the document sections in the Upload tab to verify content
            • Ensure documents contain structured recommendations with clear numbering
            """)
        else:
            st.markdown("""
            **Possible reasons:**
            • Documents don't contain structured recommendations
            • Text might be in image format (not extractable)
            • Different document format than expected
            
            **Suggestions:**
            • Try the other extraction method
            • Check document content in the Upload tab
            • Ensure documents contain readable text
            """)

def show_recommendations_compact(recommendations: List[Dict]):
    """Show recommendations in compact format"""
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"**{rec.get('id', i)}** - {rec.get('text', '')[:100]}...", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Full Text:**")
                st.write(rec.get('text', 'No text available'))
            
            with col2:
                st.write(f"**Type:** {rec.get('type', 'Unknown').replace('_', ' ').title()}")
                st.write(f"**Source:** {rec.get('source', 'Unknown').replace('_', ' ').title()}")
                st.write(f"**Confidence:** {rec.get('confidence', 0):.2f}")
                
                # Show document source if available
                doc_source = rec.get('document_source', '')
                if doc_source:
                    st.write(f"**From:** {doc_source}")

def show_recommendations_detailed(recommendations: List[Dict]):
    """Show recommendations in detailed format"""
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"### Recommendation {rec.get('id', i)}")
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Text:**")
            st.write(rec.get('text', 'No text available'))
        
        with col2:
            st.markdown("**Details:**")
            st.write(f"• **Type:** {rec.get('type', 'Unknown').replace('_', ' ').title()}")
            st.write(f"• **Source:** {rec.get('source', 'Unknown').replace('_', ' ').title()}")
            st.write(f"• **Confidence:** {rec.get('confidence', 0):.2f}")
            st.write(f"• **Theme:** {rec.get('theme', 'Not specified')}")
            
            doc_source = rec.get('document_source', '')
            if doc_source:
                st.write(f"• **Document:** {doc_source}")
            
            extraction_method = rec.get('extraction_method', '')
            if extraction_method:
                st.write(f"• **Method:** {extraction_method}")
        
        st.markdown("---")

def show_export_options(recommendations: List[Dict]):
    """Show export options for extracted recommendations"""
    st.subheader("📥 Export Recommendations")
    
    # Export format selection
    export_format = st.selectbox(
        "Choose export format:",
        ["CSV", "JSON", "Excel", "Plain Text"],
        key="export_format_selection"
    )
    
    # Generate export data
    try:
        export_data = prepare_export_data(recommendations, export_format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "CSV":
            filename = f"recommendations_{timestamp}.csv"
            mime_type = "text/csv"
        elif export_format == "JSON":
            filename = f"recommendations_{timestamp}.json"
            mime_type = "application/json"
        elif export_format == "Excel":
            filename = f"recommendations_{timestamp}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            filename = f"recommendations_{timestamp}.txt"
            mime_type = "text/plain"
        
        # Export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label=f"📥 Download {export_format}",
                data=export_data,
                file_name=filename,
                mime=mime_type,
                use_container_width=True
            )
        
        with col2:
            if st.button("📋 Preview Export Data", use_container_width=True):
                st.text_area(
                    f"Preview ({export_format}):",
                    value=str(export_data)[:1000] + ("..." if len(str(export_data)) > 1000 else ""),
                    height=200,
                    disabled=True
                )
    
    except Exception as e:
        st.error(f"Error preparing export data: {e}")

def prepare_export_data(recommendations: List[Dict], format_type: str) -> str:
    """Prepare recommendation data for export"""
    if format_type == "CSV":
        # Convert to DataFrame and then CSV
        df = pd.DataFrame(recommendations)
        return df.to_csv(index=False)
    
    elif format_type == "JSON":
        return json.dumps(recommendations, indent=2, default=str)
    
    elif format_type == "Excel":
        # For Excel, we return CSV for simplicity (could be enhanced with openpyxl)
        df = pd.DataFrame(recommendations)
        return df.to_csv(index=False)
    
    elif format_type == "Plain Text":
        lines = []
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"Recommendation {rec.get('id', i)}")
            lines.append(f"Text: {rec.get('text', '')}")
            lines.append(f"Type: {rec.get('type', '')}")
            lines.append(f"Source: {rec.get('source', '')}")
            lines.append(f"Confidence: {rec.get('confidence', 0):.2f}")
            lines.append("---")
        return "\n".join(lines)
    
    return str(recommendations)

def show_extraction_results():
    """Show previously extracted recommendations with enhanced information"""
    if not st.session_state.get('extracted_recommendations'):
        return
    
    recommendations = st.session_state.extracted_recommendations
    extraction_info = st.session_state.get('extraction_results', {})
    
    st.subheader(f"📋 Previously Extracted Recommendations ({len(recommendations)})")
    
    # Show extraction metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write(f"**Method:** {extraction_info.get('method', 'Unknown')}")
    
    with col2:
        timestamp = extraction_info.get('timestamp')
        if timestamp:
            st.write(f"**Extracted:** {timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    with col3:
        st.write(f"**Documents:** {extraction_info.get('total_docs_processed', 'Unknown')}")
    
    with col4:
        # Show section processing info if available
        doc_results = extraction_info.get('doc_results', [])
        total_sections = sum(r.get('sections_processed', 0) for r in doc_results)
        if total_sections > 0:
            st.write(f"**Sections:** {total_sections}")
        else:
            st.write("**Type:** Full document")
    
    # Recommendation display options
    display_option = st.radio(
        "Display format:",
        ["Compact View", "Detailed View", "Export Data"],
        horizontal=True,
        key="previous_results_display"
    )
    
    if display_option == "Compact View":
        show_recommendations_compact(recommendations)
    elif display_option == "Detailed View":
        show_recommendations_detailed(recommendations)
    elif display_option == "Export Data":
        show_export_options(recommendations)

# Helper functions for backward compatibility and integration
def get_extracted_recommendations():
    """Get extracted recommendations from session state"""
    return st.session_state.get('extracted_recommendations', [])

def get_extraction_statistics():
    """Get extraction statistics"""
    extraction_results = st.session_state.get('extraction_results', {})
    recommendations = st.session_state.get('extracted_recommendations', [])
    
    return {
        'total_recommendations': len(recommendations),
        'method_used': extraction_results.get('method', 'Unknown'),
        'timestamp': extraction_results.get('timestamp'),
        'documents_processed': extraction_results.get('total_docs_processed', 0)
    }

def clear_extraction_results():
    """Clear extraction results from session state"""
    st.session_state.extracted_recommendations = []
    st.session_state.extraction_results = {}
    st.success("✅ Extraction results cleared!")

# ✅ NEW: Section-aware extraction utilities
def get_sections_summary():
    """Get summary of sections across all uploaded documents"""
    docs = st.session_state.get('uploaded_documents', [])
    
    summary = {
        'total_documents': len(docs),
        'documents_with_sections': 0,
        'total_sections': 0,
        'recommendations_sections': 0,
        'responses_sections': 0,
        'section_details': []
    }
    
    for doc in docs:
        sections = doc.get('sections', [])
        if sections:
            summary['documents_with_sections'] += 1
            summary['total_sections'] += len(sections)
            
            for section in sections:
                if section['type'] == 'recommendations':
                    summary['recommendations_sections'] += 1
                elif section['type'] == 'responses':
                    summary['responses_sections'] += 1
                
                summary['section_details'].append({
                    'document': doc['filename'],
                    'type': section['type'],
                    'title': section['title'],
                    'pages': f"{section['page_start']}-{section['page_end']}",
                    'word_count': section.get('content_stats', {}).get('word_count', 0)
                })
    
    return summary

def validate_documents_for_extraction():
    """Validate that documents are ready for extraction"""
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        return False, "No documents uploaded"
    
    content_available = sum(1 for doc in docs if doc.get('content'))
    if content_available == 0:
        return False, "No documents have readable content"
    
    sections_docs = len([d for d in docs if d.get('sections')])
    if sections_docs > 0:
        return True, f"Ready: {content_available} documents ({sections_docs} with sections)"
    else:
        return True, f"Ready: {content_available} documents (full extraction)"

def show_extraction_guidance():
    """Show guidance for users on choosing extraction methods"""
    with st.expander("📖 Extraction Method Guide", expanded=False):
        st.markdown("""
        ### When to Use Each Method:
        
        **🔍 Pattern Extraction** - Best for:
        - UK Government inquiry reports
        - Coroner reports and PFD documents
        - Structured documents with clear numbering
        - Fast processing without API costs
        
        **🤖 AI Extraction** - Best for:
        - Unstructured or varied document formats
        - Documents with complex language
        - When pattern extraction misses content
        - Documents from different countries/systems
        
        **⚡ Combined Extraction** - Best for:
        - Maximum coverage and accuracy
        - Critical analysis where completeness matters
        - When document format is unknown
        - Research or comprehensive analysis
        
        ### Section vs Full Document Processing:
        
        **Sections Only**: Your documents were processed to extract only recommendations and responses sections. This focuses the extraction on the most relevant content and improves accuracy.
        
        **Full Document**: All document content is analyzed. Use this if section extraction missed important content or for documents without clear section structure.
        """)

# ✅ NEW: Add test function for extraction
def test_extraction_setup():
    """Test function to verify extraction setup is working"""
    st.subheader("🧪 Test Extraction Setup")
    
    if st.button("Run Extraction Test"):
        # Test extractor availability
        try:
            extractor = UKInquiryRecommendationExtractor()
            st.success("✅ UK Inquiry Extractor loaded successfully")
        except Exception as e:
            st.error(f"❌ Extractor loading failed: {e}")
            return
        
        # Test document availability
        docs = st.session_state.get('uploaded_documents', [])
        if not docs:
            st.warning("⚠️ No documents available for testing")
            return
        
        st.success(f"✅ Found {len(docs)} documents for testing")
        
        # Test content extraction
        test_doc = docs[0]
        content = get_document_content_for_extraction(test_doc)
        
        if content:
            st.success(f"✅ Content available: {len(content):,} characters")
            
            # Show section info if available
            sections = test_doc.get('sections', [])
            if sections:
                st.info(f"📋 Document has {len(sections)} sections ready for extraction")
            else:
                st.info("📄 Document will be processed as full content")
        else:
            st.error("❌ No content available for extraction")

# Initialize extraction-related session state
def initialize_extraction_state():
    """Initialize extraction-specific session state variables"""
    if 'extracted_recommendations' not in st.session_state:
        st.session_state.extracted_recommendations = []
    
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = {}
    
    if 'selected_extraction_docs' not in st.session_state:
        st.session_state.selected_extraction_docs = []
    
    if 'extraction_method_used' not in st.session_state:
        st.session_state.extraction_method_used = None
    
    if 'last_extraction_timestamp' not in st.session_state:
        st.session_state.last_extraction_timestamp = None

# Call initialization
initialize_extraction_state()

# Module-level configuration
EXTRACTION_CONFIG = {
    'default_method': 'pattern',
    'enable_ai_extraction': bool(os.getenv('OPENAI_API_KEY')),
    'batch_size': 10,
    'max_content_length': 50000,  # Max characters per document
    'min_recommendation_length': 20,  # Min chars for valid recommendation
}

# Logging setup for this module
logger = logging.getLogger(__name__)

def log_extraction_activity(activity: str, details: Dict[str, Any] = None):
    """Log extraction activities for debugging and monitoring"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'activity': activity,
        'details': details or {}
    }
    logger.info(f"Extraction: {activity} - {details}")

# Error handling wrapper
def safe_extraction_operation(func, *args, **kwargs):
    """Wrapper for safe execution of extraction operations"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Extraction operation failed: {e}", exc_info=True)
        st.error(f"Extraction failed: {str(e)}")
        return None

# Export this module's key functions for use by other components
__all__ = [
    'render_extraction_tab',
    'get_extracted_recommendations', 
    'get_extraction_statistics',
    'clear_extraction_results',
    'get_sections_summary',
    'validate_documents_for_extraction'
]
