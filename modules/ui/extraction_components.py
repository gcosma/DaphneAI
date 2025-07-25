# ===============================================
# FILE: modules/ui/extraction_components.py (UPDATED FOR GOVERNMENT REPORTS)
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import json

# Import the updated UK inquiry extractor
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
    """Render the main extraction tab for government inquiry documents"""
    st.header("🔍 Recommendation & Response Extraction")
    
    st.markdown("""
    Extract **recommendations** from inquiry reports and **government responses** from response documents. 
    The system automatically detects document type and uses appropriate extraction patterns.
    """)
    
    # Check if documents are uploaded
    if not st.session_state.uploaded_documents:
        st.info("📁 Please upload documents first in the Upload tab.")
        return
    
    # Show document processing status
    show_document_processing_status()
    
    # Document selection and type detection
    render_document_selection_with_type_detection()
    
    # Extraction methods
    render_extraction_methods()
    
    # Show results if available
    show_extraction_results()

def show_document_processing_status():
    """Show status of uploaded documents with document type detection"""
    docs = st.session_state.uploaded_documents
    
    # Analyze document types
    inquiry_docs = 0
    response_docs = 0
    unknown_docs = 0
    
    for doc in docs:
        doc_type = detect_document_type(doc)
        if doc_type == 'inquiry_report':
            inquiry_docs += 1
        elif doc_type == 'government_response':
            response_docs += 1
        else:
            unknown_docs += 1
    
    # Count sections
    sections_docs = len([d for d in docs if d.get('extraction_type') == 'sections_only'])
    full_docs = len([d for d in docs if d.get('extraction_type') == 'full_document'])
    total_sections = sum(len(d.get('sections', [])) for d in docs)
    
    # Display status with document type breakdown
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Documents", len(docs))
    
    with col2:
        st.metric("Inquiry Reports", inquiry_docs)
        
    with col3:
        st.metric("Response Documents", response_docs)
    
    with col4:
        st.metric("Section-Extracted", sections_docs)
    
    with col5:
        st.metric("Total Sections", total_sections)
    
    # Show status message
    if total_sections > 0:
        st.success(f"✅ Ready for extraction! Found {inquiry_docs} inquiry reports and {response_docs} response documents with {total_sections} relevant sections.")
    elif sections_docs > 0:
        st.warning("⚠️ Documents were processed for sections but none were found. Consider using different documents or full extraction.")
    
    if unknown_docs > 0:
        st.info(f"ℹ️ {unknown_docs} documents have unknown type - extraction will attempt to identify content automatically.")

def detect_document_type(doc: Dict) -> str:
    """Detect document type based on content and metadata"""
    content = doc.get('content', '').lower()
    filename = doc.get('filename', '').lower()
    
    # Check filename for indicators
    if 'response' in filename or 'government' in filename:
        return 'government_response'
    elif 'inquiry' in filename or 'report' in filename:
        return 'inquiry_report'
    
    # Check content for indicators
    inquiry_indicators = ['inquiry recommendations', 'final report', 'chair of the inquiry', 'inquiry findings']
    response_indicators = ['government response', 'presented to parliament', 'cabinet office', 'accepted in full', 'accepted in principle']
    
    inquiry_score = sum(1 for indicator in inquiry_indicators if indicator in content)
    response_score = sum(1 for indicator in response_indicators if indicator in content)
    
    if response_score > inquiry_score:
        return 'government_response'
    elif inquiry_score > 0:
        return 'inquiry_report'
    else:
        return 'unknown'

def render_document_selection_with_type_detection():
    """Render document selection with automatic type detection"""
    st.subheader("📋 Document Selection & Type Detection")
    
    docs = st.session_state.uploaded_documents
    
    # Create document type breakdown
    doc_types = {}
    for doc in docs:
        doc_type = detect_document_type(doc)
        if doc_type not in doc_types:
            doc_types[doc_type] = []
        doc_types[doc_type].append(doc)
    
    # Show documents by type
    for doc_type, documents in doc_types.items():
        type_label = {
            'inquiry_report': '📄 Inquiry Reports (Original Recommendations)',
            'government_response': '🏛️ Government Response Documents',
            'unknown': '❓ Unknown Document Type'
        }.get(doc_type, f'📁 {doc_type.title()}')
        
        with st.expander(f"{type_label} ({len(documents)} documents)", expanded=True):
            for doc in documents:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{doc['filename']}**")
                    
                with col2:
                    sections = doc.get('sections', [])
                    if sections:
                        rec_sections = len([s for s in sections if s['type'] == 'recommendations'])
                        resp_sections = len([s for s in sections if s['type'] == 'responses'])
                        st.write(f"📋 {rec_sections}R + {resp_sections}Resp")
                    else:
                        st.write("📄 Full document")
                
                with col3:
                    content_length = len(doc.get('content', ''))
                    st.write(f"{content_length:,} chars")

def render_extraction_methods():
    """Render extraction method selection with updated guidance"""
    st.subheader("⚙️ Extraction Methods")
    
    # Method selection
    col1, col2 = st.columns(2)
    
    with col1:
        extraction_method = st.selectbox(
            "Choose extraction method:",
            options=["Pattern Extraction", "AI Extraction", "Combined (Both)"],
            index=2,  # Default to combined
            help="Pattern extraction is optimized for UK government documents and is free. AI extraction is more flexible but requires OpenAI API."
        )
    
    with col2:
        if st.button("ℹ️ Method Comparison", help="See detailed comparison of extraction methods"):
            show_method_comparison()
    
    # Extraction guidance specific to government documents
    show_government_document_extraction_guidance()
    
    # Run extraction
    if st.button("🚀 Extract Recommendations & Responses", type="primary", use_container_width=True):
        run_extraction(extraction_method)

def show_method_comparison():
    """Show detailed comparison of extraction methods for government documents"""
    with st.expander("📖 Extraction Method Comparison", expanded=True):
        st.markdown("""
        ### 🎯 Pattern Extraction
        **Optimized for UK Government Documents**
        - ✅ **Free** - No API costs
        - ✅ **Fast** - Instant results  
        - ✅ **Reliable** - Works offline
        - ✅ **Specialized** patterns for:
          - Numbered recommendations ("Recommendation 1:", "1.", etc.)
          - Government responses ("accepted in full", "accepted in principle")
          - Implementation statements ("will be implemented", "action will be taken")
        
        ### 🤖 AI Extraction  
        **Flexible and Context-Aware**
        - 🧠 **Intelligent** - Understands context and meaning
        - 🔄 **Adaptive** - Works with varied document formats
        - 📝 **Comprehensive** - Finds subtle or indirect references
        - 💰 **Requires OpenAI API** (small cost per document)
        
        ### ⚡ Combined Extraction
        **Best of Both Worlds**
        - 🎯 Uses pattern extraction for structured content
        - 🤖 Uses AI for complex or ambiguous content  
        - 🔍 Deduplicates overlapping results
        - 📊 Provides comparison statistics
        
        ---
        
        **📋 Document Type Handling:**
        - **Inquiry Reports**: Extracts numbered recommendations and sub-recommendations
        - **Government Responses**: Extracts acceptance/rejection statements and implementation plans
        - **Combined Documents**: Handles both recommendations and responses in one document
        """)

def show_government_document_extraction_guidance():
    """Show specific guidance for government document extraction"""
    st.markdown("""
    ### 📋 What Will Be Extracted:
    
    **From Inquiry Reports:**
    - Numbered recommendations (1, 2, 3...)
    - Sub-recommendations (a, b, c...)
    - Action statements ("should establish", "must implement")
    
    **From Government Response Documents:**
    - Response statements ("accepted in full", "accepted in principle", "not accepted")
    - Implementation plans ("will be implemented through...", "action will be taken...")
    - Progress updates ("has been implemented", "is being reviewed")
    """)

def run_extraction(method: str):
    """Run the extraction process with the selected method"""
    docs = st.session_state.uploaded_documents
    
    if not docs:
        st.error("No documents available for extraction")
        return
    
    # Validate documents
    is_valid, message = validate_documents_for_extraction()
    if not is_valid:
        st.error(f"❌ {message}")
        return
    
    # Initialize extractor
    try:
        extractor = UKInquiryRecommendationExtractor()
    except Exception as e:
        st.error(f"❌ Failed to initialize extractor: {e}")
        return
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Results containers
    all_recommendations = []
    doc_results = []
    
    # Process each document
    for i, doc in enumerate(docs):
        doc_name = doc['filename']
        progress = (i + 1) / len(docs)
        progress_bar.progress(progress)
        status_text.text(f"Processing {doc_name}...")
        
        try:
            # Get content for extraction
            content = get_document_content_for_extraction(doc)
            
            if not content:
                doc_results.append({
                    'document': doc_name,
                    'status': '⚠️ No content available',
                    'recommendations_found': 0,
                    'quality_score': 0,
                    'document_type': 'no_content'
                })
                continue
            
            # Run extraction based on method
            if method == "Pattern Extraction":
                result = extractor._extract_with_patterns(content, doc_name, detect_document_type(doc))
                result = {'recommendations': result, 'extraction_info': {'ai_found': 0, 'pattern_found': len(result)}}
            elif method == "AI Extraction":
                result = extractor._extract_with_ai(content, detect_document_type(doc))
                result = {'recommendations': result or [], 'extraction_info': {'ai_found': len(result or []), 'pattern_found': 0}}
            else:  # Combined
                result = extractor.extract_recommendations(content, doc_name)
            
            recommendations = result.get('recommendations', [])
            
            # Get statistics and validation
            stats = extractor.get_extraction_stats(result)
            validation = extractor.validate_extraction(result)
            
            # Count by extraction source
            ai_count = len([r for r in recommendations if r.get('source') == 'ai_extraction'])
            pattern_count = len([r for r in recommendations if r.get('source') == 'pattern_extraction'])
            
            all_recommendations.extend(recommendations)
            
            # Determine document type
            doc_type = detect_document_type(doc)
            
            doc_results.append({
                'document': doc_name,
                'document_type': doc_type,
                'status': '✅ Success',
                'recommendations_found': len(recommendations),
                'ai_count': ai_count,
                'pattern_count': pattern_count,
                'quality_score': validation['quality_score'],
                'stats': stats,
                'validation': validation,
                'sections_processed': len(doc.get('sections', [])),
                'extraction_type': doc.get('extraction_type', 'unknown'),
                'extraction_method': method
            })
            
        except Exception as e:
            doc_results.append({
                'document': doc_name,
                'status': f'❌ Error: {str(e)[:50]}',
                'recommendations_found': 0,
                'quality_score': 0,
                'document_type': 'error',
                'error': str(e)
            })
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Extraction completed!")
    
    # Store results in session state
    st.session_state.extraction_results = {
        'recommendations': all_recommendations,
        'document_results': doc_results,
        'extraction_method': method,
        'extraction_date': datetime.now().isoformat(),
        'total_documents': len(docs),
        'successful_extractions': len([r for r in doc_results if '✅' in r['status']])
    }
    
    # Show results summary
    show_extraction_summary(all_recommendations, doc_results, method)

def show_extraction_summary(recommendations: List[Dict], doc_results: List[Dict], method: str):
    """Show summary of extraction results"""
    st.success(f"🎉 Extraction completed using {method}!")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items Found", len(recommendations))
    
    with col2:
        successful = len([r for r in doc_results if '✅' in r['status']])
        st.metric("Documents Processed", f"{successful}/{len(doc_results)}")
    
    with col3:
        if recommendations:
            avg_confidence = sum(r.get('confidence', 0) for r in recommendations) / len(recommendations)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col4:
        # Count by type
        inquiry_recs = len([r for r in recommendations if r.get('type') in ['inquiry_recommendation', 'main_recommendation', 'sub_recommendation']])
        gov_responses = len([r for r in recommendations if r.get('type') in ['government_response', 'implementation_plan']])
        st.metric("Recommendations vs Responses", f"{inquiry_recs}/{gov_responses}")

def show_extraction_results():
    """Show detailed extraction results if available"""
    if 'extraction_results' not in st.session_state:
        return
    
    results = st.session_state.extraction_results
    recommendations = results['recommendations']
    doc_results = results['document_results']
    
    st.subheader("📊 Extraction Results")
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Items Found", "📈 Document Results", "📥 Export"])
    
    with tab1:
        show_recommendations_by_type(recommendations)
    
    with tab2:
        show_document_results_table(doc_results)
    
    with tab3:
        show_export_options(recommendations, doc_results)

def show_recommendations_by_type(recommendations: List[Dict]):
    """Show recommendations organized by type"""
    if not recommendations:
        st.info("No recommendations or responses found.")
        return
    
    # Group by type
    by_type = {}
    for rec in recommendations:
        rec_type = rec.get('type', 'unknown')
        if rec_type not in by_type:
            by_type[rec_type] = []
        by_type[rec_type].append(rec)
    
    # Display each type
    for rec_type, items in by_type.items():
        type_labels = {
            'inquiry_recommendation': '📋 Inquiry Recommendations',
            'main_recommendation': '📋 Main Recommendations', 
            'sub_recommendation': '📝 Sub-Recommendations',
            'government_response': '🏛️ Government Responses',
            'implementation_plan': '⚙️ Implementation Plans'
        }
        
        label = type_labels.get(rec_type, f"📄 {rec_type.replace('_', ' ').title()}")
        
        with st.expander(f"{label} ({len(items)} items)", expanded=len(by_type) == 1):
            for i, item in enumerate(items, 1):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{item.get('id', i)}:** {item.get('text', 'No text')}")
                    
                    # Show additional info for responses
                    if item.get('response_type'):
                        response_type_labels = {
                            'accepted_in_full': '✅ Accepted in Full',
                            'accepted_in_principle': '🟡 Accepted in Principle', 
                            'not_accepted': '❌ Not Accepted',
                            'partially_accepted': '🟠 Partially Accepted',
                            'under_review': '🔄 Under Review'
                        }
                        label = response_type_labels.get(item['response_type'], item['response_type'])
                        st.caption(f"Response: {label}")
                    
                    if item.get('recommendation_reference'):
                        st.caption(f"References: Recommendation {item['recommendation_reference']}")
                
                with col2:
                    st.caption(f"Confidence: {item.get('confidence', 0):.2f}")
                    st.caption(f"Source: {item.get('source', 'unknown')}")
                    if item.get('document_source'):
                        st.caption(f"From: {item['document_source'][:20]}...")

def show_document_results_table(doc_results: List[Dict]):
    """Show detailed results for each document"""
    if not doc_results:
        st.info("No document results available.")
        return
    
    # Create results dataframe
    df_data = []
    for result in doc_results:
        df_data.append({
            'Document': result['document'],
            'Type': result.get('document_type', 'unknown').replace('_', ' ').title(),
            'Status': result['status'],
            'Items Found': result['recommendations_found'],
            'Quality Score': f"{result['quality_score']:.1f}",
            'Method': result.get('extraction_method', 'unknown'),
            'Sections': result.get('sections_processed', 0)
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Show document type distribution
    st.subheader("📊 Document Type Distribution")
    
    type_counts = {}
    for result in doc_results:
        doc_type = result.get('document_type', 'unknown')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    type_labels = {
        'inquiry_report': 'Inquiry Reports',
        'government_response': 'Government Responses', 
        'unknown': 'Unknown Type',
        'error': 'Processing Errors',
        'no_content': 'No Content'
    }
    
    col1, col2, col3 = st.columns(3)
    for i, (doc_type, count) in enumerate(type_counts.items()):
        label = type_labels.get(doc_type, doc_type.replace('_', ' ').title())
        with [col1, col2, col3][i % 3]:
            st.metric(label, count)

def show_export_options(recommendations: List[Dict], doc_results: List[Dict]):
    """Show export options for extraction results"""
    st.markdown("### 📥 Export Results")
    
    if not recommendations:
        st.info("No data to export.")
        return
    
    # Export format selection
    export_format = st.selectbox(
        "Choose export format:",
        options=["CSV", "JSON", "Excel"],
        help="Select the format for exporting your extracted recommendations and responses"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Export Recommendations/Responses", use_container_width=True):
            export_recommendations(recommendations, export_format)
    
    with col2:
        if st.button("📊 Export Document Results", use_container_width=True):
            export_document_results(doc_results, export_format)
    
    # Preview data
    st.markdown("### 👀 Data Preview")
    
    # Show first few recommendations
    preview_data = []
    for rec in recommendations[:5]:  # Show first 5
        preview_data.append({
            'ID': rec.get('id', ''),
            'Type': rec.get('type', ''),
            'Text Preview': rec.get('text', '')[:100] + '...' if len(rec.get('text', '')) > 100 else rec.get('text', ''),
            'Source': rec.get('source', ''),
            'Confidence': rec.get('confidence', 0)
        })
    
    if preview_data:
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        
        if len(recommendations) > 5:
            st.caption(f"Showing 5 of {len(recommendations)} total items. Export to see all data.")

def export_recommendations(recommendations: List[Dict], format_type: str):
    """Export recommendations to specified format"""
    try:
        if format_type == "CSV":
            df = pd.DataFrame(recommendations)
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"extracted_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif format_type == "JSON":
            json_data = json.dumps(recommendations, indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 Download JSON",
                data=json_data,
                file_name=f"extracted_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        elif format_type == "Excel":
            # Create Excel file with multiple sheets
            import io
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Main recommendations sheet
                df = pd.DataFrame(recommendations)
                df.to_excel(writer, sheet_name='Recommendations', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Items', 'Inquiry Recommendations', 'Government Responses', 'Average Confidence'],
                    'Value': [
                        len(recommendations),
                        len([r for r in recommendations if r.get('type') in ['inquiry_recommendation', 'main_recommendation']]),
                        len([r for r in recommendations if r.get('type') == 'government_response']),
                        f"{sum(r.get('confidence', 0) for r in recommendations) / len(recommendations):.2f}" if recommendations else "0"
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                label="💾 Download Excel",
                data=buffer.getvalue(),
                file_name=f"extracted_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    except Exception as e:
        st.error(f"Export failed: {e}")

def export_document_results(doc_results: List[Dict], format_type: str):
    """Export document processing results"""
    try:
        df = pd.DataFrame(doc_results)
        
        if format_type == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download Document Results CSV",
                data=csv,
                file_name=f"document_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        # Add other format handlers as needed
        
    except Exception as e:
        st.error(f"Export failed: {e}")

def get_document_content_for_extraction(doc: Dict) -> str:
    """Get content from document for extraction, prioritizing sections"""
    # If document has extracted sections, use only relevant sections
    sections = doc.get('sections', [])
    if sections:
        relevant_content = []
        for section in sections:
            if section['type'] in ['recommendations', 'responses']:
                relevant_content.append(f"\n=== {section['title']} ===\n")
                relevant_content.append(section['content'])
        
        if relevant_content:
            return '\n'.join(relevant_content)
    
    # Fallback to full document content
    return doc.get('content', '')

def validate_documents_for_extraction():
    """Validate that documents are ready for extraction"""
    docs = st.session_state.get('uploaded_documents', [])
    
    if not docs:
        return False, "No documents uploaded"
    
    content_available = sum(1 for doc in docs if doc.get('content'))
    if content_available == 0:
        return False, "No documents have readable content"
    
    # Check for appropriate document types
    inquiry_docs = len([d for d in docs if detect_document_type(d) == 'inquiry_report'])
    response_docs = len([d for d in docs if detect_document_type(d) == 'government_response'])
    
    if inquiry_docs + response_docs == 0:
        return True, f"Ready: {content_available} documents (document type will be auto-detected)"
    else:
        return True, f"Ready: {inquiry_docs} inquiry reports + {response_docs} response documents"

# Utility functions for backwards compatibility
def show_recommendations_compact(recommendations: List[Dict]):
    """Show recommendations in compact format - updated for government documents"""
    for i, rec in enumerate(recommendations, 1):
        # Create more descriptive title
        title = f"**{rec.get('id', i)}** - {rec.get('type', 'item').replace('_', ' ').title()}"
        preview = rec.get('text', '')[:100]
        if len(rec.get('text', '')) > 100:
            preview += "..."
        
        with st.expander(f"{title}: {preview}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Full Text:**")
                st.write(rec.get('text', 'No text available'))
                
                # Show response-specific information
                if rec.get('response_type'):
                    st.write(f"**Response Type:** {rec['response_type'].replace('_', ' ').title()}")
                
                if rec.get('recommendation_reference'):
                    st.write(f"**References:** Recommendation {rec['recommendation_reference']}")
            
            with col2:
                st.write(f"**Type:** {rec.get('type', 'Unknown').replace('_', ' ').title()}")
                st.write(f"**Source:** {rec.get('source', 'Unknown').replace('_', ' ').title()}")
                st.write(f"**Confidence:** {rec.get('confidence', 0):.2f}")
                
                # Show document source if available
                doc_source = rec.get('document_source', '')
                if doc_source:
                    st.write(f"**From:** {doc_source[:20]}...")

def show_extraction_guidance():
    """Show guidance for government document extraction"""
    with st.expander("📖 Government Document Extraction Guide", expanded=False):
        st.markdown("""
        ### Document Types Supported:
        
        **🔍 Original Inquiry Reports** - Contain:
        - Numbered recommendations (1, 2, 3...)
        - Sub-recommendations (a, b, c...)
        - Action statements ("should establish", "must implement")
        
        **🏛️ Government Response Documents** - Contain:
        - Response statements ("accepted in full", "not accepted")
        - Implementation plans ("will be implemented through...")
        - Progress updates ("has been implemented", "is being reviewed")
        
        ### Extraction Process:
        
        1. **Upload documents** using the Upload tab
        2. **Choose extraction method** based on your needs:
           - **Pattern**: Fast, free, optimized for UK government docs
           - **AI**: Flexible, context-aware, requires API key
           - **Combined**: Best coverage, uses both methods
        3. **Review results** and export in your preferred format
        
        ### Tips for Best Results:
        
        - Use "Sections Only" extraction mode for faster processing
        - Upload both inquiry reports AND response documents for complete analysis
        - Pattern extraction works best with well-formatted government documents
        - AI extraction helps with poorly formatted or scanned documents
        """)

# Test function for the updated interface
def test_extraction_interface():
    """Test function for the updated extraction interface"""
    st.write("🧪 Testing Government Document Extraction Interface")
    
    # Mock some test data
    if 'test_documents' not in st.session_state:
        st.session_state.uploaded_documents = [
            {
                'filename': 'infected_blood_inquiry_recommendations.pdf',
                'content': 'Recommendation 1: The Department should establish monitoring. Recommendation 2: Reviews must be conducted.',
                'sections': [{'type': 'recommendations', 'title': 'Final Recommendations', 'content': 'Recommendation 1: The Department should establish monitoring.'}],
                'extraction_type': 'sections_only'
            },
            {
                'filename': 'government_response_infected_blood.pdf', 
                'content': 'Recommendation 1 is accepted in full. Implementation will begin immediately. Recommendation 2 is accepted in principle.',
                'sections': [{'type': 'responses', 'title': 'Government Response', 'content': 'Recommendation 1 is accepted in full.'}],
                'extraction_type': 'sections_only'
            }
        ]
    
    render_extraction_tab()

if __name__ == "__main__":
    test_extraction_interface()
