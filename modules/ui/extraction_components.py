# ===============================================
# FILE: modules/ui/extraction_components.py
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

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
    st.header("🔍 Recommendation Extraction")
    
    st.markdown("""
    Extract recommendations from UK Government inquiry reports, reviews, and official responses.
    Choose your extraction method based on your document type and needs.
    """)
    
    # Check if documents are uploaded
    if not st.session_state.uploaded_documents:
        st.info("📁 Please upload documents first in the Upload tab.")
        return
    
    # Document selection
    render_document_selection()
    
    # Extraction methods
    render_extraction_methods()
    
    # Show results if available
    show_extraction_results()

def render_document_selection():
    """Render document selection interface"""
    st.subheader("📄 Select Documents")
    
    doc_names = [doc['filename'] for doc in st.session_state.uploaded_documents]
    
    # Selection options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_docs = st.multiselect(
            "Choose documents to process:",
            doc_names,
            default=doc_names[:3] if len(doc_names) > 3 else doc_names,  # Select first 3 by default
            key="extraction_doc_selection",
            help="Select the documents you want to extract recommendations from"
        )
    
    with col2:
        # Quick selection buttons
        if st.button("📋 Select All", use_container_width=True):
            st.session_state.extraction_doc_selection = doc_names
            st.rerun()
        
        if st.button("🔄 Clear Selection", use_container_width=True):
            st.session_state.extraction_doc_selection = []
            st.rerun()
    
    # Store selected documents in session state
    st.session_state.selected_extraction_docs = selected_docs
    
    if not selected_docs:
        st.warning("⚠️ Please select at least one document to process.")
        return False
    
    # Show selection summary
    st.info(f"📊 Selected {len(selected_docs)} document(s) for processing")
    return True

def render_extraction_methods():
    """Render extraction method selection and controls"""
    if not st.session_state.get('selected_extraction_docs'):
        return
    
    st.subheader("⚙️ Extraction Methods")
    
    # Method explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Pattern Method
        **Best for:** Structured inquiry documents
        
        **How it works:**
        - Uses regex patterns to find recommendations
        - Looks for "Recommendation 1:", "(a) should be", etc.
        - Optimized for UK government document formats
        
        **Advantages:**
        - ✅ **Free** - No API costs
        - ✅ **Fast** - Instant results
        - ✅ **Reliable** - Works offline
        - ✅ **Structured** - Maintains numbering
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Method
        **Best for:** Unstructured text, complex language
        
        **How it works:**
        - Uses OpenAI GPT to understand context
        - Identifies recommendations by meaning
        - Can handle varied document formats
        
        **Advantages:**
        - ✅ **Intelligent** - Understands context
        - ✅ **Flexible** - Adapts to any format
        - ✅ **Comprehensive** - Finds subtle recommendations
        
        **Requirements:**
        - 💰 OpenAI API key needed
        - 🌐 Internet connection required
        """)
    
    # Method selection buttons
    st.markdown("### 🚀 Choose Your Method")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Run Pattern Method", type="primary", use_container_width=True):
            run_pattern_extraction()
    
    with col2:
        if st.button("🤖 Run AI Method", type="secondary", use_container_width=True):
            run_ai_extraction()
    
    with col3:
        if st.button("⚡ Run Both Methods", type="secondary", use_container_width=True):
            run_both_methods()

def run_pattern_extraction():
    """Run pattern-based extraction"""
    selected_docs = st.session_state.get('selected_extraction_docs', [])
    
    if not selected_docs:
        st.error("No documents selected")
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
                # Extract recommendations
                result = extractor.extract_recommendations(doc['content'], doc_name)
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
                    'validation': validation
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
        'method': 'Pattern Method',
        'recommendations': all_recommendations,
        'doc_results': doc_results,
        'timestamp': datetime.now(),
        'total_docs_processed': len(selected_docs)
    }
    
    # Display results
    with results_container:
        display_extraction_results(all_recommendations, doc_results, "Pattern Method")

def run_ai_extraction():
    """Run AI-powered extraction"""
    # Check for API key first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ **OpenAI API key required** for AI method")
        st.info("💡 **Alternative:** Use the Pattern Method instead - it's free and works great for structured inquiry documents!")
        
        with st.expander("🔧 How to add OpenAI API key"):
            st.markdown("""
            **For local development:**
            1. Create a `.env` file in your project root
            2. Add: `OPENAI_API_KEY=your_api_key_here`
            
            **For Streamlit Cloud:**
            1. Go to your app settings
            2. Click on "Secrets"
            3. Add: `OPENAI_API_KEY = "your_api_key_here"`
            """)
        return
    
    selected_docs = st.session_state.get('selected_extraction_docs', [])
    
    if not selected_docs:
        st.error("No documents selected")
        return
    
    st.subheader("🤖 Running AI Extraction...")
    
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
            status_text.text(f"AI processing: {doc_name}")
            
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
                # Extract recommendations (AI will be used automatically if available)
                result = extractor.extract_recommendations(doc['content'], doc_name)
                recommendations = result['recommendations']
                
                # Filter to prioritize AI results
                ai_recommendations = [r for r in recommendations if r.get('source') == 'ai_extraction']
                if ai_recommendations:
                    recommendations = ai_recommendations
                
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
                    'ai_used': len(ai_recommendations) > 0
                })
                
            except Exception as e:
                doc_results.append({
                    'document': doc_name,
                    'status': f'❌ Error: {str(e)[:50]}',
                    'recommendations_found': 0,
                    'quality_score': 0,
                    'ai_used': False
                })
    
    # Clear progress indicators
    progress_container.empty()
    
    # Store results in session state
    st.session_state.extracted_recommendations = all_recommendations
    st.session_state.extraction_results = {
        'method': 'AI Method',
        'recommendations': all_recommendations,
        'doc_results': doc_results,
        'timestamp': datetime.now(),
        'total_docs_processed': len(selected_docs)
    }
    
    # Display results
    with results_container:
        display_extraction_results(all_recommendations, doc_results, "AI Method")

def run_both_methods():
    """Run both pattern and AI extraction for comparison"""
    selected_docs = st.session_state.get('selected_extraction_docs', [])
    
    if not selected_docs:
        st.error("No documents selected")
        return
    
    # Check for AI availability
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("⚠️ No OpenAI API key found. Running Pattern method only.")
        run_pattern_extraction()
        return
    
    st.subheader("⚡ Running Both Methods for Comparison...")
    
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
            status_text.text(f"Processing (both methods): {doc_name}")
            
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
                # Extract recommendations (both methods will be used)
                result = extractor.extract_recommendations(doc['content'], doc_name)
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
                    'combined': True
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

def display_extraction_results(recommendations: List[Dict], doc_results: List[Dict], method_name: str):
    """Display extraction results with comprehensive statistics"""
    
    if recommendations:
        st.success(f"🎉 **{method_name} completed!** Found **{len(recommendations)}** recommendations from {len(doc_results)} documents.")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Recommendations", len(recommendations))
        
        with col2:
            success_count = len([r for r in doc_results if r['recommendations_found'] > 0])
            success_rate = (success_count / len(doc_results)) * 100 if doc_results else 0
            st.metric("Success Rate", f"{success_rate:.0f}%")
        
        with col3:
            avg_quality = sum(r.get('quality_score', 0) for r in doc_results if 'quality_score' in r)
            if doc_results:
                avg_quality = avg_quality / len([r for r in doc_results if 'quality_score' in r])
            st.metric("Avg Quality Score", f"{avg_quality:.0f}/100")
        
        with col4:
            # Count recommendation types
            main_recs = len([r for r in recommendations if r.get('type') == 'main_recommendation'])
            st.metric("Main Recommendations", main_recs)
        
        # Detailed results by document
        with st.expander("📄 Results by Document", expanded=False):
            for result in doc_results:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{result['document']}**")
                    st.write(f"Status: {result['status']}")
                
                with col2:
                    st.write(f"**Found:** {result['recommendations_found']}")
                
                with col3:
                    if 'quality_score' in result:
                        st.write(f"**Quality:** {result['quality_score']}/100")
                
                # Show method breakdown for combined results
                if result.get('combined'):
                    st.write(f"AI: {result.get('ai_count', 0)}, Pattern: {result.get('pattern_count', 0)}")
        
        # Recommendation type breakdown
        with st.expander("📊 Recommendation Analysis", expanded=False):
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

def show_extraction_results():
    """Show previously extracted recommendations"""
    if not st.session_state.get('extracted_recommendations'):
        return
    
    recommendations = st.session_state.extracted_recommendations
    extraction_info = st.session_state.get('extraction_results', {})
    
    st.subheader(f"📋 Previously Extracted Recommendations ({len(recommendations)})")
    
    # Show extraction metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Method:** {extraction_info.get('method', 'Unknown')}")
    
    with col2:
        timestamp = extraction_info.get('timestamp')
        if timestamp:
            st.write(f"**Extracted:** {timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    with col3:
        st.write(f"**Documents:** {extraction_info.get('total_docs_processed', 'Unknown')}")
    
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

def show_recommendations_compact(recommendations: List[Dict]):
    """Show recommendations in compact format"""
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"**{rec.get('id', i)}** - {rec.get('text', '')[:100]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(rec.get('text', 'No text available'))
            
            with col2:
                st.write(f"**Type:** {rec.get('type', 'Unknown').replace('_', ' ').title()}")
                st.write(f"**Source:** {rec.get('source', 'Unknown').replace('_', ' ').title()}")
                st.write(f"**Confidence:** {rec.get('confidence', 0):.2f}")

def show_recommendations_detailed(recommendations: List[Dict]):
    """Show recommendations in detailed format"""
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"### Recommendation {rec.get('id', i)}")
        
        # Main content
        st.markdown(f"**Text:** {rec.get('text', 'No text available')}")
        
        # Metadata in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write(f"**Type:** {rec.get('type', 'Unknown').replace('_', ' ').title()}")
        
        with col2:
            st.write(f"**Source:** {rec.get('source', 'Unknown').replace('_', ' ').title()}")
        
        with col3:
            st.write(f"**Confidence:** {rec.get('confidence', 0):.2f}")
        
        with col4:
            st.write(f"**Document:** {rec.get('document_source', 'Unknown')}")
        
        # Additional metadata
        if rec.get('extraction_method'):
            st.write(f"**Method:** {rec.get('extraction_method', '').replace('_', ' ').title()}")
        
        st.markdown("---")

def show_export_options(recommendations: List[Dict]):
    """Show export options for recommendations"""
    st.subheader("📤 Export Recommendations")
    
    # Export format selection
    export_format = st.selectbox(
        "Choose export format:",
        ["CSV", "JSON", "Excel", "Plain Text"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Export", type="primary"):
            export_data = prepare_export_data(recommendations, export_format)
            
            if export_format == "CSV":
                st.download_button(
                    label="Download CSV",
                    data=export_data,
                    file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            elif export_format == "JSON":
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("📋 Copy to Clipboard"):
            # Show data for manual copying
            st.text_area(
                "Copy this data:",
                value=str(export_data),
                height=200
            )

def prepare_export_data(recommendations: List[Dict], format_type: str) -> str:
    """Prepare recommendation data for export"""
    if format_type == "CSV":
        # Convert to DataFrame and then CSV
        df = pd.DataFrame(recommendations)
        return df.to_csv(index=False)
    
    elif format_type == "JSON":
        import json
        return json.dumps(recommendations, indent=2, default=str)
    
    elif format_type == "Plain Text":
        lines = []
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"Recommendation {rec.get('id', i)}")
            lines.append(f"Text: {rec.get('text', '')}")
            lines.append(f"Type: {rec.get('type', '')}")
            lines.append(f"Source: {rec.get('source', '')}")
            lines.append("---")
        return "\n".join(lines)
    
    return str(recommendations)

# Helper function for backward compatibility
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
