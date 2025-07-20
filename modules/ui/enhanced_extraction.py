# ===============================================
# IMPROVED EXTRACTION INTERFACE WITH COMPARISON
# ===============================================

import re
import logging
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

def render_improved_extraction_interface():
    """Enhanced extraction interface with clear explanations and comparison capability"""
    
    st.header("🔍 Concern Extraction Methods")
    
    # Method explanation section
    st.markdown("""
    **Choose your extraction method** based on your document type and needs. You can run both methods to compare results.
    """)
    
    # Create two columns for method explanations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Enhanced Pattern Extraction
        **Best for: Coroner documents, PFD reports, structured documents**
        
        **How it works:**
        - Uses specialized patterns designed for coroner documents
        - Looks for specific phrases like "CORONER'S CONCERNS", "MATTERS OF CONCERN"
        - Multiple pattern types: standard, flexible, section-based, keyword-based
        - **Same method as PDFxtract** - ensures consistency with BERT annotation
        
        **Advantages:**
        - ✅ **Free** - No API costs
        - ✅ **Fast** - Instant results
        - ✅ **Reliable** - Works offline
        - ✅ **Optimized** for coroner documents
        - ✅ **Consistent** with PDFxtract/BERT workflow
        
        **Best when:**
        - Processing coroner/PFD documents
        - Need fast, cost-free extraction
        - Working with structured documents
        - Want consistency with other tools
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI-Powered (LLM) Extraction
        **Best for: Varied document types, unstructured text, complex language**
        
        **How it works:**
        - Uses OpenAI's ChatGPT to understand and extract concerns
        - AI reads the document and identifies concerns using context
        - Can understand nuanced language and indirect references
        - Adapts to different document formats automatically
        
        **Advantages:**
        - ✅ **Intelligent** - Understands context and meaning
        - ✅ **Flexible** - Adapts to any document format
        - ✅ **Comprehensive** - Can find subtle or indirect concerns
        - ✅ **Context-aware** - Understands relationships between ideas
        
        **Considerations:**
        - 💰 **Requires OpenAI API key** (costs ~$0.001-0.002 per document)
        - 🌐 **Needs internet** connection
        - ⏱️ **Slower** than pattern extraction
        - 📏 **Limited** to ~4000 characters per request
        """)
    
    # Recommendation box
    st.info("""
    💡 **Recommendation**: Start with **Enhanced Pattern Extraction** for coroner documents. 
    It's free, fast, and specifically designed for your document type. Use AI extraction for 
    non-standard documents or when pattern extraction doesn't capture enough content.
    """)
    
    # Document selection
    if not st.session_state.get('uploaded_documents'):
        st.warning("⚠️ Please upload documents first in the Upload tab.")
        return
    
    st.subheader("📁 Document Selection")
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_docs = st.multiselect(
            "Select documents to extract from:",
            options=doc_options,
            default=doc_options[:3] if len(doc_options) <= 3 else doc_options[:3],
            help="Choose which documents to process (showing max 3 by default for testing)"
        )
    
    with col2:
        st.markdown("**📊 Selection Info:**")
        st.write(f"• Available: {len(doc_options)}")
        st.write(f"• Selected: {len(selected_docs)}")
        if len(selected_docs) > 5:
            st.warning("⚠️ Many docs selected - may take time")
    
    if not selected_docs:
        st.info("📌 Please select at least one document to extract from.")
        return
    
    # Settings section
    with st.expander("⚙️ Extraction Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Enhanced Pattern Settings**")
            min_text_length_enhanced = st.slider(
                "Minimum concern length (Enhanced)",
                min_value=10,
                max_value=100,
                value=20,
                help="Minimum character length for extracted concerns"
            )
            
            confidence_threshold_enhanced = st.slider(
                "Confidence threshold (Enhanced)",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum confidence score to include results"
            )
        
        with col2:
            st.markdown("**🤖 AI/LLM Settings**")
            min_text_length_llm = st.slider(
                "Minimum concern length (AI)",
                min_value=10,
                max_value=100,
                value=15,
                help="Minimum character length for AI-extracted concerns"
            )
            
            confidence_threshold_llm = st.slider(
                "Confidence threshold (AI)",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Minimum confidence score for AI results"
            )
    
    # Extraction buttons and results
    st.subheader("🚀 Run Extraction")
    
    # Create columns for buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Run Enhanced Pattern", type="primary", use_container_width=True):
            run_enhanced_extraction_with_comparison(
                selected_docs, min_text_length_enhanced, confidence_threshold_enhanced, "enhanced"
            )
    
    with col2:
        if st.button("🤖 Run AI Extraction", type="secondary", use_container_width=True):
            run_ai_extraction_with_comparison(
                selected_docs, min_text_length_llm, confidence_threshold_llm, "ai"
            )
    
    with col3:
        if st.button("⚡ Run Both & Compare", type="primary", use_container_width=True):
            run_both_extractions_comparison(
                selected_docs, 
                min_text_length_enhanced, confidence_threshold_enhanced,
                min_text_length_llm, confidence_threshold_llm
            )
    
    # Display results section
    display_extraction_results_with_comparison()

def run_enhanced_extraction_with_comparison(selected_docs, min_length, confidence_threshold, result_key):
    """Run enhanced extraction and store results for comparison"""
    
    with st.spinner("🎯 Running Enhanced Pattern Extraction..."):
        extractor = EnhancedConcernExtractor()
        
        results = []
        all_concerns = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doc_name in enumerate(selected_docs):
            progress_bar.progress((i + 1) / len(selected_docs))
            status_text.text(f"Processing {doc_name}...")
            
            doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
            if not doc or not doc.get('content'):
                continue
            
            try:
                result = extractor.extract_concerns_robust(doc['content'], doc_name)
                concerns = result['concerns']
                
                # Filter by settings
                filtered_concerns = [
                    c for c in concerns 
                    if (len(c.get('text', '')) >= min_length and 
                        c.get('confidence_score', 0) >= confidence_threshold)
                ]
                
                all_concerns.extend(filtered_concerns)
                
                results.append({
                    'document': doc_name,
                    'concerns_found': len(filtered_concerns),
                    'methods_tried': result['debug_info'].get('methods_tried', []),
                    'total_before_filter': len(concerns),
                    'success': len(filtered_concerns) > 0
                })
                
            except Exception as e:
                results.append({
                    'document': doc_name,
                    'concerns_found': 0,
                    'error': str(e),
                    'success': False
                })
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state for comparison
        if 'extraction_results' not in st.session_state:
            st.session_state.extraction_results = {}
        
        st.session_state.extraction_results['enhanced'] = {
            'concerns': all_concerns,
            'results': results,
            'timestamp': datetime.now(),
            'method': 'Enhanced Pattern Extraction',
            'settings': {
                'min_length': min_length,
                'confidence_threshold': confidence_threshold
            }
        }
        
        st.success(f"✅ Enhanced Pattern Extraction completed! Found {len(all_concerns)} concerns.")

def run_ai_extraction_with_comparison(selected_docs, min_length, confidence_threshold, result_key):
    """Run AI extraction and store results for comparison"""
    
    # Check for API key
    api_key = st.secrets.get("OPENAI_API_KEY") or None
    if not api_key:
        st.error("❌ OpenAI API key not found. Please add OPENAI_API_KEY to your Streamlit secrets.")
        st.info("💡 You can still use Enhanced Pattern Extraction which doesn't require an API key.")
        return
    
    with st.spinner("🤖 Running AI-Powered Extraction..."):
        extractor = StandardLLMExtractor(api_key)
        
        results = []
        all_concerns = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doc_name in enumerate(selected_docs):
            progress_bar.progress((i + 1) / len(selected_docs))
            status_text.text(f"AI processing {doc_name}...")
            
            doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
            if not doc or not doc.get('content'):
                continue
            
            try:
                result = extractor.extract_concerns_llm(doc['content'], doc_name)
                concerns = result['concerns']
                
                # Filter by settings
                filtered_concerns = [
                    c for c in concerns 
                    if (len(c.get('text', '')) >= min_length and 
                        c.get('confidence_score', 0) >= confidence_threshold)
                ]
                
                all_concerns.extend(filtered_concerns)
                
                results.append({
                    'document': doc_name,
                    'concerns_found': len(filtered_concerns),
                    'method': 'AI/LLM',
                    'total_before_filter': len(concerns),
                    'success': len(filtered_concerns) > 0
                })
                
            except Exception as e:
                results.append({
                    'document': doc_name,
                    'concerns_found': 0,
                    'error': str(e),
                    'success': False
                })
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state for comparison
        if 'extraction_results' not in st.session_state:
            st.session_state.extraction_results = {}
        
        st.session_state.extraction_results['ai'] = {
            'concerns': all_concerns,
            'results': results,
            'timestamp': datetime.now(),
            'method': 'AI-Powered (LLM) Extraction',
            'settings': {
                'min_length': min_length,
                'confidence_threshold': confidence_threshold
            }
        }
        
        st.success(f"✅ AI Extraction completed! Found {len(all_concerns)} concerns.")

def run_both_extractions_comparison(selected_docs, min_length_enhanced, conf_enhanced, min_length_ai, conf_ai):
    """Run both extractions for direct comparison"""
    
    st.info("⚡ Running both extraction methods for comparison...")
    
    # Run enhanced first
    run_enhanced_extraction_with_comparison(selected_docs, min_length_enhanced, conf_enhanced, "enhanced")
    
    # Run AI second
    run_ai_extraction_with_comparison(selected_docs, min_length_ai, conf_ai, "ai")
    
    st.success("🎉 Both extractions completed! See comparison below.")

def display_extraction_results_with_comparison():
    """Display extraction results with comparison capability"""
    
    if 'extraction_results' not in st.session_state or not st.session_state.extraction_results:
        st.info("👆 Run an extraction method above to see results here.")
        return
    
    st.subheader("📊 Extraction Results & Comparison")
    
    results = st.session_state.extraction_results
    
    # Summary comparison
    if len(results) > 1:
        st.markdown("### 🔄 Method Comparison Summary")
        
        comparison_data = []
        for method_key, result_data in results.items():
            comparison_data.append({
                'Method': result_data['method'],
                'Total Concerns': len(result_data['concerns']),
                'Documents Processed': len([r for r in result_data['results'] if r.get('success', False)]),
                'Avg Confidence': f"{sum(c.get('confidence_score', 0) for c in result_data['concerns']) / max(len(result_data['concerns']), 1):.2f}",
                'Run Time': result_data['timestamp'].strftime("%H:%M:%S"),
                'Settings': f"Min: {result_data['settings']['min_length']}, Conf: {result_data['settings']['confidence_threshold']}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Highlight differences
        enhanced_count = len(results.get('enhanced', {}).get('concerns', []))
        ai_count = len(results.get('ai', {}).get('concerns', []))
        
        if 'enhanced' in results and 'ai' in results:
            diff = abs(enhanced_count - ai_count)
            if diff > 0:
                st.warning(f"⚠️ **Difference found**: Enhanced found {enhanced_count} concerns, AI found {ai_count} concerns ({diff} difference)")
            else:
                st.success("✅ Both methods found the same number of concerns!")
    
    # Individual method results
    for method_key, result_data in results.items():
        st.markdown(f"### {result_data['method']} Results")
        
        concerns = result_data['concerns']
        method_results = result_data['results']
        
        # Method summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Concerns Found", len(concerns))
        
        with col2:
            successful_docs = len([r for r in method_results if r.get('success', False)])
            st.metric("Docs Processed", f"{successful_docs}/{len(method_results)}")
        
        with col3:
            if concerns:
                avg_conf = sum(c.get('confidence_score', 0) for c in concerns) / len(concerns)
                st.metric("Avg Confidence", f"{avg_conf:.2f}")
            else:
                st.metric("Avg Confidence", "0.00")
        
        with col4:
            timestamp = result_data['timestamp'].strftime("%H:%M:%S")
            st.metric("Completed At", timestamp)
        
        # Document-level results
        if method_results:
            st.write("**📋 Document Results:**")
            doc_results_df = pd.DataFrame(method_results)
            st.dataframe(doc_results_df, use_container_width=True)
        
        # Sample concerns
        if concerns:
            st.write("**📝 Sample Extracted Concerns:**")
            
            # Show first 3 concerns
            for i, concern in enumerate(concerns[:3]):
                with st.expander(f"{method_key.title()} Concern {i+1} - {concern.get('document_source', 'Unknown')} (Confidence: {concern.get('confidence_score', 0):.2f})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Text:**")
                        st.write(concern.get('text', ''))
                    
                    with col2:
                        st.write(f"**Method:** {concern.get('method', 'Unknown')}")
                        st.write(f"**Type:** {concern.get('type', 'Unknown')}")
                        st.write(f"**Length:** {len(concern.get('text', ''))} chars")
                        st.write(f"**Confidence:** {concern.get('confidence_score', 0):.2f}")
                        
                        if concern.get('pattern_used'):
                            st.write(f"**Pattern:** {concern.get('pattern_used', '')[:30]}...")
            
            if len(concerns) > 3:
                st.info(f"Showing first 3 of {len(concerns)} concerns.")
        
        st.markdown("---")
    
    # Export options
    if results:
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Combine all concerns for export
            all_concerns_combined = []
            for method_key, result_data in results.items():
                for concern in result_data['concerns']:
                    concern_copy = concern.copy()
                    concern_copy['extraction_method'] = result_data['method']
                    all_concerns_combined.append(concern_copy)
            
            if all_concerns_combined:
                concerns_df = pd.DataFrame(all_concerns_combined)
                csv_data = concerns_df.to_csv(index=False)
                
                st.download_button(
                    "📄 Download All Concerns (CSV)",
                    data=csv_data,
                    file_name=f"extracted_concerns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🗑️ Clear All Results"):
                st.session_state.extraction_results = {}
                st.success("Results cleared!")
                st.rerun()

# Import the original classes (assuming they exist in the same file or are imported)
class EnhancedConcernExtractor:
    """Your existing EnhancedConcernExtractor class"""
    pass  # Replace with your actual implementation

class StandardLLMExtractor:
    """Your existing StandardLLMExtractor class"""
    pass  # Replace with your actual implementation

# Main function to replace the existing extraction interface
def render_extraction_tab():
    """Main extraction tab with improved interface"""
    render_improved_extraction_interface()

if __name__ == "__main__":
    st.title("🔍 Improved Concern Extraction with Comparison")
    render_improved_extraction_interface()
