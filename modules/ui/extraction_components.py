# ===============================================
# SIMPLE, CLEAR EXTRACTION INTERFACE
# ===============================================

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

def render_simple_extraction_tab():
    """Simple, user-friendly extraction interface"""
    
    st.title("📄 Extract Concerns from Documents")
    
    st.markdown("""
    This tool finds and extracts concerns from your uploaded documents. 
    Choose the method that works best for your document type.
    """)
    
    # Check if documents are uploaded
    if not st.session_state.get('uploaded_documents'):
        st.error("❌ **No documents found**")
        st.info("👆 Please upload documents first in the **Upload Documents** tab.")
        return
    
    # Show uploaded documents
    docs = st.session_state.uploaded_documents
    st.success(f"✅ **{len(docs)} documents ready** for extraction")
    
    # Simple document selection
    st.subheader("1️⃣ Choose Documents")
    
    doc_names = [doc['filename'] for doc in docs]
    
    if len(docs) <= 3:
        # If few documents, select all by default
        selected_docs = st.multiselect(
            "Which documents do you want to extract concerns from?",
            options=doc_names,
            default=doc_names,
            help="Select the documents to process"
        )
    else:
        # If many documents, let user choose
        selected_docs = st.multiselect(
            "Which documents do you want to extract concerns from?",
            options=doc_names,
            default=doc_names[:3],  # Default to first 3
            help=f"Select from {len(doc_names)} available documents"
        )
    
    if not selected_docs:
        st.warning("⚠️ Please select at least one document above.")
        return
    
    st.info(f"📋 **{len(selected_docs)} documents selected** for processing")
    
    # Method selection with clear explanations
    st.subheader("2️⃣ Choose Extraction Method")
    
    # Create two clear options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Pattern Method (Recommended)
        **Best for coroner documents and PFD reports**
        
        ✅ **Free** - No costs  
        ✅ **Fast** - Instant results  
        ✅ **Reliable** - Works offline  
        ✅ **Accurate** for structured documents  
        
        **How it works:** Looks for specific phrases like "Coroner's Concerns" and "Matters of Concern"
        """)
        
        use_pattern_method = st.button(
            "🎯 Extract with Pattern Method",
            type="primary",
            use_container_width=True,
            help="Recommended for coroner documents"
        )
    
    with col2:
        st.markdown("""
        ### 🤖 AI Method
        **Best for varied document types**
        
        ✅ **Smart** - Understands context  
        ✅ **Flexible** - Works with any format  
        ⚠️ **Requires API key** - Uses OpenAI  
        ⚠️ **Slower** - Takes more time  
        
        **How it works:** AI reads the document and identifies concerns using artificial intelligence
        """)
        
        use_ai_method = st.button(
            "🤖 Extract with AI Method",
            type="secondary",
            use_container_width=True,
            help="Requires OpenAI API key"
        )
    
    # Option to try both
    st.markdown("---")
    compare_methods = st.button(
        "⚡ Try Both Methods & Compare Results",
        type="primary",
        use_container_width=True,
        help="Run both methods to see which works better for your documents"
    )
    
    # Process the selection
    if use_pattern_method:
        run_pattern_extraction(selected_docs)
    
    elif use_ai_method:
        run_ai_extraction(selected_docs)
    
    elif compare_methods:
        run_both_methods(selected_docs)
    
    # Show results if available
    show_extraction_results()

def run_pattern_extraction(selected_docs):
    """Run pattern-based extraction with clear progress"""
    
    st.subheader("🎯 Running Pattern Extraction...")
    
    # Import the enhanced extractor
    try:
        from enhanced_extraction import EnhancedConcernExtractor
        extractor = EnhancedConcernExtractor()
    except ImportError:
        st.error("❌ Extraction system not available. Please check your installation.")
        return
    
    # Progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_concerns = []
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
                    'concerns_found': 0
                })
                continue
            
            try:
                # Run extraction
                result = extractor.extract_concerns_robust(doc['content'], doc_name)
                concerns = result['concerns']
                
                # Filter out very short concerns
                good_concerns = [c for c in concerns if len(c.get('text', '')) > 20]
                
                all_concerns.extend(good_concerns)
                
                doc_results.append({
                    'document': doc_name,
                    'status': f'✅ Success',
                    'concerns_found': len(good_concerns)
                })
                
            except Exception as e:
                doc_results.append({
                    'document': doc_name,
                    'status': f'❌ Error: {str(e)[:50]}',
                    'concerns_found': 0
                })
    
    # Clear progress
    progress_container.empty()
    
    # Store results
    st.session_state.extraction_results = {
        'pattern': {
            'method': 'Pattern Method',
            'concerns': all_concerns,
            'doc_results': doc_results,
            'timestamp': datetime.now()
        }
    }
    
    # Show immediate results
    with results_container:
        if all_concerns:
            st.success(f"🎉 **Pattern extraction completed!** Found **{len(all_concerns)} concerns** from {len(selected_docs)} documents.")
        else:
            st.warning("⚠️ **No concerns found** with pattern method. Try the AI method or check your documents.")

def run_ai_extraction(selected_docs):
    """Run AI extraction with clear progress"""
    
    st.subheader("🤖 Running AI Extraction...")
    
    # Check for API key first
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ **OpenAI API key required** for AI method")
        st.info("💡 **Suggestion:** Use the Pattern Method instead - it's free and works great for coroner documents!")
        return
    
    # Import the AI extractor
    try:
        from enhanced_extraction import StandardLLMExtractor
        extractor = StandardLLMExtractor(api_key)
    except ImportError:
        st.error("❌ AI extraction system not available. Please check your installation.")
        return
    
    # Progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_concerns = []
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
                    'concerns_found': 0
                })
                continue
            
            try:
                # Run AI extraction
                result = extractor.extract_concerns_llm(doc['content'], doc_name)
                concerns = result['concerns']
                
                # Filter out very short concerns
                good_concerns = [c for c in concerns if len(c.get('text', '')) > 15]
                
                all_concerns.extend(good_concerns)
                
                doc_results.append({
                    'document': doc_name,
                    'status': f'✅ Success',
                    'concerns_found': len(good_concerns)
                })
                
            except Exception as e:
                doc_results.append({
                    'document': doc_name,
                    'status': f'❌ Error: {str(e)[:50]}',
                    'concerns_found': 0
                })
    
    # Clear progress
    progress_container.empty()
    
    # Store results
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = {}
    
    st.session_state.extraction_results['ai'] = {
        'method': 'AI Method',
        'concerns': all_concerns,
        'doc_results': doc_results,
        'timestamp': datetime.now()
    }
    
    # Show immediate results
    with results_container:
        if all_concerns:
            st.success(f"🎉 **AI extraction completed!** Found **{len(all_concerns)} concerns** from {len(selected_docs)} documents.")
        else:
            st.warning("⚠️ **No concerns found** with AI method. This might be because the documents don't contain clear concern patterns.")

def run_both_methods(selected_docs):
    """Run both methods for comparison"""
    
    st.subheader("⚡ Running Both Methods for Comparison...")
    
    st.info("🔄 This will run both Pattern and AI methods so you can compare the results.")
    
    # Run pattern first
    with st.expander("🎯 Pattern Method Progress", expanded=True):
        run_pattern_extraction(selected_docs)
    
    # Run AI second
    with st.expander("🤖 AI Method Progress", expanded=True):
        run_ai_extraction(selected_docs)
    
    st.success("🎉 **Both methods completed!** See comparison below.")

def show_extraction_results():
    """Show extraction results in a clear, simple way"""
    
    if not st.session_state.get('extraction_results'):
        return
    
    results = st.session_state.extraction_results
    
    st.subheader("📊 Extraction Results")
    
    # If multiple methods, show comparison
    if len(results) > 1:
        st.markdown("### 📈 Method Comparison")
        
        comparison_data = []
        for method_key, result_data in results.items():
            comparison_data.append({
                'Method': result_data['method'],
                'Concerns Found': len(result_data['concerns']),
                'Completed': result_data['timestamp'].strftime("%H:%M")
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Show which found more
        pattern_count = len(results.get('pattern', {}).get('concerns', []))
        ai_count = len(results.get('ai', {}).get('concerns', []))
        
        if pattern_count > ai_count:
            st.info(f"🎯 **Pattern method found more concerns** ({pattern_count} vs {ai_count})")
        elif ai_count > pattern_count:
            st.info(f"🤖 **AI method found more concerns** ({ai_count} vs {pattern_count})")
        else:
            st.info(f"⚖️ **Both methods found the same number** ({pattern_count} concerns)")
    
    # Show individual results for each method
    for method_key, result_data in results.items():
        method_name = result_data['method']
        concerns = result_data['concerns']
        doc_results = result_data['doc_results']
        
        with st.expander(f"📋 {method_name} Details", expanded=len(results) == 1):
            
            # Document results summary
            st.markdown("**📁 Document Processing Results:**")
            doc_df = pd.DataFrame(doc_results)
            st.dataframe(doc_df, use_container_width=True, hide_index=True)
            
            # Show sample concerns
            if concerns:
                st.markdown(f"**📝 Sample Concerns Found ({len(concerns)} total):**")
                
                for i, concern in enumerate(concerns[:3]):  # Show first 3
                    st.markdown(f"**Concern {i+1}** from *{concern.get('document_source', 'Unknown')}*:")
                    st.write(f"'{concern.get('text', '')[:200]}{'...' if len(concern.get('text', '')) > 200 else ''}'")
                    st.markdown("---")
                
                if len(concerns) > 3:
                    st.info(f"💡 Showing first 3 of {len(concerns)} concerns found.")
            else:
                st.warning("No concerns were extracted with this method.")
    
    # Action buttons
    st.subheader("📥 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏷️ Annotate with BERT", type="primary", use_container_width=True):
            st.info("👆 Go to the **Concept Annotation** tab to analyze these concerns with BERT.")
    
    with col2:
        # Export button
        all_concerns = []
        for result_data in results.values():
            all_concerns.extend(result_data['concerns'])
        
        if all_concerns:
            concerns_df = pd.DataFrame(all_concerns)
            csv_data = concerns_df.to_csv(index=False)
            
            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name=f"extracted_concerns_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.extraction_results = {}
            st.rerun()

# Helper function to replace complex classes with simple imports
def ensure_extractors_available():
    """Ensure extraction classes are available"""
    try:
        from enhanced_extraction import EnhancedConcernExtractor, StandardLLMExtractor
        return True
    except ImportError:
        return False

# Main function to replace the existing interface
def render_extraction_tab():
    """Main extraction tab - simplified version"""
    if ensure_extractors_available():
        render_simple_extraction_tab()
    else:
        st.error("❌ Extraction system not properly configured.")
        st.info("Please ensure the enhanced_extraction module is available.")

if __name__ == "__main__":
    st.set_page_config(page_title="Simple Extraction", layout="wide")
    render_simple_extraction_tab()
