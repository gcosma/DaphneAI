# Example of how your updated extraction function should look
# File: modules/ui/extraction_components.py

import streamlit as st
from datetime import datetime
from uk_inquiry_extractor import UKInquiryRecommendationExtractor

def render_extraction_tab():
    """Render the updated recommendation extraction tab"""
    st.header("🔍 Recommendation Extraction")
    
    st.markdown("""
    Extract recommendations from UK Government inquiry reports, reviews, and official responses.
    Choose your extraction method based on your needs.
    """)
    
    if not st.session_state.uploaded_documents:
        st.info("📁 Please upload documents first in the Upload tab.")
        return
    
    # Document selection
    doc_names = [doc['filename'] for doc in st.session_state.uploaded_documents]
    selected_docs = st.multiselect(
        "Select documents to process:",
        doc_names,
        default=doc_names,
        key="extraction_doc_selection"
    )
    
    if not selected_docs:
        st.warning("Please select at least one document.")
        return
    
    # Extraction method selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Pattern Method")
        st.markdown("• Free and fast\n• Works offline\n• Great for structured documents")
        if st.button("🚀 Run Pattern Extraction", type="primary", use_container_width=True):
            run_pattern_extraction(selected_docs)
    
    with col2:
        st.markdown("### 🤖 AI Method")
        st.markdown("• Uses ChatGPT\n• Understands context\n• Requires API key")
        if st.button("🧠 Run AI Extraction", type="secondary", use_container_width=True):
            run_ai_extraction(selected_docs)

def run_pattern_extraction(selected_docs):
    """Run the new UK inquiry extraction using patterns"""
    
    st.subheader("🎯 Running Pattern Extraction...")
    
    # Initialize the new extractor
    extractor = UKInquiryRecommendationExtractor()
    
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
                'recommendations_found': 0
            })
            continue
        
        try:
            # Extract recommendations using the new method
            result = extractor.extract_recommendations(doc['content'], doc_name)
            recommendations = result['recommendations']
            
            # Get extraction stats
            stats = extractor.get_extraction_stats(result)
            validation = extractor.validate_extraction(result)
            
            all_recommendations.extend(recommendations)
            
            doc_results.append({
                'document': doc_name,
                'status': '✅ Success',
                'recommendations_found': len(recommendations),
                'quality_score': validation['quality_score'],
                'stats': stats
            })
            
        except Exception as e:
            doc_results.append({
                'document': doc_name,
                'status': f'❌ Error: {str(e)[:50]}',
                'recommendations_found': 0
            })
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Store results in session state
    st.session_state.extracted_recommendations = all_recommendations
    st.session_state.extraction_results = {
        'method': 'Pattern Extraction',
        'recommendations': all_recommendations,
        'doc_results': doc_results,
        'timestamp': datetime.now()
    }
    
    # Display results
    if all_recommendations:
        st.success(f"🎉 **Extraction completed!** Found **{len(all_recommendations)}** recommendations from {len(selected_docs)} documents.")
        
        # Show summary statistics
        with st.expander("📊 Extraction Statistics", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Recommendations", len(all_recommendations))
            
            with col2:
                avg_quality = sum(r.get('quality_score', 0) for r in doc_results if 'quality_score' in r)
                if doc_results:
                    avg_quality = avg_quality / len([r for r in doc_results if 'quality_score' in r])
                st.metric("Average Quality Score", f"{avg_quality:.0f}/100")
            
            with col3:
                success_rate = len([r for r in doc_results if r['recommendations_found'] > 0]) / len(doc_results) * 100
                st.metric("Success Rate", f"{success_rate:.0f}%")
        
        # Show per-document results
        with st.expander("📄 Per-Document Results"):
            for result in doc_results:
                if result['recommendations_found'] > 0:
                    st.write(f"**{result['document']}**: {result['status']} - {result['recommendations_found']} recommendations")
                else:
                    st.write(f"**{result['document']}**: {result['status']}")
    
    else:
        st.warning("⚠️ **No recommendations found**. This could mean:")
        st.markdown("• Documents don't contain structured recommendations\n• Try the AI method for unstructured text\n• Check if documents have readable text content")

def run_ai_extraction(selected_docs):
    """Run AI-powered extraction (similar structure but calls AI method)"""
    
    # Check for API key first
    import os
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ **OpenAI API key required** for AI method")
        st.info("💡 **Suggestion:** Use the Pattern Method instead - it's free and works great for most inquiry documents!")
        return
    
    st.subheader("🤖 Running AI Extraction...")
    
    # Similar implementation but emphasizing AI extraction
    extractor = UKInquiryRecommendationExtractor()
    
    # ... (similar structure to pattern extraction)
    
def show_extraction_results():
    """Display extracted recommendations"""
    if not st.session_state.get('extracted_recommendations'):
        st.info("No recommendations extracted yet.")
        return
    
    recommendations = st.session_state.extracted_recommendations
    
    st.subheader(f"📋 Extracted Recommendations ({len(recommendations)})")
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"Recommendation {rec.get('id', i)}: {rec.get('text', '')[:100]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Full Text:**")
                st.write(rec.get('text', 'No text available'))
            
            with col2:
                st.write("**Metadata:**")
                st.write(f"**Type:** {rec.get('type', 'Unknown')}")
                st.write(f"**Source:** {rec.get('source', 'Unknown')}")
                st.write(f"**Confidence:** {rec.get('confidence', 0):.2f}")
                st.write(f"**Document:** {rec.get('document_source', 'Unknown')}")
