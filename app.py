# Enhanced app.py with Recommendation-Response Alignment
import streamlit as st
import pandas as pd
from datetime import datetime
from modules.integration_helper import (
    setup_search_tab, 
    prepare_documents_for_search, 
    extract_text_from_file,
    get_document_statistics,
    render_analytics_tab
)

def main():
    """Main application with enhanced government analysis features"""
    st.set_page_config(
        page_title="DaphneAI - Government Document Analysis", 
        layout="wide"
    )
    
    st.title("🏛️ DaphneAI - Government Document Analysis")
    st.markdown("*Advanced document processing and search for government content*")
    
    # Enhanced tabs with new alignment feature
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload", 
        "🔍 Extract", 
        "🔍 Search",
        "🔗 Align Rec-Resp",  # NEW TAB
        "📊 Analytics"
    ])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_extract_tab()
    
    with tab3:
        setup_search_tab()
    
    with tab4:
        render_alignment_tab()  # NEW FEATURE
    
    with tab5:
        render_analytics_tab()

def render_alignment_tab():
    """Render the new recommendation-response alignment tab"""
    
    # Import the alignment system
    from modules.ui.search_components import render_recommendation_alignment_interface
    
    # Check if documents are available
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.header("🔗 Recommendation-Response Alignment")
        st.warning("📁 Please upload documents first in the Upload tab.")
        
        # Show what this feature does
        st.markdown("""
        ### 🎯 What This Feature Does:
        
        **🔍 Automatically finds:**
        - All recommendations in your documents
        - Corresponding responses to those recommendations
        - Aligns them using AI similarity matching
        
        **📊 Provides:**
        - Side-by-side view of recommendation + response
        - AI-generated summaries of each pair
        - Confidence scores for alignments
        - Export options for further analysis
        
        **💡 Perfect for:**
        - Government inquiry reports
        - Policy documents and responses
        - Committee recommendations and outcomes
        - Audit findings and management responses
        
        ### 🚀 How to Use:
        1. Upload your documents in the **Upload** tab
        2. Return here to analyze recommendations and responses
        3. Configure search patterns for your specific documents
        4. Let AI find and align recommendation-response pairs
        5. Review results with AI summaries and export findings
        """)
        
        # Show example of what results look like
        st.markdown("### 📋 Example Output:")
        st.code("""
        🟢 Recommendation 1 - Policy (Confidence: 0.85)
        
        📋 Original Extract:
        🎯 Recommendation: "We recommend implementing new cybersecurity protocols"
        📄 Document: security_audit.pdf (Page 3)
        
        ↩️ Related Response: "The department agrees to implement these protocols by Q4"
        📄 Document: govt_response.pdf (Page 1) - Similarity: 0.87
        
        🤖 AI Summary:
        The audit recommends new cybersecurity protocols for enhanced protection. 
        The government response shows acceptance with Q4 implementation timeline.
        Alignment indicates positive policy adoption.
        """)
        
        return
    
    # Render the alignment interface
    documents = st.session_state.documents
    render_recommendation_alignment_interface(documents)

def render_upload_tab():
    """Document upload and processing"""
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files for analysis"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Files", type="primary"):
            with st.spinner("Processing documents..."):
                documents = prepare_documents_for_search(uploaded_files, extract_text_from_file)
                
                st.success(f"✅ Processed {len(documents)} documents")
                
                # Show document types
                doc_types = {}
                for doc in documents:
                    doc_type = doc.get('document_type', 'general')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                if doc_types:
                    st.markdown("**📊 Document Types Detected:**")
                    for doc_type, count in doc_types.items():
                        st.markdown(f"• {doc_type.title()}: {count} documents")
                
                # Show processing summary
                total_chars = sum(len(doc.get('text', '')) for doc in documents)
                avg_pages = total_chars // 2000  # Rough estimate
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", len(documents))
                with col2:
                    st.metric("Total Characters", f"{total_chars:,}")
                with col3:
                    st.metric("Est. Total Pages", avg_pages)
                
                # Next steps
                st.markdown("""
                **✅ Files processed successfully!** 
                
                **🔍 Next Steps:**
                - Go to **Search** tab for keyword searches
                - Go to **Align Rec-Resp** tab to find recommendations and responses
                - Go to **Analytics** tab for document insights
                """)

def render_extract_tab():
    """Document extraction and preview"""
    st.header("🔍 Document Extraction")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first in the Upload tab.")
        return
    
    documents = st.session_state.documents
    
    # Document selector
    doc_names = [doc['filename'] for doc in documents]
    selected_doc = st.selectbox("Select document to preview:", doc_names)
    
    if selected_doc:
        # Find the selected document
        doc = next((d for d in documents if d['filename'] == selected_doc), None)
        
        if doc:
            # Document information
            st.markdown(f"**📄 Document:** {doc['filename']}")
            
            # Document statistics
            text = doc.get('text', '')
            stats = get_document_statistics(text)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Characters", f"{stats['char_count']:,}")
            with col2:
                st.metric("Words", f"{stats['word_count']:,}")
            with col3:
                st.metric("Sentences", f"{stats['sentence_count']:,}")
            with col4:
                st.metric("Est. Pages", stats['estimated_pages'])
            
            # Preview options
            st.markdown("### 📖 Document Preview")
            
            preview_length = st.slider(
                "Preview length (characters)", 
                min_value=500, 
                max_value=min(10000, len(text)), 
                value=min(2000, len(text))
            )
            
            # Show preview
            preview_text = text[:preview_length]
            if len(text) > preview_length:
                preview_text += "... [truncated]"
            
            st.text_area(
                "Document content:",
                value=preview_text,
                height=400,
                disabled=True
            )
            
            # Download processed text
            st.download_button(
                label="📥 Download Extracted Text",
                data=text,
                file_name=f"{selected_doc}_extracted.txt",
                mime="text/plain"
            )

# Document type classification helper
def classify_document_type(filename: str, text: str) -> str:
    """Classify document type based on filename and content"""
    
    filename_lower = filename.lower()
    text_lower = text.lower()
    
    # Policy documents
    if any(keyword in filename_lower for keyword in ['policy', 'guideline', 'framework', 'strategy']):
        return 'policy'
    
    # Reports
    if any(keyword in filename_lower for keyword in ['report', 'review', 'assessment', 'evaluation']):
        return 'report'
    
    # Responses
    if any(keyword in filename_lower for keyword in ['response', 'reply', 'answer', 'feedback']):
        return 'response'
    
    # Audit documents
    if any(keyword in filename_lower for keyword in ['audit', 'inspection', 'compliance']):
        return 'audit'
    
    # Meeting documents
    if any(keyword in filename_lower for keyword in ['meeting', 'minutes', 'agenda']):
        return 'meeting'
    
    # Content-based classification
    recommendation_keywords = ['recommend', 'suggest', 'advise', 'propose']
    response_keywords = ['accept', 'reject', 'agree', 'implement', 'consider']
    
    rec_count = sum(text_lower.count(keyword) for keyword in recommendation_keywords)
    resp_count = sum(text_lower.count(keyword) for keyword in response_keywords)
    
    if rec_count > resp_count and rec_count > 3:
        return 'recommendations'
    elif resp_count > rec_count and resp_count > 3:
        return 'responses'
    
    return 'general'

# Enhanced document preparation
def enhanced_prepare_documents_for_search(uploaded_files, extract_function):
    """Enhanced document preparation with type classification"""
    
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Extract text
            text = extract_function(uploaded_file)
            
            # Classify document type
            doc_type = classify_document_type(uploaded_file.name, text)
            
            # Create enhanced document object
            document = {
                'filename': uploaded_file.name,
                'text': text,
                'document_type': doc_type,
                'upload_time': datetime.now(),
                'file_size': len(text),
                'word_count': len(text.split()),
                'contains_recommendations': 'recommend' in text.lower(),
                'contains_responses': any(word in text.lower() for word in ['accept', 'reject', 'agree', 'implement'])
            }
            
            documents.append(document)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    # Store in session state
    st.session_state.documents = documents
    
    return documents

if __name__ == "__main__":
    main()
