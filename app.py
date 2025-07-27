# Enhanced app.py with Recommendation-Response Alignment - FIXED
import streamlit as st
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Any

# Try to import from modules, but provide fallbacks if they don't exist
try:
    from modules.integration_helper import (
        setup_search_tab, 
        prepare_documents_for_search, 
        extract_text_from_file,
        render_analytics_tab
    )
except ImportError:
    # Provide fallback implementations
    def setup_search_tab():
        from modules.ui.search_components import render_search_interface
        documents = st.session_state.get('documents', [])
        render_search_interface(documents)
    
    def prepare_documents_for_search(uploaded_files, extract_function):
        return enhanced_prepare_documents_for_search(uploaded_files, extract_function)
    
    def extract_text_from_file(uploaded_file):
        # Basic text extraction
        try:
            if uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            else:
                return str(uploaded_file.read(), "utf-8")
        except Exception as e:
            st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
            return ""
    
    def render_analytics_tab():
        st.header("📊 Document Analytics")
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("📁 Please upload documents first")
            return
        
        documents = st.session_state.documents
        
        # Basic analytics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(documents))
        with col2:
            total_words = sum(len(doc.get('text', '').split()) for doc in documents)
            st.metric("Total Words", f"{total_words:,}")
        with col3:
            avg_words = total_words // len(documents) if documents else 0
            st.metric("Avg Words per Doc", f"{avg_words:,}")

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
    try:
        from modules.ui.search_components import render_recommendation_alignment_interface
    except ImportError:
        st.error("Search components module not found. Please check your file structure.")
        return
    
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
            
            # Document statistics - FIXED FUNCTION
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

def get_document_statistics(text: str) -> Dict[str, Any]:
    """Calculate document statistics - FIXED IMPLEMENTATION"""
    
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'estimated_pages': 0,
            'line_count': 0,
            'paragraph_count': 0
        }
    
    # Basic counts
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    # Sentence count (improved)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Line count
    lines = text.split('\n')
    line_count = len(lines)
    
    # Paragraph count (empty lines separate paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    # Estimated pages (2000 characters per page)
    estimated_pages = max(1, char_count // 2000)
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'estimated_pages': estimated_pages,
        'line_count': line_count,
        'paragraph_count': paragraph_count
    }

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

# Additional utility functions for better error handling
def safe_import_search_components():
    """Safely import search components with error handling"""
    try:
        from modules.ui.search_components import render_search_interface, render_recommendation_alignment_interface
        return True, render_search_interface, render_recommendation_alignment_interface
    except ImportError as e:
        st.error(f"Could not import search components: {str(e)}")
        st.markdown("""
        **Missing Files:**
        Please ensure you have the following file structure:
        ```
        modules/
        ├── ui/
        │   └── search_components.py
        └── integration_helper.py
        ```
        """)
        return False, None, None

def create_sample_document():
    """Create a sample document for testing"""
    sample_text = """
    Sample Government Report
    
    This is a sample document for testing the DaphneAI system.
    
    Recommendations:
    1. We recommend implementing new security protocols immediately.
    2. The committee suggests reviewing the current budget allocation.
    3. We advise conducting a comprehensive audit of all systems.
    
    Responses:
    1. The department agrees to implement the security protocols by Q4.
    2. Budget review has been scheduled for next month.
    3. Audit will be conducted by external firm starting January.
    
    This demonstrates how the system can find and align recommendations with their responses.
    """
    
    return {
        'filename': 'sample_report.txt',
        'text': sample_text,
        'document_type': 'report',
        'upload_time': datetime.now(),
        'file_size': len(sample_text),
        'word_count': len(sample_text.split()),
        'contains_recommendations': True,
        'contains_responses': True
    }

# Error recovery function
def handle_app_error():
    """Handle application errors gracefully"""
    st.error("⚠️ Application Error Detected")
    
    st.markdown("""
    **Troubleshooting Steps:**
    
    1. **Check File Structure:** Ensure all required files are present
    2. **Restart Application:** Refresh the page to reset session state
    3. **Use Sample Data:** Click below to load sample data for testing
    4. **Check Dependencies:** Ensure all required packages are installed
    """)
    
    if st.button("🧪 Load Sample Data for Testing"):
        sample_doc = create_sample_document()
        st.session_state.documents = [sample_doc]
        st.success("✅ Sample document loaded! Go to Search or Align Rec-Resp tabs to test.")
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        handle_app_error()
