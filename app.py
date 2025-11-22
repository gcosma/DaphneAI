# Fixed app.py - DaphneAI Government Document Analysis
import streamlit as st
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Any
import logging
import traceback
from collections import Counter

# ry to import NLTK
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ===== RECOMMENDATION EXTRACTOR CODE =====

class SimpleRecommendationExtractor:
    """Extract recommendations by finding sentences with verbs that suggest actions"""
    
    def __init__(self):
        # More specific recommendation patterns
        self.recommendation_patterns = [
            r'we recommend\b',
            r'it is recommended\b',
            r'the inquiry recommends?\b',
            r'the committee recommends?\b',
            r'the report recommends?\b',
            r'should\b.*\b(?:implement|establish|ensure|improve)',  # Only "should" with action verbs
        ]
        
        # Bullet point patterns for unnumbered lists
        self.bullet_patterns = [
            r'^\s*[•●■▪▸►]+\s+',      # Unicode bullets
            r'^\s*[-–—]\s+',           # Dashes/hyphens  
            r'^\s*[*]\s+',             # Asterisks
            r'^\s*[○◦]\s+',            # Open circles
            r'^\s*[✓✔]\s+',            # Check marks
        ]
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.5) -> List[Dict]:
        recommendations = []
        sentences = self._split_sentences(text)
        
        for idx, sentence in enumerate(sentences):
            # Method 1: Look for explicit recommendation phrases (now more strict)
            if self._contains_recommendation_phrase(sentence):
                verb = self._extract_main_verb(sentence)
                # Only add if it's a substantial sentence (not just headers)
                if len(sentence.split()) >= 5:  # At least 5 words
                    recommendations.append({
                        'text': sentence,
                        'verb': verb,
                        'method': 'keyword',
                        'confidence': 0.85,
                        'position': idx
                    })
            
            # Method 2: Look for sentences starting with action gerunds (THIS IS THE MAIN METHOD)
            elif self._starts_with_gerund(sentence):
                verb = self._extract_first_verb(sentence)
                recommendations.append({
                    'text': sentence,
                    'verb': verb,
                    'method': 'gerund',
                    'confidence': 0.9,
                    'position': idx
                })
            
            # Method 3: Modal verbs with action (now more strict)
            elif self._contains_strong_modal(sentence):
                verb = self._extract_main_verb(sentence)
                # Only if sentence is substantial and has action verbs
                if len(sentence.split()) >= 8:  # Longer sentences for modals
                    recommendations.append({
                        'text': sentence,
                        'verb': verb,
                        'method': 'modal',
                        'confidence': 0.7,
                        'position': idx
                    })
        
        # Filter by confidence
        recommendations = [r for r in recommendations if r['confidence'] >= min_confidence]
        
        # Remove duplicates
        recommendations = self._remove_duplicates(recommendations)
        
        return recommendations
    
    def _split_sentences(self, text: str) -> List[str]:
        # This regex might not handle all cases properly
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 30 and len(s.split()) >= 5]
        
    def _contains_recommendation_phrase(self, sentence: str) -> bool:
        """Check if sentence contains explicit recommendation phrases (stricter)"""
        sentence_lower = sentence.lower()
        
        # Must contain these exact phrases
        for pattern in self.recommendation_patterns:
            if re.search(pattern, sentence_lower):
                return True
        return False
    
    def _starts_with_gerund(self, sentence: str) -> bool:
        """Check if sentence starts with a gerund (verb-ing) - MAIN DETECTION METHOD"""
        words = sentence.split()
        if not words:
            return False
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        # Must end with 'ing' and be long enough
        if first_word.endswith('ing') and len(first_word) > 5:
            # Check against common recommendation gerunds
            common_gerunds = [
                'improving', 'ensuring', 'establishing', 'enabling', 'broadening',
                'reforming', 'implementing', 'developing', 'creating', 'enhancing',
                'introducing', 'reviewing', 'updating', 'providing', 'supporting',
                'maintaining', 'expanding', 'reducing', 'addressing', 'promoting',
                'strengthening', 'facilitating', 'encouraging', 'adopting', 'requiring', 'need'
            ]
            
            if first_word in common_gerunds:
                return True
            
            # Also check with NLTK if available
            if NLP_AVAILABLE:
                try:
                    base = first_word[:-3]  # Remove 'ing'
                    tagged = pos_tag([base])
                    return tagged[0][1].startswith('VB')
                except:
                    pass
        
        return False
    
    def _contains_strong_modal(self, sentence: str) -> bool:
        """Check if sentence contains modal verbs with clear action implications"""
        sentence_lower = sentence.lower()
        
        # Only match modals followed by action verbs
        modal_action_patterns = [
            r'\bshould\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
            r'\bmust\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
            r'\bneed to\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
        ]
        
        for pattern in modal_action_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _extract_first_verb(self, sentence: str) -> str:
        """Extract the first verb from a sentence"""
        words = sentence.split()
        if not words:
            return 'unknown'
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        # If it ends with 'ing', remove the 'ing' to get base form
        if first_word.endswith('ing') and len(first_word) > 5:
            return first_word[:-3]
        
        return first_word
    
    def _extract_main_verb(self, sentence: str) -> str:
        """Extract the main verb from a sentence using NLP"""
        if NLP_AVAILABLE:
            try:
                tokens = word_tokenize(sentence[:200])  # Limit length for performance
                tagged = pos_tag(tokens)
                
                # Find action verbs (not auxiliaries)
                verbs = [word.lower() for word, pos in tagged if pos.startswith('VB')]
                
                if verbs:
                    auxiliaries = {'be', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}
                    for verb in verbs:
                        if verb not in auxiliaries:
                            return verb
                    return verbs[0]
            except:
                pass
        
        # Fallback: look for common recommendation verbs
        sentence_lower = sentence.lower()
        common_verbs = [
            'recommend', 'suggest', 'advise', 'propose', 'improve', 'ensure',
            'establish', 'enable', 'broaden', 'reform', 'implement', 'develop',
            'create', 'enhance', 'introduce', 'adopt', 'provide', 'strengthen'
        ]
        
        for verb in common_verbs:
            if verb in sentence_lower:
                return verb
        
        return 'unknown'
    
    def _remove_duplicates(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations based on text similarity"""
        if not recommendations:
            return []
        
        unique = []
        seen_texts = []
        
        for rec in recommendations:
            text = rec['text'].lower().strip()
            
            is_duplicate = False
            for seen in seen_texts:
                if self._similarity(text, seen) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(rec)
                seen_texts.append(text)
        
        return unique
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_verb_statistics(self, recommendations: List[Dict]) -> Dict:
        """Get statistics about the verbs used"""
        verb_counts = Counter(r['verb'] for r in recommendations)
        method_counts = Counter(r['method'] for r in recommendations)
        
        return {
            'total': len(recommendations),
            'unique_verbs': len(verb_counts),
            'verb_frequency': dict(verb_counts.most_common()),
            'method_distribution': dict(method_counts),
            'avg_confidence': sum(r['confidence'] for r in recommendations) / len(recommendations) if recommendations else 0
        }


def extract_recommendations_simple(text: str, min_confidence: float = 0.7) -> List[Dict]:
    """Simple function to extract recommendations from text"""
    extractor = SimpleRecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)

# ===== END RECOMMENDATION EXTRACTOR CODE =====

def safe_import_with_fallback():
    """Safely import modules with comprehensive fallbacks"""
    try:
        from modules.integration_helper import (
            setup_search_tab, 
            prepare_documents_for_search, 
            extract_text_from_file,
            render_analytics_tab
        )
        return True, setup_search_tab, prepare_documents_for_search, extract_text_from_file, render_analytics_tab
    except ImportError as e:
        logger.warning(f"Import error: {e}")
        return False, None, None, None, None

def main():
    """Main application with enhanced error handling"""
    try:
        st.set_page_config(
            page_title="DaphneAI - Government Document Analysis", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏛️ DaphneAI - Government Document Analysis")
        st.markdown("*Advanced document processing and search for government content*")
        
        # Check module availability
        modules_available, setup_search_tab, prepare_documents_for_search, extract_text_from_file, render_analytics_tab = safe_import_with_fallback()
        
        if not modules_available:
            render_fallback_interface()
            return
        
        # Enhanced tabs with error handling
        try:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📁 Upload", 
                "🔍 Extract", 
                "🔍 Search",
                "🔗 Align Rec-Resp",
                "📊 Analytics",
                "🎯 Recommendations"  # NEW - This is tab6
            ])
            
            with tab1:
                render_upload_tab_safe(prepare_documents_for_search, extract_text_from_file)
            
            with tab2:
                render_extract_tab_safe()
            
            with tab3:
                render_search_tab_safe(setup_search_tab)
            
            with tab4:
                render_alignment_tab_safe()
            
            with tab5:
                render_analytics_tab_safe(render_analytics_tab)
            
            with tab6:  # NEW - RECOMMENDATIONS TAB
                render_recommendations_tab()
                
        except Exception as e:
            st.error(f"Tab rendering error: {str(e)}")
            render_error_recovery()
            
    except Exception as e:
        st.error(f"⚠️ Application Error: {str(e)}")
        logger.error(f"Main application error: {e}")
        logger.error(traceback.format_exc())
        render_error_recovery()

def render_fallback_interface():
    """Render a basic fallback interface when modules aren't available"""
    st.warning("🔧 Module loading issues detected. Using fallback interface.")
    
    # Basic file upload
    st.header("📁 Basic Document Upload")
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Files (Basic)", type="primary"):
            documents = []
            for file in uploaded_files:
                try:
                    # Basic text extraction
                    if file.type == "text/plain":
                        text = str(file.read(), "utf-8")
                    else:
                        text = f"[Content from {file.name} - processing not available]"
                    
                    doc = {
                        'filename': file.name,
                        'text': text,
                        'word_count': len(text.split()),
                        'upload_time': datetime.now()
                    }
                    documents.append(doc)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            if documents:
                st.session_state.documents = documents
                st.success(f"✅ Processed {len(documents)} documents in basic mode")
    
    # Basic search if documents exist
    if 'documents' in st.session_state and st.session_state.documents:
        st.header("🔍 Basic Search")
        query = st.text_input("Search documents:", placeholder="Enter search terms...")
        
        if query:
            results = []
            for doc in st.session_state.documents:
                if query.lower() in doc.get('text', '').lower():
                    count = doc['text'].lower().count(query.lower())
                    results.append({
                        'filename': doc['filename'],
                        'matches': count
                    })
            
            if results:
                st.success(f"Found {len(results)} matching documents")
                for result in results:
                    st.write(f"📄 {result['filename']} - {result['matches']} matches")
            else:
                st.warning("No matches found")

def render_upload_tab_safe(prepare_documents_for_search, extract_text_from_file):
    """Safe document upload with error handling"""
    try:
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
                    try:
                        if prepare_documents_for_search and extract_text_from_file:
                            documents = prepare_documents_for_search(uploaded_files, extract_text_from_file)
                        else:
                            documents = fallback_process_documents(uploaded_files)
                        
                        st.success(f"✅ Processed {len(documents)} documents")
                        
                        # Show basic statistics
                        total_words = sum(doc.get('word_count', 0) for doc in documents)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents", len(documents))
                        with col2:
                            st.metric("Total Words", f"{total_words:,}")
                        with col3:
                            avg_words = total_words // len(documents) if documents else 0
                            st.metric("Avg Words", f"{avg_words:,}")
                        
                        st.markdown("""
                        **✅ Files processed successfully!** 
                        
                        **🔍 Next Steps:**
                        - Go to **Search** tab for keyword searches
                        - Go to **Align Rec-Resp** tab to find recommendations and responses
                        - Go to **Analytics** tab for document insights
                        """)
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        logger.error(f"Document processing error: {e}")
                        
    except Exception as e:
        st.error(f"Upload tab error: {str(e)}")
        render_basic_upload_fallback()

def render_extract_tab_safe():
    """Safe document extraction with error handling"""
    try:
        st.header("🔍 Document Extraction")
        
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("📁 Please upload documents first in the Upload tab.")
            return
        
        documents = st.session_state.documents
        doc_names = [doc['filename'] for doc in documents]
        selected_doc = st.selectbox("Select document to preview:", doc_names)
        
        if selected_doc:
            doc = next((d for d in documents if d['filename'] == selected_doc), None)
            
            if doc and 'text' in doc:
                text = doc['text']
                
                # Safe statistics calculation
                word_count = len(text.split()) if text else 0
                char_count = len(text) if text else 0
                estimated_pages = max(1, char_count // 2000)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Characters", f"{char_count:,}")
                with col2:
                    st.metric("Words", f"{word_count:,}")
                with col3:
                    # Safe sentence count
                    try:
                        sentences = re.split(r'[.!?]+', text)
                        sentence_count = len([s for s in sentences if s.strip()])
                    except:
                        sentence_count = word_count // 10  # Estimate
                    st.metric("Sentences", f"{sentence_count:,}")
                with col4:
                    st.metric("Est. Pages", estimated_pages)
                
                # Preview
                st.markdown("### 📖 Document Preview")
                preview_length = st.slider(
                    "Preview length (characters)", 
                    min_value=500, 
                    max_value=min(10000, len(text)), 
                    value=min(2000, len(text))
                )
                
                preview_text = text[:preview_length]
                if len(text) > preview_length:
                    preview_text += "... [truncated]"
                
                st.text_area(
                    "Document content:",
                    value=preview_text,
                    height=400,
                    disabled=True
                )
                
                # Download option
                st.download_button(
                    label="📥 Download Extracted Text",
                    data=text,
                    file_name=f"{selected_doc}_extracted.txt",
                    mime="text/plain"
                )
            else:
                st.error("Document text not available")
                
    except Exception as e:
        st.error(f"Extract tab error: {str(e)}")
        logger.error(f"Extract tab error: {e}")

def render_search_tab_safe(setup_search_tab):
    """Safe search tab with error handling"""
    try:
        if setup_search_tab:
            setup_search_tab()
        else:
            render_basic_search_fallback()
    except Exception as e:
        st.error(f"Search tab error: {str(e)}")
        logger.error(f"Search tab error: {e}")
        render_basic_search_fallback()

def render_alignment_tab_safe():
    """Safe alignment tab with error handling"""
    try:
        st.header("🔗 Recommendation-Response Alignment")
        
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("📁 Please upload documents first in the Upload tab.")
            show_alignment_feature_info()
            return
        
        try:
            # Try to import the alignment interface
            from modules.ui.search_components import render_recommendation_alignment_interface
            documents = st.session_state.documents
            render_recommendation_alignment_interface(documents)
        except ImportError:
            st.error("🔧 Alignment module not available. Using fallback interface.")
            render_basic_alignment_fallback()
            
    except Exception as e:
        st.error(f"Alignment tab error: {str(e)}")
        logger.error(f"Alignment tab error: {e}")
        render_basic_alignment_fallback()

def render_analytics_tab_safe(render_analytics_tab):
    """Safe analytics tab with error handling"""
    try:
        if render_analytics_tab:
            render_analytics_tab()
        else:
            render_basic_analytics_fallback()
    except Exception as e:
        st.error(f"Analytics tab error: {str(e)}")
        logger.error(f"Analytics tab error: {e}")
        render_basic_analytics_fallback()

def fallback_process_documents(uploaded_files):
    """Fallback document processing when modules aren't available"""
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Basic text extraction
            if uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                # Try basic PDF extraction
                try:
                    import PyPDF2
                    from io import BytesIO
                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                except ImportError:
                    text = f"[PDF content from {uploaded_file.name} - PDF processing not available]"
                except Exception:
                    text = f"[PDF processing failed for {uploaded_file.name}]"
            else:
                text = f"[Content from {uploaded_file.name} - processing not available for this file type]"
            
            doc = {
                'filename': uploaded_file.name,
                'text': text,
                'word_count': len(text.split()) if text else 0,
                'document_type': 'general',
                'upload_time': datetime.now(),
                'file_size': len(text) if text else 0
            }
            documents.append(doc)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            # Add error document
            documents.append({
                'filename': uploaded_file.name,
                'text': '',
                'error': str(e),
                'word_count': 0,
                'upload_time': datetime.now()
            })
    
    # Store in session state
    st.session_state.documents = documents
    return documents

def render_basic_upload_fallback():
    """Basic upload fallback interface"""
    st.markdown("### 📁 Basic File Upload")
    st.info("Using simplified upload process due to module loading issues.")
    
    uploaded_files = st.file_uploader(
        "Choose files (Basic Mode)",
        accept_multiple_files=True,
        type=['txt'],  # Only text files in basic mode
        help="Basic mode supports text files only"
    )
    
    if uploaded_files and st.button("Process Text Files"):
        documents = fallback_process_documents(uploaded_files)
        st.success(f"Processed {len(documents)} files in basic mode")

def render_basic_search_fallback():
    """Basic search fallback interface"""
    st.header("🔍 Basic Search")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first.")
        return
    
    documents = st.session_state.documents
    query = st.text_input("Search documents:", placeholder="Enter search terms...")
    
    if query:
        results = []
        query_lower = query.lower()
        
        for doc in documents:
            text = doc.get('text', '')
            if text and query_lower in text.lower():
                count = text.lower().count(query_lower)
                results.append({
                    'filename': doc['filename'],
                    'matches': count,
                    'word_count': doc.get('word_count', 0)
                })
        
        if results:
            st.success(f"Found {len(results)} matching documents")
            for result in results:
                st.write(f"📄 {result['filename']} - {result['matches']} matches ({result['word_count']} words)")
        else:
            st.warning("No matches found")

def render_basic_alignment_fallback():
    """Basic alignment fallback interface"""
    st.markdown("### 🔗 Basic Recommendation-Response Finder")
    st.info("Using simplified alignment process due to module loading issues.")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first.")
        return
    
    documents = st.session_state.documents
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Find Recommendations**")
        rec_keywords = st.text_input("Recommendation keywords:", value="recommend, suggest, advise")
    
    with col2:
        st.markdown("**↩️ Find Responses**")
        resp_keywords = st.text_input("Response keywords:", value="accept, reject, agree, implement")
    
    if st.button("🔍 Find Recommendations and Responses"):
        rec_words = [word.strip().lower() for word in rec_keywords.split(',')]
        resp_words = [word.strip().lower() for word in resp_keywords.split(',')]
        
        recommendations = []
        responses = []
        
        for doc in documents:
            text = doc.get('text', '').lower()
            filename = doc['filename']
            
            # Find recommendations
            for word in rec_words:
                if word in text:
                    count = text.count(word)
                    recommendations.append({
                        'document': filename,
                        'keyword': word,
                        'count': count
                    })
            
            # Find responses
            for word in resp_words:
                if word in text:
                    count = text.count(word)
                    responses.append({
                        'document': filename,
                        'keyword': word,
                        'count': count
                    })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Recommendations Found:**")
            if recommendations:
                for rec in recommendations:
                    st.write(f"📄 {rec['document']}: '{rec['keyword']}' ({rec['count']}x)")
            else:
                st.info("No recommendations found")
        
        with col2:
            st.markdown("**↩️ Responses Found:**")
            if responses:
                for resp in responses:
                    st.write(f"📄 {resp['document']}: '{resp['keyword']}' ({resp['count']}x)")
            else:
                st.info("No responses found")

def render_basic_analytics_fallback():
    """Basic analytics fallback interface"""
    st.header("📊 Basic Analytics")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 No documents to analyze.")
        return
    
    documents = st.session_state.documents
    
    # Basic statistics
    total_docs = len(documents)
    total_words = sum(doc.get('word_count', 0) for doc in documents)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Average Words", f"{avg_words:,}")
    
    # Document list
    st.markdown("### 📚 Document Details")
    doc_data = []
    for doc in documents:
        doc_data.append({
            'Filename': doc['filename'],
            'Words': doc.get('word_count', 0),
            'Type': doc.get('document_type', 'general').title(),
            'Status': 'Error' if 'error' in doc else 'OK'
        })
    
    df = pd.DataFrame(doc_data)
    st.dataframe(df, use_container_width=True)

def show_alignment_feature_info():
    """Show information about the alignment feature"""
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
    """)

def render_error_recovery():
    """Render error recovery options"""
    st.markdown("---")
    st.markdown("### 🛠️ Error Recovery")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Application"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Application reset. Please refresh the page.")
    
    with col2:
        if st.button("🧪 Load Sample Data"):
            create_sample_data()
    
    with col3:
        if st.button("📋 Show Debug Info"):
            show_debug_info()

def create_sample_data():
    """Create sample data for testing"""
    sample_doc = {
        'filename': 'sample_government_report.txt',
        'text': """
        Sample Government Report - Policy Review

        Executive Summary:
        This report contains several recommendations for improving government services.

        Recommendations:
        1. We recommend implementing new digital services to improve citizen access.
        2. The committee suggests reviewing current budget allocations for healthcare.
        3. We advise establishing a new framework for inter-departmental coordination.

        Government Response:
        1. The department agrees to implement digital services by Q4 2024.
        2. Budget review has been scheduled for the next fiscal year.
        3. The coordination framework proposal will be considered in the upcoming policy review.

        Conclusion:
        This demonstrates the alignment between recommendations and responses in government documentation.
        """,
        'word_count': 95,
        'document_type': 'government',
        'upload_time': datetime.now(),
        'file_size': 756
    }
    
    st.session_state.documents = [sample_doc]
    st.success("✅ Sample data loaded! You can now test the application features.")

def render_recommendations_tab():
    """Render the recommendations extraction tab"""
    st.header("🎯 Extract Recommendations")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first in the Upload tab.")
        st.info("""
        **This feature automatically finds recommendations by detecting:**
        - Sentences starting with action verbs (Improving, Ensuring, etc.)
        - Phrases like "we recommend", "should", "must"
        - Any verb-based suggestions in your documents
        
        **Uses NLTK to automatically identify ALL verbs - no predefined list needed!**
        """)
        return
    
    documents = st.session_state.documents
    doc_names = [doc['filename'] for doc in documents]
    
    # Document selection
    selected_doc = st.selectbox("Select document to analyse:", doc_names)
    
    # Confidence slider
    min_confidence = st.slider(
        "Minimum confidence:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Only show recommendations with confidence above this threshold"
    )
    
    if st.button("🔍 Extract Recommendations", type="primary"):
        doc = next((d for d in documents if d['filename'] == selected_doc), None)
        
        if doc and 'text' in doc:
            with st.spinner("Analysing document..."):
                try:
                    # Extract recommendations
                    recommendations = extract_recommendations_simple(
                        doc['text'],
                        min_confidence=min_confidence
                    )
                    
                    if recommendations:
                        st.success(f"✅ Found {len(recommendations)} recommendations")
                        
                        # Show statistics
                        extractor = SimpleRecommendationExtractor()
                        stats = extractor.get_verb_statistics(recommendations)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Found", stats['total'])
                        with col2:
                            st.metric("Unique Verbs", stats['unique_verbs'])
                        with col3:
                            st.metric("Avg Confidence", f"{stats['avg_confidence']:.0%}")
                        
                        # Display recommendations
                        st.markdown("---")
                        st.subheader("📋 Recommendations Found")
                        
                        for idx, rec in enumerate(recommendations, 1):
                            with st.expander(
                                f"**{idx}. {rec['verb'].upper()}** "
                                f"(Confidence: {rec['confidence']:.0%})"
                            ):
                                st.write(rec['text'])
                                st.caption(f"Detection method: {rec['method']}")
                        
                        # Export options
                        st.markdown("---")
                        st.subheader("💾 Export")
                        
                        # Create CSV data
                        import pandas as pd
                        df = pd.DataFrame(recommendations)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"{selected_doc}_recommendations.csv",
                            mime="text/csv"
                        )
                        
                        # Store in session state for other tabs to use
                        st.session_state.extracted_recommendations = recommendations
                        
                    else:
                        st.warning("No recommendations found. Try lowering the confidence threshold.")
                        
                except Exception as e:
                    st.error(f"Error extracting recommendations: {str(e)}")
                    st.info("Make sure you have installed nltk: pip install nltk")
        else:
            st.error("Document text not available")
            
def show_debug_info():
    """Show debug information"""
    st.markdown("### 🔍 Debug Information")
    
    # Python environment
    import sys
    import platform
    
    st.code(f"""
    Python Version: {sys.version}
    Platform: {platform.platform()}
    Streamlit Version: {st.__version__}
    
    Session State Keys: {list(st.session_state.keys())}
    
    Documents in Session: {'Yes' if 'documents' in st.session_state else 'No'}
    Document Count: {len(st.session_state.get('documents', []))}
    """)

if __name__ == "__main__":
    main()
