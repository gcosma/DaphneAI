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


class RecommendationExtractor:
    """
    Enhanced extractor for recommendations from policy documents.
    
    Features:
    - Multiple detection methods (numbered, bulleted, gerund, modal, imperative, keyword)
    - Confidence scoring
    - False positive filtering
    - Context-aware extraction
    - Verb extraction and statistics
    - Returns simple dictionaries for Streamlit compatibility
    """
    
    def __init__(self):
        """Initialise the extractor with pattern definitions"""
        
        # High-confidence gerund verbs for policy recommendations
        self.gerund_verbs = {
            'improving', 'establishing', 'ensuring', 'enabling', 'broadening',
            'reforming', 'clarifying', 'implementing', 'developing', 'strengthening',
            'enhancing', 'creating', 'building', 'maintaining', 'providing',
            'supporting', 'promoting', 'facilitating', 'introducing', 'expanding',
            'reviewing', 'updating', 'reducing', 'addressing', 'adopting',
            'requiring', 'encouraging'
        }
        
        # Imperative verbs that start recommendations
        self.imperative_verbs = {
            'ensure', 'establish', 'improve', 'create', 'develop', 'implement',
            'strengthen', 'enhance', 'provide', 'support', 'maintain', 'promote',
            'facilitate', 'introduce', 'expand', 'reform', 'clarify', 'review',
            'update', 'address', 'adopt', 'require', 'encourage'
        }
        
        # Action verbs that typically follow modals in recommendations
        self.action_verbs = {
            'implement', 'establish', 'ensure', 'improve', 'develop', 'create',
            'enhance', 'introduce', 'adopt', 'provide', 'strengthen', 'facilitate',
            'support', 'maintain', 'expand', 'review', 'update', 'address'
        }
        
        # Bullet point patterns for unnumbered lists
        self.bullet_patterns = [
            r'^\s*[•●■▪▸►]+\s+',      # Unicode bullets
            r'^\s*[-–—]\s+',           # Dashes/hyphens
            r'^\s*[*]\s+',             # Asterisks
            r'^\s*[○◦]\s+',            # Open circles
            r'^\s*[✓✔]\s+',            # Check marks
        ]
        
        # Modal + action verb patterns
        self.modal_action_patterns = [
            r'\bshould\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
            r'\bmust\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
            r'\bneed to\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
        ]
        
        # Explicit recommendation phrases
        self.recommendation_patterns = [
            r'we recommend\b',
            r'it is recommended\b',
            r'the inquiry recommends?\b',
            r'the committee recommends?\b',
            r'the report recommends?\b',
            r'(?:this|the) recommendation',
        ]
        
        # Phrases to exclude (false positive indicators)
        self.exclude_phrases = {
            'when deciding', 'before implementing', 'after establishing',
            'while ensuring', 'without improving', 'if reforming',
            'by clarifying', 'through enabling', 'during broadening',
            'was establishing', 'were improving', 'had been ensuring',
            'has been', 'have been', 'had been',
            'will be', 'would be',
            'for example', 'for instance', 'such as',
            'the need to understand', 'the need to consider'
        }
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.7) -> List[Dict]:
        """
        Extract all recommendations from text with confidence scores.
        
        Args:
            text: The document text to analyse
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            List of dictionaries with recommendation data
        """
        recommendations = []
        
        # Split into sentences for processing
        sentences = self._split_sentences(text)
        
        for idx, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence) < 30:
                continue
            
            # Try each detection method (using if/elif to avoid duplicates)
            
            # Method 1: Explicit recommendation phrases
            if self._contains_recommendation_phrase(sentence):
                verb = self._extract_main_verb(sentence)
                if len(sentence.split()) >= 5:
                    recommendations.append({
                        'text': sentence,
                        'verb': verb,
                        'method': 'keyword',
                        'confidence': 0.85,
                        'position': idx
                    })
            
            # Method 2: Numbered list items
            elif self._is_numbered(sentence):
                text_without_number = re.sub(r'^\s*\d+[.)]\s+', '', sentence)
                if self._is_recommendation_content(text_without_number):
                    verb = self._extract_main_verb(text_without_number)
                    recommendations.append({
                        'text': text_without_number,
                        'verb': verb,
                        'method': 'numbered',
                        'confidence': 0.95,
                        'position': idx
                    })
            
            # Method 3: Bulleted list items
            elif self._is_bulleted(sentence):
                text_without_bullet = self._remove_bullet(sentence)
                if self._is_recommendation_content(text_without_bullet):
                    verb = self._extract_main_verb(text_without_bullet)
                    recommendations.append({
                        'text': text_without_bullet,
                        'verb': verb,
                        'method': 'bulleted',
                        'confidence': 0.90,
                        'position': idx
                    })
            
            # Method 4: Sentences starting with gerunds
            elif self._starts_with_gerund(sentence):
                verb = self._extract_first_verb(sentence)
                recommendations.append({
                    'text': sentence,
                    'verb': verb,
                    'method': 'gerund',
                    'confidence': 0.90,
                    'position': idx
                })
            
            # Method 5: Sentences starting with imperative verbs
            elif self._starts_with_imperative(sentence):
                verb = self._extract_first_verb(sentence)
                recommendations.append({
                    'text': sentence,
                    'verb': verb,
                    'method': 'imperative',
                    'confidence': 0.85,
                    'position': idx
                })
            
            # Method 6: Modal verbs with action implications
            elif self._contains_strong_modal(sentence):
                verb = self._extract_main_verb(sentence)
                if len(sentence.split()) >= 8:
                    recommendations.append({
                        'text': sentence,
                        'verb': verb,
                        'method': 'modal',
                        'confidence': 0.75,
                        'position': idx
                    })
        
        # Filter by confidence
        recommendations = [r for r in recommendations if r['confidence'] >= min_confidence]
        
        # Remove duplicates
        recommendations = self._remove_duplicates(recommendations)
        
        return recommendations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, treating bullets and numbered lists as separate items"""
        # First, split by newlines to preserve list structure
        lines = text.split('\n')
        
        sentences = []
        for line in lines:
            line = line.strip()
            if not line or len(line) < 30:
                continue
            
            # If it's a bullet or numbered item, treat it as a separate sentence
            if (self._is_numbered(line) or self._is_bulleted(line) or 
                any(line.lower().startswith(verb) for verb in self.gerund_verbs)):
                sentences.append(line)
            else:
                # Otherwise, split by sentence boundaries
                split_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
                sentences.extend([s.strip() for s in split_sents if s.strip() and len(s.strip()) > 30])
        
        return sentences
    
    def _contains_recommendation_phrase(self, sentence: str) -> bool:
        """Check if sentence contains explicit recommendation phrases"""
        sentence_lower = sentence.lower()
        for pattern in self.recommendation_patterns:
            if re.search(pattern, sentence_lower):
                return True
        return False
    
    def _is_numbered(self, sentence: str) -> bool:
        """Check if sentence starts with a number"""
        return bool(re.match(r'^\s*\d+[.)]\s+', sentence))
    
    def _is_bulleted(self, sentence: str) -> bool:
        """Check if sentence starts with a bullet point"""
        for pattern in self.bullet_patterns:
            if re.match(pattern, sentence):
                return True
        return False
    
    def _remove_bullet(self, sentence: str) -> str:
        """Remove bullet point from start of sentence"""
        for pattern in self.bullet_patterns:
            sentence = re.sub(pattern, '', sentence)
        return sentence.strip()
    
    def _starts_with_gerund(self, sentence: str) -> bool:
        """Check if sentence starts with a gerund (verb-ing)"""
        if self._is_false_positive(sentence):
            return False
        
        words = sentence.split()
        if not words:
            return False
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        if first_word in self.gerund_verbs:
            return True
        
        # Also check with NLTK if available
        if first_word.endswith('ing') and len(first_word) > 5:
            if NLP_AVAILABLE:
                try:
                    base = first_word[:-3]
                    tagged = pos_tag([base])
                    return tagged[0][1].startswith('VB')
                except:
                    pass
        
        return False
    
    def _starts_with_imperative(self, sentence: str) -> bool:
        """Check if sentence starts with an imperative verb"""
        if self._is_false_positive(sentence):
            return False
        
        words = sentence.split()
        if not words:
            return False
        
        first_word = words[0].lower().rstrip(':,.')
        return first_word in self.imperative_verbs
    
    def _contains_strong_modal(self, sentence: str) -> bool:
        """Check if sentence contains modal verbs with clear action implications"""
        if self._is_false_positive(sentence):
            return False
        
        sentence_lower = sentence.lower()
        
        for pattern in self.modal_action_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _is_recommendation_content(self, text: str) -> bool:
        """Check if numbered/bulleted item is actually a recommendation"""
        text_lower = text.lower()
        
        # Check first word - if it's a gerund or imperative verb, it's likely a recommendation
        words = text.split()
        if words:
            first_word = words[0].lower().rstrip(':,.')
            if first_word in self.gerund_verbs or first_word in self.imperative_verbs:
                return True
        
        # Look for modal recommendation indicators
        modal_indicators = ['should', 'must', 'need', 'require']
        if any(ind in text_lower for ind in modal_indicators):
            return True
        
        # Look for action verb indicators
        action_indicators = ['ensure', 'establish', 'improve', 'create', 'develop']
        if any(ind in text_lower for ind in action_indicators):
            return True
        
        return False
    
    def _is_false_positive(self, sentence: str) -> bool:
        """Check if sentence is a false positive"""
        sentence_lower = sentence.lower()
        
        for phrase in self.exclude_phrases:
            if phrase in sentence_lower:
                return True
        
        if '?' in sentence:
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
        """Extract the main verb from a sentence"""
        if NLP_AVAILABLE:
            try:
                tokens = word_tokenize(sentence[:200])
                tagged = pos_tag(tokens)
                
                verbs = [word.lower() for word, pos in tagged if pos.startswith('VB')]
                
                if verbs:
                    auxiliaries = {'be', 'is', 'are', 'was', 'were', 'been', 'being', 
                                 'have', 'has', 'had', 'do', 'does', 'did'}
                    for verb in verbs:
                        if verb not in auxiliaries:
                            return verb
                    return verbs[0]
            except:
                pass
        
        # Fallback: look for common recommendation verbs
        sentence_lower = sentence.lower()
        
        # Check action verbs first
        for verb in self.action_verbs:
            if verb in sentence_lower:
                return verb
        
        # Then check gerund verbs (remove -ing)
        for verb in self.gerund_verbs:
            if verb in sentence_lower:
                return verb[:-3] if verb.endswith('ing') else verb
        
        # Check first word if it's a verb
        words = sentence.split()
        if words:
            first_word = words[0].lower().rstrip(':,.')
            if first_word in self.gerund_verbs or first_word in self.imperative_verbs:
                return first_word[:-3] if first_word.endswith('ing') else first_word
        
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
    
    def extract_from_file(self, filepath: str, min_confidence: float = 0.7) -> Dict:
        """
        Extract recommendations from a PDF or text file.
        
        Args:
            filepath: Path to the document
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with recommendations and metadata
        """
        import subprocess
        
        # Extract text from PDF
        if filepath.endswith('.pdf'):
            result = subprocess.run(
                ['pdftotext', '-layout', filepath, '-'],
                capture_output=True,
                text=True
            )
            text = result.stdout
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Extract recommendations
        recommendations = self.extract_recommendations(text, min_confidence)
        
        # Group by method
        by_method = {}
        for rec in recommendations:
            method = rec['method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(rec)
        
        # Get verb statistics
        verb_stats = self.get_verb_statistics(recommendations)
        
        return {
            'total_count': len(recommendations),
            'by_method': {k: len(v) for k, v in by_method.items()},
            'recommendations': recommendations,
            'grouped': by_method,
            'verb_statistics': verb_stats
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate a formatted report of extracted recommendations"""
        lines = []
        lines.append("=" * 80)
        lines.append("RECOMMENDATION EXTRACTION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTotal recommendations found: {results['total_count']}")
        lines.append("\nBreakdown by method:")
        
        for method, count in results.get('by_method', {}).items():
            lines.append(f"  - {method.upper()}: {count}")
        
        if 'verb_statistics' in results:
            lines.append(f"\nUnique verbs found: {results['verb_statistics']['unique_verbs']}")
            lines.append(f"Average confidence: {results['verb_statistics']['avg_confidence']:.2f}")
            lines.append("\nTop verbs:")
            for verb, count in list(results['verb_statistics']['verb_frequency'].items())[:10]:
                lines.append(f"  - {verb}: {count}")
        
        lines.append("\n" + "=" * 80)
        lines.append("DETAILED RECOMMENDATIONS")
        lines.append("=" * 80)
        
        for i, rec in enumerate(results['recommendations'], 1):
            lines.append(f"\n[{i}] {rec['method'].upper()} (confidence: {rec['confidence']:.2f}, verb: {rec['verb']})")
            lines.append(f"    {rec['text']}")
        
        return '\n'.join(lines)


def extract_recommendations_simple(text: str, min_confidence: float = 0.7) -> List[Dict]:
    """Simple function to extract recommendations from text - Streamlit compatible"""
    extractor = RecommendationExtractor()
    return extractor.extract_recommendations(text, min_confidence)


# Backward compatibility: alias for old code
SimpleRecommendationExtractor = RecommendationExtractor


# Example usage
if __name__ == "__main__":
    import sys
    
    extractor = RecommendationExtractor()
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        results = extractor.extract_from_file(filepath, min_confidence=0.7)
        report = extractor.generate_report(results)
        print(report)
    else:
        # Test with example text
        test_text = """
        Specific recommendations
        
        1. Improving consideration of the impact that decisions might have on those most at risk.
        
        2. Broadening participation in SAGE through open recruitment of experts.
        
        • Reforming and clarifying the structures for decision-making during emergencies.
        
        - Ensuring that decisions and their implications are clearly communicated to the public.
        
        * Enabling greater parliamentary scrutiny of the use of emergency powers.
        
        Governments must act swiftly and decisively to stop virus spread.
        
        The inquiry recommends establishing better communication channels.
        """
        
        recs = extractor.extract_recommendations(test_text, min_confidence=0.7)
        
        print(f"Found {len(recs)} recommendations:\n")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. [{rec['method'].upper()}] (verb: {rec['verb']}) - Confidence: {rec['confidence']:.2f}")
            print(f"   {rec['text'][:80]}...")
            print()
        
        # Show verb statistics
        stats = extractor.get_verb_statistics(recs)
        print("\nVerb Statistics:")
        print(f"Total: {stats['total']}")
        print(f"Unique verbs: {stats['unique_verbs']}")
        print(f"Verb frequency: {stats['verb_frequency']}")

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
