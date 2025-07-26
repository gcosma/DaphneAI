# ===============================================
# FILE: modules/ui/extraction_components.py
# Enhanced Content Extraction Components for DaphneAI
# ===============================================

import streamlit as st
import pandas as pd
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced section extractor
try:
    from modules.enhanced_section_extractor import EnhancedSectionExtractor
    ENHANCED_EXTRACTOR_AVAILABLE = True
except ImportError:
    ENHANCED_EXTRACTOR_AVAILABLE = False
    logger.warning("Enhanced Section Extractor not available")

# Import core utilities
try:
    from modules.core_utils import log_user_action
    CORE_UTILS_AVAILABLE = True
except ImportError:
    CORE_UTILS_AVAILABLE = False
    def log_user_action(action: str, data: Dict = None):
        logger.info(f"Action: {action}, Data: {data}")

# ===============================================
# ENHANCED EXTRACTION CLASSES
# ===============================================

class SmartExtractor:
    """Smart content extractor with multiple strategies"""
    
    def __init__(self):
        self.enhanced_extractor = None
        if ENHANCED_EXTRACTOR_AVAILABLE:
            try:
                self.enhanced_extractor = EnhancedSectionExtractor()
                logger.info("✅ Enhanced Section Extractor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced Section Extractor: {e}")
    
    def extract_recommendations(self, content: str, document_name: str = "") -> List[Dict]:
        """Extract recommendations using multiple strategies"""
        recommendations = []
        
        # Strategy 1: Use enhanced extractor if available
        if self.enhanced_extractor:
            try:
                enhanced_results = self.enhanced_extractor.extract_sections(content)
                if enhanced_results.get('recommendations'):
                    for rec in enhanced_results['recommendations']:
                        recommendations.append({
                            'text': rec.get('text', ''),
                            'section': rec.get('section', ''),
                            'confidence': rec.get('confidence', 0.8),
                            'method': 'enhanced_extractor',
                            'document': document_name,
                            'extracted_at': datetime.now().isoformat()
                        })
            except Exception as e:
                logger.error(f"Enhanced extractor failed: {e}")
        
        # Strategy 2: Pattern-based extraction
        pattern_recommendations = self._extract_by_patterns(content, document_name)
        recommendations.extend(pattern_recommendations)
        
        # Strategy 3: Section-based extraction
        section_recommendations = self._extract_by_sections(content, document_name)
        recommendations.extend(section_recommendations)
        
        # Remove duplicates and rank by confidence
        recommendations = self._deduplicate_recommendations(recommendations)
        
        return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    
    def extract_responses(self, content: str, document_name: str = "") -> List[Dict]:
        """Extract government responses using multiple strategies"""
        responses = []
        
        # Strategy 1: Use enhanced extractor if available
        if self.enhanced_extractor:
            try:
                enhanced_results = self.enhanced_extractor.extract_sections(content)
                if enhanced_results.get('responses'):
                    for resp in enhanced_results['responses']:
                        responses.append({
                            'text': resp.get('text', ''),
                            'section': resp.get('section', ''),
                            'confidence': resp.get('confidence', 0.8),
                            'method': 'enhanced_extractor',
                            'document': document_name,
                            'extracted_at': datetime.now().isoformat()
                        })
            except Exception as e:
                logger.error(f"Enhanced extractor failed for responses: {e}")
        
        # Strategy 2: Pattern-based response extraction
        pattern_responses = self._extract_responses_by_patterns(content, document_name)
        responses.extend(pattern_responses)
        
        # Remove duplicates and rank by confidence
        responses = self._deduplicate_recommendations(responses)
        
        return sorted(responses, key=lambda x: x['confidence'], reverse=True)
    
    def _extract_by_patterns(self, content: str, document_name: str) -> List[Dict]:
        """Extract recommendations using regex patterns"""
        recommendations = []
        
        # Common recommendation patterns
        patterns = [
            r'(?:recommend|suggests?|proposes?)[s]?\s+that\s+([^.]{50,300}\.)',
            r'(?:we|the committee|the inquiry)\s+recommend[s]?\s+([^.]{50,300}\.)',
            r'recommendation\s+\d+[:\-]\s*([^.]{50,500}\.)',
            r'it is recommended that\s+([^.]{50,300}\.)',
            r'should\s+(?:be\s+)?(?:consider|implement|establish|ensure)\s+([^.]{50,300}\.)'
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                recommendation_text = match.group(1).strip()
                if len(recommendation_text) > 30:  # Filter very short matches
                    recommendations.append({
                        'text': recommendation_text,
                        'section': f'Pattern {i+1}',
                        'confidence': 0.6 + (i * 0.05),  # Higher confidence for earlier patterns
                        'method': 'pattern_matching',
                        'document': document_name,
                        'extracted_at': datetime.now().isoformat()
                    })
        
        return recommendations
    
    def _extract_by_sections(self, content: str, document_name: str) -> List[Dict]:
        """Extract recommendations by identifying relevant sections"""
        recommendations = []
        
        # Split content into sections
        sections = re.split(r'\n\s*(?:chapter|section|part)\s+\d+', content, flags=re.IGNORECASE)
        
        for i, section in enumerate(sections):
            # Look for recommendation-heavy sections
            rec_count = len(re.findall(r'recommend', section, re.IGNORECASE))
            
            if rec_count >= 2:  # Section has multiple recommendation mentions
                # Extract sentences that likely contain recommendations
                sentences = re.split(r'[.!?]+', section)
                for sentence in sentences:
                    if re.search(r'recommend|should|must|ought|propose', sentence, re.IGNORECASE):
                        if 50 <= len(sentence.strip()) <= 400:
                            recommendations.append({
                                'text': sentence.strip(),
                                'section': f'Section {i+1}',
                                'confidence': 0.5,
                                'method': 'section_analysis',
                                'document': document_name,
                                'extracted_at': datetime.now().isoformat()
                            })
        
        return recommendations
    
    def _extract_responses_by_patterns(self, content: str, document_name: str) -> List[Dict]:
        """Extract government responses using patterns"""
        responses = []
        
        # Response patterns
        patterns = [
            r'(?:government|minister|department)\s+(?:accepts?|agrees?|acknowledges?)\s+([^.]{50,300}\.)',
            r'(?:accepted|agreed|implemented)\s*[:\-]\s*([^.]{50,400}\.)',
            r'(?:the government|we)\s+(?:will|shall|have)\s+([^.]{50,300}\.)',
            r'response[:\-]\s*([^.]{50,400}\.)',
            r'(?:action|implementation)[:\-]\s*([^.]{50,400}\.)'
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                response_text = match.group(1).strip()
                if len(response_text) > 30:
                    responses.append({
                        'text': response_text,
                        'section': f'Response Pattern {i+1}',
                        'confidence': 0.6 + (i * 0.05),
                        'method': 'response_pattern_matching',
                        'document': document_name,
                        'extracted_at': datetime.now().isoformat()
                    })
        
        return responses
    
    def _deduplicate_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Remove duplicate recommendations based on text similarity"""
        unique_recommendations = []
        seen_texts = set()
        
        for rec in recommendations:
            # Simple deduplication based on first 100 characters
            text_key = rec['text'][:100].lower().strip()
            
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_recommendations.append(rec)
        
        return unique_recommendations

# ===============================================
# EXTRACTION UTILITY FUNCTIONS
# ===============================================

def validate_documents_for_extraction() -> Tuple[bool, str]:
    """Validate that documents are available for extraction"""
    if 'uploaded_documents' not in st.session_state or not st.session_state.uploaded_documents:
        return False, "No documents uploaded. Please upload documents first."
    
    return True, "Documents available for extraction"

def get_document_content_for_extraction(selected_docs: List[str]) -> Dict[str, str]:
    """Get content from selected documents"""
    content_map = {}
    
    all_docs = st.session_state.get('uploaded_documents', [])
    
    for doc in all_docs:
        if doc['filename'] in selected_docs:
            content_map[doc['filename']] = doc['content']
    
    return content_map

# ===============================================
# MAIN EXTRACTION TAB COMPONENT
# ===============================================

def render_extraction_tab():
    """Render the enhanced content extraction interface"""
    st.header("🔍 Enhanced Content Extraction")
    st.markdown("""
    Extract recommendations, responses, and key content from uploaded documents using AI-powered analysis.
    """)
    
    # Validate documents
    docs_valid, validation_message = validate_documents_for_extraction()
    
    if not docs_valid:
        st.warning(validation_message)
        st.info("👆 Please upload documents in the Upload tab first.")
        return
    
    # Document selection
    st.markdown("### 📄 Select Documents for Extraction")
    
    all_docs = st.session_state.uploaded_documents
    doc_options = [doc['filename'] for doc in all_docs]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_docs = st.multiselect(
            "Choose documents to analyze:",
            doc_options,
            default=doc_options,  # Select all by default
            help="Select one or more documents for content extraction"
        )
    
    with col2:
        st.markdown("#### Quick Actions")
        if st.button("🔄 Select All"):
            selected_docs = doc_options
            st.rerun()
        
        if st.button("❌ Clear Selection"):
            selected_docs = []
            st.rerun()
    
    if not selected_docs:
        st.warning("Please select at least one document for extraction.")
        return
    
    # Extraction configuration
    st.markdown("### ⚙️ Extraction Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        extract_recommendations = st.checkbox("Extract Recommendations", value=True)
    
    with col2:
        extract_responses = st.checkbox("Extract Government Responses", value=True)
    
    with col3:
        use_enhanced_ai = st.checkbox("Use Enhanced AI", value=ENHANCED_EXTRACTOR_AVAILABLE)
        if not ENHANCED_EXTRACTOR_AVAILABLE:
            st.caption("⚠️ Enhanced AI not available")
    
    # Extraction button and processing
    if st.button("🚀 Start Extraction", type="primary"):
        if not extract_recommendations and not extract_responses:
            st.error("Please select at least one extraction type.")
            return
        
        # Initialize smart extractor
        smart_extractor = SmartExtractor()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_recommendations = []
        all_responses = []
        extraction_stats = {
            'documents_processed': 0,
            'recommendations_found': 0,
            'responses_found': 0,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        # Get document content
        content_map = get_document_content_for_extraction(selected_docs)
        
        # Process each document
        for i, (filename, content) in enumerate(content_map.items()):
            status_text.text(f"Processing: {filename}")
            progress = (i + 1) / len(content_map)
            progress_bar.progress(progress)
            
            try:
                # Extract recommendations
                if extract_recommendations:
                    doc_recommendations = smart_extractor.extract_recommendations(content, filename)
                    all_recommendations.extend(doc_recommendations)
                    extraction_stats['recommendations_found'] += len(doc_recommendations)
                
                # Extract responses
                if extract_responses:
                    doc_responses = smart_extractor.extract_responses(content, filename)
                    all_responses.extend(doc_responses)
                    extraction_stats['responses_found'] += len(doc_responses)
                
                extraction_stats['documents_processed'] += 1
                
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
                logger.error(f"Extraction error for {filename}: {e}")
        
        # Calculate processing time
        end_time = datetime.now()
        extraction_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Update session state
        if extract_recommendations:
            st.session_state.extracted_recommendations = all_recommendations
        
        if extract_responses:
            st.session_state.extracted_responses = all_responses
        
        st.session_state.extraction_results = {
            'recommendations': all_recommendations,
            'responses': all_responses,
            'stats': extraction_stats,
            'timestamp': end_time.isoformat()
        }
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("✅ Extraction completed!")
        
        # Show results summary
        show_extraction_summary(extraction_stats, all_recommendations, all_responses)
        
        # Log extraction action
        if CORE_UTILS_AVAILABLE:
            log_user_action("content_extraction", {
                'documents': selected_docs,
                'recommendations_found': len(all_recommendations),
                'responses_found': len(all_responses),
                'processing_time': extraction_stats['processing_time']
            })
    
    # Show existing results if available
    if st.session_state.get('extraction_results'):
        st.markdown("---")
        st.markdown("### 📊 Latest Extraction Results")
        
        results = st.session_state.extraction_results
        show_extraction_results(results)

def show_extraction_summary(stats: Dict, recommendations: List[Dict], responses: List[Dict]):
    """Show summary of extraction results"""
    st.markdown("### 🎉 Extraction Complete!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Processed", stats['documents_processed'])
    
    with col2:
        st.metric("Recommendations", stats['recommendations_found'])
    
    with col3:
        st.metric("Responses", stats['responses_found'])
    
    with col4:
        st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
    
    # Quality indicators
    if recommendations:
        avg_confidence = sum(r['confidence'] for r in recommendations) / len(recommendations)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    st.success("✅ Content extraction completed successfully!")

def show_extraction_results(results: Dict):
    """Display detailed extraction results"""
    recommendations = results.get('recommendations', [])
    responses = results.get('responses', [])
    stats = results.get('stats', {})
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "💬 Responses", "📊 Analysis"])
    
    with tab1:
        if recommendations:
            st.markdown(f"### Found {len(recommendations)} Recommendations")
            
            # Display recommendations
            for i, rec in enumerate(recommendations[:20]):  # Show first 20
                with st.expander(f"Recommendation {i+1} (Confidence: {rec['confidence']:.2f})"):
                    st.write(f"**Text:** {rec['text']}")
                    st.write(f"**Document:** {rec['document']}")
                    st.write(f"**Method:** {rec['method']}")
                    st.write(f"**Section:** {rec['section']}")
            
            if len(recommendations) > 20:
                st.info(f"Showing first 20 of {len(recommendations)} recommendations")
            
            # Export recommendations
            if st.button("📥 Export Recommendations"):
                export_extraction_results(recommendations, "recommendations")
        
        else:
            st.info("No recommendations found in the selected documents.")
    
    with tab2:
        if responses:
            st.markdown(f"### Found {len(responses)} Responses")
            
            # Display responses
            for i, resp in enumerate(responses[:20]):  # Show first 20
                with st.expander(f"Response {i+1} (Confidence: {resp['confidence']:.2f})"):
                    st.write(f"**Text:** {resp['text']}")
                    st.write(f"**Document:** {resp['document']}")
                    st.write(f"**Method:** {resp['method']}")
                    st.write(f"**Section:** {resp['section']}")
            
            if len(responses) > 20:
                st.info(f"Showing first 20 of {len(responses)} responses")
            
            # Export responses
            if st.button("📥 Export Responses"):
                export_extraction_results(responses, "responses")
        
        else:
            st.info("No responses found in the selected documents.")
    
    with tab3:
        st.markdown("### 📈 Extraction Analysis")
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Document Coverage")
            if recommendations:
                doc_coverage = {}
                for rec in recommendations:
                    doc = rec['document']
                    doc_coverage[doc] = doc_coverage.get(doc, 0) + 1
                
                coverage_df = pd.DataFrame([
                    {'Document': doc, 'Recommendations': count}
                    for doc, count in doc_coverage.items()
                ])
                st.dataframe(coverage_df)
        
        with col2:
            st.markdown("#### Method Effectiveness")
            if recommendations:
                method_stats = {}
                for rec in recommendations:
                    method = rec['method']
                    method_stats[method] = method_stats.get(method, 0) + 1
                
                methods_df = pd.DataFrame([
                    {'Method': method, 'Count': count}
                    for method, count in method_stats.items()
                ])
                st.dataframe(methods_df)
        
        # Confidence distribution
        if recommendations:
            st.markdown("#### Confidence Distribution")
            confidences = [rec['confidence'] for rec in recommendations]
            
            # Simple confidence breakdown
            high_conf = len([c for c in confidences if c >= 0.8])
            med_conf = len([c for c in confidences if 0.6 <= c < 0.8])
            low_conf = len([c for c in confidences if c < 0.6])
            
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            with conf_col1:
                st.metric("High Confidence (≥0.8)", high_conf)
            with conf_col2:
                st.metric("Medium Confidence (0.6-0.8)", med_conf)
            with conf_col3:
                st.metric("Low Confidence (<0.6)", low_conf)

def export_extraction_results(results: List[Dict], result_type: str):
    """Export extraction results to CSV"""
    try:
        # Convert to DataFrame
        export_data = []
        for item in results:
            export_data.append({
                'Text': item['text'],
                'Document': item['document'],
                'Section': item['section'],
                'Confidence': item['confidence'],
                'Method': item['method'],
                'Extracted_At': item['extracted_at']
            })
        
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        
        # Download button
        filename = f"{result_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button(
            label=f"📥 Download {result_type.title()} CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        st.success(f"✅ {result_type.title()} ready for download")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

# ===============================================
# INITIALIZATION
# ===============================================

# Initialize session state for extraction
if 'extracted_recommendations' not in st.session_state:
    st.session_state.extracted_recommendations = []

if 'extracted_responses' not in st.session_state:
    st.session_state.extracted_responses = []

if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = {}

logger.info("✅ Enhanced extraction components initialized")
