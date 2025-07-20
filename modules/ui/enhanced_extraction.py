# ===============================================
# UNIFIED BEST-OF-ALL EXTRACTION SYSTEM
# ===============================================

import re
import logging
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class UnifiedConcernExtractor:
    """
    Unified extraction system that combines the best of all methods:
    1. PDFxtract (primary) - for robust concern extraction
    2. Enhanced patterns (fallback) - for additional coverage
    3. Smart selection - automatically picks the best result
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Import PDFxtract function
        try:
            from core_utils import extract_concern_text
            self.extract_concern_text = extract_concern_text
            self.pdfxtract_available = True
        except ImportError:
            self.logger.error("Cannot import extract_concern_text from core_utils")
            self.pdfxtract_available = False
        
        # Keep enhanced patterns as fallback/supplement
        self.enhanced_patterns = {
            'high_confidence': [
                # Coroner-specific patterns (high confidence)
                r"CORONER'S\s+CONCERNS?:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|$)",
                r"MATTERS?\s+OF\s+CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|$)",
                r"The\s+MATTERS?\s+OF\s+CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|$)",
            ],
            'medium_confidence': [
                # Issue identification patterns
                r'(?:identified|raised|noted)\s+(?:a\s+)?(?:concern|issue|problem)(?:s)?[\s:]+([^.]+\.)',
                r'(?:concern|issue|problem)\s+(?:that|about|regarding|with)[\s:]+([^.]+\.)',
                r'(?:failure|inadequate|insufficient|lack\s+of)[\s:]+([^.]+\.)',
            ],
            'supplementary': [
                # Additional coverage patterns
                r'(?:matter|issue|problem)(?:s)?[\s:]+([^.]+(?:\.[^.]*){0,2}\.)',
                r'(?:prevention|recommendation)[\s:]+([^.]+\.)',
                r'(?:should\s+have|ought\s+to\s+have)[\s:]+([^.]+\.)'
            ]
        }
    
    def extract_concerns_unified(self, content: str, document_name: str = "") -> Dict:
        """
        Unified extraction that automatically selects the best method and result
        
        Process:
        1. Try PDFxtract (most reliable for coroner documents)
        2. If PDFxtract succeeds and gives good result, use it
        3. Otherwise, try enhanced patterns
        4. Return the best result with confidence scoring
        """
        if not content or len(content.strip()) < 10:
            return {
                'concerns': [],
                'extraction_summary': {
                    'method_used': 'none',
                    'reason': 'No content or content too short',
                    'content_length': len(content) if content else 0,
                    'success': False
                }
            }
        
        extraction_attempts = []
        final_concerns = []
        
        # ATTEMPT 1: PDFxtract (Primary Method)
        if self.pdfxtract_available:
            pdfxtract_result = self._try_pdfxtract_extraction(content, document_name)
            extraction_attempts.append(pdfxtract_result)
            
            # If PDFxtract gives a good result, use it
            if (pdfxtract_result['success'] and 
                pdfxtract_result['concerns'] and 
                len(pdfxtract_result['concerns'][0]['text']) > 50):
                
                final_concerns = pdfxtract_result['concerns']
                method_used = 'pdfxtract'
                confidence = 0.95
            
        # ATTEMPT 2: Enhanced Patterns (if PDFxtract failed or gave poor result)
        if not final_concerns:
            enhanced_result = self._try_enhanced_patterns(content, document_name)
            extraction_attempts.append(enhanced_result)
            
            if enhanced_result['success'] and enhanced_result['concerns']:
                final_concerns = enhanced_result['concerns']
                method_used = 'enhanced_patterns'
                confidence = 0.8
        
        # ATTEMPT 3: Smart Hybrid (combine if both methods found something)
        if len(extraction_attempts) >= 2:
            hybrid_result = self._create_hybrid_result(extraction_attempts, document_name)
            if hybrid_result['concerns'] and len(hybrid_result['concerns']) > len(final_concerns):
                final_concerns = hybrid_result['concerns']
                method_used = 'hybrid'
                confidence = 0.9
        
        # If still no results, try basic text extraction
        if not final_concerns:
            basic_result = self._try_basic_extraction(content, document_name)
            if basic_result['concerns']:
                final_concerns = basic_result['concerns']
                method_used = 'basic_fallback'
                confidence = 0.6
        
        # Prepare final result
        extraction_summary = {
            'method_used': method_used if final_concerns else 'none',
            'total_attempts': len(extraction_attempts),
            'success': len(final_concerns) > 0,
            'confidence': confidence if final_concerns else 0.0,
            'content_length': len(content),
            'concerns_found': len(final_concerns),
            'attempts_detail': [attempt['method'] for attempt in extraction_attempts]
        }
        
        # Add metadata to concerns
        for concern in final_concerns:
            concern['document_source'] = document_name
            concern['extraction_confidence'] = confidence
            concern['extraction_method'] = method_used
        
        return {
            'concerns': final_concerns,
            'extraction_summary': extraction_summary
        }
    
    def _try_pdfxtract_extraction(self, content: str, document_name: str) -> Dict:
        """Try PDFxtract extraction method"""
        try:
            extracted_text = self.extract_concern_text(content)
            
            if extracted_text and len(extracted_text.strip()) > 20:
                concern = {
                    'id': f"pdfxtract_{hash(extracted_text) % 10000}",
                    'text': extracted_text.strip(),
                    'method': 'pdfxtract',
                    'type': 'coroner_concern',
                    'confidence_score': 0.95,
                    'extracted_at': datetime.now().isoformat(),
                    'word_count': len(extracted_text.split()),
                    'character_count': len(extracted_text)
                }
                
                return {
                    'method': 'pdfxtract',
                    'success': True,
                    'concerns': [concern],
                    'text_length': len(extracted_text),
                    'quality_score': self._assess_text_quality(extracted_text)
                }
            else:
                return {
                    'method': 'pdfxtract',
                    'success': False,
                    'concerns': [],
                    'reason': 'No text extracted or too short'
                }
                
        except Exception as e:
            return {
                'method': 'pdfxtract',
                'success': False,
                'concerns': [],
                'error': str(e)
            }
    
    def _try_enhanced_patterns(self, content: str, document_name: str) -> Dict:
        """Try enhanced pattern extraction"""
        try:
            all_concerns = []
            
            # Normalize content
            normalized_content = self._normalize_content(content)
            
            # Try high confidence patterns first
            for pattern in self.enhanced_patterns['high_confidence']:
                matches = re.finditer(pattern, normalized_content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    text = match.group(1).strip()
                    if len(text) > 30:
                        all_concerns.append({
                            'id': f"enhanced_high_{len(all_concerns)}",
                            'text': text,
                            'method': 'enhanced_high_confidence',
                            'type': 'coroner_concern',
                            'confidence_score': 0.9,
                            'extracted_at': datetime.now().isoformat(),
                            'pattern_type': 'high_confidence'
                        })
            
            # If high confidence didn't work, try medium confidence
            if not all_concerns:
                for pattern in self.enhanced_patterns['medium_confidence']:
                    matches = re.finditer(pattern, normalized_content, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        text = match.group(1).strip()
                        if len(text) > 20:
                            all_concerns.append({
                                'id': f"enhanced_med_{len(all_concerns)}",
                                'text': text,
                                'method': 'enhanced_medium_confidence',
                                'type': 'identified_concern',
                                'confidence_score': 0.7,
                                'extracted_at': datetime.now().isoformat(),
                                'pattern_type': 'medium_confidence'
                            })
            
            # Deduplicate
            unique_concerns = self._deduplicate_concerns(all_concerns)
            
            return {
                'method': 'enhanced_patterns',
                'success': len(unique_concerns) > 0,
                'concerns': unique_concerns,
                'patterns_tried': len(self.enhanced_patterns['high_confidence']) + len(self.enhanced_patterns['medium_confidence']),
                'quality_score': self._assess_concerns_quality(unique_concerns)
            }
            
        except Exception as e:
            return {
                'method': 'enhanced_patterns',
                'success': False,
                'concerns': [],
                'error': str(e)
            }
    
    def _create_hybrid_result(self, attempts: List[Dict], document_name: str) -> Dict:
        """Create hybrid result combining best aspects of different methods"""
        all_concerns = []
        
        for attempt in attempts:
            if attempt['success']:
                all_concerns.extend(attempt['concerns'])
        
        if not all_concerns:
            return {'concerns': []}
        
        # Prioritize by method reliability and quality
        sorted_concerns = sorted(all_concerns, key=lambda x: (
            x.get('confidence_score', 0),
            len(x.get('text', '')),
            1 if x.get('method') == 'pdfxtract' else 0
        ), reverse=True)
        
        # Take the best concern(s)
        best_concerns = self._select_best_concerns(sorted_concerns)
        
        return {
            'concerns': best_concerns,
            'method': 'hybrid',
            'sources': [attempt['method'] for attempt in attempts if attempt['success']]
        }
    
    def _try_basic_extraction(self, content: str, document_name: str) -> Dict:
        """Basic fallback extraction for when all else fails"""
        try:
            # Look for any sentences that mention key concern words
            sentences = re.split(r'[.!?]+', content)
            concern_keywords = ['concern', 'issue', 'problem', 'failure', 'inadequate', 'insufficient', 'lack']
            
            potential_concerns = []
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 30 and 
                    any(keyword in sentence.lower() for keyword in concern_keywords)):
                    potential_concerns.append({
                        'id': f"basic_{len(potential_concerns)}",
                        'text': sentence,
                        'method': 'basic_keyword',
                        'type': 'potential_concern',
                        'confidence_score': 0.5,
                        'extracted_at': datetime.now().isoformat()
                    })
            
            # Take only the best 3
            best_concerns = sorted(potential_concerns, key=lambda x: len(x['text']), reverse=True)[:3]
            
            return {
                'method': 'basic_extraction',
                'success': len(best_concerns) > 0,
                'concerns': best_concerns
            }
            
        except Exception as e:
            return {
                'method': 'basic_extraction',
                'success': False,
                'concerns': [],
                'error': str(e)
            }
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of extracted text"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length scoring
        if len(text) > 100: score += 0.3
        elif len(text) > 50: score += 0.2
        elif len(text) > 20: score += 0.1
        
        # Content quality
        if any(word in text.lower() for word in ['concern', 'matter', 'issue']): score += 0.2
        if any(word in text.lower() for word in ['coroner', 'inquest', 'regulation']): score += 0.2
        if re.search(r'[A-Z][a-z]+:', text): score += 0.1  # Structured format
        if len(text.split('.')) > 1: score += 0.1  # Multiple sentences
        if re.search(r'\b(?:should|must|need|require)\b', text.lower()): score += 0.1  # Action words
        
        return min(score, 1.0)
    
    def _assess_concerns_quality(self, concerns: List[Dict]) -> float:
        """Assess overall quality of extracted concerns"""
        if not concerns:
            return 0.0
        
        total_score = sum(self._assess_text_quality(c.get('text', '')) for c in concerns)
        return total_score / len(concerns)
    
    def _select_best_concerns(self, concerns: List[Dict]) -> List[Dict]:
        """Select the best concerns from a list"""
        if not concerns:
            return []
        
        # If we have a high-quality PDFxtract result, prefer it
        pdfxtract_concerns = [c for c in concerns if c.get('method') == 'pdfxtract']
        if pdfxtract_concerns and len(pdfxtract_concerns[0].get('text', '')) > 50:
            return pdfxtract_concerns[:1]  # Take the best PDFxtract result
        
        # Otherwise, take the highest confidence concerns
        return concerns[:2]  # Take top 2 concerns
    
    def _normalize_content(self, content: str) -> str:
        """Minimal normalization to preserve PDFxtract compatibility"""
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)
        return content.strip()
    
    def _deduplicate_concerns(self, concerns: List[Dict]) -> List[Dict]:
        """Remove duplicate concerns"""
        unique_concerns = []
        seen_texts = set()
        
        for concern in concerns:
            text_key = concern['text'][:100].lower().strip()
            if text_key not in seen_texts:
                unique_concerns.append(concern)
                seen_texts.add(text_key)
        
        return unique_concerns


# ===============================================
# SIMPLIFIED UI FOR UNIFIED EXTRACTION
# ===============================================

def render_unified_extraction_interface():
    """Simplified, user-friendly interface for the unified extraction"""
    
    st.header("🎯 Smart Concern Extraction")
    
    st.markdown("""
    **Intelligent extraction** that automatically uses the best method for your documents:
    - 🔍 **PDFxtract**: Robust coroner document extraction
    - 🎯 **Enhanced Patterns**: Additional coverage for edge cases  
    - 🤖 **Smart Selection**: Automatically picks the best result
    
    Just select your documents and click extract - the system handles the rest!
    """)
    
    if not st.session_state.get('uploaded_documents'):
        st.warning("⚠️ Please upload documents first in the Upload tab.")
        return
    
    # Document selection
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_docs = st.multiselect(
            "📁 Select documents to extract from:",
            options=doc_options,
            default=doc_options,
            help="Choose which documents to process for concern extraction"
        )
    
    with col2:
        st.markdown("**📊 Quick Info:**")
        st.write(f"• Documents available: {len(doc_options)}")
        st.write(f"• Selected: {len(selected_docs)}")
        st.write("• Method: Smart Auto-Select")
        st.write("• Confidence: High")
    
    # Settings (simplified)
    with st.expander("⚙️ Advanced Settings (Optional)"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_text_length = st.slider(
                "Minimum text length",
                min_value=10,
                max_value=100,
                value=30,
                help="Minimum length for extracted concerns"
            )
            
            include_metadata = st.checkbox(
                "Extract metadata",
                value=True,
                help="Extract additional metadata like dates, refs, etc."
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Minimum confidence for including results"
            )
            
            max_concerns_per_doc = st.slider(
                "Max concerns per document",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of concerns to extract per document"
            )
    
    # Extraction button
    if selected_docs:
        if st.button("🚀 Extract Concerns", type="primary", use_container_width=True):
            run_unified_extraction(
                selected_docs, 
                min_text_length, 
                confidence_threshold, 
                max_concerns_per_doc,
                include_metadata
            )
    else:
        st.info("📌 Please select at least one document to extract from.")

def run_unified_extraction(selected_docs, min_text_length, confidence_threshold, max_concerns_per_doc, include_metadata):
    """Run the unified extraction process"""
    
    # Initialize extractor
    extractor = UnifiedConcernExtractor()
    
    # Results storage
    all_concerns = []
    processing_results = []
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_docs = len(selected_docs)
        
        for i, doc_name in enumerate(selected_docs):
            # Update progress
            progress = (i + 1) / total_docs
            progress_bar.progress(progress)
            status_text.text(f"Processing {doc_name}... ({i+1}/{total_docs})")
            
            # Get document
            doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
            
            if not doc or not doc.get('content'):
                processing_results.append({
                    'document': doc_name,
                    'status': 'no_content',
                    'method_used': 'none',
                    'concerns_found': 0
                })
                continue
            
            content = doc['content']
            
            try:
                # Run unified extraction
                result = extractor.extract_concerns_unified(content, doc_name)
                
                concerns = result['concerns']
                summary = result['extraction_summary']
                
                # Filter by confidence and length
                filtered_concerns = [
                    c for c in concerns 
                    if (c.get('confidence_score', 0) >= confidence_threshold and
                        len(c.get('text', '')) >= min_text_length)
                ]
                
                # Limit number of concerns
                filtered_concerns = filtered_concerns[:max_concerns_per_doc]
                
                # Add metadata if requested
                if include_metadata:
                    for concern in filtered_concerns:
                        try:
                            # Add basic metadata
                            concern['word_count'] = len(concern.get('text', '').split())
                            concern['character_count'] = len(concern.get('text', ''))
                            # You could add more metadata extraction here
                        except:
                            pass
                
                all_concerns.extend(filtered_concerns)
                
                processing_results.append({
                    'document': doc_name,
                    'status': 'success',
                    'method_used': summary.get('method_used', 'unknown'),
                    'concerns_found': len(filtered_concerns),
                    'confidence': summary.get('confidence', 0),
                    'extraction_attempts': summary.get('total_attempts', 1)
                })
                
            except Exception as e:
                processing_results.append({
                    'document': doc_name,
                    'status': 'error',
                    'method_used': 'none',
                    'concerns_found': 0,
                    'error': str(e)
                })
                st.error(f"Error processing {doc_name}: {e}")
    
    # Clear progress
    progress_container.empty()
    
    # Store results
    st.session_state.extracted_concerns = all_concerns
    st.session_state.last_extraction_results = processing_results
    
    # Display results
    display_unified_extraction_results(all_concerns, processing_results)

def display_unified_extraction_results(concerns, processing_results):
    """Display unified extraction results in a clean, informative way"""
    
    st.success("🎉 Smart Extraction Complete!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        successful_docs = len([r for r in processing_results if r['status'] == 'success'])
        total_docs = len(processing_results)
        st.metric("Documents Processed", f"{successful_docs}/{total_docs}")
    
    with col2:
        st.metric("Total Concerns Found", len(concerns))
    
    with col3:
        if concerns:
            avg_confidence = sum(c.get('confidence_score', 0) for c in concerns) / len(concerns)
            st.metric("Average Confidence", f"{avg_confidence:.2f}")
        else:
            st.metric("Average Confidence", "0.00")
    
    with col4:
        methods_used = set(r.get('method_used', 'unknown') for r in processing_results if r['status'] == 'success')
        st.metric("Methods Used", len(methods_used))
    
    # Method breakdown
    if processing_results:
        st.subheader("📊 Extraction Method Breakdown")
        
        method_counts = {}
        for result in processing_results:
            if result['status'] == 'success':
                method = result.get('method_used', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            method_df = pd.DataFrame([
                {'Method': method, 'Documents': count, 'Description': get_method_description(method)}
                for method, count in method_counts.items()
            ])
            st.dataframe(method_df, use_container_width=True)
    
    # Sample concerns
    if concerns:
        st.subheader("📋 Extracted Concerns")
        
        # Show first few concerns
        for i, concern in enumerate(concerns[:5]):
            with st.expander(f"Concern {i+1} - {concern.get('document_source', 'Unknown')} (Confidence: {concern.get('confidence_score', 0):.2f})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Text:**")
                    st.write(concern.get('text', ''))
                
                with col2:
                    st.write(f"**Method:** {concern.get('method', 'Unknown')}")
                    st.write(f"**Type:** {concern.get('type', 'Unknown')}")
                    st.write(f"**Words:** {concern.get('word_count', len(concern.get('text', '').split()))}")
                    st.write(f"**Confidence:** {concern.get('confidence_score', 0):.2f}")
        
        if len(concerns) > 5:
            st.info(f"Showing first 5 of {len(concerns)} concerns. All concerns are saved for annotation.")

def get_method_description(method):
    """Get description for extraction method"""
    descriptions = {
        'pdfxtract': 'Robust coroner document extraction',
        'enhanced_patterns': 'Pattern-based extraction',
        'hybrid': 'Combined multiple methods',
        'basic_fallback': 'Basic keyword extraction',
        'none': 'No extraction performed'
    }
    return descriptions.get(method, 'Unknown method')

# Replace the old interface function
def render_extraction_tab():
    """Main extraction tab - now uses unified extraction"""
    render_unified_extraction_interface()

if __name__ == "__main__":
    st.title("🎯 Unified Concern Extraction System")
    render_unified_extraction_interface()
