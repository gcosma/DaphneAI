# ===============================================
# COMPLETE ENHANCED EXTRACTION SYSTEM
# ===============================================

import re
import logging
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class EnhancedConcernExtractor:
    """Enhanced concern extraction with multiple robust methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced patterns for concern detection
        self.concern_patterns = {
            'standard': [
                r'(?:coroner|matter of )?concern(?:s)?[\s:]+([^.]+(?:\.[^.]*){0,3}\.)',
                r'(?:identified|raised|expressed)\s+(?:a\s+)?concern(?:s)?[\s:]+([^.]+\.)',
                r'concern\s+(?:that|about|regarding)[\s:]+([^.]+\.)',
                r'(?:this|the)\s+concern(?:s)?[\s:]+([^.]+\.)'
            ],
            'flexible': [
                r'(?:matter|issue|problem)(?:s)?[\s:]+([^.]+(?:\.[^.]*){0,2}\.)',
                r'(?:identified|noted|found)\s+(?:that|the following)[\s:]+([^.]+\.)',
                r'(?:this|the)\s+(?:matter|issue)[\s:]+([^.]+\.)',
                r'(?:prevention|recommendation)[\s:]+([^.]+\.)'
            ],
            'section': [
                r'(?:coroner|matter)\s+concern(?:s)?[\s\n]*([^.]+(?:\.[^.]*){0,5}\.)',
                r'prevention[\s\n]+of[\s\n]+future[\s\n]+death(?:s)?[\s\n]*([^.]+(?:\.[^.]*){0,5}\.)',
                r'regulation\s+28[\s\n]*report[\s\n]*([^.]+(?:\.[^.]*){0,5}\.)'
            ],
            'keyword': [
                r'(?:failure|inadequate|insufficient)[\s:]+([^.]+\.)',
                r'(?:lack\s+of|absence\s+of)[\s:]+([^.]+\.)',
                r'(?:should\s+have|ought\s+to\s+have)[\s:]+([^.]+\.)'
            ]
        }
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for better pattern matching"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Fix punctuation spacing
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)
        content = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', content)
        
        # Fix word boundaries
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
        content = re.sub(r'(\w)(\d)', r'\1 \2', content)
        content = re.sub(r'(\d)([A-Za-z])', r'\1 \2', content)
        
        return content.strip()
    
    def _extract_standard_patterns(self, content: str) -> List[Dict]:
        """Extract using standard concern patterns"""
        concerns = []
        
        for pattern in self.concern_patterns['standard']:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                text = match.group(1).strip()
                if len(text) > 20:  # Minimum length filter
                    concerns.append({
                        'id': f"std_{len(concerns)}",
                        'text': text,
                        'method': 'standard_pattern',
                        'type': 'coroner_concern',
                        'confidence_score': 0.8,
                        'extracted_at': datetime.now().isoformat(),
                        'pattern_used': pattern
                    })
        
        return concerns
    
    def _extract_flexible_patterns(self, content: str) -> List[Dict]:
        """Extract using flexible patterns"""
        concerns = []
        
        for pattern in self.concern_patterns['flexible']:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                text = match.group(1).strip()
                if len(text) > 15:
                    concerns.append({
                        'id': f"flex_{len(concerns)}",
                        'text': text,
                        'method': 'flexible_pattern',
                        'type': 'issue_matter',
                        'confidence_score': 0.7,
                        'extracted_at': datetime.now().isoformat(),
                        'pattern_used': pattern
                    })
        
        return concerns
    
    def _extract_section_patterns(self, content: str) -> List[Dict]:
        """Extract using section-based patterns"""
        concerns = []
        
        for pattern in self.concern_patterns['section']:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                text = match.group(1).strip()
                if len(text) > 30:  # Longer minimum for section patterns
                    concerns.append({
                        'id': f"sect_{len(concerns)}",
                        'text': text,
                        'method': 'section_pattern',
                        'type': 'section_concern',
                        'confidence_score': 0.9,
                        'extracted_at': datetime.now().isoformat(),
                        'pattern_used': pattern
                    })
        
        return concerns
    
    def _extract_keyword_patterns(self, content: str) -> List[Dict]:
        """Extract using keyword-based patterns"""
        concerns = []
        
        for pattern in self.concern_patterns['keyword']:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                text = match.group(1).strip()
                if len(text) > 10:
                    concerns.append({
                        'id': f"key_{len(concerns)}",
                        'text': text,
                        'method': 'keyword_pattern',
                        'type': 'failure_concern',
                        'confidence_score': 0.6,
                        'extracted_at': datetime.now().isoformat(),
                        'pattern_used': pattern
                    })
        
        return concerns
    
    def _deduplicate_concerns(self, concerns: List[Dict]) -> List[Dict]:
        """Remove duplicate concerns based on text similarity"""
        unique_concerns = []
        
        for concern in concerns:
            text = concern['text'].lower().strip()
            
            # Check for duplicates
            is_duplicate = False
            for existing in unique_concerns:
                existing_text = existing['text'].lower().strip()
                
                # Simple similarity check
                if (text in existing_text or existing_text in text or
                    self._calculate_text_similarity(text, existing_text) > 0.8):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if concern['confidence_score'] > existing['confidence_score']:
                        unique_concerns.remove(existing)
                        unique_concerns.append(concern)
                    break
            
            if not is_duplicate:
                unique_concerns.append(concern)
        
        return unique_concerns
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def extract_concerns_robust(self, content: str, document_name: str = "") -> Dict:
        """Extract concerns using multiple robust methods"""
        if not content or len(content.strip()) < 10:
            return {
                'concerns': [],
                'debug_info': {
                    'error': 'No content or content too short',
                    'content_length': len(content) if content else 0
                }
            }
        
        # Normalize content
        normalized_content = self._normalize_content(content)
        
        # Try different extraction methods
        all_concerns = []
        debug_info = {'methods_tried': [], 'results': {}}
        
        # Method 1: Standard patterns
        try:
            standard_concerns = self._extract_standard_patterns(normalized_content)
            debug_info['methods_tried'].append('standard_patterns')
            debug_info['results']['standard_patterns'] = len(standard_concerns)
            all_concerns.extend(standard_concerns)
        except Exception as e:
            debug_info['results']['standard_patterns'] = f"Error: {e}"
        
        # Method 2: Flexible patterns
        try:
            flexible_concerns = self._extract_flexible_patterns(normalized_content)
            debug_info['methods_tried'].append('flexible_patterns')
            debug_info['results']['flexible_patterns'] = len(flexible_concerns)
            all_concerns.extend(flexible_concerns)
        except Exception as e:
            debug_info['results']['flexible_patterns'] = f"Error: {e}"
        
        # Method 3: Section detection
        try:
            section_concerns = self._extract_section_patterns(normalized_content)
            debug_info['methods_tried'].append('section_detection')
            debug_info['results']['section_detection'] = len(section_concerns)
            all_concerns.extend(section_concerns)
        except Exception as e:
            debug_info['results']['section_detection'] = f"Error: {e}"
        
        # Method 4: Keyword extraction
        try:
            keyword_concerns = self._extract_keyword_patterns(normalized_content)
            debug_info['methods_tried'].append('keyword_extraction')
            debug_info['results']['keyword_extraction'] = len(keyword_concerns)
            all_concerns.extend(keyword_concerns)
        except Exception as e:
            debug_info['results']['keyword_extraction'] = f"Error: {e}"
        
        # Deduplicate concerns
        unique_concerns = self._deduplicate_concerns(all_concerns)
        
        # Add document source to all concerns
        for concern in unique_concerns:
            concern['document_source'] = document_name
        
        debug_info['total_before_dedup'] = len(all_concerns)
        debug_info['total_after_dedup'] = len(unique_concerns)
        debug_info['content_length'] = len(content)
        
        return {
            'concerns': unique_concerns,
            'debug_info': debug_info
        }


class StandardLLMExtractor:
    """Standard LLM-based extraction using OpenAI API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        self.logger = logging.getLogger(__name__)
    
    def extract_concerns_llm(self, content: str, document_name: str = "") -> Dict:
        """Extract concerns using LLM"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            prompt = f"""
            Extract concerns from the following coroner document text.
            
            Focus on:
            - Coroner concerns
            - Matters of concern
            - Issues identified
            - Problems noted
            - Failures mentioned
            
            Text: {content[:4000]}  # Limit to avoid token limits
            
            Return a JSON array of concerns with:
            - text: the concern text
            - confidence: confidence score (0-1)
            - type: concern type
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting concerns from coroner documents. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                llm_concerns = json.loads(result_text)
                
                # Format concerns
                formatted_concerns = []
                for i, concern in enumerate(llm_concerns):
                    formatted_concerns.append({
                        'id': f"llm_{i}",
                        'text': concern.get('text', ''),
                        'method': 'llm_extraction',
                        'type': concern.get('type', 'llm_concern'),
                        'confidence_score': concern.get('confidence', 0.7),
                        'extracted_at': datetime.now().isoformat(),
                        'document_source': document_name
                    })
                
                return {
                    'concerns': formatted_concerns,
                    'debug_info': {
                        'method': 'llm',
                        'response_length': len(result_text),
                        'concerns_found': len(formatted_concerns)
                    }
                }
                
            except json.JSONDecodeError as e:
                return {
                    'concerns': [],
                    'debug_info': {
                        'error': f"JSON parsing failed: {e}",
                        'raw_response': result_text[:500]
                    }
                }
                
        except Exception as e:
            return {
                'concerns': [],
                'debug_info': {
                    'error': f"LLM extraction failed: {e}"
                }
            }


def compare_extraction_methods(content: str, document_name: str = ""):
    """Compare standard and enhanced extraction methods"""
    
    st.subheader("🔍 Extraction Method Comparison")
    
    # Test standard extraction
    st.write("### Standard LLM Extraction")
    try:
        standard_extractor = StandardLLMExtractor()
        standard_result = standard_extractor.extract_concerns_llm(content, document_name)
        standard_concerns = standard_result['concerns']
        
        if standard_concerns:
            st.success(f"✅ **Standard extraction found:** {len(standard_concerns)} concerns")
            
            for i, concern in enumerate(standard_concerns[:3]):  # Show first 3
                with st.expander(f"Standard Concern {i+1} (Confidence: {concern['confidence_score']:.2f})"):
                    st.write(f"**Method:** {concern['method']}")
                    st.write(f"**Type:** {concern['type']}")
                    st.write(f"**Text:** {concern['text'][:200]}...")
        else:
            st.warning("⚠️ **Standard extraction found nothing**")
            
        with st.expander("🔍 Standard Extraction Debug Info"):
            st.json(standard_result['debug_info'])
            
    except Exception as e:
        st.error(f"❌ **Standard extraction failed:** {e}")
    
    # Test enhanced extraction
    st.write("### Enhanced Pattern Extraction")
    try:
        enhanced_extractor = EnhancedConcernExtractor()
        enhanced_result = enhanced_extractor.extract_concerns_robust(content, document_name)
        enhanced_concerns = enhanced_result['concerns']
        
        if enhanced_concerns:
            st.success(f"✅ **Enhanced extraction found:** {len(enhanced_concerns)} concerns")
            
            for i, concern in enumerate(enhanced_concerns[:3]):  # Show first 3
                with st.expander(f"Enhanced Concern {i+1} (Confidence: {concern['confidence_score']:.2f})"):
                    st.write(f"**Method:** {concern['method']}")
                    st.write(f"**Type:** {concern['type']}")
                    st.write(f"**Text:** {concern['text'][:200]}...")
        else:
            st.warning("⚠️ **Enhanced extraction found no concerns**")
        
        with st.expander("🔍 Enhanced Extraction Debug Info"):
            st.json(enhanced_result['debug_info'])
            
    except Exception as e:
        st.error(f"❌ **Enhanced extraction failed:** {e}")
    
    # Method comparison summary
    st.write("### Method Comparison Summary")
    
    comparison_data = [
        {
            'Method': 'Standard LLM',
            'Type': 'AI-powered',
            'API Required': 'Yes (OpenAI)',
            'Speed': 'Slower',
            'Accuracy': 'High (context-aware)',
            'Cost': 'Per API call',
            'Reliability': 'Depends on API'
        },
        {
            'Method': 'Enhanced Pattern',
            'Type': 'Pattern-based',
            'API Required': 'No',
            'Speed': 'Faster',
            'Accuracy': 'High (for structured docs)',
            'Cost': 'None',
            'Reliability': 'Always available'
        }
    ]
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)


def run_complete_extraction_workflow(documents: List[Dict], use_both_methods: bool = True):
    """Run complete extraction workflow with both methods"""
    
    st.subheader("🚀 Complete Extraction Workflow")
    
    # Initialize extractors
    enhanced_extractor = EnhancedConcernExtractor()
    standard_extractor = StandardLLMExtractor()
    
    # Results storage
    all_concerns = []
    workflow_results = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_docs = len(documents)
    
    for i, doc in enumerate(documents):
        progress = (i + 1) / total_docs
        progress_bar.progress(progress)
        
        doc_name = doc['filename']
        content = doc.get('content', '')
        
        status_text.text(f"Processing {doc_name}...")
        
        if not content:
            workflow_results.append({
                'document': doc_name,
                'status': 'no_content',
                'enhanced_concerns': 0,
                'standard_concerns': 0
            })
            continue
        
        doc_concerns = []
        
        # Enhanced extraction
        try:
            enhanced_result = enhanced_extractor.extract_concerns_robust(content, doc_name)
            enhanced_concerns = enhanced_result['concerns']
            doc_concerns.extend(enhanced_concerns)
            enhanced_count = len(enhanced_concerns)
        except Exception as e:
            enhanced_count = 0
            st.warning(f"Enhanced extraction failed for {doc_name}: {e}")
        
        # Standard extraction (if enabled)
        standard_count = 0
        if use_both_methods:
            try:
                standard_result = standard_extractor.extract_concerns_llm(content, doc_name)
                standard_concerns = standard_result['concerns']
                doc_concerns.extend(standard_concerns)
                standard_count = len(standard_concerns)
            except Exception as e:
                st.warning(f"Standard extraction failed for {doc_name}: {e}")
        
        all_concerns.extend(doc_concerns)
        
        workflow_results.append({
            'document': doc_name,
            'status': 'success',
            'enhanced_concerns': enhanced_count,
            'standard_concerns': standard_count,
            'total_concerns': len(doc_concerns)
        })
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Show results summary
    st.success(f"🎉 Workflow Complete! Found {len(all_concerns)} total concerns")
    
    # Results breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enhanced_total = sum(r['enhanced_concerns'] for r in workflow_results)
        st.metric("Enhanced Extraction", enhanced_total)
    
    with col2:
        standard_total = sum(r['standard_concerns'] for r in workflow_results)
        st.metric("Standard Extraction", standard_total)
    
    with col3:
        total_docs_processed = len([r for r in workflow_results if r['status'] == 'success'])
        st.metric("Documents Processed", total_docs_processed)
    
    # Detailed results table
    st.write("### Detailed Results")
    results_df = pd.DataFrame(workflow_results)
    st.dataframe(results_df, use_container_width=True)
    
    return all_concerns, workflow_results


# Usage example and testing interface
def main_extraction_interface():
    """Main interface for extraction testing and comparison"""
    
    st.title("🔧 Complete Extraction System")
    
    # Check if documents are available
    if 'uploaded_documents' not in st.session_state or not st.session_state.uploaded_documents:
        st.warning("Please upload documents first to test extraction methods.")
        return
    
    # Document selection
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    selected_doc = st.selectbox("Select document for testing:", doc_options)
    
    if selected_doc:
        # Get document content
        doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == selected_doc), None)
        if doc and doc.get('content'):
            content = doc['content']
            
            # Show content preview
            with st.expander("📄 Document Content Preview"):
                st.text_area("Content", content[:1000] + "..." if len(content) > 1000 else content, height=200)
            
            # Extraction options
            st.write("### Extraction Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎯 Test Enhanced Extraction", type="primary"):
                    compare_extraction_methods(content, selected_doc)
            
            with col2:
                if st.button("🚀 Run Full Workflow", type="secondary"):
                    selected_docs = [doc]
                    run_complete_extraction_workflow(selected_docs, use_both_methods=True)
        
        else:
            st.error("Selected document has no content.")


if __name__ == "__main__":
    main_extraction_interface()
