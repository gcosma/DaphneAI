# ===============================================
# FILE: modules/ui/extraction_components.py (FIXED - KEEP WORKING CLASSES)
# ===============================================

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import logging
import sys
import os
from pathlib import Path

# Add the current directory and parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import the working enhanced extraction classes
enhanced_extractor_class = None
standard_extractor_class = None

def import_enhanced_classes():
    """Import the enhanced extraction classes using multiple methods"""
    global enhanced_extractor_class, standard_extractor_class
    
    import_methods = [
        # Method 1: Direct import from same directory
        lambda: __import__('enhanced_extraction'),
        
        # Method 2: Import from modules.ui
        lambda: __import__('modules.ui.enhanced_extraction', fromlist=['EnhancedConcernExtractor', 'StandardLLMExtractor']),
        
        # Method 3: Relative import
        lambda: __import__('.enhanced_extraction', package='modules.ui', fromlist=['EnhancedConcernExtractor', 'StandardLLMExtractor']),
        
        # Method 4: Direct file execution
        lambda: exec(open(os.path.join(current_dir, 'enhanced_extraction.py')).read(), globals())
    ]
    
    for i, method in enumerate(import_methods):
        try:
            if i < 3:  # For import methods
                module = method()
                enhanced_extractor_class = getattr(module, 'EnhancedConcernExtractor', None)
                standard_extractor_class = getattr(module, 'StandardLLMExtractor', None)
                
                if enhanced_extractor_class and standard_extractor_class:
                    st.success(f"✅ Successfully imported extraction classes using method {i+1}")
                    return True
            else:  # For exec method
                method()
                if 'EnhancedConcernExtractor' in globals() and 'StandardLLMExtractor' in globals():
                    enhanced_extractor_class = globals()['EnhancedConcernExtractor']
                    standard_extractor_class = globals()['StandardLLMExtractor']
                    st.success(f"✅ Successfully loaded extraction classes using exec method")
                    return True
                    
        except Exception as e:
            st.warning(f"⚠️ Import method {i+1} failed: {str(e)[:100]}")
            continue
    
    return False

# Try to import the classes
import_success = import_enhanced_classes()

# If imports failed, create inline versions of the working classes
if not import_success or not enhanced_extractor_class:
    st.warning("⚠️ Creating inline extraction classes from your working code...")
    
    # Import required modules for the inline classes
    import re
    import json
    
    class EnhancedConcernExtractor:
        """Enhanced concern extraction with multiple robust methods (inline version)"""
        
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            
            # Enhanced patterns for concern detection (from your working code)
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
        """Standard LLM-based extraction using OpenAI API (inline version)"""
        
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
    
    # Set the classes
    enhanced_extractor_class = EnhancedConcernExtractor
    standard_extractor_class = StandardLLMExtractor
    
    st.success("✅ Using inline extraction classes (based on your working code)")

# =============================================================================
# MAIN EXTRACTION TAB - SIMPLIFIED VERSION
# =============================================================================

def render_extraction_tab():
    """Main extraction tab - SIMPLIFIED, CLEAR VERSION"""
    
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
        ✅ **Same as PDFxtract** - Consistent with BERT annotation
        
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
        ⚠️ **Costs money** - ~$0.001 per document
        
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
        run_pattern_extraction_simple(selected_docs)
    
    elif use_ai_method:
        run_ai_extraction_simple(selected_docs)
    
    elif compare_methods:
        run_both_methods_simple(selected_docs)
    
    # Show results if available
    show_simple_extraction_results()

# =============================================================================
# EXTRACTION FUNCTIONS - USING WORKING CLASSES
# =============================================================================

def run_pattern_extraction_simple(selected_docs):
    """Run pattern-based extraction with clear progress"""
    
    st.subheader("🎯 Running Pattern Extraction...")
    
    # Progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_concerns = []
        doc_results = []
        
        # Initialize extractor using the working class
        extractor = enhanced_extractor_class()
        
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
                # Run extraction using the working method
                result = extractor.extract_concerns_robust(doc['content'], doc_name)
                concerns = result['concerns']
                
                # Filter out very short concerns
                good_concerns = [c for c in concerns if len(c.get('text', '')) > 20]
                
                all_concerns.extend(good_concerns)
                
                doc_results.append({
                    'document': doc_name,
                    'status': f'✅ Success',
                    'concerns_found': len(good_concerns),
                    'debug_info': result.get('debug_info', {})
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
        
    st.session_state.extraction_results['pattern'] = {
        'method': 'Pattern Method',
        'concerns': all_concerns,
        'doc_results': doc_results,
        'timestamp': datetime.now()
    }
    
    # Store in the main concerns list for BERT annotation
    st.session_state.extracted_concerns = all_concerns
    
    # Show immediate results
    with results_container:
        if all_concerns:
            st.success(f"🎉 **Pattern extraction completed!** Found **{len(all_concerns)} concerns** from {len(selected_docs)} documents.")
            
            # Show debug info if available
            with st.expander("🔍 Debug Information"):
                for result in doc_results:
                    if 'debug_info' in result:
                        st.write(f"**{result['document']}:**")
                        st.json(result['debug_info'])
        else:
            st.warning("⚠️ **No concerns found** with pattern method. Try the AI method or check your documents.")

def run_ai_extraction_simple(selected_docs):
    """Run AI extraction with clear progress"""
    
    st.subheader("🤖 Running AI Extraction...")
    
    # Check for API key first
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ **OpenAI API key required** for AI method")
        st.info("💡 **Suggestion:** Use the Pattern Method instead - it's free and works great for coroner documents!")
        return
    
    # Progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_concerns = []
        doc_results = []
        
        # Initialize extractor using the working class
        extractor = standard_extractor_class(api_key)
        
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
                # Run AI extraction using the working method
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
    
    # Store in the main concerns list for BERT annotation
    if all_concerns:
        st.session_state.extracted_concerns = all_concerns
    
    # Show immediate results
    with results_container:
        if all_concerns:
            st.success(f"🎉 **AI extraction completed!** Found **{len(all_concerns)} concerns** from {len(selected_docs)} documents.")
        else:
            st.warning("⚠️ **No concerns found** with AI method. This might be because the documents don't contain clear concern patterns.")

def run_both_methods_simple(selected_docs):
    """Run both methods for comparison"""
    
    st.subheader("⚡ Running Both Methods for Comparison...")
    
    st.info("🔄 This will run both Pattern and AI methods so you can compare the results.")
    
    # Run pattern first
    with st.expander("🎯 Pattern Method Progress", expanded=True):
        run_pattern_extraction_simple(selected_docs)
    
    # Run AI second
    with st.expander("🤖 AI Method Progress", expanded=True):
        run_ai_extraction_simple(selected_docs)
    
    st.success("🎉 **Both methods completed!** See comparison below.")

# =============================================================================
# RESULTS DISPLAY - SIMPLIFIED
# =============================================================================

def show_simple_extraction_results():
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
                    concern_text = concern.get('text', '')
                    display_text = concern_text[:200] + '...' if len(concern_text) > 200 else concern_text
                    st.write(f"'{display_text}'")
                    
                    # Show extraction method and confidence
                    method_info = concern.get('method', 'Unknown')
                    confidence = concern.get('confidence_score', 0)
                    st.caption(f"Method: {method_info} | Confidence: {confidence:.2f}")
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
            if 'extracted_concerns' in st.session_state:
                del st.session_state.extracted_concerns
            st.rerun()

# =============================================================================
# LEGACY FUNCTIONS (kept for compatibility)
# =============================================================================

def display_extraction_results():
    """Legacy function - redirects to simple results"""
    show_simple_extraction_results()

def render_document_status_check():
    """Legacy function - simplified version"""
    pass  # Not needed in simplified interface

def render_extraction_configuration():
    """Legacy function - simplified version"""
    pass  # Not needed in simplified interface

def render_extraction_interface():
    """Legacy function - simplified version"""
    pass  # Not needed in simplified interface

def render_enhanced_concern_extraction():
    """Legacy function - simplified version"""
    pass  # Not needed in simplified interface

if __name__ == "__main__":
    st.title("📄 Simple Extraction Interface")
    render_extraction_tab()
