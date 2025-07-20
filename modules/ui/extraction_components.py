# ===============================================
# FILE: modules/ui/extraction_components.py (FIXED IMPORTS VERSION)
# ===============================================

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import logging
import sys
from pathlib import Path

# Add modules to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

# Try multiple import approaches to find the enhanced extraction classes
def get_extraction_classes():
    """Try different ways to import the extraction classes"""
    
    # Method 1: Try importing from enhanced_extraction in the same directory
    try:
        from .enhanced_extraction import EnhancedConcernExtractor, StandardLLMExtractor
        return EnhancedConcernExtractor, StandardLLMExtractor, "ui.enhanced_extraction"
    except ImportError:
        pass
    
    # Method 2: Try importing from modules.ui.enhanced_extraction
    try:
        from modules.ui.enhanced_extraction import EnhancedConcernExtractor, StandardLLMExtractor
        return EnhancedConcernExtractor, StandardLLMExtractor, "modules.ui.enhanced_extraction"
    except ImportError:
        pass
    
    # Method 3: Try importing from enhanced_extraction directly
    try:
        import enhanced_extraction
        return enhanced_extraction.EnhancedConcernExtractor, enhanced_extraction.StandardLLMExtractor, "enhanced_extraction"
    except ImportError:
        pass
    
    # Method 4: Try from current directory
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from enhanced_extraction import EnhancedConcernExtractor, StandardLLMExtractor
        return EnhancedConcernExtractor, StandardLLMExtractor, "current_dir.enhanced_extraction"
    except ImportError:
        pass
    
    # Method 5: Create fallback classes
    return None, None, "fallback"

# Try to get the extraction classes
EnhancedConcernExtractor, StandardLLMExtractor, import_source = get_extraction_classes()

# Create fallback classes if imports failed
if EnhancedConcernExtractor is None:
    
    class EnhancedConcernExtractor:
        """Fallback Enhanced Concern Extractor"""
        
        def __init__(self):
            self.logger = logging.getLogger(__name__)
        
        def extract_concerns_robust(self, content: str, document_name: str = "") -> Dict:
            """Fallback extraction using basic patterns"""
            
            # Try to use the core_utils extract_concern_text if available
            try:
                from core_utils import extract_concern_text
                extracted_text = extract_concern_text(content)
                
                if extracted_text and len(extracted_text.strip()) > 20:
                    return {
                        'concerns': [{
                            'id': 'fallback_1',
                            'text': extracted_text.strip(),
                            'method': 'core_utils_extract_concern_text',
                            'type': 'coroner_concern',
                            'confidence_score': 0.9,
                            'extracted_at': datetime.now().isoformat(),
                            'document_source': document_name
                        }],
                        'debug_info': {
                            'method': 'core_utils_fallback',
                            'success': True
                        }
                    }
            except ImportError:
                pass
            
            # Basic pattern fallback
            import re
            patterns = [
                r"CORONER'S\s+CONCERNS?:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|$)",
                r"MATTERS?\s+OF\s+CONCERN:?\s*(.*?)(?=ACTION\s+SHOULD\s+BE\s+TAKEN|CONCLUSIONS|YOUR\s+RESPONSE|$)",
            ]
            
            concerns = []
            for i, pattern in enumerate(patterns):
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    text = match.group(1).strip()
                    if len(text) > 20:
                        concerns.append({
                            'id': f'fallback_{i}',
                            'text': text,
                            'method': 'basic_pattern_fallback',
                            'type': 'coroner_concern',
                            'confidence_score': 0.7,
                            'extracted_at': datetime.now().isoformat(),
                            'document_source': document_name
                        })
            
            return {
                'concerns': concerns,
                'debug_info': {
                    'method': 'basic_pattern_fallback',
                    'patterns_tried': len(patterns),
                    'success': len(concerns) > 0
                }
            }

if StandardLLMExtractor is None:
    
    class StandardLLMExtractor:
        """Fallback Standard LLM Extractor"""
        
        def __init__(self, api_key: str = None):
            self.api_key = api_key or st.secrets.get("OPENAI_API_KEY")
            self.logger = logging.getLogger(__name__)
        
        def extract_concerns_llm(self, content: str, document_name: str = "") -> Dict:
            """Fallback LLM extraction"""
            
            if not self.api_key:
                return {
                    'concerns': [],
                    'debug_info': {
                        'error': 'No OpenAI API key available',
                        'method': 'llm_fallback'
                    }
                }
            
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                
                # Simple prompt
                prompt = f"""Extract concerns from this coroner document. Return as JSON array with 'text' and 'confidence' fields.

Document: {content[:3000]}"""
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Extract concerns from coroner documents. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                import json
                try:
                    llm_concerns = json.loads(result_text)
                    
                    formatted_concerns = []
                    for i, concern in enumerate(llm_concerns):
                        formatted_concerns.append({
                            'id': f"llm_fallback_{i}",
                            'text': concern.get('text', ''),
                            'method': 'llm_fallback',
                            'type': 'llm_concern',
                            'confidence_score': concern.get('confidence', 0.7),
                            'extracted_at': datetime.now().isoformat(),
                            'document_source': document_name
                        })
                    
                    return {
                        'concerns': formatted_concerns,
                        'debug_info': {
                            'method': 'llm_fallback',
                            'success': True,
                            'concerns_found': len(formatted_concerns)
                        }
                    }
                    
                except json.JSONDecodeError:
                    return {
                        'concerns': [],
                        'debug_info': {
                            'error': 'Failed to parse LLM response as JSON',
                            'method': 'llm_fallback',
                            'raw_response': result_text[:200]
                        }
                    }
                    
            except Exception as e:
                return {
                    'concerns': [],
                    'debug_info': {
                        'error': f"LLM extraction failed: {str(e)}",
                        'method': 'llm_fallback'
                    }
                }

# =============================================================================
# MAIN EXTRACTION TAB - SIMPLIFIED VERSION
# =============================================================================

def render_extraction_tab():
    """Main extraction tab - SIMPLIFIED, CLEAR VERSION"""
    
    st.title("📄 Extract Concerns from Documents")
    
    # Show import status for debugging
    if import_source == "fallback":
        st.warning("⚠️ Using fallback extraction methods. Enhanced extraction module not found.")
        st.info(f"💡 Import source: {import_source}")
    else:
        st.success(f"✅ Using extraction classes from: {import_source}")
    
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
# EXTRACTION FUNCTIONS - SIMPLIFIED
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
        
        # Initialize extractor
        extractor = EnhancedConcernExtractor()
        
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
        
        # Initialize extractor
        extractor = StandardLLMExtractor(api_key)
        
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

if __name__ == "__main__":
    st.title("📄 Simple Extraction Interface")
    render_extraction_tab()
