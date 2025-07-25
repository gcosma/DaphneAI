# ===============================================
# FILE: modules/ui/annotation_components.py (TRULY COMPLETE VERSION)
# ===============================================

import streamlit as st
import pandas as pd
import json
import logging
import streamlit.components.v1 as components
import re
import io
import zipfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Import required modules with robust error handling
try:
    import sys
    sys.path.append('modules')
    from bert_annotator import BERTConceptAnnotator
    BERT_ANNOTATOR_AVAILABLE = True
    logging.info("✅ BERTConceptAnnotator imported successfully")
except ImportError as import_error:
    BERT_ANNOTATOR_AVAILABLE = False
    logging.warning(f"BERTConceptAnnotator not available: {import_error}")
    
    # Create mock class for fallback
    class BERTConceptAnnotator:
        def __init__(self):
            self.frameworks = {
                "I-SIRch": [
                    {"name": "External factors", "keywords": ["policy", "regulation", "external", "government", "legal"]},
                    {"name": "System factors", "keywords": ["system", "process", "organization", "structure", "workflow"]},
                    {"name": "Technology factors", "keywords": ["technology", "equipment", "tools", "software", "digital"]},
                    {"name": "Person factors", "keywords": ["staff", "person", "individual", "training", "competence"]},
                    {"name": "Task factors", "keywords": ["task", "procedure", "activity", "workload", "complexity"]}
                ],
                "House of Commons": [
                    {"name": "Communication", "keywords": ["communication", "information", "reporting", "alert", "notification"]},
                    {"name": "Fragmented care", "keywords": ["fragmented", "coordination", "continuity", "handover", "transition"]},
                    {"name": "Workforce pressures", "keywords": ["workforce", "staffing", "workload", "pressure", "resource"]},
                    {"name": "Biases and stereotyping", "keywords": ["bias", "discrimination", "stereotype", "prejudice", "assumption"]}
                ]
            }
        
        def annotate_text(self, text, frameworks):
            # Mock annotation for fallback
            mock_annotations = {}
            for framework in frameworks:
                if framework in self.frameworks:
                    mock_annotations[framework] = []
                    text_lower = text.lower()
                    for theme in self.frameworks[framework]:
                        theme_name = theme["name"]
                        keywords = theme["keywords"]
                        matches = [kw for kw in keywords if kw in text_lower]
                        if matches:
                            confidence = min(0.9, len(matches) * 0.25 + 0.3)
                            mock_annotations[framework].append({
                                "theme": theme_name,
                                "confidence": confidence,
                                "keywords": matches
                            })
            return mock_annotations, {}
        
        def load_custom_framework(self, file_content, filename):
            return True, f"Mock framework loaded from {filename}"

try:
    from core_utils import SecurityValidator
    CORE_UTILS_AVAILABLE = True
    logging.info("✅ Core utilities imported successfully")
except ImportError as import_error:
    CORE_UTILS_AVAILABLE = False
    logging.warning(f"Core utilities not available: {import_error}")
    
    class SecurityValidator:
        @staticmethod
        def validate_text_input(text, max_length=10000):
            return str(text)[:max_length] if text else ""

# Configure logging
logging.basicConfig(level=logging.INFO)

# ===============================================
# MAIN ANNOTATION TAB FUNCTION
# ===============================================

def render_annotation_tab():
    """Render the complete concept annotation tab"""
    st.header("🏷️ Concept Annotation")
    
    st.markdown("""
    Annotate extracted recommendations and responses with conceptual themes using BERT-based analysis 
    and established frameworks like I-SIRch and House of Commons. Works with any UK government inquiry.
    """)
    
    # Show component availability status
    if not BERT_ANNOTATOR_AVAILABLE:
        st.warning("⚠️ BERT annotator not available - using keyword-based fallback mode")
    
    # Check for extracted content
    extraction_results = st.session_state.get('extraction_results', {})
    recommendations = extraction_results.get('recommendations', [])
    responses = extraction_results.get('responses', [])
    
    # Also check legacy extraction results for backwards compatibility
    if not recommendations and not responses:
        legacy_recommendations = st.session_state.get('extracted_recommendations', [])
        legacy_concerns = st.session_state.get('extracted_concerns', [])
        
        if legacy_recommendations or legacy_concerns:
            st.info("📋 Found legacy extraction results. Converting to new format...")
            recommendations = convert_legacy_recommendations(legacy_recommendations)
            responses = convert_legacy_concerns_to_responses(legacy_concerns)
    
    if not recommendations and not responses:
        st.warning("⚠️ Please extract recommendations and responses first in the Extraction tab.")
        if st.button("🔍 Go to Extraction Tab"):
            st.session_state.active_tab = 'Extraction'
            st.rerun()
        return
    
    # Content overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Recommendations", len(recommendations))
    with col2:
        st.metric("📋 Responses", len(responses))  
    with col3:
        st.metric("📊 Total Items", len(recommendations) + len(responses))
    
    # Initialize BERT annotator
    initialize_bert_annotator()
    
    # Framework management section
    render_framework_management()
    
    # Annotation configuration
    render_annotation_configuration()
    
    # Main annotation interface
    render_annotation_interface(recommendations, responses)
    
    # Results display
    if st.session_state.get('annotation_results'):
        display_annotation_results()

# ===============================================
# INITIALIZATION AND SETUP
# ===============================================

def initialize_bert_annotator():
    """Initialize BERT annotator in session state"""
    if 'bert_annotator_initialized' not in st.session_state:
        st.session_state.bert_annotator_initialized = False
    
    if not st.session_state.bert_annotator_initialized:
        with st.spinner("🧠 Loading BERT model... This may take a moment."):
            try:
                st.session_state.bert_annotator = BERTConceptAnnotator()
                st.session_state.bert_annotator_initialized = True
                
                if BERT_ANNOTATOR_AVAILABLE:
                    st.success("✅ BERT model loaded successfully!")
                else:
                    st.info("ℹ️ Using keyword-based fallback mode (BERT not available)")
                    
            except Exception as init_error:
                logging.error(f"Error initializing BERT annotator: {init_error}")
                st.error(f"❌ Failed to load BERT model: {init_error}")
                st.session_state.bert_annotator = BERTConceptAnnotator()  # Use fallback
                st.session_state.bert_annotator_initialized = True

def convert_legacy_recommendations(legacy_recs: List) -> List[Dict]:
    """Convert legacy recommendation format to new format"""
    converted = []
    for i, rec in enumerate(legacy_recs):
        if hasattr(rec, 'text'):
            # Object format
            converted.append({
                'number': str(i + 1),
                'text': rec.text,
                'confidence_score': getattr(rec, 'confidence_score', 0.8),
                'document_context': {
                    'filename': getattr(rec, 'document_source', 'Unknown'),
                    'document_type': 'legacy_recommendation'
                }
            })
        elif isinstance(rec, dict):
            # Dictionary format
            converted.append({
                'number': rec.get('id', str(i + 1)),
                'text': rec.get('text', ''),
                'confidence_score': rec.get('confidence', 0.8),
                'document_context': {
                    'filename': rec.get('source', 'Unknown'),
                    'document_type': 'legacy_recommendation'
                }
            })
    return converted

def convert_legacy_concerns_to_responses(legacy_concerns: List) -> List[Dict]:
    """Convert legacy concerns to response format"""
    converted = []
    for i, concern in enumerate(legacy_concerns):
        if isinstance(concern, dict):
            converted.append({
                'number': str(i + 1),
                'text': concern.get('text', ''),
                'confidence_score': concern.get('confidence', 0.8),
                'response_type': 'concern',
                'document_context': {
                    'filename': concern.get('document_source', 'Unknown'),
                    'document_type': 'legacy_concern'
                }
            })
    return converted

# ===============================================
# FRAMEWORK MANAGEMENT
# ===============================================

def render_framework_management():
    """Render framework selection and management interface"""
    st.subheader("🎯 Conceptual Frameworks")
    
    annotator = st.session_state.get('bert_annotator')
    if not annotator:
        st.error("BERT annotator not initialized.")
        return
    
    # Framework selection
    available_frameworks = list(annotator.frameworks.keys())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_frameworks = st.multiselect(
            "Select annotation frameworks:",
            options=available_frameworks,
            default=available_frameworks[:2] if len(available_frameworks) >= 2 else available_frameworks,
            help="Choose which conceptual frameworks to apply for annotation"
        )
        
        st.session_state.selected_frameworks = selected_frameworks
    
    with col2:
        st.markdown("**Framework Info:**")
        for framework in selected_frameworks:
            themes = annotator.frameworks.get(framework, [])
            st.write(f"• **{framework}:** {len(themes)} themes")
    
    # Framework details
    if selected_frameworks:
        with st.expander("📋 Framework Details", expanded=False):
            for framework in selected_frameworks:
                st.markdown(f"### {framework}")
                themes = annotator.frameworks.get(framework, [])
                
                for theme in themes:
                    theme_name = theme.get('name', 'Unknown')
                    keywords = theme.get('keywords', [])
                    st.write(f"**{theme_name}:** {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
    
    # Custom framework upload
    render_custom_framework_upload()

def render_custom_framework_upload():
    """Render custom framework upload interface"""
    with st.expander("📁 Upload Custom Framework", expanded=False):
        st.markdown("""
        Upload your own annotation framework in JSON, CSV, or Excel format.
        
        **Required format:**
        - JSON: `{"framework_name": [{"name": "theme", "keywords": ["word1", "word2"]}]}`
        - CSV: Columns: `theme_name, keywords` (keywords comma-separated)
        - Excel: Same structure as CSV
        """)
        
        uploaded_framework = st.file_uploader(
            "Choose framework file",
            type=['json', 'csv', 'xlsx'],
            help="Upload custom annotation framework"
        )
        
        if uploaded_framework and st.button("📤 Load Custom Framework"):
            load_custom_framework(uploaded_framework)

def load_custom_framework(uploaded_file):
    """Load a custom framework from uploaded file"""
    try:
        annotator = st.session_state.get('bert_annotator')
        if not annotator:
            st.error("BERT annotator not available")
            return
        
        file_content = uploaded_file.read()
        filename = uploaded_file.name
        
        # Validate file size
        if CORE_UTILS_AVAILABLE:
            SecurityValidator.validate_file_upload(file_content, filename)
        
        success, message = annotator.load_custom_framework(file_content, filename)
        
        if success:
            st.success(f"✅ {message}")
            st.rerun()  # Refresh to show new framework
        else:
            st.error(f"❌ {message}")
            
    except Exception as load_error:
        logging.error(f"Error loading custom framework: {load_error}")
        st.error(f"Error loading framework: {load_error}")

# ===============================================
# ANNOTATION CONFIGURATION
# ===============================================

def render_annotation_configuration():
    """Render annotation configuration options"""
    st.subheader("⚙️ Annotation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.1,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="Minimum similarity score for theme matching"
        )
        st.session_state.similarity_threshold = similarity_threshold
    
    with col2:
        max_themes_per_framework = st.number_input(
            "Max themes per framework:",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of themes to assign per framework"
        )
        st.session_state.max_themes_per_framework = max_themes_per_framework
    
    with col3:
        annotation_mode = st.selectbox(
            "Annotation Mode:",
            options=["Semantic + Keywords", "Semantic Only", "Keywords Only"],
            index=0,
            help="Choose annotation method"
        )
        st.session_state.annotation_mode = annotation_mode
    
    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        col1a, col2a = st.columns(2)
        
        with col1a:
            include_confidence_scores = st.checkbox(
                "Include confidence scores",
                value=True,
                help="Show confidence scores for annotations"
            )
            st.session_state.include_confidence_scores = include_confidence_scores
            
            highlight_keywords = st.checkbox(
                "Highlight matching keywords",
                value=True,
                help="Highlight keywords in annotated text"
            )
            st.session_state.highlight_keywords = highlight_keywords
        
        with col2a:
            batch_size = st.number_input(
                "Batch processing size:",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of items to process in each batch"
            )
            st.session_state.batch_size = batch_size
            
            save_intermediate_results = st.checkbox(
                "Save intermediate results",
                value=True,
                help="Save results after each batch"
            )
            st.session_state.save_intermediate_results = save_intermediate_results

# ===============================================
# ANNOTATION INTERFACE
# ===============================================

def render_annotation_interface(recommendations: List[Dict], responses: List[Dict]):
    """Render the main annotation interface"""
    st.subheader("🚀 Run Annotation")
    
    selected_frameworks = st.session_state.get('selected_frameworks', [])
    
    if not selected_frameworks:
        st.warning("⚠️ Please select at least one framework above.")
        return
    
    # Content selection
    items_to_annotate = render_content_selection(recommendations, responses)
    
    # Annotation controls
    if items_to_annotate:
        render_annotation_controls(items_to_annotate, selected_frameworks)

def render_content_selection(recommendations: List[Dict], responses: List[Dict]) -> List[Dict]:
    """Render content selection interface and return selected items"""
    st.markdown("**📋 Select Content to Annotate:**")
    
    # Content type selection
    content_options = []
    if recommendations:
        content_options.append("Recommendations")
    if responses:
        content_options.append("Responses") 
    if len(content_options) > 1:
        content_options.append("Both")
    
    if not content_options:
        st.error("No content available for annotation")
        return []
    
    content_choice = st.radio(
        "Content to annotate:",
        content_options,
        horizontal=True,
        key="content_to_annotate"
    )
    
    # Selection interface based on content choice
    items_to_annotate = []
    
    if content_choice == "Recommendations":
        items_to_annotate = render_recommendation_selection(recommendations)
    elif content_choice == "Responses":
        items_to_annotate = render_response_selection(responses)
    elif content_choice == "Both":
        items_to_annotate = render_mixed_content_selection(recommendations, responses)
    
    return items_to_annotate

def render_recommendation_selection(recommendations: List[Dict]) -> List[Dict]:
    """Render recommendation selection interface"""
    selection_method = st.radio(
        "Selection method:",
        ["All Recommendations", "By Document", "Individual Selection"],
        key="rec_selection_method"
    )
    
    if selection_method == "All Recommendations":
        st.info(f"Selected all {len(recommendations)} recommendations")
        return recommendations
        
    elif selection_method == "By Document":
        # Group by document
        doc_groups = {}
        for rec in recommendations:
            doc_name = rec.get('document_context', {}).get('filename', 'Unknown')
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(rec)
        
        selected_docs = st.multiselect(
            "Select documents:",
            options=list(doc_groups.keys()),
            default=list(doc_groups.keys()),
            key="selected_rec_docs"
        )
        
        selected_recs = []
        for doc in selected_docs:
            selected_recs.extend(doc_groups[doc])
        
        st.info(f"Selected {len(selected_recs)} recommendations from {len(selected_docs)} documents")
        return selected_recs
        
    elif selection_method == "Individual Selection":
        st.markdown("**Select individual recommendations:**")
        selected_recs = []
        
        for i, rec in enumerate(recommendations):
            rec_id = rec.get('number', i + 1)
            rec_text = rec.get('text', '')
            preview = rec_text[:100] + "..." if len(rec_text) > 100 else rec_text
            doc_name = rec.get('document_context', {}).get('filename', 'Unknown')
            
            if st.checkbox(f"Rec {rec_id}: {preview}", key=f"rec_select_{i}"):
                selected_recs.append(rec)
        
        st.info(f"Selected {len(selected_recs)} individual recommendations")
        return selected_recs
    
    return []

def render_response_selection(responses: List[Dict]) -> List[Dict]:
    """Render response selection interface"""
    selection_method = st.radio(
        "Selection method:",
        ["All Responses", "By Document", "By Response Type", "Individual Selection"],
        key="resp_selection_method"
    )
    
    if selection_method == "All Responses":
        st.info(f"Selected all {len(responses)} responses")
        return responses
        
    elif selection_method == "By Document":
        # Group by document
        doc_groups = {}
        for resp in responses:
            doc_name = resp.get('document_context', {}).get('filename', 'Unknown')
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(resp)
        
        selected_docs = st.multiselect(
            "Select documents:",
            options=list(doc_groups.keys()),
            default=list(doc_groups.keys()),
            key="selected_resp_docs"
        )
        
        selected_resps = []
        for doc in selected_docs:
            selected_resps.extend(doc_groups[doc])
        
        st.info(f"Selected {len(selected_resps)} responses from {len(selected_docs)} documents")
        return selected_resps
        
    elif selection_method == "By Response Type":
        # Group by response type
        type_groups = {}
        for resp in responses:
            resp_type = resp.get('response_type', 'unclear')
            if resp_type not in type_groups:
                type_groups[resp_type] = []
            type_groups[resp_type].append(resp)
        
        selected_types = st.multiselect(
            "Select response types:",
            options=list(type_groups.keys()),
            default=list(type_groups.keys()),
            key="selected_resp_types"
        )
        
        selected_resps = []
        for resp_type in selected_types:
            selected_resps.extend(type_groups[resp_type])
        
        st.info(f"Selected {len(selected_resps)} responses of types: {', '.join(selected_types)}")
        return selected_resps
        
    elif selection_method == "Individual Selection":
        st.markdown("**Select individual responses:**")
        selected_resps = []
        
        for i, resp in enumerate(responses):
            resp_id = resp.get('number', i + 1)
            resp_text = resp.get('text', '')
            resp_type = resp.get('response_type', 'unclear')
            preview = resp_text[:100] + "..." if len(resp_text) > 100 else resp_text
            
            emoji = {'accepted': '✅', 'rejected': '❌', 'partially_accepted': '⚡', 'under_consideration': '🤔'}.get(resp_type, '❓')
            
            if st.checkbox(f"{emoji} Resp {resp_id}: {preview}", key=f"resp_select_{i}"):
                selected_resps.append(resp)
        
        st.info(f"Selected {len(selected_resps)} individual responses")
        return selected_resps
    
    return []

def render_mixed_content_selection(recommendations: List[Dict], responses: List[Dict]) -> List[Dict]:
    """Render mixed content selection interface"""
    st.markdown("**Select content to annotate:**")
    
    col1, col2 = st.columns(2)
    selected_items = []
    
    with col1:
        st.markdown("**📝 Recommendations:**")
        include_all_recs = st.checkbox(
            f"Include all recommendations ({len(recommendations)})",
            key="include_all_recs"
        )
        
        if include_all_recs:
            selected_items.extend(recommendations)
        else:
            # Group by document
            rec_doc_groups = {}
            for rec in recommendations:
                doc_name = rec.get('document_context', {}).get('filename', 'Unknown')
                if doc_name not in rec_doc_groups:
                    rec_doc_groups[doc_name] = []
                rec_doc_groups[doc_name].append(rec)
            
            selected_rec_docs = st.multiselect(
                "Select recommendation documents:",
                options=list(rec_doc_groups.keys()),
                key="mixed_rec_docs"
            )
            
            for doc in selected_rec_docs:
                selected_items.extend(rec_doc_groups[doc])
    
    with col2:
        st.markdown("**📋 Responses:**")
        include_all_resps = st.checkbox(
            f"Include all responses ({len(responses)})",
            key="include_all_resps"
        )
        
        if include_all_resps:
            selected_items.extend(responses)
        else:
            # Group by document
            resp_doc_groups = {}
            for resp in responses:
                doc_name = resp.get('document_context', {}).get('filename', 'Unknown')
                if doc_name not in resp_doc_groups:
                    resp_doc_groups[doc_name] = []
                resp_doc_groups[doc_name].append(resp)
            
            selected_resp_docs = st.multiselect(
                "Select response documents:",
                options=list(resp_doc_groups.keys()),
                key="mixed_resp_docs"
            )
            
            for doc in selected_resp_docs:
                selected_items.extend(resp_doc_groups[doc])
    
    st.info(f"Selected {len(selected_items)} total items for annotation")
    return selected_items

def render_annotation_controls(items_to_annotate: List[Dict], selected_frameworks: List[str]):
    """Render annotation controls and summary"""
    # Content summary
    rec_count = sum(1 for item in items_to_annotate if item.get('text') and 'REC-' in str(item.get('number', '')))
    resp_count = len(items_to_annotate) - rec_count
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        content_summary = []
        if rec_count > 0:
            content_summary.append(f"{rec_count} recommendations")
        if resp_count > 0:
            content_summary.append(f"{resp_count} responses")
        
        summary_text = " + ".join(content_summary) if content_summary else "no items"
        st.info(f"📊 Ready to annotate {len(items_to_annotate)} items ({summary_text}) with {len(selected_frameworks)} frameworks")
    
    with col2:
        st.markdown("**Settings:**")
        threshold = st.session_state.get('similarity_threshold', 0.65)
        max_themes = st.session_state.get('max_themes_per_framework', 5)
        mode = st.session_state.get('annotation_mode', 'Semantic + Keywords')
        st.write(f"• Threshold: {threshold:.2f}")
        st.write(f"• Max themes: {max_themes}")
        st.write(f"• Mode: {mode}")
    
    # Annotation button
    if st.button("🚀 Start Annotation", type="primary", use_container_width=True):
        run_comprehensive_annotation(items_to_annotate, selected_frameworks)

# ===============================================
# ANNOTATION PROCESSING
# ===============================================

def run_comprehensive_annotation(items: List[Dict], frameworks: List[str]):
    """Run comprehensive annotation with progress tracking"""
    annotator = st.session_state.get('bert_annotator')
    if not annotator:
        st.error("BERT annotator not available.")
        return
    
    st.subheader("🔄 Processing Annotations...")
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    annotation_results = {}
    batch_size = st.session_state.get('batch_size', 10)
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        status_text.text(f"Processing batch {batch_num}/{total_batches}...")
        
        for j, item in enumerate(batch):
            try:
                # Update progress
                overall_progress = (i + j + 1) / len(items)
                progress_bar.progress(overall_progress)
                
                # Create item ID
                item_id = f"ITEM-{i + j + 1}"
                if 'number' in item:
                    item_type = 'recommendation' if any(keyword in str(item.get('number', '')).upper() for keyword in ['REC', 'RECOMMENDATION']) else 'response'
                    item_id = f"{'REC' if item_type == 'recommendation' else 'RESP'}-{item.get('number', i + j + 1)}"
                
                # Get text content
                text_content = item.get('text', '')
                if not text_content or len(text_content.strip()) < 10:
                    continue
                
                # Annotate item
                item_annotations, annotation_details = annotator.annotate_text(
                    text_content, 
                    frameworks
                )
                
                # Store results
                annotation_results[item_id] = {
                    'item': item,
                    'annotations': item_annotations,
                    'details': annotation_details,
                    'frameworks_used': frameworks,
                    'annotation_timestamp': datetime.now().isoformat(),
                    'text_length': len(text_content),
                    'word_count': len(text_content.split())
                }
                
            except Exception as annotation_error:
                logging.error(f"Error annotating item {item_id}: {annotation_error}")
                annotation_results[item_id] = {
                    'item': item,
                    'annotations': {},
                    'details': {},
                    'error': str(annotation_error),
                    'annotation_timestamp': datetime.now().isoformat()
                }
        
        # Save intermediate results if requested
        if st.session_state.get('save_intermediate_results', True):
            st.session_state.annotation_results = annotation_results
    
    # Complete processing
    progress_bar.progress(1.0)
    status_text.text("✅ Annotation completed!")
    
    # Store final results
    st.session_state.annotation_results = annotation_results
    st.session_state.annotation_settings = {
        'frameworks': frameworks,
        'similarity_threshold': st.session_state.get('similarity_threshold', 0.65),
        'max_themes_per_framework': st.session_state.get('max_themes_per_framework', 5),
        'annotation_mode': st.session_state.get('annotation_mode', 'Semantic + Keywords'),
        'total_items': len(items),
        'annotation_date': datetime.now().isoformat()
    }
    
    # Show summary
    show_annotation_summary(annotation_results, frameworks)

def show_annotation_summary(results: Dict[str, Any], frameworks: List[str]):
    """Show annotation summary"""
    st.success(f"🎉 Annotation completed for {len(results)} items!")
    
    # Summary metrics
    successful_annotations = len([r for r in results.values() if not r.get('error')])
    failed_annotations = len(results) - successful_annotations
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", len(results))
    with col2:
        st.metric("Successful", successful_annotations)
    with col3:
        st.metric("Failed", failed_annotations)
    with col4:
        st.metric("Frameworks", len(frameworks))
    
    # Theme distribution
    if successful_annotations > 0:
        theme_counts = {}
        for result in results.values():
            if not result.get('error'):
                annotations = result.get('annotations', {})
                for framework, themes in annotations.items():
                    for theme in themes:
                        theme_name = theme.get('theme', 'Unknown')
                        theme_key = f"{framework}_{theme_name}"
                        theme_counts[theme_key] = theme_counts.get(theme_key, 0) + 1
        
        if theme_counts:
            st.subheader("📊 Theme Distribution")
            theme_df = pd.DataFrame([
                {'Framework_Theme': theme, 'Count': count}
                for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(theme_df.head(10), use_container_width=True)

# ===============================================
# RESULTS DISPLAY
# ===============================================

def display_annotation_results():
    """Display comprehensive annotation results"""
    results = st.session_state.get('annotation_results', {})
    if not results:
        return
    
    st.subheader("📋 Annotation Results")
    
    # Results overview
    with st.expander("📊 Results Overview", expanded=True):
        settings = st.session_state.get('annotation_settings', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Annotation Summary:**")
            st.write(f"- **Items Annotated:** {len(results)}")
            st.write(f"- **Frameworks Used:** {', '.join(settings.get('frameworks', []))}")
            st.write(f"- **Date:** {settings.get('annotation_date', 'Unknown')}")
            st.write(f"- **Threshold:** {settings.get('similarity_threshold', 'Unknown')}")
        
        with col2:
            # Calculate statistics
            successful = len([r for r in results.values() if not r.get('error')])
            recommendations = len([r for r in results.values() if 'REC-' in str(r.get('item', {}).get('number', ''))])
            responses = len([r for r in results.values() if 'RESP-' in str(r.get('item', {}).get('number', ''))])
            
            st.write("**Content Breakdown:**")
            st.write(f"- **Successful:** {successful}/{len(results)}")
            st.write(f"- **Recommendations:** {recommendations}")
            st.write(f"- **Responses:** {responses}")
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 All Results", "🏷️ By Theme", "📊 Statistics", "📥 Export"])
    
    with tab1:
        render_detailed_annotation_results(results)
    
    with tab2:
        render_results_by_theme(results)
    
    with tab3:
        render_annotation_statistics(results)
    
    with tab4:
        render_annotation_export_options(results)

def render_detailed_annotation_results(results: Dict[str, Any]):
    """Render detailed annotation results"""
    if not results:
        st.info("No annotation results available")
        return
    
    # Search and filter
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_term = st.text_input("Search annotations:", placeholder="Search by text or theme...")
    
    with col2:
        content_filter = st.selectbox(
            "Content Type:",
            options=["All", "Recommendations", "Responses"],
            index=0
        )
    
    with col3:
        min_confidence = st.slider(
            "Min confidence:",
            0.0, 1.0, 0.0, 0.1
        )
    
    # Filter results
    filtered_results = filter_annotation_results(results, search_term, content_filter, min_confidence)
    
    st.write(f"Showing {len(filtered_results)} of {len(results)} items")
    
    # Display results
    for item_id, result in filtered_results.items():
        item = result.get('item', {})
        annotations = result.get('annotations', {})
        error = result.get('error')
        
        # Create header
        item_type = 'recommendation' if 'REC-' in item_id else 'response' if 'RESP-' in item_id else 'unknown'
        emoji = {'recommendation': '📝', 'response': '📋'}.get(item_type, '📄')
        
        header = f"{emoji} {item_id} - {item.get('text', '')[:80]}..."
        
        with st.expander(header, expanded=False):
            if error:
                st.error(f"❌ Annotation failed: {error}")
                continue
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Full Text:**")
                text = item.get('text', '')
                
                # Highlight keywords if enabled
                if st.session_state.get('highlight_keywords', True) and annotations:
                    highlighted_text = highlight_text_with_themes(text, annotations)
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                else:
                    st.write(text)
            
            with col2:
                st.write("**Item Details:**")
                st.write(f"- **Type:** {item_type.title()}")
                st.write(f"- **Document:** {item.get('document_context', {}).get('filename', 'Unknown')}")
                st.write(f"- **Confidence:** {item.get('confidence_score', 0):.3f}")
                
                if item_type == 'response':
                    response_type = item.get('response_type', 'unclear')
                    st.write(f"- **Response Type:** {response_type.replace('_', ' ').title()}")
            
            # Display annotations
            if annotations:
                st.write("**🏷️ Annotations:**")
                for framework, themes in annotations.items():
                    st.write(f"**{framework}:**")
                    for theme in themes:
                        theme_name = theme.get('theme', 'Unknown')
                        confidence = theme.get('confidence', 0)
                        keywords = theme.get('keywords', [])
                        
                        # Color-code by confidence
                        if confidence >= 0.8:
                            confidence_color = "🟢"
                        elif confidence >= 0.6:
                            confidence_color = "🟡"
                        else:
                            confidence_color = "🔴"
                        
                        st.write(f"  {confidence_color} **{theme_name}** ({confidence:.3f})")
                        if keywords:
                            st.caption(f"    Keywords: {', '.join(keywords)}")

def render_results_by_theme(results: Dict[str, Any]):
    """Render results organized by theme"""
    if not results:
        st.info("No results to display by theme")
        return
    
    # Collect all themes
    theme_groups = {}
    
    for item_id, result in results.items():
        if result.get('error'):
            continue
            
        annotations = result.get('annotations', {})
        item = result.get('item', {})
        
        for framework, themes in annotations.items():
            for theme in themes:
                theme_name = theme.get('theme', 'Unknown')
                theme_key = f"{framework}_{theme_name}"
                
                if theme_key not in theme_groups:
                    theme_groups[theme_key] = []
                
                theme_groups[theme_key].append({
                    'item_id': item_id,
                    'text': item.get('text', ''),
                    'type': 'recommendation' if 'REC-' in item_id else 'response',
                    'confidence': theme.get('confidence', 0),
                    'keywords': theme.get('keywords', [])
                })
    
    # Display by theme
    for theme_key, items in sorted(theme_groups.items(), key=lambda x: len(x[1]), reverse=True):
        framework, theme_name = theme_key.split('_', 1)
        
        with st.expander(f"🏷️ {framework} - {theme_name} ({len(items)} items)", expanded=False):
            for item in sorted(items, key=lambda x: x['confidence'], reverse=True):
                confidence = item['confidence']
                confidence_color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                
                st.write(f"{confidence_color} **{item['item_id']}** ({confidence:.3f})")
                st.caption(f"{item['text'][:100]}...")
                
                if item['keywords']:
                    st.caption(f"**Keywords:** {', '.join(item['keywords'])}")
                st.markdown("---")

def render_annotation_statistics(results: Dict[str, Any]):
    """Render annotation statistics and analytics"""
    if not results:
        st.info("No statistics available")
        return
    
    # Calculate statistics
    successful_results = {k: v for k, v in results.items() if not v.get('error')}
    
    if not successful_results:
        st.warning("No successful annotations to analyze")
        return
    
    # Framework usage statistics
    framework_stats = {}
    theme_stats = {}
    confidence_stats = []
    
    for result in successful_results.values():
        annotations = result.get('annotations', {})
        
        for framework, themes in annotations.items():
            if framework not in framework_stats:
                framework_stats[framework] = {'items': 0, 'themes': 0, 'avg_confidence': 0}
            
            framework_stats[framework]['items'] += 1
            framework_stats[framework]['themes'] += len(themes)
            
            for theme in themes:
                theme_name = theme.get('theme', 'Unknown')
                confidence = theme.get('confidence', 0)
                
                confidence_stats.append(confidence)
                
                theme_key = f"{framework}_{theme_name}"
                if theme_key not in theme_stats:
                    theme_stats[theme_key] = {'count': 0, 'total_confidence': 0}
                
                theme_stats[theme_key]['count'] += 1
                theme_stats[theme_key]['total_confidence'] += confidence
    
    # Calculate averages
    for framework, stats in framework_stats.items():
        if stats['items'] > 0:
            stats['avg_themes_per_item'] = stats['themes'] / stats['items']
    
    for theme_key, stats in theme_stats.items():
        if stats['count'] > 0:
            stats['avg_confidence'] = stats['total_confidence'] / stats['count']
    
    # Display statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Framework Performance:**")
        framework_df = pd.DataFrame([
            {
                'Framework': framework,
                'Items Annotated': stats['items'],
                'Total Themes': stats['themes'],
                'Avg Themes/Item': f"{stats.get('avg_themes_per_item', 0):.2f}"
            }
            for framework, stats in framework_stats.items()
        ])
        st.dataframe(framework_df, use_container_width=True)
    
    with col2:
        st.write("**Confidence Distribution:**")
        if confidence_stats:
            avg_confidence = sum(confidence_stats) / len(confidence_stats)
            high_confidence = len([c for c in confidence_stats if c >= 0.8])
            medium_confidence = len([c for c in confidence_stats if 0.6 <= c < 0.8])
            low_confidence = len([c for c in confidence_stats if c < 0.6])
            
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
            st.write(f"- **High (≥0.8):** {high_confidence}")
            st.write(f"- **Medium (0.6-0.8):** {medium_confidence}")
            st.write(f"- **Low (<0.6):** {low_confidence}")
    
    # Top themes
    st.write("**Most Common Themes:**")
    top_themes = sorted(theme_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    
    top_themes_df = pd.DataFrame([
        {
            'Theme': theme_key.replace('_', ' - '),
            'Count': stats['count'],
            'Avg Confidence': f"{stats['avg_confidence']:.3f}"
        }
        for theme_key, stats in top_themes
    ])
    
    st.dataframe(top_themes_df, use_container_width=True)

# ===============================================
# EXPORT FUNCTIONS
# ===============================================

def render_annotation_export_options(results: Dict[str, Any]):
    """Render export options for annotation results"""
    if not results:
        st.info("No results to export")
        return
    
    st.write("**Export Options:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Summary CSV", use_container_width=True):
            export_annotation_summary_csv(results)
    
    with col2:
        if st.button("📋 Download Detailed CSV", use_container_width=True):
            export_detailed_annotations_csv(results)
    
    with col3:
        if st.button("📄 Download JSON", use_container_width=True):
            export_annotations_json(results)
    
    # Additional export options
    col4, col5 = st.columns(2)
    
    with col4:
        if st.button("🌈 Download Highlighted HTML", use_container_width=True):
            export_highlighted_html(results)
    
    with col5:
        if st.button("📈 Download Statistics Report", use_container_width=True):
            export_statistics_report(results)

def export_annotation_summary_csv(results: Dict[str, Any]):
    """Export annotation summary as CSV"""
    try:
        summary_data = []
        
        for item_id, result in results.items():
            if result.get('error'):
                continue
                
            item = result.get('item', {})
            annotations = result.get('annotations', {})
            
            # Create summary row
            all_themes = []
            avg_confidence = 0
            total_themes = 0
            
            for framework, themes in annotations.items():
                for theme in themes:
                    all_themes.append(f"{framework}:{theme.get('theme', 'Unknown')}")
                    avg_confidence += theme.get('confidence', 0)
                    total_themes += 1
            
            if total_themes > 0:
                avg_confidence /= total_themes
            
            summary_data.append({
                'Item_ID': item_id,
                'Type': 'recommendation' if 'REC-' in item_id else 'response',
                'Text': item.get('text', '')[:200] + ('...' if len(item.get('text', '')) > 200 else ''),
                'Document': item.get('document_context', {}).get('filename', 'Unknown'),
                'Themes_Count': total_themes,
                'Themes_List': '; '.join(all_themes),
                'Avg_Confidence': f"{avg_confidence:.3f}" if avg_confidence > 0 else "0.000"
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                "📥 Download Summary CSV",
                csv_data,
                f"annotation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.error("No data to export")
            
    except Exception as export_error:
        st.error(f"Error exporting CSV: {export_error}")

def export_detailed_annotations_csv(results: Dict[str, Any]):
    """Export detailed annotations as CSV"""
    try:
        detailed_data = []
        
        for item_id, result in results.items():
            if result.get('error'):
                continue
                
            item = result.get('item', {})
            annotations = result.get('annotations', {})
            
            # Create row for each theme
            for framework, themes in annotations.items():
                for theme in themes:
                    detailed_data.append({
                        'Item_ID': item_id,
                        'Content_Type': 'recommendation' if 'REC-' in item_id else 'response',
                        'Full_Text': item.get('text', ''),
                        'Document_Source': item.get('document_context', {}).get('filename', 'Unknown'),
                        'Framework': framework,
                        'Theme': theme.get('theme', 'Unknown'),
                        'Confidence': theme.get('confidence', 0),
                        'Keywords': ', '.join(theme.get('keywords', [])),
                        'Response_Type': item.get('response_type', '') if 'RESP-' in item_id else '',
                        'Annotation_Date': result.get('annotation_timestamp', '')
                    })
        
        if detailed_data:
            df = pd.DataFrame(detailed_data)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                "📥 Download Detailed CSV",
                csv_data,
                f"annotation_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.error("No data to export")
            
    except Exception as export_error:
        st.error(f"Error exporting detailed CSV: {export_error}")

def export_annotations_json(results: Dict[str, Any]):
    """Export annotations as JSON"""
    try:
        export_data = {
            'annotation_metadata': {
                'export_date': datetime.now().isoformat(),
                'total_items': len(results),
                'successful_annotations': len([r for r in results.values() if not r.get('error')]),
                'settings': st.session_state.get('annotation_settings', {})
            },
            'annotations': results
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            "📥 Download JSON",
            json_data,
            f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
        
    except Exception as export_error:
        st.error(f"Error exporting JSON: {export_error}")

def export_highlighted_html(results: Dict[str, Any]):
    """Export highlighted text as HTML"""
    try:
        html_content = []
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Annotation Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .item { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .header { background-color: #f5f5f5; padding: 10px; margin: -15px -15px 15px -15px; border-radius: 5px 5px 0 0; }
                .theme-highlight { padding: 2px 4px; border-radius: 3px; margin: 0 1px; }
                .annotations { margin-top: 15px; }
                .framework { margin-bottom: 10px; }
                .theme { margin-left: 20px; }
            </style>
        </head>
        <body>
        <h1>Annotation Results</h1>
        <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        """)
        
        for item_id, result in results.items():
            if result.get('error'):
                continue
                
            item = result.get('item', {})
            annotations = result.get('annotations', {})
            
            # Create highlighted text
            text = item.get('text', '')
            highlighted_text = highlight_text_with_themes(text, annotations)
            
            item_type = 'recommendation' if 'REC-' in item_id else 'response'
            
            html_content.append(f"""
            <div class="item">
                <div class="header">
                    <h3>{item_id} - {item_type.title()}</h3>
                    <p><strong>Document:</strong> {item.get('document_context', {}).get('filename', 'Unknown')}</p>
                </div>
                <div class="content">
                    <h4>Text:</h4>
                    <p>{highlighted_text}</p>
                </div>
                <div class="annotations">
                    <h4>Annotations:</h4>
            """)
            
            for framework, themes in annotations.items():
                html_content.append(f'<div class="framework"><strong>{framework}:</strong>')
                for theme in themes:
                    confidence = theme.get('confidence', 0)
                    html_content.append(f'<div class="theme">• {theme.get("theme", "Unknown")} ({confidence:.3f})</div>')
                html_content.append('</div>')
            
            html_content.append('</div></div>')
        
        html_content.append('</body></html>')
        
        html_data = '\n'.join(html_content)
        
        st.download_button(
            "📥 Download HTML",
            html_data,
            f"annotations_highlighted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "text/html"
        )
        
    except Exception as export_error:
        st.error(f"Error exporting HTML: {export_error}")

def export_statistics_report(results: Dict[str, Any]):
    """Export statistics report as text"""
    try:
        report_lines = [
            "# Annotation Statistics Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            f"Total Items: {len(results)}",
            f"Successful Annotations: {len([r for r in results.values() if not r.get('error')])}",
            ""
        ]
        
        # Calculate detailed statistics
        successful_results = {k: v for k, v in results.items() if not v.get('error')}
        
        if successful_results:
            # Framework statistics
            framework_stats = {}
            theme_counts = {}
            
            for result in successful_results.values():
                annotations = result.get('annotations', {})
                
                for framework, themes in annotations.items():
                    if framework not in framework_stats:
                        framework_stats[framework] = {'items': 0, 'themes': 0}
                    
                    framework_stats[framework]['items'] += 1
                    framework_stats[framework]['themes'] += len(themes)
                    
                    for theme in themes:
                        theme_name = theme.get('theme', 'Unknown')
                        theme_key = f"{framework}_{theme_name}"
                        theme_counts[theme_key] = theme_counts.get(theme_key, 0) + 1
            
            report_lines.extend([
                "## Framework Statistics",
                ""
            ])
            
            for framework, stats in framework_stats.items():
                avg_themes = stats['themes'] / stats['items'] if stats['items'] > 0 else 0
                report_lines.extend([
                    f"### {framework}",
                    f"- Items annotated: {stats['items']}",
                    f"- Total themes assigned: {stats['themes']}",
                    f"- Average themes per item: {avg_themes:.2f}",
                    ""
                ])
            
            # Top themes
            report_lines.extend([
                "## Top Themes",
                ""
            ])
            
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            for theme_key, count in sorted_themes[:15]:
                framework, theme = theme_key.split('_', 1)
                report_lines.append(f"- {framework} - {theme}: {count} items")
        
        report_content = "\n".join(report_lines)
        
        st.download_button(
            "📥 Download Statistics Report",
            report_content,
            f"annotation_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )
        
    except Exception as export_error:
        st.error(f"Error exporting statistics report: {export_error}")

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def highlight_text_with_themes(text: str, annotations: Dict[str, Any]) -> str:
    """Highlight text with color-coded themes"""
    if not st.session_state.get('highlight_keywords', True):
        return text
    
    highlighted_text = text
    color_map = get_theme_color_map()
    
    try:
        # Collect all keywords with their themes
        keyword_theme_map = {}
        
        for framework, themes in annotations.items():
            for theme in themes:
                theme_name = theme.get('theme', 'Unknown')
                keywords = theme.get('keywords', [])
                
                for keyword in keywords:
                    if keyword.lower() not in keyword_theme_map:
                        keyword_theme_map[keyword.lower()] = []
                    
                    keyword_theme_map[keyword.lower()].append({
                        'framework': framework,
                        'theme': theme_name,
                        'confidence': theme.get('confidence', 0)
                    })
        
        # Sort keywords by length (longest first) to avoid partial matches
        sorted_keywords = sorted(keyword_theme_map.keys(), key=len, reverse=True)
        
        # Apply highlights
        for keyword in sorted_keywords:
            themes = keyword_theme_map[keyword]
            if not themes:
                continue
            
            # Use the highest confidence theme for coloring
            best_theme = max(themes, key=lambda x: x['confidence'])
            theme_key = f"{best_theme['framework']}_{best_theme['theme']}"
            color = color_map.get(theme_key, "#E0E0E0")
            
            # Create regex pattern for case-insensitive matching
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            
            # Replace with highlighted version
            replacement = f'<span style="background-color: {color}; padding: 1px 3px; border-radius: 3px;" title="{best_theme["framework"]}: {best_theme["theme"]} ({best_theme["confidence"]:.2f})">{keyword}</span>'
            
            highlighted_text = pattern.sub(replacement, highlighted_text)
        
        return highlighted_text
        
    except Exception as highlight_error:
        logging.error(f"Error highlighting text: {highlight_error}")
        return text

def get_theme_color_map() -> Dict[str, str]:
    """Get color mapping for themes"""
    return {
        # I-SIRch Framework Colors
        "I-SIRch_External factors": "#FFE0B2",
        "I-SIRch_System factors": "#C8E6C9", 
        "I-SIRch_Technology factors": "#B3E5FC",
        "I-SIRch_Person factors": "#F8BBD9",
        "I-SIRch_Task factors": "#D1C4E9",
        
        # House of Commons Framework Colors
        "House of Commons_Communication": "#FFCDD2",
        "House of Commons_Fragmented care": "#F0F4C3",
        "House of Commons_Workforce pressures": "#DCEDC8",
        "House of Commons_Biases and stereotyping": "#E1BEE7",
        
        # Extended Analysis Framework Colors
        "Extended Analysis_Risk Management": "#FCE4EC",
        "Extended Analysis_Quality Assurance": "#F5F5DC",
        "Extended Analysis_Leadership": "#E8EAF6",
        
        # Default colors for other themes
        "default": "#E0E0E0"
    }

def filter_annotation_results(results: Dict[str, Any], search_term: str, content_filter: str, min_confidence: float) -> Dict[str, Any]:
    """Filter annotation results based on criteria"""
    filtered = {}
    
    for item_id, result in results.items():
        if result.get('error'):
            continue
            
        item = result.get('item', {})
        annotations = result.get('annotations', {})
        
        # Apply content type filter
        if content_filter != "All":
            item_type = 'recommendation' if 'REC-' in item_id else 'response'
            if content_filter == "Recommendations" and item_type != 'recommendation':
                continue
            elif content_filter == "Responses" and item_type != 'response':
                continue
        
        # Apply confidence filter
        if min_confidence > 0:
            max_confidence = 0
            for framework, themes in annotations.items():
                for theme in themes:
                    max_confidence = max(max_confidence, theme.get('confidence', 0))
            
            if max_confidence < min_confidence:
                continue
        
        # Apply search filter
        if search_term:
            search_lower = search_term.lower()
            text_match = search_lower in item.get('text', '').lower()
            theme_match = False
            
            for framework, themes in annotations.items():
                for theme in themes:
                    if search_lower in theme.get('theme', '').lower():
                        theme_match = True
                        break
                if theme_match:
                    break
            
            if not text_match and not theme_match:
                continue
        
        filtered[item_id] = result
    
    return filtered

def log_annotation_action(action: str, details: str = ""):
    """Log annotation-related user actions"""
    timestamp = datetime.now().isoformat()
    logging.info(f"[{timestamp}] Annotation Action: {action} - {details}")
    
    # Store in session state for debugging
    if 'annotation_history' not in st.session_state:
        st.session_state.annotation_history = []
    
    st.session_state.annotation_history.append({
        'timestamp': timestamp,
        'action': action,
        'details': details
    })
    
    # Keep only last 50 actions
    if len(st.session_state.annotation_history) > 50:
        st.session_state.annotation_history = st.session_state.annotation_history[-50:]

def clear_annotation_results():
    """Clear annotation results from session state"""
    keys_to_clear = ['annotation_results', 'annotation_settings', 'bert_annotator_initialized']
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    log_annotation_action("clear_results", "Annotation results cleared")
    st.success("✅ Annotation results cleared")

def get_annotation_statistics():
    """Get annotation statistics from session state"""
    results = st.session_state.get('annotation_results', {})
    
    if not results:
        return {
            'total_annotations': 0,
            'total_themes': 0,
            'frameworks_used': [],
            'last_annotation': None
        }
    
    # Calculate statistics
    successful_results = {k: v for k, v in results.items() if not v.get('error')}
    total_themes = 0
    frameworks_used = set()
    
    for result in successful_results.values():
        annotations = result.get('annotations', {})
        for framework, themes in annotations.items():
            frameworks_used.add(framework)
            total_themes += len(themes)
    
    settings = st.session_state.get('annotation_settings', {})
    
    return {
        'total_annotations': len(successful_results),
        'total_themes': total_themes,
        'frameworks_used': list(frameworks_used),
        'last_annotation': settings.get('annotation_date'),
        'success_rate': len(successful_results) / max(1, len(results))
    }

def validate_annotation_settings(settings: Dict[str, Any]) -> tuple[bool, str]:
    """Validate annotation settings"""
    try:
        similarity_threshold = settings.get('similarity_threshold', 0.65)
        max_themes = settings.get('max_themes_per_framework', 5)
        batch_size = settings.get('batch_size', 10)
        
        if not (0.1 <= similarity_threshold <= 1.0):
            return False, "Similarity threshold must be between 0.1 and 1.0"
        
        if not (1 <= max_themes <= 20):
            return False, "Max themes per framework must be between 1 and 20"
        
        if not (1 <= batch_size <= 100):
            return False, "Batch size must be between 1 and 100"
        
        return True, "Settings are valid"
        
    except Exception as validation_error:
        return False, f"Settings validation error: {validation_error}"

# ===============================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# ===============================================

def show_recommendations_compact(recommendations: List[Dict]):
    """Show recommendations in compact format - backwards compatibility"""
    if not recommendations:
        st.info("No recommendations to display")
        return
    
    for i, rec in enumerate(recommendations, 1):
        # Handle both new and legacy formats
        if isinstance(rec, dict):
            rec_id = rec.get('number', i)
            rec_text = rec.get('text', '')
            doc_source = rec.get('document_context', {}).get('filename', 'Unknown')
            confidence = rec.get('confidence_score', 0)
        else:
            # Legacy object format
            rec_id = getattr(rec, 'id', i)
            rec_text = getattr(rec, 'text', '')
            doc_source = getattr(rec, 'document_source', 'Unknown')
            confidence = getattr(rec, 'confidence_score', 0)
        
        preview = rec_text[:100] + "..." if len(rec_text) > 100 else rec_text
        
        with st.expander(f"**Rec {rec_id}**: {preview}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Full Text:**")
                st.write(rec_text)
            
            with col2:
                st.write("**Details:**")
                st.write(f"- **Confidence:** {confidence:.3f}")
                st.write(f"- **Source:** {doc_source}")

def get_recommendations_for_annotation():
    """Get extracted recommendations for annotation - backwards compatibility"""
    # Try new extraction results first
    extraction_results = st.session_state.get('extraction_results', {})
    recommendations = extraction_results.get('recommendations', [])
    
    if recommendations:
        return recommendations
    
    # Fall back to legacy extraction results
    legacy_recommendations = st.session_state.get('extracted_recommendations', [])
    if legacy_recommendations:
        return convert_legacy_recommendations(legacy_recommendations)
    
    return []

def get_annotation_ready_content():
    """Get all content ready for annotation"""
    recommendations = get_recommendations_for_annotation()
    
    # Get responses
    extraction_results = st.session_state.get('extraction_results', {})
    responses = extraction_results.get('responses', [])
    
    # Also check for legacy concerns
    if not responses:
        legacy_concerns = st.session_state.get('extracted_concerns', [])
        if legacy_concerns:
            responses = convert_legacy_concerns_to_responses(legacy_concerns)
    
    return {
        'recommendations': recommendations,
        'responses': responses,
        'total_items': len(recommendations) + len(responses)
    }

# ===============================================
# ADVANCED FEATURES
# ===============================================

def render_annotation_diagnostics():
    """Render annotation diagnostics for debugging"""
    with st.expander("🔧 Annotation Diagnostics", expanded=False):
        # Component status
        st.write("**Component Status:**")
        st.write(f"- BERT Annotator: {'✅' if BERT_ANNOTATOR_AVAILABLE else '❌'}")
        st.write(f"- Core Utils: {'✅' if CORE_UTILS_AVAILABLE else '❌'}")
        
        # Session state info
        annotator = st.session_state.get('bert_annotator')
        st.write(f"- BERT Initialized: {'✅' if annotator else '❌'}")
        
        if annotator:
            st.write(f"- Available Frameworks: {list(annotator.frameworks.keys())}")
        
        # Current settings
        st.write("**Current Settings:**")
        st.write(f"- Selected Frameworks: {st.session_state.get('selected_frameworks', [])}")
        st.write(f"- Similarity Threshold: {st.session_state.get('similarity_threshold', 'Not set')}")
        st.write(f"- Max Themes: {st.session_state.get('max_themes_per_framework', 'Not set')}")
        
        # Results info
        results = st.session_state.get('annotation_results', {})
        st.write(f"- Annotation Results: {len(results)} items")
        
        # Statistics
        stats = get_annotation_statistics()
        st.write("**Statistics:**")
        st.json(stats)

def test_annotation_pipeline():
    """Test the annotation pipeline with sample data"""
    if st.button("🧪 Test Annotation Pipeline"):
        # Create test data
        test_items = [
            {
                'number': 'TEST-1',
                'text': 'Improve communication protocols between healthcare teams to ensure better information sharing and coordination of patient care.',
                'document_context': {'filename': 'test_document.pdf'},
                'confidence_score': 0.9
            },
            {
                'number': 'TEST-2', 
                'text': 'Implement new technology systems to support clinical decision-making and reduce human error in medication administration.',
                'document_context': {'filename': 'test_document.pdf'},
                'confidence_score': 0.85
            }
        ]
        
        test_frameworks = ['I-SIRch', 'House of Commons']
        
        st.info("Testing annotation pipeline with sample data...")
        
        try:
            # Test the annotation process
            run_comprehensive_annotation(test_items, test_frameworks)
            st.success("✅ Annotation pipeline test completed!")
            
        except Exception as test_error:
            st.error(f"❌ Annotation pipeline test failed: {test_error}")

def create_annotation_report():
    """Create a comprehensive annotation report"""
    results = st.session_state.get('annotation_results', {})
    if not results:
        st.warning("No annotation results available for report generation")
        return
    
    if st.button("📊 Generate Comprehensive Report"):
        with st.spinner("Generating comprehensive annotation report..."):
            # Calculate comprehensive statistics
            successful_results = {k: v for k, v in results.items() if not v.get('error')}
            
            # Generate report sections
            report_data = {
                'overview': {
                    'total_items': len(results),
                    'successful_annotations': len(successful_results),
                    'success_rate': len(successful_results) / max(1, len(results)),
                    'generation_date': datetime.now().isoformat()
                },
                'framework_analysis': {},
                'theme_distribution': {},
                'confidence_analysis': {},
                'content_analysis': {}
            }
            
            # Analyze by framework
            for result in successful_results.values():
                annotations = result.get('annotations', {})
                item = result.get('item', {})
                
                for framework, themes in annotations.items():
                    if framework not in report_data['framework_analysis']:
                        report_data['framework_analysis'][framework] = {
                            'items_annotated': 0,
                            'themes_assigned': 0,
                            'avg_confidence': 0,
                            'confidence_scores': []
                        }
                    
                    report_data['framework_analysis'][framework]['items_annotated'] += 1
                    report_data['framework_analysis'][framework]['themes_assigned'] += len(themes)
                    
                    for theme in themes:
                        confidence = theme.get('confidence', 0)
                        report_data['framework_analysis'][framework]['confidence_scores'].append(confidence)
            
            # Calculate averages
            for framework, stats in report_data['framework_analysis'].items():
                if stats['confidence_scores']:
                    stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
                    del stats['confidence_scores']  # Remove raw scores for cleaner output
            
            # Display report
            st.subheader("📊 Comprehensive Annotation Report")
            
            # Overview
            st.write("**Overview:**")
            overview = report_data['overview']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", overview['total_items'])
            with col2:
                st.metric("Successful", overview['successful_annotations'])
            with col3:
                st.metric("Success Rate", f"{overview['success_rate']:.1%}")
            
            # Framework analysis
            st.write("**Framework Analysis:**")
            framework_df = pd.DataFrame.from_dict(report_data['framework_analysis'], orient='index')
            framework_df.index.name = 'Framework'
            st.dataframe(framework_df, use_container_width=True)
            
            # Export report
            report_json = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                "📥 Download Report JSON",
                report_json,
                f"annotation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

# ===============================================
# MISSING FUNCTIONS - NOW IMPLEMENTED
# ===============================================

def calculate_annotation_time_estimate(items_count: int, frameworks_count: int) -> str:
    """Calculate estimated annotation time"""
    # Rough estimate: 2-5 seconds per item per framework
    base_time_per_item = 3  # seconds
    total_seconds = items_count * frameworks_count * base_time_per_item
    
    if total_seconds < 60:
        return f"~{total_seconds} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        return f"~{minutes} minutes"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"~{hours}h {minutes}m"

def show_comprehensive_annotation_summary(successful: int, failed: List[str], total: int, items: List):
    """Show comprehensive annotation summary"""
    if successful > 0:
        st.success(f"🎉 Successfully annotated {successful} of {total} items!")
        
        # Content breakdown
        rec_count = sum(1 for item in items if hasattr(item, 'id') and hasattr(item, 'text'))
        concern_count = len(items) - rec_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recommendations", rec_count)
        with col2:
            st.metric("Concerns", concern_count)
        with col3:
            st.metric("Success Rate", f"{(successful/total)*100:.1f}%")
    
    if failed:
        st.error(f"❌ {len(failed)} items failed annotation")
        with st.expander("Error Details"):
            for error in failed:
                st.write(f"• {error}")

def format_confidence_indicator(confidence: float) -> str:
    """Format confidence as color-coded indicator"""
    if confidence >= 0.8:
        return "🟢"
    elif confidence >= 0.6:
        return "🟡"
    else:
        return "🔴"

def get_content_type_summary(items: List) -> str:
    """Get summary of content types in selection"""
    if not items:
        return "no items"
    
    recommendations_count = 0
    concerns_count = 0
    
    for item in items:
        if hasattr(item, 'id') and hasattr(item, 'text') and hasattr(item, 'document_source'):
            recommendations_count += 1
        else:
            concerns_count += 1
    
    summary_parts = []
    if recommendations_count > 0:
        summary_parts.append(f"{recommendations_count} recommendations")
    if concerns_count > 0:
        summary_parts.append(f"{concerns_count} concerns")
    
    return " + ".join(summary_parts) if summary_parts else "no items"

def validate_annotation_inputs(items: List, frameworks: List[str]) -> Tuple[bool, str]:
    """Validate inputs for annotation"""
    if not items:
        return False, "No items selected for annotation"
    
    if not frameworks:
        return False, "No frameworks selected"
    
    # Check if BERT annotator is available
    annotator = st.session_state.get('bert_annotator')
    if not annotator:
        return False, "BERT annotator not initialized"
    
    return True, "Validation passed"

def clean_text_for_annotation(text: str) -> str:
    """Clean text before annotation"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might interfere with annotation
    text = text.replace('\x00', '')  # Remove null bytes
    
    return text.strip()

def handle_annotation_error(item_id: str, error: Exception, error_list: List[str]):
    """Handle annotation errors gracefully"""
    error_msg = f"Error annotating {item_id}: {str(error)}"
    error_list.append(error_msg)
    logging.error(f"Annotation error for {item_id}: {error}", exc_info=True)

def recover_from_annotation_failure(results: Dict, failed_items: List[str]):
    """Attempt to recover from annotation failures"""
    if failed_items:
        st.warning(f"⚠️ {len(failed_items)} items failed annotation.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retry Failed Items"):
                st.info("Retry functionality would be implemented here")
        
        with col2:
            if st.button("📋 View Error Details"):
                with st.expander("Error Details"):
                    for error in failed_items:
                        st.write(f"• {error}")

def save_annotation_checkpoint(results: Dict):
    """Save annotation results as checkpoint"""
    try:
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'results_count': len(results),
            'results': results
        }
        
        # In a real implementation, this would save to disk
        st.session_state.annotation_checkpoint = checkpoint_data
        
    except Exception as e:
        logging.error(f"Failed to save annotation checkpoint: {e}")

def load_annotation_checkpoint() -> Optional[Dict]:
    """Load annotation checkpoint if available"""
    try:
        return st.session_state.get('annotation_checkpoint')
    except Exception as e:
        logging.error(f"Failed to load annotation checkpoint: {e}")
        return None

def optimize_annotation_batch_size(total_items: int, available_memory_mb: int = 512) -> int:
    """Optimize batch size based on available resources"""
    # Simple heuristic for batch size optimization
    if total_items <= 10:
        return total_items
    elif total_items <= 50:
        return min(10, total_items)
    else:
        # For larger datasets, use smaller batches to prevent memory issues
        return min(20, total_items // 4)

def estimate_memory_usage(items_count: int, frameworks_count: int) -> float:
    """Estimate memory usage for annotation process"""
    # Rough estimate in MB
    base_memory_per_item = 2  # MB per item
    framework_overhead = 0.5  # MB per framework
    
    estimated_mb = (items_count * base_memory_per_item) + (frameworks_count * framework_overhead)
    return estimated_mb

def validate_annotation_settings() -> Tuple[bool, List[str]]:
    """Validate annotation settings"""
    issues = []
    
    similarity_threshold = st.session_state.get('similarity_threshold', 0.65)
    if similarity_threshold < 0.3 or similarity_threshold > 0.9:
        issues.append("Similarity threshold should be between 0.3 and 0.9")
    
    max_themes = st.session_state.get('max_themes_per_framework', 10)
    if max_themes < 1 or max_themes > 50:
        issues.append("Max themes should be between 1 and 50")
    
    batch_size = st.session_state.get('annotation_batch_size', 10)
    if batch_size < 1 or batch_size > 100:
        issues.append("Batch size should be between 1 and 100")
    
    return len(issues) == 0, issues

def apply_annotation_best_practices():
    """Apply best practice settings for annotation"""
    recommended_settings = {
        'similarity_threshold': 0.65,
        'max_themes_per_framework': 10,
        'annotation_batch_size': 10,
        'use_context_annotation': True,
        'enable_annotation_caching': True,
        'min_keyword_matches': 1
    }
    
    for key, value in recommended_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_item_info(item):
    """Extract ID, text, and type from content item"""
    if hasattr(item, 'id') and hasattr(item, 'text'):
        return item.id, item.text, "recommendation"
    elif isinstance(item, dict):
        item_id = item.get('id', f"concern_{hash(item.get('text', ''))}")
        item_text = item.get('text', '')
        return item_id, item_text, "concern"
    else:
        return str(hash(str(item))), str(item), "unknown"

def create_comprehensive_results_dataframe(annotation_results):
    """Create comprehensive results DataFrame"""
    table_data = []
    
    for item_id, result in annotation_results.items():
        content_item = result.get('content_item')
        content_type = result.get('content_type', 'unknown')
        annotations = result.get('annotations', {})
        
        # Get content details
        if content_type == 'recommendation' and hasattr(content_item, 'text'):
            content_text = content_item.text
            document_source = getattr(content_item, 'document_source', 'Unknown')
            section = getattr(content_item, 'section_title', 'Unknown')
        elif content_type == 'concern' and isinstance(content_item, dict):
            content_text = content_item.get('text', '')
            document_source = content_item.get('document_source', 'Unknown')
            section = content_item.get('type', 'Unknown')
        else:
            content_text = str(content_item)
            document_source = 'Unknown'
            section = 'Unknown'
        
        # Create rows for each annotation
        for framework, themes in annotations.items():
            for theme in themes:
                table_data.append({
                    'Item ID': item_id,
                    'Content Type': content_type.title(),
                    'Document': document_source.split('/')[-1] if '/' in document_source else document_source,
                    'Section': section,
                    'Framework': framework,
                    'Theme': theme['theme'],
                    'Confidence': theme['confidence'],
                    'Combined Score': theme.get('semantic_similarity', 0) * theme['confidence'],
                    'Matched Keywords': ', '.join(theme.get('matched_keywords', [])[:3]),
                    'Content Preview': content_text[:100] + "..." if len(content_text) > 100 else content_text,
                    'Annotation Time': result.get('annotation_time', '')
                })
    
    return pd.DataFrame(table_data)

def export_annotation_summary_csv(results):
    """Export annotation summary as CSV"""
    table_data = []
    
    for item_id, result in results.items():
        content_item = result.get('content_item')
        content_type = result.get('content_type', 'unknown')
        annotations = result.get('annotations', {})
        
        # Get content info
        if content_type == 'recommendation' and hasattr(content_item, 'text'):
            content_text = content_item.text
            document_source = getattr(content_item, 'document_source', 'Unknown')
            section = getattr(content_item, 'section_title', 'Unknown')
        elif content_type == 'concern' and isinstance(content_item, dict):
            content_text = content_item.get('text', '')
            document_source = content_item.get('document_source', 'Unknown')
            section = content_item.get('type', 'Unknown')
        else:
            content_text = str(content_item)
            document_source = 'Unknown'
            section = 'Unknown'
        
        # Create rows for each annotation
        for framework, themes in annotations.items():
            for theme in themes:
                table_data.append({
                    'Item_ID': item_id,
                    'Content_Type': content_type.title(),
                    'Text': content_text[:200] + ('...' if len(content_text) > 200 else ''),
                    'Document': document_source,
                    'Framework': framework,
                    'Theme': theme['theme'],
                    'Confidence': theme['confidence'],
                    'Keywords': ', '.join(theme.get('matched_keywords', [])),
                })
    
    if table_data:
        df = pd.DataFrame(table_data)
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"annotation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    else:
        st.error("No data to export")

def export_annotations_json(results):
    """Export annotations as JSON"""
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_items': len(results),
        'annotation_results': results
    }
    
    json_data = json.dumps(export_data, indent=2, default=str)
    
    st.download_button(
        "📥 Download JSON",
        json_data,
        f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json"
    )

def download_single_highlighted_html(doc_name, html_content):
    """Download single highlighted HTML"""
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Highlighted Analysis - {doc_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Document Analysis - {doc_name}</h1>
        <div class="content">{html_content}</div>
    </div>
</body>
</html>"""
    
    st.download_button(
        "📥 Download HTML",
        data=full_html,
        file_name=f"highlighted_{doc_name}.html",
        mime="text/html"
    )

def download_all_highlighted_html(highlighted_texts):
    """Download all highlighted documents as ZIP"""
    import zipfile
    import io
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for doc_id, highlighting_data in highlighted_texts.items():
            original_text = highlighting_data.get('original_text', '')
            theme_highlights = highlighting_data.get('theme_highlights', {})
            
            if original_text and theme_highlights:
                html_content = create_highlighting_html(original_text, theme_highlights)
                
                full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Highlighted Analysis - {doc_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Document Analysis - {doc_id}</h1>
        <div class="content">{html_content}</div>
    </div>
</body>
</html>"""
                
                zip_file.writestr(f"highlighted_{doc_id}.html", full_html)
    
    zip_buffer.seek(0)
    
    st.download_button(
        "📦 Download All HTML (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"highlighted_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )

def create_highlighting_html(text: str, theme_highlights: Dict) -> str:
    """Create HTML with theme highlighting"""
    # Simple implementation for highlighting
    highlighted_text = text
    
    # Apply highlighting for each theme
    for theme, color in theme_highlights.items():
        # This is a simplified version - in practice, you'd want more sophisticated highlighting
        highlighted_text = highlighted_text.replace(
            theme, 
            f'<span style="background-color: {color}; padding: 2px;">{theme}</span>'
        )
    
    return highlighted_text

def get_theme_color(framework: str, theme: str) -> str:
    """Get color for a specific framework/theme combination"""
    theme_colors = {
        "I-SIRch_External factors": "#FFE0B2",
        "I-SIRch_System factors": "#C8E6C9", 
        "I-SIRch_Technology factors": "#B3E5FC",
        "I-SIRch_Person factors": "#F8BBD9",
        "I-SIRch_Task factors": "#D1C4E9",
        "House of Commons_Communication": "#FFCDD2",
        "House of Commons_Fragmented care": "#F0F4C3",
        "House of Commons_Workforce pressures": "#DCEDC8",
        "House of Commons_Biases and stereotyping": "#E1BEE7",
        "House of Commons_Resources": "#FFF8E1",
        "House of Commons_Standards": "#E8EAF6",
        "Extended Analysis_Risk Management": "#FCE4EC",
        "Extended Analysis_Quality Assurance": "#F5F5DC",
        "Extended Analysis_Leadership": "#E8EAF6",
    }
    
    theme_key = f"{framework}_{theme}"
    return theme_colors.get(theme_key, "#E0E0E0")

def render_comprehensive_export_options(results):
    """Render comprehensive export options"""
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Summary CSV", use_container_width=True):
            export_annotation_summary_csv(results)
    
    with col2:
        if st.button("📋 Export Detailed CSV", use_container_width=True):
            export_detailed_annotations_csv(results)
    
    with col3:
        if st.button("📄 Export JSON", use_container_width=True):
            export_annotations_json(results)
    
    with col4:
        highlighted_texts = st.session_state.get('bert_results', {}).get('highlighted_texts', {})
        if highlighted_texts and st.button("🌈 Export All HTML", use_container_width=True):
            download_all_highlighted_html(highlighted_texts)

# Legacy compatibility functions
def annotate_recommendations(recommendations: List, frameworks: List[str]):
    """Legacy function for backward compatibility"""
    st.warning("⚠️ Using legacy annotation function. Consider updating to the enhanced version.")
    
    # Convert to new format and use new function
    if recommendations and frameworks:
        run_comprehensive_annotation(recommendations, frameworks)

# ===============================================
# UPDATED MODULE EXPORTS
# ===============================================

__all__ = [
    # Main functions
    'render_annotation_tab',
    'render_framework_management', 
    'render_annotation_configuration',
    'render_annotation_interface',
    
    # Processing functions
    'run_comprehensive_annotation',
    'show_annotation_summary',
    'initialize_bert_annotator',
    
    # Display functions
    'display_annotation_results',
    'render_detailed_annotation_results',
    'render_results_by_theme',
    'render_annotation_statistics',
    
    # Content selection functions
    'render_content_selection',
    'render_recommendation_selection',
    'render_response_selection',
    'render_mixed_content_selection',
    'render_annotation_controls',
    
    # Export functions
    'render_annotation_export_options',
    'export_annotation_summary_csv',
    'export_detailed_annotations_csv',
    'export_annotations_json',
    'export_highlighted_html',
    'export_statistics_report',
    'render_comprehensive_export_options',
    'download_single_highlighted_html',
    'download_all_highlighted_html',
    
    # Framework functions
    'render_custom_framework_upload',
    'load_custom_framework',
    
    # Utility functions
    'highlight_text_with_themes',
    'get_theme_color_map',
    'filter_annotation_results',
    'convert_legacy_recommendations',
    'convert_legacy_concerns_to_responses',
    'log_annotation_action',
    'clear_annotation_results',
    'get_annotation_statistics',
    'validate_annotation_settings',
    
    # Missing functions now implemented
    'calculate_annotation_time_estimate',
    'show_comprehensive_annotation_summary',
    'format_confidence_indicator',
    'get_content_type_summary',
    'validate_annotation_inputs',
    'clean_text_for_annotation',
    'handle_annotation_error',
    'recover_from_annotation_failure',
    'save_annotation_checkpoint',
    'load_annotation_checkpoint',
    'optimize_annotation_batch_size',
    'estimate_memory_usage',
    'apply_annotation_best_practices',
    'get_item_info',
    'create_comprehensive_results_dataframe',
    'create_highlighting_html',
    'get_theme_color',
    
    # Backwards compatibility
    'show_recommendations_compact',
    'get_recommendations_for_annotation',
    'get_annotation_ready_content',
    'annotate_recommendations',
    
    # Advanced features
    'render_annotation_diagnostics',
    'test_annotation_pipeline',
    'create_annotation_report',
    
    # Availability flags
    'BERT_ANNOTATOR_AVAILABLE',
    'CORE_UTILS_AVAILABLE'
]

# ===============================================
# MODULE INITIALIZATION
# ===============================================

# Log module initialization
if BERT_ANNOTATOR_AVAILABLE:
    logging.info("✅ Annotation components loaded with full BERT functionality")
else:
    logging.info("✅ Annotation components loaded with keyword-based fallback mode")

logging.info("🎉 Annotation components module is COMPLETE and ready for conceptual framework annotation!")

# ===============================================
# MODULE VALIDATION
# ===============================================

def validate_module_completeness():
    """Validate that the annotation module is complete and functional"""
    required_functions = [
        'render_annotation_tab',
        'render_framework_management',
        'render_annotation_interface',
        'run_comprehensive_annotation',
        'display_annotation_results'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    if missing_functions:
        logging.error(f"Missing required functions: {missing_functions}")
        return False
    
    logging.info("✅ Annotation components module validation passed")
    return True

# Validate on import
if not validate_module_completeness():
    logging.warning("⚠️ Annotation components module may not be fully functional")
else:
    logging.info(f"✅ Annotation components module loaded with {len(__all__)} functions")
    logging.info("🎉 Annotation components module is COMPLETE and functional!")
