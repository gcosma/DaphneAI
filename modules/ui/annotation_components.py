# ===============================================
# FILE: modules/ui/annotation_components.py
# COMPLETE VERSION - ALL ORIGINAL FUNCTIONALITY + COLOR HIGHLIGHTING
# ===============================================

import streamlit as st
import pandas as pd
import json
import logging
import streamlit.components.v1 as components
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Import required modules with error handling
try:
    import sys
    sys.path.append('modules')
    from bert_annotator import BERTConceptAnnotator
    from .shared_components import add_error_message, show_progress_indicator
except ImportError as e:
    logging.error(f"Import error in annotation_components: {e}")
    # Create mock class for development
    class BERTConceptAnnotator:
        def __init__(self):
            self.frameworks = {
                "I-SIRch": [{"name": "Sample Theme", "keywords": ["sample"]}],
                "House of Commons": [{"name": "Sample Theme", "keywords": ["sample"]}]
            }
        def annotate_text(self, text, frameworks): return {}, {}
        def load_custom_framework(self, file): return True, "Mock framework loaded"

# =============================================================================
# MAIN TAB RENDERING FUNCTION
# =============================================================================

def render_annotation_tab():
    """Render the complete concept annotation tab"""
    st.header("🏷️ Concept Annotation")
    
    # Check for content
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not recommendations and not concerns:
        st.warning("⚠️ Please extract recommendations or concerns first in the Extract Content tab.")
        return
    
    st.markdown("""
    Annotate recommendations and concerns with conceptual themes using BERT-based analysis and 
    established frameworks like I-SIRch and House of Commons.
    """)
    
    # Content overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Recommendations", len(recommendations))
    with col2:
        st.metric("⚠️ Concerns", len(concerns))
    with col3:
        st.metric("📊 Total Items", len(recommendations) + len(concerns))
    
    # Initialize BERT
    initialize_bert_annotator()
    
    # Framework management
    render_framework_management()
    
    # Annotation configuration
    render_annotation_configuration()
    
    # Annotation interface
    render_annotation_interface(recommendations, concerns)
    
    # Results display with color-coded highlighting
    display_annotation_results()

# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_bert_annotator():
    """Initialize BERT annotator in session state"""
    if not st.session_state.get('bert_annotator'):
        with st.spinner("🧠 Loading BERT model... This may take a moment."):
            try:
                st.session_state.bert_annotator = BERTConceptAnnotator()
                st.success("✅ BERT model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load BERT model: {str(e)}")
                add_error_message(f"BERT initialization failed: {str(e)}")
                st.session_state.bert_annotator = BERTConceptAnnotator()

# =============================================================================
# FRAMEWORK MANAGEMENT FUNCTIONS
# =============================================================================

def render_framework_management():
    """Render comprehensive framework selection and management"""
    st.subheader("🎯 Conceptual Frameworks")
    
    annotator = st.session_state.get('bert_annotator')
    if not annotator:
        st.error("BERT annotator not initialized.")
        return
    
    # Available frameworks display
    available_frameworks = list(annotator.frameworks.keys())
    
    st.markdown("**Available Frameworks:**")
    for framework in available_frameworks:
        themes_count = len(annotator.frameworks[framework])
        st.write(f"• **{framework}:** {themes_count} themes")
    
    # Framework selection
    selected_frameworks = st.multiselect(
        "Select frameworks to use for annotation:",
        available_frameworks,
        default=available_frameworks[:2] if len(available_frameworks) >= 2 else available_frameworks,
        help="Choose which conceptual frameworks to apply",
        key="selected_frameworks"
    )
    
    # Framework details
    if selected_frameworks:
        with st.expander("📋 Framework Details"):
            for framework in selected_frameworks:
                st.markdown(f"**{framework}:**")
                themes = annotator.frameworks[framework]
                for theme in themes[:5]:  # Show first 5 themes
                    theme_name = theme.get('name', 'Unknown')
                    keywords = theme.get('keywords', [])
                    st.write(f"• {theme_name}: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
                if len(themes) > 5:
                    st.write(f"... and {len(themes) - 5} more themes")
    
    # Custom framework upload
    render_custom_framework_upload(annotator)

def render_custom_framework_upload(annotator):
    """Render custom framework upload interface"""
    with st.expander("📁 Upload Custom Framework"):
        st.markdown("Upload your own conceptual framework in JSON, CSV, or Excel format.")
        
        uploaded_file = st.file_uploader(
            "Choose framework file:",
            type=['json', 'csv', 'xlsx'],
            help="Upload your own conceptual framework",
            key="custom_framework_upload"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if uploaded_file and st.button("📤 Load Custom Framework"):
                try:
                    success, message = annotator.load_custom_framework(uploaded_file)
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"❌ Error loading framework: {str(e)}")
        
        with col2:
            if st.button("📖 Framework Format Help"):
                st.info("""
                **JSON Format:**
                ```json
                [
                  {
                    "name": "Theme Name",
                    "keywords": ["keyword1", "keyword2"]
                  }
                ]
                ```
                
                **CSV Format:**
                - Column 1: Theme Name
                - Column 2: Keywords (comma-separated)
                """)

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def render_annotation_configuration():
    """Render comprehensive annotation configuration settings"""
    st.subheader("⚙️ Annotation Settings")
    
    # Basic settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.30, 0.90, 0.65, 0.05,
            help="Minimum similarity score for theme matching",
            key="similarity_threshold"
        )
    
    with col2:
        max_themes = st.slider(
            "Max Themes per Framework",
            1, 15, 10,
            help="Maximum themes to identify per framework",
            key="max_themes_per_framework"
        )
    
    with col3:
        annotation_mode = st.selectbox(
            "Annotation Mode",
            ["Semantic + Keywords", "Semantic Only", "Keywords Only"],
            help="Choose how to perform concept matching",
            key="annotation_mode"
        )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input(
                "Batch Size",
                1, 50, 10,
                help="Number of items to process at once",
                key="annotation_batch_size"
            )
            
            use_context = st.checkbox(
                "Use Context Window",
                value=True,
                help="Include surrounding text for better context",
                key="use_context_annotation"
            )
        
        with col2:
            min_keyword_match = st.number_input(
                "Min Keyword Matches",
                1, 5, 1,
                help="Minimum keyword matches required",
                key="min_keyword_matches"
            )
            
            enable_caching = st.checkbox(
                "Enable Caching",
                value=True,
                help="Cache results for faster re-annotation",
                key="enable_annotation_caching"
            )

# =============================================================================
# ANNOTATION INTERFACE FUNCTIONS
# =============================================================================

def render_annotation_interface(recommendations, concerns):
    """Render the comprehensive annotation interface"""
    st.subheader("🔬 Annotation Process")
    
    selected_frameworks = st.session_state.get('selected_frameworks', [])
    
    if not selected_frameworks:
        st.warning("⚠️ Please select at least one framework above.")
        return
    
    # Content type selection
    content_options = []
    if recommendations:
        content_options.append("Recommendations")
    if concerns:
        content_options.append("Concerns") 
    if len(content_options) > 1:
        content_options.append("Both")
    
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
    elif content_choice == "Concerns":
        items_to_annotate = render_concern_selection(concerns)
    elif content_choice == "Both":
        items_to_annotate = render_mixed_content_selection(recommendations, concerns)
    
    # Annotation summary and controls
    if items_to_annotate:
        render_annotation_controls(items_to_annotate, selected_frameworks)

def render_recommendation_selection(recommendations):
    """Render recommendation selection interface"""
    selection_method = st.radio(
        "Selection method:",
        ["All Recommendations", "By Document", "Individual Selection"],
        key="rec_selection_method"
    )
    
    if selection_method == "All Recommendations":
        return recommendations
        
    elif selection_method == "By Document":
        doc_sources = list(set(rec.document_source for rec in recommendations))
        selected_docs = st.multiselect(
            "Select documents:",
            doc_sources,
            default=doc_sources,
            key="selected_rec_docs"
        )
        return [rec for rec in recommendations if rec.document_source in selected_docs]
        
    elif selection_method == "Individual Selection":
        # Individual recommendation selection
        st.markdown("**Select individual recommendations:**")
        
        # Group by document for better organization
        docs_recs = {}
        for rec in recommendations:
            doc = rec.document_source
            if doc not in docs_recs:
                docs_recs[doc] = []
            docs_recs[doc].append(rec)
        
        selected_recs = []
        
        for doc, doc_recs in docs_recs.items():
            with st.expander(f"📄 {doc.split('/')[-1]} ({len(doc_recs)} recommendations)"):
                for rec in doc_recs:
                    preview = rec.text[:100] + "..." if len(rec.text) > 100 else rec.text
                    if st.checkbox(f"{rec.id}: {preview}", key=f"rec_select_{rec.id}"):
                        selected_recs.append(rec)
        
        return selected_recs
    
    return []

def render_concern_selection(concerns):
    """Render concern selection interface"""
    selection_method = st.radio(
        "Selection method:",
        ["All Concerns", "By Document", "Individual Selection"],
        key="concern_selection_method"
    )
    
    if selection_method == "All Concerns":
        return concerns
        
    elif selection_method == "By Document":
        doc_sources = list(set(concern.get('document_source', 'Unknown') for concern in concerns))
        selected_docs = st.multiselect(
            "Select documents:",
            doc_sources,
            default=doc_sources,
            key="selected_concern_docs"
        )
        return [concern for concern in concerns if concern.get('document_source', 'Unknown') in selected_docs]
        
    elif selection_method == "Individual Selection":
        # Individual concern selection
        st.markdown("**Select individual concerns:**")
        
        selected_concerns = []
        for i, concern in enumerate(concerns):
            concern_id = concern.get('id', f'concern_{i}')
            concern_text = concern.get('text', '')
            preview = concern_text[:100] + "..." if len(concern_text) > 100 else concern_text
            
            if st.checkbox(f"{concern_id}: {preview}", key=f"concern_select_{concern_id}"):
                selected_concerns.append(concern)
        
        return selected_concerns
    
    return []

def render_mixed_content_selection(recommendations, concerns):
    """Render mixed content selection interface"""
    st.markdown("**Select content to annotate:**")
    
    col1, col2 = st.columns(2)
    selected_items = []
    
    with col1:
        st.markdown("**📋 Recommendations:**")
        include_all_recs = st.checkbox(
            f"Include all recommendations ({len(recommendations)})",
            key="include_all_recs"
        )
        
        if include_all_recs:
            selected_items.extend(recommendations)
        else:
            rec_docs = list(set(rec.document_source for rec in recommendations))
            selected_rec_docs = st.multiselect(
                "Select recommendation documents:",
                rec_docs,
                key="mixed_rec_docs"
            )
            selected_items.extend([rec for rec in recommendations if rec.document_source in selected_rec_docs])
    
    with col2:
        st.markdown("**⚠️ Concerns:**")
        include_all_concerns = st.checkbox(
            f"Include all concerns ({len(concerns)})",
            key="include_all_concerns"
        )
        
        if include_all_concerns:
            selected_items.extend(concerns)
        else:
            concern_docs = list(set(concern.get('document_source', 'Unknown') for concern in concerns))
            selected_concern_docs = st.multiselect(
                "Select concern documents:",
                concern_docs,
                key="mixed_concern_docs"
            )
            selected_items.extend([concern for concern in concerns if concern.get('document_source', 'Unknown') in selected_concern_docs])
    
    return selected_items

def render_annotation_controls(items_to_annotate, selected_frameworks):
    """Render annotation controls and summary"""
    # Content summary
    rec_count = sum(1 for item in items_to_annotate if hasattr(item, 'id') and hasattr(item, 'text'))
    concern_count = len(items_to_annotate) - rec_count
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        content_summary = []
        if rec_count > 0:
            content_summary.append(f"{rec_count} recommendations")
        if concern_count > 0:
            content_summary.append(f"{concern_count} concerns")
        
        summary_text = " + ".join(content_summary) if content_summary else "no items"
        st.info(f"📊 Ready to annotate {len(items_to_annotate)} items ({summary_text}) with {len(selected_frameworks)} frameworks")
    
    with col2:
        st.markdown("**Settings:**")
        threshold = st.session_state.get('similarity_threshold', 0.65)
        max_themes = st.session_state.get('max_themes_per_framework', 10)
        mode = st.session_state.get('annotation_mode', 'Semantic + Keywords')
        st.write(f"• Threshold: {threshold:.2f}")
        st.write(f"• Max themes: {max_themes}")
        st.write(f"• Mode: {mode}")
    
    # Annotation button
    if st.button("🚀 Start Annotation", type="primary", use_container_width=True):
        run_comprehensive_annotation(items_to_annotate, selected_frameworks)

# =============================================================================
# ANNOTATION PROCESSING FUNCTIONS
# =============================================================================

def run_comprehensive_annotation(items, frameworks):
    """Run comprehensive annotation with full error handling"""
    annotator = st.session_state.get('bert_annotator')
    if not annotator:
        st.error("BERT annotator not available.")
        return
    
    # Get configuration
    similarity_threshold = st.session_state.get('similarity_threshold', 0.65)
    max_themes = st.session_state.get('max_themes_per_framework', 10)
    batch_size = st.session_state.get('annotation_batch_size', 10)
    annotation_mode = st.session_state.get('annotation_mode', 'Semantic + Keywords')
    
    # Update annotator configuration
    if hasattr(annotator, 'config'):
        annotator.config.update({
            "base_similarity_threshold": similarity_threshold,
            "max_themes_per_framework": max_themes,
            "annotation_mode": annotation_mode
        })
    
    # Progress tracking
    total_items = len(items)
    progress_container = st.container()
    status_container = st.container()
    
    annotation_results = {}
    highlighted_texts = {}
    successful_annotations = 0
    failed_annotations = []
    
    # Set processing status
    st.session_state.processing_status = "annotating"
    
    try:
        for i, item in enumerate(items):
            current_step = i + 1
            
            # Get item info
            item_id, item_text, item_type = get_item_info(item)
            
            # Update progress
            with progress_container:
                show_progress_indicator(current_step, total_items, f"Annotating {item_id}")
            
            with status_container:
                status_text = st.empty()
                status_text.info(f"🏷️ Processing: {item_id} ({item_type})")
            
            try:
                # Perform annotation
                framework_results, highlighting = annotator.annotate_text(item_text, frameworks)
                
                # Store results
                annotation_results[item_id] = {
                    'content_item': item,
                    'content_type': item_type,
                    'annotations': framework_results,
                    'highlighting': highlighting,
                    'annotation_time': datetime.now().isoformat(),
                    'frameworks_used': frameworks,
                    'settings': {
                        'similarity_threshold': similarity_threshold,
                        'max_themes': max_themes,
                        'annotation_mode': annotation_mode
                    }
                }
                
                # Store highlighting for color display
                if highlighting:
                    highlighted_texts[item_id] = {
                        'original_text': item_text,
                        'theme_highlights': highlighting,
                        'item_type': item_type
                    }
                
                successful_annotations += 1
                status_text.success(f"✅ Annotated: {item_id}")
                
            except Exception as e:
                error_msg = f"Error annotating {item_id}: {str(e)}"
                failed_annotations.append(error_msg)
                add_error_message(error_msg)
                status_text.error(f"❌ Failed: {item_id}")
                logging.error(f"Annotation error: {e}", exc_info=True)
        
        # Update session state
        if annotation_results:
            existing_results = st.session_state.get('annotation_results', {})
            existing_results.update(annotation_results)
            st.session_state.annotation_results = existing_results
        
        # Update BERT results for highlighting
        st.session_state.bert_results = {
            'results_df': create_comprehensive_results_dataframe(annotation_results),
            'highlighted_texts': highlighted_texts
        }
        
        # Clear progress displays
        progress_container.empty()
        status_container.empty()
        
        # Show comprehensive results summary
        show_comprehensive_annotation_summary(successful_annotations, failed_annotations, total_items, items)
        
    finally:
        st.session_state.processing_status = "idle"

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

def show_comprehensive_annotation_summary(successful: int, failed: List[str], total: int, items: List):
    """Show comprehensive annotation summary"""
    if successful > 0:
        st.success(f"🎉 Successfully annotated {successful} of {total} items!")
    
    # Content type breakdown
    rec_count = sum(1 for item in items if hasattr(item, 'id') and hasattr(item, 'text'))
    concern_count = len(items) - rec_count
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Successful", successful)
    with col2:
        st.metric("📋 Recommendations", rec_count)
    with col3:
        st.metric("⚠️ Concerns", concern_count)
    
    if failed:
        st.error(f"❌ Failed to annotate {len(failed)} items:")
        for error in failed[:5]:
            st.write(f"• {error}")
        if len(failed) > 5:
            st.write(f"... and {len(failed) - 5} more errors")
    
    # Quick stats
    if successful > 0:
        results = st.session_state.get('annotation_results', {})
        if results:
            total_annotations = sum(
                len(themes) 
                for result in results.values() 
                for themes in result.get('annotations', {}).values()
            )
            st.info(f"📊 Total annotations created: {total_annotations}")

# =============================================================================
# COMPREHENSIVE RESULTS DISPLAY WITH COLOR-CODED HIGHLIGHTING
# =============================================================================

def display_annotation_results():
    """Display comprehensive annotation results with color-coded highlighting"""
    results = st.session_state.get('annotation_results', {})
    
    if not results:
        st.info("💡 No annotations yet. Configure settings above and click 'Start Annotation' to begin.")
        return
    
    st.subheader("🏷️ Annotation Results")
    
    # Overview metrics
    display_annotation_overview_metrics(results)
    
    # Results table
    display_comprehensive_results_table(results)
    
    # Color-coded highlighting
    highlighted_texts = st.session_state.get('bert_results', {}).get('highlighted_texts', {})
    if highlighted_texts:
        render_color_coded_highlighting_section(highlighted_texts)
    
    # Export options
    render_comprehensive_export_options(results)

def display_annotation_overview_metrics(results: Dict):
    """Display comprehensive overview metrics"""
    total_items = len(results)
    total_annotations = 0
    framework_counts = {}
    confidence_scores = []
    content_type_counts = {'recommendation': 0, 'concern': 0, 'unknown': 0}
    
    for result in results.values():
        content_type = result.get('content_type', 'unknown')
        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        annotations = result.get('annotations', {})
        for framework, themes in annotations.items():
            framework_counts[framework] = framework_counts.get(framework, 0) + len(themes)
            total_annotations += len(themes)
            
            for theme in themes:
                confidence_scores.append(theme['confidence'])
    
    # Display metrics
    st.markdown("### 📊 Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Items Annotated", total_items)
    with col2:
        st.metric("Total Annotations", total_annotations)
    with col3:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    with col4:
        st.metric("Frameworks Used", len(framework_counts))
    
    # Framework distribution
    if framework_counts:
        st.markdown("### 🎯 Framework Distribution")
        framework_cols = st.columns(len(framework_counts))
        
        for i, (framework, count) in enumerate(framework_counts.items()):
            with framework_cols[i]:
                st.metric(framework, count)

def display_comprehensive_results_table(results: Dict):
    """Display comprehensive results table with filtering"""
    results_df = st.session_state.get('bert_results', {}).get('results_df')
    
    if results_df is None or results_df.empty:
        st.warning("No annotation data to display.")
        return
    
    st.markdown("### 📋 Detailed Results")
    
    # Filtering options
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        content_types = ['All'] + sorted(results_df['Content Type'].unique().tolist())
        content_filter = st.selectbox("Content Type:", content_types, key="results_content_filter")
    
    with col2:
        frameworks = ['All'] + sorted(results_df['Framework'].unique().tolist())
        framework_filter = st.selectbox("Framework:", frameworks, key="results_framework_filter")
    
    with col3:
        min_confidence = st.slider("Min Confidence:", 0.0, 1.0, 0.0, 0.05, key="results_confidence_filter")
    
    with col4:
        documents = ['All'] + sorted(results_df['Document'].unique().tolist())
        doc_filter = st.selectbox("Document:", documents, key="results_doc_filter")
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if content_filter != 'All':
        filtered_df = filtered_df[filtered_df['Content Type'] == content_filter]
    if framework_filter != 'All':
        filtered_df = filtered_df[filtered_df['Framework'] == framework_filter]
    if doc_filter != 'All':
        filtered_df = filtered_df[filtered_df['Document'] == doc_filter]
    
    filtered_df = filtered_df[filtered_df['Confidence'] >= min_confidence]
    
    if filtered_df.empty:
        st.warning("No annotations match the current filters.")
        return
    
    st.write(f"Showing {len(filtered_df)} of {len(results_df)} annotations")
    
    # Display results table with pagination
    items_per_page = 20
    total_pages = (len(filtered_df) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.selectbox("Page:", range(1, total_pages + 1), key="results_page")
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_df = filtered_df.iloc[start_idx:end_idx]
    else:
        page_df = filtered_df
    
    # Display main results table
    st.dataframe(
        page_df,
        use_container_width=True,
        column_config={
            "Item ID": st.column_config.TextColumn("Item ID", width="small"),
            "Content Type": st.column_config.TextColumn("Type", width="small"),
            "Document": st.column_config.TextColumn("Document", width="medium"),
            "Framework": st.column_config.TextColumn("Framework", width="small"),
            "Theme": st.column_config.TextColumn("Theme", width="medium"),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f", width="small"),
            "Combined Score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
            "Matched Keywords": st.column_config.TextColumn("Keywords", width="large"),
        },
        hide_index=True
    )
    
    # Individual annotation details
    if st.checkbox("🔍 Show Individual Annotation Details", key="show_annotation_details"):
        display_individual_annotations(page_df, results)

def display_individual_annotations(display_df: pd.DataFrame, all_results: Dict):
    """Display individual annotation details with expandable sections"""
    st.markdown("### 🔍 Individual Annotation Details")
    
    for _, row in display_df.iterrows():
        item_id = row['Item ID']
        framework = row['Framework']
        theme_name = row['Theme']
        confidence = row['Confidence']
        
        # Color coding based on confidence
        if confidence >= 0.8:
            confidence_color = "🟢"
        elif confidence >= 0.6:
            confidence_color = "🟡"
        else:
            confidence_color = "🔴"
        
        # Find the full result data
        result = all_results.get(item_id)
        if not result:
            continue
        
        content_item = result.get('content_item')
        content_type = result.get('content_type', 'unknown')
        
        # Get the specific theme data
        theme_data = None
        annotations = result.get('annotations', {})
        if framework in annotations:
            for theme in annotations[framework]:
                if theme['theme'] == theme_name:
                    theme_data = theme
                    break
        
        if not theme_data:
            continue
        
        with st.expander(f"{confidence_color} {item_id} - {framework}: {theme_name} ({confidence:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Content Text:**")
                if content_type == 'recommendation' and hasattr(content_item, 'text'):
                    st.write(content_item.text)
                elif content_type == 'concern' and isinstance(content_item, dict):
                    st.write(content_item.get('text', 'No text available'))
                else:
                    st.write("Content not available")
                
                st.markdown("**Matched Keywords:**")
                keywords = theme_data.get('matched_keywords', [])
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No keywords matched")
            
            with col2:
                st.markdown("**Annotation Metrics:**")
                st.write(f"**Type:** {content_type.title()}")
                st.write(f"**Framework:** {framework}")
                st.write(f"**Theme:** {theme_data['theme']}")
                st.write(f"**Confidence:** {theme_data['confidence']:.3f}")
                st.write(f"**Semantic Similarity:** {theme_data.get('semantic_similarity', 0):.3f}")
                st.write(f"**Keyword Count:** {theme_data.get('keyword_count', 0)}")
                
                # Source information
                st.markdown("**Source Details:**")
                if content_type == 'recommendation' and hasattr(content_item, 'document_source'):
                    st.write(f"**Document:** {content_item.document_source}")
                    if hasattr(content_item, 'section_title'):
                        st.write(f"**Section:** {content_item.section_title}")
                elif content_type == 'concern' and isinstance(content_item, dict):
                    st.write(f"**Document:** {content_item.get('document_source', 'Unknown')}")
                    if content_item.get('type'):
                        st.write(f"**Concern Type:** {content_item.get('type')}")

# =============================================================================
# COLOR-CODED HIGHLIGHTING SECTION
# =============================================================================

def render_color_coded_highlighting_section(highlighted_texts):
    """Render the color-coded highlighting section"""
    st.markdown("---")
    st.subheader("🌈 Color-Coded Text Highlighting")
    st.markdown("**View documents with themes highlighted in different colors:**")
    
    # Document selector
    doc_options = list(highlighted_texts.keys())
    selected_doc = st.selectbox(
        "Select document to view:",
        doc_options,
        key="highlighted_doc_selector"
    )
    
    if selected_doc and selected_doc in highlighted_texts:
        highlighting_data = highlighted_texts[selected_doc]
        
        # Display options
        col1, col2 = st.columns([3, 1])
        
        with col2:
            show_legend = st.checkbox("Show Color Legend", value=True)
            show_stats = st.checkbox("Show Statistics", value=True)
        
        # Display statistics
        if show_stats:
            display_highlighting_statistics(highlighting_data)
        
        # Display theme legend
        if show_legend:
            display_theme_legend(highlighting_data.get('theme_highlights', {}))
        
        # Generate and display highlighted HTML
        original_text = highlighting_data.get('original_text', '')
        theme_highlights = highlighting_data.get('theme_highlights', {})
        
        if original_text and theme_highlights:
            html_content = create_highlighting_html(original_text, theme_highlights)
            
            # Display in styled container
            styled_html = f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                background-color: #fafafa;
                margin: 10px 0;
                max-height: 600px;
                overflow-y: auto;
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
            ">
                {html_content}
            </div>
            """
            
            components.html(styled_html, height=650, scrolling=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download This HTML"):
                    download_single_highlighted_html(selected_doc, html_content)
            with col2:
                if st.button("📦 Download All HTML"):
                    download_all_highlighted_html(highlighted_texts)

def display_highlighting_statistics(highlighting_data):
    """Display highlighting statistics"""
    theme_highlights = highlighting_data.get('theme_highlights', {})
    total_highlights = sum(len(positions) for positions in theme_highlights.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Highlights", total_highlights)
    with col2:
        st.metric("Themes Found", len(theme_highlights))
    with col3:
        frameworks = set(theme.split('_')[0] for theme in theme_highlights.keys() if '_' in theme)
        st.metric("Frameworks", len(frameworks))

def create_highlighting_html(text, theme_highlights):
    """Create HTML with color-coded highlighting"""
    if not text or not theme_highlights:
        return f"<p>{text}</p>"
    
    # Theme colors mapping
    theme_colors = {
        "I-SIRch_Communication": "#FFD580",
        "I-SIRch_Documentation": "#FFECB3", 
        "I-SIRch_Training": "#E1F5FE",
        "I-SIRch_Policies": "#E8F5E9",
        "I-SIRch_Equipment": "#F3E5F5",
        "I-SIRch_Monitoring": "#FFF3E0",
        "House of Commons_Safety": "#E0F7FA",
        "House of Commons_Oversight": "#F1F8E9",
        "House of Commons_Resources": "#FFF8E1",
        "House of Commons_Standards": "#E8EAF6",
        "Extended Analysis_Risk Management": "#FCE4EC",
        "Extended Analysis_Quality Assurance": "#F5F5DC",
    }
    
    # Convert highlights to positions
    all_positions = []
    for theme_key, positions in theme_highlights.items():
        color = theme_colors.get(theme_key, "#E0E0E0")
        
        for pos_info in positions:
            all_positions.append((
                pos_info[0],  # start
                pos_info[1],  # end  
                theme_key,    # theme
                pos_info[2] if len(pos_info) > 2 else '',  # keywords
                color         # color
            ))
    
    # Sort by start position
    all_positions.sort()
    
    # Merge overlapping highlights
    merged_positions = []
    if all_positions:
        current = all_positions[0]
        for i in range(1, len(all_positions)):
            if all_positions[i][0] <= current[1]:  # Overlap
                # Merge highlights - combine themes and use first color
                combined_theme = current[2] + " + " + all_positions[i][2]
                combined_keywords = current[3] + " + " + all_positions[i][3]
                current = (
                    current[0],
                    max(current[1], all_positions[i][1]),
                    combined_theme,
                    combined_keywords,
                    current[4]  # Keep first color
                )
            else:
                merged_positions.append(current)
                current = all_positions[i]
        merged_positions.append(current)
    
    # Create highlighted HTML
    result = []
    last_end = 0
    
    for start, end, theme_key, keywords, color in merged_positions:
        # Add text before highlight
        if start > last_end:
            result.append(text[last_end:start])
        
        # Add highlighted text with tooltip
        theme_display = theme_key.split('_', 1)[1] if '_' in theme_key else theme_key
        tooltip = f"Theme: {theme_display}\\nKeywords: {keywords}"
        style = f"background-color: {color}; border: 1px solid #666; border-radius: 3px; padding: 2px 4px; margin: 0 1px;"
        
        result.append(f'<span style="{style}" title="{tooltip}">{text[start:end]}</span>')
        last_end = end
    
    # Add remaining text
    if last_end < len(text):
        result.append(text[last_end:])
    
    return ''.join(result)

def display_theme_legend(theme_highlights):
    """Display color legend for themes"""
    if not theme_highlights:
        return
    
    st.markdown("**🎨 Theme Color Legend:**")
    
    # Group themes by framework
    frameworks = {}
    for theme_key in theme_highlights.keys():
        if '_' in theme_key:
            framework, theme = theme_key.split('_', 1)
        else:
            framework, theme = "Other", theme_key
        
        if framework not in frameworks:
            frameworks[framework] = []
        frameworks[framework].append((theme, theme_key))
    
    # Display legend by framework
    for framework, themes in frameworks.items():
        st.markdown(f"**{framework} Framework:**")
        
        cols = st.columns(min(3, len(themes)))
        for i, (theme, theme_key) in enumerate(themes):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                color = get_theme_color(theme_key)
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    border: 1px solid #666;
                    border-radius: 3px;
                    padding: 5px 8px;
                    margin: 2px 0;
                    font-size: 12px;
                    text-align: center;
                ">
                    {theme}
                </div>
                """, unsafe_allow_html=True)

def get_theme_color(theme_key):
    """Get color for a theme"""
    theme_colors = {
        "I-SIRch_Communication": "#FFD580",
        "I-SIRch_Documentation": "#FFECB3", 
        "I-SIRch_Training": "#E1F5FE",
        "I-SIRch_Policies": "#E8F5E9",
        "I-SIRch_Equipment": "#F3E5F5",
        "I-SIRch_Monitoring": "#FFF3E0",
        "House of Commons_Safety": "#E0F7FA",
        "House of Commons_Oversight": "#F1F8E9",
        "House of Commons_Resources": "#FFF8E1",
        "House of Commons_Standards": "#E8EAF6",
        "Extended Analysis_Risk Management": "#FCE4EC",
        "Extended Analysis_Quality Assurance": "#F5F5DC",
    }
    return theme_colors.get(theme_key, "#E0E0E0")

# =============================================================================
# COMPREHENSIVE EXPORT FUNCTIONS
# =============================================================================

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
                    'Document': document_source,
                    'Section': section,
                    'Framework': framework,
                    'Theme': theme['theme'],
                    'Confidence': theme['confidence'],
                    'Combined_Score': theme.get('semantic_similarity', 0) * theme['confidence'],
                    'Matched_Keywords': ', '.join(theme.get('matched_keywords', [])),
                    'Content_Text': content_text,
                    'Annotation_Time': result.get('annotation_time', '')
                })
    
    df = pd.DataFrame(table_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "📥 Download Summary CSV",
        data=csv,
        file_name=f"annotation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_detailed_annotations_csv(results):
    """Export detailed annotations as CSV"""
    detailed_data = []
    
    for item_id, result in results.items():
        content_item = result['content_item']
        content_type = result.get('content_type', 'unknown')
        annotations = result.get('annotations', {})
        settings = result.get('settings', {})
        
        # Get text and source based on content type
        if content_type == 'recommendation' and hasattr(content_item, 'text'):
            text = content_item.text
            source_doc = getattr(content_item, 'document_source', 'Unknown')
            section = getattr(content_item, 'section_title', 'Unknown')
        elif content_type == 'concern' and isinstance(content_item, dict):
            text = content_item.get('text', '')
            source_doc = content_item.get('document_source', 'Unknown')
            section = content_item.get('type', 'Unknown')
        else:
            text = str(content_item)
            source_doc = 'Unknown'
            section = 'Unknown'
        
        for framework, themes in annotations.items():
            for theme in themes:
                detailed_data.append({
                    'Item_ID': item_id,
                    'Content_Type': content_type.title(),
                    'Item_Text': text,
                    'Source_Document': source_doc,
                    'Section': section,
                    'Framework': framework,
                    'Theme': theme['theme'],
                    'Confidence': theme['confidence'],
                    'Semantic_Similarity': theme.get('semantic_similarity', 0),
                    'Keyword_Count': theme.get('keyword_count', 0),
                    'Matched_Keywords': ', '.join(theme.get('matched_keywords', [])),
                    'Similarity_Threshold': settings.get('similarity_threshold', 0),
                    'Annotation_Mode': settings.get('annotation_mode', 'Unknown'),
                    'Annotation_Time': result.get('annotation_time', '')
                })
    
    df = pd.DataFrame(detailed_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "📥 Download Detailed CSV",
        data=csv,
        file_name=f"detailed_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_annotations_json(results):
    """Export annotations as JSON"""
    # Make results JSON serializable
    export_data = {}
    
    for item_id, result in results.items():
        content_item = result.get('content_item')
        content_type = result.get('content_type', 'unknown')
        
        # Create serializable content item data
        if content_type == 'recommendation' and hasattr(content_item, 'text'):
            content_data = {
                'id': getattr(content_item, 'id', item_id),
                'text': content_item.text,
                'document_source': getattr(content_item, 'document_source', 'Unknown'),
                'section_title': getattr(content_item, 'section_title', 'Unknown'),
                'confidence_score': getattr(content_item, 'confidence_score', 0)
            }
        elif content_type == 'concern' and isinstance(content_item, dict):
            content_data = {
                'id': content_item.get('id', item_id),
                'text': content_item.get('text', ''),
                'document_source': content_item.get('document_source', 'Unknown'),
                'type': content_item.get('type', 'Unknown'),
                'confidence_score': content_item.get('confidence_score', 0)
            }
        else:
            content_data = {
                'id': item_id,
                'text': str(content_item),
                'type': 'unknown'
            }
        
        export_data[item_id] = {
            'content_item': content_data,
            'content_type': content_type,
            'annotations': result.get('annotations', {}),
            'annotation_time': result.get('annotation_time', ''),
            'frameworks_used': result.get('frameworks_used', []),
            'settings': result.get('settings', {})
        }
    
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    st.download_button(
        "📥 Download JSON",
        data=json_str.encode('utf-8'),
        file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
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

# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def annotate_recommendations(recommendations: List, frameworks: List[str]):
    """Legacy function for backward compatibility"""
    st.warning("⚠️ Using legacy annotation function. Consider updating to the enhanced version.")
    
    # Convert to new format and use new function
    if recommendations and frameworks:
        run_comprehensive_annotation(recommendations, frameworks)

def show_annotation_summary(successful: int, failed: List[str], total: int):
    """Legacy summary function for backward compatibility"""
    # Create dummy items list for compatibility
    dummy_items = [{'type': 'recommendation'} for _ in range(total)]
    show_comprehensive_annotation_summary(successful, failed, total, dummy_items)

def render_annotation_interface():
    """Legacy annotation interface for backward compatibility"""
    st.subheader("🔬 Legacy Annotation Interface")
    
    recommendations = st.session_state.get('extracted_recommendations', [])
    selected_frameworks = st.session_state.get('selected_frameworks', [])
    
    if not selected_frameworks:
        st.warning("⚠️ Please select at least one framework above.")
        return
    
    # Simple legacy interface
    if recommendations:
        st.info(f"📊 Ready to annotate {len(recommendations)} recommendations with {len(selected_frameworks)} frameworks")
        
        if st.button("🚀 Start Legacy Annotation", type="secondary"):
            run_comprehensive_annotation(recommendations, selected_frameworks)
    else:
        st.warning("No recommendations available for annotation.")

def display_annotation_results():
    """Main results display function"""
    results = st.session_state.get('annotation_results', {})
    
    if not results:
        st.info("💡 No annotations yet. Configure settings above and click 'Start Annotation' to begin.")
        return
    
    # Call the comprehensive display function
    display_annotation_results()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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

# =============================================================================
# ERROR HANDLING AND RECOVERY
# =============================================================================

def handle_annotation_error(item_id: str, error: Exception, error_list: List[str]):
    """Handle annotation errors gracefully"""
    error_msg = f"Error annotating {item_id}: {str(error)}"
    error_list.append(error_msg)
    add_error_message(error_msg)
    logging.error(f"Annotation error for {item_id}: {error}", exc_info=True)

def recover_from_annotation_failure(results: Dict, failed_items: List[str]):
    """Attempt to recover from annotation failures"""
    if failed_items:
        st.warning(f"⚠️ {len(failed_items)} items failed annotation. You can:")
        
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

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

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

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

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

# =============================================================================
# DEBUGGING AND DIAGNOSTICS
# =============================================================================

def show_annotation_diagnostics():
    """Show annotation diagnostics for debugging"""
    if st.checkbox("🔧 Show Annotation Diagnostics"):
        with st.expander("Annotation System Diagnostics"):
            # BERT annotator status
            annotator = st.session_state.get('bert_annotator')
            st.write(f"**BERT Annotator Status:** {'✅ Loaded' if annotator else '❌ Not loaded'}")
            
            if annotator:
                st.write(f"**Available Frameworks:** {list(annotator.frameworks.keys())}")
            
            # Session state info
            st.write(f"**Selected Frameworks:** {st.session_state.get('selected_frameworks', [])}")
            st.write(f"**Similarity Threshold:** {st.session_state.get('similarity_threshold', 'Not set')}")
            st.write(f"**Processing Status:** {st.session_state.get('processing_status', 'idle')}")
            
            # Results info
            results = st.session_state.get('annotation_results', {})
            st.write(f"**Annotation Results:** {len(results)} items")
            
            highlighted_texts = st.session_state.get('bert_results', {}).get('highlighted_texts', {})
            st.write(f"**Highlighted Texts:** {len(highlighted_texts)} documents")

def test_annotation_pipeline():
    """Test the annotation pipeline with sample data"""
    if st.button("🧪 Test Annotation Pipeline"):
        # Create test data
        test_item = {
            'id': 'test_item_001',
            'text': 'This is a test recommendation about improving communication protocols and training procedures.',
            'document_source': 'test_document.pdf'
        }
        
        test_frameworks = ['I-SIRch']
        
        st.info("Testing annotation pipeline with sample data...")
        
        try:
            # Test the annotation process
            run_comprehensive_annotation([test_item], test_frameworks)
            st.success("✅ Annotation pipeline test completed!")
            
        except Exception as e:
            st.error(f"❌ Annotation pipeline test failed: {e}")

# =============================================================================
# END OF FILE
# =============================================================================
