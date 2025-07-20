# ===============================================
# FILE: modules/ui/annotation_components.py
# COMPLETE SOLUTION FOR RECOMMENDATIONS AND CONCERNS ANNOTATION
# ===============================================

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

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
# MAIN TAB RENDERING FUNCTION - UPDATED FOR RECOMMENDATIONS AND CONCERNS
# =============================================================================

def render_annotation_tab():
    """Render the enhanced concept annotation tab supporting both recommendations and concerns"""
    st.header("🏷️ Concept Annotation")
    
    # Check for both recommendations and concerns
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not recommendations and not concerns:
        st.warning("⚠️ Please extract recommendations or concerns first in the Extract Content tab.")
        return
    
    # Let user choose what to annotate
    st.markdown("### 📋 Select Content to Annotate")
    
    # Content selection
    content_options = []
    if recommendations:
        content_options.append("Recommendations")
    if concerns:
        content_options.append("Concerns")
    if recommendations and concerns:
        content_options.append("Both")
    
    if len(content_options) > 1:
        content_choice = st.radio(
            "What would you like to annotate?",
            content_options,
            horizontal=True,
            key="annotation_content_choice"
        )
    else:
        content_choice = content_options[0] if content_options else "None"
    
    # Show what's available
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Recommendations Available", len(recommendations))
    with col2:
        st.metric("⚠️ Concerns Available", len(concerns))
    with col3:
        total_items = len(recommendations) + len(concerns)
        st.metric("📊 Total Items", total_items)
    
    st.markdown(f"""
    Annotate {content_choice.lower()} with conceptual themes using BERT-based analysis and 
    established frameworks like I-SIRch and House of Commons.
    """)
    
    # Initialize BERT annotator
    initialize_bert_annotator()
    
    # Framework management
    render_framework_management()
    
    # Annotation configuration
    render_annotation_configuration()
    
    # Enhanced annotation interface that handles the content choice
    render_enhanced_annotation_interface(content_choice, recommendations, concerns)
    
    # Display annotation results
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
                # Use fallback
                st.session_state.bert_annotator = BERTConceptAnnotator()

# =============================================================================
# FRAMEWORK MANAGEMENT FUNCTIONS
# =============================================================================

def render_framework_management():
    """Render framework selection and management"""
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
    
    # Framework selection - DON'T manually set session state after widget creation
    selected_frameworks = st.multiselect(
        "Select frameworks to use for annotation:",
        available_frameworks,
        default=available_frameworks[:1] if available_frameworks else [],
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
    with st.expander("📁 Upload Custom Framework"):
        uploaded_file = st.file_uploader(
            "Upload custom framework file:",
            type=['json', 'csv', 'xlsx'],
            help="Upload your own conceptual framework",
            key="custom_framework_upload"
        )
        
        if uploaded_file and st.button("Load Custom Framework"):
            try:
                success, message = annotator.load_custom_framework(uploaded_file)
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ Error loading framework: {str(e)}")
                
# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def render_annotation_configuration():
    """Render annotation configuration settings"""
    st.subheader("⚙️ Annotation Settings")
    
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
            ["Semantic Only", "Semantic + Keywords", "Keywords Only"],
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
# ENHANCED ANNOTATION INTERFACE FUNCTIONS
# =============================================================================

def render_enhanced_annotation_interface(content_choice, recommendations, concerns):
    """Render the annotation interface based on content choice"""
    st.subheader("🔬 Annotation Process")
    
    # Get selected frameworks from session state (managed by the widget)
    selected_frameworks = st.session_state.get('selected_frameworks', [])
    
    if not selected_frameworks:
        st.warning("⚠️ Please select at least one framework above.")
        return
    
    # Determine items to process based on content choice
    items_to_annotate = []
    
    if content_choice == "Recommendations":
        items_to_annotate = render_recommendation_selection_interface(recommendations)
        
    elif content_choice == "Concerns":
        items_to_annotate = render_concern_selection_interface(concerns)
        
    elif content_choice == "Both":
        items_to_annotate = render_mixed_content_interface(recommendations, concerns)
    
    # Show annotation summary and start button
    if items_to_annotate:
        content_type_summary = get_content_type_summary(items_to_annotate)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📊 Ready to annotate {len(items_to_annotate)} items ({content_type_summary}) with {len(selected_frameworks)} frameworks")
        
        with col2:
            st.markdown("**Annotation Settings:**")
            threshold = st.session_state.get('similarity_threshold', 0.65)
            max_themes = st.session_state.get('max_themes_per_framework', 10)
            st.write(f"• Threshold: {threshold:.2f}")
            st.write(f"• Max themes: {max_themes}")
        
        # Start annotation button
        if st.button("🚀 Start Annotation", type="primary", use_container_width=True):
            annotate_content_items(items_to_annotate, selected_frameworks)
    
    else:
        st.info("Select items above to proceed with annotation.")

def render_recommendation_selection_interface(recommendations):
    """Handle recommendation selection sub-interface"""
    if not recommendations:
        return []
    
    selection_method = st.radio(
        "How to select recommendations:",
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
            default=doc_sources,  # Select all by default
            key="selected_rec_docs"
        )
        return [rec for rec in recommendations if rec.document_source in selected_docs]
        
    elif selection_method == "Individual Selection":
        rec_options = []
        for i, rec in enumerate(recommendations):
            preview = rec.text[:60] + "..." if len(rec.text) > 60 else rec.text
            doc_short = rec.document_source.split('/')[-1] if '/' in rec.document_source else rec.document_source
            rec_options.append(f"📋 {rec.id} | {doc_short} | {preview}")
        
        selected_indices = st.multiselect(
            f"Select recommendations ({len(recommendations)} available):",
            range(len(rec_options)),
            format_func=lambda x: rec_options[x],
            key="individual_rec_selection"
        )
        return [recommendations[i] for i in selected_indices]
    
    return []

def render_concern_selection_interface(concerns):
    """Handle concern selection sub-interface"""
    if not concerns:
        return []
    
    selection_method = st.radio(
        "How to select concerns:",
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
            default=doc_sources,  # Select all by default
            key="selected_concern_docs"
        )
        return [concern for concern in concerns if concern.get('document_source', 'Unknown') in selected_docs]
        
    elif selection_method == "Individual Selection":
        concern_options = []
        for i, concern in enumerate(concerns):
            concern_id = concern.get('id', f'concern_{i}')
            concern_text = concern.get('text', '')
            preview = concern_text[:60] + "..." if len(concern_text) > 60 else concern_text
            doc_source = concern.get('document_source', 'Unknown')
            doc_short = doc_source.split('/')[-1] if '/' in doc_source else doc_source
            concern_options.append(f"⚠️ {concern_id} | {doc_short} | {preview}")
        
        selected_indices = st.multiselect(
            f"Select concerns ({len(concerns)} available):",
            range(len(concern_options)),
            format_func=lambda x: concern_options[x],
            key="individual_concern_selection"
        )
        return [concerns[i] for i in selected_indices]
    
    return []

def render_mixed_content_interface(recommendations, concerns):
    """Handle mixed content selection interface"""
    st.markdown("**Select content to annotate:**")
    
    col1, col2 = st.columns(2)
    
    selected_items = []
    
    with col1:
        st.markdown("**📋 Recommendations:**")
        if recommendations:
            include_all_recs = st.checkbox(
                f"Include all recommendations ({len(recommendations)})",
                key="include_all_recs"
            )
            
            if include_all_recs:
                selected_items.extend(recommendations)
            else:
                # Show document-level selection for recommendations
                rec_docs = list(set(rec.document_source for rec in recommendations))
                selected_rec_docs = st.multiselect(
                    "Select recommendation documents:",
                    rec_docs,
                    key="mixed_rec_docs"
                )
                selected_items.extend([rec for rec in recommendations if rec.document_source in selected_rec_docs])
        else:
            st.write("No recommendations available")
    
    with col2:
        st.markdown("**⚠️ Concerns:**")
        if concerns:
            include_all_concerns = st.checkbox(
                f"Include all concerns ({len(concerns)})",
                key="include_all_concerns"
            )
            
            if include_all_concerns:
                selected_items.extend(concerns)
            else:
                # Show document-level selection for concerns
                concern_docs = list(set(concern.get('document_source', 'Unknown') for concern in concerns))
                selected_concern_docs = st.multiselect(
                    "Select concern documents:",
                    concern_docs,
                    key="mixed_concern_docs"
                )
                selected_items.extend([concern for concern in concerns if concern.get('document_source', 'Unknown') in selected_concern_docs])
        else:
            st.write("No concerns available")
    
    return selected_items

# =============================================================================
# ANNOTATION PROCESSING FUNCTIONS
# =============================================================================

def annotate_content_items(items, frameworks):
    """Perform annotation on selected content items (recommendations and/or concerns)"""
    if not items or not frameworks:
        st.warning("No items or frameworks selected.")
        return
    
    annotator = st.session_state.get('bert_annotator')
    if not annotator:
        st.error("BERT annotator not available.")
        return
    
    # Get configuration
    similarity_threshold = st.session_state.get('similarity_threshold', 0.65)
    max_themes = st.session_state.get('max_themes_per_framework', 10)
    batch_size = st.session_state.get('annotation_batch_size', 10)
    
    # Update annotator configuration
    if hasattr(annotator, 'config'):
        annotator.config["base_similarity_threshold"] = similarity_threshold
        annotator.config["max_themes_per_framework"] = max_themes
    
    # Progress tracking
    total_items = len(items)
    progress_container = st.container()
    status_container = st.container()
    
    annotation_results = {}
    successful_annotations = 0
    failed_annotations = []
    
    # Set processing status
    st.session_state.processing_status = "annotating"
    
    try:
        # Process items
        for i, item in enumerate(items):
            current_step = i + 1
            
            # Determine item type and get ID/text
            item_id, item_text, item_type = extract_item_info(item)
            
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
                        'max_themes': max_themes
                    }
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
            # Merge with existing results
            existing_results = st.session_state.get('annotation_results', {})
            existing_results.update(annotation_results)
            st.session_state.annotation_results = existing_results
        
        # Clear progress displays
        progress_container.empty()
        status_container.empty()
        
        # Show results summary
        show_enhanced_annotation_summary(successful_annotations, failed_annotations, total_items, items)
    
    finally:
        st.session_state.processing_status = "idle"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_item_info(item):
    """Extract ID, text, and type from content item (recommendation or concern)"""
    if hasattr(item, 'id') and hasattr(item, 'text'):
        # This is a recommendation object
        return item.id, item.text, "recommendation"
    elif isinstance(item, dict):
        # This is a concern dictionary
        item_id = item.get('id', f"concern_{hash(item.get('text', ''))}")
        item_text = item.get('text', '')
        return item_id, item_text, "concern"
    else:
        # Fallback
        return str(hash(str(item))), str(item), "unknown"

def get_content_type_summary(items):
    """Get summary of content types in the selection"""
    if not items:
        return "no items"
    
    recommendations_count = 0
    concerns_count = 0
    
    for item in items:
        if hasattr(item, 'id') and hasattr(item, 'text') and hasattr(item, 'document_source'):
            # This looks like a recommendation object
            recommendations_count += 1
        else:
            # This looks like a concern dictionary
            concerns_count += 1
    
    summary_parts = []
    if recommendations_count > 0:
        summary_parts.append(f"{recommendations_count} recommendations")
    if concerns_count > 0:
        summary_parts.append(f"{concerns_count} concerns")
    
    return " + ".join(summary_parts) if summary_parts else "no items"

def show_enhanced_annotation_summary(successful: int, failed: List[str], total: int, items: List):
    """Show enhanced summary of annotation results"""
    if successful > 0:
        st.success(f"🎉 Successfully annotated {successful} of {total} items!")
    
    # Show content type breakdown
    content_summary = get_content_type_summary(items)
    st.info(f"📊 Processed: {content_summary}")
    
    if failed:
        st.error(f"❌ Failed to annotate {len(failed)} items:")
        for error in failed[:5]:  # Show first 5 errors
            st.write(f"• {error}")
        
        if len(failed) > 5:
            st.write(f"... and {len(failed) - 5} more errors")
    
    # Quick stats
    if successful > 0:
        results = st.session_state.get('annotation_results', {})
        if results:
            total_annotations = 0
            for result in results.values():
                for framework, themes in result['annotations'].items():
                    total_annotations += len(themes)
            
            st.info(f"📊 Total annotations created: {total_annotations}")

# =============================================================================
# RESULTS DISPLAY FUNCTIONS - ENHANCED FOR MIXED CONTENT
# =============================================================================

def display_annotation_results():
    """Display annotation results with interactive features for both recommendations and concerns"""
    results = st.session_state.get('annotation_results', {})
    
    if not results:
        st.info("💡 No annotations yet. Configure settings above and click 'Start Annotation' to begin.")
        return
    
    st.subheader("🏷️ Annotation Results")
    
    # Results overview
    display_enhanced_annotation_overview(results)
    
    # Detailed results
    display_enhanced_detailed_annotations(results)
    
    # Export options
    render_annotation_export_options(results)

def display_enhanced_annotation_overview(results: Dict):
    """Display overview statistics of annotation results for mixed content"""
    # Calculate statistics
    total_items = len(results)
    total_annotations = 0
    framework_counts = {}
    theme_counts = {}
    confidence_scores = []
    content_type_counts = {'recommendation': 0, 'concern': 0, 'unknown': 0}
    
    for result in results.values():
        # Count content types
        content_type = result.get('content_type', 'unknown')
        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        annotations = result.get('annotations', {})
        for framework, themes in annotations.items():
            framework_counts[framework] = framework_counts.get(framework, 0) + len(themes)
            total_annotations += len(themes)
            
            for theme in themes:
                theme_name = f"{framework}: {theme['theme']}"
                theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
                confidence_scores.append(theme['confidence'])
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Items", total_items)
    
    with col2:
        st.metric("📋 Recommendations", content_type_counts.get('recommendation', 0))
    
    with col3:
        st.metric("⚠️ Concerns", content_type_counts.get('concern', 0))
    
    with col4:
        st.metric("Total Annotations", total_annotations)
    
    with col5:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Framework breakdown
    if framework_counts:
        st.markdown("**Annotations by Framework:**")
        framework_df = pd.DataFrame([
            {'Framework': fw, 'Annotations': count} 
            for fw, count in framework_counts.items()
        ])
        st.dataframe(framework_df, use_container_width=True, hide_index=True)

def display_enhanced_detailed_annotations(results: Dict):
    """Display detailed annotation results with enhanced filtering for mixed content"""
    st.markdown("### 📋 Detailed Annotations")
    
    # Enhanced filtering options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Content type filter
        content_types = set()
        for result in results.values():
            content_types.add(result.get('content_type', 'unknown'))
        
        content_filter = st.selectbox(
            "Filter by content type:",
            ['All'] + sorted(list(content_types)),
            key="detailed_content_filter"
        )
    
    with col2:
        # Framework filter
        frameworks_used = set()
        for result in results.values():
            frameworks_used.update(result.get('frameworks_used', []))
        
        framework_filter = st.selectbox(
            "Filter by framework:",
            ['All'] + sorted(list(frameworks_used)),
            key="detailed_framework_filter"
        )
    
    with col3:
        # Confidence filter
        confidence_filter = st.slider(
            "Min confidence:",
            0.0, 1.0, 0.0, 0.05,
            key="detailed_confidence_filter"
        )
    
    # Apply filters and create display data
    display_data = []
    
    for item_id, result in results.items():
        # Apply content type filter
        if content_filter != 'All' and result.get('content_type') != content_filter:
            continue
        
        # Apply framework filter
        if framework_filter != 'All' and framework_filter not in result.get('frameworks_used', []):
            continue
        
        # Get item info
        content_item = result.get('content_item')
        content_type = result.get('content_type', 'unknown')
        annotations = result.get('annotations', {})
        
        # Process annotations
        for framework, themes in annotations.items():
            for theme in themes:
                # Apply confidence filter
                if theme.get('confidence', 0) < confidence_filter:
                    continue
                
                display_data.append({
                    'item_id': item_id,
                    'content_type': content_type,
                    'content_item': content_item,
                    'framework': framework,
                    'theme': theme,
                    'result': result
                })
    
    if not display_data:
        st.warning("No annotations match the current filters.")
        return
    
    st.write(f"Showing {len(display_data)} annotations")
    
    # Paginated display
    items_per_page = 10
    total_pages = (len(display_data) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.selectbox(
            "Page:",
            range(1, total_pages + 1),
            key="detailed_annotation_page"
        )
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_data = display_data[start_idx:end_idx]
    else:
        page_data = display_data
    
    # Display annotations
    for item in page_data:
        display_enhanced_single_annotation(item)

def display_enhanced_single_annotation(item: Dict):
    """Display a single annotation result for recommendations or concerns"""
    item_id = item['item_id']
    content_type = item['content_type']
    content_item = item['content_item']
    theme = item['theme']
    framework = item['framework']
    
    # Get icon and color based on content type
    if content_type == 'recommendation':
        icon = "📋"
    elif content_type == 'concern':
        icon = "⚠️"
    else:
        icon = "📄"
    
    # Color coding based on confidence
    confidence = theme.get('confidence', 0)
    if confidence >= 0.8:
        confidence_color = "🟢"
    elif confidence >= 0.6:
        confidence_color = "🟡"
    else:
        confidence_color = "🔴"
    
    with st.expander(f"{confidence_color} {icon} {item_id} - {framework}: {theme['theme']} ({confidence:.2f})"):
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
            keywords = theme.get('matched_keywords', [])
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.write("No keywords matched")
        
        with col2:
            st.markdown("**Annotation Details:**")
            st.write(f"**Type:** {content_type.title()}")
            st.write(f"**Framework:** {framework}")
            st.write(f"**Theme:** {theme['theme']}")
            st.write(f"**Confidence:** {confidence:.3f}")
            st.write(f"**Semantic Similarity:** {theme.get('semantic_similarity', 0):.3f}")
            st.write(f"**Keyword Count:** {theme.get('keyword_count', 0)}")
            
            # Source information
            st.markdown("**Source:**")
            if content_type == 'recommendation' and hasattr(content_item, 'document_source'):
                st.write(f"**Document:** {content_item.document_source}")
                if hasattr(content_item, 'section_title'):
                    st.write(f"**Section:** {content_item.section_title}")
            elif content_type == 'concern' and isinstance(content_item, dict):
                st.write(f"**Document:** {content_item.get('document_source', 'Unknown')}")
                if content_item.get('type'):
                    st.write(f"**Concern Type:** {content_item.get('type')}")

# =============================================================================
# EXPORT FUNCTIONS - ENHANCED FOR MIXED CONTENT
# =============================================================================

def render_annotation_export_options(results: Dict):
    """Render export options for annotation results"""
    st.subheader("📥 Export Annotations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Summary CSV", use_container_width=True):
            export_enhanced_annotation_summary(results)
    
    with col2:
        if st.button("📋 Export Detailed CSV", use_container_width=True):
            export_enhanced_detailed_annotations(results)
    
    with col3:
        if st.button("📄 Export JSON", use_container_width=True):
            export_enhanced_annotations_json(results)

def export_enhanced_annotation_summary(results: Dict):
    """Export enhanced annotation summary as CSV for mixed content"""
    if not results:
        st.warning("No annotations to export.")
        return
    
    # Prepare summary data
    summary_data = []
    
    for item_id, result in results.items():
        content_item = result['content_item']
        content_type = result.get('content_type', 'unknown')
        annotations = result.get('annotations', {})
        
        # Count annotations by framework
        framework_counts = {}
        total_annotations = 0
        avg_confidence = 0
        confidences = []
        
        for framework, themes in annotations.items():
            framework_counts[framework] = len(themes)
            total_annotations += len(themes)
            for theme in themes:
                confidences.append(theme['confidence'])
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        
        # Get text based on content type
        if content_type == 'recommendation' and hasattr(content_item, 'text'):
            text = content_item.text
            source_doc = getattr(content_item, 'document_source', 'Unknown')
        elif content_type == 'concern' and isinstance(content_item, dict):
            text = content_item.get('text', '')
            source_doc = content_item.get('document_source', 'Unknown')
        else:
            text = str(content_item)
            source_doc = 'Unknown'
        
        summary_data.append({
            'Item_ID': item_id,
            'Content_Type': content_type.title(),
            'Item_Text': text[:100] + "..." if len(text) > 100 else text,
            'Source_Document': source_doc,
            'Total_Annotations': total_annotations,
            'Frameworks_Applied': len(framework_counts),
            'Average_Confidence': avg_confidence,
            'Annotation_Time': result.get('annotation_time', ''),
            **{f'{fw}_Count': count for fw, count in framework_counts.items()}
        })
    
    df = pd.DataFrame(summary_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Summary CSV",
        data=csv,
        file_name=f"annotation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_enhanced_detailed_annotations(results: Dict):
    """Export enhanced detailed annotations as CSV for mixed content"""
    if not results:
        st.warning("No annotations to export.")
        return
    
    # Prepare detailed data
    detailed_data = []
    
    for item_id, result in results.items():
        content_item = result['content_item']
        content_type = result.get('content_type', 'unknown')
        annotations = result.get('annotations', {})
        
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
                    'Annotation_Time': result.get('annotation_time', '')
                })
    
    df = pd.DataFrame(detailed_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Detailed CSV",
        data=csv,
        file_name=f"detailed_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def export_enhanced_annotations_json(results: Dict):
    """Export enhanced annotations as JSON for mixed content"""
    if not results:
        st.warning("No annotations to export.")
        return
    
    # Prepare JSON data (make it serializable)
    export_data = {}
    
    for item_id, result in results.items():
        content_item = result['content_item']
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
        label="📥 Download JSON",
        data=json_str.encode('utf-8'),
        file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

# =============================================================================
# LEGACY INTERFACE SUPPORT FUNCTIONS
# =============================================================================

def render_annotation_interface():
    """Legacy annotation interface - maintained for backward compatibility"""
    st.subheader("🔬 Annotation Process")
    
    recommendations = st.session_state.get('extracted_recommendations', [])
    selected_frameworks = st.session_state.get('selected_frameworks', [])
    
    if not selected_frameworks:
        st.warning("⚠️ Please select at least one framework above.")
        return
    
    # Selection options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        annotation_scope = st.radio(
            "What to annotate:",
            ["All Recommendations", "Selected Recommendations", "By Document"],
            help="Choose which recommendations to annotate",
            key="legacy_annotation_scope"
        )
    
    with col2:
        st.markdown("**Annotation Info:**")
        st.write(f"• Total recommendations: {len(recommendations)}")
        st.write(f"• Selected frameworks: {len(selected_frameworks)}")
        st.write(f"• Similarity threshold: {st.session_state.get('similarity_threshold', 0.65):.2f}")
    
    # Recommendation selection based on scope
    recommendations_to_annotate = []
    
    if annotation_scope == "All Recommendations":
        recommendations_to_annotate = recommendations
    
    elif annotation_scope == "Selected Recommendations":
        # Allow user to select specific recommendations
        rec_options = [f"{rec.id}: {rec.text[:50]}..." for rec in recommendations]
        selected_indices = st.multiselect(
            "Select recommendations to annotate:",
            range(len(rec_options)),
            format_func=lambda x: rec_options[x],
            key="legacy_selected_recommendations_indices"
        )
        recommendations_to_annotate = [recommendations[i] for i in selected_indices]
    
    elif annotation_scope == "By Document":
        # Allow selection by document source
        doc_sources = list(set(rec.document_source for rec in recommendations))
        selected_docs = st.multiselect(
            "Select documents:",
            doc_sources,
            key="legacy_selected_docs_annotation"
        )
        recommendations_to_annotate = [rec for rec in recommendations if rec.document_source in selected_docs]
    
    # Show what will be annotated
    if recommendations_to_annotate:
        st.info(f"📊 Ready to annotate {len(recommendations_to_annotate)} recommendations with {len(selected_frameworks)} frameworks")
        
        # Annotation button
        if st.button("🚀 Start Annotation (Legacy)", type="secondary", use_container_width=True):
            # Convert to new format and use new function
            annotate_content_items(recommendations_to_annotate, selected_frameworks)
    else:
        st.warning("No recommendations selected for annotation.")

def annotate_recommendations(recommendations: List, frameworks: List[str]):
    """Legacy annotation function - redirects to new enhanced function"""
    st.warning("⚠️ Using legacy annotation function. Consider updating to the enhanced version.")
    annotate_content_items(recommendations, frameworks)

def show_annotation_summary(successful: int, failed: List[str], total: int):
    """Legacy summary function - redirects to enhanced version"""
    # Create dummy items list for compatibility
    dummy_items = [{'type': 'recommendation'} for _ in range(total)]
    show_enhanced_annotation_summary(successful, failed, total, dummy_items)
