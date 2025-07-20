# ===============================================
# FILE: modules/ui/annotation_components.py
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

# EXACT REPLACEMENT for render_annotation_tab() function

def render_annotation_tab():
    """Render the concept annotation tab"""
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
    
    # Annotation interface
    render_annotation_interface()
    
    # Display annotation results
    display_annotation_results()

def initialize_bert_annotator():
    """Initialize BERT annotator in session state"""
    if not st.session_state.bert_annotator:
        with st.spinner("🧠 Loading BERT model... This may take a moment."):
            try:
                st.session_state.bert_annotator = BERTConceptAnnotator()
                st.success("✅ BERT model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load BERT model: {str(e)}")
                add_error_message(f"BERT initialization failed: {str(e)}")
                # Use fallback
                st.session_state.bert_annotator = BERTConceptAnnotator()

def render_framework_management():
    """Render framework selection and management"""
    st.subheader("🎯 Conceptual Frameworks")
    
    annotator = st.session_state.bert_annotator
    if not annotator:
        st.error("BERT annotator not initialized.")
        return
    
    # Available frameworks
    available_frameworks = list(annotator.frameworks.keys())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_frameworks = st.multiselect(
            "Select frameworks to use for annotation:",
            options=available_frameworks,
            default=available_frameworks,
            help="Choose which conceptual frameworks to apply",
            key="selected_frameworks_annotation"
        )
        
        # Store in session state
        st.session_state.selected_frameworks = selected_frameworks
    
    with col2:
        st.markdown("**Available Frameworks:**")
        for framework in available_frameworks:
            theme_count = len(annotator.frameworks.get(framework, []))
            st.write(f"• **{framework}:** {theme_count} themes")
    
    # Framework details
    if selected_frameworks:
        with st.expander("📋 Framework Details"):
            for framework in selected_frameworks:
                display_framework_details(framework, annotator.frameworks.get(framework, []))
    
    # Custom framework upload
    render_custom_framework_upload()

def display_framework_details(framework_name: str, themes: List[Dict]):
    """Display detailed information about a framework"""
    st.markdown(f"**{framework_name} Framework**")
    
    if not themes:
        st.warning(f"No themes available for {framework_name}")
        return
    
    # Create DataFrame for better display
    theme_data = []
    for theme in themes:
        theme_data.append({
            'Theme': theme.get('name', 'Unknown'),
            'Keywords': ', '.join(theme.get('keywords', [])),
            'Keyword Count': len(theme.get('keywords', []))
        })
    
    if theme_data:
        df = pd.DataFrame(theme_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_custom_framework_upload():
    """Render custom framework upload interface"""
    with st.expander("📁 Upload Custom Framework"):
        st.markdown("""
        Upload your own conceptual framework in JSON, CSV, or Excel format.
        
        **Expected Format:**
        - **JSON:** `{"themes": [{"name": "Theme Name", "keywords": ["keyword1", "keyword2"]}]}`
        - **CSV/Excel:** Columns: `theme`, `keywords` (comma-separated)
        """)
        
        custom_file = st.file_uploader(
            "Upload custom taxonomy",
            type=['json', 'csv', 'xlsx', 'xls'],
            help="Upload your own conceptual framework",
            key="custom_framework_upload"
        )
        
        if custom_file:
            if st.button("📤 Load Custom Framework", use_container_width=True):
                load_custom_framework(custom_file)

def load_custom_framework(custom_file):
    """Load custom framework from uploaded file"""
    annotator = st.session_state.bert_annotator
    if not annotator:
        st.error("BERT annotator not available.")
        return
    
    try:
        success, message = annotator.load_custom_framework(custom_file)
        
        if success:
            st.success(f"✅ {message}")
            # Add custom framework to selected frameworks if not already there
            if "Custom" not in st.session_state.get('selected_frameworks', []):
                current_selected = st.session_state.get('selected_frameworks', [])
                current_selected.append("Custom")
                st.session_state.selected_frameworks = current_selected
            st.rerun()
        else:
            st.error(f"❌ {message}")
            add_error_message(f"Custom framework loading failed: {message}")
    
    except Exception as e:
        error_msg = f"Error loading custom framework: {str(e)}"
        st.error(f"❌ {error_msg}")
        add_error_message(error_msg)
        logging.error(f"Custom framework error: {e}", exc_info=True)

def render_annotation_configuration():
    """Render annotation configuration options"""
    st.subheader("⚙️ Annotation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.3, 0.9, 0.65, 0.05,
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
                help="Number of recommendations to process at once",
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

def render_annotation_interface():
    """Render the main annotation interface"""
    st.subheader("🔬 Annotation Process")
    
    # Get both recommendations and concerns
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    selected_frameworks = st.session_state.get('selected_frameworks', [])
    
    # Determine what content to use based on user selection
    content_choice = st.session_state.get('annotation_content_choice', 'Recommendations')
    
    # Prepare items based on selection
    if content_choice == "Recommendations":
        items = recommendations
        item_type = "recommendations"
    elif content_choice == "Concerns":
        # Convert concerns to recommendation-like objects
        items = []
        for concern in concerns:
            pseudo_rec = type('obj', (object,), {
                'id': concern.get('id', 'unknown'),
                'text': concern.get('text', ''),
                'document_source': concern.get('document_source', 'unknown'),
                'section_title': 'Concern',
                'page_number': None,
                'confidence_score': concern.get('confidence_score', 0),
                'original_type': 'concern'
            })
            items.append(pseudo_rec)
        item_type = "concerns"
    else:  # Both
        items = list(recommendations)
        for concern in concerns:
            pseudo_rec = type('obj', (object,), {
                'id': concern.get('id', 'unknown'),
                'text': concern.get('text', ''),
                'document_source': concern.get('document_source', 'unknown'),
                'section_title': 'Concern',
                'page_number': None,
                'confidence_score': concern.get('confidence_score', 0),
                'original_type': 'concern'
            })
            items.append(pseudo_rec)
        item_type = "recommendations and concerns"
    
    if not selected_frameworks:
        st.warning("⚠️ Please select at least one framework above.")
        return
    
    # Selection options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        annotation_scope = st.radio(
            "What to annotate:",
            [f"All {item_type.title()}", f"Selected {item_type.title()}", "By Document"],
            help=f"Choose which {item_type} to annotate",
            key="annotation_scope"
        )
    
    with col2:
        st.markdown("**Annotation Info:**")
        st.write(f"• Total {item_type}: {len(items)}")
        st.write(f"• Selected frameworks: {len(selected_frameworks)}")
        st.write(f"• Similarity threshold: {st.session_state.get('similarity_threshold', 0.65):.2f}")
    
    # Item selection based on scope
    recommendations_to_annotate = []  # Keep same variable name for compatibility
    
    if annotation_scope == f"All {item_type.title()}":
        recommendations_to_annotate = items
    
    elif annotation_scope == f"Selected {item_type.title()}":
        # Allow user to select specific items
        rec_options = [f"{item.id}: {item.text[:50]}..." for item in items]
        selected_indices = st.multiselect(
            f"Select {item_type} to annotate:",
            range(len(rec_options)),
            format_func=lambda x: rec_options[x],
            key="selected_recommendations_indices"
        )
        recommendations_to_annotate = [items[i] for i in selected_indices]
    
    elif annotation_scope == "By Document":
        # Allow selection by document source
        doc_sources = list(set(item.document_source for item in items))
        selected_docs = st.multiselect(
            "Select documents:",
            doc_sources,
            key="selected_docs_annotation"
        )
        recommendations_to_annotate = [item for item in items if item.document_source in selected_docs]
    
    # Show what will be annotated
    if recommendations_to_annotate:
        st.info(f"📊 Ready to annotate {len(recommendations_to_annotate)} {item_type} with {len(selected_frameworks)} frameworks")
        
        # Annotation button
        if st.button("🚀 Start Annotation", type="primary", use_container_width=True):
            annotate_recommendations(recommendations_to_annotate, selected_frameworks)
    else:
        st.warning(f"No {item_type} selected for annotation.")

def annotate_recommendations(recommendations: List, frameworks: List[str]):
    """Perform annotation on selected recommendations"""
    if not recommendations or not frameworks:
        st.warning("No recommendations or frameworks selected.")
        return
    
    annotator = st.session_state.bert_annotator
    if not annotator:
        st.error("BERT annotator not available.")
        return
    
    # Get configuration
    similarity_threshold = st.session_state.get('similarity_threshold', 0.65)
    max_themes = st.session_state.get('max_themes_per_framework', 10)
    batch_size = st.session_state.get('annotation_batch_size', 10)
    
    # Update annotator configuration
    annotator.config["base_similarity_threshold"] = similarity_threshold
    annotator.config["max_themes_per_framework"] = max_themes
    
    # Progress tracking
    total_recommendations = len(recommendations)
    progress_container = st.container()
    status_container = st.container()
    
    annotation_results = {}
    successful_annotations = 0
    failed_annotations = []
    
    # Set processing status
    st.session_state.processing_status = "annotating"
    
    try:
        # Process in batches
        for batch_start in range(0, total_recommendations, batch_size):
            batch_end = min(batch_start + batch_size, total_recommendations)
            batch = recommendations[batch_start:batch_end]
            
            for i, rec in enumerate(batch):
                current_step = batch_start + i + 1
                
                # Update progress
                with progress_container:
                    show_progress_indicator(current_step, total_recommendations, f"Annotating {rec.id}")
                
                with status_container:
                    status_text = st.empty()
                    status_text.info(f"🏷️ Processing: {rec.id}")
                
                try:
                    # Perform annotation
                    framework_results, highlighting = annotator.annotate_text(rec.text, frameworks)
                    
                    # Store results
                    annotation_results[rec.id] = {
                        'recommendation': rec,
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
                    status_text.success(f"✅ Annotated: {rec.id}")
                    
                except Exception as e:
                    error_msg = f"Error annotating {rec.id}: {str(e)}"
                    failed_annotations.append(error_msg)
                    add_error_message(error_msg)
                    status_text.error(f"❌ Failed: {rec.id}")
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
        show_annotation_summary(successful_annotations, failed_annotations, total_recommendations)
    
    finally:
        st.session_state.processing_status = "idle"

def show_annotation_summary(successful: int, failed: List[str], total: int):
    """Show summary of annotation results"""
    if successful > 0:
        st.success(f"🎉 Successfully annotated {successful} of {total} recommendations!")
    
    if failed:
        st.error(f"❌ Failed to annotate {len(failed)} recommendations:")
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

def display_annotation_results():
    """Display annotation results with interactive features"""
    results = st.session_state.get('annotation_results', {})
    
    if not results:
        st.info("💡 No annotations yet. Configure settings above and click 'Start Annotation' to begin.")
        return
    
    st.subheader("🏷️ Annotation Results")
    
    # Results overview
    display_annotation_overview(results)
    
    # Detailed results
    display_detailed_annotations(results)
    
    # Export options
    render_annotation_export_options(results)

def display_annotation_overview(results: Dict):
    """Display overview statistics of annotation results"""
    # Calculate statistics
    total_recommendations = len(results)
    total_annotations = 0
    framework_counts = {}
    theme_counts = {}
    confidence_scores = []
    
    for result in results.values():
        annotations = result.get('annotations', {})
        for framework, themes in annotations.items():
            framework_counts[framework] = framework_counts.get(framework, 0) + len(themes)
            total_annotations += len(themes)
            
            for theme in themes:
                theme_name = f"{framework}: {theme['theme']}"
                theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
                confidence_scores.append(theme['confidence'])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annotated Recommendations", total_recommendations)
    
    with col2:
        st.metric("Total Annotations", total_annotations)
    
    with col3:
        st.metric("Frameworks Used", len(framework_counts))
    
    with col4:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Framework distribution
    if framework_counts:
        st.subheader("📊 Framework Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            framework_df = pd.DataFrame(
                list(framework_counts.items()),
                columns=['Framework', 'Annotation Count']
            )
            st.bar_chart(framework_df.set_index('Framework'))
        
        with col2:
            # Top themes
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_themes:
                st.markdown("**Top 10 Themes:**")
                for theme, count in top_themes:
                    st.write(f"• {theme}: {count}")
    
    # Confidence distribution
    if confidence_scores:
        st.subheader("📈 Confidence Distribution")
        confidence_df = pd.DataFrame({'Confidence': confidence_scores})
        st.histogram_chart(confidence_df, x='Confidence')

def display_detailed_annotations(results: Dict):
    """Display detailed annotation results"""
    st.subheader("📋 Detailed Annotations")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        framework_filter = st.selectbox(
            "Filter by Framework:",
            options=['All'] + list(set(
                framework 
                for result in results.values() 
                for framework in result.get('annotations', {}).keys()
            )),
            key="annotation_framework_filter"
        )
    
    with col2:
        confidence_filter = st.slider(
            "Min Confidence:",
            0.0, 1.0, 0.0, 0.05,
            key="annotation_confidence_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["Confidence (High to Low)", "Framework", "Recommendation ID"],
            key="annotation_sort_by"
        )
    
    # Create display data
    display_data = []
    
    for rec_id, result in results.items():
        rec = result['recommendation']
        annotations = result.get('annotations', {})
        
        for framework, themes in annotations.items():
            if framework_filter != 'All' and framework != framework_filter:
                continue
                
            for theme in themes:
                if theme['confidence'] >= confidence_filter:
                    display_data.append({
                        'rec_id': rec_id,
                        'recommendation': rec,
                        'framework': framework,
                        'theme': theme,
                        'full_result': result
                    })
    
    # Apply sorting
    if sort_by == "Confidence (High to Low)":
        display_data.sort(key=lambda x: x['theme']['confidence'], reverse=True)
    elif sort_by == "Framework":
        display_data.sort(key=lambda x: x['framework'])
    elif sort_by == "Recommendation ID":
        display_data.sort(key=lambda x: x['rec_id'])
    
    # Display results
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
            key="annotation_page"
        )
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_data = display_data[start_idx:end_idx]
    else:
        page_data = display_data
    
    # Display annotations
    for item in page_data:
        display_single_annotation(item)

def display_single_annotation(item: Dict):
    """Display a single annotation result"""
    rec = item['recommendation']
    theme = item['theme']
    framework = item['framework']
    
    # Color coding based on confidence
    if theme['confidence'] >= 0.8:
        confidence_color = "🟢"
    elif theme['confidence'] >= 0.6:
        confidence_color = "🟡"
    else:
        confidence_color = "🔴"
    
    with st.expander(f"{confidence_color} {rec.id} - {framework}: {theme['theme']} ({theme['confidence']:.2f})"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Recommendation Text:**")
            st.write(rec.text)
            
            st.markdown("**Matched Keywords:**")
            keywords = theme.get('matched_keywords', [])
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.write("No keywords matched")
        
        with col2:
            st.markdown("**Annotation Details:**")
            st.write(f"**Framework:** {framework}")
            st.write(f"**Theme:** {theme['theme']}")
            st.write(f"**Confidence:** {theme['confidence']:.3f}")
            st.write(f"**Semantic Similarity:** {theme.get('semantic_similarity', 0):.3f}")
            st.write(f"**Keyword Count:** {theme.get('keyword_count', 0)}")
            
            # Source information
            st.markdown("**Source:**")
            st.write(f"**Document:** {rec.document_source}")
            st.write(f"**Section:** {rec.section_title}")

def render_annotation_export_options(results: Dict):
    """Render export options for annotation results"""
    st.subheader("📥 Export Annotations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Summary CSV", use_container_width=True):
            export_annotation_summary(results)
    
    with col2:
        if st.button("📋 Export Detailed CSV", use_container_width=True):
            export_detailed_annotations(results)
    
    with col3:
        if st.button("📄 Export JSON", use_container_width=True):
            export_annotations_json(results)

def export_annotation_summary(results: Dict):
    """Export annotation summary as CSV"""
    if not results:
        st.warning("No annotations to export.")
        return
    
    # Prepare summary data
    summary_data = []
    
    for rec_id, result in results.items():
        rec = result['recommendation']
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
        
        summary_data.append({
            'Recommendation_ID': rec_id,
            'Recommendation_Text': rec.text[:100] + "..." if len(rec.text) > 100 else rec.text,
            'Source_Document': rec.document_source,
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

def export_detailed_annotations(results: Dict):
    """Export detailed annotations as CSV"""
    if not results:
        st.warning("No annotations to export.")
        return
    
    # Prepare detailed data
    detailed_data = []
    
    for rec_id, result in results.items():
        rec = result['recommendation']
        annotations = result.get('annotations', {})
        
        for framework, themes in annotations.items():
            for theme in themes:
                detailed_data.append({
                    'Recommendation_ID': rec_id,
                    'Recommendation_Text': rec.text,
                    'Source_Document': rec.document_source,
                    'Section': rec.section_title,
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

def export_annotations_json(results: Dict):
    """Export annotations as JSON"""
    if not results:
        st.warning("No annotations to export.")
        return
    
    # Prepare JSON data (make it serializable)
    export_data = {}
    
    for rec_id, result in results.items():
        rec = result['recommendation']
        
        export_data[rec_id] = {
            'recommendation': {
                'id': rec.id,
                'text': rec.text,
                'document_source': rec.document_source,
                'section_title': rec.section_title,
                'confidence_score': rec.confidence_score
            },
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
