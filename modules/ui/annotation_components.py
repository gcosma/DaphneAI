# ===============================================
# FILE: modules/ui/annotation_components.py
# Concept Annotation Components for DaphneAI
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core utilities
try:
    from modules.core_utils import log_user_action
    CORE_UTILS_AVAILABLE = True
except ImportError:
    CORE_UTILS_AVAILABLE = False
    def log_user_action(action: str, data: Dict = None):
        logger.info(f"Action: {action}, Data: {data}")

# ===============================================
# ANNOTATION FRAMEWORK DEFINITIONS
# ===============================================

GOVERNMENT_FRAMEWORKS = {
    "Policy Areas": [
        "Healthcare", "Education", "Transport", "Housing", "Environment",
        "Economy", "Defence", "Foreign Affairs", "Justice", "Social Care"
    ],
    "Implementation Status": [
        "Accepted", "Partially Accepted", "Rejected", "Under Review",
        "Implemented", "In Progress", "Not Started", "Delayed"
    ],
    "Priority Level": [
        "Critical", "High", "Medium", "Low"
    ],
    "Stakeholders": [
        "Department of Health", "Department for Education", "Home Office",
        "Treasury", "Cabinet Office", "Local Authorities", "NHS", "Private Sector"
    ],
    "Timeline": [
        "Immediate (0-6 months)", "Short-term (6-12 months)", 
        "Medium-term (1-2 years)", "Long-term (2+ years)"
    ]
}

def render_annotation_tab():
    """Render the concept annotation interface"""
    st.header("🏷️ Concept Annotation")
    st.markdown("""
    Annotate extracted recommendations and responses with government-specific categories and metadata.
    """)
    
    # Check if extraction results are available
    recommendations = st.session_state.get('extracted_recommendations', [])
    responses = st.session_state.get('extracted_responses', [])
    
    if not recommendations and not responses:
        st.warning("No extracted content available for annotation.")
        st.info("👆 Please extract content from documents first in the Extraction tab.")
        return
    
    # Content type selection
    st.markdown("### 📊 Select Content Type")
    content_type = st.radio(
        "Choose content to annotate:",
        ["Recommendations", "Responses"],
        horizontal=True
    )
    
    # Get content based on selection
    if content_type == "Recommendations":
        content_items = recommendations
        content_key = "recommendations"
    else:
        content_items = responses
        content_key = "responses"
    
    if not content_items:
        st.warning(f"No {content_type.lower()} available for annotation.")
        return
    
    st.markdown(f"### 🎯 Annotate {content_type}")
    st.info(f"Found {len(content_items)} {content_type.lower()} to annotate")
    
    # Annotation interface
    render_annotation_interface(content_items, content_type, content_key)

def render_annotation_interface(content_items: List[Dict], content_type: str, content_key: str):
    """Render the main annotation interface"""
    
    # Pagination for large datasets
    items_per_page = 5
    total_pages = (len(content_items) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(content_items))
        page_items = content_items[start_idx:end_idx]
    else:
        page_items = content_items
        start_idx = 0
    
    # Annotation form
    annotations = []
    
    for i, item in enumerate(page_items):
        item_idx = start_idx + i
        
        with st.expander(f"{content_type[:-1]} {item_idx + 1}: {item['text'][:100]}..."):
            st.markdown(f"**Full Text:** {item['text']}")
            st.markdown(f"**Source:** {item['document']}")
            st.markdown(f"**Confidence:** {item['confidence']:.2f}")
            
            # Annotation form
            col1, col2 = st.columns(2)
            
            with col1:
                # Policy area
                policy_area = st.selectbox(
                    "Policy Area",
                    ["Not Specified"] + GOVERNMENT_FRAMEWORKS["Policy Areas"],
                    key=f"policy_{item_idx}"
                )
                
                # Priority level
                priority = st.selectbox(
                    "Priority Level",
                    ["Not Specified"] + GOVERNMENT_FRAMEWORKS["Priority Level"],
                    key=f"priority_{item_idx}"
                )
                
                # Timeline
                timeline = st.selectbox(
                    "Implementation Timeline",
                    ["Not Specified"] + GOVERNMENT_FRAMEWORKS["Timeline"],
                    key=f"timeline_{item_idx}"
                )
            
            with col2:
                # Implementation status (for recommendations)
                if content_type == "Recommendations":
                    status = st.selectbox(
                        "Implementation Status",
                        ["Not Specified"] + GOVERNMENT_FRAMEWORKS["Implementation Status"],
                        key=f"status_{item_idx}"
                    )
                else:
                    status = "Not Applicable"
                
                # Stakeholders
                stakeholders = st.multiselect(
                    "Relevant Stakeholders",
                    GOVERNMENT_FRAMEWORKS["Stakeholders"],
                    key=f"stakeholders_{item_idx}"
                )
                
                # Custom tags
                custom_tags = st.text_input(
                    "Custom Tags (comma-separated)",
                    key=f"tags_{item_idx}",
                    help="Add custom tags separated by commas"
                )
            
            # Notes
            notes = st.text_area(
                "Additional Notes",
                key=f"notes_{item_idx}",
                height=100
            )
            
            # Create annotation object
            annotation = {
                'item_index': item_idx,
                'original_item': item,
                'policy_area': policy_area,
                'priority': priority,
                'timeline': timeline,
                'implementation_status': status,
                'stakeholders': stakeholders,
                'custom_tags': [tag.strip() for tag in custom_tags.split(',') if tag.strip()],
                'notes': notes,
                'annotated_at': datetime.now().isoformat(),
                'annotated_by': 'user'  # Could be enhanced with user management
            }
            annotations.append(annotation)
    
    # Save annotations
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save Annotations", type="primary"):
            save_annotations(annotations, content_key)
    
    with col2:
        if st.button("📊 View Summary"):
            show_annotation_summary(content_key)
    
    with col3:
        if st.button("📥 Export Annotations"):
            export_annotations(content_key)

def save_annotations(annotations: List[Dict], content_key: str):
    """Save annotations to session state"""
    try:
        # Initialize annotation storage if needed
        if 'annotation_results' not in st.session_state:
            st.session_state.annotation_results = {}
        
        # Save annotations
        st.session_state.annotation_results[content_key] = annotations
        
        # Update original items with annotations
        if content_key == "recommendations":
            original_items = st.session_state.get('extracted_recommendations', [])
        else:
            original_items = st.session_state.get('extracted_responses', [])
        
        # Add annotations to original items
        for annotation in annotations:
            item_idx = annotation['item_index']
            if item_idx < len(original_items):
                original_items[item_idx]['annotations'] = annotation
        
        st.success(f"✅ Saved annotations for {len(annotations)} items")
        
        # Log action
        if CORE_UTILS_AVAILABLE:
            log_user_action("annotations_saved", {
                'content_type': content_key,
                'annotation_count': len(annotations)
            })
            
    except Exception as e:
        st.error(f"❌ Failed to save annotations: {str(e)}")
        logger.error(f"Annotation save error: {e}")

def show_annotation_summary(content_key: str):
    """Show summary of annotations"""
    annotations = st.session_state.get('annotation_results', {}).get(content_key, [])
    
    if not annotations:
        st.warning("No annotations available for summary.")
        return
    
    st.markdown("### 📊 Annotation Summary")
    
    # Policy area distribution
    policy_areas = {}
    priorities = {}
    timelines = {}
    statuses = {}
    
    for annotation in annotations:
        # Policy areas
        pa = annotation['policy_area']
        if pa != "Not Specified":
            policy_areas[pa] = policy_areas.get(pa, 0) + 1
        
        # Priorities
        pr = annotation['priority']
        if pr != "Not Specified":
            priorities[pr] = priorities.get(pr, 0) + 1
        
        # Timelines
        tl = annotation['timeline']
        if tl != "Not Specified":
            timelines[tl] = timelines.get(tl, 0) + 1
        
        # Implementation status
        st = annotation.get('implementation_status')
        if st and st != "Not Specified" and st != "Not Applicable":
            statuses[st] = statuses.get(st, 0) + 1
    
    # Display summaries
    col1, col2 = st.columns(2)
    
    with col1:
        if policy_areas:
            st.markdown("#### Policy Area Distribution")
            pa_df = pd.DataFrame([
                {'Policy Area': area, 'Count': count}
                for area, count in policy_areas.items()
            ])
            st.dataframe(pa_df)
        
        if priorities:
            st.markdown("#### Priority Distribution")
            pr_df = pd.DataFrame([
                {'Priority': priority, 'Count': count}
                for priority, count in priorities.items()
            ])
            st.dataframe(pr_df)
    
    with col2:
        if timelines:
            st.markdown("#### Timeline Distribution")
            tl_df = pd.DataFrame([
                {'Timeline': timeline, 'Count': count}
                for timeline, count in timelines.items()
            ])
            st.dataframe(tl_df)
        
        if statuses:
            st.markdown("#### Implementation Status")
            st_df = pd.DataFrame([
                {'Status': status, 'Count': count}
                for status, count in statuses.items()
            ])
            st.dataframe(st_df)

def export_annotations(content_key: str):
    """Export annotations to CSV"""
    try:
        annotations = st.session_state.get('annotation_results', {}).get(content_key, [])
        
        if not annotations:
            st.warning("No annotations to export.")
            return
        
        # Prepare export data
        export_data = []
        for annotation in annotations:
            item = annotation['original_item']
            
            export_data.append({
                'Text': item['text'],
                'Document': item['document'],
                'Original_Confidence': item['confidence'],
                'Policy_Area': annotation['policy_area'],
                'Priority': annotation['priority'],
                'Timeline': annotation['timeline'],
                'Implementation_Status': annotation['implementation_status'],
                'Stakeholders': '; '.join(annotation['stakeholders']),
                'Custom_Tags': '; '.join(annotation['custom_tags']),
                'Notes': annotation['notes'],
                'Annotated_At': annotation['annotated_at']
            })
        
        # Create DataFrame and CSV
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        
        # Download button
        filename = f"annotations_{content_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button(
            label=f"📥 Download {content_key.title()} Annotations",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        st.success(f"✅ {content_key.title()} annotations ready for download")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        logger.error(f"Annotation export error: {e}")

# ===============================================
# INITIALIZATION
# ===============================================

# Initialize session state for annotations
if 'annotation_results' not in st.session_state:
    st.session_state.annotation_results = {}

logger.info("✅ Annotation components initialized")
