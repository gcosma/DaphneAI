# ===============================================
# FILE: modules/ui/dashboard_components.py
# Analytics Dashboard Components for DaphneAI
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

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
# DASHBOARD ANALYTICS CLASS
# ===============================================

class DashboardAnalytics:
    """Analytics engine for dashboard metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_document_metrics(self, documents: List[Dict]) -> Dict:
        """Calculate document-related metrics"""
        if not documents:
            return {
                'total_documents': 0,
                'total_size_mb': 0,
                'avg_document_size': 0,
                'file_types': {},
                'document_count_by_date': {}
            }
        
        total_size = sum(doc['metadata']['file_size'] for doc in documents)
        file_types = {}
        
        for doc in documents:
            file_type = doc['metadata'].get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_size_mb': total_size / (1024 * 1024),
            'avg_document_size': total_size / len(documents) if documents else 0,
            'file_types': file_types,
            'largest_document': max(documents, key=lambda x: x['metadata']['file_size'])['filename'] if documents else None
        }
    
    def calculate_extraction_metrics(self, recommendations: List[Dict], responses: List[Dict]) -> Dict:
        """Calculate extraction-related metrics"""
        metrics = {
            'total_recommendations': len(recommendations),
            'total_responses': len(responses),
            'avg_recommendation_confidence': 0,
            'avg_response_confidence': 0,
            'extraction_methods': {},
            'document_coverage': {}
        }
        
        # Confidence calculations
        if recommendations:
            metrics['avg_recommendation_confidence'] = sum(r['confidence'] for r in recommendations) / len(recommendations)
            
            # Method distribution
            for rec in recommendations:
                method = rec.get('method', 'unknown')
                metrics['extraction_methods'][method] = metrics['extraction_methods'].get(method, 0) + 1
        
        if responses:
            metrics['avg_response_confidence'] = sum(r['confidence'] for r in responses) / len(responses)
        
        # Document coverage
        all_items = recommendations + responses
        for item in all_items:
            doc = item.get('document', 'unknown')
            metrics['document_coverage'][doc] = metrics['document_coverage'].get(doc, 0) + 1
        
        return metrics
    
    def calculate_annotation_metrics(self, annotation_results: Dict) -> Dict:
        """Calculate annotation-related metrics"""
        metrics = {
            'total_annotations': 0,
            'policy_areas': {},
            'priorities': {},
            'implementation_status': {},
            'annotation_coverage': 0
        }
        
        all_annotations = []
        for content_type, annotations in annotation_results.items():
            all_annotations.extend(annotations)
        
        metrics['total_annotations'] = len(all_annotations)
        
        for annotation in all_annotations:
            # Policy areas
            policy_area = annotation.get('policy_area', 'Not Specified')
            if policy_area != 'Not Specified':
                metrics['policy_areas'][policy_area] = metrics['policy_areas'].get(policy_area, 0) + 1
            
            # Priorities
            priority = annotation.get('priority', 'Not Specified')
            if priority != 'Not Specified':
                metrics['priorities'][priority] = metrics['priorities'].get(priority, 0) + 1
            
            # Implementation status
            status = annotation.get('implementation_status', 'Not Specified')
            if status not in ['Not Specified', 'Not Applicable']:
                metrics['implementation_status'][status] = metrics['implementation_status'].get(status, 0) + 1
        
        return metrics
    
    def calculate_matching_metrics(self, matching_results: Dict) -> Dict:
        """Calculate matching-related metrics"""
        if not matching_results:
            return {
                'total_matches': 0,
                'matching_rate': 0,
                'avg_similarity': 0,
                'match_types': {}
            }
        
        stats = matching_results.get('statistics', {})
        matches = matching_results.get('matches', [])
        
        # Match type distribution
        match_types = {}
        all_similarities = []
        
        for match in matches:
            for resp_match in match.get('matches', []):
                match_type = resp_match.get('match_type', 'Unknown')
                match_types[match_type] = match_types.get(match_type, 0) + 1
                all_similarities.append(resp_match.get('similarity', 0))
        
        avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0
        
        return {
            'total_matches': stats.get('total_matches', 0),
            'matching_rate': stats.get('matching_rate', 0),
            'avg_similarity': avg_similarity,
            'match_types': match_types,
            'matched_recommendations': stats.get('matched_recommendations', 0)
        }

def render_dashboard_tab():
    """Render the analytics dashboard interface"""
    st.header("📊 Analytics Dashboard")
    st.markdown("""
    Comprehensive analytics and insights for your document analysis workflow.
    """)
    
    # Initialize analytics engine
    analytics = DashboardAnalytics()
    
    # Get data from session state
    documents = st.session_state.get('uploaded_documents', [])
    recommendations = st.session_state.get('extracted_recommendations', [])
    responses = st.session_state.get('extracted_responses', [])
    annotation_results = st.session_state.get('annotation_results', {})
    matching_results = st.session_state.get('matching_results', {})
    
    # Check if data is available
    if not any([documents, recommendations, responses]):
        render_empty_dashboard()
        return
    
    # Calculate metrics
    doc_metrics = analytics.calculate_document_metrics(documents)
    extraction_metrics = analytics.calculate_extraction_metrics(recommendations, responses)
    annotation_metrics = analytics.calculate_annotation_metrics(annotation_results)
    matching_metrics = analytics.calculate_matching_metrics(matching_results)
    
    # Render dashboard sections
    render_overview_section(doc_metrics, extraction_metrics, annotation_metrics, matching_metrics)
    render_document_analysis(doc_metrics, documents)
    render_extraction_analysis(extraction_metrics, recommendations, responses)
    render_annotation_analysis(annotation_metrics, annotation_results)
    render_matching_analysis(matching_metrics, matching_results)
    render_export_section(doc_metrics, extraction_metrics, annotation_metrics, matching_metrics)

def render_empty_dashboard():
    """Render dashboard when no data is available"""
    st.info("📊 Welcome to the Analytics Dashboard!")
    st.markdown("""
    Your dashboard will show comprehensive analytics once you:
    
    1. **📁 Upload Documents** - Add inquiry reports and response documents
    2. **🔍 Extract Content** - Extract recommendations and responses  
    3. **🏷️ Annotate Content** - Add categories and metadata
    4. **🔗 Find Matches** - Link recommendations with responses
    
    Start by uploading documents in the Upload tab.
    """)
    
    # Show workflow progress
    st.markdown("### 🚀 Workflow Progress")
    
    documents = st.session_state.get('uploaded_documents', [])
    recommendations = st.session_state.get('extracted_recommendations', [])
    responses = st.session_state.get('extracted_responses', [])
    annotations = st.session_state.get('annotation_results', {})
    matches = st.session_state.get('matching_results', {})
    
    progress_items = [
        ("📁 Documents Uploaded", len(documents) > 0, len(documents)),
        ("🔍 Content Extracted", len(recommendations) > 0 or len(responses) > 0, len(recommendations) + len(responses)),
        ("🏷️ Content Annotated", len(annotations) > 0, sum(len(v) for v in annotations.values())),
        ("🔗 Matches Found", len(matches) > 0, matches.get('statistics', {}).get('total_matches', 0))
    ]
    
    for item_name, completed, count in progress_items:
        status = "✅" if completed else "⏸️"
        st.markdown(f"{status} {item_name}: {count} items")

def render_overview_section(doc_metrics: Dict, extraction_metrics: Dict, annotation_metrics: Dict, matching_metrics: Dict):
    """Render overview metrics section"""
    st.markdown("### 📈 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📁 Documents",
            doc_metrics['total_documents'],
            help="Total number of uploaded documents"
        )
    
    with col2:
        st.metric(
            "📋 Recommendations",
            extraction_metrics['total_recommendations'],
            help="Total recommendations extracted"
        )
    
    with col3:
        st.metric(
            "💬 Responses",
            extraction_metrics['total_responses'],
            help="Total government responses extracted"
        )
    
    with col4:
        st.metric(
            "🔗 Matches",
            matching_metrics['total_matches'],
            help="Total recommendation-response matches found"
        )
    
    # Secondary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💾 Total Size",
            f"{doc_metrics['total_size_mb']:.1f} MB",
            help="Total size of uploaded documents"
        )
    
    with col2:
        confidence = extraction_metrics['avg_recommendation_confidence']
        st.metric(
            "🎯 Avg Confidence",
            f"{confidence:.2f}" if confidence > 0 else "N/A",
            help="Average extraction confidence score"
        )
    
    with col3:
        st.metric(
            "🏷️ Annotations",
            annotation_metrics['total_annotations'],
            help="Total number of annotations added"
        )
    
    with col4:
        matching_rate = matching_metrics['matching_rate']
        st.metric(
            "📊 Matching Rate",
            f"{matching_rate:.1%}" if matching_rate > 0 else "N/A",
            help="Percentage of recommendations with matches"
        )

def render_document_analysis(doc_metrics: Dict, documents: List[Dict]):
    """Render document analysis section"""
    st.markdown("### 📄 Document Analysis")
    
    if not documents:
        st.info("No documents to analyze.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### File Type Distribution")
        file_types = doc_metrics['file_types']
        
        if file_types:
            file_type_df = pd.DataFrame([
                {'File Type': ftype, 'Count': count, 'Percentage': f"{(count/doc_metrics['total_documents']*100):.1f}%"}
                for ftype, count in file_types.items()
            ])
            st.dataframe(file_type_df)
        
        # Document size analysis
        st.markdown("#### Document Sizes")
        sizes_mb = [doc['metadata']['file_size'] / (1024*1024) for doc in documents]
        
        if sizes_mb:
            st.metric("Largest Document", f"{max(sizes_mb):.1f} MB")
            st.metric("Smallest Document", f"{min(sizes_mb):.1f} MB")
            st.metric("Average Size", f"{sum(sizes_mb)/len(sizes_mb):.1f} MB")
    
    with col2:
        st.markdown("#### Recent Uploads")
        
        # Sort documents by upload time
        sorted_docs = sorted(
            documents,
            key=lambda x: x['metadata'].get('upload_timestamp', ''),
            reverse=True
        )
        
        recent_docs = sorted_docs[:5]
        
        for doc in recent_docs:
            upload_time = doc['metadata'].get('upload_timestamp', 'Unknown')
            if upload_time != 'Unknown':
                upload_time = upload_time[:19].replace('T', ' ')
            
            st.markdown(f"**{doc['filename']}**")
            st.caption(f"Uploaded: {upload_time} | Size: {doc['metadata']['file_size']/(1024*1024):.1f} MB")

def render_extraction_analysis(extraction_metrics: Dict, recommendations: List[Dict], responses: List[Dict]):
    """Render extraction analysis section"""
    st.markdown("### 🔍 Extraction Analysis")
    
    if not recommendations and not responses:
        st.info("No extraction data to analyze.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Extraction Methods")
        methods = extraction_metrics['extraction_methods']
        
        if methods:
            method_df = pd.DataFrame([
                {'Method': method, 'Count': count}
                for method, count in methods.items()
            ])
            st.dataframe(method_df)
        
        # Confidence distribution
        if recommendations:
            st.markdown("#### Confidence Distribution")
            confidences = [rec['confidence'] for rec in recommendations]
            
            high_conf = len([c for c in confidences if c >= 0.8])
            med_conf = len([c for c in confidences if 0.6 <= c < 0.8])
            low_conf = len([c for c in confidences if c < 0.6])
            
            conf_df = pd.DataFrame([
                {'Range': 'High (≥0.8)', 'Count': high_conf},
                {'Range': 'Medium (0.6-0.8)', 'Count': med_conf},
                {'Range': 'Low (<0.6)', 'Count': low_conf}
            ])
            st.dataframe(conf_df)
    
    with col2:
        st.markdown("#### Document Coverage")
        coverage = extraction_metrics['document_coverage']
        
        if coverage:
            coverage_df = pd.DataFrame([
                {'Document': doc, 'Extractions': count}
                for doc, count in coverage.items()
            ])
            st.dataframe(coverage_df)
        
        # Top recommendations by confidence
        if recommendations:
            st.markdown("#### Top Recommendations")
            top_recs = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)[:3]
            
            for i, rec in enumerate(top_recs):
                st.markdown(f"**{i+1}.** {rec['text'][:100]}...")
                st.caption(f"Confidence: {rec['confidence']:.2f} | Document: {rec['document']}")

def render_annotation_analysis(annotation_metrics: Dict, annotation_results: Dict):
    """Render annotation analysis section"""
    st.markdown("### 🏷️ Annotation Analysis")
    
    if annotation_metrics['total_annotations'] == 0:
        st.info("No annotation data to analyze.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Policy Area Distribution")
        policy_areas = annotation_metrics['policy_areas']
        
        if policy_areas:
            policy_df = pd.DataFrame([
                {'Policy Area': area, 'Count': count}
                for area, count in policy_areas.items()
            ])
            st.dataframe(policy_df)
        
        st.markdown("#### Priority Levels")
        priorities = annotation_metrics['priorities']
        
        if priorities:
            priority_df = pd.DataFrame([
                {'Priority': priority, 'Count': count}
                for priority, count in priorities.items()
            ])
            st.dataframe(priority_df)
    
    with col2:
        st.markdown("#### Implementation Status")
        statuses = annotation_metrics['implementation_status']
        
        if statuses:
            status_df = pd.DataFrame([
                {'Status': status, 'Count': count}
                for status, count in statuses.items()
            ])
            st.dataframe(status_df)
        
        # Annotation coverage
        total_extracted = len(st.session_state.get('extracted_recommendations', [])) + \
                         len(st.session_state.get('extracted_responses', []))
        
        if total_extracted > 0:
            coverage_rate = annotation_metrics['total_annotations'] / total_extracted
            st.metric("Annotation Coverage", f"{coverage_rate:.1%}")

def render_matching_analysis(matching_metrics: Dict, matching_results: Dict):
    """Render matching analysis section"""
    st.markdown("### 🔗 Matching Analysis")
    
    if matching_metrics['total_matches'] == 0:
        st.info("No matching data to analyze.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Match Quality")
        match_types = matching_metrics['match_types']
        
        if match_types:
            match_type_df = pd.DataFrame([
                {'Match Type': mtype, 'Count': count}
                for mtype, count in match_types.items()
            ])
            st.dataframe(match_type_df)
        
        # Similarity statistics
        avg_sim = matching_metrics['avg_similarity']
        st.metric("Average Similarity", f"{avg_sim:.3f}")
    
    with col2:
        st.markdown("#### Matching Performance")
        
        total_recs = len(st.session_state.get('extracted_recommendations', []))
        matched_recs = matching_metrics['matched_recommendations']
        
        if total_recs > 0:
            st.metric("Recommendations Matched", f"{matched_recs}/{total_recs}")
            st.metric("Matching Success Rate", f"{(matched_recs/total_recs):.1%}")
        
        st.metric("Total Matches Found", matching_metrics['total_matches'])

def render_export_section(doc_metrics: Dict, extraction_metrics: Dict, annotation_metrics: Dict, matching_metrics: Dict):
    """Render data export section"""
    st.markdown("### 📥 Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Dashboard Summary"):
            export_dashboard_summary(doc_metrics, extraction_metrics, annotation_metrics, matching_metrics)
    
    with col2:
        if st.button("📋 Export All Data"):
            export_all_data()
    
    with col3:
        if st.button("📈 Generate Report"):
            generate_analysis_report(doc_metrics, extraction_metrics, annotation_metrics, matching_metrics)

def export_dashboard_summary(doc_metrics: Dict, extraction_metrics: Dict, annotation_metrics: Dict, matching_metrics: Dict):
    """Export dashboard summary as JSON"""
    try:
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'document_metrics': doc_metrics,
            'extraction_metrics': extraction_metrics,
            'annotation_metrics': annotation_metrics,
            'matching_metrics': matching_metrics
        }
        
        json_str = json.dumps(summary, indent=2, default=str)
        filename = f"dashboard_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        st.download_button(
            label="📥 Download Summary JSON",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
        
        st.success("✅ Dashboard summary ready for download!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_all_data():
    """Export all session data as JSON"""
    try:
        all_data = {
            'export_timestamp': datetime.now().isoformat(),
            'uploaded_documents': st.session_state.get('uploaded_documents', []),
            'extracted_recommendations': st.session_state.get('extracted_recommendations', []),
            'extracted_responses': st.session_state.get('extracted_responses', []),
            'annotation_results': st.session_state.get('annotation_results', {}),
            'matching_results': st.session_state.get('matching_results', {}),
            'search_results': st.session_state.get('search_results', {})
        }
        
        # Convert to JSON
        json_str = json.dumps(all_data, indent=2, default=str)
        filename = f"daphneai_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        st.download_button(
            label="📥 Download Complete Dataset",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
        
        st.success("✅ Complete dataset ready for download!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def generate_analysis_report(doc_metrics: Dict, extraction_metrics: Dict, annotation_metrics: Dict, matching_metrics: Dict):
    """Generate comprehensive analysis report"""
    try:
        report_lines = [
            "DAPHNEAI ANALYSIS REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
            f"• Documents Processed: {doc_metrics['total_documents']}",
            f"• Recommendations Extracted: {extraction_metrics['total_recommendations']}",
            f"• Responses Extracted: {extraction_metrics['total_responses']}",
            f"• Total Annotations: {annotation_metrics['total_annotations']}",
            f"• Matches Found: {matching_metrics['total_matches']}",
            "",
            "DOCUMENT ANALYSIS",
            "-" * 20,
            f"• Total Size: {doc_metrics['total_size_mb']:.1f} MB",
            f"• Average Document Size: {doc_metrics['avg_document_size']/(1024*1024):.1f} MB",
            f"• File Types: {', '.join(doc_metrics['file_types'].keys())}"
        ]
        
        # Add extraction analysis
        if extraction_metrics['avg_recommendation_confidence'] > 0:
            report_lines.extend([
                "",
                "EXTRACTION QUALITY",
                "-" * 20,
                f"• Average Confidence: {extraction_metrics['avg_recommendation_confidence']:.2f}",
                f"• Extraction Methods: {', '.join(extraction_metrics['extraction_methods'].keys())}"
            ])
        
        # Add annotation analysis
        if annotation_metrics['total_annotations'] > 0:
            report_lines.extend([
                "",
                "ANNOTATION SUMMARY",
                "-" * 20,
                f"• Top Policy Areas: {', '.join(list(annotation_metrics['policy_areas'].keys())[:3])}",
                f"• Priority Distribution: {', '.join(annotation_metrics['priorities'].keys())}"
            ])
        
        # Add matching analysis
        if matching_metrics['total_matches'] > 0:
            report_lines.extend([
                "",
                "MATCHING PERFORMANCE",
                "-" * 20,
                f"• Matching Rate: {matching_metrics['matching_rate']:.1%}",
                f"• Average Similarity: {matching_metrics['avg_similarity']:.3f}",
                f"• Match Types: {', '.join(matching_metrics['match_types'].keys())}"
            ])
        
        report_text = "\n".join(report_lines)
        filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        st.download_button(
            label="📈 Download Analysis Report",
            data=report_text,
            file_name=filename,
            mime="text/plain"
        )
        
        st.success("✅ Analysis report ready for download!")
        
    except Exception as e:
        st.error(f"❌ Report generation failed: {str(e)}")

# ===============================================
# INITIALIZATION
# ===============================================

logger.info("✅ Dashboard components initialized")
