# ===============================================
# FILE: modules/streamlit_components.py
# ===============================================

import streamlit as st
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os
from datetime import datetime
import io

# Import other modules
from core_utils import Recommendation, Response, AnnotationResult, clean_text
from document_processor import DocumentProcessor
from llm_extractor import LLMRecommendationExtractor
from bert_annotator import BERTConceptAnnotator
from vector_store import VectorStoreManager
from rag_engine import RAGQueryEngine
from recommendation_matcher import RecommendationResponseMatcher

def initialize_session_state():
    """Initialize session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.uploaded_documents = []
        st.session_state.extracted_recommendations = []
        st.session_state.annotation_results = {}
        st.session_state.matching_results = {}
        st.session_state.vector_store_manager = None
        st.session_state.rag_engine = None
        st.session_state.bert_annotator = None

def render_header():
    """Render application header"""
    st.title("📋 Recommendation-Response Tracker")
    st.markdown("""
    **AI-Powered Document Analysis System**
    
    Upload documents → Extract recommendations → Annotate with concepts → Find responses → Analyze patterns
    """)
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        docs_count = len(st.session_state.get('uploaded_documents', []))
        st.metric("Documents", docs_count)
    
    with col2:
        recs_count = len(st.session_state.get('extracted_recommendations', []))
        st.metric("Recommendations", recs_count)
    
    with col3:
        annotations_count = len(st.session_state.get('annotation_results', {}))
        st.metric("Annotated", annotations_count)
    
    with col4:
        matches_count = len(st.session_state.get('matching_results', {}))
        st.metric("Matched", matches_count)

def render_navigation_tabs():
    """Render navigation tabs"""
    return st.tabs([
        "📁 Upload Documents",
        "🔍 Extract Recommendations", 
        "🏷️ Concept Annotation",
        "🔗 Find Responses",
        "📊 Dashboard"
    ])

def render_upload_tab():
    """Render document upload tab"""
    st.header("📁 Document Upload")
    
    st.markdown("""
    Upload PDF documents containing recommendations and responses.
    The system will automatically process and categorize them.
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDF documents containing recommendations or responses"
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)
    
    # Display uploaded documents
    if st.session_state.uploaded_documents:
        st.subheader("📋 Uploaded Documents")
        
        docs_data = []
        for doc in st.session_state.uploaded_documents:
            docs_data.append({
                "Filename": doc.get('filename', 'Unknown'),
                "Type": doc.get('document_type', 'Unknown'),
                "Pages": doc.get('metadata', {}).get('page_count', 'N/A'),
                "Size (KB)": round(doc.get('metadata', {}).get('file_size', 0) / 1024, 1),
                "Status": "✅ Processed"
            })
        
        df = pd.DataFrame(docs_data)
        st.dataframe(df, use_container_width=True)
        
        # Export documents data
        if st.button("📥 Export Documents List"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Documents CSV",
                csv,
                "uploaded_documents.csv",
                "text/csv"
            )

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files"""
    processor = DocumentProcessor()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Extract text and metadata
            doc_data = processor.extract_text_from_pdf(tmp_file_path)
            
            if doc_data:
                # Determine document type
                doc_type = determine_document_type(doc_data['content'])
                
                # Add to session state
                doc_info = {
                    'filename': uploaded_file.name,
                    'content': doc_data['content'],
                    'metadata': doc_data['metadata'],
                    'document_type': doc_type,
                    'upload_time': datetime.now().isoformat()
                }
                
                # Check if already uploaded
                existing_names = [doc['filename'] for doc in st.session_state.uploaded_documents]
                if uploaded_file.name not in existing_names:
                    st.session_state.uploaded_documents.append(doc_info)
            
            # Cleanup
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            logging.error(f"File processing error: {e}", exc_info=True)
    
    progress_bar.empty()
    status_text.empty()
    
    if uploaded_files:
        st.success(f"✅ Processed {len(uploaded_files)} files successfully!")

def determine_document_type(content: str) -> str:
    """Determine if document contains recommendations or responses"""
    content_lower = content.lower()
    
    # Response indicators
    response_indicators = [
        'in response to', 'responding to', 'implementation', 
        'accepted', 'rejected', 'under review', 'action taken',
        'following the recommendation', 'as recommended',
        'we have implemented', 'steps taken'
    ]
    
    # Recommendation indicators
    recommendation_indicators = [
        'recommendation', 'recommend that', 'should implement',
        'must establish', 'needs to', 'ought to'
    ]
    
    response_score = sum(1 for indicator in response_indicators if indicator in content_lower)
    recommendation_score = sum(1 for indicator in recommendation_indicators if indicator in content_lower)
    
    if response_score > recommendation_score:
        return 'Response'
    elif recommendation_score > 0:
        return 'Recommendation'
    else:
        return 'Unknown'

def render_extraction_tab():
    """Render recommendation extraction tab"""
    st.header("🔍 Recommendation Extraction")
    
    if not st.session_state.uploaded_documents:
        st.warning("⚠️ Please upload documents first.")
        return
    
    st.markdown("""
    Extract recommendations from uploaded documents using AI-powered analysis.
    """)
    
    # Document selection
    doc_options = [doc['filename'] for doc in st.session_state.uploaded_documents]
    selected_docs = st.multiselect(
        "Select documents to extract recommendations from:",
        options=doc_options,
        default=doc_options
    )
    
    # Extraction settings
    col1, col2 = st.columns(2)
    
    with col1:
        extraction_method = st.selectbox(
            "Extraction Method",
            ["AI-Powered (GPT)", "Pattern-Based", "Hybrid"],
            help="Choose how to extract recommendations"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.6,
            help="Minimum confidence for including recommendations"
        )
    
    # Extract button
    if st.button("🔍 Extract Recommendations", type="primary"):
        extract_recommendations(selected_docs, extraction_method, confidence_threshold)
    
    # Display extracted recommendations
    if st.session_state.extracted_recommendations:
        display_extracted_recommendations()

def extract_recommendations(selected_docs: List[str], method: str, threshold: float):
    """Extract recommendations from selected documents"""
    extractor = LLMRecommendationExtractor()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_recommendations = []
    
    for i, doc_name in enumerate(selected_docs):
        try:
            # Find document
            doc = next((d for d in st.session_state.uploaded_documents if d['filename'] == doc_name), None)
            if not doc:
                continue
            
            # Update progress
            progress = (i + 1) / len(selected_docs)
            progress_bar.progress(progress)
            status_text.text(f"Extracting from {doc_name}...")
            
            # Extract recommendations
            recommendations = extractor.extract_recommendations(
                doc['content'], 
                doc['filename']
            )
            
            # Filter by confidence
            filtered_recs = [rec for rec in recommendations if rec.confidence_score >= threshold]
            all_recommendations.extend(filtered_recs)
            
        except Exception as e:
            st.error(f"Error extracting from {doc_name}: {str(e)}")
            logging.error(f"Extraction error: {e}", exc_info=True)
    
    progress_bar.empty()
    status_text.empty()
    
    # Update session state
    st.session_state.extracted_recommendations = all_recommendations
    
    if all_recommendations:
        st.success(f"✅ Extracted {len(all_recommendations)} recommendations!")
    else:
        st.warning("⚠️ No recommendations found meeting the criteria.")

def display_extracted_recommendations():
    """Display extracted recommendations"""
    st.subheader("📋 Extracted Recommendations")
    
    recommendations_data = []
    for rec in st.session_state.extracted_recommendations:
        recommendations_data.append({
            "ID": rec.id,
            "Text": rec.text[:100] + "..." if len(rec.text) > 100 else rec.text,
            "Source": rec.document_source,
            "Section": rec.section_title,
            "Confidence": f"{rec.confidence_score:.2f}",
            "Page": rec.page_number or "N/A"
        })
    
    df = pd.DataFrame(recommendations_data)
    
    # Display with selection
    selected_indices = st.dataframe(
        df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row"
    )
    
    # Show full text for selected recommendations
    if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
        st.subheader("📖 Full Text")
        for idx in selected_indices.selection.rows:
            rec = st.session_state.extracted_recommendations[idx]
            with st.expander(f"Recommendation {rec.id}"):
                st.write(f"**Source:** {rec.document_source}")
                st.write(f"**Section:** {rec.section_title}")
                st.write(f"**Confidence:** {rec.confidence_score:.2f}")
                st.write("**Full Text:**")
                st.write(rec.text)

def render_annotation_tab():
    """Render concept annotation tab"""
    st.header("🏷️ Concept Annotation")
    
    if not st.session_state.extracted_recommendations:
        st.warning("⚠️ Please extract recommendations first.")
        return
    
    st.markdown("""
    Annotate recommendations with conceptual themes using BERT-based analysis.
    """)
    
    # Initialize BERT annotator
    if not st.session_state.bert_annotator:
        with st.spinner("Loading BERT model..."):
            st.session_state.bert_annotator = BERTConceptAnnotator()
    
    # Framework selection
    available_frameworks = list(st.session_state.bert_annotator.frameworks.keys())
    selected_frameworks = st.multiselect(
        "Select Annotation Frameworks:",
        options=available_frameworks,
        default=available_frameworks,
        help="Choose which conceptual frameworks to use for annotation"
    )
    
    # Custom framework upload
    with st.expander("📁 Upload Custom Framework"):
        custom_file = st.file_uploader(
            "Upload custom taxonomy (JSON/CSV/Excel)",
            type=['json', 'csv', 'xlsx'],
            help="Upload your own conceptual framework"
        )
        
        if custom_file:
            success, message = st.session_state.bert_annotator.load_custom_framework(custom_file)
            if success:
                st.success(f"✅ {message}")
                if "Custom" not in selected_frameworks:
                    selected_frameworks.append("Custom")
            else:
                st.error(f"❌ {message}")
    
    # Annotation settings
    col1, col2 = st.columns(2)
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.3, 0.9, 0.65,
            help="Minimum similarity for theme matching"
        )
    
    with col2:
        max_themes = st.slider(
            "Max Themes per Framework",
            1, 15, 10,
            help="Maximum themes to identify per framework"
        )
    
    # Annotate button
    if st.button("🏷️ Annotate Recommendations", type="primary"):
        annotate_recommendations(selected_frameworks, similarity_threshold, max_themes)
    
    # Display annotation results
    if st.session_state.annotation_results:
        display_annotation_results()

def annotate_recommendations(frameworks: List[str], threshold: float, max_themes: int):
    """Annotate recommendations with concepts"""
    annotator = st.session_state.bert_annotator
    annotator.config["base_similarity_threshold"] = threshold
    annotator.config["max_themes_per_framework"] = max_themes
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    annotation_results = {}
    
    for i, rec in enumerate(st.session_state.extracted_recommendations):
        try:
            # Update progress
            progress = (i + 1) / len(st.session_state.extracted_recommendations)
            progress_bar.progress(progress)
            status_text.text(f"Annotating recommendation {rec.id}...")
            
            # Annotate text
            framework_results, highlighting = annotator.annotate_text(rec.text, frameworks)
            
            annotation_results[rec.id] = {
                'recommendation': rec,
                'annotations': framework_results,
                'highlighting': highlighting
            }
            
        except Exception as e:
            st.error(f"Error annotating {rec.id}: {str(e)}")
            logging.error(f"Annotation error: {e}", exc_info=True)
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.annotation_results = annotation_results
    
    if annotation_results:
        st.success(f"✅ Annotated {len(annotation_results)} recommendations!")

def display_annotation_results():
    """Display annotation results"""
    st.subheader("🏷️ Annotation Results")
    
    # Summary statistics
    total_annotations = 0
    framework_counts = {}
    
    for result in st.session_state.annotation_results.values():
        for framework, themes in result['annotations'].items():
            framework_counts[framework] = framework_counts.get(framework, 0) + len(themes)
            total_annotations += len(themes)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Annotations", total_annotations)
    with col2:
        st.metric("Frameworks Used", len(framework_counts))
    with col3:
        st.metric("Avg per Recommendation", round(total_annotations / len(st.session_state.annotation_results), 1))
    
    # Framework distribution
    if framework_counts:
        st.subheader("📊 Framework Distribution")
        framework_df = pd.DataFrame(
            list(framework_counts.items()),
            columns=['Framework', 'Theme Count']
        )
        st.bar_chart(framework_df.set_index('Framework'))
    
    # Detailed results
    st.subheader("📋 Detailed Annotations")
    
    for rec_id, result in st.session_state.annotation_results.items():
        with st.expander(f"Recommendation {rec_id}"):
            rec = result['recommendation']
            
            st.write(f"**Text:** {rec.text}")
            st.write("**Identified Themes:**")
            
            for framework, themes in result['annotations'].items():
                if themes:
                    st.write(f"**{framework}:**")
                    for theme in themes:
                        confidence_color = "🟢" if theme['confidence'] > 0.8 else "🟡" if theme['confidence'] > 0.6 else "🔴"
                        st.write(f"  {confidence_color} {theme['theme']} (confidence: {theme['confidence']:.2f})")
                        st.write(f"    Keywords: {', '.join(theme['matched_keywords'])}")

def render_matching_tab():
    """Render response matching tab"""
    st.header("🔗 Find Responses")
    
    if not st.session_state.extracted_recommendations:
        st.warning("⚠️ Please extract recommendations first.")
        return
    
    st.markdown("""
    Find responses to recommendations using AI-powered semantic search and concept matching.
    """)
    
    # Initialize systems
    if not st.session_state.vector_store_manager:
        st.session_state.vector_store_manager = VectorStoreManager()
    
    if not st.session_state.rag_engine:
        st.session_state.rag_engine = RAGQueryEngine(st.session_state.vector_store_manager)
    
    # Index documents for searching
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📚 Index Documents for Search"):
            index_documents_for_search()
    
    with col2:
        # Vector store stats
        if st.session_state.vector_store_manager:
            stats = st.session_state.vector_store_manager.get_collection_stats()
            st.metric("Indexed Documents", stats.get('total_documents', 0))
    
    # Recommendation selection
    rec_options = [f"{rec.id}: {rec.text[:50]}..." for rec in st.session_state.extracted_recommendations]
    selected_rec_index = st.selectbox(
        "Select recommendation to find responses for:",
        range(len(rec_options)),
        format_func=lambda x: rec_options[x]
    )
    
    # Search button
    if st.button("🔍 Find Responses", type="primary"):
        find_responses_for_recommendation(selected_rec_index)
    
    # Display matching results
    if st.session_state.matching_results:
        display_matching_results(selected_rec_index)

def index_documents_for_search():
    """Index uploaded documents in vector store"""
    if not st.session_state.uploaded_documents:
        st.warning("No documents to index.")
        return
    
    with st.spinner("Indexing documents for search..."):
        try:
            documents_for_indexing = []
            
            for doc in st.session_state.uploaded_documents:
                doc_for_index = {
                    'content': doc['content'],
                    'source': doc['filename'],
                    'document_type': doc['document_type'],
                    'metadata': doc['metadata']
                }
                documents_for_indexing.append(doc_for_index)
            
            success = st.session_state.vector_store_manager.add_documents(documents_for_indexing)
            
            if success:
                st.success("✅ Documents indexed successfully!")
            else:
                st.error("❌ Failed to index documents.")
                
        except Exception as e:
            st.error(f"Error indexing documents: {str(e)}")
            logging.error(f"Indexing error: {e}", exc_info=True)

def find_responses_for_recommendation(rec_index: int):
    """Find responses for selected recommendation"""
    if rec_index >= len(st.session_state.extracted_recommendations):
        st.error("Invalid recommendation selected.")
        return
    
    recommendation = st.session_state.extracted_recommendations[rec_index]
    
    # Initialize matcher
    if not st.session_state.bert_annotator:
        st.session_state.bert_annotator = BERTConceptAnnotator()
    
    matcher = RecommendationResponseMatcher(
        st.session_state.rag_engine,
        st.session_state.bert_annotator
    )
    
    with st.spinner("Finding responses..."):
        try:
            responses = matcher.match_recommendation_to_responses(recommendation)
            
            # Store results
            if rec_index not in st.session_state.matching_results:
                st.session_state.matching_results[rec_index] = {}
            
            st.session_state.matching_results[rec_index] = {
                'recommendation': recommendation,
                'responses': responses,
                'search_time': datetime.now().isoformat()
            }
            
            if responses:
                st.success(f"✅ Found {len(responses)} potential responses!")
            else:
                st.warning("⚠️ No matching responses found.")
                
        except Exception as e:
            st.error(f"Error finding responses: {str(e)}")
            logging.error(f"Response matching error: {e}", exc_info=True)

def display_matching_results(rec_index: int):
    """Display matching results"""
    if rec_index not in st.session_state.matching_results:
        return
    
    result = st.session_state.matching_results[rec_index]
    recommendation = result['recommendation']
    responses = result['responses']
    
    st.subheader("🎯 Matching Results")
    
    # Show recommendation
    with st.expander("📋 Original Recommendation", expanded=True):
        st.write(f"**ID:** {recommendation.id}")
        st.write(f"**Source:** {recommendation.document_source}")
        st.write(f"**Text:** {recommendation.text}")
    
    # Show responses
    if responses:
        st.subheader(f"📄 Found {len(responses)} Potential Responses")
        
        for i, response in enumerate(responses):
            confidence = response.get('combined_confidence', response.get('similarity_score', 0))
            match_type = response.get('match_type', 'UNKNOWN')
            
            # Color code by confidence
            if confidence >= 0.8:
                confidence_color = "🟢"
            elif confidence >= 0.6:
                confidence_color = "🟡" 
            else:
                confidence_color = "🔴"
            
            with st.expander(f"{confidence_color} Response {i+1} - {match_type} (confidence: {confidence:.2f})"):
                st.write(f"**Source:** {response.get('source', 'Unknown')}")
                st.write(f"**Similarity Score:** {response.get('similarity_score', 0):.2f}")
                st.write(f"**Combined Confidence:** {confidence:.2f}")
                
                # Show concept overlap if available
                if 'concept_overlap' in response:
                    overlap = response['concept_overlap']
                    if overlap.get('shared_themes'):
                        st.write(f"**Shared Themes:** {', '.join(overlap['shared_themes'])}")
                
                st.write("**Response Text:**")
                st.write(response.get('text', 'No text available'))
    else:
        st.info("🔍 No responses found. Try adjusting search parameters or adding more response documents.")

def render_dashboard_tab():
    """Render analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    if not any([
        st.session_state.uploaded_documents,
        st.session_state.extracted_recommendations,
        st.session_state.annotation_results,
        st.session_state.matching_results
    ]):
        st.warning("⚠️ No data available. Please process some documents first.")
        return
    
    # Overview metrics
    st.subheader("📈 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Documents",
            len(st.session_state.uploaded_documents),
            help="Total uploaded documents"
        )
    
    with col2:
        st.metric(
            "Recommendations",
            len(st.session_state.extracted_recommendations),
            help="Total extracted recommendations"
        )
    
    with col3:
        annotated_count = len(st.session_state.annotation_results)
        st.metric(
            "Annotated",
            annotated_count,
            help="Recommendations with concept annotations"
        )
    
    with col4:
        matched_count = len(st.session_state.matching_results)
        st.metric(
            "Matched",
            matched_count,
            help="Recommendations with found responses"
        )
    
    # Document type distribution
    if st.session_state.uploaded_documents:
        st.subheader("📁 Document Types")
        
        doc_types = [doc['document_type'] for doc in st.session_state.uploaded_documents]
        type_counts = pd.Series(doc_types).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(type_counts)
        
        with col2:
            for doc_type, count in type_counts.items():
                percentage = (count / len(st.session_state.uploaded_documents)) * 100
                st.write(f"**{doc_type}:** {count} ({percentage:.1f}%)")
    
    # Annotation analysis
    if st.session_state.annotation_results:
        st.subheader("🏷️ Concept Analysis")
        
        # Framework usage
        framework_counts = {}
        theme_counts = {}
        
        for result in st.session_state.annotation_results.values():
            for framework, themes in result['annotations'].items():
                framework_counts[framework] = framework_counts.get(framework, 0) + len(themes)
                
                for theme in themes:
                    theme_name = f"{framework}: {theme['theme']}"
                    theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Framework Usage:**")
            framework_df = pd.DataFrame(
                list(framework_counts.items()),
                columns=['Framework', 'Count']
            )
            st.dataframe(framework_df, use_container_width=True)
        
        with col2:
            st.write("**Top Themes:**")
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            theme_df = pd.DataFrame(top_themes, columns=['Theme', 'Count'])
            st.dataframe(theme_df, use_container_width=True)
    
    # Response matching analysis
    if st.session_state.matching_results:
        st.subheader("🔗 Response Matching")
        
        # Confidence distribution
        confidences = []
        match_types = []
        
        for result in st.session_state.matching_results.values():
            for response in result.get('responses', []):
                conf = response.get('combined_confidence', response.get('similarity_score', 0))
                confidences.append(conf)
                match_types.append(response.get('match_type', 'UNKNOWN'))
        
        if confidences:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Confidence Distribution:**")
                confidence_df = pd.DataFrame({'Confidence': confidences})
                st.histogram_chart(confidence_df, x='Confidence')
            
            with col2:
                st.write("**Match Types:**")
                match_type_counts = pd.Series(match_types).value_counts()
                st.bar_chart(match_type_counts)
    
    # Export options
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Recommendations"):
            export_recommendations()
    
    with col2:
        if st.button("🏷️ Export Annotations"):
            export_annotations()
    
    with col3:
        if st.button("🔗 Export Matches"):
            export_matches()

def export_recommendations():
    """Export recommendations to CSV"""
    if not st.session_state.extracted_recommendations:
        st.warning("No recommendations to export.")
        return
    
    data = []
    for rec in st.session_state.extracted_recommendations:
        data.append({
            'ID': rec.id,
            'Text': rec.text,
            'Source': rec.document_source,
            'Section': rec.section_title,
            'Page': rec.page_number,
            'Confidence': rec.confidence_score
        })
    
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "Download Recommendations CSV",
        csv,
        f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

def export_annotations():
    """Export annotations to JSON"""
    if not st.session_state.annotation_results:
        st.warning("No annotations to export.")
        return
    
    # Convert to serializable format
    export_data = {}
    for rec_id, result in st.session_state.annotation_results.items():
        export_data[rec_id] = {
            'recommendation_text': result['recommendation'].text,
            'annotations': result['annotations'],
            'frameworks_used': list(result['annotations'].keys())
        }
    
    json_str = json.dumps(export_data, indent=2)
    
    st.download_button(
        "Download Annotations JSON",
        json_str.encode('utf-8'),
        f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json"
    )

def export_matches():
    """Export matches to CSV"""
    if not st.session_state.matching_results:
        st.warning("No matches to export.")
        return
    
    data = []
    for rec_index, result in st.session_state.matching_results.items():
        rec = result['recommendation']
        
        for response in result.get('responses', []):
            data.append({
                'Recommendation_ID': rec.id,
                'Recommendation_Text': rec.text[:100] + "...",
                'Response_Source': response.get('source', 'Unknown'),
                'Response_Text': response.get('text', '')[:100] + "...",
                'Similarity_Score': response.get('similarity_score', 0),
                'Combined_Confidence': response.get('combined_confidence', 0),
                'Match_Type': response.get('match_type', 'UNKNOWN')
            })
    
    if data:
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "Download Matches CSV",
            csv,
            f"matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    else:
        st.warning("No match data to export.")
