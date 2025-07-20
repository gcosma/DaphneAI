# ===============================================
# FILE: modules/ui/dashboard_components.py
# ===============================================

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import io
import zipfile

# Import plotting libraries with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def render_dashboard_tab():
    """Render the analytics dashboard tab"""
    st.header("📊 Analytics Dashboard")
    
    # Check if there's any data to analyze
    if not has_analysis_data():
        render_empty_dashboard()
        return
    
    # Dashboard overview
    render_dashboard_overview()
    
    # Main dashboard sections
    dashboard_sections = st.tabs([
        "📈 Processing Overview", 
        "📋 Content Analysis", 
        "🏷️ Concept Analytics", 
        "🔗 Matching Insights",
        "🔍 Search Analytics",
        "📥 Export Center"
    ])
    
    with dashboard_sections[0]:
        render_processing_overview()
    
    with dashboard_sections[1]:
        render_content_analysis()
    
    with dashboard_sections[2]:
        render_concept_analytics()
    
    with dashboard_sections[3]:
        render_matching_insights()
    
    with dashboard_sections[4]:
        render_search_analytics()
    
    with dashboard_sections[5]:
        render_export_center()

def has_analysis_data():
    """Check if there's any data available for analysis"""
    return any([
        st.session_state.get('uploaded_documents'),
        st.session_state.get('extracted_recommendations'),
        st.session_state.get('extracted_concerns'),
        st.session_state.get('annotation_results'),
        st.session_state.get('matching_results'),
        st.session_state.get('search_results')
    ])

def render_empty_dashboard():
    """Render dashboard when no data is available"""
    st.info("📊 Welcome to the Analytics Dashboard!")
    
    st.markdown("""
    This dashboard will show comprehensive analytics once you start processing documents.
    
    **To get started:**
    1. 📁 Upload documents in the **Upload Documents** tab
    2. 🔍 Extract recommendations in the **Extract Content** tab  
    3. 🏷️ Annotate with concepts in the **Concept Annotation** tab
    4. 🔗 Find responses in the **Find Responses** tab
    5. 🔎 Use the **Smart Search** to explore your data
    
    Once you have processed some documents, this dashboard will display:
    - Document processing statistics
    - Content analysis and trends
    - Concept annotation insights
    - Response matching analytics
    - Search patterns and usage
    - Comprehensive export options
    """)
    
    # Show progress towards getting analytics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        docs_uploaded = len(st.session_state.get('uploaded_documents', []))
        if docs_uploaded > 0:
            st.success(f"✅ {docs_uploaded} documents uploaded")
        else:
            st.info("📁 Upload documents to start")
    
    with col2:
        recs_extracted = len(st.session_state.get('extracted_recommendations', []))
        if recs_extracted > 0:
            st.success(f"✅ {recs_extracted} recommendations extracted")
        else:
            st.info("🔍 Extract content")
    
    with col3:
        annotations_done = len(st.session_state.get('annotation_results', {}))
        if annotations_done > 0:
            st.success(f"✅ {annotations_done} items annotated")
        else:
            st.info("🏷️ Add annotations")
    
    with col4:
        matches_found = len(st.session_state.get('matching_results', {}))
        if matches_found > 0:
            st.success(f"✅ {matches_found} items matched")
        else:
            st.info("🔗 Find responses")

def render_dashboard_overview():
    """Render the main dashboard overview metrics"""
    st.subheader("📈 System Overview")
    
    # Calculate key metrics
    metrics = calculate_overview_metrics()
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📁 Documents", 
            metrics['documents'],
            help="Total uploaded documents"
        )
    
    with col2:
        st.metric(
            "💡 Recommendations", 
            metrics['recommendations'],
            help="Extracted recommendations"
        )
    
    with col3:
        st.metric(
            "⚠️ Concerns", 
            metrics['concerns'],
            help="Identified concerns"
        )
    
    with col4:
        st.metric(
            "🏷️ Annotations", 
            metrics['annotations'],
            help="Applied concept annotations"
        )
    
    with col5:
        st.metric(
            "🔗 Matches", 
            metrics['matches'],
            help="Found response matches"
        )
    
    # Processing efficiency metrics
    if metrics['documents'] > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extraction_rate = (metrics['recommendations'] + metrics['concerns']) / metrics['documents']
            st.metric(
                "📊 Extraction Rate", 
                f"{extraction_rate:.1f}",
                help="Average extractions per document"
            )
        
        with col2:
            if metrics['recommendations'] > 0:
                annotation_rate = metrics['annotations'] / metrics['recommendations'] * 100
                st.metric(
                    "🎯 Annotation Coverage", 
                    f"{annotation_rate:.0f}%",
                    help="Percentage of recommendations annotated"
                )
        
        with col3:
            if metrics['recommendations'] > 0:
                matching_rate = metrics['matches'] / metrics['recommendations'] * 100
                st.metric(
                    "🔍 Matching Coverage", 
                    f"{matching_rate:.0f}%",
                    help="Percentage of recommendations with found responses"
                )

def calculate_overview_metrics():
    """Calculate overview metrics for the dashboard"""
    return {
        'documents': len(st.session_state.get('uploaded_documents', [])),
        'recommendations': len(st.session_state.get('extracted_recommendations', [])),
        'concerns': len(st.session_state.get('extracted_concerns', [])),
        'annotations': len(st.session_state.get('annotation_results', {})),
        'matches': len(st.session_state.get('matching_results', {})),
        'searches': len(st.session_state.get('search_history', []))
    }

def render_processing_overview():
    """Render processing overview analytics"""
    st.subheader("📈 Processing Pipeline Overview")
    
    documents = st.session_state.get('uploaded_documents', [])
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not documents:
        st.info("No processing data available yet.")
        return
    
    # Document analysis
    render_document_analysis(documents)
    
    # Processing timeline
    render_processing_timeline(documents, recommendations, concerns)
    
    # Content distribution
    render_content_distribution(documents)

def render_document_analysis(documents: List[Dict]):
    """Render document analysis charts"""
    st.markdown("### 📄 Document Analysis")
    
    # Document type distribution
    doc_types = [doc.get('document_type', 'Unknown') for doc in documents]
    type_counts = pd.Series(doc_types).value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if PLOTLY_AVAILABLE and len(type_counts) > 0:
            fig = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title="Document Type Distribution"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(type_counts)
    
    with col2:
        # Document size analysis
        sizes = [doc.get('file_size', 0) / 1024 for doc in documents]  # Convert to KB
        size_df = pd.DataFrame({
            'Document': [doc.get('filename', f'Doc {i}') for i, doc in enumerate(documents)],
            'Size (KB)': sizes
        })
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                size_df, 
                x='Document', 
                y='Size (KB)',
                title="Document Sizes"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(size_df.set_index('Document'))

def render_processing_timeline(documents: List[Dict], recommendations: List, concerns: List):
    """Render processing timeline"""
    st.markdown("### ⏱️ Processing Timeline")
    
    # Create timeline data
    timeline_data = []
    
    # Add document uploads
    for doc in documents:
        upload_time = doc.get('upload_time', '')
        if upload_time:
            try:
                timestamp = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                timeline_data.append({
                    'timestamp': timestamp,
                    'event': 'Document Upload',
                    'details': doc.get('filename', 'Unknown'),
                    'type': 'Upload'
                })
            except:
                pass
    
    # Add extraction events (if timestamps available)
    for rec in recommendations:
        if hasattr(rec, 'metadata') and rec.metadata.get('extraction_time'):
            try:
                timestamp = datetime.fromisoformat(rec.metadata['extraction_time'].replace('Z', '+00:00'))
                timeline_data.append({
                    'timestamp': timestamp,
                    'event': 'Recommendation Extracted',
                    'details': rec.id,
                    'type': 'Extraction'
                })
            except:
                pass
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        timeline_df = timeline_df.sort_values('timestamp')
        
        if PLOTLY_AVAILABLE:
            fig = px.scatter(
                timeline_df, 
                x='timestamp', 
                y='type',
                color='type',
                hover_data=['event', 'details'],
                title="Processing Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback display
            st.dataframe(timeline_df[['timestamp', 'event', 'details']], use_container_width=True)
    else:
        st.info("No timeline data available.")

def render_content_distribution(documents: List[Dict]):
    """Render content distribution analysis"""
    st.markdown("### 📊 Content Distribution")
    
    # Content length analysis
    content_lengths = []
    for doc in documents:
        content = doc.get('content', '')
        content_lengths.append({
            'Document': doc.get('filename', 'Unknown'),
            'Content Length': len(content),
            'Word Count': len(content.split()) if content else 0,
            'Document Type': doc.get('document_type', 'Unknown')
        })
    
    if content_lengths:
        content_df = pd.DataFrame(content_lengths)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    content_df, 
                    x='Content Length',
                    nbins=20,
                    title="Content Length Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(content_df.set_index('Document')['Content Length'])
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig = px.box(
                    content_df, 
                    x='Document Type', 
                    y='Word Count',
                    title="Word Count by Document Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                type_word_counts = content_df.groupby('Document Type')['Word Count'].mean()
                st.bar_chart(type_word_counts)

def render_content_analysis():
    """Render content analysis section"""
    st.subheader("📋 Content Analysis")
    
    recommendations = st.session_state.get('extracted_recommendations', [])
    concerns = st.session_state.get('extracted_concerns', [])
    
    if not recommendations and not concerns:
        st.info("No extracted content available for analysis.")
        return
    
    # Content metrics
    render_content_metrics(recommendations, concerns)
    
    # Confidence analysis
    render_confidence_analysis(recommendations, concerns)
    
    # Source analysis
    render_source_analysis(recommendations, concerns)

def render_content_metrics(recommendations: List, concerns: List):
    """Render content metrics"""
    st.markdown("### 📊 Content Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_items = len(recommendations) + len(concerns)
        st.metric("Total Extracted Items", total_items)
    
    with col2:
        st.metric("Recommendations", len(recommendations))
    
    with col3:
        st.metric("Concerns", len(concerns))
    
    with col4:
        if total_items > 0:
            rec_percentage = len(recommendations) / total_items * 100
            st.metric("Recommendation %", f"{rec_percentage:.1f}%")
    
    # Content length analysis
    if recommendations or concerns:
        lengths_data = []
        
        for rec in recommendations:
            lengths_data.append({
                'Type': 'Recommendation',
                'Length': len(rec.text),
                'Source': rec.document_source,
                'Confidence': rec.confidence_score
            })
        
        for concern in concerns:
            lengths_data.append({
                'Type': 'Concern',
                'Length': len(concern.get('text', '')),
                'Source': concern.get('document_source', 'Unknown'),
                'Confidence': concern.get('confidence_score', 0)
            })
        
        if lengths_data:
            lengths_df = pd.DataFrame(lengths_data)
            
            if PLOTLY_AVAILABLE:
                fig = px.box(
                    lengths_df, 
                    x='Type', 
                    y='Length',
                    title="Content Length Distribution by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                type_lengths = lengths_df.groupby('Type')['Length'].mean()
                st.bar_chart(type_lengths)

def render_confidence_analysis(recommendations: List, concerns: List):
    """Render confidence score analysis"""
    st.markdown("### 🎯 Confidence Analysis")
    
    confidence_data = []
    
    for rec in recommendations:
        confidence_data.append({
            'Type': 'Recommendation',
            'Confidence': rec.confidence_score,
            'Source': rec.document_source,
            'ID': rec.id
        })
    
    for concern in concerns:
        confidence_data.append({
            'Type': 'Concern', 
            'Confidence': concern.get('confidence_score', 0),
            'Source': concern.get('document_source', 'Unknown'),
            'ID': concern.get('id', 'Unknown')
        })
    
    if confidence_data:
        conf_df = pd.DataFrame(confidence_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    conf_df, 
                    x='Confidence',
                    color='Type',
                    nbins=20,
                    title="Confidence Score Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(conf_counts)
        
        with col2:
            # Confidence statistics
            st.markdown("**Confidence Statistics:**")
            
            for content_type in ['Recommendation', 'Concern']:
                type_data = conf_df[conf_df['Type'] == content_type]['Confidence']
                if len(type_data) > 0:
                    st.write(f"**{content_type}:**")
                    st.write(f"  • Mean: {type_data.mean():.3f}")
                    st.write(f"  • Median: {type_data.median():.3f}")
                    st.write(f"  • Std Dev: {type_data.std():.3f}")
                    st.write(f"  • Min: {type_data.min():.3f}")
                    st.write(f"  • Max: {type_data.max():.3f}")

def render_source_analysis(recommendations: List, concerns: List):
    """Render source document analysis"""
    st.markdown("### 📚 Source Analysis")
    
    source_data = {}
    
    # Count by source
    for rec in recommendations:
        source = rec.document_source
        if source not in source_data:
            source_data[source] = {'recommendations': 0, 'concerns': 0}
        source_data[source]['recommendations'] += 1
    
    for concern in concerns:
        source = concern.get('document_source', 'Unknown')
        if source not in source_data:
            source_data[source] = {'recommendations': 0, 'concerns': 0}
        source_data[source]['concerns'] += 1
    
    if source_data:
        # Create chart data
        chart_data = []
        for source, counts in source_data.items():
            chart_data.append({
                'Source': source,
                'Recommendations': counts['recommendations'],
                'Concerns': counts['concerns'],
                'Total': counts['recommendations'] + counts['concerns']
            })
        
        chart_df = pd.DataFrame(chart_data)
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                chart_df, 
                x='Source', 
                y=['Recommendations', 'Concerns'],
                title="Content Distribution by Source Document"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(chart_df.set_index('Source')[['Recommendations', 'Concerns']])

def render_concept_analytics():
    """Render concept annotation analytics"""
    st.subheader("🏷️ Concept Analytics")
    
    annotation_results = st.session_state.get('annotation_results', {})
    
    if not annotation_results:
        st.info("No concept annotations available for analysis.")
        return
    
    # Framework usage
    render_framework_usage(annotation_results)
    
    # Theme analysis
    render_theme_analysis(annotation_results)
    
    # Confidence patterns
    render_annotation_confidence_patterns(annotation_results)

def render_framework_usage(annotation_results: Dict):
    """Render framework usage analytics"""
    st.markdown("### 🎯 Framework Usage")
    
    framework_stats = {}
    
    for result in annotation_results.values():
        annotations = result.get('annotations', {})
        for framework, themes in annotations.items():
            if framework not in framework_stats:
                framework_stats[framework] = {
                    'usage_count': 0,
                    'total_themes': 0,
                    'avg_confidence': 0,
                    'confidences': []
                }
            
            framework_stats[framework]['usage_count'] += 1
            framework_stats[framework]['total_themes'] += len(themes)
            
            for theme in themes:
                confidence = theme.get('confidence', 0)
                framework_stats[framework]['confidences'].append(confidence)
    
    # Calculate averages
    for framework, stats in framework_stats.items():
        if stats['confidences']:
            stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
    
    if framework_stats:
        # Usage chart
        usage_data = []
        for framework, stats in framework_stats.items():
            usage_data.append({
                'Framework': framework,
                'Usage Count': stats['usage_count'],
                'Total Themes': stats['total_themes'],
                'Avg Confidence': stats['avg_confidence']
            })
        
        usage_df = pd.DataFrame(usage_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    usage_df, 
                    x='Framework', 
                    y='Usage Count',
                    title="Framework Usage Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(usage_df.set_index('Framework')['Usage Count'])
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    usage_df, 
                    x='Framework', 
                    y='Avg Confidence',
                    title="Average Confidence by Framework"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(usage_df.set_index('Framework')['Avg Confidence'])

def render_theme_analysis(annotation_results: Dict):
    """Render theme analysis"""
    st.markdown("### 🎨 Theme Analysis")
    
    theme_counts = {}
    theme_confidences = {}
    
    for result in annotation_results.values():
        annotations = result.get('annotations', {})
        for framework, themes in annotations.items():
            for theme in themes:
                theme_name = f"{framework}: {theme['theme']}"
                theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1
                
                if theme_name not in theme_confidences:
                    theme_confidences[theme_name] = []
                theme_confidences[theme_name].append(theme.get('confidence', 0))
    
    if theme_counts:
        # Top themes
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        theme_analysis_data = []
        for theme, count in top_themes:
            avg_confidence = sum(theme_confidences[theme]) / len(theme_confidences[theme])
            theme_analysis_data.append({
                'Theme': theme.split(': ', 1)[1] if ': ' in theme else theme,
                'Framework': theme.split(': ', 1)[0] if ': ' in theme else 'Unknown',
                'Count': count,
                'Avg Confidence': avg_confidence
            })
        
        theme_df = pd.DataFrame(theme_analysis_data)
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                theme_df, 
                x='Count', 
                y='Theme',
                color='Framework',
                orientation='h',
                title="Top 15 Most Common Themes"
            )
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(theme_df.set_index('Theme')['Count'])

def render_annotation_confidence_patterns(annotation_results: Dict):
    """Render annotation confidence patterns"""
    st.markdown("### 📈 Confidence Patterns")
    
    confidence_data = []
    
    for rec_id, result in annotation_results.items():
        annotations = result.get('annotations', {})
        rec = result.get('recommendation')
        
        for framework, themes in annotations.items():
            for theme in themes:
                confidence_data.append({
                    'Recommendation_ID': rec_id,
                    'Framework': framework,
                    'Theme': theme['theme'],
                    'Confidence': theme.get('confidence', 0),
                    'Semantic_Similarity': theme.get('semantic_similarity', 0),
                    'Keyword_Count': theme.get('keyword_count', 0),
                    'Source': rec.document_source if rec else 'Unknown'
                })
    
    if confidence_data:
        conf_pattern_df = pd.DataFrame(confidence_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.scatter(
                    conf_pattern_df, 
                    x='Semantic_Similarity', 
                    y='Confidence',
                    color='Framework',
                    size='Keyword_Count',
                    title="Confidence vs Semantic Similarity"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig = px.box(
                    conf_pattern_df, 
                    x='Framework', 
                    y='Confidence',
                    title="Confidence Distribution by Framework"
                )
                st.plotly_chart(fig, use_container_width=True)

def render_matching_insights():
    """Render matching insights section"""
    st.subheader("🔗 Matching Insights")
    
    matching_results = st.session_state.get('matching_results', {})
    
    if not matching_results:
        st.info("No matching results available for analysis.")
        return
    
    # Matching success rates
    render_matching_success_rates(matching_results)
    
    # Confidence analysis
    render_matching_confidence_analysis(matching_results)
    
    # Match type analysis
    render_match_type_analysis(matching_results)

def render_matching_success_rates(matching_results: Dict):
    """Render matching success rates"""
    st.markdown("### 📊 Matching Success Rates")
    
    total_recommendations = len(matching_results)
    successful_matches = 0
    total_responses = 0
    high_confidence_matches = 0
    
    for result in matching_results.values():
        responses = result.get('responses', [])
        if responses:
            successful_matches += 1
            total_responses += len(responses)
            
            # Count high confidence matches
            for response in responses:
                confidence = response.get('combined_confidence', response.get('similarity_score', 0))
                if confidence >= 0.8:
                    high_confidence_matches += 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recommendations", total_recommendations)
    
    with col2:
        success_rate = (successful_matches / total_recommendations * 100) if total_recommendations > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_responses = (total_responses / successful_matches) if successful_matches > 0 else 0
        st.metric("Avg Responses/Match", f"{avg_responses:.1f}")
    
    with col4:
        high_conf_rate = (high_confidence_matches / total_responses * 100) if total_responses > 0 else 0
        st.metric("High Confidence %", f"{high_conf_rate:.1f}%")

def render_matching_confidence_analysis(matching_results: Dict):
    """Render matching confidence analysis"""
    st.markdown("### 🎯 Confidence Analysis")
    
    confidence_data = []
    
    for rec_index, result in matching_results.items():
        recommendation = result.get('recommendation')
        responses = result.get('responses', [])
        
        for response in responses:
            confidence_data.append({
                'Recommendation_ID': recommendation.id if recommendation else f"Rec_{rec_index}",
                'Combined_Confidence': response.get('combined_confidence', response.get('similarity_score', 0)),
                'Semantic_Similarity': response.get('similarity_score', 0),
                'Concept_Overlap': response.get('concept_overlap', {}).get('overlap_score', 0),
                'Match_Type': match_type,
                'Count': count,
                'Avg_Confidence': avg_confidence
            })
        
        match_df = pd.DataFrame(match_analysis_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    match_df, 
                    values='Count', 
                    names='Match_Type',
                    title="Match Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(match_df.set_index('Match_Type')['Count'])
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    match_df, 
                    x='Match_Type', 
                    y='Avg_Confidence',
                    title="Average Confidence by Match Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(match_df.set_index('Match_Type')['Avg_Confidence'])

def render_search_analytics():
    """Render search analytics section"""
    st.subheader("🔍 Search Analytics")
    
    search_history = st.session_state.get('search_history', [])
    search_results = st.session_state.get('search_results', {})
    
    if not search_history:
        st.info("No search activity available for analysis.")
        return
    
    # Search usage patterns
    render_search_usage_patterns(search_history)
    
    # Query analysis
    render_query_analysis(search_history, search_results)
    
    # Search effectiveness
    render_search_effectiveness(search_results)

def render_search_usage_patterns(search_history: List[Dict]):
    """Render search usage patterns"""
    st.markdown("### 📈 Search Usage Patterns")
    
    if not search_history:
        st.info("No search history available.")
        return
    
    # Time-based analysis
    search_times = []
    for search in search_history:
        timestamp_str = search.get('timestamp', '')
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            search_times.append(timestamp)
        except:
            pass
    
    if search_times:
        # Daily search counts
        daily_counts = {}
        for timestamp in search_times:
            date_key = timestamp.date()
            daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
        
        if daily_counts:
            daily_df = pd.DataFrame([
                {'Date': date, 'Searches': count} 
                for date, count in sorted(daily_counts.items())
            ])
            
            if PLOTLY_AVAILABLE:
                fig = px.line(
                    daily_df, 
                    x='Date', 
                    y='Searches',
                    title="Daily Search Activity"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(daily_df.set_index('Date'))
    
    # Search frequency metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Searches", len(search_history))
    
    with col2:
        if len(search_history) > 1 and search_times:
            time_span = (max(search_times) - min(search_times)).days + 1
            avg_daily = len(search_history) / time_span if time_span > 0 else 0
            st.metric("Avg Searches/Day", f"{avg_daily:.1f}")
    
    with col3:
        avg_results = sum(s.get('result_count', 0) for s in search_history) / len(search_history)
        st.metric("Avg Results/Search", f"{avg_results:.1f}")

def render_query_analysis(search_history: List[Dict], search_results: Dict):
    """Render query analysis"""
    st.markdown("### 🔤 Query Analysis")
    
    if not search_history:
        return
    
    # Query length analysis
    query_lengths = [len(search.get('query', '').split()) for search in search_history]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if query_lengths:
            length_df = pd.DataFrame({'Query Length (words)': query_lengths})
            
            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    length_df, 
                    x='Query Length (words)',
                    nbins=10,
                    title="Query Length Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                length_counts = length_df['Query Length (words)'].value_counts().sort_index()
                st.bar_chart(length_counts)
    
    with col2:
        # Most common query terms
        all_terms = []
        for search in search_history:
            query = search.get('query', '').lower()
            terms = [term.strip('.,!?') for term in query.split() if len(term.strip('.,!?')) > 2]
            all_terms.extend(terms)
        
        if all_terms:
            term_counts = pd.Series(all_terms).value_counts().head(10)
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    x=term_counts.values,
                    y=term_counts.index,
                    orientation='h',
                    title="Top 10 Query Terms"
                )
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(term_counts)

def render_search_effectiveness(search_results: Dict):
    """Render search effectiveness analysis"""
    st.markdown("### 🎯 Search Effectiveness")
    
    if not search_results:
        st.info("No detailed search results available.")
        return
    
    effectiveness_data = []
    
    for search_id, result in search_results.items():
        semantic_results = result.get('semantic_results', [])
        rag_response = result.get('rag_response', {})
        
        # Calculate effectiveness metrics
        avg_semantic_score = 0
        if semantic_results:
            avg_semantic_score = sum(r.get('score', 0) for r in semantic_results) / len(semantic_results)
        
        rag_confidence = rag_response.get('confidence', 0) if rag_response else 0
        
        effectiveness_data.append({
            'Search_ID': search_id,
            'Query': result.get('query', ''),
            'Semantic_Results': len(semantic_results),
            'Avg_Semantic_Score': avg_semantic_score,
            'RAG_Confidence': rag_confidence,
            'Has_RAG_Response': bool(rag_response),
            'Search_Mode': result.get('mode', 'Unknown')
        })
    
    if effectiveness_data:
        eff_df = pd.DataFrame(effectiveness_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.scatter(
                    eff_df, 
                    x='Semantic_Results', 
                    y='Avg_Semantic_Score',
                    color='Search_Mode',
                    title="Search Results vs Quality"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            mode_counts = eff_df['Search_Mode'].value_counts()
            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    values=mode_counts.values,
                    names=mode_counts.index,
                    title="Search Mode Usage"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(mode_counts)

def render_export_center():
    """Render comprehensive export center"""
    st.subheader("📥 Export Center")
    
    st.markdown("""
    Export your analysis data in various formats for further analysis or reporting.
    Choose from individual components or comprehensive packages.
    """)
    
    # Quick export options
    render_quick_exports()
    
    # Comprehensive export packages
    render_comprehensive_exports()
    
    # Custom export builder
    render_custom_export_builder()

def render_quick_exports():
    """Render quick export options"""
    st.markdown("### ⚡ Quick Exports")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Export Documents List", use_container_width=True):
            export_documents_list()
    
    with col2:
        if st.button("💡 Export Recommendations", use_container_width=True):
            export_recommendations_data()
    
    with col3:
        if st.button("🏷️ Export Annotations", use_container_width=True):
            export_annotations_data()
    
    with col4:
        if st.button("🔗 Export Matches", use_container_width=True):
            export_matches_data()

def render_comprehensive_exports():
    """Render comprehensive export packages"""
    st.markdown("### 📦 Comprehensive Packages")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analytics Package", use_container_width=True, type="primary"):
            export_analytics_package()
    
    with col2:
        if st.button("🔬 Research Package", use_container_width=True, type="primary"):
            export_research_package()
    
    with col3:
        if st.button("📋 Executive Summary", use_container_width=True, type="primary"):
            export_executive_summary()

def render_custom_export_builder():
    """Render custom export builder"""
    st.markdown("### 🔧 Custom Export Builder")
    
    with st.expander("Build Custom Export Package"):
        st.markdown("Select the components you want to include in your export:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_documents = st.checkbox("📁 Document metadata", value=True)
            include_extractions = st.checkbox("🔍 Extracted content", value=True)
            include_annotations = st.checkbox("🏷️ Concept annotations", value=True)
            include_matches = st.checkbox("🔗 Response matches", value=True)
        
        with col2:
            include_search_data = st.checkbox("🔎 Search analytics", value=False)
            include_charts = st.checkbox("📊 Visualization data", value=False)
            include_config = st.checkbox("⚙️ Configuration settings", value=False)
            include_logs = st.checkbox("📝 Processing logs", value=False)
        
        # Export format selection
        export_format = st.selectbox(
            "Export Format:",
            ["ZIP Package", "JSON", "CSV Bundle", "Excel Workbook"],
            key="custom_export_format"
        )
        
        if st.button("📦 Create Custom Export", use_container_width=True):
            create_custom_export(
                include_documents, include_extractions, include_annotations, 
                include_matches, include_search_data, include_charts,
                include_config, include_logs, export_format
            )

def export_documents_list():
    """Export documents list as CSV"""
    documents = st.session_state.get('uploaded_documents', [])
    
    if not documents:
        st.warning("No documents to export.")
        return
    
    export_data = []
    for doc in documents:
        export_data.append({
            'Filename': doc.get('filename', ''),
            'Document_Type': doc.get('document_type', ''),
            'File_Size_KB': round(doc.get('file_size', 0) / 1024, 1),
            'Upload_Time': doc.get('upload_time', ''),
            'Page_Count': doc.get('metadata', {}).get('page_count', ''),
            'Content_Length': len(doc.get('content', '')),
            'Processing_Status': doc.get('processing_status', '')
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Documents List",
        data=csv,
        file_name=f"documents_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_recommendations_data():
    """Export recommendations data"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    
    if not recommendations:
        st.warning("No recommendations to export.")
        return
    
    export_data = []
    for rec in recommendations:
        export_data.append({
            'ID': rec.id,
            'Text': rec.text,
            'Document_Source': rec.document_source,
            'Section_Title': rec.section_title,
            'Page_Number': rec.page_number,
            'Confidence_Score': rec.confidence_score,
            'Text_Length': len(rec.text),
            'Extraction_Method': getattr(rec, 'metadata', {}).get('extraction_method', 'Unknown')
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Recommendations",
        data=csv,
        file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_annotations_data():
    """Export annotations data"""
    annotation_results = st.session_state.get('annotation_results', {})
    
    if not annotation_results:
        st.warning("No annotations to export.")
        return
    
    export_data = []
    for rec_id, result in annotation_results.items():
        rec = result['recommendation']
        annotations = result.get('annotations', {})
        
        for framework, themes in annotations.items():
            for theme in themes:
                export_data.append({
                    'Recommendation_ID': rec_id,
                    'Recommendation_Text': rec.text,
                    'Document_Source': rec.document_source,
                    'Framework': framework,
                    'Theme': theme['theme'],
                    'Confidence': theme['confidence'],
                    'Semantic_Similarity': theme.get('semantic_similarity', 0),
                    'Keyword_Count': theme.get('keyword_count', 0),
                    'Matched_Keywords': ', '.join(theme.get('matched_keywords', [])),
                    'Annotation_Time': result.get('annotation_time', '')
                })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Annotations",
        data=csv,
        file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_matches_data():
    """Export matching results data"""
    matching_results = st.session_state.get('matching_results', {})
    
    if not matching_results:
        st.warning("No matching results to export.")
        return
    
    export_data = []
    for rec_index, result in matching_results.items():
        recommendation = result['recommendation']
        responses = result.get('responses', [])
        
        for response in responses:
            export_data.append({
                'Recommendation_ID': recommendation.id,
                'Recommendation_Text': recommendation.text,
                'Recommendation_Source': recommendation.document_source,
                'Response_Source': response.get('source', 'Unknown'),
                'Response_Text': response.get('text', ''),
                'Similarity_Score': response.get('similarity_score', 0),
                'Combined_Confidence': response.get('combined_confidence', response.get('similarity_score', 0)),
                'Match_Type': response.get('match_type', 'UNKNOWN'),
                'Concept_Overlap_Score': response.get('concept_overlap', {}).get('overlap_score', 0),
                'Shared_Themes': ', '.join(response.get('concept_overlap', {}).get('shared_themes', [])),
                'Search_Time': result.get('search_time', '')
            })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Matching Results",
        data=csv,
        file_name=f"matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_analytics_package():
    """Export comprehensive analytics package"""
    try:
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add overview metrics
            metrics = calculate_overview_metrics()
            metrics_json = json.dumps(metrics, indent=2)
            zip_file.writestr("overview_metrics.json", metrics_json)
            
            # Add all data exports
            documents = st.session_state.get('uploaded_documents', [])
            if documents:
                docs_data = []
                for doc in documents:
                    docs_data.append({
                        'filename': doc.get('filename', ''),
                        'document_type': doc.get('document_type', ''),
                        'file_size': doc.get('file_size', 0),
                        'upload_time': doc.get('upload_time', ''),
                        'content_length': len(doc.get('content', ''))
                    })
                docs_csv = pd.DataFrame(docs_data).to_csv(index=False)
                zip_file.writestr("documents.csv", docs_csv)
            
            # Add recommendations
            recommendations = st.session_state.get('extracted_recommendations', [])
            if recommendations:
                recs_data = []
                for rec in recommendations:
                    recs_data.append({
                        'id': rec.id,
                        'text': rec.text,
                        'document_source': rec.document_source,
                        'confidence_score': rec.confidence_score
                    })
                recs_csv = pd.DataFrame(recs_data).to_csv(index=False)
                zip_file.writestr("recommendations.csv", recs_csv)
            
            # Add package info
            package_info = {
                'package_type': 'Analytics Package',
                'export_time': datetime.now().isoformat(),
                'contents': ['overview_metrics.json', 'documents.csv', 'recommendations.csv'],
                'total_documents': len(documents),
                'total_recommendations': len(recommendations)
            }
            package_json = json.dumps(package_info, indent=2)
            zip_file.writestr("package_info.json", package_json)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Analytics Package",
            data=zip_buffer.getvalue(),
            file_name=f"analytics_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )
        
        st.success("✅ Analytics package created successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to create analytics package: {str(e)}")
        logging.error(f"Analytics package export error: {e}", exc_info=True)

def export_research_package():
    """Export research-focused package"""
    st.info("🔬 Research package export functionality coming soon!")

def export_executive_summary():
    """Export executive summary"""
    st.info("📋 Executive summary export functionality coming soon!")


def create_custom_export(include_documents, include_extractions, include_annotations, 
                         include_matches, include_search_data, include_charts,
                         include_config, include_logs, export_format):
    """Create custom export based on user selections"""
    try:
        if export_format == "ZIP Package":
            # Create ZIP package
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                files_added = []
                
                if include_documents:
                    documents = st.session_state.get('uploaded_documents', [])
                    if documents:
                        docs_data = [
                            {
                                'filename': doc.get('filename', ''),
                                'document_type': doc.get('document_type', ''),
                                'file_size': doc.get('file_size', 0),
                                'upload_time': doc.get('upload_time', '')
                            }
                            for doc in documents
                        ]
                        docs_csv = pd.DataFrame(docs_data).to_csv(index=False)
                        zip_file.writestr("documents.csv", docs_csv)
                        files_added.append("documents.csv")
                
                if include_extractions:
                    recommendations = st.session_state.get('extracted_recommendations', [])
                    if recommendations:
                        recs_data = [
                            {
                                'id': rec.id,
                                'text': rec.text,
                                'document_source': rec.document_source,
                                'confidence_score': rec.confidence_score
                            }
                            for rec in recommendations
                        ]
                        recs_csv = pd.DataFrame(recs_data).to_csv(index=False)
                        zip_file.writestr("recommendations.csv", recs_csv)
                        files_added.append("recommendations.csv")
                
                # Add package manifest
                manifest = {
                    'export_type': 'Custom Export',
                    'export_time': datetime.now().isoformat(),
                    'files_included': files_added,
                    'options_selected': {
                        'documents': include_documents,
                        'extractions': include_extractions,
                        'annotations': include_annotations,
                        'matches': include_matches,
                        'search_data': include_search_data,
                        'charts': include_charts,
                        'config': include_config,
                        'logs': include_logs
                    }
                }
                manifest_json = json.dumps(manifest, indent=2)
                zip_file.writestr("manifest.json", manifest_json)
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Custom Export",
                data=zip_buffer.getvalue(),
                file_name=f"custom_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            
            st.success("✅ Custom export package created successfully!")
        
        else:
            st.info(f"Export format '{export_format}' is not yet implemented.")

    except Exception as e:
        match_type = response.get('match_type', 'UNKNOWN') if 'response' in locals() else 'N/A'
        source = response.get('source', 'Unknown') if 'response' in locals() else 'N/A'
        st.error(f"❌ Failed to create custom export: {str(e)}")
        logging.error(f"Custom export error: {e}, match_type: {match_type}, source: {source}", exc_info=True)

                            

    
    if confidence_data:
        conf_df = pd.DataFrame(confidence_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    conf_df, 
                    x='Combined_Confidence',
                    nbins=20,
                    title="Combined Confidence Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                conf_counts = conf_df['Combined_Confidence'].value_counts().sort_index()
                st.bar_chart(conf_counts) 
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig = px.scatter(
                    conf_df, 
                    x='Semantic_Similarity', 
                    y='Concept_Overlap',
                    color='Match_Type',
                    size='Combined_Confidence',
                    title="Semantic vs Concept Matching"
                )
                st.plotly_chart(fig, use_container_width=True)

def render_match_type_analysis(matching_results: Dict):
    """Render match type analysis"""
    st.markdown("### 🏷️ Match Type Analysis")
    
    match_type_counts = {}
    match_type_confidences = {}
    
    for result in matching_results.values():
        for response in result.get('responses', []):
            match_type = response.get('match_type', 'UNKNOWN')
            confidence = response.get('combined_confidence', response.get('similarity_score', 0))
            
            match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1
            
            if match_type not in match_type_confidences:
                match_type_confidences[match_type] = []
            match_type_confidences[match_type].append(confidence)
    
    if match_type_counts:
        # Calculate averages
        match_analysis_data = []
        for match_type, count in match_type_counts.items():
            avg_confidence = sum(match_type_confidences[match_type]) / len(match_type_confidences[match_type])
            match_analysis_data.append({
                # ===============================================
# MISSING ENDING FOR: modules/ui/dashboard_components.py
# ===============================================

# Continue from where it was cut off...

                'Match_Type': match_type,
                'Count': count,
                'Avg_Confidence': avg_confidence
            })
        
        match_df = pd.DataFrame(match_analysis_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    match_df, 
                    values='Count', 
                    names='Match_Type',
                    title="Match Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(match_df.set_index('Match_Type')['Count'])
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    match_df, 
                    x='Match_Type', 
                    y='Avg_Confidence',
                    title="Average Confidence by Match Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(match_df.set_index('Match_Type')['Avg_Confidence'])

def render_search_analytics():
    """Render search analytics section"""
    st.subheader("🔍 Search Analytics")
    
    search_history = st.session_state.get('search_history', [])
    search_results = st.session_state.get('search_results', {})
    
    if not search_history:
        st.info("No search activity available for analysis.")
        return
    
    # Search usage patterns
    render_search_usage_patterns(search_history)
    
    # Query analysis
    render_query_analysis(search_history, search_results)
    
    # Search effectiveness
    render_search_effectiveness(search_results)

def render_search_usage_patterns(search_history: List[Dict]):
    """Render search usage patterns"""
    st.markdown("### 📈 Search Usage Patterns")
    
    if not search_history:
        st.info("No search history available.")
        return
    
    # Time-based analysis
    search_times = []
    for search in search_history:
        timestamp_str = search.get('timestamp', '')
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            search_times.append(timestamp)
        except:
            pass
    
    if search_times:
        # Daily search counts
        daily_counts = {}
        for timestamp in search_times:
            date_key = timestamp.date()
            daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
        
        if daily_counts:
            daily_df = pd.DataFrame([
                {'Date': date, 'Searches': count} 
                for date, count in sorted(daily_counts.items())
            ])
            
            if PLOTLY_AVAILABLE:
                fig = px.line(
                    daily_df, 
                    x='Date', 
                    y='Searches',
                    title="Daily Search Activity"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(daily_df.set_index('Date'))
    
    # Search frequency metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Searches", len(search_history))
    
    with col2:
        if len(search_history) > 1 and search_times:
            time_span = (max(search_times) - min(search_times)).days + 1
            avg_daily = len(search_history) / time_span if time_span > 0 else 0
            st.metric("Avg Searches/Day", f"{avg_daily:.1f}")
    
    with col3:
        avg_results = sum(s.get('result_count', 0) for s in search_history) / len(search_history)
        st.metric("Avg Results/Search", f"{avg_results:.1f}")

def render_query_analysis(search_history: List[Dict], search_results: Dict):
    """Render query analysis"""
    st.markdown("### 🔤 Query Analysis")
    
    if not search_history:
        return
    
    # Query length analysis
    query_lengths = [len(search.get('query', '').split()) for search in search_history]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if query_lengths:
            length_df = pd.DataFrame({'Query Length (words)': query_lengths})
            
            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    length_df, 
                    x='Query Length (words)',
                    nbins=10,
                    title="Query Length Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.histogram_chart(length_df, x='Query Length (words)')
    
    with col2:
        # Most common query terms
        all_terms = []
        for search in search_history:
            query = search.get('query', '').lower()
            terms = [term.strip('.,!?') for term in query.split() if len(term.strip('.,!?')) > 2]
            all_terms.extend(terms)
        
        if all_terms:
            term_counts = pd.Series(all_terms).value_counts().head(10)
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    x=term_counts.values,
                    y=term_counts.index,
                    orientation='h',
                    title="Top 10 Query Terms"
                )
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(term_counts)

def render_search_effectiveness(search_results: Dict):
    """Render search effectiveness analysis"""
    st.markdown("### 🎯 Search Effectiveness")
    
    if not search_results:
        st.info("No detailed search results available.")
        return
    
    effectiveness_data = []
    
    for search_id, result in search_results.items():
        semantic_results = result.get('semantic_results', [])
        rag_response = result.get('rag_response', {})
        
        # Calculate effectiveness metrics
        avg_semantic_score = 0
        if semantic_results:
            avg_semantic_score = sum(r.get('score', 0) for r in semantic_results) / len(semantic_results)
        
        rag_confidence = rag_response.get('confidence', 0) if rag_response else 0
        
        effectiveness_data.append({
            'Search_ID': search_id,
            'Query': result.get('query', ''),
            'Semantic_Results': len(semantic_results),
            'Avg_Semantic_Score': avg_semantic_score,
            'RAG_Confidence': rag_confidence,
            'Has_RAG_Response': bool(rag_response),
            'Search_Mode': result.get('mode', 'Unknown')
        })
    
    if effectiveness_data:
        eff_df = pd.DataFrame(effectiveness_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.scatter(
                    eff_df, 
                    x='Semantic_Results', 
                    y='Avg_Semantic_Score',
                    color='Search_Mode',
                    title="Search Results vs Quality"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            mode_counts = eff_df['Search_Mode'].value_counts()
            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    values=mode_counts.values,
                    names=mode_counts.index,
                    title="Search Mode Usage"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(mode_counts)

def render_export_center():
    """Render comprehensive export center"""
    st.subheader("📥 Export Center")
    
    st.markdown("""
    Export your analysis data in various formats for further analysis or reporting.
    Choose from individual components or comprehensive packages.
    """)
    
    # Quick export options
    render_quick_exports()
    
    # Comprehensive export packages
    render_comprehensive_exports()
    
    # Custom export builder
    render_custom_export_builder()

def render_quick_exports():
    """Render quick export options"""
    st.markdown("### ⚡ Quick Exports")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Export Documents List", use_container_width=True):
            export_documents_list()
    
    with col2:
        if st.button("💡 Export Recommendations", use_container_width=True):
            export_recommendations_data()
    
    with col3:
        if st.button("🏷️ Export Annotations", use_container_width=True):
            export_annotations_data()
    
    with col4:
        if st.button("🔗 Export Matches", use_container_width=True):
            export_matches_data()

def render_comprehensive_exports():
    """Render comprehensive export packages"""
    st.markdown("### 📦 Comprehensive Packages")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analytics Package", use_container_width=True, type="primary"):
            export_analytics_package()
    
    with col2:
        if st.button("🔬 Research Package", use_container_width=True, type="primary"):
            export_research_package()
    
    with col3:
        if st.button("📋 Executive Summary", use_container_width=True, type="primary"):
            export_executive_summary()

def render_custom_export_builder():
    """Render custom export builder"""
    st.markdown("### 🔧 Custom Export Builder")
    
    with st.expander("Build Custom Export Package"):
        st.markdown("Select the components you want to include in your export:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_documents = st.checkbox("📁 Document metadata", value=True)
            include_extractions = st.checkbox("🔍 Extracted content", value=True)
            include_annotations = st.checkbox("🏷️ Concept annotations", value=True)
            include_matches = st.checkbox("🔗 Response matches", value=True)
        
        with col2:
            include_search_data = st.checkbox("🔎 Search analytics", value=False)
            include_charts = st.checkbox("📊 Visualization data", value=False)
            include_config = st.checkbox("⚙️ Configuration settings", value=False)
            include_logs = st.checkbox("📝 Processing logs", value=False)
        
        # Export format selection
        export_format = st.selectbox(
            "Export Format:",
            ["ZIP Package", "JSON", "CSV Bundle", "Excel Workbook"],
            key="custom_export_format"
        )
        
        if st.button("📦 Create Custom Export", use_container_width=True):
            create_custom_export(
                include_documents, include_extractions, include_annotations, 
                include_matches, include_search_data, include_charts,
                include_config, include_logs, export_format
            )

def export_documents_list():
    """Export documents list as CSV"""
    documents = st.session_state.get('uploaded_documents', [])
    
    if not documents:
        st.warning("No documents to export.")
        return
    
    export_data = []
    for doc in documents:
        export_data.append({
            'Filename': doc.get('filename', ''),
            'Document_Type': doc.get('document_type', ''),
            'File_Size_KB': round(doc.get('file_size', 0) / 1024, 1),
            'Upload_Time': doc.get('upload_time', ''),
            'Page_Count': doc.get('metadata', {}).get('page_count', ''),
            'Content_Length': len(doc.get('content', '')),
            'Processing_Status': doc.get('processing_status', '')
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Documents List",
        data=csv,
        file_name=f"documents_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_recommendations_data():
    """Export recommendations data"""
    recommendations = st.session_state.get('extracted_recommendations', [])
    
    if not recommendations:
        st.warning("No recommendations to export.")
        return
    
    export_data = []
    for rec in recommendations:
        export_data.append({
            'ID': rec.id,
            'Text': rec.text,
            'Document_Source': rec.document_source,
            'Section_Title': rec.section_title,
            'Page_Number': rec.page_number,
            'Confidence_Score': rec.confidence_score,
            'Text_Length': len(rec.text),
            'Extraction_Method': getattr(rec, 'metadata', {}).get('extraction_method', 'Unknown')
        })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Recommendations",
        data=csv,
        file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_annotations_data():
    """Export annotations data"""
    annotation_results = st.session_state.get('annotation_results', {})
    
    if not annotation_results:
        st.warning("No annotations to export.")
        return
    
    export_data = []
    for rec_id, result in annotation_results.items():
        rec = result['recommendation']
        annotations = result.get('annotations', {})
        
        for framework, themes in annotations.items():
            for theme in themes:
                export_data.append({
                    'Recommendation_ID': rec_id,
                    'Recommendation_Text': rec.text,
                    'Document_Source': rec.document_source,
                    'Framework': framework,
                    'Theme': theme['theme'],
                    'Confidence': theme['confidence'],
                    'Semantic_Similarity': theme.get('semantic_similarity', 0),
                    'Keyword_Count': theme.get('keyword_count', 0),
                    'Matched_Keywords': ', '.join(theme.get('matched_keywords', [])),
                    'Annotation_Time': result.get('annotation_time', '')
                })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Annotations",
        data=csv,
        file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_matches_data():
    """Export matching results data"""
    matching_results = st.session_state.get('matching_results', {})
    
    if not matching_results:
        st.warning("No matching results to export.")
        return
    
    export_data = []
    for rec_index, result in matching_results.items():
        recommendation = result['recommendation']
        responses = result.get('responses', [])
        
        for response in responses:
            export_data.append({
                'Recommendation_ID': recommendation.id,
                'Recommendation_Text': recommendation.text,
                'Recommendation_Source': recommendation.document_source,
                'Response_Source': response.get('source', 'Unknown'),
                'Response_Text': response.get('text', ''),
                'Similarity_Score': response.get('similarity_score', 0),
                'Combined_Confidence': response.get('combined_confidence', response.get('similarity_score', 0)),
                'Match_Type': response.get('match_type', 'UNKNOWN'),
                'Concept_Overlap_Score': response.get('concept_overlap', {}).get('overlap_score', 0),
                'Shared_Themes': ', '.join(response.get('concept_overlap', {}).get('shared_themes', [])),
                'Search_Time': result.get('search_time', '')
            })
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Matching Results",
        data=csv,
        file_name=f"matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_analytics_package():
    """Export comprehensive analytics package"""
    try:
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add overview metrics
            metrics = calculate_overview_metrics()
            metrics_json = json.dumps(metrics, indent=2)
            zip_file.writestr("overview_metrics.json", metrics_json)
            
            # Add all data exports if available
            documents = st.session_state.get('uploaded_documents', [])
            if documents:
                docs_data = []
                for doc in documents:
                    docs_data.append({
                        'filename': doc.get('filename', ''),
                        'document_type': doc.get('document_type', ''),
                        'file_size': doc.get('file_size', 0),
                        'upload_time': doc.get('upload_time', ''),
                        'content_length': len(doc.get('content', ''))
                    })
                docs_csv = pd.DataFrame(docs_data).to_csv(index=False)
                zip_file.writestr("documents.csv", docs_csv)
            
            # Add recommendations
            recommendations = st.session_state.get('extracted_recommendations', [])
            if recommendations:
                recs_data = []
                for rec in recommendations:
                    recs_data.append({
                        'id': rec.id,
                        'text': rec.text,
                        'document_source': rec.document_source,
                        'confidence_score': rec.confidence_score
                    })
                recs_csv = pd.DataFrame(recs_data).to_csv(index=False)
                zip_file.writestr("recommendations.csv", recs_csv)
            
            # Add package info
            package_info = {
                'package_type': 'Analytics Package',
                'export_time': datetime.now().isoformat(),
                'contents': ['overview_metrics.json', 'documents.csv', 'recommendations.csv'],
                'total_documents': len(documents),
                'total_recommendations': len(recommendations)
            }
            package_json = json.dumps(package_info, indent=2)
            zip_file.writestr("package_info.json", package_json)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Analytics Package",
            data=zip_buffer.getvalue(),
            file_name=f"analytics_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )
        
        st.success("✅ Analytics package created successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to create analytics package: {str(e)}")
        logging.error(f"Analytics package export error: {e}", exc_info=True)

def export_research_package():
    """Export research-focused package"""
    st.info("🔬 Research package export functionality coming soon!")

def export_executive_summary():
    """Export executive summary"""
    st.info("📋 Executive summary export functionality coming soon!")

def create_custom_export(include_documents, include_extractions, include_annotations, 
                        include_matches, include_search_data, include_charts,
                        include_config, include_logs, export_format):
    """Create custom export based on user selections"""
    try:
        if export_format == "ZIP Package":
            # Create ZIP package
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                files_added = []
                
                if include_documents:
                    documents = st.session_state.get('uploaded_documents', [])
                    if documents:
                        docs_data = [
                            {
                                'filename': doc.get('filename', ''),
                                'document_type': doc.get('document_type', ''),
                                'file_size': doc.get('file_size', 0),
                                'upload_time': doc.get('upload_time', '')
                            }
                            for doc in documents
                        ]
                        docs_csv = pd.DataFrame(docs_data).to_csv(index=False)
                        zip_file.writestr("documents.csv", docs_csv)
                        files_added.append("documents.csv")
                
                if include_extractions:
                    recommendations = st.session_state.get('extracted_recommendations', [])
                    if recommendations:
                        recs_data = [
                            {
                                'id': rec.id,
                                'text': rec.text,
                                'document_source': rec.document_source,
                                'confidence_score': rec.confidence_score
                            }
                            for rec in recommendations
                        ]
                        recs_csv = pd.DataFrame(recs_data).to_csv(index=False)
                        zip_file.writestr("recommendations.csv", recs_csv)
                        files_added.append("recommendations.csv")
                
                # Add more data based on selections
                if include_annotations:
                    annotation_results = st.session_state.get('annotation_results', {})
                    if annotation_results:
                        ann_data = []
                        for rec_id, result in annotation_results.items():
                            for framework, themes in result.get('annotations', {}).items():
                                for theme in themes:
                                    ann_data.append({
                                        'recommendation_id': rec_id,
                                        'framework': framework,
                                        'theme': theme['theme'],
                                        'confidence': theme['confidence']
                                    })
                        if ann_data:
                            ann_csv = pd.DataFrame(ann_data).to_csv(index=False)
                            zip_file.writestr("annotations.csv", ann_csv)
                            files_added.append("annotations.csv")
                
                # Add package manifest
                manifest = {
                    'export_type': 'Custom Export',
                    'export_time': datetime.now().isoformat(),
                    'files_included': files_added,
                    'options_selected': {
                        'documents': include_documents,
                        'extractions': include_extractions,
                        'annotations': include_annotations,
                        'matches': include_matches,
                        'search_data': include_search_data,
                        'charts': include_charts,
                        'config': include_config,
                        'logs': include_logs
                    }
                }
                manifest_json = json.dumps(manifest, indent=2)
                zip_file.writestr("manifest.json", manifest_json)
                files_added.append("manifest.json")
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Custom Export",
                data=zip_buffer.getvalue(),
                file_name=f"custom_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            
            st.success(f"✅ Custom export package created successfully! ({len(files_added)} files)")
        
        else:
            st.info(f"Export format '{export_format}' is not yet implemented.")
    
    except Exception as e:
        st.error(f"❌ Failed to create custom export: {str(e)}")
        logging.error(f"Custom export error: {e}", exc_info=True)
