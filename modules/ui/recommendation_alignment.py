# modules/ui/recommendation_alignment.py
"""
🎯 IMPROVED Recommendation-Response Alignment Interface
With enhanced configuration and better alignment display
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any

# Import the improved matching functions
from .recommendation_alignment_matching import (
    perform_alignment_matching,
    SemanticMatcher
)
from .recommendation_alignment_extraction import (
    extract_recommendations_ultimate,
    extract_responses_ultimate
)
from .recommendation_alignment_core import AlignmentConfig

logger = logging.getLogger(__name__)

# =============================================================================
# 🎨 MAIN INTERFACE FUNCTION
# =============================================================================

def render_recommendation_alignment_interface():
    """Main function to render the improved alignment interface"""
    
    st.markdown("## 🎯 Recommendation-Response Alignment System")
    st.markdown("*Enhanced with semantic similarity and self-match prevention*")
    
    # Check for documents
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload government documents first!")
        st.info("""
        This system requires:
        1. **Recommendation documents** (inquiry reports, committee recommendations)
        2. **Response documents** (government responses, implementation reports)
        """)
        return
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Configuration", 
        "🔄 Run Analysis", 
        "📊 Results", 
        "🔍 Debug"
    ])
    
    # =============================================================================
    # TAB 1: CONFIGURATION
    # =============================================================================
    
    with tab1:
        st.markdown("### ⚙️ Alignment Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Matching Settings")
            
            # Self-match prevention (CRITICAL)
            self_match_prevention = st.checkbox(
                "Enable Self-Match Prevention",
                value=True,
                help="🛡️ CRITICAL: Prevents recommendations from matching with themselves"
            )
            
            # Minimum similarity threshold
            min_similarity = st.slider(
                "Minimum Similarity Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.4,
                step=0.05,
                help="Lower = more matches (including weak ones)"
            )
            
            # Maximum results per recommendation
            max_results = st.selectbox(
                "Max Responses per Recommendation",
                options=[1, 2, 3, 5, 10, "All"],
                index=2,
                help="Limit the number of responses shown per recommendation"
            )
            
            # Document pairing mode
            doc_pairing = st.selectbox(
                "Document Pairing Mode",
                options=[
                    "Smart (Recommend↔Response)",
                    "Cross-Document Only",
                    "Same-Document Only",
                    "All Combinations"
                ],
                index=0,
                help="Control which document combinations to consider"
            )
        
        with col2:
            st.markdown("#### 🧠 Semantic Analysis")
            
            # Semantic matcher type
            use_ai_semantic = st.checkbox(
                "Use AI Semantic Matching",
                value=True,
                help="Uses transformer models for better semantic understanding"
            )
            
            # Scoring weights
            st.markdown("**Scoring Weights:**")
            
            weights = {}
            weights['semantic'] = st.slider("Semantic Similarity", 0.0, 1.0, 0.35, 0.05)
            weights['reference'] = st.slider("Reference Matching", 0.0, 1.0, 0.25, 0.05)
            weights['context'] = st.slider("Context Similarity", 0.0, 1.0, 0.15, 0.05)
            weights['topic'] = st.slider("Topic Alignment", 0.0, 1.0, 0.10, 0.05)
            weights['document'] = st.slider("Document Relationship", 0.0, 1.0, 0.10, 0.05)
            weights['position'] = st.slider("Position Proximity", 0.0, 1.0, 0.05, 0.05)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
                st.info(f"Weights normalized (sum = 1.0)")
        
        # Save configuration
        config = {
            'self_match_prevention': self_match_prevention,
            'min_similarity_threshold': min_similarity,
            'max_results': max_results,
            'doc_pairing': doc_pairing,
            'use_ai_semantic': use_ai_semantic,
            'scoring_weights': weights
        }
        
        st.session_state['alignment_config'] = config
        
        # Show config summary
        with st.expander("📋 Configuration Summary", expanded=False):
            st.json(config)
    
    # =============================================================================
    # TAB 2: RUN ANALYSIS
    # =============================================================================
    
    with tab2:
        st.markdown("### 🔄 Run Alignment Analysis")
        
        # Document selection
        st.markdown("#### 📁 Select Documents")
        
        documents = st.session_state.documents
        doc_names = [doc['filename'] for doc in documents]
        
        col1, col2 = st.columns(2)
        
        with col1:
            rec_docs = st.multiselect(
                "Recommendation Documents",
                options=doc_names,
                help="Select documents containing recommendations"
            )
        
        with col2:
            resp_docs = st.multiselect(
                "Response Documents",
                options=doc_names,
                help="Select documents containing government responses"
            )
        
        # Auto-detect button
        if st.button("🔍 Auto-Detect Document Types"):
            rec_docs, resp_docs = auto_detect_document_types(documents)
            st.success(f"Detected {len(rec_docs)} recommendation docs, {len(resp_docs)} response docs")
            st.rerun()
        
        # Extraction settings
        st.markdown("#### 🔧 Extraction Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extraction_mode = st.selectbox(
                "Extraction Mode",
                ["Balanced", "High Precision", "High Recall"],
                help="Balance between finding all items vs. accuracy"
            )
        
        with col2:
            filter_narrative = st.checkbox(
                "Filter Narrative Content",
                value=True,
                help="Remove story-telling sentences"
            )
        
        with col3:
            min_confidence = st.slider(
                "Min Extraction Confidence",
                0.1, 0.9, 0.3, 0.1,
                help="Minimum confidence for extraction"
            )
        
        # Run analysis button
        if st.button("🚀 Run Alignment Analysis", type="primary"):
            
            if not rec_docs or not resp_docs:
                st.error("Please select at least one document of each type!")
                return
            
            with st.spinner("🔄 Processing documents..."):
                
                # Progress tracking
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    # Step 1: Extract recommendations
                    status.text("📝 Extracting recommendations...")
                    progress.progress(0.25)
                    
                    rec_documents = [d for d in documents if d['filename'] in rec_docs]
                    recommendations = extract_recommendations_ultimate(
                        rec_documents,
                        {
                            'quality_mode': extraction_mode,
                            'filter_narrative': filter_narrative,
                            'min_confidence': min_confidence
                        }
                    )
                    
                    # Step 2: Extract responses
                    status.text("📝 Extracting responses...")
                    progress.progress(0.50)
                    
                    resp_documents = [d for d in documents if d['filename'] in resp_docs]
                    responses = extract_responses_ultimate(
                        resp_documents,
                        {
                            'quality_mode': extraction_mode,
                            'filter_narrative': filter_narrative,
                            'min_confidence': min_confidence
                        }
                    )
                    
                    # Step 3: Perform alignment
                    status.text("🔗 Aligning recommendations with responses...")
                    progress.progress(0.75)
                    
                    alignments = perform_alignment_matching(
                        recommendations,
                        responses,
                        st.session_state['alignment_config']
                    )
                    
                    # Step 4: Store results
                    st.session_state['alignment_results'] = {
                        'alignments': alignments,
                        'recommendations': recommendations,
                        'responses': responses,
                        'timestamp': datetime.now(),
                        'config': st.session_state['alignment_config']
                    }
                    
                    progress.progress(1.0)
                    status.text("✅ Analysis complete!")
                    
                    # Show summary
                    st.success(f"""
                    ✅ **Analysis Complete!**
                    - Found {len(recommendations)} recommendations
                    - Found {len(responses)} responses  
                    - Created {len([a for a in alignments if a['has_response']])} alignments
                    """)
                    
                    # Auto-switch to results tab
                    st.info("💡 Switch to the Results tab to see the alignments")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    logger.exception("Analysis error")
                    
                finally:
                    progress.empty()
                    status.empty()
    
    # =============================================================================
    # TAB 3: RESULTS
    # =============================================================================
    
    with tab3:
        st.markdown("### 📊 Alignment Results")
        
        if 'alignment_results' not in st.session_state:
            st.info("🔄 Run the analysis first to see results")
            return
        
        results = st.session_state['alignment_results']
        alignments = results['alignments']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_recs = len(results['recommendations'])
            st.metric("Total Recommendations", total_recs)
        
        with col2:
            aligned = len([a for a in alignments if a['has_response']])
            st.metric("Aligned", f"{aligned} ({aligned/total_recs*100:.1f}%)")
        
        with col3:
            high_conf = len([a for a in alignments if a.get('alignment_confidence', 0) > 0.7])
            st.metric("High Confidence", high_conf)
        
        with col4:
            avg_score = sum(a.get('alignment_confidence', 0) for a in alignments) / len(alignments) if alignments else 0
            st.metric("Avg Confidence", f"{avg_score:.2%}")
        
        # Filter controls
        st.markdown("#### 🔍 Filter Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_only_aligned = st.checkbox("Show only aligned", value=True)
        
        with col2:
            min_display_confidence = st.slider(
                "Min confidence to display",
                0.0, 1.0, 0.0, 0.1
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Confidence (High→Low)", "Confidence (Low→High)", "Document Order"]
            )
        
        # Display alignments
        st.markdown("#### 📋 Alignment Details")
        
        # Filter and sort
        filtered_alignments = alignments
        
        if show_only_aligned:
            filtered_alignments = [a for a in filtered_alignments if a['has_response']]
        
        if min_display_confidence > 0:
            filtered_alignments = [
                a for a in filtered_alignments 
                if a.get('alignment_confidence', 0) >= min_display_confidence
            ]
        
        if sort_by == "Confidence (High→Low)":
            filtered_alignments.sort(key=lambda x: x.get('alignment_confidence', 0), reverse=True)
        elif sort_by == "Confidence (Low→High)":
            filtered_alignments.sort(key=lambda x: x.get('alignment_confidence', 0))
        
        # Display each alignment
        for i, alignment in enumerate(filtered_alignments):
            rec = alignment['recommendation']
            responses = alignment['responses']
            
            # Create expandable section
            confidence = alignment.get('alignment_confidence', 0)
            status = alignment.get('alignment_status', 'Unknown')
            
            # Color coding based on confidence
            if confidence >= 0.7:
                color = "🟢"
            elif confidence >= 0.5:
                color = "🟡"
            else:
                color = "🔴"
            
            with st.expander(
                f"{color} Recommendation {i+1} | {status} | Confidence: {confidence:.2%}",
                expanded=(i < 3)  # Expand first 3
            ):
                # Show recommendation
                st.markdown("**📝 Recommendation:**")
                st.info(rec.sentence)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"📁 {rec.document['filename']}")
                with col2:
                    st.caption(f"📍 Page ~{rec.position_info.get('page_number', 'Unknown')}")
                
                if responses:
                    st.markdown("**📢 Government Response(s):**")
                    
                    for j, resp_match in enumerate(responses):
                        resp = resp_match['response']
                        score = resp_match['combined_score']
                        quality = resp_match['match_quality']
                        explanation = resp_match['explanation']
                        
                        # Response container
                        with st.container():
                            st.success(f"**Response {j+1} - {quality} Match ({score:.2%})**")
                            st.write(resp.sentence)
                            
                            # Details
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.caption(f"📁 {resp.document['filename']}")
                            with col2:
                                st.caption(f"📍 Page ~{resp.position_info.get('page_number', 'Unknown')}")
                            with col3:
                                st.caption(f"💡 {explanation}")
                            
                            # Show detailed scores
                            if st.checkbox(f"Show scoring details #{i}-{j}", key=f"detail_{i}_{j}"):
                                scores = resp_match['similarity_scores']
                                df_scores = pd.DataFrame([
                                    {"Metric": k.title(), "Score": f"{v:.2%}"} 
                                    for k, v in scores.items()
                                ])
                                st.dataframe(df_scores, use_container_width=True)
                else:
                    st.warning("❌ No responses found for this recommendation")
        
        # Export options
        st.markdown("#### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export to CSV"):
                csv = export_alignments_to_csv(filtered_alignments)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"alignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("📊 Export to Excel"):
                excel_data = export_alignments_to_excel(filtered_alignments)
                st.download_button(
                    "Download Excel",
                    excel_data,
                    f"alignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col3:
            if st.button("📝 Generate Report"):
                report = generate_alignment_report(filtered_alignments, results)
                st.download_button(
                    "Download Report",
                    report,
                    f"alignment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown"
                )
    
    # =============================================================================
    # TAB 4: DEBUG
    # =============================================================================
    
    with tab4:
        st.markdown("### 🔍 Debug Information")
        
        if 'alignment_results' in st.session_state:
            results = st.session_state['alignment_results']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Extraction Stats")
                st.json({
                    "Recommendations found": len(results['recommendations']),
                    "Responses found": len(results['responses']),
                    "Alignments created": len(results['alignments']),
                    "Timestamp": str(results['timestamp'])
                })
            
            with col2:
                st.markdown("#### Configuration Used")
                st.json(results['config'])
            
            # Sample data
            st.markdown("#### Sample Extracted Data")
            
            if results['recommendations']:
                st.markdown("**Sample Recommendation:**")
                rec = results['recommendations'][0]
                st.json({
                    "Text": rec.sentence[:200] + "...",
                    "Confidence": rec.confidence_score,
                    "Document": rec.document['filename'],
                    "Type": rec.content_type
                })
            
            if results['responses']:
                st.markdown("**Sample Response:**")
                resp = results['responses'][0]
                st.json({
                    "Text": resp.sentence[:200] + "...",
                    "Confidence": resp.confidence_score,
                    "Document": resp.document['filename'],
                    "Type": resp.content_type
                })
            
            # Semantic matcher status
            st.markdown("#### Semantic Matcher Status")
            try:
                matcher = SemanticMatcher()
                st.success(f"✅ Semantic matcher initialized (AI: {matcher.use_transformer})")
            except Exception as e:
                st.error(f"❌ Semantic matcher error: {e}")
        else:
            st.info("No results to debug. Run analysis first.")

# =============================================================================
# 🔧 HELPER FUNCTIONS
# =============================================================================

def auto_detect_document_types(documents: List[Dict]) -> tuple:
    """Auto-detect recommendation and response documents"""
    
    rec_docs = []
    resp_docs = []
    
    for doc in documents:
        filename_lower = doc['filename'].lower()
        text_preview = doc.get('text', '')[:5000].lower()
        
        # Check for response indicators
        if ('response' in filename_lower or 
            'government response' in text_preview or
            'accepting in full' in text_preview or
            'accepting in principle' in text_preview):
            resp_docs.append(doc['filename'])
        
        # Check for recommendation indicators
        if ('inquiry' in filename_lower or
            'report' in filename_lower or
            'recommendation' in filename_lower or
            'we recommend' in text_preview or
            'the committee recommends' in text_preview):
            rec_docs.append(doc['filename'])
    
    # If a document appears in both, prioritize based on stronger signals
    overlap = set(rec_docs) & set(resp_docs)
    for doc_name in overlap:
        doc = next(d for d in documents if d['filename'] == doc_name)
        if 'response' in doc['filename'].lower():
            rec_docs.remove(doc_name)
        else:
            resp_docs.remove(doc_name)
    
    return rec_docs, resp_docs

def export_alignments_to_csv(alignments: List[Dict]) -> str:
    """Export alignments to CSV format"""
    
    rows = []
    for alignment in alignments:
        rec = alignment['recommendation']
        
        if alignment['responses']:
            for resp_match in alignment['responses']:
                resp = resp_match['response']
                rows.append({
                    'Recommendation': rec.sentence,
                    'Rec_Document': rec.document['filename'],
                    'Rec_Page': rec.position_info.get('page_number', ''),
                    'Response': resp.sentence,
                    'Resp_Document': resp.document['filename'],
                    'Resp_Page': resp.position_info.get('page_number', ''),
                    'Confidence': resp_match['combined_score'],
                    'Quality': resp_match['match_quality'],
                    'Explanation': resp_match['explanation']
                })
        else:
            rows.append({
                'Recommendation': rec.sentence,
                'Rec_Document': rec.document['filename'],
                'Rec_Page': rec.position_info.get('page_number', ''),
                'Response': 'No response found',
                'Resp_Document': '',
                'Resp_Page': '',
                'Confidence': 0,
                'Quality': 'No Match',
                'Explanation': ''
            })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def export_alignments_to_excel(alignments: List[Dict]) -> bytes:
    """Export alignments to Excel format"""
    import io
    
    # Create DataFrame
    rows = []
    for alignment in alignments:
        rec = alignment['recommendation']
        
        if alignment['responses']:
            for resp_match in alignment['responses']:
                resp = resp_match['response']
                rows.append({
                    'Recommendation': rec.sentence,
                    'Rec_Document': rec.document['filename'],
                    'Rec_Page': rec.position_info.get('page_number', ''),
                    'Response': resp.sentence,
                    'Resp_Document': resp.document['filename'],
                    'Resp_Page': resp.position_info.get('page_number', ''),
                    'Confidence': resp_match['combined_score'],
                    'Quality': resp_match['match_quality']
                })
        else:
            rows.append({
                'Recommendation': rec.sentence,
                'Rec_Document': rec.document['filename'],
                'Rec_Page': rec.position_info.get('page_number', ''),
                'Response': 'No response found',
                'Resp_Document': '',
                'Resp_Page': '',
                'Confidence': 0,
                'Quality': 'No Match'
            })
    
    df = pd.DataFrame(rows)
    
    # Create Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Alignments', index=False)
    
    return output.getvalue()

def generate_alignment_report(alignments: List[Dict], results: Dict) -> str:
    """Generate a markdown report of alignments"""
    
    report = f"""# Recommendation-Response Alignment Report
    
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

- **Total Recommendations:** {len(results['recommendations'])}
- **Total Responses:** {len(results['responses'])}
- **Aligned Recommendations:** {len([a for a in alignments if a['has_response']])}
- **Average Confidence:** {sum(a.get('alignment_confidence', 0) for a in alignments) / len(alignments):.2%}

## Configuration Used

```json
{results['config']}
```

## Detailed Alignments

"""
    
    for i, alignment in enumerate(alignments, 1):
        rec = alignment['recommendation']
        
        report += f"""
### Alignment {i}

**Recommendation:**
> {rec.sentence}

- Document: {rec.document['filename']}
- Page: ~{rec.position_info.get('page_number', 'Unknown')}
- Confidence: {rec.confidence_score:.2%}

"""
        
        if alignment['responses']:
            report += "**Responses:**\n\n"
            for j, resp_match in enumerate(alignment['responses'], 1):
                resp = resp_match['response']
                report += f"""
**Response {j}** (Match: {resp_match['combined_score']:.2%} - {resp_match['match_quality']})
> {resp.sentence}

- Document: {resp.document['filename']}
- Page: ~{resp.position_info.get('page_number', 'Unknown')}
- Explanation: {resp_match['explanation']}

"""
        else:
            report += "**No responses found**\n\n"
        
        report += "---\n\n"
    
    return report

# Export the main function
__all__ = ['render_recommendation_alignment_interface']
