# simplified_alignment_ui.py
"""
SIMPLIFIED Recommendation-Response Alignment
- Self-match prevention ALWAYS ON (no option needed)
- ALL keywords included by default
- Much simpler interface
"""

import streamlit as st
from typing import List, Dict, Any
import re
from difflib import SequenceMatcher

def render_simple_alignment_interface(documents: List[Dict[str, Any]]):
    """Simple alignment interface with self-match prevention built in"""
    
    # Removed the heading - it's already shown by the main app
    st.markdown("*Automatically finds and aligns recommendations with government responses*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        return
    
    # Initialize session state for results
    if 'alignment_results' not in st.session_state:
        st.session_state.alignment_results = None
    if 'alignment_settings' not in st.session_state:
        st.session_state.alignment_settings = {}
    
    # Just one slider for similarity threshold
    similarity_threshold = st.slider(
        "Match sensitivity",
        min_value=0.2,
        max_value=0.8,
        value=0.4,
        step=0.1,
        help="Lower = more matches, Higher = only strong matches"
    )
    
    # Document selection (optional)
    doc_names = [doc['filename'] for doc in documents]
    selected_docs = st.multiselect(
        "📄 Select documents (or leave empty for all):",
        doc_names,
        default=[],  # Empty by default = use all
        help="Leave empty to search all documents"
    )
    
    # If no docs selected, use all
    if not selected_docs:
        selected_docs = doc_names
    
    # Single button to run everything
    if st.button("🚀 Find & Align Recommendations", type="primary"):
        with st.spinner("Finding recommendations and responses..."):
            
            # Step 1: Find recommendations (ALL keywords)
            rec_keywords = [
                'recommend', 'recommends', 'recommendation', 'we recommend',
                'should', 'must', 'suggest', 'advise', 'propose', 'urge',
                'it is recommended', 'the committee recommends',
                'we advise', 'we suggest', 'we propose'
            ]
            
            recommendations = find_all_patterns(
                [d for d in documents if d['filename'] in selected_docs],
                rec_keywords,
                "recommendation"
            )
            
            st.success(f"✅ Found {len(recommendations)} recommendations")
            
            # Step 2: Find responses (ALL keywords)
            resp_keywords = [
                'accept', 'accepts', 'accepted', 'accepting',
                'reject', 'rejects', 'rejected', 'rejecting',
                'agree', 'agrees', 'agreed', 'implement', 'implemented',
                'approve', 'approved', 'support', 'supported',
                'will implement', 'will consider', 'under consideration',
                'accept in principle', 'accept in full'
            ]
            
            responses = find_all_patterns(
                [d for d in documents if d['filename'] in selected_docs],
                resp_keywords,
                "response"
            )
            
            st.success(f"✅ Found {len(responses)} responses")
            
            # Step 3: Align with BUILT-IN self-match prevention
            alignments = align_with_self_match_prevention(
                recommendations,
                responses,
                similarity_threshold
            )
            
            st.success(f"✅ Created {len(alignments)} alignments (self-matches prevented)")
            
            # Store results in session state so they persist
            st.session_state.alignment_results = alignments
            st.session_state.alignment_settings = {
                'threshold': similarity_threshold,
                'documents': selected_docs
            }
    
    # Display results from session state (persists when switching tabs)
    if st.session_state.alignment_results is not None:
        display_simple_results(
            st.session_state.alignment_results, 
            st.session_state.alignment_settings.get('threshold', 0.4)
        )

def find_all_patterns(documents: List[Dict], keywords: List[str], match_type: str) -> List[Dict]:
    """Find all patterns in documents"""
    
    matches = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sent_idx, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
            
            sentence_lower = sentence.lower()
            
            # Check if any keyword is in sentence
            for keyword in keywords:
                if keyword in sentence_lower:
                    # Calculate position
                    char_position = text.find(sentence)
                    
                    matches.append({
                        'document': doc,
                        'sentence': sentence.strip(),
                        'keyword': keyword,
                        'type': match_type,
                        'position': char_position,
                        'sent_idx': sent_idx
                    })
                    break  # One match per sentence is enough

    return matches

def align_with_self_match_prevention(
    recommendations: List[Dict],
    responses: List[Dict],
    threshold: float
) -> List[Dict]:
    """Align recommendations with responses, PREVENTING self-matches"""
    
    alignments = []
    
    for rec in recommendations:
        matched_responses = []
        
        for resp in responses:
            # SELF-MATCH PREVENTION CHECKS
            
            # 1. Check if exact same text
            if rec['sentence'].strip().lower() == resp['sentence'].strip().lower():
                continue  # Skip - it's a self-match!
            
            # 2. Check if very similar (>90% similar)
            similarity = SequenceMatcher(
                None,
                rec['sentence'].lower(),
                resp['sentence'].lower()
            ).ratio()
            
            if similarity > 0.9:
                continue  # Skip - too similar, likely same content
            
            # 3. Check if same document and very close position
            if rec['document']['filename'] == resp['document']['filename']:
                if abs(rec['position'] - resp['position']) < 500:
                    continue  # Skip - too close in same document
            
            # 4. Calculate meaningful similarity for alignment
            if similarity >= threshold:
                matched_responses.append({
                    'response': resp,
                    'similarity': similarity,
                    'same_doc': rec['document']['filename'] == resp['document']['filename']
                })
        
        # Sort by similarity
        matched_responses.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Keep top 3 matches
        matched_responses = matched_responses[:3]
        
        alignments.append({
            'recommendation': rec,
            'responses': matched_responses,
            'has_response': len(matched_responses) > 0
        })
    
    return alignments

def display_simple_results(alignments: List[Dict], threshold: float):
    """Display results in a simple, clear format"""
    
    # Summary
    total = len(alignments)
    with_responses = sum(1 for a in alignments if a['has_response'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Recommendations", total)
    with col2:
        st.metric("With Responses", with_responses)
    with col3:
        st.metric("Without Responses", total - with_responses)
    
    # Filter options
    show_only_matched = st.checkbox("Show only recommendations with responses", value=True)
    
    # Display each alignment
    st.markdown("---")
    st.markdown("### 📋 Results")
    
    displayed = 0
    for alignment in alignments:
        if show_only_matched and not alignment['has_response']:
            continue
        
        displayed += 1
        
        rec = alignment['recommendation']
        responses = alignment['responses']
        
        # Color based on whether we found responses
        if responses:
            color = "🟢"
            status = f"{len(responses)} response(s) found"
        else:
            color = "🔴"
            status = "No response found"
        
        with st.expander(f"{color} Rec #{displayed}: {rec['sentence'][:80]}... | {status}"):
            # Show full recommendation
            st.markdown("**📝 Full Recommendation:**")
            st.info(rec['sentence'])
            st.caption(f"📁 Document: {rec['document']['filename']}")
            st.caption(f"🔍 Triggered by: '{rec['keyword']}'")
            
            if responses:
                st.markdown("**📢 Matched Response(s):**")
                
                for i, resp_match in enumerate(responses, 1):
                    resp = resp_match['response']
                    similarity = resp_match['similarity']
                    
                    # Show similarity as percentage
                    match_quality = "Strong" if similarity > 0.6 else "Moderate" if similarity > 0.4 else "Weak"
                    
                    st.success(f"Response {i} - {match_quality} match ({similarity:.0%})")
                    st.write(resp['sentence'])
                    st.caption(f"📁 Document: {resp['document']['filename']}")
                    st.caption(f"🔍 Triggered by: '{resp['keyword']}'")
                    
                    if resp_match['same_doc']:
                        st.caption("📌 Same document")
                    else:
                        st.caption("📤 Different document")
            else:
                st.warning("No responses found above the similarity threshold")
                st.caption(f"Try lowering the match sensitivity (currently {threshold:.1f})")
    
    if displayed == 0:
        st.info("No results to display. Try adjusting the filters.")
    
    # Export option - show download button directly if we have alignments
    if alignments and len(alignments) > 0:
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        csv_data = export_simple_csv(alignments)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="alignment_results.csv",
            mime="text/csv",
            help="Download all alignment results as a CSV file"
        )

def export_simple_csv(alignments: List[Dict]) -> str:
    """Export to CSV format with all details including confidence"""
    
    lines = ["ID,Recommendation,Rec_Document,Rec_Keyword,Response,Resp_Document,Resp_Keyword,Similarity,Match_Quality,Same_Document"]
    
    id_counter = 1
    for alignment in alignments:
        rec = alignment['recommendation']
        
        if alignment['responses']:
            for resp_match in alignment['responses']:
                resp = resp_match['response']
                similarity = resp_match['similarity']
                
                # Determine match quality
                if similarity > 0.7:
                    match_quality = "Strong"
                elif similarity > 0.5:
                    match_quality = "Good"
                elif similarity > 0.4:
                    match_quality = "Moderate"
                else:
                    match_quality = "Weak"
                
                # Clean text for CSV (remove line breaks and extra spaces)
                rec_text = rec["sentence"].replace('\n', ' ').replace('\r', ' ').strip()
                resp_text = resp["sentence"].replace('\n', ' ').replace('\r', ' ').strip()
                
                # Create CSV line
                lines.append(
                    f'{id_counter},'
                    f'"{rec_text}",'
                    f'"{rec["document"]["filename"]}",'
                    f'"{rec["keyword"]}",'
                    f'"{resp_text}",'
                    f'"{resp["document"]["filename"]}",'
                    f'"{resp["keyword"]}",'
                    f'{similarity:.2%},'
                    f'{match_quality},'
                    f'{"Yes" if resp_match["same_doc"] else "No"}'
                )
                id_counter += 1
        else:
            # No response found
            rec_text = rec["sentence"].replace('\n', ' ').replace('\r', ' ').strip()
            lines.append(
                f'{id_counter},'
                f'"{rec_text}",'
                f'"{rec["document"]["filename"]}",'
                f'"{rec["keyword"]}",'
                f'"No response found",'
                f'"",'
                f'"",'
                f'0%,'
                f'No Match,'
                f'N/A'
            )
            id_counter += 1
    
    return "\n".join(lines)

# Export for use in your app
__all__ = ['render_simple_alignment_interface']
