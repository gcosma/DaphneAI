# ===============================================
# FILE: modules/ui/matching_components.py
# Response Matching Components for DaphneAI
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

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
# MATCHING ALGORITHM CLASSES
# ===============================================

class RecommendationMatcher:
    """Matches recommendations with government responses"""
    
    def __init__(self):
        self.similarity_threshold = 0.3
    
    def find_matches(self, recommendations: List[Dict], responses: List[Dict]) -> List[Dict]:
        """Find matches between recommendations and responses"""
        matches = []
        
        for i, rec in enumerate(recommendations):
            rec_matches = []
            
            for j, resp in enumerate(responses):
                similarity = self.calculate_similarity(rec, resp)
                
                if similarity >= self.similarity_threshold:
                    rec_matches.append({
                        'response': resp,
                        'response_index': j,
                        'similarity': similarity,
                        'match_type': self.determine_match_type(similarity)
                    })
            
            # Sort by similarity
            rec_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            matches.append({
                'recommendation': rec,
                'recommendation_index': i,
                'matches': rec_matches,
                'best_match': rec_matches[0] if rec_matches else None,
                'match_count': len(rec_matches)
            })
        
        return matches
    
    def calculate_similarity(self, rec: Dict, resp: Dict) -> float:
        """Calculate similarity between recommendation and response"""
        rec_text = rec['text'].lower()
        resp_text = resp['text'].lower()
        
        # Simple keyword-based similarity
        rec_words = set(re.findall(r'\b\w+\b', rec_text))
        resp_words = set(re.findall(r'\b\w+\b', resp_text))
        
        # Remove common words
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'upon', 'against',
            'within', 'without', 'throughout', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'
        }
        
        rec_words -= common_words
        resp_words -= common_words
        
        if not rec_words or not resp_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(rec_words & resp_words)
        union = len(rec_words | resp_words)
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Boost similarity for key government terms
        government_terms = {
            'implement', 'implementation', 'accept', 'accepted', 'reject', 'rejected',
            'consider', 'review', 'action', 'policy', 'department', 'government',
            'minister', 'parliament', 'legislation', 'guidance', 'framework'
        }
        
        gov_boost = 0.0
        for term in government_terms:
            if term in rec_text and term in resp_text:
                gov_boost += 0.05
        
        return min(1.0, jaccard_similarity + gov_boost)
    
    def determine_match_type(self, similarity: float) -> str:
        """Determine the type of match based on similarity score"""
        if similarity >= 0.7:
            return "Strong Match"
        elif similarity >= 0.5:
            return "Good Match"
        elif similarity >= 0.3:
            return "Possible Match"
        else:
            return "Weak Match"

def render_matching_tab():
    """Render the response matching interface"""
    st.header("🔗 Find Responses")
    st.markdown("""
    Match extracted recommendations with government responses to track implementation and acceptance.
    """)
    
    # Check if extraction results are available
    recommendations = st.session_state.get('extracted_recommendations', [])
    responses = st.session_state.get('extracted_responses', [])
    
    if not recommendations:
        st.warning("No recommendations available for matching.")
        st.info("👆 Please extract recommendations from documents first in the Extraction tab.")
        return
    
    if not responses:
        st.warning("No responses available for matching.")
        st.info("👆 Please extract government responses from documents first in the Extraction tab.")
        return
    
    # Matching configuration
    st.markdown("### ⚙️ Matching Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Minimum similarity score for considering a match"
        )
    
    with col2:
        max_matches_per_rec = st.number_input(
            "Max Matches per Recommendation",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of response matches to show per recommendation"
        )
    
    with col3:
        match_algorithm = st.selectbox(
            "Matching Algorithm",
            ["Keyword Similarity", "Advanced NLP"],
            help="Choose the matching algorithm to use"
        )
        
        if match_algorithm == "Advanced NLP":
            st.caption("⚠️ Advanced NLP requires additional setup")
    
    # Data overview
    st.markdown("### 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recommendations", len(recommendations))
    
    with col2:
        st.metric("Responses", len(responses))
    
    with col3:
        potential_matches = len(recommendations) * len(responses)
        st.metric("Potential Matches", potential_matches)
    
    with col4:
        # Count existing matches if available
        existing_matches = len(st.session_state.get('matching_results', {}).get('matches', []))
        st.metric("Found Matches", existing_matches)
    
    # Run matching
    if st.button("🔍 Find Matches", type="primary"):
        run_matching_analysis(recommendations, responses, similarity_threshold, max_matches_per_rec)
    
    # Show existing results
    if st.session_state.get('matching_results'):
        st.markdown("---")
        show_matching_results()

def run_matching_analysis(recommendations: List[Dict], responses: List[Dict], threshold: float, max_matches: int):
    """Run the matching analysis"""
    
    with st.spinner("Analyzing matches between recommendations and responses..."):
        # Initialize matcher
        matcher = RecommendationMatcher()
        matcher.similarity_threshold = threshold
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Find matches
        status_text.text("Finding matches...")
        matches = matcher.find_matches(recommendations, responses)
        
        # Filter and limit matches
        filtered_matches = []
        total_matches = 0
        
        for match in matches:
            # Limit matches per recommendation
            match['matches'] = match['matches'][:max_matches]
            
            if match['matches']:
                filtered_matches.append(match)
                total_matches += len(match['matches'])
        
        progress_bar.progress(1.0)
        status_text.text("✅ Matching analysis completed!")
        
        # Store results
        matching_results = {
            'matches': filtered_matches,
            'statistics': {
                'total_recommendations': len(recommendations),
                'total_responses': len(responses),
                'matched_recommendations': len(filtered_matches),
                'total_matches': total_matches,
                'matching_rate': len(filtered_matches) / len(recommendations) if recommendations else 0,
                'avg_matches_per_rec': total_matches / len(filtered_matches) if filtered_matches else 0
            },
            'parameters': {
                'similarity_threshold': threshold,
                'max_matches_per_rec': max_matches,
                'algorithm': 'keyword_similarity'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.matching_results = matching_results
        
        # Show summary
        show_matching_summary(matching_results)
        
        # Log action
        if CORE_UTILS_AVAILABLE:
            log_user_action("matching_analysis", {
                'recommendations_count': len(recommendations),
                'responses_count': len(responses),
                'matches_found': total_matches,
                'matching_rate': matching_results['statistics']['matching_rate']
            })

def show_matching_summary(results: Dict):
    """Show summary of matching results"""
    stats = results['statistics']
    
    st.markdown("### 🎉 Matching Analysis Complete!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Matched Recommendations", stats['matched_recommendations'])
    
    with col2:
        st.metric("Total Matches", stats['total_matches'])
    
    with col3:
        st.metric("Matching Rate", f"{stats['matching_rate']:.1%}")
    
    with col4:
        st.metric("Avg Matches/Rec", f"{stats['avg_matches_per_rec']:.1f}")
    
    if stats['matched_recommendations'] > 0:
        st.success("✅ Successfully found matches between recommendations and responses!")
    else:
        st.warning("⚠️ No matches found. Try lowering the similarity threshold.")

def show_matching_results():
    """Display detailed matching results"""
    results = st.session_state.get('matching_results', {})
    matches = results.get('matches', [])
    
    if not matches:
        st.info("No matching results available.")
        return
    
    st.markdown("### 🔍 Matching Results")
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Detailed Matches", "📊 Statistics", "📥 Export"])
    
    with tab1:
        show_detailed_matches(matches)
    
    with tab2:
        show_matching_statistics(results)
    
    with tab3:
        show_export_options(results)

def show_detailed_matches(matches: List[Dict]):
    """Show detailed match results"""
    st.markdown(f"### Found {len(matches)} Recommendations with Matches")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        min_similarity = st.slider(
            "Minimum Similarity to Display",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
    
    with col2:
        match_type_filter = st.multiselect(
            "Filter by Match Type",
            ["Strong Match", "Good Match", "Possible Match", "Weak Match"],
            default=["Strong Match", "Good Match", "Possible Match"]
        )
    
    # Display matches
    for i, match in enumerate(matches):
        recommendation = match['recommendation']
        rec_matches = match['matches']
        
        # Filter matches based on criteria
        filtered_rec_matches = [
            m for m in rec_matches 
            if m['similarity'] >= min_similarity and m['match_type'] in match_type_filter
        ]
        
        if not filtered_rec_matches:
            continue
        
        with st.expander(f"Recommendation {i+1}: {recommendation['text'][:100]}..."):
            st.markdown(f"**Full Recommendation:** {recommendation['text']}")
            st.markdown(f"**Source Document:** {recommendation['document']}")
            st.markdown(f"**Found {len(filtered_rec_matches)} matching responses:**")
            
            for j, resp_match in enumerate(filtered_rec_matches):
                response = resp_match['response']
                similarity = resp_match['similarity']
                match_type = resp_match['match_type']
                
                # Color code match types
                if match_type == "Strong Match":
                    color = "🟢"
                elif match_type == "Good Match":
                    color = "🟡"
                elif match_type == "Possible Match":
                    color = "🟠"
                else:
                    color = "🔴"
                
                st.markdown(f"**{color} Match {j+1} ({match_type}, {similarity:.2f}):**")
                st.markdown(f"Response: {response['text']}")
                st.markdown(f"Source: {response['document']}")
                st.markdown("---")

def show_matching_statistics(results: Dict):
    """Show matching statistics and analysis"""
    stats = results['statistics']
    matches = results['matches']
    
    st.markdown("### 📈 Matching Statistics")
    
    # Overall statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Overall Performance")
        st.metric("Total Recommendations", stats['total_recommendations'])
        st.metric("Recommendations with Matches", stats['matched_recommendations'])
        st.metric("Total Matches Found", stats['total_matches'])
        st.metric("Overall Matching Rate", f"{stats['matching_rate']:.1%}")
    
    with col2:
        st.markdown("#### Match Quality Distribution")
        
        # Count match types
        match_type_counts = {
            "Strong Match": 0,
            "Good Match": 0,
            "Possible Match": 0,
            "Weak Match": 0
        }
        
        for match in matches:
            for resp_match in match['matches']:
                match_type = resp_match['match_type']
                match_type_counts[match_type] += 1
        
        # Display match type distribution
        for match_type, count in match_type_counts.items():
            st.metric(match_type, count)
    
    # Similarity distribution
    st.markdown("#### Similarity Score Distribution")
    
    if matches:
        all_similarities = []
        for match in matches:
            for resp_match in match['matches']:
                all_similarities.append(resp_match['similarity'])
        
        if all_similarities:
            # Simple statistics
            avg_similarity = sum(all_similarities) / len(all_similarities)
            max_similarity = max(all_similarities)
            min_similarity = min(all_similarities)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Similarity", f"{avg_similarity:.3f}")
            with col2:
                st.metric("Highest Similarity", f"{max_similarity:.3f}")
            with col3:
                st.metric("Lowest Similarity", f"{min_similarity:.3f}")
    
    # Document coverage analysis
    st.markdown("#### Document Coverage")
    
    rec_docs = {}
    resp_docs = {}
    
    for match in matches:
        rec_doc = match['recommendation']['document']
        rec_docs[rec_doc] = rec_docs.get(rec_doc, 0) + 1
        
        for resp_match in match['matches']:
            resp_doc = resp_match['response']['document']
            resp_docs[resp_doc] = resp_docs.get(resp_doc, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Recommendation Documents:**")
        rec_df = pd.DataFrame([
            {'Document': doc, 'Matched Recommendations': count}
            for doc, count in rec_docs.items()
        ])
        st.dataframe(rec_df)
    
    with col2:
        st.markdown("**Response Documents:**")
        resp_df = pd.DataFrame([
            {'Document': doc, 'Matches Found': count}
            for doc, count in resp_docs.items()
        ])
        st.dataframe(resp_df)

def show_export_options(results: Dict):
    """Show export options for matching results"""
    st.markdown("### 📥 Export Matching Results")
    
    matches = results['matches']
    
    if not matches:
        st.info("No matches to export.")
        return
    
    # Export format selection
    export_format = st.radio(
        "Choose export format:",
        ["Detailed CSV", "Summary CSV", "JSON"],
        horizontal=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Export File"):
            generate_export_file(matches, export_format)
    
    with col2:
        if st.button("📋 Copy Summary to Clipboard"):
            generate_summary_text(results)

def generate_export_file(matches: List[Dict], export_format: str):
    """Generate export file in specified format"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == "Detailed CSV":
            # Detailed CSV with all match information
            export_data = []
            
            for match in matches:
                recommendation = match['recommendation']
                
                for resp_match in match['matches']:
                    response = resp_match['response']
                    
                    export_data.append({
                        'Recommendation_Text': recommendation['text'],
                        'Recommendation_Document': recommendation['document'],
                        'Recommendation_Confidence': recommendation['confidence'],
                        'Response_Text': response['text'],
                        'Response_Document': response['document'],
                        'Response_Confidence': response['confidence'],
                        'Match_Similarity': resp_match['similarity'],
                        'Match_Type': resp_match['match_type']
                    })
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            filename = f"detailed_matches_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download Detailed CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
        
        elif export_format == "Summary CSV":
            # Summary CSV with one row per recommendation
            export_data = []
            
            for match in matches:
                recommendation = match['recommendation']
                best_match = match['best_match']
                
                export_data.append({
                    'Recommendation_Text': recommendation['text'],
                    'Recommendation_Document': recommendation['document'],
                    'Match_Count': len(match['matches']),
                    'Best_Match_Similarity': best_match['similarity'] if best_match else 0,
                    'Best_Match_Type': best_match['match_type'] if best_match else 'No Match',
                    'Best_Response_Text': best_match['response']['text'] if best_match else '',
                    'Best_Response_Document': best_match['response']['document'] if best_match else ''
                })
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            filename = f"summary_matches_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
        
        elif export_format == "JSON":
            # JSON export with full structure
            import json
            
            json_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_matches': len(matches),
                'matches': matches
            }
            
            json_str = json.dumps(json_data, indent=2, default=str)
            filename = f"matches_{timestamp}.json"
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )
        
        st.success(f"✅ {export_format} export ready for download!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        logger.error(f"Export error: {e}")

def generate_summary_text(results: Dict):
    """Generate summary text for clipboard"""
    try:
        stats = results['statistics']
        matches = results['matches']
        
        summary = f"""
MATCHING ANALYSIS SUMMARY
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total Recommendations: {stats['total_recommendations']}
- Total Responses: {stats['total_responses']}
- Recommendations with Matches: {stats['matched_recommendations']}
- Total Matches Found: {stats['total_matches']}
- Overall Matching Rate: {stats['matching_rate']:.1%}

TOP MATCHES:
"""
        
        # Add top 5 matches
        sorted_matches = sorted(matches, 
                              key=lambda x: x['best_match']['similarity'] if x['best_match'] else 0, 
                              reverse=True)
        
        for i, match in enumerate(sorted_matches[:5]):
            if match['best_match']:
                best = match['best_match']
                summary += f"\n{i+1}. {match['recommendation']['text'][:100]}..."
                summary += f"\n   → {best['response']['text'][:100]}..."
                summary += f"\n   Similarity: {best['similarity']:.3f} ({best['match_type']})\n"
        
        # Copy to clipboard (note: this is a simplified approach)
        st.text_area("Summary Text (copy manually):", summary, height=300)
        st.info("📋 Copy the text above to your clipboard")
        
    except Exception as e:
        st.error(f"❌ Failed to generate summary: {str(e)}")

# ===============================================
# INITIALIZATION
# ===============================================

# Initialize session state for matching
if 'matching_results' not in st.session_state:
    st.session_state.matching_results = {}

logger.info("✅ Matching components initialized")
