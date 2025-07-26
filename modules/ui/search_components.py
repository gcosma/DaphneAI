# ===============================================
# FILE: modules/ui/search_components.py
# Smart Search Components for DaphneAI
# ===============================================

import streamlit as st
import pandas as pd
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

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
# SEARCH ENGINE CLASS
# ===============================================

class SmartSearchEngine:
    """Smart search engine for recommendations and responses"""
    
    def __init__(self):
        self.search_history = []
    
    def search(self, query: str, content_items: List[Dict], search_type: str = "all") -> List[Dict]:
        """Perform smart search across content items"""
        
        if not query.strip():
            return []
        
        results = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        for item in content_items:
            score = self.calculate_relevance_score(query, query_words, item)
            
            if score > 0:
                results.append({
                    'item': item,
                    'score': score,
                    'match_type': self.determine_match_type(score),
                    'query': query,
                    'search_type': search_type
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add to search history
        self.add_to_history(query, len(results), search_type)
        
        return results
    
    def calculate_relevance_score(self, query: str, query_words: set, item: Dict) -> float:
        """Calculate relevance score for an item"""
        text = item.get('text', '').lower()
        
        if not text:
            return 0.0
        
        score = 0.0
        
        # Exact phrase match (highest score)
        if query.lower() in text:
            score += 1.0
        
        # Word matches
        text_words = set(re.findall(r'\b\w+\b', text))
        word_matches = len(query_words & text_words)
        word_score = word_matches / len(query_words) if query_words else 0
        score += word_score * 0.7
        
        # Boost for key terms
        key_terms = {
            'recommend', 'implementation', 'accept', 'reject', 'consider',
            'policy', 'government', 'minister', 'department', 'action'
        }
        
        for term in key_terms:
            if term in query.lower() and term in text:
                score += 0.1
        
        # Document metadata boost
        if 'metadata' in item:
            metadata_text = str(item['metadata']).lower()
            if any(word in metadata_text for word in query_words):
                score += 0.2
        
        # Confidence boost (if available)
        if 'confidence' in item:
            score += item['confidence'] * 0.1
        
        return score
    
    def determine_match_type(self, score: float) -> str:
        """Determine match type based on score"""
        if score >= 1.5:
            return "Excellent Match"
        elif score >= 1.0:
            return "Very Good Match"
        elif score >= 0.7:
            return "Good Match"
        elif score >= 0.4:
            return "Fair Match"
        else:
            return "Weak Match"
    
    def add_to_history(self, query: str, result_count: int, search_type: str):
        """Add search to history"""
        self.search_history.append({
            'query': query,
            'result_count': result_count,
            'search_type': search_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 50 searches
        if len(self.search_history) > 50:
            self.search_history = self.search_history[-50:]

def render_search_tab():
    """Render the smart search interface"""
    st.header("🔎 Smart Search")
    st.markdown("""
    Search across all extracted recommendations and responses using intelligent matching algorithms.
    """)
    
    # Check if content is available
    recommendations = st.session_state.get('extracted_recommendations', [])
    responses = st.session_state.get('extracted_responses', [])
    
    if not recommendations and not responses:
        st.warning("No content available for search.")
        st.info("👆 Please extract content from documents first in the Extraction tab.")
        return
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = SmartSearchEngine()
    
    search_engine = st.session_state.search_engine
    
    # Search interface
    render_search_interface(search_engine, recommendations, responses)
    
    # Search history
    if st.session_state.get('show_search_history', False):
        render_search_history(search_engine)

def render_search_interface(search_engine: SmartSearchEngine, recommendations: List[Dict], responses: List[Dict]):
    """Render the main search interface"""
    
    # Search configuration
    st.markdown("### 🔍 Search Configuration")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter keywords, phrases, or topics...",
            help="Search across recommendations and responses"
        )
    
    with col2:
        search_scope = st.selectbox(
            "Search Scope",
            ["All Content", "Recommendations Only", "Responses Only"]
        )
    
    with col3:
        max_results = st.number_input(
            "Max Results",
            min_value=5,
            max_value=100,
            value=20,
            help="Maximum number of results to display"
        )
    
    # Advanced search options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_score = st.slider(
                "Minimum Relevance Score",
                min_value=0.0,
                max_value=2.0,
                value=0.1,
                step=0.1,
                help="Filter results by minimum relevance score"
            )
            
            include_metadata = st.checkbox(
                "Search in Metadata",
                value=True,
                help="Include document metadata in search"
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort Results By",
                ["Relevance Score", "Confidence", "Document Name", "Date"]
            )
            
            case_sensitive = st.checkbox(
                "Case Sensitive",
                value=False,
                help="Make search case sensitive"
            )
    
    # Search execution
    if st.button("🔍 Search", type="primary") or search_query:
        if search_query.strip():
            execute_search(search_engine, search_query, search_scope, recommendations, responses, 
                         max_results, min_score, sort_by)
        else:
            st.warning("Please enter a search query.")
    
    # Quick search suggestions
    render_quick_search_suggestions(search_engine, recommendations, responses)

def execute_search(search_engine: SmartSearchEngine, query: str, scope: str, 
                  recommendations: List[Dict], responses: List[Dict], 
                  max_results: int, min_score: float, sort_by: str):
    """Execute the search and display results"""
    
    with st.spinner("Searching..."):
        # Determine content to search
        if scope == "Recommendations Only":
            content_items = recommendations
            search_type = "recommendations"
        elif scope == "Responses Only":
            content_items = responses
            search_type = "responses"
        else:
            content_items = recommendations + responses
            search_type = "all"
        
        # Perform search
        results = search_engine.search(query, content_items, search_type)
        
        # Filter by minimum score
        filtered_results = [r for r in results if r['score'] >= min_score]
        
        # Limit results
        limited_results = filtered_results[:max_results]
        
        # Store results in session state
        st.session_state.search_results = {
            'query': query,
            'results': limited_results,
            'total_found': len(filtered_results),
            'scope': scope,
            'timestamp': datetime.now().isoformat()
        }
        
        # Display results
        display_search_results(limited_results, query, len(filtered_results), max_results)
        
        # Log search
        if CORE_UTILS_AVAILABLE:
            log_user_action("smart_search", {
                'query': query,
                'scope': scope,
                'results_found': len(filtered_results),
                'results_shown': len(limited_results)
            })

def display_search_results(results: List[Dict], query: str, total_found: int, max_results: int):
    """Display search results"""
    
    if not results:
        st.warning(f"No results found for '{query}'")
        return
    
    st.markdown(f"### 🎯 Search Results for '{query}'")
    st.info(f"Found {total_found} results" + (f", showing top {len(results)}" if total_found > max_results else ""))
    
    # Results display options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_mode = st.radio(
            "Display Mode",
            ["Expanded", "Compact", "Table"],
            horizontal=True
        )
    
    with col2:
        if st.button("📥 Export Results"):
            export_search_results(results, query)
    
    with col3:
        if st.button("🔄 Clear Results"):
            if 'search_results' in st.session_state:
                del st.session_state.search_results
            st.rerun()
    
    # Display results based on mode
    if display_mode == "Expanded":
        display_expanded_results(results)
    elif display_mode == "Compact":
        display_compact_results(results)
    else:
        display_table_results(results)

def display_expanded_results(results: List[Dict]):
    """Display results in expanded format"""
    
    for i, result in enumerate(results):
        item = result['item']
        score = result['score']
        match_type = result['match_type']
        
        # Determine content type
        content_type = "📋 Recommendation" if 'extracted_recommendations' in str(type(item)) else "💬 Response"
        
        with st.expander(f"Result {i+1}: {match_type} (Score: {score:.2f})"):
            st.markdown(f"**{content_type}**")
            st.markdown(f"**Text:** {item['text']}")
            st.markdown(f"**Document:** {item.get('document', 'Unknown')}")
            st.markdown(f"**Confidence:** {item.get('confidence', 'N/A')}")
            st.markdown(f"**Relevance Score:** {score:.3f}")
            
            if 'metadata' in item:
                with st.expander("📄 Metadata"):
                    st.json(item['metadata'])

def display_compact_results(results: List[Dict]):
    """Display results in compact format"""
    
    for i, result in enumerate(results):
        item = result['item']
        score = result['score']
        match_type = result['match_type']
        
        # Truncate text for compact display
        text_preview = item['text'][:150] + "..." if len(item['text']) > 150 else item['text']
        
        st.markdown(f"**{i+1}.** {text_preview}")
        st.caption(f"Score: {score:.2f} | {match_type} | Document: {item.get('document', 'Unknown')}")
        st.markdown("---")

def display_table_results(results: List[Dict]):
    """Display results in table format"""
    
    table_data = []
    for i, result in enumerate(results):
        item = result['item']
        score = result['score']
        match_type = result['match_type']
        
        table_data.append({
            'Rank': i + 1,
            'Text Preview': item['text'][:100] + "..." if len(item['text']) > 100 else item['text'],
            'Document': item.get('document', 'Unknown'),
            'Confidence': item.get('confidence', 'N/A'),
            'Score': round(score, 3),
            'Match Type': match_type
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

def render_quick_search_suggestions(search_engine: SmartSearchEngine, recommendations: List[Dict], responses: List[Dict]):
    """Render quick search suggestions"""
    
    st.markdown("### ⚡ Quick Search")
    
    # Generate suggested searches based on content
    suggestions = generate_search_suggestions(recommendations, responses)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏥 Healthcare"):
            execute_quick_search("healthcare health medical NHS", search_engine, recommendations, responses)
    
    with col2:
        if st.button("🏫 Education"):
            execute_quick_search("education school university learning", search_engine, recommendations, responses)
    
    with col3:
        if st.button("🏛️ Policy"):
            execute_quick_search("policy government implementation", search_engine, recommendations, responses)
    
    with col4:
        if st.button("💰 Finance"):
            execute_quick_search("finance funding budget cost", search_engine, recommendations, responses)
    
    # Recent searches
    if search_engine.search_history:
        st.markdown("#### 🕒 Recent Searches")
        recent_searches = search_engine.search_history[-5:]
        
        for search in reversed(recent_searches):
            if st.button(f"🔍 '{search['query']}'  ({search['result_count']} results)", key=f"recent_{search['query']}"):
                execute_quick_search(search['query'], search_engine, recommendations, responses)

def execute_quick_search(query: str, search_engine: SmartSearchEngine, recommendations: List[Dict], responses: List[Dict]):
    """Execute a quick search"""
    execute_search(search_engine, query, "All Content", recommendations, responses, 20, 0.1, "Relevance Score")

def generate_search_suggestions(recommendations: List[Dict], responses: List[Dict]) -> List[str]:
    """Generate search suggestions based on content analysis"""
    
    all_text = ""
    for item in recommendations + responses:
        all_text += " " + item.get('text', '')
    
    # Extract common terms (simplified approach)
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 4:  # Only words longer than 4 characters
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top words
    common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return [word for word, freq in common_words]

def render_search_history(search_engine: SmartSearchEngine):
    """Render search history"""
    
    st.markdown("### 📚 Search History")
    
    if not search_engine.search_history:
        st.info("No search history available.")
        return
    
    # Display history in table format
    history_data = []
    for search in reversed(search_engine.search_history):
        history_data.append({
            'Query': search['query'],
            'Results': search['result_count'],
            'Type': search['search_type'],
            'Time': search['timestamp'][:19].replace('T', ' ')
        })
    
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True)
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        search_engine.search_history = []
        st.success("Search history cleared.")
        st.rerun()

def export_search_results(results: List[Dict], query: str):
    """Export search results to CSV"""
    
    try:
        export_data = []
        
        for i, result in enumerate(results):
            item = result['item']
            
            export_data.append({
                'Rank': i + 1,
                'Query': query,
                'Text': item['text'],
                'Document': item.get('document', 'Unknown'),
                'Original_Confidence': item.get('confidence', 'N/A'),
                'Search_Score': result['score'],
                'Match_Type': result['match_type'],
                'Search_Timestamp': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        
        filename = f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button(
            label="📥 Download Search Results",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        st.success("✅ Search results ready for download")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

# ===============================================
# INITIALIZATION
# ===============================================

# Initialize session state for search
if 'search_results' not in st.session_state:
    st.session_state.search_results = {}

if 'show_search_history' not in st.session_state:
    st.session_state.show_search_history = False

logger.info("✅ Search components initialized")
