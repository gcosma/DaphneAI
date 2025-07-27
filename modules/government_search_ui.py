# government_search_ui.py
"""
Streamlit UI components for Government Document Search Engine
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import re

# Import the search engine (adjusted for your existing search_engine.py file)
from .search_engine import GovernmentSearchEngine, DocumentSearchResult, SearchFragment, SearchResultDisplay

class GovernmentSearchUI:
    """Streamlit UI for government document search"""
    
    def __init__(self):
        self.search_engine = None
        self._initialize_search_engine()
    
    def _initialize_search_engine(self):
        """Initialize search engine in session state"""
        if 'gov_search_engine' not in st.session_state:
            st.session_state.gov_search_engine = GovernmentSearchEngine()
            st.session_state.search_history = []
        
        self.search_engine = st.session_state.gov_search_engine
    
    def render_search_interface(self):
        """Main search interface"""
        st.title("🏛️ Government Document Search Engine")
        st.markdown("*Advanced search with stemming for recommendations, responses, and policy documents*")
        
        # Display search engine status
        self._render_search_status()
        
        # Check if documents are loaded
        if not hasattr(st.session_state, 'documents') or not st.session_state.documents:
            self._render_no_documents_message()
            return
        
        # Ensure documents are added to search engine
        if len(self.search_engine.documents) != len(st.session_state.documents):
            with st.spinner("🔍 Building search index with stemming..."):
                self.search_engine.add_documents(st.session_state.documents)
            st.success(f"✅ Indexed {len(st.session_state.documents)} documents for search")
        
        # Search configuration
        self._render_search_configuration()
        
        # Search history and quick searches
        self._render_search_shortcuts()
        
        # Main search results
        if 'current_search_results' in st.session_state:
            self._render_search_results(st.session_state.current_search_results)
    
    def _render_search_status(self):
        """Display search engine capabilities"""
        stats = self.search_engine.get_search_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("📝 Sentences", stats.get('total_sentences', 0))
        with col3:
            semantic_status = "✅ Enabled" if stats.get('semantic_enabled') else "❌ Disabled"
            st.metric("🤖 Semantic Search", semantic_status)
        with col4:
            stemming_status = "✅ Active" if hasattr(self.search_engine, 'stemmer') else "❌ Inactive"
            st.metric("🌱 Stemming", stemming_status)
    
    def _render_no_documents_message(self):
        """Show message when no documents are available"""
        st.warning("📁 No documents available for search")
        st.info("Please upload and process documents first using the Upload/Extract tabs.")
        
        # Sample data for testing
        if st.button("🎯 Load Sample Government Data"):
            sample_docs = [
                {
                    'id': 'doc1',
                    'filename': 'healthcare_recommendations.txt',
                    'text': '''
                    The Health Select Committee recommends that the Department of Health and Social Care should implement a comprehensive review of NHS waiting times. 
                    This recommendation is made with immediate effect and should be considered a priority.
                    The government should establish clear targets for reducing waiting times in emergency departments.
                    Implementation should begin within the next fiscal quarter.
                    The committee proposes additional funding for emergency services.
                    These recommendations require urgent governmental action.
                    ''',
                    'document_type': 'recommendation'
                },
                {
                    'id': 'doc2', 
                    'filename': 'government_response_health.txt',
                    'text': '''
                    The Government accepts the recommendation regarding NHS waiting times.
                    The Department of Health and Social Care will establish a task force to review current procedures.
                    We reject the proposal for immediate implementation due to budget constraints.
                    However, we will consider a phased approach beginning in Q2 2025.
                    The Minister agrees that this is a priority issue requiring urgent attention.
                    Implementation will be coordinated across all departments.
                    We are implementing new policies to address these concerns.
                    ''',
                    'document_type': 'response'
                },
                {
                    'id': 'doc3',
                    'filename': 'education_policy.txt', 
                    'text': '''
                    The Education Committee proposes new guidelines for school funding allocation.
                    These recommendations focus on supporting disadvantaged communities.
                    The policy framework should prioritize early childhood education.
                    Implementation requires coordination between local authorities and central government.
                    We recommend establishing dedicated funding streams for educational technology.
                    The government should implement these proposals within the academic year.
                    ''',
                    'document_type': 'policy'
                }
            ]
            st.session_state.documents = sample_docs
            st.rerun()
    
    def _render_search_configuration(self):
        """Render search configuration panel"""
        st.markdown("### 🔍 Search Configuration")
        
        # Main search input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="e.g., implement recommendations, NHS waiting, education funding...",
                help="Enter your search terms. The system will find exact matches and stemmed variations (e.g., 'implement' will also find 'implementation', 'implementing')"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Show stemming examples
        if query:
            with st.expander("🌱 Stemming Preview"):
                stemmed_query = self.search_engine._stem_text(query)
                st.write(f"**Original:** {query}")
                st.write(f"**Stemmed:** {stemmed_query}")
                
                # Show what variations will be searched
                variations = self.search_engine._generate_stemmed_query_variations(query)
                st.write("**Search Variations:**")
                for orig, stemmed, score in variations[:5]:
                    st.write(f"- {orig} → {stemmed} (weight: {score:.1f})")
        
        # Advanced options
        with st.expander("⚙️ Advanced Search Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_type = st.selectbox(
                    "Search Type",
                    ["comprehensive", "exact", "semantic"],
                    index=0,
                    help="Comprehensive: exact + stemmed + semantic; Exact: exact + stemmed matching; Semantic: AI-based meaning search"
                )
            
            with col2:
                max_fragments = st.number_input(
                    "Max Fragments per Document",
                    min_value=5,
                    max_value=100,
                    value=20,
                    help="Maximum number of relevant sentences to show per document"
                )
            
            with col3:
                min_score = st.slider(
                    "Relevance Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Filter out results below this relevance score"
                )
            
            # Document type filter
            st.markdown("**Filter by Document Type:**")
            doc_types = st.multiselect(
                "Document Types",
                ["recommendation", "response", "policy", "general"],
                default=["recommendation", "response", "policy", "general"],
                help="Filter results by document classification"
            )
            
            # Enable stemming toggle (for demonstration)
            use_stemming = st.checkbox(
                "Enable Stemming",
                value=True,
                help="Use word stemming to find variations (recommend → recommendation, implementing → implement)"
            )
        
        # Execute search
        if search_button and query.strip():
            self._execute_search(query, search_type, max_fragments, min_score, doc_types)
        elif query.strip() and st.session_state.get('auto_search', False):
            self._execute_search(query, search_type, max_fragments, min_score, doc_types)
    
    def _execute_search(self, query: str, search_type: str, max_fragments: int, 
                       min_score: float, doc_types: List[str]):
        """Execute search and store results"""
        with st.spinner("🔍 Searching with stemming and semantic analysis..."):
            start_time = datetime.now()
            
            # Perform search
            results = self.search_engine.search(
                query=query,
                search_type=search_type,
                max_results_per_doc=max_fragments,
                min_score=min_score
            )
            
            # Filter by document type
            if doc_types:
                results = [r for r in results if r.document_type in doc_types]
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            # Store results and add to history
            st.session_state.current_search_results = {
                'query': query,
                'results': results,
                'search_time': search_time,
                'total_fragments': sum(len(r.fragments) for r in results),
                'search_type': search_type,
                'timestamp': datetime.now()
            }
            
            # Add to search history
            self._add_to_search_history(query, len(results))
            
            # Show detailed results summary
            if results:
                st.success(f"✅ Found {len(results)} documents with {st.session_state.current_search_results['total_fragments']} relevant fragments in {search_time:.2f}s")
                
                # Show match type breakdown
                match_types = {}
                for result in results:
                    for fragment in result.fragments:
                        match_type = fragment.match_type
                        match_types[match_type] = match_types.get(match_type, 0) + 1
                
                if match_types:
                    st.info(f"Match types: {', '.join([f'{k}: {v}' for k, v in match_types.items()])}")
            else:
                st.warning(f"No results found for '{query}'. Try reducing the relevance threshold or using different terms.")
    
    def _render_search_shortcuts(self):
        """Render quick search options and history"""
        st.markdown("### ⚡ Quick Searches")
        
        # Predefined government searches with stemming-aware queries
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🏥 Healthcare"):
                self._quick_search("healthcare NHS medical recommendations")
        
        with col2:
            if st.button("🏫 Education"):
                self._quick_search("education school funding implement")
        
        with col3:
            if st.button("📋 Recommendations"):
                self._quick_search("recommend propose suggest advise")
        
        with col4:
            if st.button("📝 Responses"):
                self._quick_search("response accept reject implement")
        
        # Additional quick searches
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⚡ Implementation"):
                self._quick_search("implement implementation executing")
        
        with col2:
            if st.button("🎯 Priorities"):
                self._quick_search("priority urgent immediate critical")
        
        with col3:
            if st.button("💰 Funding"):
                self._quick_search("funding budget financial resources")
        
        with col4:
            if st.button("📊 Policy"):
                self._quick_search("policy framework strategy guidelines")
        
        # Search history
        if st.session_state.search_history:
            st.markdown("**🕒 Recent Searches:**")
            
            for i, search_item in enumerate(reversed(st.session_state.search_history[-5:])):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"🔍 '{search_item['query']}'", key=f"history_{i}"):
                        self._quick_search(search_item['query'])
                with col2:
                    st.caption(f"{search_item['results']} docs")
    
    def _quick_search(self, query: str):
        """Execute a quick search with default settings"""
        self._execute_search(query, "comprehensive", 20, 0.1, ["recommendation", "response", "policy", "general"])
    
    def _add_to_search_history(self, query: str, result_count: int):
        """Add search to history"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        # Remove duplicate if exists
        st.session_state.search_history = [
            item for item in st.session_state.search_history 
            if item['query'].lower() != query.lower()
        ]
        
        # Add new search
        st.session_state.search_history.append({
            'query': query,
            'results': result_count,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 searches
        if len(st.session_state.search_history) > 20:
            st.session_state.search_history = st.session_state.search_history[-20:]
    
    def _render_search_results(self, search_data: Dict[str, Any]):
        """Render comprehensive search results with all fragments"""
        results = search_data['results']
        
        if not results:
            st.warning(f"No results found for '{search_data['query']}'")
            self._show_search_tips()
            return
        
        # Results summary
        st.markdown(f"### 📊 Results for: *{search_data['query']}*")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents Found", len(results))
        with col2:
            st.metric("Total Fragments", search_data['total_fragments'])
        with col3:
            st.metric("Search Time", f"{search_data['search_time']:.2f}s")
        with col4:
            st.metric("Search Type", search_data['search_type'].title())
        
        # Results display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_mode = st.selectbox(
                "Display Mode",
                ["Fragment List", "Document View", "Compact View"],
                help="Fragment List shows all sentences; Document View groups by document"
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Relevance Score", "Document Type", "Fragment Count", "Match Type"],
                help="Sort results by different criteria"
            )
        
        with col3:
            if st.button("📊 Export Results"):
                self._export_results(search_data)
        
        # Filter by match type
        all_match_types = set()
        for result in results:
            for fragment in result.fragments:
                all_match_types.add(fragment.match_type)
        
        if len(all_match_types) > 1:
            st.markdown("**Filter by Match Type:**")
            selected_match_types = st.multiselect(
                "Match Types",
                list(all_match_types),
                default=list(all_match_types),
                help="Filter fragments by how they matched your query"
            )
        else:
            selected_match_types = list(all_match_types)
        
        # Sort results
        if sort_by == "Document Type":
            results.sort(key=lambda x: x.document_type)
        elif sort_by == "Fragment Count":
            results.sort(key=lambda x: len(x.fragments), reverse=True)
        elif sort_by == "Match Type":
            results.sort(key=lambda x: x.fragments[0].match_type if x.fragments else "")
        # Default is by relevance score (already sorted)
        
        # Display results based on mode
        if display_mode == "Fragment List":
            self._render_fragment_list(results, search_data['query'], selected_match_types)
        elif display_mode == "Document View":
            self._render_document_view(results, search_data['query'], selected_match_types)
        else:  # Compact View
            self._render_compact_results(results, search_data['query'], selected_match_types)
    
    def _render_fragment_list(self, results: List[DocumentSearchResult], query: str, match_types: List[str]):
        """Render all fragments as a flat list"""
        st.markdown("### 📝 All Relevant Fragments")
        
        # Collect all fragments
        all_fragments = []
        for doc_result in results:
            for fragment in doc_result.fragments:
                if fragment.match_type in match_types:
                    all_fragments.append({
                        'fragment': fragment,
                        'document': doc_result
                    })
        
        # Sort fragments by score
        all_fragments.sort(key=lambda x: x['fragment'].score, reverse=True)
        
        if not all_fragments:
            st.warning("No fragments match the selected criteria.")
            return
        
        st.info(f"Showing {len(all_fragments)} fragments from {len(results)} documents")
        
        # Display fragments
        for i, item in enumerate(all_fragments, 1):
            fragment = item['fragment']
            doc_result = item['document']
            
            # Fragment container
            with st.container():
                # Header with document info and score
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{i}. {doc_result.filename}**")
                
                with col2:
                    score_color = "🟢" if fragment.score > 0.8 else "🟡" if fragment.score > 0.5 else "🔴"
                    st.markdown(f"{score_color} {fragment.score:.2f}")
                
                with col3:
                    match_emoji = {"exact": "🎯", "stemmed": "🌱", "semantic": "🤖", "partial": "📝"}
                    st.markdown(f"{match_emoji.get(fragment.match_type, '📄')} {fragment.match_type}")
                
                with col4:
                    st.markdown(f"📑 {doc_result.document_type}")
                
                # Highlighted sentence
                highlighted_text = self._highlight_text(fragment.sentence, fragment.highlights)
                st.markdown(f"**Fragment:** {highlighted_text}")
                
                # Context (if different from sentence)
                if fragment.context != fragment.sentence:
                    with st.expander("📖 View Context"):
                        context_highlighted = self._highlight_text(fragment.context, fragment.highlights)
                        st.markdown(context_highlighted)
                
                # Show highlights
                if fragment.highlights:
                    st.markdown(f"**🔍 Highlights:** {', '.join(fragment.highlights)}")
                
                st.divider()
    
    def _render_document_view(self, results: List[DocumentSearchResult], query: str, match_types: List[str]):
        """Render results grouped by document"""
        st.markdown("### 📄 Results by Document")
        
        for i, doc_result in enumerate(results, 1):
            # Filter fragments by match type
            filtered_fragments = [f for f in doc_result.fragments if f.match_type in match_types]
            
            if not filtered_fragments:
                continue
            
            with st.expander(
                f"{i}. {doc_result.filename} ({len(filtered_fragments)} fragments, score: {doc_result.total_score:.2f})",
                expanded=i <= 3  # Expand first 3 results
            ):
                # Document metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document Type", doc_result.document_type.title())
                with col2:
                    st.metric("Fragments Found", len(filtered_fragments))
                with col3:
                    st.metric("Average Score", f"{doc_result.total_score:.2f}")
                
                # Show fragments
                for j, fragment in enumerate(filtered_fragments, 1):
                    st.markdown(f"**Fragment {j}:**")
                    
                    # Fragment details
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        highlighted_text = self._highlight_text(fragment.sentence, fragment.highlights)
                        st.markdown(highlighted_text)
                    
                    with col2:
                        match_emoji = {"exact": "🎯", "stemmed": "🌱", "semantic": "🤖", "partial": "📝"}
                        st.markdown(f"{match_emoji.get(fragment.match_type, '📄')} {fragment.match_type} ({fragment.score:.2f})")
                    
                    # Show highlights
                    if fragment.highlights:
                        st.caption(f"🔍 {', '.join(fragment.highlights)}")
                    
                    if j < len(filtered_fragments):
                        st.markdown("---")
    
    def _render_compact_results(self, results: List[DocumentSearchResult], query: str, match_types: List[str]):
        """Render compact view of results"""
        st.markdown("### 📋 Compact Results")
        
        # Create summary table
        table_data = []
        for doc_result in results:
            filtered_fragments = [f for f in doc_result.fragments if f.match_type in match_types]
            if filtered_fragments:
                table_data.append({
                    'Document': doc_result.filename,
                    'Type': doc_result.document_type.title(),
                    'Fragments': len(filtered_fragments),
                    'Avg Score': f"{doc_result.total_score:.2f}",
                    'Top Match': filtered_fragments[0].match_type,
                    'Best Fragment': filtered_fragments[0].sentence[:100] + "..." if len(filtered_fragments[0].sentence) > 100 else filtered_fragments[0].sentence
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
            # Detailed view for selected row
            if st.button("📖 View Selected Document Details"):
                selected_doc = st.selectbox("Select Document", [row['Document'] for row in table_data])
                
                # Find and display the selected document
                for doc_result in results:
                    if doc_result.filename == selected_doc:
                        filtered_fragments = [f for f in doc_result.fragments if f.match_type in match_types]
                        
                        st.markdown(f"### 📄 {selected_doc}")
                        for fragment in filtered_fragments:
                            highlighted_text = self._highlight_text(fragment.sentence, fragment.highlights)
                            st.markdown(f"- {highlighted_text} *({fragment.match_type}, {fragment.score:.2f})*")
                        break
        else:
            st.warning("No fragments match the selected criteria.")
    
    def _highlight_text(self, text: str, highlights: List[str]) -> str:
        """Add highlighting to text"""
        if not highlights:
            return text
        
        highlighted = text
        for highlight in highlights:
            if highlight and len(highlight) > 1:
                # Use case-insensitive replacement with markdown bold
                pattern = re.compile(re.escape(highlight), re.IGNORECASE)
                highlighted = pattern.sub(f'**{highlight}**', highlighted)
        
        return highlighted
    
    def _export_results(self, search_data: Dict[str, Any]):
        """Export search results to CSV"""
        results = search_data['results']
        export_data = []
        
        for doc_result in results:
            for fragment in doc_result.fragments:
                export_data.append({
                    'Query': search_data['query'],
                    'Document': doc_result.filename,
                    'Document_Type': doc_result.document_type,
                    'Fragment': fragment.sentence,
                    'Score': fragment.score,
                    'Match_Type': fragment.match_type,
                    'Highlights': ', '.join(fragment.highlights),
                    'Search_Time': search_data['search_time'],
                    'Search_Type': search_data['search_type']
                })
        
        if export_data:
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"search_results_{search_data['query'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data to export.")
    
    def _show_search_tips(self):
        """Show search tips when no results found"""
        st.markdown("### 💡 Search Tips")
        
        st.markdown("""
        **To improve your search results:**
        
        1. **Use different word forms** - The system automatically finds stemmed variations:
           - "recommend" finds "recommendation", "recommends", "recommended"
           - "implement" finds "implementation", "implementing", "implemented"
        
        2. **Try government-specific terms:**
           - Use "accept", "reject", "consider" for responses
           - Use "recommend", "propose", "suggest" for recommendations
           - Use "policy", "framework", "strategy" for policies
        
        3. **Reduce the relevance threshold** in Advanced Options
        
        4. **Use shorter, more focused queries** rather than long sentences
        
        5. **Try the Quick Search buttons** for common government topics
        """)

# Main function to render the search interface
def render_government_search():
    """Main function to render the government search interface"""
    search_ui = GovernmentSearchUI()
    search_ui.render_search_interface()

# For backwards compatibility
def render_search_tab():
    """Alias for backwards compatibility"""
    render_government_search()

# Export functions
__all__ = [
    'GovernmentSearchUI',
    'render_government_search', 
    'render_search_tab'
]
