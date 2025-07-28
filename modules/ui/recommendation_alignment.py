# modules/ui/recommendation_alignment.py - COMPLETELY FIXED VERSION
import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

def render_recommendation_alignment_interface(documents: List[Dict[str, Any]]):
    """FIXED recommendation-response alignment interface"""
    
    st.header("🏛️ Recommendation-Response Alignment")
    st.markdown("*Automatically find recommendations and their corresponding responses*")
    
    if not documents:
        st.warning("📁 Please upload documents first")
        return
    
    # Simple tab structure
    tab_selection = st.radio(
        "Choose alignment mode:",
        ["🔄 Auto Alignment", "🔍 Manual Search"],
        horizontal=True
    )
    
    if tab_selection == "🔄 Auto Alignment":
        render_auto_alignment_fixed(documents)
    else:
        render_manual_search_fixed(documents)

def render_auto_alignment_fixed(documents: List[Dict[str, Any]]):
    """FIXED automatic alignment with proper filtering"""
    
    st.markdown("### 🔄 Automatic Recommendation-Response Alignment")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        rec_patterns = st.multiselect(
            "Recommendation Keywords",
            ["recommend", "suggest", "advise", "propose", "urge", "should", "must"],
            default=["recommend", "suggest", "advise"],
        )
    
    with col2:
        resp_patterns = st.multiselect(
            "Response Keywords", 
            ["accept", "reject", "agree", "disagree", "implement", "consider", "approved", "declined"],
            default=["accept", "reject", "agree", "implement"],
        )
    
    # FIXED: Add minimum confidence threshold
    min_confidence = st.slider(
        "Minimum Alignment Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Lower values show more potential matches but may include false positives"
    )
    
    # Analysis button
    if st.button("🔍 Find & Align Recommendations", type="primary"):
        with st.spinner("🔍 Analyzing documents..."):
            
            try:
                # FIXED: Use improved pattern matching
                recommendations = find_recommendations_improved(documents, rec_patterns)
                responses = find_responses_improved(documents, resp_patterns)
                
                st.info(f"Found {len(recommendations)} potential recommendations and {len(responses)} potential responses")
                
                # FIXED: Use improved alignment logic
                alignments = create_improved_alignments(recommendations, responses, min_confidence)
                
                # FIXED: Filter out self-matches and low confidence
                alignments = filter_valid_alignments(alignments, min_confidence)
                
                display_alignment_results_improved(alignments)
                
            except Exception as e:
                logger.error(f"Alignment analysis error: {e}")
                st.error(f"Analysis error: {str(e)}")
                show_basic_pattern_analysis(documents, rec_patterns, resp_patterns)

def find_recommendations_improved(documents: List[Dict], patterns: List[str]) -> List[Dict]:
    """IMPROVED recommendation finding with better filtering"""
    
    recommendations = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Split into sentences more carefully
        sentences = split_into_proper_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if not is_valid_sentence(sentence):
                continue
            
            sentence_lower = sentence.lower()
            
            # Check for recommendation patterns
            for pattern in patterns:
                if pattern.lower() in sentence_lower:
                    
                    # FIXED: Validate this is actually a recommendation
                    if is_actual_recommendation(sentence, pattern):
                        
                        context = get_enhanced_context(sentences, i, doc)
                        char_position = text.find(sentence) if sentence in text else i * 100
                        
                        recommendation = {
                            'id': f"rec_{len(recommendations) + 1}",
                            'document': doc,
                            'sentence': sentence.strip(),
                            'context': context,
                            'pattern_matched': pattern,
                            'sentence_index': i,
                            'char_position': char_position,
                            'page_number': estimate_page_number_safe(char_position, text),
                            'recommendation_type': classify_recommendation_type_improved(sentence),
                            'confidence_score': calculate_recommendation_confidence(sentence, pattern),
                            'document_section': identify_document_section(sentence, text, char_position)
                        }
                        
                        recommendations.append(recommendation)
                        break  # Only match one pattern per sentence
    
    # FIXED: Sort by confidence and remove duplicates
    recommendations = deduplicate_recommendations(recommendations)
    recommendations.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
    
    return recommendations

def find_responses_improved(documents: List[Dict], patterns: List[str]) -> List[Dict]:
    """IMPROVED response finding with better filtering"""
    
    responses = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        sentences = split_into_proper_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if not is_valid_sentence(sentence):
                continue
            
            sentence_lower = sentence.lower()
            
            # Check for response patterns
            for pattern in patterns:
                if pattern.lower() in sentence_lower:
                    
                    # FIXED: Validate this is actually a response
                    if is_actual_response(sentence, pattern):
                        
                        context = get_enhanced_context(sentences, i, doc)
                        char_position = text.find(sentence) if sentence in text else i * 100
                        
                        response = {
                            'id': f"resp_{len(responses) + 1}",
                            'document': doc,
                            'sentence': sentence.strip(),
                            'context': context,
                            'pattern_matched': pattern,
                            'sentence_index': i,
                            'char_position': char_position,
                            'page_number': estimate_page_number_safe(char_position, text),
                            'response_type': classify_response_type_improved(sentence),
                            'confidence_score': calculate_response_confidence(sentence, pattern),
                            'document_section': identify_document_section(sentence, text, char_position)
                        }
                        
                        responses.append(response)
                        break
    
    # FIXED: Sort by confidence and remove duplicates
    responses = deduplicate_responses(responses)
    responses.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
    
    return responses

def create_improved_alignments(recommendations: List[Dict], responses: List[Dict], 
                             min_confidence: float) -> List[Dict]:
    """IMPROVED alignment with proper cross-document matching"""
    
    alignments = []
    
    for rec in recommendations:
        potential_matches = []
        
        for resp in responses:
            # FIXED: Prevent self-matching (same document, same position)
            if is_self_match(rec, resp):
                continue
            
            # Calculate multiple similarity scores
            semantic_similarity = calculate_enhanced_semantic_similarity(
                rec.get('sentence', ''), 
                resp.get('sentence', '')
            )
            
            contextual_similarity = calculate_contextual_similarity(
                rec.get('context', ''),
                resp.get('context', '')
            )
            
            # FIXED: Consider document relationship
            document_relevance = calculate_document_relevance(rec, resp)
            
            # FIXED: Enhanced combined scoring
            combined_score = (
                semantic_similarity * 0.4 +
                contextual_similarity * 0.3 +
                document_relevance * 0.3
            )
            
            # Only consider matches above minimum threshold
            if combined_score >= min_confidence:
                potential_matches.append({
                    'response': resp,
                    'semantic_similarity': semantic_similarity,
                    'contextual_similarity': contextual_similarity,
                    'document_relevance': document_relevance,
                    'combined_score': combined_score
                })
        
        # Sort by combined score
        potential_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Create alignment
        alignment = {
            'recommendation': rec,
            'responses': potential_matches[:3],  # Top 3 matches
            'alignment_confidence': potential_matches[0]['combined_score'] if potential_matches else 0,
            'alignment_status': determine_alignment_status_improved(potential_matches),
            'cross_document': is_cross_document_alignment(rec, potential_matches)
        }
        
        alignments.append(alignment)
    
    return alignments

def filter_valid_alignments(alignments: List[Dict], min_confidence: float) -> List[Dict]:
    """FIXED: Filter out invalid alignments"""
    
    valid_alignments = []
    
    for alignment in alignments:
        confidence = alignment.get('alignment_confidence', 0)
        
        # Filter by minimum confidence
        if confidence < min_confidence:
            continue
        
        # FIXED: Check for self-matches and invalid patterns
        rec = alignment.get('recommendation', {})
        responses = alignment.get('responses', [])
        
        # Skip if no valid responses
        if not responses:
            continue
        
        # FIXED: Validate alignment quality
        if is_valid_alignment(rec, responses):
            valid_alignments.append(alignment)
    
    return valid_alignments

def display_alignment_results_improved(alignments: List[Dict]):
    """IMPROVED display with better error detection"""
    
    if not alignments:
        st.warning("❌ No valid recommendation-response alignments found")
        st.info("""
        **Possible reasons:**
        - Documents contain narrative text rather than formal recommendations
        - Recommendations and responses may use different terminology
        - Minimum confidence threshold may be too high
        - Documents may not contain matching recommendation-response pairs
        """)
        return
    
    # Enhanced summary statistics
    st.markdown("### 📊 Alignment Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Valid Alignments Found", len(alignments))
    
    with col2:
        cross_doc_count = sum(1 for a in alignments if a.get('cross_document', False))
        st.metric("Cross-Document Alignments", cross_doc_count)
    
    with col3:
        avg_confidence = sum(a.get('alignment_confidence', 0) for a in alignments) / len(alignments)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        high_confidence = sum(1 for a in alignments if a.get('alignment_confidence', 0) > 0.7)
        st.metric("High Confidence (>0.7)", high_confidence)
    
    # Display individual alignments with improved formatting
    st.markdown("### 🔗 Recommendation-Response Alignments")
    
    for i, alignment in enumerate(alignments, 1):
        display_single_alignment_improved(alignment, i)

def display_single_alignment_improved(alignment: Dict, index: int):
    """IMPROVED single alignment display with validation info"""
    
    rec = alignment.get('recommendation', {})
    responses = alignment.get('responses', [])
    confidence = alignment.get('alignment_confidence', 0)
    is_cross_doc = alignment.get('cross_document', False)
    
    # Enhanced confidence indicator
    if confidence > 0.8:
        confidence_color = "🟢"
        confidence_text = "Very High"
    elif confidence > 0.6:
        confidence_color = "🟡" 
        confidence_text = "High"
    elif confidence > 0.4:
        confidence_color = "🟠"
        confidence_text = "Medium"
    else:
        confidence_color = "🔴"
        confidence_text = "Low"
    
    rec_type = rec.get('recommendation_type', 'General')
    cross_doc_indicator = " 📄↔️📄" if is_cross_doc else ""
    
    with st.expander(f"{confidence_color} Alignment {index} - {rec_type} - {confidence_text} Confidence ({confidence:.2f}){cross_doc_indicator}", 
                    expanded=index <= 2):
        
        # FIXED: Show document information clearly
        rec_doc_name = rec.get('document', {}).get('filename', 'Unknown Document')
        rec_page = rec.get('page_number', 1)
        rec_section = rec.get('document_section', 'Unknown Section')
        
        st.markdown(f"""
        **🎯 RECOMMENDATION**
        
        📄 **Source:** {rec_doc_name} | **Page:** {rec_page} | **Section:** {rec_section}
        """)
        
        # Display recommendation with highlighting
        rec_sentence = rec.get('sentence', 'No sentence available')
        st.markdown(f"> {rec_sentence}")
        
        # Show recommendation context if different
        rec_context = rec.get('context', '')
        if rec_context and rec_context != rec_sentence:
            with st.expander("📖 Full Recommendation Context"):
                st.markdown(rec_context)
        
        # Display responses
        if responses:
            st.markdown("**↩️ MATCHED RESPONSES**")
            
            for j, resp_match in enumerate(responses, 1):
                resp = resp_match.get('response', {})
                match_score = resp_match.get('combined_score', 0)
                
                resp_doc_name = resp.get('document', {}).get('filename', 'Unknown Document')
                resp_page = resp.get('page_number', 1)
                resp_section = resp.get('document_section', 'Unknown Section')
                
                # FIXED: Show detailed matching information
                semantic_sim = resp_match.get('semantic_similarity', 0)
                contextual_sim = resp_match.get('contextual_similarity', 0)
                doc_relevance = resp_match.get('document_relevance', 0)
                
                st.markdown(f"""
                **Response {j} - Match Score: {match_score:.2f}**
                
                📄 **Source:** {resp_doc_name} | **Page:** {resp_page} | **Section:** {resp_section}
                
                📊 **Similarity Breakdown:**
                - Semantic: {semantic_sim:.2f} | Contextual: {contextual_sim:.2f} | Document: {doc_relevance:.2f}
                """)
                
                # Display response
                resp_sentence = resp.get('sentence', 'No sentence available')
                st.markdown(f"> {resp_sentence}")
                
                # Show response context if different
                resp_context = resp.get('context', '')
                if resp_context and resp_context != resp_sentence:
                    with st.expander(f"📖 Full Response {j} Context"):
                        st.markdown(resp_context)
                
                if j < len(responses):
                    st.markdown("---")
        else:
            st.warning("❌ No matching responses found")

# HELPER FUNCTIONS WITH IMPROVED LOGIC

def split_into_proper_sentences(text: str) -> List[str]:
    """Split text into proper sentences with better handling"""
    
    # Use regex to split on sentence endings but preserve structure
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean and filter sentences
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Must be reasonable length and contain actual content
        if len(sentence) > 20 and len(sentence.split()) > 4:
            # Remove sentences that are clearly navigation or metadata
            if not is_metadata_sentence(sentence):
                clean_sentences.append(sentence)
    
    return clean_sentences

def is_valid_sentence(sentence: str) -> bool:
    """Check if sentence is valid for analysis"""
    
    if not sentence or len(sentence.strip()) < 20:
        return False
    
    # Check for minimum word count
    words = sentence.split()
    if len(words) < 5:
        return False
    
    # Filter out obviously bad sentences
    bad_patterns = [
        r'^\d+\s*$',  # Just numbers
        r'^page \d+',  # Page numbers
        r'^figure \d+',  # Figure references
        r'^table \d+',  # Table references
        r'^\w+\s*:\s*$',  # Just headers with colons
    ]
    
    sentence_lower = sentence.lower()
    for pattern in bad_patterns:
        if re.match(pattern, sentence_lower):
            return False
    
    return True

def is_metadata_sentence(sentence: str) -> bool:
    """Check if sentence is metadata/navigation rather than content"""
    
    sentence_lower = sentence.lower().strip()
    
    metadata_indicators = [
        'page ', 'figure ', 'table ', 'appendix ', 'chapter ',
        'see section', 'see page', 'see appendix',
        'written statement of', 'witness statement',
        'witn', 'para ', 'paras ',  # Common in legal documents
        'infected blood inquiry', 'the report',  # Document titles
        'commentary on the government response'  # Section headers
    ]
    
    for indicator in metadata_indicators:
        if indicator in sentence_lower:
            return True
    
    # Check for patterns like "1505 Professor Marc Turner"
    if re.match(r'^\d{3,5}\s+[A-Z]', sentence):
        return True
    
    return False

def is_actual_recommendation(sentence: str, pattern: str) -> bool:
    """Validate this is actually a recommendation, not just narrative"""
    
    sentence_lower = sentence.lower()
    
    # Look for recommendation structure patterns
    recommendation_structures = [
        f"we {pattern}",
        f"i {pattern}",
        f"the committee {pattern}",
        f"the report {pattern}",
        f"it is {pattern}ed",
        f"{pattern}ation:",  # "recommendation:"
        f"{pattern} that",   # "recommend that"
    ]
    
    # Must have at least one proper recommendation structure
    has_structure = any(structure in sentence_lower for structure in recommendation_structures)
    
    if not has_structure:
        return False
    
    # FIXED: Filter out narrative descriptions
    narrative_indicators = [
        'was advised', 'were advised', 'the meeting was advised',
        'they were told', 'he was told', 'she was told',
        'it was suggested to', 'it was recommended to them',
        'the directors were advised', 'members were advised'
    ]
    
    # If it's clearly narrative, not a recommendation
    for indicator in narrative_indicators:
        if indicator in sentence_lower:
            return False
    
    return True

def is_actual_response(sentence: str, pattern: str) -> bool:
    """Validate this is actually a response, not just narrative"""
    
    sentence_lower = sentence.lower()
    
    # Look for response structure patterns
    response_structures = [
        f"we {pattern}",
        f"the government {pattern}s",
        f"the department {pattern}s",
        f"the ministry {pattern}s",
        f"in response",
        f"our response",
        f"we will {pattern}",
        f"this {pattern}s the",
    ]
    
    # Must have at least one proper response structure
    has_structure = any(structure in sentence_lower for structure in response_structures)
    
    if not has_structure:
        return False
    
    # Filter out the same narrative indicators as recommendations
    narrative_indicators = [
        'was advised', 'were advised', 'the meeting was advised',
        'they were told', 'he was told', 'she was told'
    ]
    
    for indicator in narrative_indicators:
        if indicator in sentence_lower:
            return False
    
    return True

def is_self_match(rec: Dict, resp: Dict) -> bool:
    """FIXED: Detect self-matches to prevent false positives"""
    
    # Same document and similar position
    rec_doc = rec.get('document', {}).get('filename', '')
    resp_doc = resp.get('document', {}).get('filename', '')
    
    if rec_doc == resp_doc:
        rec_pos = rec.get('char_position', 0)
        resp_pos = resp.get('char_position', 0)
        
        # If positions are very close (within 500 characters), likely same content
        if abs(rec_pos - resp_pos) < 500:
            return True
    
    # Check for identical or very similar text
    rec_text = rec.get('sentence', '').strip()
    resp_text = resp.get('sentence', '').strip()
    
    if not rec_text or not resp_text:
        return False
    
    # Identical text
    if rec_text == resp_text:
        return True
    
    # Very similar text (>95% overlap)
    similarity = calculate_text_similarity(rec_text, resp_text)
    if similarity > 0.95:
        return True
    
    return False

def calculate_enhanced_semantic_similarity(text1: str, text2: str) -> float:
    """Enhanced semantic similarity calculation"""
    
    if not text1 or not text2:
        return 0.0
    
    # Tokenize and clean
    words1 = set(word.lower() for word in re.findall(r'\b\w+\b', text1) if len(word) > 2)
    words2 = set(word.lower() for word in re.findall(r'\b\w+\b', text2) if len(word) > 2)
    
    if not words1 or not words2:
        return 0.0
    
    # Basic Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    jaccard = intersection / union if union > 0 else 0.0
    
    # Boost for government-specific term matches
    gov_terms = {'government', 'department', 'ministry', 'policy', 'implement', 'recommend', 'accept', 'reject'}
    gov_matches = len((words1 & words2) & gov_terms)
    gov_boost = min(gov_matches * 0.1, 0.3)  # Up to 30% boost
    
    return min(jaccard + gov_boost, 1.0)

def calculate_contextual_similarity(context1: str, context2: str) -> float:
    """Calculate similarity between contexts"""
    
    if not context1 or not context2:
        return 0.0
    
    # Similar to semantic similarity but for longer contexts
    return calculate_enhanced_semantic_similarity(context1, context2)

def calculate_document_relevance(rec: Dict, resp: Dict) -> float:
    """Calculate how relevant documents are to each other"""
    
    rec_doc = rec.get('document', {}).get('filename', '').lower()
    resp_doc = resp.get('document', {}).get('filename', '').lower()
    
    # Same document gets lower score (prefer cross-document matches)
    if rec_doc == resp_doc:
        return 0.3
    
    # Look for document relationship indicators
    if 'response' in resp_doc and 'response' not in rec_doc:
        return 1.0  # Perfect: recommendation doc -> response doc
    
    if 'government' in resp_doc and 'committee' in rec_doc:
        return 0.9  # Very good: committee -> government
    
    if 'volume' in rec_doc and 'volume' in resp_doc:
        return 0.7  # Good: related volumes
    
    # Default for different documents
    return 0.6

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity for duplicate detection"""
    
    if not text1 or not text2:
        return 0.0
    
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    if not words1 or not words2:
        return 0.0
    
    # Simple word overlap ratio
    common_words = len(set(words1) & set(words2))
    total_words = len(set(words1) | set(words2))
    
    return common_words / total_words if total_words > 0 else 0.0

def calculate_recommendation_confidence(sentence: str, pattern: str) -> float:
    """Calculate confidence that this is a real recommendation"""
    
    sentence_lower = sentence.lower()
    
    base_score = 0.5
    
    # Boost for strong recommendation indicators
    strong_indicators = [
        f'we {pattern}', f'i {pattern}', f'committee {pattern}s', 
        f'{pattern}ation ', f'{pattern} that'
    ]
    
    for indicator in strong_indicators:
        if indicator in sentence_lower:
            base_score += 0.2
    
    # Boost for formal language
    formal_terms = ['committee', 'report', 'inquiry', 'government', 'department']
    formal_count = sum(1 for term in formal_terms if term in sentence_lower)
    base_score += min(formal_count * 0.1, 0.3)
    
    # Penalty for narrative indicators
    narrative_terms = ['was advised', 'were told', 'meeting was']
    narrative_count = sum(1 for term in narrative_terms if term in sentence_lower)
    base_score -= min(narrative_count * 0.3, 0.5)
    
    return max(0.1, min(base_score, 1.0))

def calculate_response_confidence(sentence: str, pattern: str) -> float:
    """Calculate confidence that this is a real response"""
    
    sentence_lower = sentence.lower()
    
    base_score = 0.5
    
    # Boost for strong response indicators
    strong_indicators = [
        f'government {pattern}s', f'department {pattern}s', f'we {pattern}',
        'in response', 'our response', f'we will {pattern}'
    ]
    
    for indicator in strong_indicators:
        if indicator in sentence_lower:
            base_score += 0.2
    
    # Similar formal language boost as recommendations
    formal_terms = ['government', 'department', 'ministry', 'policy', 'implementation']
    formal_count = sum(1 for term in formal_terms if term in sentence_lower)
    base_score += min(formal_count * 0.1, 0.3)
    
    # Same narrative penalty
    narrative_terms = ['was advised', 'were told', 'meeting was']
    narrative_count = sum(1 for term in narrative_terms if term in sentence_lower)
    base_score -= min(narrative_count * 0.3, 0.5)
    
    return max(0.1, min(base_score, 1.0))

def deduplicate_recommendations(recommendations: List[Dict]) -> List[Dict]:
    """Remove duplicate recommendations"""
    
    unique_recs = []
    seen_texts = set()
    
    for rec in recommendations:
        text = rec.get('sentence', '').strip().lower()
        
        # Skip if we've seen very similar text
        is_duplicate = False
        for seen_text in seen_texts:
            if calculate_text_similarity(text, seen_text) > 0.9:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_recs.append(rec)
            seen_texts.add(text)
    
    return unique_recs

def deduplicate_responses(responses: List[Dict]) -> List[Dict]:
    """Remove duplicate responses"""
    return deduplicate_recommendations(responses)  # Same logic

def get_enhanced_context(sentences: List[str], index: int, doc: Dict) -> str:
    """Get enhanced context with document info"""
    
    # Get surrounding sentences
    start = max(0, index - 2)
    end = min(len(sentences), index + 3)
    
    context_sentences = sentences[start:end]
    context = ' '.join(s.strip() for s in context_sentences if s.strip())
    
    return context

def identify_document_section(sentence: str, full_text: str, position: int) -> str:
    """Identify which section of document this sentence is from"""
    
    # Look for section headers before this position
    text_before = full_text[:position]
    
    # Common section patterns
    section_patterns = [
        r'(chapter \d+)',
        r'(section \d+)',
        r'(part \d+)',
        r'(recommendations?)',
        r'(government response)',
        r'(executive summary)',
        r'(conclusions?)',
        r'(findings?)',
    ]
    
    for pattern in section_patterns:
        matches = list(re.finditer(pattern, text_before, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).title()
    
    return "Main Content"

def estimate_page_number_safe(char_position: int, text: str) -> int:
    """Safely estimate page number"""
    try:
        if char_position <= 0:
            return 1
        return max(1, char_position // 2000 + 1)
    except Exception:
        return 1

def classify_recommendation_type_improved(sentence: str) -> str:
    """Improved recommendation classification"""
    
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['urgent', 'immediate', 'critical', 'emergency']):
        return 'Urgent'
    elif any(word in sentence_lower for word in ['policy', 'regulation', 'framework', 'legislation']):
        return 'Policy'
    elif any(word in sentence_lower for word in ['implement', 'establish', 'create', 'develop']):
        return 'Implementation'
    elif any(word in sentence_lower for word in ['consider', 'review', 'explore', 'examine']):
        return 'Consideration'
    elif any(word in sentence_lower for word in ['financial', 'budget', 'funding', 'cost']):
        return 'Financial'
    else:
        return 'General'

def classify_response_type_improved(sentence: str) -> str:
    """Improved response classification"""
    
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['accept', 'agree', 'approve', 'endorse']):
        return 'Acceptance'
    elif any(word in sentence_lower for word in ['reject', 'decline', 'disagree', 'oppose']):
        return 'Rejection'
    elif any(word in sentence_lower for word in ['consider', 'review', 'evaluate', 'assess']):
        return 'Under Review'
    elif any(word in sentence_lower for word in ['implement', 'will implement', 'implementing']):
        return 'Implementation'
    else:
        return 'General Response'

def determine_alignment_status_improved(matches: List[Dict]) -> str:
    """Improved alignment status determination"""
    
    if not matches:
        return "No Response Found"
    
    best_score = matches[0].get('combined_score', 0)
    
    if best_score > 0.8:
        return "Strong Alignment"
    elif best_score > 0.6:
        return "Good Alignment"
    elif best_score > 0.4:
        return "Moderate Alignment"
    else:
        return "Weak Alignment"

def is_cross_document_alignment(rec: Dict, responses: List[Dict]) -> bool:
    """Check if this is a cross-document alignment"""
    
    if not responses:
        return False
    
    rec_doc = rec.get('document', {}).get('filename', '')
    
    for resp_match in responses:
        resp = resp_match.get('response', {})
        resp_doc = resp.get('document', {}).get('filename', '')
        
        if rec_doc != resp_doc:
            return True
    
    return False

def is_valid_alignment(rec: Dict, responses: List[Dict]) -> bool:
    """Validate that this alignment makes sense"""
    
    if not responses:
        return False
    
    # Check recommendation quality
    rec_confidence = rec.get('confidence_score', 0)
    if rec_confidence < 0.3:
        return False
    
    # Check best response quality
    best_response = responses[0].get('response', {})
    resp_confidence = best_response.get('confidence_score', 0)
    if resp_confidence < 0.3:
        return False
    
    # Check overall alignment score
    best_match_score = responses[0].get('combined_score', 0)
    if best_match_score < 0.4:
        return False
    
    return True

def render_manual_search_fixed(documents: List[Dict[str, Any]]):
    """FIXED manual search with better validation"""
    
    st.markdown("### 🔍 Manual Sentence Search")
    st.markdown("*Find similar recommendations or responses by pasting a sentence*")
    
    # Text input for manual search
    search_sentence = st.text_area(
        "📝 Paste your sentence here:",
        placeholder="e.g., 'The committee recommends that the Department implement new protocols'",
        help="Paste any sentence to find similar recommendations or responses",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_type = st.selectbox(
            "What are you looking for?",
            ["Similar content", "Recommendations only", "Responses only"]
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower values find more matches but may be less relevant"
        )
    
    with col2:
        max_matches = st.selectbox(
            "Maximum matches to show",
            [5, 10, 20, 50],
            index=1
        )
        
        show_scores = st.checkbox("Show similarity scores", value=True)
    
    # Search execution
    if st.button("🔎 Find Similar Content", type="primary") and search_sentence.strip():
        
        start_time = time.time()
        
        with st.spinner("🔍 Searching for similar content..."):
            
            try:
                # Use improved search
                matches = find_similar_content_improved(
                    documents=documents,
                    target_sentence=search_sentence,
                    search_type=search_type,
                    similarity_threshold=similarity_threshold,
                    max_matches=max_matches
                )
                
                search_time = time.time() - start_time
                
                display_manual_search_results_improved(
                    matches=matches,
                    target_sentence=search_sentence,
                    search_time=search_time,
                    show_scores=show_scores,
                    search_type=search_type
                )
                
            except Exception as e:
                logger.error(f"Manual search error: {e}")
                st.error(f"Search error: {str(e)}")

def find_similar_content_improved(documents: List[Dict], target_sentence: str, 
                                search_type: str, similarity_threshold: float, 
                                max_matches: int) -> List[Dict]:
    """IMPROVED similar content search with better filtering"""
    
    matches = []
    
    # Clean and validate target sentence
    target_sentence = target_sentence.strip()
    if not target_sentence or len(target_sentence) < 10:
        return matches
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        sentences = split_into_proper_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if not is_valid_sentence(sentence):
                continue
            
            # Skip sentences that are too similar to target (avoid self-matches)
            if calculate_text_similarity(target_sentence, sentence) > 0.95:
                continue
            
            # Calculate enhanced similarity
            similarity = calculate_enhanced_semantic_similarity(target_sentence, sentence)
            
            if similarity >= similarity_threshold:
                
                # Filter by search type if specified
                if search_type == "Recommendations only":
                    if not contains_recommendation_pattern(sentence):
                        continue
                elif search_type == "Responses only":
                    if not contains_response_pattern(sentence):
                        continue
                
                # Get enhanced context
                context = get_enhanced_context(sentences, i, doc)
                char_position = text.find(sentence) if sentence in text else i * 100
                
                match = {
                    'sentence': sentence.strip(),
                    'context': context,
                    'similarity_score': similarity,
                    'document': doc,
                    'position': char_position,
                    'page_number': estimate_page_number_safe(char_position, text),
                    'content_type': classify_content_type_improved(sentence),
                    'document_section': identify_document_section(sentence, text, char_position),
                    'is_cross_document': doc.get('filename', '') != 'target_document'
                }
                
                matches.append(match)
    
    # Sort by similarity score and limit results
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matches[:max_matches]

def contains_recommendation_pattern(sentence: str) -> bool:
    """Check if sentence contains recommendation patterns"""
    
    sentence_lower = sentence.lower()
    
    patterns = [
        'recommend', 'suggest', 'advise', 'propose', 'urge',
        'should', 'must', 'ought to', 'it is recommended'
    ]
    
    return any(pattern in sentence_lower for pattern in patterns)

def contains_response_pattern(sentence: str) -> bool:
    """Check if sentence contains response patterns"""
    
    sentence_lower = sentence.lower()
    
    patterns = [
        'accept', 'reject', 'agree', 'disagree', 'implement',
        'consider', 'approved', 'declined', 'response', 'reply',
        'government', 'department', 'ministry'
    ]
    
    return any(pattern in sentence_lower for pattern in patterns)

def classify_content_type_improved(sentence: str) -> str:
    """Improved content type classification"""
    
    sentence_lower = sentence.lower()
    
    # More specific classification
    if contains_recommendation_pattern(sentence):
        if any(word in sentence_lower for word in ['urgent', 'immediate', 'critical']):
            return 'Urgent Recommendation'
        elif any(word in sentence_lower for word in ['policy', 'framework']):
            return 'Policy Recommendation'
        else:
            return 'Recommendation'
    
    elif contains_response_pattern(sentence):
        if any(word in sentence_lower for word in ['accept', 'agree', 'approve']):
            return 'Positive Response'
        elif any(word in sentence_lower for word in ['reject', 'decline', 'oppose']):
            return 'Negative Response'
        else:
            return 'Response'
    
    elif any(word in sentence_lower for word in ['finding', 'conclusion', 'result']):
        return 'Finding'
    
    elif any(word in sentence_lower for word in ['background', 'context', 'history']):
        return 'Background'
    
    else:
        return 'General Content'

def display_manual_search_results_improved(matches: List[Dict], target_sentence: str, 
                                         search_time: float, show_scores: bool, 
                                         search_type: str):
    """IMPROVED display for manual search results"""
    
    if not matches:
        st.warning(f"No matches found for your sentence in {search_type.lower()}")
        st.info("""
        **Try:**
        - Lowering the similarity threshold
        - Changing the search type to 'Similar content'
        - Using different keywords from your sentence
        """)
        return
    
    # Enhanced summary
    st.success(f"🎯 Found **{len(matches)}** matches in **{search_time:.3f}** seconds")
    
    # Show target sentence
    st.markdown("### 📝 Your Target Sentence")
    st.info(f"*{target_sentence}*")
    
    # Group by document for better organization
    doc_groups = {}
    for match in matches:
        doc_name = match['document']['filename']
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(match)
    
    st.markdown(f"### 🔍 Similar Content Found ({len(doc_groups)} documents)")
    
    for doc_name, doc_matches in doc_groups.items():
        
        with st.expander(f"📄 {doc_name} ({len(doc_matches)} matches)", expanded=len(doc_groups) <= 3):
            
            for i, match in enumerate(doc_matches, 1):
                
                similarity = match['similarity_score']
                content_type = match['content_type']
                page_num = match['page_number']
                section = match['document_section']
                
                # Confidence indicator
                if similarity > 0.8:
                    confidence_icon = "🟢"
                    confidence_text = "Very High"
                elif similarity > 0.6:
                    confidence_icon = "🟡"
                    confidence_text = "High"
                elif similarity > 0.4:
                    confidence_icon = "🟠"
                    confidence_text = "Medium"
                else:
                    confidence_icon = "🔴"
                    confidence_text = "Low"
                
                score_display = f" ({similarity:.3f})" if show_scores else ""
                
                st.markdown(f"""
                **{confidence_icon} Match {i} - {content_type} - {confidence_text} Similarity{score_display}**
                
                📍 **Location:** Page {page_num} | Section: {section}
                """)
                
                # Display matched sentence
                sentence = match['sentence']
                st.markdown(f"> {sentence}")
                
                # Show context if different
                context = match.get('context', '')
                if context and context != sentence and len(context) > len(sentence) + 50:
                    with st.expander(f"📖 Full Context for Match {i}"):
                        st.markdown(context)
                
                if i < len(doc_matches):
                    st.markdown("---")

def show_basic_pattern_analysis(documents: List[Dict], rec_patterns: List[str], resp_patterns: List[str]):
    """Show basic pattern analysis when alignment fails"""
    
    st.markdown("### 📊 Basic Pattern Analysis")
    st.info("Showing simple keyword counts since full alignment had issues")
    
    total_rec = 0
    total_resp = 0
    doc_analysis = []
    
    for doc in documents:
        text = doc.get('text', '').lower()
        filename = doc.get('filename', 'Unknown')
        
        doc_rec_count = sum(text.count(pattern.lower()) for pattern in rec_patterns)
        doc_resp_count = sum(text.count(pattern.lower()) for pattern in resp_patterns)
        
        total_rec += doc_rec_count
        total_resp += doc_resp_count
        
        if doc_rec_count > 0 or doc_resp_count > 0:
            doc_analysis.append({
                'Document': filename,
                'Recommendation Keywords': doc_rec_count,
                'Response Keywords': doc_resp_count
            })
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Recommendation Keywords", total_rec)
    with col2:
        st.metric("Total Response Keywords", total_resp)
    with col3:
        st.metric("Documents with Keywords", len(doc_analysis))
    
    # Document breakdown
    if doc_analysis:
        st.markdown("### 📄 Keyword Distribution by Document")
        df = pd.DataFrame(doc_analysis)
        st.dataframe(df, use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Keyword Analysis",
            data=csv,
            file_name=f"keyword_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No recommendation or response keywords found in any documents")
        
        st.markdown("""
        **Possible issues:**
        - Documents may use different terminology
        - Content may be narrative rather than formal recommendations/responses
        - Try different keyword patterns
        """)

# Export the main function
__all__ = ['render_recommendation_alignment_interface']
