# modules/ui/recommendation_alignment_extraction.py
"""
🎯 Pattern Extraction Engine for Recommendation-Response Alignment
Handles advanced pattern recognition and content extraction with military-grade accuracy
"""

import re
import logging
from typing import Dict, List, Any
from .recommendation_alignment_core import (
    ContentItem, AlignmentConfig, split_sentences_with_intelligence,
    passes_basic_validation, is_narrative_content, calculate_character_position,
    estimate_page_number_intelligent
)

logger = logging.getLogger(__name__)

# =============================================================================
# ADVANCED PATTERN EXTRACTION ENGINE
# =============================================================================

def extract_recommendations_ultimate(documents: List[Dict[str, Any]], narrative_filtering: str) -> List[ContentItem]:
    """🎯 ULTIMATE recommendation extraction with enterprise-grade accuracy"""
    
    recommendations = []
    processing_stats = {'processed': 0, 'excluded': 0, 'narrative_filtered': 0}
    
    for doc_idx, doc in enumerate(documents):
        text = doc.get('text', '')
        if not text:
            continue
        
        # Enhanced sentence splitting with government document intelligence
        sentences = split_sentences_with_intelligence(text)
        
        for sent_idx, sentence in enumerate(sentences):
            processing_stats['processed'] += 1
            
            # Multi-layered validation
            if not passes_basic_validation(sentence):
                processing_stats['excluded'] += 1
                continue
            
            # 🛡️ CRITICAL: Narrative filtering based on configuration
            if narrative_filtering != "Disabled" and is_narrative_content(sentence, narrative_filtering):
                processing_stats['narrative_filtered'] += 1
                continue
            
            # Pattern matching with confidence assessment
            pattern_matches = find_recommendation_patterns_advanced(sentence)
            
            if pattern_matches:
                # Create ContentItem with comprehensive metadata
                content_item = create_recommendation_content_item(
                    doc_idx, sent_idx, doc, sentence, sentences, pattern_matches
                )
                
                # Apply quality threshold
                if content_item.confidence_score >= 0.25:  # Minimum quality threshold
                    recommendations.append(content_item)
    
    # Advanced deduplication with similarity analysis
    deduplicated_recommendations = deduplicate_content_items_advanced(recommendations)
    
    logger.info(f"Recommendation extraction: {len(deduplicated_recommendations)} final recommendations from {processing_stats['processed']} sentences")
    
    return deduplicated_recommendations

def extract_responses_ultimate(documents: List[Dict[str, Any]], narrative_filtering: str) -> List[ContentItem]:
    """↩️ ULTIMATE response extraction with government intelligence"""
    
    responses = []
    processing_stats = {'processed': 0, 'excluded': 0, 'narrative_filtered': 0}
    
    for doc_idx, doc in enumerate(documents):
        text = doc.get('text', '')
        if not text:
            continue
        
        sentences = split_sentences_with_intelligence(text)
        
        for sent_idx, sentence in enumerate(sentences):
            processing_stats['processed'] += 1
            
            if not passes_basic_validation(sentence):
                processing_stats['excluded'] += 1
                continue
            
            if narrative_filtering != "Disabled" and is_narrative_content(sentence, narrative_filtering):
                processing_stats['narrative_filtered'] += 1
                continue
            
            # Response-specific pattern matching
            pattern_matches = find_response_patterns_advanced(sentence)
            
            if pattern_matches:
                content_item = create_response_content_item(
                    doc_idx, sent_idx, doc, sentence, sentences, pattern_matches
                )
                
                if content_item.confidence_score >= 0.25:
                    responses.append(content_item)
    
    deduplicated_responses = deduplicate_content_items_advanced(responses)
    
    logger.info(f"Response extraction: {len(deduplicated_responses)} final responses from {processing_stats['processed']} sentences")
    
    return deduplicated_responses

# =============================================================================
# ADVANCED PATTERN MATCHING FUNCTIONS
# =============================================================================

def find_recommendation_patterns_advanced(sentence: str) -> List[Dict[str, Any]]:
    """🎯 Advanced recommendation pattern matching with confidence scoring"""
    
    matches = []
    sentence_lower = sentence.lower()
    
    # Check strong patterns first
    for pattern in AlignmentConfig.RECOMMENDATION_PATTERNS['strong']:
        if pattern in sentence_lower:
            position = sentence_lower.find(pattern)
            matches.append({
                'pattern': pattern,
                'type': 'strong_recommendation',
                'position': position,
                'confidence': 0.9,
                'context': extract_pattern_context(sentence, position, len(pattern))
            })
    
    # Check moderate patterns
    for pattern in AlignmentConfig.RECOMMENDATION_PATTERNS['moderate']:
        if pattern in sentence_lower:
            position = sentence_lower.find(pattern)
            matches.append({
                'pattern': pattern,
                'type': 'moderate_recommendation',
                'position': position,
                'confidence': 0.7,
                'context': extract_pattern_context(sentence, position, len(pattern))
            })
    
    # Check weak patterns (only if no strong/moderate found)
    if not matches:
        for pattern in AlignmentConfig.RECOMMENDATION_PATTERNS['weak']:
            if pattern in sentence_lower:
                position = sentence_lower.find(pattern)
                matches.append({
                    'pattern': pattern,
                    'type': 'weak_recommendation',
                    'position': position,
                    'confidence': 0.5,
                    'context': extract_pattern_context(sentence, position, len(pattern))
                })
    
    # Additional validation for recommendation structure
    validated_matches = []
    for match in matches:
        if validate_recommendation_structure(sentence, match):
            validated_matches.append(match)
    
    return validated_matches

def find_response_patterns_advanced(sentence: str) -> List[Dict[str, Any]]:
    """↩️ Advanced response pattern matching with government intelligence"""
    
    matches = []
    sentence_lower = sentence.lower()
    
    # Check strong response patterns
    for pattern in AlignmentConfig.RESPONSE_PATTERNS['strong']:
        if pattern in sentence_lower:
            position = sentence_lower.find(pattern)
            matches.append({
                'pattern': pattern,
                'type': 'strong_response',
                'position': position,
                'confidence': 0.9,
                'context': extract_pattern_context(sentence, position, len(pattern))
            })
    
    # Check moderate patterns
    for pattern in AlignmentConfig.RESPONSE_PATTERNS['moderate']:
        if pattern in sentence_lower:
            position = sentence_lower.find(pattern)
            matches.append({
                'pattern': pattern,
                'type': 'moderate_response',
                'position': position,
                'confidence': 0.7,
                'context': extract_pattern_context(sentence, position, len(pattern))
            })
    
    # Check weak patterns
    if not matches:
        for pattern in AlignmentConfig.RESPONSE_PATTERNS['weak']:
            if pattern in sentence_lower:
                position = sentence_lower.find(pattern)
                matches.append({
                    'pattern': pattern,
                    'type': 'weak_response',
                    'position': position,
                    'confidence': 0.5,
                    'context': extract_pattern_context(sentence, position, len(pattern))
                })
    
    # Validate response structure
    validated_matches = []
    for match in matches:
        if validate_response_structure(sentence, match):
            validated_matches.append(match)
    
    return validated_matches

def extract_pattern_context(sentence: str, position: int, pattern_length: int) -> str:
    """Extract context around pattern match"""
    
    start = max(0, position - 20)
    end = min(len(sentence), position + pattern_length + 20)
    
    return sentence[start:end]

def validate_recommendation_structure(sentence: str, pattern_match: Dict[str, Any]) -> bool:
    """✅ Validate that sentence has proper recommendation structure"""
    
    sentence_lower = sentence.lower()
    
    # Must have action-oriented language
    action_indicators = [
        'implement', 'establish', 'create', 'develop', 'ensure',
        'provide', 'improve', 'enhance', 'strengthen', 'introduce',
        'adopt', 'pursue', 'consider', 'review', 'assess'
    ]
    
    has_action = any(indicator in sentence_lower for indicator in action_indicators)
    
    # Must not be pure narrative
    narrative_indicators = [
        'was told', 'were told', 'explained that', 'mentioned that',
        'described how', 'indicated that', 'stated that'
    ]
    
    is_narrative = any(indicator in sentence_lower for indicator in narrative_indicators)
    
    # Must have reasonable sentence structure
    has_structure = len(sentence.split()) >= 8 and '.' not in sentence[:50]
    
    return has_action and not is_narrative and has_structure

def validate_response_structure(sentence: str, pattern_match: Dict[str, Any]) -> bool:
    """✅ Validate that sentence has proper response structure"""
    
    sentence_lower = sentence.lower()
    
    # Should have government/official language
    official_indicators = [
        'government', 'department', 'ministry', 'official',
        'policy', 'implementation', 'consideration'
    ]
    
    has_official_language = any(indicator in sentence_lower for indicator in official_indicators)
    
    # Should have response indicators
    response_indicators = [
        'accept', 'reject', 'agree', 'disagree', 'implement',
        'consider', 'review', 'response', 'position'
    ]
    
    has_response_language = any(indicator in sentence_lower for indicator in response_indicators)
    
    # Must not be pure narrative
    narrative_indicators = [
        'was told', 'were told', 'meeting was', 'directors were'
    ]
    
    is_narrative = any(indicator in sentence_lower for indicator in narrative_indicators)
    
    # Must have reasonable sentence structure  
    has_structure = len(sentence.split()) >= 6
    
    return (has_official_language or has_response_language) and not is_narrative and has_structure

# =============================================================================
# CONTENT ITEM CREATION FUNCTIONS
# =============================================================================

def create_recommendation_content_item(doc_idx: int, sent_idx: int, doc: Dict[str, Any], 
                                     sentence: str, sentences: List[str], 
                                     pattern_matches: List[Dict[str, Any]]) -> ContentItem:
    """Create enhanced ContentItem for recommendations"""
    
    # Generate unique ID
    content_id = f"rec_{doc_idx}_{sent_idx}_{hash(sentence) % 10000}"
    
    # Calculate character position
    char_position = calculate_character_position(doc.get('text', ''), sentence)
    
    # Enhanced position information
    position_info = {
        'character_position': char_position,
        'sentence_index': sent_idx,
        'page_number': estimate_page_number_intelligent(char_position, doc.get('text', '')),
        'percentage_through': (char_position / len(doc.get('text', ''))) * 100 if doc.get('text') else 0
    }
    
    # Get surrounding context
    context_window = AlignmentConfig.CONTEXT_WINDOW_SIZE
    context_start = max(0, sent_idx - context_window)
    context_end = min(len(sentences), sent_idx + context_window + 1)
    context = ' '.join(sentences[context_start:context_end])
    
    # Calculate confidence score with advanced metrics
    confidence_score = calculate_recommendation_confidence_advanced(sentence, pattern_matches)
    
    # Content type classification
    content_type = classify_recommendation_content_type(sentence)
    
    # Topic classification
    topic_classification = classify_topic_advanced(sentence)
    
    # Urgency assessment
    urgency_score = assess_urgency_score(sentence)
    
    # Formal language assessment
    formal_language_score = assess_formal_language_score(sentence)
    
    # Validation flags
    validation_flags = {
        'passes_narrative_filter': not is_narrative_content(sentence, 'Aggressive'),
        'has_action_language': has_action_oriented_language(sentence),
        'has_formal_structure': has_formal_sentence_structure(sentence),
        'pattern_confidence': max(match['confidence'] for match in pattern_matches)
    }
    
    return ContentItem(
        id=content_id,
        document=doc,
        sentence=sentence,
        context=context,
        patterns_matched=pattern_matches,
        confidence_score=confidence_score,
        content_type=content_type,
        topic_classification=topic_classification,
        urgency_score=urgency_score,
        formal_language_score=formal_language_score,
        position_info=position_info,
        validation_flags=validation_flags
    )

def create_response_content_item(doc_idx: int, sent_idx: int, doc: Dict[str, Any], 
                               sentence: str, sentences: List[str], 
                               pattern_matches: List[Dict[str, Any]]) -> ContentItem:
    """Create enhanced ContentItem for responses"""
    
    # Generate unique ID
    content_id = f"resp_{doc_idx}_{sent_idx}_{hash(sentence) % 10000}"
    
    # Calculate character position
    char_position = calculate_character_position(doc.get('text', ''), sentence)
    
    # Enhanced position information
    position_info = {
        'character_position': char_position,
        'sentence_index': sent_idx,
        'page_number': estimate_page_number_intelligent(char_position, doc.get('text', '')),
        'percentage_through': (char_position / len(doc.get('text', ''))) * 100 if doc.get('text') else 0
    }
    
    # Get surrounding context
    context_window = AlignmentConfig.CONTEXT_WINDOW_SIZE
    context_start = max(0, sent_idx - context_window)
    context_end = min(len(sentences), sent_idx + context_window + 1)
    context = ' '.join(sentences[context_start:context_end])
    
    # Calculate confidence score
    confidence_score = calculate_response_confidence_advanced(sentence, pattern_matches)
    
    # Content type classification
    content_type = classify_response_content_type(sentence)
    
    # Topic classification
    topic_classification = classify_topic_advanced(sentence)
    
    # Urgency assessment
    urgency_score = assess_urgency_score(sentence)
    
    # Formal language assessment
    formal_language_score = assess_formal_language_score(sentence)
    
    # Validation flags
    validation_flags = {
        'passes_narrative_filter': not is_narrative_content(sentence, 'Aggressive'),
        'has_official_language': has_official_language(sentence),
        'has_response_indicators': has_response_indicators(sentence),
        'pattern_confidence': max(match['confidence'] for match in pattern_matches)
    }
    
    return ContentItem(
        id=content_id,
        document=doc,
        sentence=sentence,
        context=context,
        patterns_matched=pattern_matches,
        confidence_score=confidence_score,
        content_type=content_type,
        topic_classification=topic_classification,
        urgency_score=urgency_score,
        formal_language_score=formal_language_score,
        position_info=position_info,
        validation_flags=validation_flags
    )

# =============================================================================
# ADVANCED SCORING AND CLASSIFICATION FUNCTIONS
# =============================================================================

def calculate_recommendation_confidence_advanced(sentence: str, pattern_matches: List[Dict[str, Any]]) -> float:
    """Calculate advanced confidence score for recommendations"""
    
    base_score = 0.0
    
    # Pattern strength scoring
    if pattern_matches:
        pattern_score = max(match['confidence'] for match in pattern_matches)
        base_score += pattern_score * 0.4
    
    # Action language bonus
    action_score = assess_action_language_strength(sentence)
    base_score += action_score * 0.3
    
    # Formal structure bonus
    structure_score = assess_formal_structure_strength(sentence)
    base_score += structure_score * 0.2
    
    # Length and complexity bonus
    complexity_score = assess_sentence_complexity(sentence)
    base_score += complexity_score * 0.1
    
    # Apply negative indicators penalty
    negative_indicators = [
        ('meeting was advised', 0.9),   # MAXIMUM PENALTY - YOUR SPECIFIC ISSUE
        ('directors were advised', 0.8), # HIGH PENALTY
        ('was advised', 0.8),           # HIGH PENALTY
        ('were told', 0.7),             # MEDIUM PENALTY
        ('explained that', 0.6),        # MEDIUM PENALTY
        ('mentioned that', 0.5),        # LOW PENALTY
        ('witness statement', 0.9),     # HIGH PENALTY
        ('stated that', 0.5)            # LOW PENALTY
    ]
    
    sentence_lower = sentence.lower()
    for indicator, penalty in negative_indicators:
        if indicator in sentence_lower:
            base_score *= (1.0 - penalty)  # Apply penalty as multiplier
    
    return min(max(base_score, 0.0), 1.0)  # Clamp between 0 and 1

def calculate_response_confidence_advanced(sentence: str, pattern_matches: List[Dict[str, Any]]) -> float:
    """Calculate advanced confidence score for responses"""
    
    base_score = 0.0
    
    # Pattern strength scoring
    if pattern_matches:
        pattern_score = max(match['confidence'] for match in pattern_matches)
        base_score += pattern_score * 0.4
    
    # Official language bonus
    official_score = assess_official_language_strength(sentence)
    base_score += official_score * 0.3
    
    # Response indicators bonus
    response_score = assess_response_indicators_strength(sentence)
    base_score += response_score * 0.2
    
    # Government context bonus
    gov_context_score = assess_government_context(sentence)
    base_score += gov_context_score * 0.1
    
    # Apply negative indicators penalty (same as recommendations)
    negative_indicators = [
        ('meeting was advised', 0.9),
        ('directors were advised', 0.8),
        ('was advised', 0.8),
        ('were told', 0.7),
        ('witness statement', 0.9)
    ]
    
    sentence_lower = sentence.lower()
    for indicator, penalty in negative_indicators:
        if indicator in sentence_lower:
            base_score *= (1.0 - penalty)
    
    return min(max(base_score, 0.0), 1.0)

def classify_recommendation_content_type(sentence: str) -> str:
    """Classify recommendation content type"""
    
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['urgent', 'immediate', 'critical', 'emergency']):
        return 'Urgent Recommendation'
    elif any(word in sentence_lower for word in ['policy', 'framework', 'strategy']):
        return 'Policy Recommendation'
    elif any(word in sentence_lower for word in ['financial', 'budget', 'funding']):
        return 'Financial Recommendation'
    elif any(word in sentence_lower for word in ['implement', 'implementation']):
        return 'Implementation Recommendation'
    else:
        return 'General Recommendation'

def classify_response_content_type(sentence: str) -> str:
    """Classify response content type"""
    
    sentence_lower = sentence.lower()
    
    if any(word in sentence_lower for word in ['accept', 'agree', 'approved']):
        return 'Positive Response'
    elif any(word in sentence_lower for word in ['reject', 'decline', 'oppose']):
        return 'Negative Response'
    elif any(word in sentence_lower for word in ['consider', 'review', 'under consideration']):
        return 'Conditional Response'
    elif any(word in sentence_lower for word in ['implement', 'implementation']):
        return 'Implementation Response'
    else:
        return 'General Response'

def classify_topic_advanced(sentence: str) -> str:
    """Advanced topic classification"""
    
    sentence_lower = sentence.lower()
    
    # Healthcare and safety
    if any(word in sentence_lower for word in ['health', 'medical', 'patient', 'safety', 'clinical']):
        return 'Healthcare & Safety'
    
    # Financial and budget
    elif any(word in sentence_lower for word in ['financial', 'budget', 'funding', 'cost', 'expenditure']):
        return 'Financial'
    
    # Legal and regulatory
    elif any(word in sentence_lower for word in ['legal', 'regulation', 'compliance', 'law', 'regulatory']):
        return 'Legal & Regulatory'
    
    # Technology and systems
    elif any(word in sentence_lower for word in ['technology', 'system', 'IT', 'digital', 'data']):
        return 'Technology & Systems'
    
    # Education and training
    elif any(word in sentence_lower for word in ['education', 'training', 'learning', 'curriculum']):
        return 'Education & Training'
    
    # Administrative and operational
    elif any(word in sentence_lower for word in ['administrative', 'operational', 'management', 'process']):
        return 'Administrative'
    
    else:
        return 'General'

def assess_urgency_score(sentence: str) -> float:
    """Assess urgency level of sentence"""
    
    sentence_lower = sentence.lower()
    urgency_indicators = {
        'immediate': 1.0,
        'urgent': 0.9,
        'critical': 0.8,
        'priority': 0.7,
        'emergency': 1.0,
        'asap': 0.8,
        'quickly': 0.6,
        'soon': 0.5
    }
    
    max_urgency = 0.0
    for indicator, score in urgency_indicators.items():
        if indicator in sentence_lower:
            max_urgency = max(max_urgency, score)
    
    return max_urgency

def assess_formal_language_score(sentence: str) -> float:
    """Assess formal language level"""
    
    sentence_lower = sentence.lower()
    formal_indicators = [
        'committee', 'department', 'ministry', 'government',
        'recommendation', 'implementation', 'consideration',
        'furthermore', 'therefore', 'consequently', 'moreover',
        'shall', 'pursuant', 'whereas', 'aforementioned'
    ]
    
    formal_count = sum(1 for indicator in formal_indicators if indicator in sentence_lower)
    words = sentence.split()
    
    if not words:
        return 0.0
    
    return min(formal_count / len(words) * 10, 1.0)  # Normalize to 0-1

# =============================================================================
# UTILITY ASSESSMENT FUNCTIONS
# =============================================================================

def assess_action_language_strength(sentence: str) -> float:
    """Assess strength of action-oriented language"""
    
    sentence_lower = sentence.lower()
    action_words = [
        'implement', 'establish', 'create', 'develop', 'ensure',
        'provide', 'improve', 'enhance', 'strengthen', 'introduce',
        'adopt', 'pursue', 'execute', 'deliver', 'deploy'
    ]
    
    action_count = sum(1 for word in action_words if word in sentence_lower)
    return min(action_count / 3.0, 1.0)  # Normalize to 0-1

def assess_formal_structure_strength(sentence: str) -> float:
    """Assess formal sentence structure strength"""
    
    words = sentence.split()
    
    # Check for proper length
    if len(words) < 8:
        return 0.3
    elif len(words) > 30:
        return 0.7
    else:
        return 0.8
    
    # Additional checks could be added here

def assess_sentence_complexity(sentence: str) -> float:
    """Assess sentence complexity"""
    
    words = sentence.split()
    
    # Basic complexity based on length and vocabulary
    length_score = min(len(words) / 20.0, 1.0)
    
    # Check for complex sentence structures
    complexity_indicators = [',', ';', ':', 'which', 'that', 'where', 'when']
    complexity_count = sum(1 for indicator in complexity_indicators if indicator in sentence)
    complexity_score = min(complexity_count / 3.0, 1.0)
    
    return (length_score + complexity_score) / 2.0

def assess_official_language_strength(sentence: str) -> float:
    """Assess strength of official government language"""
    
    sentence_lower = sentence.lower()
    official_terms = [
        'government', 'department', 'ministry', 'official',
        'policy', 'framework', 'legislation', 'regulation',
        'administration', 'authority', 'public', 'crown'
    ]
    
    official_count = sum(1 for term in official_terms if term in sentence_lower)
    return min(official_count / 3.0, 1.0)

def assess_response_indicators_strength(sentence: str) -> float:
    """Assess strength of response indicators"""
    
    sentence_lower = sentence.lower()
    response_terms = [
        'accept', 'reject', 'agree', 'disagree', 'implement',
        'consider', 'review', 'response', 'position', 'stance',
        'approve', 'decline', 'endorse', 'support'
    ]
    
    response_count = sum(1 for term in response_terms if term in sentence_lower)
    return min(response_count / 2.0, 1.0)

def assess_government_context(sentence: str) -> float:
    """Assess government context strength"""
    
    sentence_lower = sentence.lower()
    gov_context_terms = [
        'parliament', 'committee', 'inquiry', 'cabinet', 'minister',
        'secretary', 'commissioner', 'director', 'agency', 'board'
    ]
    
    context_count = sum(1 for term in gov_context_terms if term in sentence_lower)
    return min(context_count / 2.0, 1.0)

def has_action_oriented_language(sentence: str) -> bool:
    """Check if sentence has action-oriented language"""
    
    sentence_lower = sentence.lower()
    action_indicators = [
        'implement', 'establish', 'create', 'develop', 'ensure',
        'provide', 'improve', 'enhance', 'strengthen', 'introduce'
    ]
    
    return any(indicator in sentence_lower for indicator in action_indicators)

def has_formal_sentence_structure(sentence: str) -> bool:
    """Check if sentence has formal structure"""
    
    words = sentence.split()
    return len(words) >= 8 and len(words) <= 50

def has_official_language(sentence: str) -> bool:
    """Check if sentence has official government language"""
    
    sentence_lower = sentence.lower()
    official_indicators = [
        'government', 'department', 'ministry', 'official',
        'policy', 'administration', 'authority'
    ]
    
    return any(indicator in sentence_lower for indicator in official_indicators)

def has_response_indicators(sentence: str) -> bool:
    """Check if sentence has response indicators"""
    
    sentence_lower = sentence.lower()
    response_indicators = [
        'accept', 'reject', 'agree', 'disagree', 'implement',
        'consider', 'review', 'response', 'position'
    ]
    
    return any(indicator in sentence_lower for indicator in response_indicators)

# =============================================================================
# DEDUPLICATION AND OPTIMIZATION
# =============================================================================

def deduplicate_content_items_advanced(content_items: List[ContentItem]) -> List[ContentItem]:
    """Advanced deduplication with similarity analysis"""
    
    if not content_items:
        return content_items
    
    # Sort by confidence score (highest first)
    sorted_items = sorted(content_items, key=lambda x: x.confidence_score, reverse=True)
    
    deduplicated = []
    seen_sentences = set()
    
    for item in sorted_items:
        sentence_normalized = normalize_sentence_for_comparison(item.sentence)
        
        # Check for exact duplicates
        if sentence_normalized in seen_sentences:
            continue
        
        # Check for high similarity with existing items
        is_duplicate = False
        for existing_item in deduplicated:
            if calculate_sentence_similarity(item.sentence, existing_item.sentence) > 0.9:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated.append(item)
            seen_sentences.add(sentence_normalized)
    
    logger.info(f"Deduplication: {len(content_items)} → {len(deduplicated)} items")
    
    return deduplicated

def normalize_sentence_for_comparison(sentence: str) -> str:
    """Normalize sentence for comparison"""
    
    # Convert to lowercase
    normalized = sentence.lower()
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    # Remove common punctuation
    normalized = re.sub(r'[.,;:!?()]', '', normalized)
    
    return normalized

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """Calculate similarity between two sentences"""
    
    # Normalize both sentences
    norm1 = normalize_sentence_for_comparison(sentence1)
    norm2 = normalize_sentence_for_comparison(sentence2)
    
    # Simple word-based similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

# =============================================================================
# QUALITY ASSURANCE AND VALIDATION
# =============================================================================

def validate_extraction_quality(content_items: List[ContentItem]) -> Dict[str, Any]:
    """Validate extraction quality with comprehensive metrics"""
    
    if not content_items:
        return {
            'total_items': 0,
            'quality_score': 0.0,
            'confidence_distribution': {},
            'validation_issues': ['No content items extracted']
        }
    
    # Calculate quality metrics
    total_items = len(content_items)
    confidence_scores = [item.confidence_score for item in content_items]
    avg_confidence = sum(confidence_scores) / total_items
    
    # Confidence distribution
    confidence_ranges = {
        'high (>0.7)': sum(1 for score in confidence_scores if score > 0.7),
        'medium (0.4-0.7)': sum(1 for score in confidence_scores if 0.4 <= score <= 0.7),
        'low (<0.4)': sum(1 for score in confidence_scores if score < 0.4)
    }
    
    # Validation issues
    validation_issues = []
    
    # Check for narrative content that slipped through
    narrative_items = sum(1 for item in content_items 
                         if not item.validation_flags.get('passes_narrative_filter', True))
    if narrative_items > 0:
        validation_issues.append(f"{narrative_items} items may contain narrative content")
    
    # Check for low-confidence items
    low_confidence_items = confidence_ranges['low (<0.4)']
    if low_confidence_items > total_items * 0.3:
        validation_issues.append(f"High proportion of low-confidence items: {low_confidence_items}/{total_items}")
    
    # Check for duplicate content
    unique_sentences = len(set(item.sentence for item in content_items))
    if unique_sentences < total_items:
        validation_issues.append(f"Potential duplicates detected: {total_items - unique_sentences} duplicates")
    
    # Overall quality score
    quality_factors = {
        'avg_confidence': avg_confidence,
        'high_confidence_ratio': confidence_ranges['high (>0.7)'] / total_items,
        'narrative_filter_success': 1.0 - (narrative_items / total_items),
        'uniqueness': unique_sentences / total_items
    }
    
    quality_score = sum(quality_factors.values()) / len(quality_factors)
    
    return {
        'total_items': total_items,
        'quality_score': quality_score,
        'avg_confidence': avg_confidence,
        'confidence_distribution': confidence_ranges,
        'validation_issues': validation_issues,
        'quality_factors': quality_factors
    }

def generate_extraction_report(recommendations: List[ContentItem], 
                             responses: List[ContentItem]) -> str:
    """Generate comprehensive extraction report"""
    
    # Validate quality
    rec_quality = validate_extraction_quality(recommendations)
    resp_quality = validate_extraction_quality(responses)
    
    report = f"""
RECOMMENDATION-RESPONSE EXTRACTION REPORT
========================================

EXTRACTION SUMMARY:
- Recommendations Extracted: {rec_quality['total_items']}
- Responses Extracted: {resp_quality['total_items']}
- Overall Quality Score: {(rec_quality['quality_score'] + resp_quality['quality_score']) / 2:.2f}

RECOMMENDATION ANALYSIS:
- Average Confidence: {rec_quality['avg_confidence']:.2f}
- High Confidence (>0.7): {rec_quality['confidence_distribution']['high (>0.7)']}
- Medium Confidence (0.4-0.7): {rec_quality['confidence_distribution']['medium (0.4-0.7)']}
- Low Confidence (<0.4): {rec_quality['confidence_distribution']['low (<0.4)']}

RESPONSE ANALYSIS:
- Average Confidence: {resp_quality['avg_confidence']:.2f}
- High Confidence (>0.7): {resp_quality['confidence_distribution']['high (>0.7)']}
- Medium Confidence (0.4-0.7): {resp_quality['confidence_distribution']['medium (0.4-0.7)']}
- Low Confidence (<0.4): {resp_quality['confidence_distribution']['low (<0.4)']}

QUALITY ISSUES:
"""
    
    all_issues = rec_quality['validation_issues'] + resp_quality['validation_issues']
    if all_issues:
        for issue in all_issues:
            report += f"- {issue}\n"
    else:
        report += "- No significant quality issues detected\n"
    
    report += "\nRECOMMENDATIONS FOR IMPROVEMENT:\n"
    
    if rec_quality['quality_score'] < 0.7:
        report += "- Consider adjusting recommendation extraction patterns\n"
    
    if resp_quality['quality_score'] < 0.7:
        report += "- Consider adjusting response extraction patterns\n"
    
    if len(all_issues) > 3:
        report += "- Enable more aggressive narrative filtering\n"
        report += "- Review document quality and preprocessing\n"
    
    return report

# =============================================================================
# EXPORT AND UTILITY FUNCTIONS
# =============================================================================

def export_content_items_to_csv(content_items: List[ContentItem], output_type: str) -> str:
    """Export content items to CSV format"""
    
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    header = [
        'ID', 'Document', 'Sentence', 'Confidence_Score', 'Content_Type',
        'Topic_Classification', 'Urgency_Score', 'Formal_Language_Score',
        'Page_Number', 'Character_Position', 'Percentage_Through',
        'Pattern_Matches', 'Validation_Flags'
    ]
    writer.writerow(header)
    
    # Data rows
    for item in content_items:
        pattern_matches_str = '; '.join([f"{m['pattern']} ({m['confidence']:.2f})" 
                                       for m in item.patterns_matched])
        
        validation_str = '; '.join([f"{k}: {v}" for k, v in item.validation_flags.items()])
        
        row = [
            item.id,
            item.document.get('filename', 'Unknown'),
            item.sentence,
            item.confidence_score,
            item.content_type,
            item.topic_classification,
            item.urgency_score,
            item.formal_language_score,
            item.position_info.get('page_number', 1),
            item.position_info.get('character_position', 0),
            item.position_info.get('percentage_through', 0),
            pattern_matches_str,
            validation_str
        ]
        writer.writerow(row)
    
    return output.getvalue()

def get_extraction_statistics(content_items: List[ContentItem]) -> Dict[str, Any]:
    """Get comprehensive extraction statistics"""
    
    if not content_items:
        return {'total_items': 0}
    
    # Basic stats
    total_items = len(content_items)
    confidence_scores = [item.confidence_score for item in content_items]
    
    # Content type distribution
    content_types = {}
    for item in content_items:
        content_types[item.content_type] = content_types.get(item.content_type, 0) + 1
    
    # Topic distribution
    topics = {}
    for item in content_items:
        topics[item.topic_classification] = topics.get(item.topic_classification, 0) + 1
    
    # Document distribution
    documents = {}
    for item in content_items:
        doc_name = item.document.get('filename', 'Unknown')
        documents[doc_name] = documents.get(doc_name, 0) + 1
    
    # Pattern analysis
    pattern_types = {}
    for item in content_items:
        for pattern in item.patterns_matched:
            pattern_type = pattern.get('type', 'unknown')
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
    
    return {
        'total_items': total_items,
        'avg_confidence': sum(confidence_scores) / total_items,
        'min_confidence': min(confidence_scores),
        'max_confidence': max(confidence_scores),
        'content_type_distribution': content_types,
        'topic_distribution': topics,
        'document_distribution': documents,
        'pattern_type_distribution': pattern_types,
        'avg_urgency': sum(item.urgency_score for item in content_items) / total_items,
        'avg_formality': sum(item.formal_language_score for item in content_items) / total_items
    }

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

def optimize_extraction_performance(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze and optimize extraction performance"""
    
    total_chars = sum(len(doc.get('text', '')) for doc in documents)
    total_words = sum(len(doc.get('text', '').split()) for doc in documents)
    
    # Estimate processing time
    estimated_time = (total_words / 1000) * 2  # Rough estimate: 2 seconds per 1000 words
    
    # Memory estimation
    estimated_memory = total_chars * 0.000001  # Rough estimate: 1MB per 1M characters
    
    # Performance recommendations
    recommendations = []
    
    if total_words > 100000:
        recommendations.append("Consider processing documents in batches for large datasets")
    
    if estimated_memory > 500:
        recommendations.append("Large memory usage expected - consider memory-efficient processing")
    
    if estimated_time > 300:  # 5 minutes
        recommendations.append("Long processing time expected - consider parallel processing")
    
    return {
        'total_documents': len(documents),
        'total_characters': total_chars,
        'total_words': total_words,
        'estimated_processing_time_seconds': estimated_time,
        'estimated_memory_mb': estimated_memory,
        'performance_recommendations': recommendations
    }

# Export all functions
__all__ = [
    'extract_recommendations_ultimate',
    'extract_responses_ultimate',
    'find_recommendation_patterns_advanced',
    'find_response_patterns_advanced',
    'create_recommendation_content_item',
    'create_response_content_item',
    'calculate_recommendation_confidence_advanced',
    'calculate_response_confidence_advanced',
    'classify_recommendation_content_type',
    'classify_response_content_type',
    'classify_topic_advanced',
    'assess_urgency_score',
    'assess_formal_language_score',
    'deduplicate_content_items_advanced',
    'validate_extraction_quality',
    'generate_extraction_report',
    'export_content_items_to_csv',
    'get_extraction_statistics',
    'optimize_extraction_performance'
]
