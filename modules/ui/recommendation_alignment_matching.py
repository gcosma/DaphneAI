# modules/ui/recommendation_alignment_matching.py
"""
🔄 IMPROVED Alignment Engine for Recommendation-Response Matching
With enhanced self-match prevention and semantic similarity
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np

# Import from your existing core module
from .recommendation_alignment_core import ContentItem, AlignmentMatch, AlignmentConfig

logger = logging.getLogger(__name__)

# =============================================================================
# 🛡️ ENHANCED SELF-MATCH PREVENTION (FIXES YOUR MAIN ISSUE)
# =============================================================================

def is_self_match_advanced(rec: ContentItem, resp: ContentItem, prevention_enabled: bool = True) -> bool:
    """
    🛡️ CRITICAL FIX: Enhanced self-match prevention
    This directly addresses your issue of recommendations matching with themselves
    """
    
    if not prevention_enabled:
        return False
    
    # 1. EXACT TEXT MATCH - Most common issue
    if rec.sentence.strip().lower() == resp.sentence.strip().lower():
        logger.info(f"🛡️ BLOCKED: Exact text self-match detected")
        return True
    
    # 2. Near-duplicate detection (>90% similarity)
    similarity = SequenceMatcher(None, rec.sentence.lower(), resp.sentence.lower()).ratio()
    if similarity > 0.9:
        logger.info(f"🛡️ BLOCKED: High similarity self-match ({similarity:.2%})")
        return True
    
    # 3. Same document + close position
    if rec.document.get('filename') == resp.document.get('filename'):
        rec_pos = rec.position_info.get('character_position', 0)
        resp_pos = resp.position_info.get('character_position', 0)
        
        # If within 500 characters in same document, likely same content
        if abs(rec_pos - resp_pos) < 500:
            logger.info(f"🛡️ BLOCKED: Same document, close position (diff: {abs(rec_pos - resp_pos)})")
            return True
    
    # 4. Check if both have same recommendation number reference
    rec_nums = extract_recommendation_numbers(rec.sentence)
    resp_nums = extract_recommendation_numbers(resp.sentence)
    
    if rec_nums and resp_nums and rec_nums == resp_nums:
        # If they reference the same recommendation number and are very similar
        if similarity > 0.7:
            logger.info(f"🛡️ BLOCKED: Same recommendation reference with high similarity")
            return True
    
    # 5. Pattern-based detection
    # If both extracted with same patterns and start similarly
    if rec.patterns_matched and resp.patterns_matched:
        rec_patterns = {p.get('pattern', '') for p in rec.patterns_matched}
        resp_patterns = {p.get('pattern', '') for p in resp.patterns_matched}
        
        if rec_patterns == resp_patterns and rec.sentence[:50] == resp.sentence[:50]:
            logger.info(f"🛡️ BLOCKED: Identical extraction patterns")
            return True
    
    return False

# =============================================================================
# 🧠 SEMANTIC SIMILARITY CALCULATION
# =============================================================================

class SemanticMatcher:
    """Enhanced semantic similarity calculator with fallback options"""
    
    def __init__(self):
        self.model = None
        self.use_transformer = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Try to initialize transformer model with fallback"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_transformer = True
            logger.info("✅ Transformer model loaded successfully")
        except Exception as e:
            logger.info(f"📊 Using enhanced keyword matching (transformer unavailable: {e})")
            self.use_transformer = False
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity with automatic fallback"""
        
        if not text1 or not text2:
            return 0.0
        
        if self.use_transformer and self.model:
            try:
                # Use transformer-based similarity
                embeddings = self.model.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(similarity)
            except Exception as e:
                logger.warning(f"Transformer failed, using fallback: {e}")
        
        # Fallback to enhanced keyword matching
        return self._keyword_similarity_enhanced(text1, text2)
    
    def _keyword_similarity_enhanced(self, text1: str, text2: str) -> float:
        """Enhanced keyword-based similarity for government documents"""
        
        # Extract meaningful words
        words1 = set(self._extract_keywords(text1))
        words2 = set(self._extract_keywords(text2))
        
        if not words1 or not words2:
            return 0.0
        
        # Base Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        base_score = intersection / union if union > 0 else 0.0
        
        # Government terminology boost
        gov_terms = {
            'recommend', 'recommendation', 'accept', 'reject', 'implement',
            'government', 'department', 'ministry', 'policy', 'response',
            'committee', 'inquiry', 'report', 'principle', 'full'
        }
        
        gov_matches = len((words1 & words2) & gov_terms)
        gov_boost = min(gov_matches * 0.05, 0.2)  # Up to 20% boost
        
        # Synonym matching boost
        synonym_boost = self._calculate_synonym_boost(words1, words2)
        
        return min(base_score + gov_boost + synonym_boost, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        import re
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _calculate_synonym_boost(self, words1: set, words2: set) -> float:
        """Calculate boost from synonym matching"""
        
        synonym_groups = [
            {'recommend', 'suggest', 'advise', 'propose', 'urge'},
            {'accept', 'agree', 'approve', 'endorse', 'support'},
            {'reject', 'decline', 'refuse', 'oppose', 'dismiss'},
            {'implement', 'execute', 'deploy', 'establish', 'introduce'},
            {'review', 'examine', 'assess', 'evaluate', 'consider'}
        ]
        
        total_boost = 0.0
        for group in synonym_groups:
            if (words1 & group) and (words2 & group):
                total_boost += 0.02
        
        return min(total_boost, 0.1)  # Cap at 10% boost

# =============================================================================
# 📊 COMPREHENSIVE ALIGNMENT CALCULATION
# =============================================================================

def calculate_comprehensive_similarity(
    rec: ContentItem, 
    resp: ContentItem, 
    semantic_matcher: SemanticMatcher,
    config: Dict[str, Any]
) -> Tuple[Dict[str, float], float]:
    """
    Calculate multi-factor similarity scores
    Returns: (individual_scores, combined_score)
    """
    
    scores = {}
    
    # 1. Semantic similarity (most important)
    scores['semantic'] = semantic_matcher.calculate_semantic_similarity(
        rec.sentence, resp.sentence
    )
    
    # 2. Reference number matching (very reliable for government docs)
    scores['reference'] = calculate_reference_similarity(rec.sentence, resp.sentence)
    
    # 3. Context similarity
    scores['context'] = semantic_matcher.calculate_semantic_similarity(
        rec.context, resp.context
    )
    
    # 4. Topic alignment
    scores['topic'] = calculate_topic_alignment(rec, resp)
    
    # 5. Document relationship score
    scores['document'] = calculate_document_relationship(rec, resp)
    
    # 6. Position proximity (for same document)
    if rec.document.get('filename') == resp.document.get('filename'):
        scores['position'] = calculate_position_proximity(rec, resp)
    else:
        scores['position'] = 0.5  # Neutral score for different documents
    
    # Calculate weighted combination
    weights = config.get('scoring_weights', {
        'semantic': 0.35,
        'reference': 0.25,
        'context': 0.15,
        'topic': 0.10,
        'document': 0.10,
        'position': 0.05
    })
    
    combined_score = sum(scores[key] * weights.get(key, 0) for key in scores)
    
    return scores, combined_score

# =============================================================================
# 🔍 REFERENCE NUMBER MATCHING
# =============================================================================

def extract_recommendation_numbers(text: str) -> set:
    """Extract recommendation numbers from text"""
    
    # Patterns for recommendation numbers
    patterns = [
        r'recommendation\s+(\d+[a-z]?(?:\s*\([ivxIVX]+\))?)',
        r'rec\s+(\d+[a-z]?)',
        r'recommendation\s+([ivxIVX]+)',
        r'\b(\d+[a-z]?(?:\s*\([ivxIVX]+\))?)\s+(?:accepted|rejected|implemented)'
    ]
    
    numbers = set()
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        numbers.update(matches)
    
    return numbers

def calculate_reference_similarity(rec_text: str, resp_text: str) -> float:
    """Calculate similarity based on recommendation number references"""
    
    rec_nums = extract_recommendation_numbers(rec_text)
    resp_nums = extract_recommendation_numbers(resp_text)
    
    if not rec_nums and not resp_nums:
        return 0.0  # No references found
    
    if not rec_nums or not resp_nums:
        return 0.0  # One has references, other doesn't
    
    # Calculate overlap
    intersection = len(rec_nums & resp_nums)
    union = len(rec_nums | resp_nums)
    
    return intersection / union if union > 0 else 0.0

# =============================================================================
# 📐 ADDITIONAL SCORING FUNCTIONS
# =============================================================================

def calculate_topic_alignment(rec: ContentItem, resp: ContentItem) -> float:
    """Calculate topic alignment between recommendation and response"""
    
    # Use the topic classifications from ContentItem
    if rec.topic_classification == resp.topic_classification:
        return 1.0
    
    # Check for related topics
    related_topics = {
        'healthcare': {'health', 'medical', 'nhs', 'patient'},
        'policy': {'governance', 'regulation', 'framework', 'legislation'},
        'emergency': {'crisis', 'pandemic', 'response', 'preparedness'}
    }
    
    rec_topic = rec.topic_classification.lower()
    resp_topic = resp.topic_classification.lower()
    
    for main_topic, related in related_topics.items():
        if rec_topic in related and resp_topic in related:
            return 0.7
    
    return 0.3  # Different topics

def calculate_document_relationship(rec: ContentItem, resp: ContentItem) -> float:
    """Calculate document relationship score"""
    
    rec_doc = rec.document
    resp_doc = resp.document
    
    # Check document types
    rec_type = classify_document_type(rec_doc)
    resp_type = classify_document_type(resp_doc)
    
    if rec_type == 'recommendation' and resp_type == 'response':
        return 1.0  # Perfect document type match
    elif rec_type == 'recommendation' and resp_type == 'mixed':
        return 0.7  # Response might be in mixed document
    else:
        return 0.3  # Less likely to be correct match

def classify_document_type(document: Dict[str, Any]) -> str:
    """Classify document as recommendation, response, or mixed"""
    
    text_lower = document.get('text', '').lower()[:5000]  # Check first 5000 chars
    filename_lower = document.get('filename', '').lower()
    
    # Strong indicators
    if 'government response' in text_lower or 'response' in filename_lower:
        return 'response'
    if 'inquiry report' in text_lower or 'recommendations' in filename_lower:
        return 'recommendation'
    
    # Count indicators
    response_count = text_lower.count('accept') + text_lower.count('reject') + text_lower.count('response')
    rec_count = text_lower.count('recommend') + text_lower.count('suggest') + text_lower.count('should')
    
    if response_count > rec_count * 1.5:
        return 'response'
    elif rec_count > response_count * 1.5:
        return 'recommendation'
    else:
        return 'mixed'

def calculate_position_proximity(rec: ContentItem, resp: ContentItem) -> float:
    """Calculate position proximity score for items in same document"""
    
    rec_pos = rec.position_info.get('character_position', 0)
    resp_pos = resp.position_info.get('character_position', 0)
    
    # Calculate distance
    distance = abs(rec_pos - resp_pos)
    
    # Convert to score (closer = higher score, but not too close to avoid self-matches)
    if distance < 500:  # Too close, might be same content
        return 0.0
    elif distance < 2000:  # Good proximity
        return 1.0 - (distance - 500) / 1500
    elif distance < 10000:  # Moderate proximity
        return 0.5 - (distance - 2000) / 16000
    else:  # Far apart
        return 0.2

# =============================================================================
# 🎯 MAIN ALIGNMENT FUNCTION
# =============================================================================

def perform_alignment_matching(
    recommendations: List[ContentItem],
    responses: List[ContentItem],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Main function to align recommendations with responses
    This is the function you'll call from your UI
    """
    
    logger.info(f"🔄 Starting alignment: {len(recommendations)} recommendations, {len(responses)} responses")
    
    # Initialize semantic matcher
    semantic_matcher = SemanticMatcher()
    
    # Configuration
    min_similarity = config.get('min_similarity_threshold', 0.4)
    max_results_per_rec = config.get('max_results', 3)
    enable_self_match_prevention = config.get('self_match_prevention', True)
    
    alignments = []
    
    for rec in recommendations:
        potential_matches = []
        
        for resp in responses:
            # CRITICAL: Check for self-match first
            if is_self_match_advanced(rec, resp, enable_self_match_prevention):
                logger.debug(f"Skipped self-match: {rec.id} <-> {resp.id}")
                continue
            
            # Calculate comprehensive similarity
            scores, combined_score = calculate_comprehensive_similarity(
                rec, resp, semantic_matcher, config
            )
            
            # Only consider if above threshold
            if combined_score >= min_similarity:
                match_info = {
                    'response': resp,
                    'similarity_scores': scores,
                    'combined_score': combined_score,
                    'match_quality': determine_match_quality(combined_score),
                    'confidence_level': determine_confidence_level(combined_score),
                    'explanation': generate_match_explanation(scores, combined_score)
                }
                potential_matches.append(match_info)
        
        # Sort by score and limit results
        potential_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        if max_results_per_rec != "All":
            potential_matches = potential_matches[:max_results_per_rec]
        
        # Create alignment entry
        alignment = {
            'recommendation': rec,
            'responses': potential_matches,
            'alignment_confidence': potential_matches[0]['combined_score'] if potential_matches else 0,
            'alignment_status': determine_alignment_status(potential_matches),
            'has_response': len(potential_matches) > 0
        }
        
        alignments.append(alignment)
    
    # Log statistics
    aligned_count = sum(1 for a in alignments if a['has_response'])
    logger.info(f"✅ Alignment complete: {aligned_count}/{len(recommendations)} recommendations have responses")
    
    return alignments

# =============================================================================
# 🏷️ CLASSIFICATION FUNCTIONS
# =============================================================================

def determine_match_quality(score: float) -> str:
    """Determine match quality based on score"""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Poor"

def determine_confidence_level(score: float) -> str:
    """Determine confidence level based on score"""
    if score >= 0.75:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"

def determine_alignment_status(matches: List[Dict]) -> str:
    """Determine overall alignment status"""
    if not matches:
        return "No Response Found"
    
    top_score = matches[0]['combined_score']
    if top_score >= 0.7:
        return "Well Aligned"
    elif top_score >= 0.5:
        return "Partially Aligned"
    else:
        return "Weakly Aligned"

def generate_match_explanation(scores: Dict[str, float], combined_score: float) -> str:
    """Generate explanation for the match"""
    
    explanations = []
    
    if scores['semantic'] >= 0.7:
        explanations.append("Strong semantic similarity")
    elif scores['semantic'] >= 0.4:
        explanations.append("Moderate semantic similarity")
    
    if scores['reference'] >= 0.8:
        explanations.append("Matching recommendation references")
    
    if scores['context'] >= 0.6:
        explanations.append("Similar context")
    
    if scores['document'] >= 0.7:
        explanations.append("Appropriate document pairing")
    
    if not explanations:
        if combined_score >= 0.5:
            explanations.append("General alignment detected")
        else:
            explanations.append("Weak alignment")
    
    return "; ".join(explanations)

# Export main functions
__all__ = [
    'perform_alignment_matching',
    'is_self_match_advanced',
    'SemanticMatcher',
    'calculate_comprehensive_similarity'
]
