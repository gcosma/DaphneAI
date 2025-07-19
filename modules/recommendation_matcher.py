# ===============================================
# FILE: modules/recommendation_matcher.py
# ===============================================

import logging
from typing import List, Dict, Any, Tuple, Optional

from core_utils import Recommendation
from rag_engine import RAGQueryEngine
from bert_annotator import BERTConceptAnnotator

class RecommendationResponseMatcher:
    def __init__(self, rag_engine: RAGQueryEngine, bert_annotator: BERTConceptAnnotator):
        self.rag_engine = rag_engine
        self.bert_annotator = bert_annotator
        self.logger = logging.getLogger(__name__)
    
    def match_recommendation_to_responses(self, recommendation: Recommendation) -> List[Dict[str, Any]]:
        """Match recommendation to responses using multi-modal approach"""
        try:
            # RAG-based matching
            rag_responses = self.rag_engine.find_responses_for_recommendation(recommendation)
            
            # Annotate recommendation if needed
            if not recommendation.annotations:
                rec_annotations, _ = self.bert_annotator.annotate_text(recommendation.text)
                recommendation.annotations = rec_annotations
            
            # Enhanced matching with concept validation
            enhanced_matches = []
            
            for rag_response in rag_responses:
                try:
                    # Annotate response text
                    response_annotations, _ = self.bert_annotator.annotate_text(rag_response['text'])
                    
                    # Calculate concept overlap
                    concept_overlap = self._calculate_concept_overlap(
                        recommendation.annotations, 
                        response_annotations
                    )
                    
                    # Combined confidence score
                    combined_confidence = self._calculate_combined_confidence(
                        rag_response['similarity_score'],
                        concept_overlap['overlap_score']
                    )
                    
                    # Create enhanced match
                    enhanced_match = {
                        **rag_response,
                        'response_annotations': response_annotations,
                        'concept_overlap': concept_overlap,
                        'combined_confidence': combined_confidence,
                        'match_type': self._determine_match_type(combined_confidence)
                    }
                    
                    enhanced_matches.append(enhanced_match)
                    
                except Exception as e:
                    self.logger.error(f"Error processing individual match: {e}")
                    enhanced_matches.append({
                        **rag_response,
                        'response_annotations': {},
                        'concept_overlap': {'overlap_score': 0, 'shared_themes': []},
                        'combined_confidence': rag_response['similarity_score'],
                        'match_type': 'BASIC_MATCH'
                    })
            
            enhanced_matches.sort(key=lambda x: x['combined_confidence'], reverse=True)
            return enhanced_matches
            
        except Exception as e:
            self.logger.error(f"Error matching recommendation to responses: {e}")
            return []
    
    def batch_match_recommendations(self, recommendations: List[Recommendation]) -> Dict[str, List[Dict[str, Any]]]:
        """Match multiple recommendations to responses"""
        matches = {}
        
        for recommendation in recommendations:
            try:
                matches[recommendation.id] = self.match_recommendation_to_responses(recommendation)
            except Exception as e:
                self.logger.error(f"Error matching recommendation {recommendation.id}: {e}")
                matches[recommendation.id] = []
        
        return matches
    
    def _calculate_concept_overlap(self, rec_annotations: Dict, resp_annotations: Dict) -> Dict[str, Any]:
        """Calculate concept overlap between recommendation and response"""
        overlap_info = {
            'shared_frameworks': [],
            'shared_themes': [],
            'overlap_score': 0.0,
            'total_rec_concepts': 0,
            'total_resp_concepts': 0
        }
        
        if not rec_annotations or not resp_annotations:
            return overlap_info
        
        try:
            # Count total concepts
            for framework, themes in rec_annotations.items():
                overlap_info['total_rec_concepts'] += len(themes)
            
            for framework, themes in resp_annotations.items():
                overlap_info['total_resp_concepts'] += len(themes)
            
            # Find overlapping frameworks and themes
            shared_themes = []
            
            for framework in rec_annotations:
                if framework in resp_annotations:
                    overlap_info['shared_frameworks'].append(framework)
                    
                    rec_themes = {theme['theme'] for theme in rec_annotations[framework]}
                    resp_themes = {theme['theme'] for theme in resp_annotations[framework]}
                    
                    framework_shared = rec_themes.intersection(resp_themes)
                    for theme in framework_shared:
                        shared_themes.append(f"{framework}: {theme}")
            
            overlap_info['shared_themes'] = shared_themes
            
            # Calculate overlap score
            if overlap_info['total_rec_concepts'] > 0:
                overlap_info['overlap_score'] = len(shared_themes) / overlap_info['total_rec_concepts']
            
            return overlap_info
            
        except Exception as e:
            self.logger.error(f"Error calculating concept overlap: {e}")
            return overlap_info
    
    def _calculate_combined_confidence(self, semantic_score: float, concept_score: float) -> float:
        """Calculate combined confidence"""
        try:
            weights = {'semantic': 0.7, 'concept': 0.3}
            combined = (semantic_score * weights['semantic']) + (concept_score * weights['concept'])
            return min(max(combined, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating combined confidence: {e}")
            return semantic_score
    
    def _determine_match_type(self, confidence: float) -> str:
        """Determine match type based on confidence"""
        if confidence >= 0.85:
            return "HIGH_CONFIDENCE"
        elif confidence >= 0.65:
            return "MEDIUM_CONFIDENCE" 
        elif confidence >= 0.45:
            return "LOW_CONFIDENCE"
        else:
            return "POOR_MATCH"
