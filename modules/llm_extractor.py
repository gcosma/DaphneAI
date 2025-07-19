# ===============================================
# FILE: modules/llm_extractor.py
# ===============================================

import json
import re
import logging
from typing import List, Dict, Any, Optional
from core_utils import Recommendation

try:
    import openai
    from openai import OpenAI as OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class LLMRecommendationExtractor:
    def __init__(self, llm_provider="openai", model="gpt-3.5-turbo"):
        self.llm_provider = llm_provider
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        self.extraction_prompt = """
You are an expert at extracting recommendations from documents.

Analyze the following document text and extract all recommendations, suggestions,
or required actions. Look for patterns like:
- "Recommendation 1:", "Recommendation 2:", etc.
- "We recommend that..."
- "Should implement..."
- "Must establish..."

For each recommendation found, extract:
1. The full recommendation text
2. Any recommendation ID or number
3. The section it appears in
4. Confidence level (0-1)

Document text:
{text}

Return as valid JSON array only:
[
    {{
        "id": "recommendation_identifier",
        "text": "full recommendation text",
        "section": "section name where found",
        "confidence": 0.95,
        "page_number": 1
    }}
]
"""
    
    def extract_recommendations(self, document_text: str, document_source: str = "") -> List[Recommendation]:
        """Extract recommendations using LLM with fallback"""
        try:
            if OPENAI_AVAILABLE:
                response = self._call_openai(document_text)
                
                if response:
                    try:
                        recommendations_data = json.loads(response)
                        
                        recommendations = []
                        for i, rec_data in enumerate(recommendations_data):
                            rec = Recommendation(
                                id=rec_data.get('id', f"REC-{i+1}"),
                                text=rec_data.get('text', ''),
                                document_source=document_source,
                                section_title=rec_data.get('section', 'Unknown'),
                                page_number=rec_data.get('page_number'),
                                confidence_score=rec_data.get('confidence', 0.5)
                            )
                            recommendations.append(rec)
                        
                        self.logger.info(f"LLM extracted {len(recommendations)} recommendations")
                        return recommendations
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse LLM response: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in LLM extraction: {e}")
        
        # Fallback to pattern-based extraction
        self.logger.info("Using pattern-based extraction fallback")
        return self._pattern_based_extraction(document_text, document_source)
    
    def _call_openai(self, text: str) -> Optional[str]:
        """Call OpenAI API"""
        try:
            import os
            if not os.getenv("OPENAI_API_KEY"):
                self.logger.warning("OpenAI API key not found")
                return None
            
            # Truncate text if too long
            max_chars = 30000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            client = OpenAIClient()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting recommendations from documents."},
                    {"role": "user", "content": self.extraction_prompt.format(text=text)}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return None
    
    def _pattern_based_extraction(self, text: str, source: str) -> List[Recommendation]:
        """Fallback pattern-based extraction"""
        recommendations = []
        
        patterns = [
            (r"recommendation\s+(\d+)[:\.]?\s*([^.]+(?:\.[^.]*){0,2}\.)", "numbered"),
            (r"(?:we\s+)?recommend\s+that\s+([^.]+(?:\.[^.]*){0,1}\.)", "recommend_that"),
            (r"(?:should|must|ought\s+to)\s+([^.]+\.)", "modal_verbs"),
            (r"it\s+is\s+(?:recommended|suggested)\s+that\s+([^.]+\.)", "passive_voice")
        ]
        
        for pattern, pattern_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                if pattern_type == "numbered":
                    rec_id = f"REC-{match.group(1)}"
                    rec_text = match.group(2).strip()
                else:
                    rec_id = f"REC-{len(recommendations)+1}"
                    rec_text = match.group(1).strip()
                
                rec_text = re.sub(r'\s+', ' ', rec_text)
                
                if len(rec_text) < 20 or not rec_text.endswith('.'):
                    continue
                
                recommendation = Recommendation(
                    id=rec_id,
                    text=rec_text,
                    document_source=source,
                    section_title="Pattern-based extraction",
                    confidence_score=0.6
                )
                recommendations.append(recommendation)
        
        recommendations = self._remove_duplicate_recommendations(recommendations)
        
        self.logger.info(f"Pattern-based extraction found {len(recommendations)} recommendations")
        return recommendations
    
    def _remove_duplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations"""
        if len(recommendations) <= 1:
            return recommendations
        
        unique_recs = []
        seen_texts = set()
        
        for rec in recommendations:
            normalized_text = re.sub(r'\s+', ' ', rec.text.lower().strip())
            
            is_duplicate = False
            
            for seen_text in seen_texts:
                            if len(set(normalized_text.split()) & set(seen_text.split())) / len(set(normalized_text.split()) | set(seen_text.split())) > 0.7:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            unique_recs.append(rec)
                            seen_texts.add(normalized_text)
                    
                    return unique_recs
