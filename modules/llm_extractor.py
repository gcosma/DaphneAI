# ===============================================
# FILE: modules/llm_extractor.py
# ===============================================

import json
import re
import logging
from typing import List, Dict, Any, Optional
from core_utils import Recommendation
import time

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
        
        # Improved extraction prompt with concern extraction patterns
        self.extraction_prompt = """
You are an expert at extracting recommendations and concerns from documents.

Analyze the following document text and extract:
1. RECOMMENDATIONS - explicit suggestions, required actions, or directives
2. CONCERNS - issues, problems, risks, or areas of worry mentioned

Look for patterns like:
RECOMMENDATIONS:
- "Recommendation 1:", "Recommendation 2:", etc.
- "We recommend that..."
- "Should implement..."
- "Must establish..."
- "It is recommended..."
- "The following action should be taken..."

CONCERNS:
- "Concern about..."
- "Issue with..."
- "Problem identified..."
- "Risk of..."
- "Worried about..."
- "Difficulty in..."
- "Challenge faced..."

For each item found, extract:
1. The full text
2. Type (recommendation or concern)
3. Category/theme if identifiable
4. Confidence level (0-1)
5. Any reference numbers or IDs

Document text:
{text}

Return as valid JSON array only:
[
    {{
        "type": "recommendation",
        "id": "rec_1",
        "text": "full recommendation text",
        "category": "safety/quality/process/etc",
        "confidence": 0.95,
        "page_number": 1,
        "section": "section name"
    }},
    {{
        "type": "concern", 
        "id": "concern_1",
        "text": "full concern text",
        "category": "safety/quality/process/etc",
        "confidence": 0.90,
        "page_number": 1,
        "section": "section name"
    }}
]
"""
    
    def extract_recommendations_and_concerns(self, document_text: str, document_source: str = "") -> Dict[str, List]:
        """Extract both recommendations and concerns using LLM with fallback"""
        try:
            if OPENAI_AVAILABLE:
                response = self._call_openai(document_text)
                
                if response:
                    try:
                        items_data = json.loads(response)
                        
                        recommendations = []
                        concerns = []
                        
                        for i, item_data in enumerate(items_data):
                            item_type = item_data.get('type', 'recommendation').lower()
                            
                            if item_type == 'recommendation':
                                rec = Recommendation(
                                    id=item_data.get('id', f"REC-{len(recommendations)+1}"),
                                    text=item_data.get('text', ''),
                                    document_source=document_source,
                                    section_title=item_data.get('section', 'Unknown'),
                                    page_number=item_data.get('page_number'),
                                    confidence_score=item_data.get('confidence', 0.5),
                                    metadata={
                                        'category': item_data.get('category', 'general'),
                                        'extraction_method': 'llm'
                                    }
                                )
                                recommendations.append(rec)
                            
                            elif item_type == 'concern':
                                concern = {
                                    'id': item_data.get('id', f"CONCERN-{len(concerns)+1}"),
                                    'text': item_data.get('text', ''),
                                    'document_source': document_source,
                                    'section': item_data.get('section', 'Unknown'),
                                    'page_number': item_data.get('page_number'),
                                    'confidence_score': item_data.get('confidence', 0.5),
                                    'category': item_data.get('category', 'general'),
                                    'extraction_method': 'llm'
                                }
                                concerns.append(concern)
                        
                        self.logger.info(f"LLM extracted {len(recommendations)} recommendations and {len(concerns)} concerns")
                        return {
                            'recommendations': recommendations,
                            'concerns': concerns
                        }
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse LLM response: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in LLM extraction: {e}")
        
        # Fallback to pattern-based extraction
        self.logger.info("Using pattern-based extraction fallback")
        return self._pattern_based_extraction(document_text, document_source)
    
    def extract_recommendations(self, document_text: str, document_source: str = "") -> List[Recommendation]:
        """Extract only recommendations (backward compatibility)"""
        result = self.extract_recommendations_and_concerns(document_text, document_source)
        return result.get('recommendations', [])
    
    def _call_openai(self, text: str) -> Optional[str]:
        """Call OpenAI API with retry logic"""
        try:
            import os
            if not os.getenv("OPENAI_API_KEY"):
                self.logger.warning("OpenAI API key not found")
                return None
            
            # Truncate text if too long
            max_chars = 25000  # Reduced for better processing
            if len(text) > max_chars:
                # Smart truncation - try to keep complete sentences
                truncated = text[:max_chars]
                last_period = truncated.rfind('.')
                if last_period > max_chars * 0.8:  # Keep if we don't lose too much
                    text = truncated[:last_period + 1]
                else:
                    text = truncated + "..."
            
            client = OpenAIClient()
            
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are an expert at extracting recommendations and concerns from documents. Always respond with valid JSON only."
                            },
                            {
                                "role": "user", 
                                "content": self.extraction_prompt.format(text=text)
                            }
                        ],
                        temperature=0.1,
                        max_tokens=2500,
                        timeout=30
                    )
                    
                    return response.choices[0].message.content.strip()
                    
                except Exception as e:
                    self.logger.warning(f"OpenAI API attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return None
    
    def _pattern_based_extraction(self, text: str, source: str) -> Dict[str, List]:
        """Enhanced pattern-based extraction for recommendations and concerns"""
        recommendations = []
        concerns = []
        
        # Recommendation patterns
        rec_patterns = [
            (r"recommendation\s+(\d+)[:\.]?\s*([^.]+(?:\.[^.]*){0,3}\.)", "numbered"),
            (r"(?:we\s+)?recommend\s+that\s+([^.]+(?:\.[^.]*){0,2}\.)", "recommend_that"),
            (r"(?:should|must|ought\s+to)\s+([^.]+\.)", "modal_verbs"),
            (r"it\s+is\s+(?:recommended|suggested)\s+that\s+([^.]+\.)", "passive_voice"),
            (r"the\s+following\s+(?:action|step)s?\s+(?:should|must)\s+be\s+taken[:\s]+([^.]+\.)", "action_required"),
            (r"(?:action|step)\s+(?:required|needed)[:\s]+([^.]+\.)", "action_needed")
        ]
