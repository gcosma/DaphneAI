"""
Universal UK Government Inquiry Recommendation Extractor

Designed to extract recommendations from UK Government inquiry reports and responses.
Optimized for common patterns found in public inquiries, statutory inquiries, 
and government response documents.

Works with various inquiry types including:
- Public Health inquiries
- Safety inquiries  
- Administrative inquiries
- Independent reviews
- Government departmental reviews
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

class UKInquiryRecommendationExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Universal patterns for UK Government inquiry documents
        self.recommendation_patterns = {
            # Main numbered recommendations (1-50+ in large inquiries)
            'main_numbered': [
                r'(\d{1,3})\.\s+([^0-9]+?)(?=\d{1,3}\.|$|\n\n|\(a\))',
                r'Recommendation\s+(\d{1,3})[:\.]?\s*([^.]+(?:\.[^.]*){0,15}?)(?=Recommendation\s+\d|\Z)',
                r'(\d{1,3})\s*[-–]\s*([^0-9]+?)(?=\d{1,3}\s*[-–]|$|\n\n)',
            ],
            
            # Sub-recommendations with letters (a), (b), (c), etc.
            'sub_lettered': [
                r'\(([a-z])\)\s+([^(]+?)(?=\([a-z]\)|\d{1,3}\.|$)',
                r'(\d{1,3}[a-z])\)\s+([^0-9]+?)(?=\d{1,3}[a-z]\)|\d{1,3}\.|$)',
                r'([a-z])\.\s+([^a-z\.]+?)(?=[a-z]\.|$|\n\n)',
            ],
            
            # Roman numeral sub-recommendations (i), (ii), (iii), etc.
            'sub_roman': [
                r'\(([ivx]+)\)\s+([^(]+?)(?=\([ivx]+\)|\([a-z]\)|\d{1,3}\.|$)',
                r'([ivx]+)\.\s+([^ivx\.]+?)(?=[ivx]+\.|$|\n\n)',
            ],
            
            # Common UK inquiry language patterns
            'inquiry_language': [
                r'((?:should|must|ought to)[^.]+\.)',
                r'((?:I|We) recommend(?:s)? that[^.]+\.)',
                r'(It is recommended that[^.]+\.)',
                r'(The (?:Government|Department|Authority|Trust|Board)[^.]+should[^.]+\.)',
                r'((?:steps|action|measures) (?:should|must) be taken[^.]+\.)',
                r'(consideration (?:should )?be given[^.]+\.)',
                r'((?:should|must) be (?:established|implemented|introduced|created)[^.]+\.)',
                r'(There (?:should|must) be[^.]+\.)',
            ],
            
            # Government response patterns (universal across inquiries)
            'government_response': [
                r'(Recommendation\s+\d+[a-z]*[)\.]?\s*[^.]+\.)',
                r'((?:is|are) accepted in full[^.]*\.)',
                r'((?:is|are) accepted in principle[^.]*\.)',
                r'((?:is|are) not accepted[^.]*\.)',
                r'(This recommendation[^.]+\.)',
                r'(The Government (?:accepts|rejects|supports)[^.]+\.)',
                r'((?:will|shall) be implemented[^.]+\.)',
            ],
            
            # Action-oriented recommendations
            'action_patterns': [
                r'((?:establish|create|implement|introduce|develop)[^.]+\.)',
                r'((?:review|assess|evaluate|examine)[^.]+\.)',
                r'((?:ensure|guarantee|require)[^.]+\.)',
                r'((?:provide|deliver|offer)[^.]+\.)',
                r'((?:improve|enhance|strengthen)[^.]+\.)',
            ]
        }
        
        # AI prompt for general UK inquiry documents
        self.ai_prompt = """
You are an expert at extracting recommendations from UK Government inquiry reports, reviews, and official responses.

TASK: Extract ALL recommendations from this text.

LOOK FOR THESE COMMON PATTERNS:
1. **Numbered recommendations**: "1. The Department should...", "Recommendation 5:", "15 - We recommend"
2. **Sub-recommendations**: "(a) consideration should be given", "(i) steps must be taken"
3. **Action statements**: "should be established", "must be implemented", "ought to be reviewed"
4. **Government responses**: "is accepted in full", "accepted in principle", "not accepted"
5. **Inquiry language**: "I recommend that", "It is recommended", "We recommend"

DOCUMENT TYPES: This could be from:
- Public inquiry reports
- Government departmental reviews  
- Independent investigations
- Statutory inquiry responses
- Policy reviews and recommendations

For each recommendation found, extract:
- The recommendation ID/number if available
- The full text of the recommendation
- Whether it's from the inquiry/review or government response
- The main topic/theme if identifiable

Return as JSON array:
[
    {
        "id": "1" or "auto_generated_id",
        "text": "The full recommendation text",
        "type": "main_recommendation|sub_recommendation|government_response|action_item",
        "source": "inquiry_report|government_response|review_document",
        "theme": "topic if identifiable",
        "confidence": 0.95
    }
]

Document text:
{text}
"""

    def extract_recommendations(self, document_text: str, document_source: str = "") -> Dict[str, List]:
        """
        Extract recommendations using multiple methods optimized for UK inquiry documents
        """
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(document_text)
            
            # Try AI extraction first (if available)
            ai_results = self._extract_with_ai(cleaned_text)
            
            # Always run pattern extraction as backup/supplement
            pattern_results = self._extract_with_patterns(cleaned_text, document_source)
            
            # Combine and deduplicate results
            combined_results = self._combine_results(ai_results, pattern_results)
            
            return {
                'recommendations': combined_results,
                'extraction_info': {
                    'ai_found': len(ai_results) if ai_results else 0,
                    'pattern_found': len(pattern_results),
                    'total_combined': len(combined_results),
                    'document_source': document_source,
                    'extraction_date': datetime.now().isoformat(),
                    'extractor_version': 'UK_Inquiry_v1.0'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return {'recommendations': [], 'extraction_info': {'error': str(e)}}

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better extraction"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken word boundaries
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix numbered list formatting
        text = re.sub(r'(\d+)\.([A-Z])', r'\1. \2', text)
        text = re.sub(r'\(([a-z])\)([A-Z])', r'(\1) \2', text)
        
        # Fix recommendation numbering
        text = re.sub(r'Recommendation(\d+)', r'Recommendation \1', text)
        
        # Fix common formatting issues in government documents
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
        
        return text.strip()

    def _extract_with_patterns(self, text: str, source: str) -> List[Dict]:
        """Extract using regex patterns common across UK inquiry documents"""
        recommendations = []
        
        # Extract main numbered recommendations (1-999)
        for pattern in self.recommendation_patterns['main_numbered']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                rec_id = match.group(1)
                rec_text = match.group(2).strip()
                
                if len(rec_text) > 20 and self._is_valid_recommendation(rec_text):
                    recommendations.append({
                        'id': rec_id,
                        'text': rec_text,
                        'type': 'main_recommendation',
                        'source': 'pattern_extraction',
                        'document_source': source,
                        'confidence': 0.9,
                        'extraction_method': 'numbered_pattern'
                    })
        
        # Extract sub-recommendations with letters
        for pattern in self.recommendation_patterns['sub_lettered']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                rec_id = match.group(1)
                rec_text = match.group(2).strip()
                
                if len(rec_text) > 30 and self._is_valid_recommendation(rec_text):
                    recommendations.append({
                        'id': rec_id,
                        'text': rec_text,
                        'type': 'sub_recommendation',
                        'source': 'pattern_extraction',
                        'document_source': source,
                        'confidence': 0.85,
                        'extraction_method': 'lettered_pattern'
                    })
        
        # Extract roman numeral sub-recommendations
        for pattern in self.recommendation_patterns['sub_roman']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                rec_id = match.group(1)
                rec_text = match.group(2).strip()
                
                if len(rec_text) > 25 and self._is_valid_recommendation(rec_text):
                    recommendations.append({
                        'id': rec_id,
                        'text': rec_text,
                        'type': 'sub_recommendation',
                        'source': 'pattern_extraction',
                        'document_source': source,
                        'confidence': 0.8,
                        'extraction_method': 'roman_pattern'
                    })
        
        # Extract inquiry-specific language patterns
        for pattern in self.recommendation_patterns['inquiry_language']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                rec_text = match.group(1).strip()
                
                if len(rec_text) > 40 and self._is_valid_recommendation(rec_text):
                    recommendations.append({
                        'id': f"REC_{len(recommendations)+1}",
                        'text': rec_text,
                        'type': 'action_recommendation',
                        'source': 'pattern_extraction',
                        'document_source': source,
                        'confidence': 0.8,
                        'extraction_method': 'language_pattern'
                    })
        
        # Extract action-oriented patterns
        for pattern in self.recommendation_patterns['action_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                rec_text = match.group(1).strip()
                
                if len(rec_text) > 30 and self._is_valid_recommendation(rec_text):
                    recommendations.append({
                        'id': f"ACTION_{len(recommendations)+1}",
                        'text': rec_text,
                        'type': 'action_item',
                        'source': 'pattern_extraction',
                        'document_source': source,
                        'confidence': 0.75,
                        'extraction_method': 'action_pattern'
                    })
        
        # Extract government responses (if applicable)
        if any(keyword in source.lower() for keyword in ['government', 'response', 'reply', 'answer']):
            for pattern in self.recommendation_patterns['government_response']:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    rec_text = match.group(1).strip()
                    
                    if len(rec_text) > 25:
                        recommendations.append({
                            'id': f"GOV_{len(recommendations)+1}",
                            'text': rec_text,
                            'type': 'government_response',
                            'source': 'pattern_extraction',
                            'document_source': source,
                            'confidence': 0.9,
                            'extraction_method': 'response_pattern'
                        })
        
        return recommendations

    def _is_valid_recommendation(self, text: str) -> bool:
        """Check if extracted text is likely a valid recommendation"""
        # Filter out common false positives
        invalid_patterns = [
            r'^(page|chapter|section|figure|table)\s+\d+',
            r'^(the|this|that|it|he|she|they)\s',
            r'^\d+\s*(st|nd|rd|th)\s',
            r'^(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'^\s*(and|or|but|however|therefore|furthermore|moreover)\s',
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in invalid_patterns:
            if re.match(pattern, text_lower):
                return False
        
        # Must contain some recommendation-like words
        recommendation_indicators = [
            'should', 'must', 'ought', 'recommend', 'suggest', 'propose',
            'establish', 'implement', 'create', 'develop', 'ensure',
            'consider', 'review', 'assess', 'evaluate', 'improve'
        ]
        
        return any(indicator in text_lower for indicator in recommendation_indicators)

    def _extract_with_ai(self, text: str) -> Optional[List[Dict]]:
        """Extract using AI (OpenAI) if available"""
        try:
            import openai
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.info("No OpenAI API key found, skipping AI extraction")
                return None
            
            # Truncate if too long (GPT-3.5 limit consideration)
            if len(text) > 30000:
                text = text[:30000] + "..."
                self.logger.info("Text truncated for AI processing")
            
            client = openai.OpenAI(api_key=api_key)
            
            # Retry logic for robustness
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert at extracting recommendations from UK Government inquiry reports and official documents. Return only valid JSON."},
                            {"role": "user", "content": self.ai_prompt.format(text=text)}
                        ],
                        temperature=0.1,
                        max_tokens=4000,
                        timeout=60
                    )
                    
                    ai_response = response.choices[0].message.content.strip()
                    
                    # Clean JSON response (remove markdown formatting if present)
                    ai_response = re.sub(r'^```json\s*', '', ai_response)
                    ai_response = re.sub(r'\s*```$', '', ai_response)
                    
                    # Parse JSON response
                    recommendations = json.loads(ai_response)
                    
                    # Validate and add metadata
                    validated_recs = []
                    for rec in recommendations:
                        if isinstance(rec, dict) and 'text' in rec and len(rec['text'].strip()) > 10:
                            # Add standardized metadata
                            rec['source'] = 'ai_extraction'
                            rec['extraction_method'] = 'openai_gpt'
                            if 'confidence' not in rec:
                                rec['confidence'] = 0.8
                            if 'id' not in rec:
                                rec['id'] = f"AI_{len(validated_recs)+1}"
                            if 'type' not in rec:
                                rec['type'] = 'general_recommendation'
                            
                            validated_recs.append(rec)
                    
                    self.logger.info(f"AI extracted {len(validated_recs)} recommendations")
                    return validated_recs
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"AI response parsing failed (attempt {attempt+1}): {e}")
                    if attempt == max_retries - 1:
                        self.logger.error("All AI extraction attempts failed due to invalid JSON")
                        return None
                    
                except Exception as e:
                    self.logger.warning(f"AI API call failed (attempt {attempt+1}): {e}")
                    if attempt == max_retries - 1:
                        raise
            
        except ImportError:
            self.logger.info("OpenAI library not available, skipping AI extraction")
            return None
        except Exception as e:
            self.logger.warning(f"AI extraction failed: {e}")
            return None

    def _combine_results(self, ai_results: Optional[List[Dict]], pattern_results: List[Dict]) -> List[Dict]:
        """Combine AI and pattern results, removing duplicates"""
        combined = []
        
        # Add pattern results first (they're more reliable for structure)
        combined.extend(pattern_results)
        
        # Add AI results if available, avoiding duplicates
        if ai_results:
            for ai_rec in ai_results:
                ai_text = ai_rec.get('text', '').lower().strip()
                
                # Check for duplicates using similarity
                is_duplicate = False
                for existing in combined:
                    existing_text = existing.get('text', '').lower().strip()
                    
                    # Multiple similarity checks
                    if len(ai_text) > 20 and len(existing_text) > 20:
                        # Check if significant portion overlaps
                        if (ai_text[:60] in existing_text) or (existing_text[:60] in ai_text):
                            is_duplicate = True
                            break
                        
                        # Check word overlap for shorter texts
                        ai_words = set(ai_text.split())
                        existing_words = set(existing_text.split())
                        
                        if len(ai_words & existing_words) > max(3, len(ai_words) * 0.7):
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    combined.append(ai_rec)
        
        # Sort by ID if possible, maintaining structure
        def sort_key(rec):
            rec_id = rec.get('id', '').strip()
            
            # Handle numeric IDs
            if rec_id.isdigit():
                return (0, int(rec_id), '')
            
            # Handle alphanumeric IDs like "1a", "2b"
            match = re.match(r'^(\d+)([a-z])$', rec_id.lower())
            if match:
                return (1, int(match.group(1)), match.group(2))
            
            # Handle complex IDs like "4a) i)"
            match = re.match(r'^(\d+)([a-z])\)\s*([ivx]+)\)$', rec_id.lower())
            if match:
                roman_to_num = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10}
                roman_val = roman_to_num.get(match.group(3), 99)
                return (2, int(match.group(1)), match.group(2), roman_val)
            
            # Handle "Recommendation X" format
            match = re.match(r'^recommendation\s+(\d+)', rec_id.lower())
            if match:
                return (0, int(match.group(1)), '')
            
            # Everything else goes to the end
            return (99, rec_id)
        
        try:
            combined.sort(key=sort_key)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not sort recommendations: {e}")
            # Keep original order if sorting fails
        
        return combined

    def get_extraction_stats(self, results: Dict) -> Dict:
        """Get comprehensive statistics about the extraction"""
        recommendations = results.get('recommendations', [])
        
        stats = {
            'total_recommendations': len(recommendations),
            'by_type': {},
            'by_source': {},
            'by_extraction_method': {},
            'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
            'average_confidence': 0,
            'id_patterns': {
                'numeric': 0,
                'alphanumeric': 0,
                'complex': 0,
                'auto_generated': 0
            }
        }
        
        confidence_sum = 0
        
        for rec in recommendations:
            # Count by type
            rec_type = rec.get('type', 'unknown')
            stats['by_type'][rec_type] = stats['by_type'].get(rec_type, 0) + 1
            
            # Count by source
            source = rec.get('source', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            # Count by extraction method
            method = rec.get('extraction_method', 'unknown')
            stats['by_extraction_method'][method] = stats['by_extraction_method'].get(method, 0) + 1
            
            # Count by confidence level
            confidence = rec.get('confidence', 0.5)
            confidence_sum += confidence
            
            if confidence >= 0.8:
                stats['by_confidence']['high'] += 1
            elif confidence >= 0.6:
                stats['by_confidence']['medium'] += 1
            else:
                stats['by_confidence']['low'] += 1
            
            # Analyze ID patterns
            rec_id = str(rec.get('id', '')).strip()
            if rec_id.isdigit():
                stats['id_patterns']['numeric'] += 1
            elif re.match(r'^\d+[a-z]$', rec_id.lower()):
                stats['id_patterns']['alphanumeric'] += 1
            elif any(prefix in rec_id.upper() for prefix in ['REC_', 'AI_', 'GOV_', 'ACTION_']):
                stats['id_patterns']['auto_generated'] += 1
            else:
                stats['id_patterns']['complex'] += 1
        
        # Calculate average confidence
        if recommendations:
            stats['average_confidence'] = round(confidence_sum / len(recommendations), 2)
        
        return stats

    def validate_extraction(self, results: Dict) -> Dict:
        """Validate extraction results and provide quality metrics"""
        recommendations = results.get('recommendations', [])
        
        validation = {
            'is_valid': True,
            'issues': [],
            'quality_score': 0,
            'completeness_indicators': {
                'has_main_recommendations': False,
                'has_sub_recommendations': False,
                'has_government_responses': False,
                'has_action_items': False
            }
        }
        
        if not recommendations:
            validation['is_valid'] = False
            validation['issues'].append("No recommendations found")
            return validation
        
        # Check for different types of recommendations
        types_found = set()
        for rec in recommendations:
            rec_type = rec.get('type', '')
            types_found.add(rec_type)
            
            # Check individual recommendation quality
            text = rec.get('text', '')
            if len(text) < 10:
                validation['issues'].append(f"Very short recommendation text: '{text[:50]}'")
            
            if not rec.get('id'):
                validation['issues'].append("Recommendation missing ID")
            
            confidence = rec.get('confidence', 0)
            if confidence < 0.5:
                validation['issues'].append(f"Low confidence recommendation: {confidence}")
        
        # Update completeness indicators
        validation['completeness_indicators']['has_main_recommendations'] = 'main_recommendation' in types_found
        validation['completeness_indicators']['has_sub_recommendations'] = 'sub_recommendation' in types_found
        validation['completeness_indicators']['has_government_responses'] = 'government_response' in types_found
        validation['completeness_indicators']['has_action_items'] = 'action_item' in types_found
        
        # Calculate quality score (0-100)
        quality_factors = []
        
        # Factor 1: Number of recommendations found
        quality_factors.append(min(len(recommendations) * 5, 30))
        
        # Factor 2: Variety of types
        quality_factors.append(len(types_found) * 10)
        
        # Factor 3: Average confidence
        avg_confidence = sum(rec.get('confidence', 0.5) for rec in recommendations) / len(recommendations)
        quality_factors.append(avg_confidence * 30)
        
        # Factor 4: Completeness
        completeness_score = sum(validation['completeness_indicators'].values()) * 5
        quality_factors.append(completeness_score)
        
        validation['quality_score'] = min(sum(quality_factors), 100)
        
        return validation


# For backward compatibility with existing system
class Recommendation:
    """Simple recommendation data class for compatibility"""
    def __init__(self, id: str, text: str, document_source: str = "", 
                 section_title: str = "", page_number: int = None, 
                 confidence_score: float = 0.8, metadata: Dict = None):
        self.id = id
        self.text = text
        self.document_source = document_source
        self.section_title = section_title
        self.page_number = page_number
        self.confidence_score = confidence_score
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'document_source': self.document_source,
            'section_title': self.section_title,
            'page_number': self.page_number,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }


# Usage examples for different inquiry types:
if __name__ == "__main__":
    extractor = UKInquiryRecommendationExtractor()
    
    # Example 1: Infected Blood Inquiry
    ibi_text = """
    12. Giving effect to Recommendations of this Inquiry
    (a) Within the next 12 months, the Government should consider and either commit 
    to implementing the recommendations which I make, or give sufficient reason.
    (b) During that period, the Government should report back to Parliament.
    """
    
    # Example 2: Generic Public Inquiry
    generic_text = """
    Recommendation 1: The Department should establish a new oversight body.
    Recommendation 2: Regular reviews must be conducted annually.
    (a) The review should include stakeholder consultation.
    (b) Results should be published within 6 months.
    """
    
    # Example 3: Government Response
    response_text = """
    Recommendation 1 is accepted in full by the Government.
    Recommendation 2 is accepted in principle, subject to resource allocation.
    This recommendation will be implemented through existing frameworks.
    """
    
    # Test all examples
    for i, (name, text) in enumerate([
        ("Infected Blood Inquiry", ibi_text),
        ("Generic Public Inquiry", generic_text), 
        ("Government Response", response_text)
    ], 1):
        print(f"\n=== Test {i}: {name} ===")
        results = extractor.extract_recommendations(text, f"test_{name.lower().replace(' ', '_')}")
        stats = extractor.get_extraction_stats(results)
        validation = extractor.validate_extraction(results)
        
        print(f"Found {len(results['recommendations'])} recommendations")
        print(f"Quality Score: {validation['quality_score']}/100")
        print(f"Extraction stats: {stats}")
        
        # Show first recommendation as example
        if results['recommendations']:
            first_rec = results['recommendations'][0]
            print(f"Example: ID='{first_rec['id']}', Type='{first_rec['type']}', Text='{first_rec['text'][:100]}...'")
        
        if validation['issues']:
            print(f"Issues found: {validation['issues']}")
    
    print("\n=== Universal UK Inquiry Extractor Ready! ===")
    print("Compatible with: Public Health inquiries, Safety inquiries, Administrative reviews, etc.")
    print("Features: Dual AI+Pattern extraction, Quality validation, Comprehensive statistics")
