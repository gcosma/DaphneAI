# ===============================================
# FILE: modules/uk_inquiry_extractor.py (UPDATED FOR GOVERNMENT RESPONSES)
# ===============================================

import re
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
import os

class UKInquiryRecommendationExtractor:
    """
    Updated extractor for UK Government inquiry reports and responses
    Handles both original inquiry recommendations and government responses
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Updated patterns for government inquiry documents
        self.recommendation_patterns = {
            # Original inquiry recommendations
            'main_numbered': [
                r'(?i)(?:^|\n)\s*(?:Recommendation\s+)?(\d+)[\.:]\s*([^.\n]+(?:\.[^.\n]+)*)',
                r'(?i)(?:^|\n)\s*(\d+)[\.:]\s+((?:The|I|We|It)\s+(?:recommend|should|must|ought)[^.\n]+(?:\.[^.\n]+)*)',
                r'(?i)(?:^|\n)\s*(?:Recommendation\s+)?(\d+)[\s\-–]\s*([^.\n]+(?:\.[^.\n]+)*)'
            ],
            
            # Sub-recommendations with letters
            'sub_lettered': [
                r'(?i)\(([a-z])\)\s*([^.\n]+(?:\.[^.\n]+)*)',
                r'(?i)(?:^|\n)\s*([a-z])[\.:]\s*([^.\n]+(?:\.[^.\n]+)*)'
            ],
            
            # Government response patterns
            'response_statements': [
                r'(?i)(Recommendation\s+\d+)\s+(?:is\s+)?(accepted\s+in\s+full|accepted\s+in\s+principle|not\s+accepted|partially\s+accepted)([^.\n]+(?:\.[^.\n]+)*)',
                r'(?i)(Recommendation\s+\d+):\s*([^.\n]+(?:\.[^.\n]+)*?)(?=Recommendation\s+\d+|$)',
                r'(?i)(?:The\s+Government|We)\s+(accept[s]?|reject[s]?|agree[s]?|disagree[s]?)\s+([^.\n]+(?:\.[^.\n]+)*)'
            ],
            
            # Implementation statements
            'implementation': [
                r'(?i)(?:This\s+)?(?:recommendation\s+)?(?:will\s+be\s+|shall\s+be\s+|is\s+being\s+)?implemented\s+([^.\n]+(?:\.[^.\n]+)*)',
                r'(?i)(?:Action\s+)?(?:will\s+be\s+taken|has\s+been\s+taken)\s+([^.\n]+(?:\.[^.\n]+)*)',
                r'(?i)(?:Steps\s+)?(?:are\s+being\s+taken|have\s+been\s+taken)\s+([^.\n]+(?:\.[^.\n]+)*)'
            ]
        }
        
        # Keywords that help identify document types
        self.document_type_indicators = {
            'inquiry_report': [
                'inquiry recommendations', 'final report', 'findings and recommendations',
                'chair of the inquiry', 'panel recommendations', 'inquiry findings'
            ],
            'government_response': [
                'government response', 'official response', 'response to the inquiry',
                'presented to parliament', 'cabinet office', 'minister for'
            ]
        }
        
        # Setup OpenAI if available
        self.openai_client = None
        if os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            except Exception as e:
                self.logger.warning(f"OpenAI setup failed: {e}")

    def extract_recommendations(self, document_text: str, document_source: str = "") -> Dict[str, Any]:
        """
        Extract recommendations and responses using methods optimized for UK inquiry documents
        """
        try:
            # Determine document type
            doc_type = self._determine_document_type(document_text)
            
            # Clean and normalize text
            cleaned_text = self._clean_text(document_text)
            
            # Extract using both methods
            ai_results = self._extract_with_ai(cleaned_text, doc_type) if self.openai_client else []
            pattern_results = self._extract_with_patterns(cleaned_text, document_source, doc_type)
            
            # Combine and deduplicate results
            combined_results = self._combine_results(ai_results, pattern_results)
            
            return {
                'recommendations': combined_results,
                'extraction_info': {
                    'document_type': doc_type,
                    'ai_found': len(ai_results) if ai_results else 0,
                    'pattern_found': len(pattern_results),
                    'total_combined': len(combined_results),
                    'document_source': document_source,
                    'extraction_date': datetime.now().isoformat(),
                    'extractor_version': 'UK_Inquiry_v2.0_Gov_Response'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return {'recommendations': [], 'extraction_info': {'error': str(e)}}

    def _determine_document_type(self, text: str) -> str:
        """Determine if document is original inquiry or government response"""
        text_lower = text.lower()
        
        inquiry_score = sum(1 for keyword in self.document_type_indicators['inquiry_report'] 
                           if keyword in text_lower)
        
        response_score = sum(1 for keyword in self.document_type_indicators['government_response'] 
                            if keyword in text_lower)
        
        if response_score > inquiry_score:
            return 'government_response'
        elif inquiry_score > 0:
            return 'inquiry_report'
        else:
            return 'unknown'

    def _extract_with_ai(self, text: str, doc_type: str) -> List[Dict]:
        """Extract using AI with document-type-specific prompts"""
        if not self.openai_client:
            return []
        
        try:
            # Create document-type-specific prompt
            if doc_type == 'government_response':
                prompt = self._create_response_extraction_prompt(text)
            else:
                prompt = self._create_recommendation_extraction_prompt(text)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            recommendations = json.loads(content)
            
            # Add AI source marker
            for rec in recommendations:
                rec['source'] = 'ai_extraction'
                rec['extraction_method'] = f'ai_{doc_type}'
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"AI extraction failed: {e}")
            return []

    def _create_recommendation_extraction_prompt(self, text: str) -> str:
        """Create prompt for extracting original inquiry recommendations"""
        return f"""
TASK: Extract ALL recommendations from this UK Government inquiry report.

DOCUMENT TYPE: Original inquiry report containing recommendations

LOOK FOR:
1. **Numbered recommendations**: "Recommendation 1:", "1.", "Recommendation 15:"
2. **Sub-recommendations**: "(a)", "(b)", "(i)", "(ii)"
3. **Action statements**: "should establish", "must implement", "ought to review"
4. **Inquiry language**: "I recommend that", "The inquiry recommends", "We recommend"

For each recommendation found, extract:
- The recommendation ID/number
- The full text of the recommendation
- The type (main recommendation, sub-recommendation, etc.)

Return as JSON array:
[
    {{
        "id": "1",
        "text": "The full recommendation text",
        "type": "main_recommendation",
        "theme": "topic if identifiable",
        "confidence": 0.95
    }}
]

Document text:
{text[:3000]}
"""

    def _create_response_extraction_prompt(self, text: str) -> str:
        """Create prompt for extracting government responses"""
        return f"""
TASK: Extract ALL government responses to recommendations from this document.

DOCUMENT TYPE: Government response to inquiry recommendations

LOOK FOR:
1. **Response statements**: "Recommendation 1 is accepted in full", "accepted in principle", "not accepted"
2. **Implementation plans**: "will be implemented through", "action will be taken"
3. **Government commitments**: "The Government will", "We will ensure"
4. **Status updates**: "has been implemented", "is being implemented", "will be reviewed"

For each response found, extract:
- Which recommendation it responds to
- The government's acceptance/rejection
- Implementation details

Return as JSON array:
[
    {{
        "id": "Response_1",
        "text": "The full response text",
        "type": "government_response",
        "recommendation_reference": "1",
        "response_type": "accepted_in_full",
        "confidence": 0.90
    }}
]

Document text:
{text[:3000]}
"""

    def _extract_with_patterns(self, text: str, source: str, doc_type: str) -> List[Dict]:
        """Extract using regex patterns optimized for document type"""
        recommendations = []
        
        if doc_type == 'government_response':
            # Extract government responses
            recommendations.extend(self._extract_response_patterns(text, source))
        else:
            # Extract original recommendations
            recommendations.extend(self._extract_recommendation_patterns(text, source))
        
        return recommendations

    def _extract_recommendation_patterns(self, text: str, source: str) -> List[Dict]:
        """Extract original inquiry recommendations using patterns"""
        recommendations = []
        
        # Extract main numbered recommendations
        for pattern in self.recommendation_patterns['main_numbered']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            for match in matches:
                rec_id = match.group(1)
                rec_text = match.group(2).strip()
                
                if len(rec_text) > 20 and self._is_valid_recommendation(rec_text):
                    recommendations.append({
                        'id': rec_id,
                        'text': rec_text,
                        'type': 'inquiry_recommendation',
                        'source': 'pattern_extraction',
                        'document_source': source,
                        'confidence': 0.9,
                        'extraction_method': 'numbered_pattern'
                    })
        
        # Extract sub-recommendations
        for pattern in self.recommendation_patterns['sub_lettered']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                rec_id = match.group(1)
                rec_text = match.group(2).strip()
                
                if len(rec_text) > 30 and self._is_valid_recommendation(rec_text):
                    recommendations.append({
                        'id': f"sub_{rec_id}",
                        'text': rec_text,
                        'type': 'sub_recommendation',
                        'source': 'pattern_extraction',
                        'document_source': source,
                        'confidence': 0.85,
                        'extraction_method': 'lettered_pattern'
                    })
        
        return recommendations

    def _extract_response_patterns(self, text: str, source: str) -> List[Dict]:
        """Extract government responses using patterns"""
        responses = []
        
        # Extract response statements
        for pattern in self.recommendation_patterns['response_statements']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.groups()) >= 2:
                    rec_ref = match.group(1) if len(match.groups()) > 2 else "unknown"
                    response_text = match.group(-1).strip()  # Last group
                    
                    # Determine response type
                    response_type = self._classify_response_type(response_text)
                    
                    if len(response_text) > 20:
                        responses.append({
                            'id': f"response_{rec_ref}",
                            'text': response_text,
                            'type': 'government_response',
                            'recommendation_reference': rec_ref,
                            'response_type': response_type,
                            'source': 'pattern_extraction',
                            'document_source': source,
                            'confidence': 0.85,
                            'extraction_method': 'response_pattern'
                        })
        
        # Extract implementation statements
        for pattern in self.recommendation_patterns['implementation']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                impl_text = match.group(1).strip()
                
                if len(impl_text) > 20:
                    responses.append({
                        'id': f"implementation_{len(responses)}",
                        'text': impl_text,
                        'type': 'implementation_plan',
                        'source': 'pattern_extraction',
                        'document_source': source,
                        'confidence': 0.80,
                        'extraction_method': 'implementation_pattern'
                    })
        
        return responses

    def _classify_response_type(self, text: str) -> str:
        """Classify the type of government response"""
        text_lower = text.lower()
        
        if 'accepted in full' in text_lower or 'accept in full' in text_lower:
            return 'accepted_in_full'
        elif 'accepted in principle' in text_lower or 'accept in principle' in text_lower:
            return 'accepted_in_principle'
        elif 'not accepted' in text_lower or 'reject' in text_lower:
            return 'not_accepted'
        elif 'partially accepted' in text_lower:
            return 'partially_accepted'
        elif 'under consideration' in text_lower or 'being reviewed' in text_lower:
            return 'under_review'
        else:
            return 'general_response'

    def _is_valid_recommendation(self, text: str) -> bool:
        """Validate if extracted text is a genuine recommendation"""
        text = text.lower().strip()
        
        # Skip if too short or just a title
        if len(text) < 15:
            return False
        
        # Skip common false positives
        skip_phrases = [
            'table of contents', 'index', 'page ', 'see section', 'see chapter',
            'see appendix', 'figure ', 'diagram ', 'chart ', 'graph ',
            'copyright', '©', 'all rights reserved'
        ]
        
        if any(phrase in text for phrase in skip_phrases):
            return False
        
        # Look for recommendation indicators
        positive_indicators = [
            'should', 'must', 'ought', 'recommend', 'ensure', 'establish',
            'implement', 'develop', 'create', 'review', 'consider', 'examine',
            'accepted', 'rejected', 'will be', 'has been', 'government', 'department'
        ]
        
        return any(indicator in text for indicator in positive_indicators)

    def _combine_results(self, ai_results: List[Dict], pattern_results: List[Dict]) -> List[Dict]:
        """Combine and deduplicate results from AI and pattern extraction"""
        combined = []
        seen_texts = set()
        
        # Add AI results first (higher priority)
        for rec in ai_results or []:
            text_key = self._normalize_text_for_comparison(rec.get('text', ''))
            if text_key not in seen_texts and len(text_key) > 10:
                seen_texts.add(text_key)
                combined.append(rec)
        
        # Add pattern results that aren't duplicates
        for rec in pattern_results:
            text_key = self._normalize_text_for_comparison(rec.get('text', ''))
            if text_key not in seen_texts and len(text_key) > 10:
                seen_texts.add(text_key)
                combined.append(rec)
        
        return combined

    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normalize text for duplicate detection"""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove common prefixes/suffixes that might vary
        normalized = re.sub(r'^(?:recommendation\s+\d+[:\-\.]?\s*)', '', normalized)
        normalized = re.sub(r'^\d+[:\-\.]?\s*', '', normalized)
        
        return normalized[:100]  # First 100 chars for comparison

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
        
        # Fix common formatting issues
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        
        return text.strip()

    def get_extraction_stats(self, results: Dict) -> Dict[str, Any]:
        """Get detailed statistics about the extraction"""
        recommendations = results.get('recommendations', [])
        
        stats = {
            'total_recommendations': len(recommendations),
            'by_type': {},
            'by_source': {},
            'by_response_type': {},
            'confidence_distribution': {},
            'average_confidence': 0,
            'extraction_info': results.get('extraction_info', {})
        }
        
        if not recommendations:
            return stats
        
        # Count by type
        for rec in recommendations:
            rec_type = rec.get('type', 'unknown')
            stats['by_type'][rec_type] = stats['by_type'].get(rec_type, 0) + 1
            
            source = rec.get('source', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            # For government responses, track response type
            if rec.get('response_type'):
                resp_type = rec['response_type']
                stats['by_response_type'][resp_type] = stats['by_response_type'].get(resp_type, 0) + 1
        
        # Confidence statistics
        confidences = [rec.get('confidence', 0) for rec in recommendations]
        if confidences:
            stats['average_confidence'] = sum(confidences) / len(confidences)
            
            # Distribution
            for conf in confidences:
                bucket = f"{int(conf * 10) * 10}-{int(conf * 10) * 10 + 9}%"
                stats['confidence_distribution'][bucket] = stats['confidence_distribution'].get(bucket, 0) + 1
        
        return stats

    def validate_extraction(self, results: Dict) -> Dict[str, Any]:
        """Validate extraction quality and provide feedback"""
        recommendations = results.get('recommendations', [])
        extraction_info = results.get('extraction_info', {})
        
        validation = {
            'is_valid': False,
            'quality_score': 0,
            'issues': [],
            'recommendations': [],
            'document_type': extraction_info.get('document_type', 'unknown')
        }
        
        if not recommendations:
            validation['issues'].append("No recommendations found")
            return validation
        
        # Check recommendation quality
        valid_recs = 0
        for rec in recommendations:
            text_length = len(rec.get('text', ''))
            confidence = rec.get('confidence', 0)
            
            if text_length >= 20 and confidence >= 0.5:
                valid_recs += 1
        
        # Calculate quality score (0-100)
        if recommendations:
            length_score = min(100, sum(min(100, len(rec.get('text', ''))) for rec in recommendations) / len(recommendations))
            confidence_score = sum(rec.get('confidence', 0) for rec in recommendations) / len(recommendations) * 100
            validity_score = (valid_recs / len(recommendations)) * 100
            
            validation['quality_score'] = (length_score + confidence_score + validity_score) / 3
        
        # Provide recommendations
        if validation['quality_score'] < 50:
            validation['recommendations'].append("Consider trying AI extraction if pattern extraction is insufficient")
        
        if len(recommendations) > 50:
            validation['recommendations'].append("Large number of recommendations found - review for duplicates")
        
        validation['is_valid'] = validation['quality_score'] >= 30 and len(recommendations) > 0
        
        return validation


# Test function for the updated extractor
def test_extraction():
    """Test the updated extractor with sample government response text"""
    extractor = UKInquiryRecommendationExtractor()
    
    # Sample government response text
    sample_text = """
    Government Response to the Infected Blood Inquiry
    
    Recommendation 1: The Department of Health should establish a new monitoring system.
    This recommendation is accepted in full by the Government. Implementation will begin immediately through existing NHS structures.
    
    Recommendation 2: Regular reviews must be conducted annually.
    This recommendation is accepted in principle. The Government will establish a review process within 12 months.
    
    Recommendation 3: All staff should receive additional training.
    This recommendation is not accepted as current training programmes are deemed sufficient.
    """
    
    results = extractor.extract_recommendations(sample_text, "test_government_response")
    stats = extractor.get_extraction_stats(results)
    validation = extractor.validate_extraction(results)
    
    print(f"Found {len(results['recommendations'])} items")
    print(f"Quality Score: {validation['quality_score']:.1f}/100")
    print(f"Document Type: {validation['document_type']}")
    
    for rec in results['recommendations']:
        print(f"- {rec['type']}: {rec['text'][:80]}...")
    
    return results

if __name__ == "__main__":
    test_extraction()
