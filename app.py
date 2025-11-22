# Fixed app.py - DaphneAI Government Document Analysis
import streamlit as st
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Any
import logging
import traceback
from collections import Counter

# ry to import NLTK
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ===== RECOMMENDATION EXTRACTOR CODE =====

"""
Improved Recommendation Extractor for UK COVID-19 Inquiry Documents

This module provides an enhanced approach to extracting recommendations from 
policy documents, handling both explicit numbered recommendations and 
implicit recommendations embedded in prose.

Author: Prof. Georgina Brown
Date: November 2024
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter

# Optional NLP support
try:
    from nltk import word_tokenize, pos_tag
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


class RecommendationType(Enum):
    """Types of recommendations that can be detected"""
    NUMBERED = "numbered"           # Explicitly numbered recommendations
    GERUND = "gerund"              # Starting with -ing verbs (Improving, Establishing, etc.)
    MODAL = "modal"                # Using modal verbs (should, must, need to)
    IMPERATIVE = "imperative"      # Direct commands (Ensure, Establish, etc.)
    THERE_IS = "there_is"          # "There is a need to..." patterns
    KEYWORD = "keyword"            # Explicit recommendation phrases


@dataclass
class Recommendation:
    """A detected recommendation with metadata"""
    text: str
    type: RecommendationType
    confidence: float
    verb: str = "unknown"
    line_number: int = None
    position: int = None
    section: str = None
    
    def __repr__(self):
        return f"Recommendation(type={self.type.value}, confidence={self.confidence:.2f}, verb='{self.verb}', text='{self.text[:60]}...')"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'text': self.text,
            'type': self.type.value,
            'confidence': self.confidence,
            'verb': self.verb,
            'line_number': self.line_number,
            'position': self.position,
            'section': self.section
        }


class RecommendationExtractor:
    """
    Enhanced extractor for recommendations from policy documents.
    
    Features:
    - Multiple detection methods (numbered, gerund, modal, imperative, keyword)
    - Confidence scoring
    - False positive filtering
    - Context-aware extraction
    - Verb extraction and statistics
    """
    
    def __init__(self):
        """Initialise the extractor with pattern definitions"""
        
        # High-confidence gerund verbs for policy recommendations
        self.gerund_verbs = {
            'improving', 'establishing', 'ensuring', 'enabling', 'broadening',
            'reforming', 'clarifying', 'implementing', 'developing', 'strengthening',
            'enhancing', 'creating', 'building', 'maintaining', 'providing',
            'supporting', 'promoting', 'facilitating', 'introducing', 'expanding',
            'reviewing', 'updating', 'reducing', 'addressing', 'adopting',
            'requiring', 'encouraging'
        }
        
        # Imperative verbs that start recommendations
        self.imperative_verbs = {
            'ensure', 'establish', 'improve', 'create', 'develop', 'implement',
            'strengthen', 'enhance', 'provide', 'support', 'maintain', 'promote',
            'facilitate', 'introduce', 'expand', 'reform', 'clarify', 'review',
            'update', 'address', 'adopt', 'require', 'encourage'
        }
        
        # Action verbs that typically follow modals in recommendations
        self.action_verbs = {
            'implement', 'establish', 'ensure', 'improve', 'develop', 'create',
            'enhance', 'introduce', 'adopt', 'provide', 'strengthen', 'facilitate',
            'support', 'maintain', 'expand', 'review', 'update', 'address'
        }
        
        # Modal patterns (high confidence)
        self.modal_patterns = [
            r'\bmust\b',
            r'\bshould\b',
            r'\bneed to\b',
            r'\brequire[sd]?\b',
            r'\bis essential to\b',
            r'\bis critical to\b',
            r'\bis important to\b'
        ]
        
        # Modal + action verb patterns (more specific)
        self.modal_action_patterns = [
            r'\bshould\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
            r'\bmust\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
            r'\bneed to\s+(?:implement|establish|ensure|improve|develop|create|enhance|introduce|adopt|provide)',
        ]
        
        # "There is" patterns
        self.there_is_patterns = [
            r'there (?:is|are|remains?) (?:a |an )?(?:urgent )?need',
            r'there (?:is|are) (?:a )?lack of',
            r'there should be'
        ]
        
        # Explicit recommendation phrases
        self.recommendation_patterns = [
            r'we recommend\b',
            r'it is recommended\b',
            r'the inquiry recommends?\b',
            r'the committee recommends?\b',
            r'the report recommends?\b',
            r'(?:this|the) recommendation',
        ]
        
        # Phrases to exclude (false positive indicators)
        self.exclude_phrases = {
            'when deciding', 'before implementing', 'after establishing',
            'while ensuring', 'without improving', 'if reforming',
            'by clarifying', 'through enabling', 'during broadening',
            'was establishing', 'were improving', 'had been ensuring',
            'has been', 'have been', 'had been',  # Past tense indicators
            'will be', 'would be',  # Conditional/future (less certain)
            'for example', 'for instance', 'such as',  # Examples
            'the need to understand', 'the need to consider'  # Analysis, not action
        }
        
        # Section headers that indicate recommendation sections
        self.recommendation_sections = {
            'recommendation', 'action', 'proposal', 'should', 'must',
            'key issue', 'future', 'lesson', 'improvement'
        }
    
    def extract_recommendations(self, text: str, min_confidence: float = 0.7) -> List[Recommendation]:
        """
        Extract all recommendations from text with confidence scores.
        
        Args:
            text: The document text to analyse
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            List of Recommendation objects sorted by confidence
        """
        recommendations = []
        
        # Split into lines for line-based processing
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and very short lines
            if len(line) < 20:
                continue
            
            # Try each detection method
            recs = []
            
            # Method 1: Numbered recommendations
            recs.extend(self._detect_numbered(line, line_num))
            
            # Method 2: Gerund-based recommendations (PRIMARY METHOD)
            recs.extend(self._detect_gerund(line, line_num))
            
            # Method 3: Modal verb recommendations
            recs.extend(self._detect_modal(line, line_num))
            
            # Method 4: Imperative recommendations
            recs.extend(self._detect_imperative(line, line_num))
            
            # Method 5: "There is a need to..." patterns
            recs.extend(self._detect_there_is(line, line_num))
            
            # Method 6: Explicit recommendation keywords
            recs.extend(self._detect_keyword(line, line_num))
            
            # Filter by confidence
            for rec in recs:
                if rec.confidence >= min_confidence:
                    recommendations.append(rec)
        
        # Also process by sentences for better context
        sentences = self._split_sentences(text)
        for pos, sentence in enumerate(sentences):
            # Only process if not already captured by line processing
            if not any(self._similarity(sentence, r.text) > 0.8 for r in recommendations):
                recs = []
                
                # Apply same methods to sentences
                recs.extend(self._detect_gerund_sentence(sentence, pos))
                recs.extend(self._detect_modal_sentence(sentence, pos))
                recs.extend(self._detect_keyword_sentence(sentence, pos))
                
                for rec in recs:
                    if rec.confidence >= min_confidence:
                        recommendations.append(rec)
        
        # Remove duplicates and sort by confidence
        recommendations = self._deduplicate(recommendations)
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations
    
    def _detect_numbered(self, line: str, line_num: int) -> List[Recommendation]:
        """Detect explicitly numbered recommendations"""
        recs = []
        
        # Pattern: Number followed by period or parenthesis at start of line
        numbered_pattern = r'^\s*(\d+)[.)]\s+(.+)$'
        match = re.match(numbered_pattern, line)
        
        if match:
            number, text = match.groups()
            
            # Check if it's actually a recommendation (not just a list item)
            if self._is_recommendation_content(text):
                verb = self._extract_main_verb(text)
                recs.append(Recommendation(
                    text=text.strip(),
                    type=RecommendationType.NUMBERED,
                    confidence=0.95,
                    verb=verb,
                    line_number=line_num
                ))
        
        return recs
    
    def _detect_gerund(self, line: str, line_num: int) -> List[Recommendation]:
        """Detect gerund-based recommendations (Improving..., Establishing...)"""
        recs = []
        
        # Check if line starts with a gerund verb
        words = line.split()
        if not words:
            return recs
        
        first_word = words[0].lower().rstrip(':,.')
        
        if first_word in self.gerund_verbs:
            # Check it's not a false positive
            if not self._is_false_positive(line):
                # Extract verb (remove -ing)
                verb = first_word[:-3] if first_word.endswith('ing') else first_word
                recs.append(Recommendation(
                    text=line.strip(),
                    type=RecommendationType.GERUND,
                    confidence=0.90,
                    verb=verb,
                    line_number=line_num
                ))
        
        return recs
    
    def _detect_gerund_sentence(self, sentence: str, position: int) -> List[Recommendation]:
        """Detect gerund-based recommendations from sentences"""
        recs = []
        
        words = sentence.split()
        if not words or len(sentence) < 30:
            return recs
        
        first_word = words[0].strip('.,;:!?"\'').lower()
        
        if first_word in self.gerund_verbs:
            if not self._is_false_positive(sentence):
                verb = first_word[:-3] if first_word.endswith('ing') else first_word
                recs.append(Recommendation(
                    text=sentence.strip(),
                    type=RecommendationType.GERUND,
                    confidence=0.90,
                    verb=verb,
                    position=position
                ))
        
        return recs
    
    def _detect_modal(self, line: str, line_num: int) -> List[Recommendation]:
        """Detect modal verb recommendations (should, must, need to)"""
        recs = []
        
        line_lower = line.lower()
        
        # First check for modal + action patterns (higher confidence)
        for pattern in self.modal_action_patterns:
            if re.search(pattern, line_lower):
                if not self._is_false_positive(line):
                    verb = self._extract_main_verb(line)
                    recs.append(Recommendation(
                        text=line.strip(),
                        type=RecommendationType.MODAL,
                        confidence=0.85,
                        verb=verb,
                        line_number=line_num
                    ))
                    return recs  # Only add once per line
        
        # Then check general modal patterns (lower confidence)
        for pattern in self.modal_patterns:
            if re.search(pattern, line_lower):
                if not self._is_false_positive(line) and len(line.split()) >= 8:
                    verb = self._extract_main_verb(line)
                    confidence = 0.75
                    
                    # Higher confidence for "must" and "should" at sentence start
                    if re.match(r'^\s*(?:we |governments? |the uk )?(?:must|should)', line_lower):
                        confidence = 0.80
                    
                    recs.append(Recommendation(
                        text=line.strip(),
                        type=RecommendationType.MODAL,
                        confidence=confidence,
                        verb=verb,
                        line_number=line_num
                    ))
                    break  # Only add once per line
        
        return recs
    
    def _detect_modal_sentence(self, sentence: str, position: int) -> List[Recommendation]:
        """Detect modal recommendations from sentences"""
        recs = []
        
        if len(sentence) < 30:
            return recs
        
        sentence_lower = sentence.lower()
        
        # Check modal + action patterns
        for pattern in self.modal_action_patterns:
            if re.search(pattern, sentence_lower):
                if not self._is_false_positive(sentence):
                    verb = self._extract_main_verb(sentence)
                    recs.append(Recommendation(
                        text=sentence.strip(),
                        type=RecommendationType.MODAL,
                        confidence=0.85,
                        verb=verb,
                        position=position
                    ))
                    return recs
        
        return recs
    
    def _detect_imperative(self, line: str, line_num: int) -> List[Recommendation]:
        """Detect imperative recommendations (Ensure..., Establish...)"""
        recs = []
        
        words = line.split()
        if not words:
            return recs
        
        first_word = words[0].lower().rstrip(':,.')
        
        if first_word in self.imperative_verbs:
            # Check it's not a false positive
            if not self._is_false_positive(line):
                recs.append(Recommendation(
                    text=line.strip(),
                    type=RecommendationType.IMPERATIVE,
                    confidence=0.85,
                    verb=first_word,
                    line_number=line_num
                ))
        
        return recs
    
    def _detect_there_is(self, line: str, line_num: int) -> List[Recommendation]:
        """Detect 'There is a need to...' style recommendations"""
        recs = []
        
        line_lower = line.lower()
        
        for pattern in self.there_is_patterns:
            if re.search(pattern, line_lower):
                # Check it's not a false positive
                if not self._is_false_positive(line):
                    verb = self._extract_main_verb(line)
                    recs.append(Recommendation(
                        text=line.strip(),
                        type=RecommendationType.THERE_IS,
                        confidence=0.80,
                        verb=verb,
                        line_number=line_num
                    ))
                    break
        
        return recs
    
    def _detect_keyword(self, line: str, line_num: int) -> List[Recommendation]:
        """Detect explicit recommendation keywords"""
        recs = []
        
        line_lower = line.lower()
        
        for pattern in self.recommendation_patterns:
            if re.search(pattern, line_lower):
                # Must be substantial
                if len(line.split()) >= 5 and not self._is_false_positive(line):
                    verb = self._extract_main_verb(line)
                    recs.append(Recommendation(
                        text=line.strip(),
                        type=RecommendationType.KEYWORD,
                        confidence=0.85,
                        verb=verb,
                        line_number=line_num
                    ))
                    break
        
        return recs
    
    def _detect_keyword_sentence(self, sentence: str, position: int) -> List[Recommendation]:
        """Detect explicit recommendation keywords in sentences"""
        recs = []
        
        if len(sentence) < 30:
            return recs
        
        sentence_lower = sentence.lower()
        
        for pattern in self.recommendation_patterns:
            if re.search(pattern, sentence_lower):
                if not self._is_false_positive(sentence):
                    verb = self._extract_main_verb(sentence)
                    recs.append(Recommendation(
                        text=sentence.strip(),
                        type=RecommendationType.KEYWORD,
                        confidence=0.85,
                        verb=verb,
                        position=position
                    ))
                    break
        
        return recs
    
    def _is_recommendation_content(self, text: str) -> bool:
        """Check if numbered item is actually a recommendation"""
        text_lower = text.lower()
        
        # Look for recommendation indicators
        indicators = ['should', 'must', 'need', 'require', 'ensure', 
                     'establish', 'improve', 'create', 'develop']
        
        return any(ind in text_lower for ind in indicators)
    
    def _is_false_positive(self, line: str) -> bool:
        """Check if line is a false positive"""
        line_lower = line.lower()
        
        # Check for excluding phrases
        for phrase in self.exclude_phrases:
            if phrase in line_lower:
                return True
        
        # Check if it's a question (unlikely to be a recommendation)
        if '?' in line:
            return True
        
        # Check if it's a citation or reference without recommendation content
        if re.search(r'\[\w+\s+\d{4}\]|\(\d{4}\)', line):
            if not any(verb in line_lower for verb in self.gerund_verbs):
                return True
        
        return False
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Better sentence splitting that preserves sentence integrity
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Only keep sentences that are substantial (more than 30 chars and have multiple words)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 30 and len(s.split()) >= 5]
    
    def _extract_main_verb(self, text: str) -> str:
        """Extract the main verb from text"""
        if NLP_AVAILABLE:
            try:
                tokens = word_tokenize(text[:200])  # Limit length for performance
                tagged = pos_tag(tokens)
                
                # Find action verbs (not auxiliaries)
                verbs = [word.lower() for word, pos in tagged if pos.startswith('VB')]
                
                if verbs:
                    auxiliaries = {'be', 'is', 'are', 'was', 'were', 'been', 'being', 
                                 'have', 'has', 'had', 'do', 'does', 'did'}
                    for verb in verbs:
                        if verb not in auxiliaries:
                            return verb
                    return verbs[0]
            except:
                pass
        
        # Fallback: look for common recommendation verbs
        text_lower = text.lower()
        
        # Check action verbs first
        for verb in self.action_verbs:
            if verb in text_lower:
                return verb
        
        # Then check gerund verbs (remove -ing)
        for verb in self.gerund_verbs:
            if verb in text_lower:
                return verb[:-3] if verb.endswith('ing') else verb
        
        # Check first word if it's a verb
        words = text.split()
        if words:
            first_word = words[0].lower().rstrip(':,.')
            if first_word in self.gerund_verbs or first_word in self.imperative_verbs:
                return first_word[:-3] if first_word.endswith('ing') else first_word
        
        return 'unknown'
    
    def _deduplicate(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations based on text similarity"""
        unique_recs = []
        seen_texts = set()
        
        for rec in recommendations:
            # Normalise text for comparison
            normalised = rec.text.lower().strip()
            normalised = re.sub(r'\s+', ' ', normalised)
            
            # Check if we've seen this text (or very similar)
            is_duplicate = False
            for seen in seen_texts:
                # If texts are very similar (>80% overlap), consider duplicate
                if self._similarity(normalised, seen) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_recs.append(rec)
                seen_texts.add(normalised)
        
        return unique_recs
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (0-1)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_verb_statistics(self, recommendations: List[Recommendation]) -> Dict:
        """Get statistics about the verbs used"""
        verb_counts = Counter(r.verb for r in recommendations)
        method_counts = Counter(r.type.value for r in recommendations)
        
        return {
            'total': len(recommendations),
            'unique_verbs': len(verb_counts),
            'verb_frequency': dict(verb_counts.most_common()),
            'method_distribution': dict(method_counts),
            'avg_confidence': sum(r.confidence for r in recommendations) / len(recommendations) if recommendations else 0
        }
    
    def extract_from_file(self, filepath: str, min_confidence: float = 0.7) -> Dict:
        """
        Extract recommendations from a PDF or text file.
        
        Args:
            filepath: Path to the document
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with recommendations and metadata
        """
        import subprocess
        
        # Extract text from PDF
        if filepath.endswith('.pdf'):
            result = subprocess.run(
                ['pdftotext', '-layout', filepath, '-'],
                capture_output=True,
                text=True
            )
            text = result.stdout
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Extract recommendations
        recommendations = self.extract_recommendations(text, min_confidence)
        
        # Group by type
        by_type = {}
        for rec in recommendations:
            rec_type = rec.type.value
            if rec_type not in by_type:
                by_type[rec_type] = []
            by_type[rec_type].append(rec)
        
        # Get verb statistics
        verb_stats = self.get_verb_statistics(recommendations)
        
        return {
            'total_count': len(recommendations),
            'by_type': {k: len(v) for k, v in by_type.items()},
            'recommendations': recommendations,
            'grouped': by_type,
            'verb_statistics': verb_stats
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate a formatted report of extracted recommendations"""
        lines = []
        lines.append("=" * 80)
        lines.append("RECOMMENDATION EXTRACTION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTotal recommendations found: {results['total_count']}")
        lines.append("\nBreakdown by type:")
        
        for rec_type, count in results['by_type'].items():
            lines.append(f"  - {rec_type.upper()}: {count}")
        
        if 'verb_statistics' in results:
            lines.append(f"\nUnique verbs found: {results['verb_statistics']['unique_verbs']}")
            lines.append(f"Average confidence: {results['verb_statistics']['avg_confidence']:.2f}")
            lines.append("\nTop verbs:")
            for verb, count in list(results['verb_statistics']['verb_frequency'].items())[:10]:
                lines.append(f"  - {verb}: {count}")
        
        lines.append("\n" + "=" * 80)
        lines.append("DETAILED RECOMMENDATIONS")
        lines.append("=" * 80)
        
        for i, rec in enumerate(results['recommendations'], 1):
            lines.append(f"\n[{i}] {rec.type.value.upper()} (confidence: {rec.confidence:.2f}, verb: {rec.verb})")
            if rec.line_number:
                lines.append(f"    Line: {rec.line_number}")
            if rec.position is not None:
                lines.append(f"    Position: {rec.position}")
            lines.append(f"    {rec.text}")
        
        return '\n'.join(lines)


def extract_recommendations_simple(text: str, min_confidence: float = 0.7) -> List[Dict]:
    """Simple function to extract recommendations from text"""
    extractor = RecommendationExtractor()
    recommendations = extractor.extract_recommendations(text, min_confidence)
    return [rec.to_dict() for rec in recommendations]


# Example usage
if __name__ == "__main__":
    import sys
    
    extractor = RecommendationExtractor()
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        results = extractor.extract_from_file(filepath, min_confidence=0.7)
        report = extractor.generate_report(results)
        print(report)
    else:
        # Test with example text
        test_text = """
        Specific recommendations
        
        1. Improving consideration of the impact that decisions might have on those most at risk.
        
        2. Broadening participation in SAGE through open recruitment of experts.
        
        Reforming and clarifying the structures for decision-making during emergencies.
        
        Governments must act swiftly and decisively to stop virus spread.
        
        There is a need to continue Long Covid services including clinics.
        
        The inquiry recommends establishing better communication channels.
        
        The pandemic was unprecedented in scale.
        """
        
        recs = extractor.extract_recommendations(test_text, min_confidence=0.7)
        
        print(f"Found {len(recs)} recommendations:\n")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. [{rec.type.value}] (verb: {rec.verb}) {rec.text}")
            print(f"   Confidence: {rec.confidence:.2f}\n")
        
        # Show verb statistics
        stats = extractor.get_verb_statistics(recs)
        print("\nVerb Statistics:")
        print(f"Total: {stats['total']}")
        print(f"Unique verbs: {stats['unique_verbs']}")
        print(f"Verb frequency: {stats['verb_frequency']}")


# ===== END RECOMMENDATION EXTRACTOR CODE =====

def safe_import_with_fallback():
    """Safely import modules with comprehensive fallbacks"""
    try:
        from modules.integration_helper import (
            setup_search_tab, 
            prepare_documents_for_search, 
            extract_text_from_file,
            render_analytics_tab
        )
        return True, setup_search_tab, prepare_documents_for_search, extract_text_from_file, render_analytics_tab
    except ImportError as e:
        logger.warning(f"Import error: {e}")
        return False, None, None, None, None

def main():
    """Main application with enhanced error handling"""
    try:
        st.set_page_config(
            page_title="DaphneAI - Government Document Analysis", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏛️ DaphneAI - Government Document Analysis")
        st.markdown("*Advanced document processing and search for government content*")
        
        # Check module availability
        modules_available, setup_search_tab, prepare_documents_for_search, extract_text_from_file, render_analytics_tab = safe_import_with_fallback()
        
        if not modules_available:
            render_fallback_interface()
            return
        
        # Enhanced tabs with error handling
        try:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📁 Upload", 
                "🔍 Extract", 
                "🔍 Search",
                "🔗 Align Rec-Resp",
                "📊 Analytics",
                "🎯 Recommendations"  # NEW - This is tab6
            ])
            
            with tab1:
                render_upload_tab_safe(prepare_documents_for_search, extract_text_from_file)
            
            with tab2:
                render_extract_tab_safe()
            
            with tab3:
                render_search_tab_safe(setup_search_tab)
            
            with tab4:
                render_alignment_tab_safe()
            
            with tab5:
                render_analytics_tab_safe(render_analytics_tab)
            
            with tab6:  # NEW - RECOMMENDATIONS TAB
                render_recommendations_tab()
                
        except Exception as e:
            st.error(f"Tab rendering error: {str(e)}")
            render_error_recovery()
            
    except Exception as e:
        st.error(f"⚠️ Application Error: {str(e)}")
        logger.error(f"Main application error: {e}")
        logger.error(traceback.format_exc())
        render_error_recovery()

def render_fallback_interface():
    """Render a basic fallback interface when modules aren't available"""
    st.warning("🔧 Module loading issues detected. Using fallback interface.")
    
    # Basic file upload
    st.header("📁 Basic Document Upload")
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Files (Basic)", type="primary"):
            documents = []
            for file in uploaded_files:
                try:
                    # Basic text extraction
                    if file.type == "text/plain":
                        text = str(file.read(), "utf-8")
                    else:
                        text = f"[Content from {file.name} - processing not available]"
                    
                    doc = {
                        'filename': file.name,
                        'text': text,
                        'word_count': len(text.split()),
                        'upload_time': datetime.now()
                    }
                    documents.append(doc)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            if documents:
                st.session_state.documents = documents
                st.success(f"✅ Processed {len(documents)} documents in basic mode")
    
    # Basic search if documents exist
    if 'documents' in st.session_state and st.session_state.documents:
        st.header("🔍 Basic Search")
        query = st.text_input("Search documents:", placeholder="Enter search terms...")
        
        if query:
            results = []
            for doc in st.session_state.documents:
                if query.lower() in doc.get('text', '').lower():
                    count = doc['text'].lower().count(query.lower())
                    results.append({
                        'filename': doc['filename'],
                        'matches': count
                    })
            
            if results:
                st.success(f"Found {len(results)} matching documents")
                for result in results:
                    st.write(f"📄 {result['filename']} - {result['matches']} matches")
            else:
                st.warning("No matches found")

def render_upload_tab_safe(prepare_documents_for_search, extract_text_from_file):
    """Safe document upload with error handling"""
    try:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files for analysis"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        if prepare_documents_for_search and extract_text_from_file:
                            documents = prepare_documents_for_search(uploaded_files, extract_text_from_file)
                        else:
                            documents = fallback_process_documents(uploaded_files)
                        
                        st.success(f"✅ Processed {len(documents)} documents")
                        
                        # Show basic statistics
                        total_words = sum(doc.get('word_count', 0) for doc in documents)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents", len(documents))
                        with col2:
                            st.metric("Total Words", f"{total_words:,}")
                        with col3:
                            avg_words = total_words // len(documents) if documents else 0
                            st.metric("Avg Words", f"{avg_words:,}")
                        
                        st.markdown("""
                        **✅ Files processed successfully!** 
                        
                        **🔍 Next Steps:**
                        - Go to **Search** tab for keyword searches
                        - Go to **Align Rec-Resp** tab to find recommendations and responses
                        - Go to **Analytics** tab for document insights
                        """)
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        logger.error(f"Document processing error: {e}")
                        
    except Exception as e:
        st.error(f"Upload tab error: {str(e)}")
        render_basic_upload_fallback()

def render_extract_tab_safe():
    """Safe document extraction with error handling"""
    try:
        st.header("🔍 Document Extraction")
        
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("📁 Please upload documents first in the Upload tab.")
            return
        
        documents = st.session_state.documents
        doc_names = [doc['filename'] for doc in documents]
        selected_doc = st.selectbox("Select document to preview:", doc_names)
        
        if selected_doc:
            doc = next((d for d in documents if d['filename'] == selected_doc), None)
            
            if doc and 'text' in doc:
                text = doc['text']
                
                # Safe statistics calculation
                word_count = len(text.split()) if text else 0
                char_count = len(text) if text else 0
                estimated_pages = max(1, char_count // 2000)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Characters", f"{char_count:,}")
                with col2:
                    st.metric("Words", f"{word_count:,}")
                with col3:
                    # Safe sentence count
                    try:
                        sentences = re.split(r'[.!?]+', text)
                        sentence_count = len([s for s in sentences if s.strip()])
                    except:
                        sentence_count = word_count // 10  # Estimate
                    st.metric("Sentences", f"{sentence_count:,}")
                with col4:
                    st.metric("Est. Pages", estimated_pages)
                
                # Preview
                st.markdown("### 📖 Document Preview")
                preview_length = st.slider(
                    "Preview length (characters)", 
                    min_value=500, 
                    max_value=min(10000, len(text)), 
                    value=min(2000, len(text))
                )
                
                preview_text = text[:preview_length]
                if len(text) > preview_length:
                    preview_text += "... [truncated]"
                
                st.text_area(
                    "Document content:",
                    value=preview_text,
                    height=400,
                    disabled=True
                )
                
                # Download option
                st.download_button(
                    label="📥 Download Extracted Text",
                    data=text,
                    file_name=f"{selected_doc}_extracted.txt",
                    mime="text/plain"
                )
            else:
                st.error("Document text not available")
                
    except Exception as e:
        st.error(f"Extract tab error: {str(e)}")
        logger.error(f"Extract tab error: {e}")

def render_search_tab_safe(setup_search_tab):
    """Safe search tab with error handling"""
    try:
        if setup_search_tab:
            setup_search_tab()
        else:
            render_basic_search_fallback()
    except Exception as e:
        st.error(f"Search tab error: {str(e)}")
        logger.error(f"Search tab error: {e}")
        render_basic_search_fallback()

def render_alignment_tab_safe():
    """Safe alignment tab with error handling"""
    try:
        st.header("🔗 Recommendation-Response Alignment")
        
        if 'documents' not in st.session_state or not st.session_state.documents:
            st.warning("📁 Please upload documents first in the Upload tab.")
            show_alignment_feature_info()
            return
        
        try:
            # Try to import the alignment interface
            from modules.ui.search_components import render_recommendation_alignment_interface
            documents = st.session_state.documents
            render_recommendation_alignment_interface(documents)
        except ImportError:
            st.error("🔧 Alignment module not available. Using fallback interface.")
            render_basic_alignment_fallback()
            
    except Exception as e:
        st.error(f"Alignment tab error: {str(e)}")
        logger.error(f"Alignment tab error: {e}")
        render_basic_alignment_fallback()

def render_analytics_tab_safe(render_analytics_tab):
    """Safe analytics tab with error handling"""
    try:
        if render_analytics_tab:
            render_analytics_tab()
        else:
            render_basic_analytics_fallback()
    except Exception as e:
        st.error(f"Analytics tab error: {str(e)}")
        logger.error(f"Analytics tab error: {e}")
        render_basic_analytics_fallback()

def fallback_process_documents(uploaded_files):
    """Fallback document processing when modules aren't available"""
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Basic text extraction
            if uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                # Try basic PDF extraction
                try:
                    import PyPDF2
                    from io import BytesIO
                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                except ImportError:
                    text = f"[PDF content from {uploaded_file.name} - PDF processing not available]"
                except Exception:
                    text = f"[PDF processing failed for {uploaded_file.name}]"
            else:
                text = f"[Content from {uploaded_file.name} - processing not available for this file type]"
            
            doc = {
                'filename': uploaded_file.name,
                'text': text,
                'word_count': len(text.split()) if text else 0,
                'document_type': 'general',
                'upload_time': datetime.now(),
                'file_size': len(text) if text else 0
            }
            documents.append(doc)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            # Add error document
            documents.append({
                'filename': uploaded_file.name,
                'text': '',
                'error': str(e),
                'word_count': 0,
                'upload_time': datetime.now()
            })
    
    # Store in session state
    st.session_state.documents = documents
    return documents

def render_basic_upload_fallback():
    """Basic upload fallback interface"""
    st.markdown("### 📁 Basic File Upload")
    st.info("Using simplified upload process due to module loading issues.")
    
    uploaded_files = st.file_uploader(
        "Choose files (Basic Mode)",
        accept_multiple_files=True,
        type=['txt'],  # Only text files in basic mode
        help="Basic mode supports text files only"
    )
    
    if uploaded_files and st.button("Process Text Files"):
        documents = fallback_process_documents(uploaded_files)
        st.success(f"Processed {len(documents)} files in basic mode")

def render_basic_search_fallback():
    """Basic search fallback interface"""
    st.header("🔍 Basic Search")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first.")
        return
    
    documents = st.session_state.documents
    query = st.text_input("Search documents:", placeholder="Enter search terms...")
    
    if query:
        results = []
        query_lower = query.lower()
        
        for doc in documents:
            text = doc.get('text', '')
            if text and query_lower in text.lower():
                count = text.lower().count(query_lower)
                results.append({
                    'filename': doc['filename'],
                    'matches': count,
                    'word_count': doc.get('word_count', 0)
                })
        
        if results:
            st.success(f"Found {len(results)} matching documents")
            for result in results:
                st.write(f"📄 {result['filename']} - {result['matches']} matches ({result['word_count']} words)")
        else:
            st.warning("No matches found")

def render_basic_alignment_fallback():
    """Basic alignment fallback interface"""
    st.markdown("### 🔗 Basic Recommendation-Response Finder")
    st.info("Using simplified alignment process due to module loading issues.")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first.")
        return
    
    documents = st.session_state.documents
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Find Recommendations**")
        rec_keywords = st.text_input("Recommendation keywords:", value="recommend, suggest, advise")
    
    with col2:
        st.markdown("**↩️ Find Responses**")
        resp_keywords = st.text_input("Response keywords:", value="accept, reject, agree, implement")
    
    if st.button("🔍 Find Recommendations and Responses"):
        rec_words = [word.strip().lower() for word in rec_keywords.split(',')]
        resp_words = [word.strip().lower() for word in resp_keywords.split(',')]
        
        recommendations = []
        responses = []
        
        for doc in documents:
            text = doc.get('text', '').lower()
            filename = doc['filename']
            
            # Find recommendations
            for word in rec_words:
                if word in text:
                    count = text.count(word)
                    recommendations.append({
                        'document': filename,
                        'keyword': word,
                        'count': count
                    })
            
            # Find responses
            for word in resp_words:
                if word in text:
                    count = text.count(word)
                    responses.append({
                        'document': filename,
                        'keyword': word,
                        'count': count
                    })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Recommendations Found:**")
            if recommendations:
                for rec in recommendations:
                    st.write(f"📄 {rec['document']}: '{rec['keyword']}' ({rec['count']}x)")
            else:
                st.info("No recommendations found")
        
        with col2:
            st.markdown("**↩️ Responses Found:**")
            if responses:
                for resp in responses:
                    st.write(f"📄 {resp['document']}: '{resp['keyword']}' ({resp['count']}x)")
            else:
                st.info("No responses found")

def render_basic_analytics_fallback():
    """Basic analytics fallback interface"""
    st.header("📊 Basic Analytics")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 No documents to analyze.")
        return
    
    documents = st.session_state.documents
    
    # Basic statistics
    total_docs = len(documents)
    total_words = sum(doc.get('word_count', 0) for doc in documents)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Average Words", f"{avg_words:,}")
    
    # Document list
    st.markdown("### 📚 Document Details")
    doc_data = []
    for doc in documents:
        doc_data.append({
            'Filename': doc['filename'],
            'Words': doc.get('word_count', 0),
            'Type': doc.get('document_type', 'general').title(),
            'Status': 'Error' if 'error' in doc else 'OK'
        })
    
    df = pd.DataFrame(doc_data)
    st.dataframe(df, use_container_width=True)

def show_alignment_feature_info():
    """Show information about the alignment feature"""
    st.markdown("""
    ### 🎯 What This Feature Does:
    
    **🔍 Automatically finds:**
    - All recommendations in your documents
    - Corresponding responses to those recommendations
    - Aligns them using AI similarity matching
    
    **📊 Provides:**
    - Side-by-side view of recommendation + response
    - AI-generated summaries of each pair
    - Confidence scores for alignments
    - Export options for further analysis
    
    **💡 Perfect for:**
    - Government inquiry reports
    - Policy documents and responses
    - Committee recommendations and outcomes
    - Audit findings and management responses
    """)

def render_error_recovery():
    """Render error recovery options"""
    st.markdown("---")
    st.markdown("### 🛠️ Error Recovery")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Application"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Application reset. Please refresh the page.")
    
    with col2:
        if st.button("🧪 Load Sample Data"):
            create_sample_data()
    
    with col3:
        if st.button("📋 Show Debug Info"):
            show_debug_info()

def create_sample_data():
    """Create sample data for testing"""
    sample_doc = {
        'filename': 'sample_government_report.txt',
        'text': """
        Sample Government Report - Policy Review

        Executive Summary:
        This report contains several recommendations for improving government services.

        Recommendations:
        1. We recommend implementing new digital services to improve citizen access.
        2. The committee suggests reviewing current budget allocations for healthcare.
        3. We advise establishing a new framework for inter-departmental coordination.

        Government Response:
        1. The department agrees to implement digital services by Q4 2024.
        2. Budget review has been scheduled for the next fiscal year.
        3. The coordination framework proposal will be considered in the upcoming policy review.

        Conclusion:
        This demonstrates the alignment between recommendations and responses in government documentation.
        """,
        'word_count': 95,
        'document_type': 'government',
        'upload_time': datetime.now(),
        'file_size': 756
    }
    
    st.session_state.documents = [sample_doc]
    st.success("✅ Sample data loaded! You can now test the application features.")

def render_recommendations_tab():
    """Render the recommendations extraction tab"""
    st.header("🎯 Extract Recommendations")
    
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("📁 Please upload documents first in the Upload tab.")
        st.info("""
        **This feature automatically finds recommendations by detecting:**
        - Sentences starting with action verbs (Improving, Ensuring, etc.)
        - Phrases like "we recommend", "should", "must"
        - Any verb-based suggestions in your documents
        
        **Uses NLTK to automatically identify ALL verbs - no predefined list needed!**
        """)
        return
    
    documents = st.session_state.documents
    doc_names = [doc['filename'] for doc in documents]
    
    # Document selection
    selected_doc = st.selectbox("Select document to analyse:", doc_names)
    
    # Confidence slider
    min_confidence = st.slider(
        "Minimum confidence:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Only show recommendations with confidence above this threshold"
    )
    
    if st.button("🔍 Extract Recommendations", type="primary"):
        doc = next((d for d in documents if d['filename'] == selected_doc), None)
        
        if doc and 'text' in doc:
            with st.spinner("Analysing document..."):
                try:
                    # Extract recommendations
                    recommendations = extract_recommendations_simple(
                        doc['text'],
                        min_confidence=min_confidence
                    )
                    
                    if recommendations:
                        st.success(f"✅ Found {len(recommendations)} recommendations")
                        
                        # Show statistics
                        extractor = SimpleRecommendationExtractor()
                        stats = extractor.get_verb_statistics(recommendations)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Found", stats['total'])
                        with col2:
                            st.metric("Unique Verbs", stats['unique_verbs'])
                        with col3:
                            st.metric("Avg Confidence", f"{stats['avg_confidence']:.0%}")
                        
                        # Display recommendations
                        st.markdown("---")
                        st.subheader("📋 Recommendations Found")
                        
                        for idx, rec in enumerate(recommendations, 1):
                            with st.expander(
                                f"**{idx}. {rec['verb'].upper()}** "
                                f"(Confidence: {rec['confidence']:.0%})"
                            ):
                                st.write(rec['text'])
                                st.caption(f"Detection method: {rec['method']}")
                        
                        # Export options
                        st.markdown("---")
                        st.subheader("💾 Export")
                        
                        # Create CSV data
                        import pandas as pd
                        df = pd.DataFrame(recommendations)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"{selected_doc}_recommendations.csv",
                            mime="text/csv"
                        )
                        
                        # Store in session state for other tabs to use
                        st.session_state.extracted_recommendations = recommendations
                        
                    else:
                        st.warning("No recommendations found. Try lowering the confidence threshold.")
                        
                except Exception as e:
                    st.error(f"Error extracting recommendations: {str(e)}")
                    st.info("Make sure you have installed nltk: pip install nltk")
        else:
            st.error("Document text not available")
            
def show_debug_info():
    """Show debug information"""
    st.markdown("### 🔍 Debug Information")
    
    # Python environment
    import sys
    import platform
    
    st.code(f"""
    Python Version: {sys.version}
    Platform: {platform.platform()}
    Streamlit Version: {st.__version__}
    
    Session State Keys: {list(st.session_state.keys())}
    
    Documents in Session: {'Yes' if 'documents' in st.session_state else 'No'}
    Document Count: {len(st.session_state.get('documents', []))}
    """)

if __name__ == "__main__":
    main()
