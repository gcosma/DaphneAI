# modules/ui/search_utils.py - Search Utility Functions
import re
from typing import Dict, List

# Comprehensive stop words list
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 
    'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 
    'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 
    'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 
    'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 
    'come', 'made', 'may', 'part'
}

def filter_stop_words(query: str) -> str:
    """Remove stop words from query"""
    words = query.lower().split()
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    
    # If all words are stop words, keep the original query
    if not filtered_words:
        return query
    
    return ' '.join(filtered_words)

def preprocess_query(query: str, method: str) -> str:
    """Preprocess query based on search method"""
    
    # For exact search, don't filter stop words
    if method == "exact":
        return query
    
    # For other methods, filter stop words
    filtered_query = filter_stop_words(query)
    
    return filtered_query

def estimate_page_number(char_position: int, text: str) -> int:
    """Estimate page number based on character position"""
    if char_position <= 0:
        return 1
    return max(1, char_position // 2000 + 1)  # ~2000 chars per page

def check_rag_availability() -> bool:
    """Check if RAG dependencies are available and working"""
    try:
        import sentence_transformers
        import torch
        # Try to load a small model to verify it works
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return True
    except ImportError:
        return False
    except Exception:
        return False

def remove_overlapping_matches(matches: List[Dict]) -> List[Dict]:
    """Remove overlapping matches, keeping the highest scored ones"""
    
    if not matches:
        return matches
    
    # Sort by score descending
    sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)
    
    unique_matches = []
    used_positions = set()
    
    for match in sorted_matches:
        pos = match['position']
        matched_text = match.get('matched_text', '')
        match_length = len(matched_text)
        
        # Check if this match overlaps with any existing match
        overlap = False
        for used_start, used_end in used_positions:
            # Check for overlap
            if not (pos + match_length <= used_start or pos >= used_end):
                overlap = True
                break
        
        if not overlap:
            unique_matches.append(match)
            used_positions.add((pos, pos + match_length))
    
    return unique_matches

def highlight_search_terms(text: str, query: str) -> str:
    """Highlight search terms in text - FIXED to exclude stop words"""
    
    # Filter out stop words from highlighting (same as search filtering)
    filtered_query = filter_stop_words(query)
    query_words = filtered_query.split() if filtered_query else []
    
    highlighted = text
    
    # Sort words by length (longest first) to avoid partial highlighting conflicts
    query_words.sort(key=len, reverse=True)
    
    for word in query_words:
        if len(word) > 1 and word.lower() not in STOP_WORDS:  # Double-check stop words
            
            # Create multiple patterns for better matching
            patterns = [
                word,                    # Exact word
                word.rstrip('s'),        # Remove plural 's'
                word + 'ing',           # Add 'ing'
                word + 'ed',            # Add 'ed'
                word + 'tion',          # Add 'tion'
                word + 'ment',          # Add 'ment'
            ]
            
            # Also try root forms
            if word.endswith('ing'):
                patterns.append(word[:-3])  # Remove 'ing'
            if word.endswith('ed'):
                patterns.append(word[:-2])  # Remove 'ed'
            if word.endswith('tion'):
                patterns.append(word[:-4])  # Remove 'tion'
            
            # Remove short or duplicate patterns, and filter stop words again
            patterns = list(set([p for p in patterns if len(p) > 2 and p.lower() not in STOP_WORDS]))
            
            # Apply highlighting for each pattern
            for pattern in patterns:
                try:
                    # Case-insensitive word boundary matching
                    regex_pattern = rf'\b{re.escape(pattern)}\w*'
                    compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)
                    
                    def highlight_match(match):
                        matched_word = match.group()
                        # Double-check: don't highlight if it's a stop word
                        if matched_word.lower() in STOP_WORDS:
                            return matched_word
                        return f"<mark style='background-color: #FFEB3B; padding: 2px; border-radius: 2px; font-weight: bold;'>{matched_word}</mark>"
                    
                    highlighted = compiled_pattern.sub(highlight_match, highlighted)
                except re.error:
                    # Skip invalid patterns
                    continue
    
    return highlighted
