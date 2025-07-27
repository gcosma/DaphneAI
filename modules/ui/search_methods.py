# modules/ui/search_methods.py - Search Method Implementations
import streamlit as st
import re
import difflib
from typing import Dict, List
from .search_utils import (
    filter_stop_words, preprocess_query, estimate_page_number,
    remove_overlapping_matches, STOP_WORDS
)

def execute_search(documents: List[Dict], query: str, method: str, max_results: int = None, case_sensitive: bool = False) -> List[Dict]:
    """Execute search using the specified method with stop word filtering"""
    
    # Preprocess query
    processed_query = preprocess_query(query, method)
    
    # Show query transformation if different
    if processed_query != query and method != "exact":
        st.info(f"🔍 Search terms after filtering stop words: '{processed_query}'")
    
    all_results = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Execute search based on method using processed query
        if method == "exact":
            matches = exact_search(text, query, case_sensitive)  # Use original query for exact
        elif method == "smart":
            matches = smart_search_filtered(text, processed_query, case_sensitive)
        elif method == "fuzzy":
            matches = fuzzy_search_filtered(text, processed_query, case_sensitive)
        elif method == "semantic":
            matches = semantic_search(text, processed_query)
        elif method == "hybrid":
            matches = hybrid_search_filtered(text, processed_query, case_sensitive)
        else:
            matches = smart_search_filtered(text, processed_query, case_sensitive)
        
        # CHANGED: Only limit results if max_results is specified
        if max_results is not None:
            matches = matches[:max_results]
        
        # Add document info to each match
        for match in matches:
            match['document'] = doc
            match['search_method'] = method
            match['original_query'] = query
            match['processed_query'] = processed_query
            all_results.append(match)
    
    # Sort by relevance score
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return all_results

def exact_search(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Find exact matches only"""
    
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()
    
    matches = []
    start = 0
    
    while True:
        pos = search_text.find(search_query, start)
        if pos == -1:
            break
        
        # Extract context
        context_start = max(0, pos - 100)
        context_end = min(len(text), pos + len(query) + 100)
        context = text[context_start:context_end]
        
        match = {
            'position': pos,
            'matched_text': text[pos:pos + len(query)],
            'context': context,
            'score': 100.0,  # Exact matches get highest score
            'match_type': 'exact',
            'page_number': estimate_page_number(pos, text),
            'word_position': len(text[:pos].split()),
            'percentage_through': (pos / len(text)) * 100 if text else 0
        }
        
        matches.append(match)
        start = pos + 1
    
    return matches

def smart_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Smart search with stop word filtering - ENHANCED for word variations"""
    
    if not query.strip():
        return []
    
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()
    
    matches = []
    
    # Split query into meaningful words (stop words already filtered)
    query_words = [word for word in query.split() if len(word) > 1]
    
    if not query_words:
        return []
    
    # Find each meaningful word with better pattern matching
    for word in query_words:
        # ENHANCED: Create root word for better matching
        root_word = word
        if word.endswith(('ing', 'ed', 'er', 's', 'ly', 'tion', 'ment')):
            # Try to find root word
            if word.endswith('ing'):
                root_word = word[:-3]
            elif word.endswith('ed'):
                root_word = word[:-2]
            elif word.endswith(('er', 'ly')):
                root_word = word[:-2]
            elif word.endswith('tion'):
                root_word = word[:-4]
            elif word.endswith('ment'):
                root_word = word[:-4]
            elif word.endswith('s'):
                root_word = word[:-1]
        
        # ENHANCED: More flexible pattern to catch variations
        # This will match: recommend, recommending, recommended, recommendation, etc.
        patterns = [
            rf'\b{re.escape(word)}\w*',           # Original word + endings
            rf'\b{re.escape(root_word)}\w*',      # Root word + endings
        ]
        
        # Remove duplicates
        patterns = list(set(patterns))
        
        for pattern in patterns:
            try:
                for match in re.finditer(pattern, search_text):
                    pos = match.start()
                    matched_text = text[match.start():match.end()]
                    
                    # Calculate score based on match quality
                    matched_lower = match.group().lower()
                    word_lower = word.lower()
                    
                    if matched_lower == word_lower:
                        score = 100.0  # Exact word match
                    elif matched_lower.startswith(word_lower):
                        score = 95.0   # Word starts with query (recommend -> recommending)
                    elif matched_lower.startswith(root_word.lower()):
                        score = 90.0   # Root word match (recommendation -> recommend)
                    elif word_lower in matched_lower:
                        score = 85.0   # Query word contained in match
                    else:
                        score = 70.0   # Partial match
                    
                    # Bonus for common word variations
                    if any(matched_lower.endswith(suffix) for suffix in ['ing', 'ed', 'tion', 'ment']):
                        score += 5.0
                    
                    # Extract context
                    context_start = max(0, pos - 150)
                    context_end = min(len(text), pos + len(matched_text) + 150)
                    context = text[context_start:context_end]
                    
                    match_info = {
                        'position': pos,
                        'matched_text': matched_text,
                        'context': context,
                        'score': score,
                        'match_type': 'smart',
                        'page_number': estimate_page_number(pos, text),
                        'word_position': len(text[:pos].split()),
                        'percentage_through': (pos / len(text)) * 100 if text else 0,
                        'query_word': word,
                        'pattern_used': pattern
                    }
                    
                    matches.append(match_info)
            except re.error:
                # Skip invalid regex patterns
                continue
    
    # Remove overlapping matches and sort by score
    matches = remove_overlapping_matches(matches)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches

def fuzzy_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Fuzzy search with stop word filtering - FIXED highlighting"""
    
    if not query.strip():
        return []
    
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()
    
    # Split into meaningful words
    words = text.split()
    search_words = search_text.split()
    query_words = query.split()
    
    matches = []
    
    # Check each word against each query word
    for query_word in query_words:
        if len(query_word) < 2:
            continue
            
        for i, word in enumerate(search_words):
            similarity = difflib.SequenceMatcher(None, query_word, word).ratio()
            
            if similarity > 0.6:  # Lower threshold for better results (was 0.7)
                # Find position in original text
                pos = len(' '.join(words[:i]))
                if i > 0:
                    pos += 1  # Add space
                
                # Extract context with proper highlighting context
                context_start = max(0, pos - 150)
                context_end = min(len(text), pos + len(words[i]) + 150)
                context = text[context_start:context_end]
                
                # FIXED: Use original case word for matched_text
                matched_text = words[i]
                
                match = {
                    'position': pos,
                    'matched_text': matched_text,  # FIXED: Original case preserved
                    'context': context,
                    'score': similarity * 100,
                    'match_type': 'fuzzy',
                    'similarity': similarity,
                    'page_number': estimate_page_number(pos, text),
                    'word_position': i,
                    'percentage_through': (pos / len(text)) * 100 if text else 0,
                    'query_word': query_word
                }
                
                matches.append(match)
    
    # Remove overlapping matches and sort by similarity
    matches = remove_overlapping_matches(matches)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches

def semantic_search(text: str, query: str) -> List[Dict]:
    """Enhanced semantic search with proper fallback"""
    
    try:
        # First try direct sentence transformers
        return semantic_search_direct(text, query)
    except Exception as e:
        st.warning(f"🤖 AI semantic search failed: {str(e)}. Using enhanced similarity matching.")
        return semantic_fallback_search(text, query)

def semantic_search_direct(text: str, query: str) -> List[Dict]:
    """Direct semantic search using sentence transformers - FIXED for Streamlit Cloud"""
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import torch
        
        # FIXED: Force CPU usage and handle meta tensor issues
        device = 'cpu'  # Force CPU on Streamlit Cloud
        torch.set_default_device('cpu')
        
        # Initialize model if not cached with proper device handling
        if 'semantic_model' not in st.session_state:
            try:
                # Try loading with explicit CPU device
                model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                # Ensure model is on CPU
                model = model.to(device)
                st.session_state.semantic_model = model
                st.session_state.model_device = device
                
            except Exception as model_error:
                # Fallback to a more compatible model
                try:
                    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
                    model = model.to(device)
                    st.session_state.semantic_model = model
                    st.session_state.model_device = device
                except Exception:
                    # If both fail, raise to trigger fallback search
                    raise Exception(f"Model loading failed: {str(model_error)}")
        
        model = st.session_state.semantic_model
        
        # Ensure model is on correct device
        if hasattr(model, 'device') and model.device != torch.device(device):
            model = model.to(device)
        
        # Split text into chunks (sentences)
        sentences = re.split(r'[.!?]+', text)
        chunks = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if not chunks:
            return []
        
        # Limit chunks to avoid memory issues on Streamlit Cloud
        if len(chunks) > 100:
            # Take first 50 and last 50 chunks
            chunks = chunks[:50] + chunks[-50:]
        
        # Generate embeddings with error handling
        try:
            with torch.no_grad():  # Disable gradients for inference
                query_embedding = model.encode([query], convert_to_tensor=False, device=device)
                chunk_embeddings = model.encode(chunks, convert_to_tensor=False, device=device, batch_size=16)
            
            # Ensure numpy arrays
            if torch.is_tensor(query_embedding):
                query_embedding = query_embedding.cpu().numpy()
            if torch.is_tensor(chunk_embeddings):
                chunk_embeddings = chunk_embeddings.cpu().numpy()
                
        except Exception as encoding_error:
            st.warning(f"Encoding failed: {str(encoding_error)}")
            # Try with smaller batch size
            try:
                query_embedding = model.encode([query], convert_to_tensor=False, batch_size=1)
                chunk_embeddings = model.encode(chunks[:20], convert_to_tensor=False, batch_size=1)  # Limit to 20 chunks
                chunks = chunks[:20]  # Update chunks list
            except Exception:
                raise Exception(f"Encoding failed: {str(encoding_error)}")
        
        # Calculate cosine similarity
        similarities = np.dot(query_embedding, chunk_embeddings.T).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:5]  # Top 5 matches
        
        matches = []
        for idx in top_indices:
            if idx >= len(chunks):  # Safety check
                continue
                
            similarity = similarities[idx]
            
            if similarity > 0.3:  # Minimum similarity threshold
                chunk = chunks[idx]
                
                # Find position in original text
                pos = text.find(chunk)
                if pos == -1:
                    pos = 0
                
                match = {
                    'position': pos,
                    'matched_text': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'context': chunk,
                    'score': similarity * 100,
                    'match_type': 'semantic',
                    'page_number': estimate_page_number(pos, text),
                    'word_position': len(text[:pos].split()),
                    'percentage_through': (pos / len(text)) * 100 if text else 0,
                    'semantic_score': similarity
                }
                
                matches.append(match)
        
        return matches
        
    except ImportError:
        raise Exception("Sentence transformers not available")
    except RuntimeError as e:
        if "meta tensor" in str(e) or "CUDA" in str(e):
            raise Exception(f"PyTorch device error: {str(e)}")
        else:
            raise Exception(f"Runtime error: {str(e)}")
    except Exception as e:
        raise Exception(f"Semantic search error: {str(e)}")

def check_rag_availability() -> bool:
    """Check if RAG dependencies are available and working - ENHANCED"""
    try:
        import sentence_transformers
        import torch
        
        # Try to load a small model to verify it works
        from sentence_transformers import SentenceTransformer
        
        # FIXED: Test with CPU device only
        device = 'cpu'
        torch.set_default_device('cpu')
        
        # Quick test with a simple model
        test_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        test_model = test_model.to(device)
        
        # Test encoding a simple sentence
        with torch.no_grad():
            test_embedding = test_model.encode(["test sentence"], convert_to_tensor=False, device=device)
        
        return True
        
    except ImportError:
        return False
    except RuntimeError as e:
        if "meta tensor" in str(e) or "CUDA" in str(e):
            # PyTorch/device issues, but libraries are available
            return False
        return False
    except Exception:
        return False

def semantic_fallback_search(text: str, query: str) -> List[Dict]:
    """Enhanced fallback semantic search using word similarity and synonyms"""
    
    # Enhanced semantic word groups (synonyms and related terms)
    semantic_groups = {
        'recommend': ['recommend', 'suggestion', 'suggest', 'advise', 'propose', 'urge', 'advocate', 'endorse', 'recommendation', 'recommendations'],
        'suggest': ['suggest', 'recommend', 'proposal', 'propose', 'advise', 'hint', 'indicate', 'suggestion', 'suggestions'],
        'respond': ['respond', 'response', 'reply', 'answer', 'feedback', 'reaction', 'comment', 'responses', 'replies'],
        'response': ['response', 'respond', 'reply', 'answer', 'feedback', 'reaction', 'comment', 'responses', 'replies'],
        'implement': ['implement', 'execute', 'carry out', 'put into practice', 'apply', 'deploy', 'implementation'],
        'review': ['review', 'examine', 'assess', 'evaluate', 'analyze', 'inspect', 'analysis'],
        'policy': ['policy', 'procedure', 'guideline', 'protocol', 'framework', 'strategy', 'policies'],
        'accept': ['accept', 'agree', 'approve', 'endorse', 'support', 'adopt', 'acceptance'],
        'reject': ['reject', 'decline', 'refuse', 'dismiss', 'deny', 'oppose', 'rejection'],
        'government': ['government', 'department', 'ministry', 'agency', 'authority', 'administration'],
        'report': ['report', 'document', 'paper', 'study', 'analysis', 'investigation']
    }
    
    matches = []
    query_words = query.lower().split()
    
    # Remove stop words from query
    meaningful_words = [word for word in query_words if word not in STOP_WORDS and len(word) > 2]
    
    if not meaningful_words:
        return []
    
    # Find semantic matches for each meaningful word
    for query_word in meaningful_words:
        
        # Find related words for this query term
        related_words = []
        
        # Direct lookup in semantic groups
        if query_word in semantic_groups:
            related_words.extend(semantic_groups[query_word])
        
        # Check if query word is contained in any synonym
        for key, synonyms in semantic_groups.items():
            if any(query_word in synonym for synonym in synonyms):
                related_words.extend(synonyms)
        
        # If no semantic group found, use the word itself and common variations
        if not related_words:
            related_words = [query_word]
            # Add common word endings
            if len(query_word) > 4:
                related_words.extend([
                    query_word + 's',
                    query_word + 'ing', 
                    query_word + 'ed',
                    query_word + 'ion'
                ])
        
        # Remove duplicates
        related_words = list(set(related_words))
        
        # Search for related words in text
        search_text = text.lower()
        words = text.split()
        search_words = search_text.split()
        
        for i, word in enumerate(search_words):
            for related_word in related_words:
                # Check for matches (word contains related word or vice versa)
                if (related_word in word or word in related_word) and len(word) > 2:
                    
                    # Calculate semantic similarity score
                    if word == related_word:
                        score = 95.0  # Exact semantic match
                    elif word == query_word:
                        score = 100.0  # Exact query match
                    elif related_word in word:
                        score = 85.0  # Contains semantic word
                    elif word in related_word:
                        score = 80.0  # Word contained in semantic term
                    elif word.startswith(related_word) or related_word.startswith(word):
                        score = 75.0  # Prefix match
                    else:
                        score = 70.0  # Partial semantic match
                    
                    # Find position in original text
                    pos = len(' '.join(words[:i]))
                    if i > 0:
                        pos += 1
                    
                    # Extract context
                    context_start = max(0, pos - 150)
                    context_end = min(len(text), pos + len(words[i]) + 150)
                    context = text[context_start:context_end]
                    
                    match = {
                        'position': pos,
                        'matched_text': words[i],
                        'context': context,
                        'score': score,
                        'match_type': 'semantic',
                        'page_number': estimate_page_number(pos, text),
                        'word_position': i,
                        'percentage_through': (pos / len(text)) * 100 if text else 0,
                        'semantic_relation': f"{query_word} → {related_word}",
                        'query_word': query_word
                    }
                    
                    matches.append(match)
                    break  # Only match once per word
    
    # Remove overlapping matches and sort by score
    matches = remove_overlapping_matches(matches)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches

def hybrid_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict]:
    """Hybrid search combining smart and semantic search"""
    
    # Get smart search results with filtering
    smart_results = smart_search_filtered(text, query, case_sensitive)
    
    # Get semantic search results (semantic search handles its own processing)
    semantic_results = semantic_search(text, query)
    
    # Combine and deduplicate
    all_matches = smart_results + semantic_results
    
    # Remove overlapping matches
    unique_matches = remove_overlapping_matches(all_matches)
    
    # Sort by score
    unique_matches.sort(key=lambda x: x['score'], reverse=True)
    
    return unique_matches
