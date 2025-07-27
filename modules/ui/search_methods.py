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
    """Enhanced smart search with alignment-quality techniques"""
    
    if not query.strip():
        return []
    
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()
    
    matches = []
    
    # Split query into meaningful words (stop words already filtered)
    query_words = [word for word in query.split() if len(word) > 1]
    
    if not query_words:
        return []
    
    # ENHANCED: Split text into sentences for better context
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Find matches in sentences (much better than word-by-word)
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        
        # Calculate multiple similarity scores
        word_matches = 0
        semantic_score = 0
        total_query_words = len(query_words)
        
        for word in query_words:
            # ENHANCED: Better pattern matching with variations
            root_word = get_root_word(word)
            patterns = [
                rf'\b{re.escape(word)}\w*',
                rf'\b{re.escape(root_word)}\w*',
            ]
            
            word_found = False
            for pattern in patterns:
                try:
                    if re.search(pattern, sentence_lower):
                        word_matches += 1
                        word_found = True
                        break
                except re.error:
                    continue
            
            # ENHANCED: Semantic matching using government groups
            if not word_found:
                semantic_match = find_semantic_match(word, sentence_lower)
                if semantic_match:
                    semantic_score += 1
        
        # Calculate combined similarity score
        word_similarity = word_matches / total_query_words if total_query_words > 0 else 0
        semantic_similarity = semantic_score / total_query_words if total_query_words > 0 else 0
        
        # ENHANCED: Combined scoring like alignment system
        combined_score = (word_similarity * 0.7) + (semantic_similarity * 0.3)
        
        if combined_score > 0.2:  # Lower threshold for more results
            
            # Calculate position in original text
            char_position = text.find(sentence)
            if char_position == -1:
                char_position = 0
            
            # ENHANCED: Content classification
            content_type = classify_sentence_content_type(sentence)
            content_relevance = calculate_content_relevance(sentence, query)
            
            # ENHANCED: Better scoring system
            base_score = combined_score * 100
            content_bonus = content_relevance * 20  # Bonus for relevant content types
            final_score = min(base_score + content_bonus, 100)
            
            # Extract enhanced context (previous and next sentence)
            context_start = max(0, i - 1)
            context_end = min(len(sentences), i + 2)
            context = ' '.join(sentences[context_start:context_end])
            
            match_info = {
                'position': char_position,
                'matched_text': sentence,
                'context': context,
                'score': final_score,
                'match_type': 'smart_enhanced',
                'page_number': estimate_page_number(char_position, text),
                'word_position': len(text[:char_position].split()),
                'percentage_through': (char_position / len(text)) * 100 if text else 0,
                'content_type': content_type,
                'content_relevance': content_relevance,
                'word_similarity': word_similarity,
                'semantic_similarity': semantic_similarity,
                'query_coverage': f"{word_matches + semantic_score}/{total_query_words}"
            }
            
            matches.append(match_info)
    
    # ENHANCED: Smart deduplication and ranking
    matches = remove_overlapping_matches(matches)
    
    # ENHANCED: Multi-factor sorting
    matches.sort(key=lambda x: (
        x['score'],  # Primary: Overall score
        x['content_relevance'],  # Secondary: Content relevance
        -x['position']  # Tertiary: Earlier in document
    ), reverse=True)
    
    return matches

def get_root_word(word: str) -> str:
    """Extract root word for better matching"""
    if word.endswith(('ing', 'ed', 'er', 's', 'ly', 'tion', 'ment')):
        if word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith(('er', 'ly')):
            return word[:-2]
        elif word.endswith('tion'):
            return word[:-4]
        elif word.endswith('ment'):
            return word[:-4]
        elif word.endswith('s'):
            return word[:-1]
    return word

def find_semantic_match(query_word: str, sentence_lower: str) -> bool:
    """Find semantic matches using government terminology"""
    
    # Enhanced semantic groups from alignment system
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
        'report': ['report', 'document', 'paper', 'study', 'analysis', 'investigation'],
        'meeting': ['meeting', 'conference', 'session', 'discussion', 'consultation', 'briefing'],
        'budget': ['budget', 'funding', 'financial', 'cost', 'expenditure', 'allocation'],
        'urgent': ['urgent', 'immediate', 'critical', 'priority', 'emergency', 'pressing']
    }
    
    # Check if query word has semantic matches in the sentence
    for key, synonyms in semantic_groups.items():
        if query_word in synonyms:
            # Look for any synonym in the sentence
            for synonym in synonyms:
                if synonym in sentence_lower:
                    return True
    
    return False

def classify_sentence_content_type(sentence: str) -> str:
    """Classify sentence content type like alignment system"""
    
    sentence_lower = sentence.lower()
    
    # Government document classification
    if any(word in sentence_lower for word in ['recommend', 'suggest', 'advise', 'propose', 'should', 'must', 'ought']):
        return 'Recommendation'
    elif any(word in sentence_lower for word in ['accept', 'reject', 'agree', 'disagree', 'implement', 'consider', 'response', 'reply']):
        return 'Response'
    elif any(word in sentence_lower for word in ['policy', 'procedure', 'guideline', 'framework', 'protocol', 'strategy']):
        return 'Policy'
    elif any(word in sentence_lower for word in ['review', 'analyze', 'assess', 'evaluate', 'examine', 'investigation']):
        return 'Analysis'
    elif any(word in sentence_lower for word in ['budget', 'funding', 'financial', 'cost', 'expenditure']):
        return 'Financial'
    elif any(word in sentence_lower for word in ['urgent', 'immediate', 'critical', 'priority', 'emergency']):
        return 'Urgent'
    elif any(word in sentence_lower for word in ['meeting', 'conference', 'discussion', 'consultation']):
        return 'Meeting'
    elif any(word in sentence_lower for word in ['government', 'department', 'ministry', 'agency', 'authority']):
        return 'Government'
    else:
        return 'General'

def calculate_content_relevance(sentence: str, query: str) -> float:
    """Calculate how relevant the content type is to the query"""
    
    sentence_type = classify_sentence_content_type(sentence)
    query_lower = query.lower()
    
    # High relevance mappings
    high_relevance = {
        'Recommendation': ['recommend', 'suggest', 'advise', 'propose'],
        'Response': ['respond', 'response', 'reply', 'answer'],
        'Policy': ['policy', 'procedure', 'framework', 'guideline'],
        'Financial': ['budget', 'cost', 'funding', 'financial'],
        'Government': ['government', 'department', 'ministry'],
        'Urgent': ['urgent', 'immediate', 'critical', 'emergency']
    }
    
    # Check if query matches content type
    for content_type, keywords in high_relevance.items():
        if sentence_type == content_type:
            if any(keyword in query_lower for keyword in keywords):
                return 1.0  # Perfect match
            else:
                return 0.5  # Type match but not keyword match
    
    return 0.3  # General relevance

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
    """Enhanced semantic search with immediate fallback for Streamlit Cloud"""
    
    # STREAMLIT CLOUD FIX: Skip problematic model loading entirely
    # Go straight to enhanced fallback which works better anyway
    try:
        # Quick check if we can use AI models
        if 'semantic_model_available' not in st.session_state:
            st.session_state.semantic_model_available = False
            
            # Only try model loading if explicitly requested and not on Streamlit Cloud
            import os
            is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') or 'streamlit.app' in os.getenv('HOSTNAME', '')
            
            if not is_streamlit_cloud:
                try:
                    # Quick test for local development
                    return semantic_search_direct(text, query)
                except Exception:
                    pass
        
        # Use enhanced fallback (which is actually better for government docs)
        return semantic_fallback_search(text, query)
        
    except Exception as e:
        st.warning(f"🤖 AI semantic search had issues: {str(e)}. Using enhanced similarity matching.")
        return semantic_fallback_search(text, query)

def check_rag_availability() -> bool:
    """Check if RAG dependencies are available - STREAMLIT CLOUD OPTIMIZED"""
    try:
        import sentence_transformers
        import torch
        import os
        
        # Check if we're on Streamlit Cloud
        is_streamlit_cloud = (
            os.getenv('STREAMLIT_SHARING_MODE') or 
            'streamlit.app' in os.getenv('HOSTNAME', '') or
            '/mount/src/' in os.getcwd()
        )
        
        if is_streamlit_cloud:
            # On Streamlit Cloud, always return False to use fallback
            # This avoids the PyTorch meta tensor issues entirely
            return False
        
        # Only test model loading on local development
        try:
            from sentence_transformers import SentenceTransformer
            device = 'cpu'
            torch.set_default_device('cpu')
            
            # Quick test
            test_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            with torch.no_grad():
                test_model.encode(["test"], convert_to_tensor=False, device=device)
            return True
            
        except Exception:
            return False
        
    except ImportError:
        return False
    except Exception:
        return False

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
