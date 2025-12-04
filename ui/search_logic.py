"""Shared search and alignment helpers extracted from the legacy search_components module."""

from __future__ import annotations

import difflib
import logging
import os
import re
from typing import Any, Dict, List

import streamlit as st

from daphne_core.alignment_engine import (
    align_recommendations_with_responses,
    calculate_simple_similarity,
    classify_content_type,
    determine_alignment_status,
    find_pattern_matches,
)
from daphne_core.search_utils import STOP_WORDS

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Search helpers
# --------------------------------------------------------------------------- #

def execute_search_with_ai(
    documents: List[Dict[str, Any]],
    query: str,
    method: str,
    max_results: int | None = None,
    case_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """Execute search with all methods including AI semantic search."""
    results: List[Dict[str, Any]] = []

    for doc in documents:
        text = doc.get("text", "")
        if not text:
            continue

        if "Smart" in method:
            matches = smart_search_filtered(text, query, case_sensitive)
        elif "Exact" in method:
            matches = exact_search_unfiltered(text, query, case_sensitive)
        elif "Fuzzy" in method:
            matches = fuzzy_search_filtered(text, query, case_sensitive)
        elif "AI Semantic" in method:
            matches = ai_semantic_search(text, query, case_sensitive)
        elif "Hybrid" in method:
            matches = hybrid_search_smart_ai(text, query, case_sensitive)
        else:
            matches = smart_search_filtered(text, query, case_sensitive)

        if max_results:
            matches = matches[:max_results]

        for match in matches:
            match["document"] = doc
            results.append(match)

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


def smart_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """Smart search with stop word filtering for better results."""
    matches: List[Dict[str, Any]] = []
    meaningful_words = get_meaningful_words(query)

    if not meaningful_words:
        return matches

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    for i, sentence in enumerate(sentences):
        sentence_search = sentence if case_sensitive else sentence.lower()

        word_matches = sum(1 for word in meaningful_words if word.lower() in sentence_search)
        if word_matches <= 0:
            continue

        score = (word_matches / len(meaningful_words)) * 100
        if word_matches > 1:
            score *= 1.2

        pos = text.find(sentence)
        if pos == -1:
            pos = i * 100

        context = get_context_simple(sentences, i, 2)
        matches.append(
            {
                "position": pos,
                "matched_text": sentence,
                "context": context,
                "score": min(score, 100),
                "match_type": "smart",
                "page_number": max(1, pos // 2000 + 1),
                "word_matches": word_matches,
                "total_meaningful_words": len(meaningful_words),
                "meaningful_words_found": [w for w in meaningful_words if w.lower() in sentence_search],
                "percentage_through": (pos / len(text)) * 100 if text else 0,
            }
        )

    return matches


def exact_search_unfiltered(text: str, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """Exact search without filtering (preserves exact phrases)."""
    matches: List[Dict[str, Any]] = []
    search_text = text if case_sensitive else text.lower()
    search_query = query if case_sensitive else query.lower()

    start = 0
    while True:
        pos = search_text.find(search_query, start)
        if pos == -1:
            break

        context_start = max(0, pos - 150)
        context_end = min(len(text), pos + len(query) + 150)
        context = text[context_start:context_end]

        matches.append(
            {
                "position": pos,
                "matched_text": text[pos : pos + len(query)],
                "context": context,
                "score": 100.0,
                "match_type": "exact",
                "page_number": max(1, pos // 2000 + 1),
                "percentage_through": (pos / len(text)) * 100 if text else 0,
            }
        )
        start = pos + 1

    return matches


def fuzzy_search_filtered(text: str, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """Fuzzy search focusing on meaningful words."""
    matches: List[Dict[str, Any]] = []
    meaningful_words = get_meaningful_words(query)
    if not meaningful_words:
        return matches

    words = text.split()
    search_words = [w if case_sensitive else w.lower() for w in words]

    for query_word in meaningful_words:
        query_word_search = query_word if case_sensitive else query_word.lower()
        for i, word in enumerate(search_words):
            similarity = difflib.SequenceMatcher(None, query_word_search, word).ratio()
            if similarity <= 0.6:
                continue

            pos = len(" ".join(words[:i]))
            if i > 0:
                pos += 1

            context_start = max(0, i - 10)
            context_end = min(len(words), i + 10)
            context = " ".join(words[context_start:context_end])

            matches.append(
                {
                    "position": pos,
                    "matched_text": words[i],
                    "context": context,
                    "score": similarity * 100,
                    "match_type": "fuzzy",
                    "page_number": max(1, pos // 2000 + 1),
                    "similarity": similarity,
                    "query_word": query_word,
                    "percentage_through": (pos / len(text)) * 100 if text else 0,
                }
            )

    unique_matches = []
    seen_positions = set()
    for match in sorted(matches, key=lambda x: x["score"], reverse=True):
        pos = match["position"]
        if pos not in seen_positions:
            unique_matches.append(match)
            seen_positions.add(pos)

    return unique_matches


def ai_semantic_search(text: str, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """AI Semantic search with fallback."""
    try:
        return ai_semantic_search_direct(text, query, case_sensitive)
    except Exception as exc:
        logger.info(f"Using semantic fallback: {exc}")
        return semantic_fallback_search(text, query, case_sensitive)


def ai_semantic_search_direct(text: str, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """Direct AI semantic search using sentence transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import torch

        device = "cpu"
        torch.set_default_device("cpu")

        if "semantic_model" not in st.session_state:
            model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            model = model.to(device)
            st.session_state.semantic_model = model
            st.session_state.model_device = device

        model = st.session_state.semantic_model
        meaningful_query = " ".join(get_meaningful_words(query)) or query

        sentences = re.split(r"[.!?]+", text)
        chunks = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        if not chunks:
            return []
        if len(chunks) > 100:
            chunks = chunks[:50] + chunks[-50:]

        with torch.no_grad():
            query_embedding = model.encode([meaningful_query], convert_to_tensor=False, device=device)
            chunk_embeddings = model.encode(chunks, convert_to_tensor=False, device=device, batch_size=16)

        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.cpu().numpy()
        if torch.is_tensor(chunk_embeddings):
            chunk_embeddings = chunk_embeddings.cpu().numpy()

        similarities = np.dot(query_embedding, chunk_embeddings.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:10]

        matches: List[Dict[str, Any]] = []
        for idx in top_indices:
            if idx >= len(chunks):
                continue

            similarity = similarities[idx]
            if similarity <= 0.3:
                continue

            chunk = chunks[idx]
            pos = text.find(chunk)
            if pos == -1:
                pos = 0

            sentences_for_context = re.split(r"[.!?]+", text)
            chunk_idx = -1
            for i, sent in enumerate(sentences_for_context):
                if chunk in sent:
                    chunk_idx = i
                    break

            context = (
                get_context_simple(sentences_for_context, chunk_idx, 2) if chunk_idx != -1 else chunk
            )

            matches.append(
                {
                    "position": pos,
                    "matched_text": f"{chunk[:100]}..." if len(chunk) > 100 else chunk,
                    "context": context,
                    "score": similarity * 100,
                    "match_type": "semantic",
                    "page_number": max(1, pos // 2000 + 1),
                    "word_position": len(text[:pos].split()),
                    "percentage_through": (pos / len(text)) * 100 if text else 0,
                    "semantic_score": similarity,
                    "semantic_relation": f"AI semantic match for '{meaningful_query}'",
                }
            )

        return matches

    except ImportError as exc:
        raise Exception("Sentence transformers not available") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise Exception(f"AI semantic search error: {exc}") from exc


def semantic_fallback_search(text: str, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """Enhanced semantic fallback using government terminology."""
    semantic_groups = {
        "recommend": [
            "recommend",
            "suggestion",
            "suggest",
            "advise",
            "propose",
            "urge",
            "advocate",
            "endorse",
            "recommendation",
            "recommendations",
        ],
        "suggest": ["suggest", "recommend", "proposal", "propose", "advise", "hint", "indicate", "suggestion", "suggestions"],
        "respond": ["respond", "response", "reply", "answer", "feedback", "reaction", "comment", "responses", "replies"],
        "response": ["response", "respond", "reply", "answer", "feedback", "reaction", "comment", "responses", "replies"],
        "implement": ["implement", "execute", "carry out", "put into practice", "apply", "deploy", "implementation"],
        "review": ["review", "examine", "assess", "evaluate", "analyze", "inspect", "analysis"],
        "policy": ["policy", "procedure", "guideline", "protocol", "framework", "strategy", "policies"],
        "accept": ["accept", "agree", "approve", "endorse", "support", "adopt", "acceptance"],
        "reject": ["reject", "decline", "refuse", "dismiss", "deny", "oppose", "rejection"],
        "government": ["government", "department", "ministry", "agency", "authority", "administration"],
        "report": ["report", "document", "paper", "study", "analysis", "investigation"],
        "committee": ["committee", "panel", "board", "commission", "group", "team"],
        "budget": ["budget", "funding", "financial", "cost", "expenditure", "allocation"],
        "urgent": ["urgent", "immediate", "critical", "priority", "emergency", "pressing"],
    }

    matches: List[Dict[str, Any]] = []
    meaningful_words = get_meaningful_words(query)
    if not meaningful_words:
        return matches

    for query_word in meaningful_words:
        related_words: List[str] = []
        if query_word in semantic_groups:
            related_words.extend(semantic_groups[query_word])

        for synonyms in semantic_groups.values():
            if any(query_word in synonym for synonym in synonyms):
                related_words.extend(synonyms)

        if not related_words:
            related_words = [query_word]
            if len(query_word) > 4:
                related_words.extend([query_word + "s", query_word + "ing", query_word + "ed", query_word + "ion"])

        related_words = list(set(related_words))
        sentences = re.split(r"[.!?]+", text)

        for i, sentence in enumerate(sentences):
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            sentence_search = sentence_clean if case_sensitive else sentence_clean.lower()
            for related_word in related_words:
                related_search = related_word if case_sensitive else related_word.lower()
                if related_search not in sentence_search:
                    continue

                if related_word.lower() == query_word.lower():
                    score = 100.0
                elif query_word.lower() in related_word.lower() or related_word.lower() in query_word.lower():
                    score = 95.0
                else:
                    score = 85.0

                pos = text.find(sentence_clean)
                if pos == -1:
                    pos = i * 100

                context = get_context_simple(sentences, i, 2)
                matches.append(
                    {
                        "position": pos,
                        "matched_text": sentence_clean,
                        "context": context,
                        "score": score,
                        "match_type": "semantic",
                        "page_number": max(1, pos // 2000 + 1),
                        "word_position": i,
                        "percentage_through": (pos / len(text)) * 100 if text else 0,
                        "semantic_relation": f"{query_word} → {related_word}",
                        "query_word": query_word,
                    }
                )
                break

    matches = remove_overlapping_matches(matches)
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


def hybrid_search_smart_ai(text: str, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """Combine smart and AI semantic search results."""
    smart_matches = smart_search_filtered(text, query, case_sensitive)
    semantic_matches = ai_semantic_search(text, query, case_sensitive)
    combined = smart_matches + semantic_matches

    seen = set()
    unique_matches = []
    for match in sorted(combined, key=lambda x: x.get("score", 0), reverse=True):
        key = (match.get("position"), match.get("matched_text"))
        if key in seen:
            continue
        seen.add(key)
        unique_matches.append(match)

    return unique_matches


def find_similar_content_filtered(
    documents: List[Dict[str, Any]], target_sentence: str, search_type: str, threshold: float, max_matches: int
) -> List[Dict[str, Any]]:
    """Find similar content using meaningful words only."""
    matches: List[Dict[str, Any]] = []
    target_meaningful = set(get_meaningful_words(target_sentence))
    if not target_meaningful:
        return matches

    for doc in documents:
        text = doc.get("text", "")
        if not text:
            continue

        sentences = re.split(r"[.!?]+", text)
        for i, sentence in enumerate(sentences):
            if not sentence.strip() or len(sentence.strip()) < 20:
                continue

            sentence_meaningful = set(get_meaningful_words(sentence))
            if not sentence_meaningful:
                continue

            intersection = len(target_meaningful & sentence_meaningful)
            union = len(target_meaningful | sentence_meaningful)
            similarity = intersection / union if union > 0 else 0
            if similarity < threshold:
                continue

            if search_type == "Recommendations":
                if not any(word in sentence.lower() for word in ["recommend", "suggest", "advise"]):
                    continue
            elif search_type == "Responses":
                if not any(word in sentence.lower() for word in ["accept", "reject", "agree", "implement"]):
                    continue

            context = get_context_simple(sentences, i, 2)
            position = text.find(sentence)
            matches.append(
                {
                    "sentence": sentence.strip(),
                    "context": context,
                    "similarity_score": similarity,
                    "document": doc,
                    "position": position,
                    "page_number": max(1, position // 2000 + 1) if position >= 0 else 1,
                    "content_type": classify_content_type(sentence),
                    "matched_meaningful_words": list(target_meaningful & sentence_meaningful),
                    "total_meaningful_words": len(target_meaningful),
                }
            )

    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matches[:max_matches]


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def filter_stop_words(query: str) -> str:
    """Remove stop words from query, keeping only meaningful words."""
    words = query.lower().split()
    meaningful_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    return " ".join(meaningful_words) if meaningful_words else query


def get_meaningful_words(text: str) -> List[str]:
    """Extract meaningful words (non-stop words) from text."""
    words = re.findall(r"\b\w+\b", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def get_context_simple(sentences: List[str], index: int, window: int = 1) -> str:
    """Get context around a sentence."""
    start = max(0, index - window)
    end = min(len(sentences), index + window + 1)
    context_sentences = [s.strip() for s in sentences[start:end] if s.strip()]
    return " ".join(context_sentences)


def remove_overlapping_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove overlapping matches, keeping the highest scored ones."""
    if not matches:
        return matches

    sorted_matches = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)
    unique_matches: List[Dict[str, Any]] = []
    used_positions = set()

    for match in sorted_matches:
        pos = match.get("position", 0)
        matched_text = match.get("matched_text", "")
        match_length = len(matched_text)

        overlap = False
        for used_start, used_end in used_positions:
            if not (pos + match_length <= used_start or pos >= used_end):
                overlap = True
                break

        if not overlap:
            unique_matches.append(match)
            used_positions.add((pos, pos + match_length))

    return unique_matches


def check_rag_availability() -> bool:
    """Check if RAG dependencies are available - STREAMLIT CLOUD OPTIMIZED."""
    try:
        import torch

        is_streamlit_cloud = os.getenv("STREAMLIT_SHARING_MODE") or "streamlit.app" in os.getenv("HOSTNAME", "") or "/mount/src/" in os.getcwd()
        if is_streamlit_cloud:
            return False

        from sentence_transformers import SentenceTransformer

        device = "cpu"
        torch.set_default_device("cpu")
        test_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        with torch.no_grad():
            test_model.encode(["test"], convert_to_tensor=False, device=device)
        return True

    except ImportError:
        return False
    except Exception:
        return False


__all__ = [
    "STOP_WORDS",
    "align_recommendations_with_responses",
    "calculate_simple_similarity",
    "check_rag_availability",
    "classify_content_type",
    "determine_alignment_status",
    "execute_search_with_ai",
    "exact_search_unfiltered",
    "find_pattern_matches",
    "find_similar_content_filtered",
    "fuzzy_search_filtered",
    "get_context_simple",
    "get_meaningful_words",
    "hybrid_search_smart_ai",
    "ai_semantic_search",
    "ai_semantic_search_direct",
    "semantic_fallback_search",
    "smart_search_filtered",
    "filter_stop_words",
    "remove_overlapping_matches",
]
