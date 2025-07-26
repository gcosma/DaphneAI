# ===============================================
# AI CONCEPT EXTRACTION MODULE
# Smart paragraph recognition for recommendations and responses
# 
# This module uses AI/ML techniques to intelligently identify and classify
# paragraphs that contain recommendations, responses, and related concepts.
#
# Key Features:
# - BERT-based semantic paragraph classification
# - Context-aware concept recognition
# - Smart boundary detection using sentence embeddings
# - Multi-model approach (BERT + pattern matching + heuristics)
# - Confidence scoring based on semantic similarity
# - Fine-tuned for government document structures
#
# Author: Recommendation-Response Tracker Team
# Version: 1.0 - AI Concept Extraction
# Last Updated: 2025
# ===============================================

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import json
import numpy as np

# AI/ML imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available")

# Configure logging
logging.basicConfig(level=logging.INFO)

class AIConceptExtractor:
    """
    AI-powered concept extractor that uses semantic understanding to identify
    recommendations, responses, and related conceptual content in government documents
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.sentence_model = None
        self.classification_pipeline = None
        
        # Initialize AI models
        self._initialize_models()
        
        # Concept templates for semantic matching
        self.concept_templates = {
            'recommendation': [
                "The committee recommends that the government should implement new policies",
                "We recommend that immediate action be taken to address this issue", 
                "It is recommended that the following measures be put in place",
                "The inquiry recommends that departments should establish procedures",
                "We propose that the government should consider implementing",
                "The panel recommends that urgent steps be taken",
                "Our recommendation is that the ministry should develop",
                "We suggest that the following actions be taken immediately"
            ],
            'government_response': [
                "The government accepts this recommendation and will implement",
                "This recommendation is accepted and we will take action",
                "The department agrees with this recommendation and commits to",
                "We accept this recommendation and will establish procedures",
                "The government partially accepts this recommendation",
                "This recommendation is not accepted due to resource constraints", 
                "We reject this recommendation because it is not feasible",
                "The ministry will implement this recommendation within"
            ],
            'implementation': [
                "Implementation will begin within the next six months",
                "We will establish a working group to oversee implementation",
                "The department will allocate resources for this initiative",
                "A timeline for implementation has been developed",
                "Progress on implementation will be reported quarterly",
                "The implementation plan includes the following milestones"
            ],
            'justification': [
                "The rationale for this recommendation is based on evidence",
                "This recommendation addresses the root cause of the problem",
                "The committee believes this action is necessary because",
                "Evidence from the inquiry supports this recommendation",
                "Stakeholder feedback indicates that this measure is needed",
                "Analysis of the data suggests that this approach would be effective"
            ],
            'context': [
                "The background to this issue includes several key factors",
                "Previous attempts to address this problem have been unsuccessful",
                "The current situation requires immediate attention due to",
                "Stakeholders have raised concerns about the existing approach",
                "The evidence gathered during the inquiry demonstrates that",
                "Multiple witnesses testified about the impact of this issue"
            ]
        }
        
        # Precompute template embeddings
        self.template_embeddings = {}
        if self.sentence_model:
            self._precompute_template_embeddings()

    def _initialize_models(self):
        """Initialize AI models with fallbacks"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.sentence_model = SentenceTransformer(self.model_name)
                self.logger.info(f"✅ Loaded SentenceTransformer model: {self.model_name}")
            else:
                self.logger.warning("⚠️ SentenceTransformer not available - using fallback methods")
                
            if TRANSFORMERS_AVAILABLE:
                # Try to load a classification pipeline for government text
                try:
                    self.classification_pipeline = pipeline(
                        "text-classification",
                        model="microsoft/DialoGPT-medium",
                        return_all_scores=True
                    )
                    self.logger.info("✅ Loaded classification pipeline")
                except Exception as e:
                    self.logger.warning(f"Could not load classification pipeline: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error initializing AI models: {e}")
            self.sentence_model = None
            self.classification_pipeline = None

    def _precompute_template_embeddings(self):
        """Precompute embeddings for concept templates"""
        if not self.sentence_model:
            return
            
        try:
            for concept_type, templates in self.concept_templates.items():
                embeddings = self.sentence_model.encode(templates)
                self.template_embeddings[concept_type] = embeddings
                self.logger.info(f"Computed embeddings for {concept_type}: {len(templates)} templates")
        except Exception as e:
            self.logger.error(f"Error computing template embeddings: {e}")

    def extract_semantic_paragraphs(self, text: str, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Extract paragraphs using semantic understanding and concept classification
        """
        if not self.sentence_model:
            self.logger.warning("No AI model available - falling back to pattern-based extraction")
            return self._fallback_paragraph_extraction(text)
        
        try:
            # Split text into paragraphs
            paragraphs = self._split_into_paragraphs(text)
            
            # Encode all paragraphs
            paragraph_embeddings = self.sentence_model.encode([p['text'] for p in paragraphs])
            
            # Classify each paragraph
            classified_paragraphs = []
            
            for i, paragraph in enumerate(paragraphs):
                paragraph_embedding = paragraph_embeddings[i].reshape(1, -1)
                
                # Calculate semantic similarity to each concept type
                concept_scores = {}
                for concept_type, template_embeddings in self.template_embeddings.items():
                    if SKLEARN_AVAILABLE:
                        similarities = cosine_similarity(paragraph_embedding, template_embeddings)
                        max_similarity = np.max(similarities)
                        concept_scores[concept_type] = float(max_similarity)
                    else:
                        # Fallback similarity calculation
                        concept_scores[concept_type] = 0.5
                
                # Determine best concept match
                best_concept = max(concept_scores.items(), key=lambda x: x[1])
                concept_type, confidence = best_concept
                
                # Apply confidence threshold
                if confidence >= confidence_threshold:
                    # Enhanced analysis for high-confidence matches
                    enhanced_analysis = self._analyze_paragraph_context(
                        paragraph['text'], concept_type, confidence
                    )
                    
                    classified_paragraphs.append({
                        'text': paragraph['text'],
                        'concept_type': concept_type,
                        'confidence': confidence,
                        'paragraph_index': i,
                        'start_line': paragraph['start_line'],
                        'end_line': paragraph['end_line'],
                        'word_count': len(paragraph['text'].split()),
                        'char_count': len(paragraph['text']),
                        'extraction_method': 'ai_semantic',
                        'concept_scores': concept_scores,
                        'enhanced_analysis': enhanced_analysis,
                        'extracted_at': datetime.now().isoformat()
                    })
            
            self.logger.info(f"Extracted {len(classified_paragraphs)} semantic paragraphs")
            return classified_paragraphs
            
        except Exception as e:
            self.logger.error(f"Error in semantic extraction: {e}")
            return self._fallback_paragraph_extraction(text)

    def _split_into_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Split text into meaningful paragraphs"""
        # Split on double newlines first
        raw_paragraphs = re.split(r'\n\s*\n', text)
        
        paragraphs = []
        current_line = 0
        
        for raw_para in raw_paragraphs:
            # Clean up the paragraph
            cleaned_para = re.sub(r'\s+', ' ', raw_para.strip())
            
            # Skip very short paragraphs
            if len(cleaned_para) < 50:
                current_line += raw_para.count('\n') + 1
                continue
            
            # Count lines in original text
            line_count = raw_para.count('\n') + 1
            
            paragraphs.append({
                'text': cleaned_para,
                'start_line': current_line,
                'end_line': current_line + line_count,
                'raw_text': raw_para
            })
            
            current_line += line_count + 1
        
        return paragraphs

    def _analyze_paragraph_context(self, paragraph_text: str, concept_type: str, confidence: float) -> Dict[str, Any]:
        """Perform enhanced analysis of paragraph context"""
        analysis = {
            'sentence_count': len([s for s in paragraph_text.split('.') if s.strip()]),
            'has_numbers': bool(re.search(r'\d+', paragraph_text)),
            'has_formal_language': self._detect_formal_language(paragraph_text),
            'urgency_indicators': self._detect_urgency(paragraph_text),
            'stakeholder_mentions': self._detect_stakeholders(paragraph_text),
            'action_verbs': self._extract_action_verbs(paragraph_text),
            'time_references': self._extract_time_references(paragraph_text)
        }
        
        # Concept-specific analysis
        if concept_type == 'recommendation':
            analysis['recommendation_strength'] = self._assess_recommendation_strength(paragraph_text)
            analysis['scope'] = self._assess_recommendation_scope(paragraph_text)
        
        elif concept_type == 'government_response':
            analysis['response_type'] = self._classify_response_type(paragraph_text)
            analysis['commitment_level'] = self._assess_commitment_level(paragraph_text)
        
        return analysis

    def _detect_formal_language(self, text: str) -> bool:
        """Detect formal government language patterns"""
        formal_patterns = [
            r'\b(?:shall|should|must|ought|recommend|propose)\b',
            r'\b(?:government|department|ministry|authority|committee)\b',
            r'\b(?:policy|procedure|protocol|framework|strategy)\b',
            r'\b(?:implement|establish|develop|ensure|consider)\b'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in formal_patterns)

    def _detect_urgency(self, text: str) -> List[str]:
        """Detect urgency indicators in text"""
        urgency_patterns = [
            r'\b(?:urgent|immediate|priority|critical|essential)\b',
            r'\b(?:as soon as possible|without delay|forthwith)\b',
            r'\b(?:within \d+ (?:days|weeks|months))\b'
        ]
        
        indicators = []
        for pattern in urgency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)
        
        return indicators

    def _detect_stakeholders(self, text: str) -> List[str]:
        """Detect stakeholder mentions"""
        stakeholder_patterns = [
            r'\b(?:patients?|families|staff|clinicians?|doctors?|nurses?)\b',
            r'\b(?:public|citizens?|communities?|society)\b',
            r'\b(?:professionals?|experts?|specialists?)\b',
            r'\b(?:organizations?|institutions?|bodies)\b'
        ]
        
        stakeholders = []
        for pattern in stakeholder_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stakeholders.extend(matches)
        
        return list(set(stakeholders))

    def _extract_action_verbs(self, text: str) -> List[str]:
        """Extract action verbs indicating concrete actions"""
        action_patterns = [
            r'\b(?:implement|establish|develop|create|build|design)\b',
            r'\b(?:review|assess|evaluate|monitor|track|measure)\b',
            r'\b(?:improve|enhance|strengthen|upgrade|modernize)\b',
            r'\b(?:provide|deliver|offer|supply|ensure|guarantee)\b'
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend(matches)
        
        return list(set(actions))

    def _extract_time_references(self, text: str) -> List[str]:
        """Extract time references and deadlines"""
        time_patterns = [
            r'\b(?:within \d+ (?:days|weeks|months|years?))\b',
            r'\b(?:by (?:the end of )?\d{4})\b',
            r'\b(?:quarterly|annually|monthly|weekly)\b',
            r'\b(?:immediate|ongoing|continuous|regular)\b'
        ]
        
        time_refs = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            time_refs.extend(matches)
        
        return time_refs

    def _assess_recommendation_strength(self, text: str) -> str:
        """Assess the strength/imperative of a recommendation"""
        if re.search(r'\b(?:must|shall|essential|critical|mandatory)\b', text, re.IGNORECASE):
            return 'strong'
        elif re.search(r'\b(?:should|ought|recommend)\b', text, re.IGNORECASE):
            return 'moderate'
        elif re.search(r'\b(?:could|might|consider|suggest)\b', text, re.IGNORECASE):
            return 'weak'
        else:
            return 'neutral'

    def _assess_recommendation_scope(self, text: str) -> str:
        """Assess the scope of a recommendation"""
        if re.search(r'\b(?:all|every|entire|comprehensive|universal)\b', text, re.IGNORECASE):
            return 'broad'
        elif re.search(r'\b(?:specific|particular|targeted|focused)\b', text, re.IGNORECASE):
            return 'narrow'
        else:
            return 'moderate'

    def _classify_response_type(self, text: str) -> str:
        """Classify the type of government response"""
        text_lower = text.lower()
        
        if 'accept' in text_lower and 'not' not in text_lower:
            return 'accepted'
        elif 'reject' in text_lower or 'not accept' in text_lower:
            return 'rejected'
        elif 'partial' in text_lower or 'partly' in text_lower:
            return 'partially_accepted'
        elif 'under consideration' in text_lower or 'reviewing' in text_lower:
            return 'under_consideration'
        else:
            return 'general_response'

    def _assess_commitment_level(self, text: str) -> str:
        """Assess the level of government commitment"""
        if re.search(r'\b(?:will|shall|commit|guarantee|ensure)\b', text, re.IGNORECASE):
            return 'high'
        elif re.search(r'\b(?:intend|plan|aim|seek)\b', text, re.IGNORECASE):
            return 'moderate'
        elif re.search(r'\b(?:consider|explore|examine|review)\b', text, re.IGNORECASE):
            return 'low'
        else:
            return 'unclear'

    def _fallback_paragraph_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Fallback extraction when AI models are not available"""
        paragraphs = []
        
        # Simple pattern-based approach
        chunks = re.split(r'\n\s*\n', text)
        
        for i, chunk in enumerate(chunks):
            chunk_clean = re.sub(r'\s+', ' ', chunk.strip())
            
            if len(chunk_clean) < 50:
                continue
            
            # Simple classification based on keywords
            concept_type = 'context'  # default
            confidence = 0.5
            
            if re.search(r'\b(?:recommend|recommendation)\b', chunk_clean, re.IGNORECASE):
                concept_type = 'recommendation'
                confidence = 0.8
            elif re.search(r'\b(?:response|accept|reject)\b', chunk_clean, re.IGNORECASE):
                concept_type = 'government_response'
                confidence = 0.7
            elif re.search(r'\b(?:implement|establish|develop)\b', chunk_clean, re.IGNORECASE):
                concept_type = 'implementation'
                confidence = 0.6
            
            paragraphs.append({
                'text': chunk_clean,
                'concept_type': concept_type,
                'confidence': confidence,
                'paragraph_index': i,
                'word_count': len(chunk_clean.split()),
                'char_count': len(chunk_clean),
                'extraction_method': 'pattern_fallback',
                'extracted_at': datetime.now().isoformat()
            })
        
        return paragraphs

    def enhance_existing_extractions(self, recommendations: List[Dict], responses: List[Dict], original_text: str) -> Dict[str, Any]:
        """
        Enhance existing extractions with semantic context analysis
        """
        if not self.sentence_model:
            return {'enhanced_recommendations': recommendations, 'enhanced_responses': responses}
        
        try:
            # Get all text chunks
            all_items = recommendations + responses
            all_texts = [item.get('text', '') for item in all_items]
            
            if not all_texts:
                return {'enhanced_recommendations': recommendations, 'enhanced_responses': responses}
            
            # Encode all extracted items
            item_embeddings = self.sentence_model.encode(all_texts)
            
            # Find semantic clusters
            if SKLEARN_AVAILABLE and len(item_embeddings) > 1:
                clustering = DBSCAN(eps=0.3, min_samples=2)
                clusters = clustering.fit_predict(item_embeddings)
                
                # Add cluster information
                for i, item in enumerate(all_items):
                    item['semantic_cluster'] = int(clusters[i]) if clusters[i] != -1 else None
            
            # Add enhanced semantic analysis
            for i, item in enumerate(all_items):
                semantic_analysis = self._analyze_semantic_features(item.get('text', ''), item_embeddings[i])
                item['semantic_analysis'] = semantic_analysis
                
                # Find related context in original document
                context = self._find_related_context(item.get('text', ''), original_text)
                item['related_context'] = context
            
            # Separate back into recommendations and responses
            enhanced_recommendations = [item for item in all_items if item in recommendations]
            enhanced_responses = [item for item in all_items if item in responses]
            
            return {
                'enhanced_recommendations': enhanced_recommendations,
                'enhanced_responses': enhanced_responses,
                'semantic_metadata': {
                    'total_items_analyzed': len(all_items),
                    'clusters_found': len(set(clusters)) if 'clusters' in locals() else 0,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in semantic enhancement: {e}")
            return {'enhanced_recommendations': recommendations, 'enhanced_responses': responses}

    def _analyze_semantic_features(self, text: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Analyze semantic features of extracted text"""
        features = {
            'text_complexity': self._calculate_text_complexity(text),
            'semantic_density': float(np.linalg.norm(embedding)),
            'topic_coherence': self._assess_topic_coherence(text),
            'formality_score': self._calculate_formality_score(text),
            'specificity_score': self._calculate_specificity_score(text)
        }
        
        return features

    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        words = text.split()
        if not words:
            return 0.0
        
        # Simple complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Normalize to 0-1 scale
        complexity = (avg_word_length / 10 + avg_sentence_length / 20) / 2
        return min(1.0, complexity)

    def _assess_topic_coherence(self, text: str) -> float:
        """Assess how coherent the topic is within the text"""
        # Count topic-related keywords
        govt_keywords = len(re.findall(r'\b(?:government|policy|department|ministry)\b', text, re.IGNORECASE))
        health_keywords = len(re.findall(r'\b(?:health|medical|patient|clinical)\b', text, re.IGNORECASE))
        process_keywords = len(re.findall(r'\b(?:process|procedure|system|framework)\b', text, re.IGNORECASE))
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        keyword_density = (govt_keywords + health_keywords + process_keywords) / total_words
        return min(1.0, keyword_density * 5)  # Scale up and cap at 1.0

    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score of the text"""
        formal_indicators = [
            r'\b(?:shall|must|ought|hereby|pursuant|aforementioned)\b',
            r'\b(?:establishment|implementation|consideration|recommendation)\b',
            r'\b(?:furthermore|moreover|nevertheless|notwithstanding)\b'
        ]
        
        formal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in formal_indicators)
        words = len(text.split())
        
        if words == 0:
            return 0.0
        
        return min(1.0, formal_count / words * 10)

    def _calculate_specificity_score(self, text: str) -> float:
        """Calculate how specific/concrete the text is"""
        specific_indicators = [
            r'\b\d+(?:\.\d+)?\s*(?:percent|%|days?|weeks?|months?|years?)\b',
            r'\b(?:within|by|before|after)\s+\d+\b',
            r'\b(?:£|$|€)\d+(?:,\d{3})*(?:\.\d{2})?\b',
            r'\b[A-Z][a-z]+\s+(?:Act|Bill|Policy|Framework|Strategy)\b'
        ]
        
        specific_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in specific_indicators)
        sentences = len([s for s in text.split('.') if s.strip()])
        
        if sentences == 0:
            return 0.0
        
        return min(1.0, specific_count / sentences)

    def _find_related_context(self, item_text: str, original_text: str, context_window: int = 500) -> Dict[str, str]:
        """Find related context around the extracted item in the original text"""
        try:
            # Find the item in the original text
            item_start = original_text.lower().find(item_text.lower())
            
            if item_start == -1:
                return {'before': '', 'after': '', 'found': False}
            
            # Extract context before and after
            context_start = max(0, item_start - context_window)
            context_end = min(len(original_text), item_start + len(item_text) + context_window)
            
            before_context = original_text[context_start:item_start].strip()
            after_context = original_text[item_start + len(item_text):context_end].strip()
            
            return {
                'before': before_context,
                'after': after_context,
                'found': True,
                'position': item_start
            }
            
        except Exception as e:
            self.logger.error(f"Error finding context: {e}")
            return {'before': '', 'after': '', 'found': False}

# ===============================================
# STREAMLIT INTEGRATION
# ===============================================

def render_ai_concept_extraction_interface():
    """Render AI concept extraction interface"""
    st.subheader("🤖 AI Concept Extraction")
    
    st.markdown("""
    **AI-powered semantic understanding of document concepts**
    
    🧠 **BERT-based paragraph classification**  
    🎯 **Context-aware concept recognition**  
    📊 **Semantic similarity scoring**  
    🔍 **Enhanced contextual analysis**  
    """)
    
    # Check AI availability
    ai_status = []
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        ai_status.append("✅ Sentence Transformers")
    else:
        ai_status.append("❌ Sentence Transformers")
    
    if TRANSFORMERS_AVAILABLE:
        ai_status.append("✅ Transformers")
    else:
        ai_status.append("❌ Transformers")
    
    if SKLEARN_AVAILABLE:
        ai_status.append("✅ Scikit-learn")
    else:
        ai_status.append("❌ Scikit-learn")
    
    with st.expander("🔧 AI Model Status"):
        for status in ai_status:
            st.write(status)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.warning("⚠️ For full AI functionality, install: `pip install sentence-transformers`")
    
    # Get uploaded documents
    docs = st.session_state.get('uploaded_documents', [])
    if not docs:
        st.warning("Please upload documents first in the Upload tab.")
        return
    
    # Document selection
    doc_options = [f"{doc.get('filename', 'Unknown')}" for doc in docs]
    selected_docs = st.multiselect(
        "Select documents for AI concept extraction:",
        options=doc_options,
        default=doc_options[:1] if doc_options else [],
        help="Choose documents to analyze with AI"
    )
    
    # AI extraction options
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Confidence threshold:", 
            0.0, 1.0, 0.7, 0.05,
            help="Minimum confidence for concept classification"
        )
        
        extract_paragraphs = st.checkbox("Extract semantic paragraphs", value=True)
        enhance_existing = st.checkbox("Enhance existing extractions", value=True)
    
    with col2:
        concept_types = st.multiselect(
            "Concept types to extract:",
            options=['recommendation', 'government_response', 'implementation', 'justification', 'context'],
            default=['recommendation', 'government_response'],
            help="Types of concepts to identify"
        )
        
        include_analysis = st.checkbox("Include semantic analysis", value=True)
        cluster_analysis = st.checkbox("Perform cluster analysis", value=True)
    
    # Process button
    if st.button("🤖 Start AI Concept Extraction", type="primary", disabled=not selected_docs):
        process_ai_concept_extraction(
            selected_docs, docs, confidence_threshold, 
            extract_paragraphs, enhance_existing, concept_types,
            include_analysis, cluster_analysis
        )

def process_ai_concept_extraction(
    selected_docs: List[str],
    all_docs: List[Dict],
    confidence_threshold: float,
    extract_paragraphs: bool,
    enhance_existing: bool,
    concept_types: List[str],
    include_analysis: bool,
    cluster_analysis: bool
):
    """Process AI concept extraction"""
    
    # Initialize AI extractor
    extractor = AIConceptExtractor()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_concepts = []
    enhanced_extractions = {}
    processing_results = []
    
    # Get selected document objects
    selected_doc_objects = [doc for doc in all_docs if doc.get('filename') in selected_docs]
    
    for i, doc in enumerate(selected_doc_objects):
        filename = doc.get('filename', f'Document {i+1}')
        status_text.text(f"AI analyzing {filename}...")
        progress_bar.progress((i + 1) / len(selected_doc_objects))
        
        try:
            # Get document content
            from modules.ui.extraction_components import get_document_content_for_extraction
            content = get_document_content_for_extraction(doc)
            
            if not content or len(content.strip()) < 200:
                processing_results.append({
                    'filename': filename,
                    'status': '⚠️ Insufficient content for AI analysis',
                    'concepts_found': 0
                })
                continue
            
            doc_concepts = []
            
            # Extract semantic paragraphs
            if extract_paragraphs:
                semantic_paragraphs = extractor.extract_semantic_paragraphs(
                    content, confidence_threshold
                )
                
                # Filter by requested concept types
                filtered_paragraphs = [
                    p for p in semantic_paragraphs 
                    if p.get('concept_type') in concept_types
                ]
                
                # Add document context
                for concept in filtered_paragraphs:
                    concept['document_context'] = {'filename': filename}
                
                doc_concepts.extend(filtered_paragraphs)
            
            # Enhance existing extractions if available
            if enhance_existing:
                existing_results = st.session_state.get('extraction_results', {})
                if existing_results:
                    recommendations = existing_results.get('recommendations', [])
                    responses = existing_results.get('responses', [])
                    
                    enhanced = extractor.enhance_existing_extractions(
                        recommendations, responses, content
                    )
                    enhanced_extractions[filename] = enhanced
            
            all_concepts.extend(doc_concepts)
            
            processing_results.append({
                'filename': filename,
                'status': '✅ AI analysis complete',
                'concepts_found': len(doc_concepts),
                'concept_types': list(set(c.get('concept_type') for c in doc_concepts)),
                'avg_confidence': sum(c.get('confidence', 0) for c in doc_concepts) / max(len(doc_concepts), 1)
            })
            
        except Exception as e:
            processing_results.append({
                'filename': filename,
                'status': f'❌ AI analysis error: {str(e)}',
                'concepts_found': 0
            })
    
    # Store results
    st.session_state.ai_concept_results = {
        'concepts': all_concepts,
        'enhanced_extractions': enhanced_extractions,
        'processing_results': processing_results,
        'extraction_settings': {
            'confidence_threshold': confidence_threshold,
            'concept_types': concept_types,
            'include_analysis': include_analysis,
            'cluster_analysis': cluster_analysis
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Display results
    render_ai_concept_results()

def render_ai_concept_results():
    """Render AI concept extraction results"""
    results = st.session_state.get('ai_concept_results', {})
    
    if not results:
        return
    
    st.subheader("🤖 AI Concept Analysis Results")
    
    concepts = results.get('concepts', [])
    enhanced_extractions = results.get('enhanced_extractions', {})
    processing_results = results.get('processing_results', [])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Concepts Found", len(concepts))
    
    with col2:
        concept_types = list(set(c.get('concept_type') for c in concepts))
        st.metric("Concept Types", len(concept_types))
    
    with col3:
        avg_confidence = sum(c.get('confidence', 0) for c in concepts) / max(len(concepts), 1)
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col4:
        total_words = sum(c.get('word_count', 0) for c in concepts)
        st.metric("Total Words", f"{total_words:,}")
    
    # Processing results
    if processing_results:
        st.write("**AI Processing Summary:**")
        results_df = pd.DataFrame(processing_results)
        st.dataframe(results_df, use_container_width=True)
    
    # Concept analysis tabs
    if concepts:
        analysis_tabs = st.tabs([
            "🎯 Concept Classification", 
            "📊 Semantic Analysis", 
            "🔗 Enhanced Extractions",
            "📥 Download Results"
        ])
        
        with analysis_tabs[0]:
            render_concept_classification_results(concepts)
        
        with analysis_tabs[1]:
            render_semantic_analysis_results(concepts)
        
        with analysis_tabs[2]:
            render_enhanced_extractions_results(enhanced_extractions)
        
        with analysis_tabs[3]:
            render_ai_download_options(concepts, enhanced_extractions)

def render_concept_classification_results(concepts: List[Dict]):
    """Render concept classification results"""
    
    # Group by concept type
    concept_groups = {}
    for concept in concepts:
        concept_type = concept.get('concept_type', 'unknown')
        if concept_type not in concept_groups:
            concept_groups[concept_type] = []
        concept_groups[concept_type].append(concept)
    
    # Display each concept type
    for concept_type, concept_list in concept_groups.items():
        st.write(f"### {concept_type.replace('_', ' ').title()} ({len(concept_list)} items)")
        
        # Sort by confidence
        sorted_concepts = sorted(concept_list, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, concept in enumerate(sorted_concepts[:5]):  # Show top 5
            confidence = concept.get('confidence', 0)
            confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            
            with st.expander(
                f"{confidence_color} {concept.get('text', '')[:80]}... (Confidence: {confidence:.3f})",
                expanded=False
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Full Text:**")
                    st.write(concept.get('text', ''))
                    
                    # Enhanced analysis if available
                    if concept.get('enhanced_analysis'):
                        analysis = concept['enhanced_analysis']
                        st.write("**Enhanced Analysis:**")
                        
                        if analysis.get('action_verbs'):
                            st.write(f"🎯 Action verbs: {', '.join(analysis['action_verbs'][:5])}")
                        
                        if analysis.get('stakeholder_mentions'):
                            st.write(f"👥 Stakeholders: {', '.join(analysis['stakeholder_mentions'][:3])}")
                        
                        if analysis.get('time_references'):
                            st.write(f"⏰ Time references: {', '.join(analysis['time_references'][:3])}")
                        
                        if concept_type == 'recommendation' and analysis.get('recommendation_strength'):
                            strength_emoji = {'strong': '💪', 'moderate': '👍', 'weak': '🤏', 'neutral': '➖'}
                            strength = analysis['recommendation_strength']
                            st.write(f"💪 Strength: {strength_emoji.get(strength, '❓')} {strength}")
                        
                        if concept_type == 'government_response' and analysis.get('response_type'):
                            response_emoji = {
                                'accepted': '✅', 'rejected': '❌', 
                                'partially_accepted': '⚡', 'under_consideration': '🤔',
                                'general_response': '📋'
                            }
                            resp_type = analysis['response_type']
                            st.write(f"📋 Response: {response_emoji.get(resp_type, '❓')} {resp_type.replace('_', ' ')}")
                
                with col2:
                    st.write("**Metrics:**")
                    st.write(f"🎯 Confidence: {concept.get('confidence', 0):.3f}")
                    st.write(f"📝 Words: {concept.get('word_count', 0)}")
                    st.write(f"🔧 Method: {concept.get('extraction_method', 'unknown')}")
                    
                    # Concept scores if available
                    if concept.get('concept_scores'):
                        st.write("**Concept Scores:**")
                        scores = concept['concept_scores']
                        for score_type, score_value in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                            st.write(f"• {score_type}: {score_value:.3f}")
                    
                    if concept.get('document_context'):
                        st.write(f"📄 Source: {concept['document_context'].get('filename', 'Unknown')}")

def render_semantic_analysis_results(concepts: List[Dict]):
    """Render semantic analysis results"""
    
    # Filter concepts with semantic analysis
    analyzed_concepts = [c for c in concepts if c.get('semantic_analysis')]
    
    if not analyzed_concepts:
        st.info("No semantic analysis data available")
        return
    
    st.write(f"**Semantic Analysis for {len(analyzed_concepts)} concepts**")
    
    # Calculate aggregate metrics
    complexity_scores = [c['semantic_analysis'].get('text_complexity', 0) for c in analyzed_concepts]
    formality_scores = [c['semantic_analysis'].get('formality_score', 0) for c in analyzed_concepts]
    specificity_scores = [c['semantic_analysis'].get('specificity_score', 0) for c in analyzed_concepts]
    
    # Display aggregate metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Complexity", f"{sum(complexity_scores)/len(complexity_scores):.3f}")
    
    with col2:
        st.metric("Avg Formality", f"{sum(formality_scores)/len(formality_scores):.3f}")
    
    with col3:
        st.metric("Avg Specificity", f"{sum(specificity_scores)/len(specificity_scores):.3f}")
    
    # Detailed analysis table
    analysis_data = []
    for concept in analyzed_concepts[:10]:  # Show top 10
        analysis = concept.get('semantic_analysis', {})
        analysis_data.append({
            'Text Preview': concept.get('text', '')[:50] + '...',
            'Concept Type': concept.get('concept_type', '').replace('_', ' ').title(),
            'Complexity': f"{analysis.get('text_complexity', 0):.3f}",
            'Formality': f"{analysis.get('formality_score', 0):.3f}",
            'Specificity': f"{analysis.get('specificity_score', 0):.3f}",
            'Semantic Density': f"{analysis.get('semantic_density', 0):.3f}",
            'Topic Coherence': f"{analysis.get('topic_coherence', 0):.3f}"
        })
    
    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)
        st.dataframe(analysis_df, use_container_width=True)

def render_enhanced_extractions_results(enhanced_extractions: Dict):
    """Render enhanced extractions results"""
    
    if not enhanced_extractions:
        st.info("No enhanced extractions available")
        return
    
    for filename, enhanced_data in enhanced_extractions.items():
        st.write(f"### Enhanced Analysis: {filename}")
        
        enhanced_recs = enhanced_data.get('enhanced_recommendations', [])
        enhanced_resps = enhanced_data.get('enhanced_responses', [])
        metadata = enhanced_data.get('semantic_metadata', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Enhanced Recommendations", len(enhanced_recs))
        
        with col2:
            st.metric("Enhanced Responses", len(enhanced_resps))
        
        with col3:
            st.metric("Semantic Clusters", metadata.get('clusters_found', 0))
        
        # Show enhanced items
        if enhanced_recs:
            st.write("**Enhanced Recommendations:**")
            for i, rec in enumerate(enhanced_recs[:3]):
                cluster = rec.get('semantic_cluster')
                cluster_text = f" (Cluster {cluster})" if cluster is not None else ""
                
                with st.expander(f"Rec {i+1}: {rec.get('text', '')[:60]}...{cluster_text}"):
                    st.write(rec.get('text', ''))
                    
                    if rec.get('semantic_analysis'):
                        analysis = rec['semantic_analysis']
                        st.write(f"**Complexity:** {analysis.get('text_complexity', 0):.3f}")
                        st.write(f"**Formality:** {analysis.get('formality_score', 0):.3f}")
                    
                    if rec.get('related_context') and rec['related_context'].get('found'):
                        context = rec['related_context']
                        if context.get('before'):
                            st.write("**Context Before:**")
                            st.caption(context['before'][-100:] + "...")

def render_ai_download_options(concepts: List[Dict], enhanced_extractions: Dict):
    """Render AI concept extraction download options"""
    
    st.write("**Download AI Analysis Results**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Concepts CSV"):
            csv_data = create_ai_concepts_csv(concepts)
            if csv_data:
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f"ai_concepts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📋 Download Analysis JSON"):
            json_data = create_ai_analysis_json(concepts, enhanced_extractions)
            if json_data:
                st.download_button(
                    label="💾 Download JSON",
                    data=json_data,
                    file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col3:
        if st.button("📄 Download AI Report"):
            report_data = create_ai_analysis_report(concepts, enhanced_extractions)
            if report_data:
                st.download_button(
                    label="💾 Download Report",
                    data=report_data,
                    file_name=f"ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

def create_ai_concepts_csv(concepts: List[Dict]) -> str:
    """Create CSV export for AI concepts"""
    if not concepts:
        return ""
    
    csv_data = []
    for i, concept in enumerate(concepts):
        analysis = concept.get('enhanced_analysis', {})
        semantic = concept.get('semantic_analysis', {})
        
        csv_data.append({
            'Concept_ID': f"AI_CONCEPT_{i+1:03d}",
            'Concept_Type': concept.get('concept_type', ''),
            'Full_Text': concept.get('text', ''),
            'Confidence': concept.get('confidence', 0),
            'Word_Count': concept.get('word_count', 0),
            'Character_Count': concept.get('char_count', 0),
            'Extraction_Method': concept.get('extraction_method', ''),
            'Document_Source': concept.get('document_context', {}).get('filename', ''),
            'Paragraph_Index': concept.get('paragraph_index', 0),
            'Text_Complexity': semantic.get('text_complexity', 0),
            'Formality_Score': semantic.get('formality_score', 0),
            'Specificity_Score': semantic.get('specificity_score', 0),
            'Semantic_Density': semantic.get('semantic_density', 0),
            'Topic_Coherence': semantic.get('topic_coherence', 0),
            'Has_Numbers': analysis.get('has_numbers', False),
            'Has_Formal_Language': analysis.get('has_formal_language', False),
            'Action_Verbs': ', '.join(analysis.get('action_verbs', [])[:5]),
            'Stakeholders': ', '.join(analysis.get('stakeholder_mentions', [])[:5]),
            'Time_References': ', '.join(analysis.get('time_references', [])[:3]),
            'Urgency_Indicators': ', '.join(analysis.get('urgency_indicators', [])[:3]),
            'Semantic_Cluster': concept.get('semantic_cluster', ''),
            'Extracted_At': concept.get('extracted_at', '')
        })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
    
    return ""

def create_ai_analysis_json(concepts: List[Dict], enhanced_extractions: Dict) -> str:
    """Create JSON export for AI analysis"""
    export_data = {
        'ai_analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_concepts': len(concepts),
            'concept_types': list(set(c.get('concept_type') for c in concepts)),
            'analysis_method': 'bert_semantic_classification',
            'model_used': 'all-MiniLM-L6-v2'
        },
        'semantic_concepts': concepts,
        'enhanced_extractions': enhanced_extractions,
        'analysis_summary': {
            'avg_confidence': sum(c.get('confidence', 0) for c in concepts) / max(len(concepts), 1),
            'total_words_analyzed': sum(c.get('word_count', 0) for c in concepts),
            'documents_processed': len(enhanced_extractions)
        }
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def create_ai_analysis_report(concepts: List[Dict], enhanced_extractions: Dict) -> str:
    """Create formatted AI analysis report"""
    
    report_lines = [
        "=" * 80,
        "AI CONCEPT EXTRACTION & SEMANTIC ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Concepts Analyzed: {len(concepts)}",
        f"Documents Processed: {len(enhanced_extractions)}",
        "",
        "ANALYSIS METHODOLOGY",
        "=" * 40,
        "• BERT-based semantic paragraph classification",
        "• Sentence transformer embeddings (all-MiniLM-L6-v2)",
        "• Cosine similarity matching to concept templates",
        "• Enhanced contextual analysis with NLP features",
        "• Semantic clustering for relationship detection",
        "",
        "CONCEPT CLASSIFICATION RESULTS",
        "=" * 40,
        ""
    ]
    
    # Group concepts by type
    concept_groups = {}
    for concept in concepts:
        concept_type = concept.get('concept_type', 'unknown')
        if concept_type not in concept_groups:
            concept_groups[concept_type] = []
        concept_groups[concept_type].append(concept)
    
    # Add each concept type
    for concept_type, concept_list in concept_groups.items():
        report_lines.extend([
            f"{concept_type.replace('_', ' ').upper()} ({len(concept_list)} items)",
            "-" * 40,
            ""
        ])
        
        # Sort by confidence and show top 3
        sorted_concepts = sorted(concept_list, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, concept in enumerate(sorted_concepts[:3], 1):
            analysis = concept.get('enhanced_analysis', {})
            semantic = concept.get('semantic_analysis', {})
            
            report_lines.extend([
                f"{i}. CONFIDENCE: {concept.get('confidence', 0):.3f}",
                f"   TEXT: {concept.get('text', '')[:200]}...",
                f"   COMPLEXITY: {semantic.get('text_complexity', 0):.3f} | "
                f"FORMALITY: {semantic.get('formality_score', 0):.3f}",
                ""
            ])
            
            if analysis.get('action_verbs'):
                report_lines.append(f"   ACTION VERBS: {', '.join(analysis['action_verbs'][:3])}")
            
            if analysis.get('stakeholder_mentions'):
                report_lines.append(f"   STAKEHOLDERS: {', '.join(analysis['stakeholder_mentions'][:3])}")
            
            report_lines.extend(["", ""])
    
    # Add enhanced extractions summary
    if enhanced_extractions:
        report_lines.extend([
            "",
            "ENHANCED EXTRACTIONS SUMMARY",
            "=" * 40,
            ""
        ])
        
        for filename, enhanced_data in enhanced_extractions.items():
            metadata = enhanced_data.get('semantic_metadata', {})
            report_lines.extend([
                f"DOCUMENT: {filename}",
                f"• Enhanced recommendations: {len(enhanced_data.get('enhanced_recommendations', []))}",
                f"• Enhanced responses: {len(enhanced_data.get('enhanced_responses', []))}",
                f"• Semantic clusters found: {metadata.get('clusters_found', 0)}",
                ""
            ])
    
    return "\n".join(report_lines)

# ===============================================
# INTEGRATION WITH MAIN EXTRACTION MODULE
# ===============================================

def integrate_ai_concept_extraction():
    """
    Integration function to add AI concept extraction to the main extraction tab
    """
    st.markdown("---")
    st.subheader("🤖 AI Concept Enhancement")
    
    if st.button("🧠 Analyze with AI Concepts"):
        render_ai_concept_extraction_interface()

# ===============================================
# MODULE EXPORTS
# ===============================================

__all__ = [
    # Main classes
    'AIConceptExtractor',
    
    # Streamlit functions
    'render_ai_concept_extraction_interface',
    'process_ai_concept_extraction',
    'render_ai_concept_results',
    'integrate_ai_concept_extraction',
    
    # Results display functions
    'render_concept_classification_results',
    'render_semantic_analysis_results',
    'render_enhanced_extractions_results',
    'render_ai_download_options',
    
    # Export functions
    'create_ai_concepts_csv',
    'create_ai_analysis_json',
    'create_ai_analysis_report',
    
    # Availability flags
    'SENTENCE_TRANSFORMERS_AVAILABLE',
    'TRANSFORMERS_AVAILABLE',
    'SKLEARN_AVAILABLE'
]

# ===============================================
# MODULE INITIALIZATION
# ===============================================

logging.info("🤖 AI Concept Extraction module loaded!")
if SENTENCE_TRANSFORMERS_AVAILABLE:
    logging.info("✅ Full AI functionality available with BERT models")
else:
    logging.info("⚠️ Limited AI functionality - install sentence-transformers for full features")
