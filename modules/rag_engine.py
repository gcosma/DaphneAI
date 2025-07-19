# ===============================================
# FILE: modules/rag_engine.py
# ===============================================

import os
import logging
from typing import List, Dict, Any, Tuple, Optional

try:
    from langchain.chains import RetrievalQA
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_FULL_AVAILABLE = True
except ImportError:
    try:
        from langchain.chains import RetrievalQA
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        LANGCHAIN_FULL_AVAILABLE = True
    except ImportError:
        LANGCHAIN_FULL_AVAILABLE = False

from core_utils import Recommendation
from vector_store import VectorStoreManager

class MockLLM:
    """Mock LLM for testing"""
    def __call__(self, prompt: str) -> str:
        return "This is a mock response. Please configure your API key for full functionality."
    
    def predict(self, text: str) -> str:
        return self.__call__(text)

class RAGQueryEngine:
    def __init__(self, vector_store_manager: VectorStoreManager, llm_model="gpt-3.5-turbo"):
        self.vector_store = vector_store_manager
        self.llm = None
        self.llm_model = llm_model
        self.logger = logging.getLogger(__name__)
        
        self._initialize_llm()
        
        if LANGCHAIN_FULL_AVAILABLE:
            self.rag_prompt = PromptTemplate(
                template="""
                You are an expert at analyzing recommendations and their responses.
                
                Use the following context to answer questions about recommendations and responses:
                
                Context: {context}
                
                Question: {question}
                
                Provide a detailed answer based on the context. If you cannot find relevant information
                in the context, say so clearly. Include confidence scores and source references where possible.
                
                Answer:
                """,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = None
            self._initialize_qa_chain()
    
    def _initialize_llm(self):
        """Initialize LLM"""
        try:
            if LANGCHAIN_FULL_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                self.llm = OpenAI(model_name=self.llm_model, temperature=0)
                self.logger.info("LLM initialized successfully")
            else:
                self.logger.warning("Using mock LLM - OpenAI key not found or LangChain unavailable")
                self.llm = MockLLM()
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            self.llm = MockLLM()
    
    def _initialize_qa_chain(self):
        """Initialize QA chain"""
        try:
            if (LANGCHAIN_FULL_AVAILABLE and 
                self.llm and 
                hasattr(self.vector_store.vector_store, 'as_retriever')):
                
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.vector_store.as_retriever(search_kwargs={"k": 5}),
                    chain_type_kwargs={"prompt": self.rag_prompt},
                    return_source_documents=True
                )
                self.logger.info("QA chain initialized successfully")
            else:
                self.logger.warning("Cannot initialize QA chain - using fallback method")
        except Exception as e:
            self.logger.error(f"Error initializing QA chain: {e}")
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Process RAG query"""
        try:
            if self.qa_chain:
                result = self.qa_chain({"query": question})
                
                response = {
                    "answer": result.get("result", "No answer generated"),
                    "confidence": self._calculate_confidence(result),
                    "sources": []
                }
                
                if include_sources and "source_documents" in result:
                    response["sources"] = self._format_sources(result["source_documents"])
                
                return response
            else:
                docs = self.vector_store.similarity_search(question, k=3)
                return {
                    "answer": f"Found {len(docs)} relevant documents. Please check the sources for details.",
                    "confidence": 0.5,
                    "sources": self._format_sources(docs)
                }
            
        except Exception as e:
            self.logger.error(f"Error processing RAG query: {e}")
            return {
                "answer": "Sorry, I encountered an error processing your query.",
                "confidence": 0.0,
                "sources": []
            }
    
    def find_responses_for_recommendation(self, recommendation: Recommendation) -> List[Dict[str, Any]]:
        """Find responses for a specific recommendation"""
        try:
            query = f"response implementation action taken regarding: {recommendation.text}"
            
            similar_docs = self.vector_store.similarity_search_with_score(query, k=10)
            
            responses = []
            for doc, score in similar_docs:
                if self._is_response_document(doc):
                    response_info = {
                        "text": doc.page_content,
                        "source": doc.metadata.get('source', 'Unknown'),
                        "similarity_score": float(score),
                        "metadata": doc.metadata,
                        "recommendation_id": recommendation.id
                    }
                    responses.append(response_info)
            
            responses.sort(key=lambda x: x['similarity_score'], reverse=True)
            return responses
            
        except Exception as e:
            self.logger.error(f"Error finding responses: {e}")
            return []
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate confidence score"""
        try:
            answer = result.get("result", "")
            sources = result.get("source_documents", [])
            
            confidence = 0.5
            
            if len(answer) > 100:
                confidence += 0.2
            
            if len(sources) > 1:
                confidence += 0.1 * min(len(sources), 3)
            
            uncertainty_phrases = ["i don't know", "unclear", "cannot determine", "not sure"]
            if any(phrase in answer.lower() for phrase in uncertainty_phrases):
                confidence -= 0.3
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _format_sources(self, source_docs: List) -> List[Dict[str, Any]]:
        """Format source documents"""
        sources = []
        for doc in source_docs:
            try:
                source = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": getattr(doc, 'metadata', {})
                }
                sources.append(source)
            except Exception as e:
                self.logger.error(f"Error formatting source: {e}")
                continue
        return sources
    
    def _is_response_document(self, doc) -> bool:
        """Determine if document is a response"""
        try:
            doc_type = doc.metadata.get('document_type', '').lower()
            if 'response' in doc_type:
                return True
            
            content = doc.page_content.lower()
            response_indicators = [
                "in response to", "responding to", "implementation", 
                "accepted", "rejected", "under review", "action taken",
                "following the recommendation", "as recommended",
                "we have implemented", "steps taken"
            ]
            
            return any(indicator in content for indicator in response_indicators)
            
        except Exception as e:
            self.logger.error(f"Error checking if document is response: {e}")
            return False

# ===============================================
# FILE: modules/recommendation_matcher.py
# ===============================================

import logging
from typing import List, Dict, Any, Tuple, Optional

from core_utils import Recommendation
from rag_engine import RAGQueryEngine
from bert_annotator import BERTConceptAnnotator

class RecommendationResponseMatcher:
    def __init__(self, rag_engine: RAGQueryEngine, bert_annotator: BERTConceptAnnotator):
        self.rag_engine = rag_engine
        self.bert_annotator = bert_annotator
        self.logger = logging.getLogger(__name__)
    
    def match_recommendation_to_responses(self, recommendation: Recommendation) -> List[Dict[str, Any]]:
        """Match recommendation to responses using multi-modal approach"""
        try:
            # RAG-based matching
            rag_responses = self.rag_engine.find_responses_for_recommendation(recommendation)
            
            # Annotate recommendation if needed
            if not recommendation.annotations:
                rec_annotations, _ = self.bert_annotator.annotate_text(recommendation.text)
                recommendation.annotations = rec_annotations
            
            # Enhanced matching with concept validation
            enhanced_matches = []
            
            for rag_response in rag_responses:
                try:
                    # Annotate response text
                    response_annotations, _ = self.bert_annotator.annotate_text(rag_response['text'])
                    
                    # Calculate concept overlap
                    concept_overlap = self._calculate_concept_overlap(
                        recommendation.annotations, 
                        response_annotations
                    )
                    
                    # Combined confidence score
                    combined_confidence = self._calculate_combined_confidence(
                        rag_response['similarity_score'],
                        concept_overlap['overlap_score']
                    )
                    
                    # Create enhanced match
                    enhanced_match = {
                        **rag_response,
                        'response_annotations': response_annotations,
                        'concept_overlap': concept_overlap,
                        'combined_confidence': combined_confidence,
                        'match_type': self._determine_match_type(combined_confidence)
                    }
                    
                    enhanced_matches.append(enhanced_match)
                    
                except Exception as e:
                    self.logger.error(f"Error processing individual match: {e}")
                    enhanced_matches.append({
                        **rag_response,
                        'response_annotations': {},
                        'concept_overlap': {'overlap_score': 0, 'shared_themes': []},
                        'combined_confidence': rag_response['similarity_score'],
                        'match_type': 'BASIC_MATCH'
                    })
            
            enhanced_matches.sort(key=lambda x: x['combined_confidence'], reverse=True)
            return enhanced_matches
            
        except Exception as e:
            self.logger.error(f"Error matching recommendation to responses: {e}")
            return []
    
    def batch_match_recommendations(self, recommendations: List[Recommendation]) -> Dict[str, List[Dict[str, Any]]]:
        """Match multiple recommendations to responses"""
        matches = {}
        
        for recommendation in recommendations:
            try:
                matches[recommendation.id] = self.match_recommendation_to_responses(recommendation)
            except Exception as e:
                self.logger.error(f"Error matching recommendation {recommendation.id}: {e}")
                matches[recommendation.id] = []
        
        return matches
    
    def _calculate_concept_overlap(self, rec_annotations: Dict, resp_annotations: Dict) -> Dict[str, Any]:
        """Calculate concept overlap between recommendation and response"""
        overlap_info = {
            'shared_frameworks': [],
            'shared_themes': [],
            'overlap_score': 0.0,
            'total_rec_concepts': 0,
            'total_resp_concepts': 0
        }
        
        if not rec_annotations or not resp_annotations:
            return overlap_info
        
        try:
            # Count total concepts
            for framework, themes in rec_annotations.items():
                overlap_info['total_rec_concepts'] += len(themes)
            
            for framework, themes in resp_annotations.items():
                overlap_info['total_resp_concepts'] += len(themes)
            
            # Find overlapping frameworks and themes
            shared_themes = []
            
            for framework in rec_annotations:
                if framework in resp_annotations:
                    overlap_info['shared_frameworks'].append(framework)
                    
                    rec_themes = {theme['theme'] for theme in rec_annotations[framework]}
                    resp_themes = {theme['theme'] for theme in resp_annotations[framework]}
                    
                    framework_shared = rec_themes.intersection(resp_themes)
                    for theme in framework_shared:
                        shared_themes.append(f"{framework}: {theme}")
            
            overlap_info['shared_themes'] = shared_themes
            
            # Calculate overlap score
            if overlap_info['total_rec_concepts'] > 0:
                overlap_info['overlap_score'] = len(shared_themes) / overlap_info['total_rec_concepts']
            
            return overlap_info
            
        except Exception as e:
            self.logger.error(f"Error calculating concept overlap: {e}")
            return overlap_info
    
    def _calculate_combined_confidence(self, semantic_score: float, concept_score: float) -> float:
        """Calculate combined confidence"""
        try:
            weights = {'semantic': 0.7, 'concept': 0.3}
            combined = (semantic_score * weights['semantic']) + (concept_score * weights['concept'])
            return min(max(combined, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating combined confidence: {e}")
            return semantic_score
    
    def _determine_match_type(self, confidence: float) -> str:
        """Determine match type based on confidence"""
        if confidence >= 0.85:
            return "HIGH_CONFIDENCE"
        elif confidence >= 0.65:
            return "MEDIUM_CONFIDENCE" 
        elif confidence >= 0.45:
            return "LOW_CONFIDENCE"
        else:
            return "POOR_MATCH"
