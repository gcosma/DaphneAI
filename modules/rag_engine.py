# ===============================================
# FILE: modules/rag_engine.py - CORRECTED VERSION
# ===============================================

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import time

# LangChain imports with version compatibility
try:
    # Try new LangChain 0.1.0+ imports first
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    LANGCHAIN_NEW = True
except ImportError:
    try:
        # Fallback to older LangChain imports
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        from langchain.schema.runnable import RunnablePassthrough
        LANGCHAIN_NEW = False
    except ImportError:
        # No LangChain available
        ChatOpenAI = None
        PromptTemplate = None
        StrOutputParser = None
        RunnablePassthrough = None
        LANGCHAIN_NEW = None

# OpenAI imports with version compatibility
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_NEW = True
except ImportError:
    try:
        import openai
        OpenAIClient = None
        OPENAI_NEW = False
    except ImportError:
        OpenAIClient = None
        OPENAI_NEW = None

from core_utils import Recommendation

class MockLLM:
    """Mock LLM for testing when API keys or libraries are unavailable"""
    
    def __init__(self, model_name="mock-model"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    def predict(self, text: str) -> str:
        """Mock prediction method"""
        self.logger.info("Using mock LLM response")
        return "This is a mock response. Please configure your OpenAI API key for full functionality."
    
    def __call__(self, prompt: str) -> str:
        return self.predict(prompt)
    
    def invoke(self, input_data) -> str:
        """For compatibility with new LangChain interfaces"""
        if isinstance(input_data, dict):
            return self.predict(str(input_data))
        return self.predict(str(input_data))

class RAGQueryEngine:
    """Enhanced RAG Query Engine with fallback mechanisms"""
    
    def __init__(self, vector_store_manager, llm_model="gpt-3.5-turbo"):
        self.vector_store = vector_store_manager
        self.llm_model = llm_model
        self.llm = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM with fallbacks
        self._initialize_llm()
        
        # Define RAG prompt template
        self.rag_prompt_template = """
You are an expert at analyzing recommendations and their responses.

Use the following context to answer questions about recommendations and responses:

Context: {context}

Question: {question}

Provide a detailed answer based on the context. If you cannot find relevant information
in the context, say so clearly. Include confidence scores and source references where possible.

Focus on:
1. Direct answers to the question
2. Relevant details from the context
3. Any implementation status or progress mentioned
4. Connections between recommendations and responses

Answer:
"""
        
        # Initialize processing chain
        self._initialize_chain()
    
    def _initialize_llm(self):
        """Initialize LLM with comprehensive fallbacks"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                self.logger.warning("OpenAI API key not found - using mock LLM")
                self.llm = MockLLM(self.llm_model)
                return
            
            # Try new OpenAI client first
            if OPENAI_NEW and OpenAIClient:
                try:
                    # Test the API key works
                    client = OpenAIClient(api_key=api_key)
                    # Simple test call
                    response = client.chat.completions.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    
                    # If test successful, create LangChain wrapper if available
                    if ChatOpenAI and LANGCHAIN_NEW is not None:
                        self.llm = ChatOpenAI(
                            model_name=self.llm_model,
                            temperature=0.1,
                            openai_api_key=api_key
                        )
                        self.logger.info(f"Initialized ChatOpenAI with model: {self.llm_model}")
                    else:
                        # Use direct OpenAI client
                        self.llm = OpenAIDirectWrapper(client, self.llm_model)
                        self.logger.info(f"Initialized direct OpenAI client with model: {self.llm_model}")
                    
                    return
                    
                except Exception as e:
                    self.logger.warning(f"New OpenAI client initialization failed: {e}")
            
            # Try older OpenAI API
            if not OPENAI_NEW and 'openai' in globals():
                try:
                    openai.api_key = api_key
                    # Test old API
                    response = openai.ChatCompletion.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    
                    if ChatOpenAI:
                        self.llm = ChatOpenAI(
                            model_name=self.llm_model,
                            temperature=0.1,
                            openai_api_key=api_key
                        )
                        self.logger.info(f"Initialized ChatOpenAI (legacy) with model: {self.llm_model}")
                    else:
                        self.llm = OpenAILegacyWrapper(self.llm_model)
                        self.logger.info(f"Initialized legacy OpenAI wrapper with model: {self.llm_model}")
                    
                    return
                    
                except Exception as e:
                    self.logger.warning(f"Legacy OpenAI initialization failed: {e}")
            
            # If all else fails, use mock
            self.logger.warning("All LLM initialization attempts failed - using mock LLM")
            self.llm = MockLLM(self.llm_model)
            
        except Exception as e:
            self.logger.error(f"Error during LLM initialization: {e}")
            self.llm = MockLLM(self.llm_model)
    
    def _initialize_chain(self):
        """Initialize the RAG processing chain"""
        try:
            if PromptTemplate and self.llm and not isinstance(self.llm, MockLLM):
                self.prompt = PromptTemplate.from_template(self.rag_prompt_template)
                
                # Try to create modern LangChain chain
                if LANGCHAIN_NEW and hasattr(self.llm, 'invoke'):
                    try:
                        if StrOutputParser and RunnablePassthrough:
                            self.chain = (
                                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                                | self.prompt
                                | self.llm
                                | StrOutputParser()
                            )
                            self.logger.info("Initialized modern LangChain chain")
                        else:
                            self.chain = None
                            self.logger.warning("LangChain components missing - using direct method")
                    except Exception as e:
                        self.logger.warning(f"Modern chain creation failed: {e}")
                        self.chain = None
                else:
                    self.chain = None
                    self.logger.info("Using legacy LLM interface")
            else:
                self.chain = None
                self.logger.warning("Chain initialization skipped - using fallback methods")
                
        except Exception as e:
            self.logger.error(f"Error initializing chain: {e}")
            self.chain = None
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Process RAG query with comprehensive error handling"""
        try:
            if not question or not question.strip():
                return {
                    "answer": "Please provide a valid question.",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Get relevant documents from vector store
            relevant_docs = []
            try:
                if self.vector_store:
                    relevant_docs = self.vector_store.similarity_search(question, k=5)
                    self.logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
                else:
                    self.logger.warning("Vector store not available")
            except Exception as e:
                self.logger.error(f"Error retrieving documents: {e}")
            
            # Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)
            
            # Generate answer
            answer = self._generate_answer(question, context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(answer, relevant_docs, question)
            
            # Prepare sources
            sources = []
            if include_sources:
                sources = self._format_sources(relevant_docs)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "context_length": len(context),
                "documents_retrieved": len(relevant_docs)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing RAG query: {e}")
            return {
                "answer": f"I encountered an error processing your query: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "error": str(e)
            }
    
    def _prepare_context(self, documents: List) -> str:
        """Prepare context string from retrieved documents"""
        try:
            if not documents:
                return "No relevant context found."
            
            context_parts = []
            for i, doc in enumerate(documents, 1):
                content = getattr(doc, 'page_content', str(doc))
                source = getattr(doc, 'metadata', {}).get('source', f'Document {i}')
                
                # Limit content length to prevent prompt overflow
                max_content_length = 500
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                context_parts.append(f"Source {i} ({source}):\n{content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error preparing context: {e}")
            return "Error preparing context from retrieved documents."
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        try:
            if isinstance(self.llm, MockLLM):
                return self.llm.predict(f"Question: {question}\nContext: {context[:200]}...")
            
            # Try modern LangChain chain first
            if self.chain and hasattr(self.chain, 'invoke'):
                try:
                    response = self.chain.invoke({
                        "question": question,
                        "context": context
                    })
                    return str(response)
                except Exception as e:
                    self.logger.warning(f"Modern chain invocation failed: {e}")
            
            # Try direct LLM invocation
            if hasattr(self.llm, 'invoke'):
                try:
                    prompt_text = self.rag_prompt_template.format(
                        question=question,
                        context=context
                    )
                    response = self.llm.invoke(prompt_text)
                    return str(response.content if hasattr(response, 'content') else response)
                except Exception as e:
                    self.logger.warning(f"Direct LLM invoke failed: {e}")
            
            # Try predict method
            if hasattr(self.llm, 'predict'):
                try:
                    prompt_text = self.rag_prompt_template.format(
                        question=question,
                        context=context
                    )
                    response = self.llm.predict(prompt_text)
                    return str(response)
                except Exception as e:
                    self.logger.warning(f"LLM predict failed: {e}")
            
            # Try call method
            if callable(self.llm):
                try:
                    prompt_text = self.rag_prompt_template.format(
                        question=question,
                        context=context
                    )
                    response = self.llm(prompt_text)
                    return str(response)
                except Exception as e:
                    self.logger.warning(f"LLM call failed: {e}")
            
            # If all methods fail
            return "I'm unable to generate a response at this time. Please check your API configuration."
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _calculate_confidence(self, answer: str, documents: List, question: str) -> float:
        """Calculate confidence score for the generated answer"""
        try:
            confidence = 0.5  # Base confidence
            
            # Factor 1: Answer length and completeness
            if len(answer) > 50:
                confidence += 0.1
            if len(answer) > 150:
                confidence += 0.1
            
            # Factor 2: Number of source documents
            if len(documents) > 0:
                confidence += 0.1
            if len(documents) > 2:
                confidence += 0.1
            
            # Factor 3: Check for uncertainty indicators
            uncertainty_phrases = [
                "i don't know", "unclear", "cannot determine", 
                "not sure", "unable to", "no information", "error"
            ]
            
            answer_lower = answer.lower()
            uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer_lower)
            confidence -= (uncertainty_count * 0.2)
            
            # Factor 4: Question-answer relevance (simple keyword check)
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(question_words.intersection(answer_words))
            if overlap > 2:
                confidence += 0.1
            
            # Ensure confidence is between 0 and 1
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _format_sources(self, documents: List) -> List[Dict[str, Any]]:
        """Format source documents for response"""
        try:
            sources = []
            for doc in documents:
                try:
                    content = getattr(doc, 'page_content', str(doc))
                    metadata = getattr(doc, 'metadata', {})
                    
                    # Limit content for display
                    display_content = content[:300] + "..." if len(content) > 300 else content
                    
                    source = {
                        "content": display_content,
                        "metadata": metadata,
                        "source": metadata.get('source', 'Unknown'),
                        "content_type": metadata.get('content_type', 'general')
                    }
                    sources.append(source)
                    
                except Exception as e:
                    self.logger.error(f"Error formatting source document: {e}")
                    continue
            
            return sources
            
        except Exception as e:
            self.logger.error(f"Error formatting sources: {e}")
            return []
    
    def find_responses_for_recommendation(self, recommendation: Recommendation) -> List[Dict[str, Any]]:
        """Find responses for a specific recommendation using enhanced search"""
        try:
            # Create enhanced query for finding responses
            query_parts = [
                f"response to {recommendation.text}",
                f"implementation of {recommendation.text[:100]}",
                f"action taken regarding {recommendation.text[:100]}",
                recommendation.text
            ]
            
            all_responses = []
            seen_sources = set()
            
            # Try multiple query strategies
            for query in query_parts:
                try:
                    if len(query.strip()) < 10:  # Skip very short queries
                        continue
                    
                    # Get similar documents
                    similar_docs = self.vector_store.similarity_search_with_score(query, k=8)
                    
                    for doc, score in similar_docs:
                        source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
                        
                        # Skip duplicates
                        if source in seen_sources:
                            continue
                        
                        # Check if this looks like a response document
                        if self._is_response_document(doc):
                            response_info = {
                                "text": getattr(doc, 'page_content', ''),
                                "source": source,
                                "similarity_score": float(score),
                                "metadata": getattr(doc, 'metadata', {}),
                                "recommendation_id": recommendation.id,
                                "query_used": query[:50] + "..." if len(query) > 50 else query
                            }
                            all_responses.append(response_info)
                            seen_sources.add(source)
                            
                            # Limit total responses to prevent overflow
                            if len(all_responses) >= 15:
                                break
                    
                    if len(all_responses) >= 15:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error in query '{query[:30]}...': {e}")
                    continue
            
            # Sort by similarity score and return top results
            all_responses.sort(key=lambda x: x['similarity_score'], reverse=True)
            return all_responses[:10]  # Return top 10 responses
            
        except Exception as e:
            self.logger.error(f"Error finding responses for recommendation {recommendation.id}: {e}")
            return []
    
    def _is_response_document(self, doc) -> bool:
        """Enhanced method to determine if document is a response"""
        try:
            # Check metadata first
            metadata = getattr(doc, 'metadata', {})
            doc_type = metadata.get('document_type', '').lower()
            content_type = metadata.get('content_type', '').lower()
            
            if 'response' in doc_type or 'response' in content_type:
                return True
            
            # Check content for response indicators
            content = getattr(doc, 'page_content', '').lower()
            
            # Strong response indicators
            strong_indicators = [
                "in response to", "responding to", "we have implemented",
                "action taken", "actions completed", "implementation",
                "following the recommendation", "as recommended",
                "we accept", "we reject", "under review",
                "progress report", "status update", "completion",
                "addressed", "resolved", "actioned"
            ]
            
            strong_matches = sum(1 for indicator in strong_indicators if indicator in content)
            
            # Weak response indicators
            weak_indicators = [
                "update", "progress", "status", "review", "assessment",
                "evaluation", "outcome", "result", "measure", "step"
            ]
            
            weak_matches = sum(1 for indicator in weak_indicators if indicator in content)
            
            # Decision logic
            if strong_matches >= 1:
                return True
            elif strong_matches == 0 and weak_matches >= 3:
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking if document is response: {e}")
            return False


class OpenAIDirectWrapper:
    """Direct wrapper for new OpenAI client"""
    
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    def predict(self, text: str) -> str:
        """Predict method for compatibility"""
        return self.invoke(text)
    
    def invoke(self, text: str) -> str:
        """Invoke method for new LangChain compatibility"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": str(text)}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI direct wrapper error: {e}")
            return f"Error generating response: {str(e)}"
    
    def __call__(self, text: str) -> str:
        return self.invoke(text)


class OpenAILegacyWrapper:
    """Wrapper for legacy OpenAI API"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    def predict(self, text: str) -> str:
        """Predict method for compatibility"""
        try:
            import openai
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": str(text)}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI legacy wrapper error: {e}")
            return f"Error generating response: {str(e)}"
    
    def invoke(self, text: str) -> str:
        """Invoke method for compatibility"""
        return self.predict(text)
    
    def __call__(self, text: str) -> str:
        return self.predict(text)
