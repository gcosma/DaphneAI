# ===============================================
# FILE: modules/data_models.py
# ===============================================

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_recommendations: int = 0
    annotated_recommendations: int = 0
    matched_recommendations: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)

@dataclass
class FrameworkConfig:
    """Configuration for annotation frameworks"""
    name: str
    description: str
    themes: List[Dict[str, Any]]
    enabled: bool = True
    confidence_threshold: float = 0.65

@dataclass
class ExtractionConfig:
    """Configuration for recommendation extraction"""
    method: str = "hybrid"  # "ai", "pattern", "hybrid"
    confidence_threshold: float = 0.6
    max_recommendations: int = 100
    include_context: bool = True

@dataclass
class MatchingConfig:
    """Configuration for response matching"""
    similarity_threshold: float = 0.7
    concept_weight: float = 0.3
    semantic_weight: float = 0.7
    max_responses: int = 10

@dataclass
class SystemHealth:
    """System health monitoring"""
    vector_store_status: str = "unknown"
    bert_model_status: str = "unknown"
    llm_status: str = "unknown"
    last_check: Optional[str] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
