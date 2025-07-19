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


# ===============================================
# FILE: modules/health_checker.py
# ===============================================

import logging
import psutil
import torch
from typing import Dict, Any
from datetime import datetime
from data_models import SystemHealth

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_system_health(self, 
                          vector_store=None, 
                          bert_annotator=None, 
                          rag_engine=None) -> SystemHealth:
        """Comprehensive system health check"""
        
        health = SystemHealth()
        health.last_check = datetime.now().isoformat()
        
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            health.memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            health.disk_usage = disk.percent
            
            # Vector store status
            if vector_store:
                try:
                    stats = vector_store.get_collection_stats()
                    if stats.get('total_documents', 0) >= 0:
                        health.vector_store_status = "healthy"
                    else:
                        health.vector_store_status = "error"
                except Exception as e:
                    health.vector_store_status = f"error: {str(e)}"
                    self.logger.error(f"Vector store health check failed: {e}")
            
            # BERT model status
            if bert_annotator:
                try:
                    if bert_annotator.model is not None:
                        # Test embedding
                        test_emb = bert_annotator.get_bert_embedding("test")
                        if test_emb is not None:
                            health.bert_model_status = "healthy"
                        else:
                            health.bert_model_status = "degraded"
                    else:
                        health.bert_model_status = "not_loaded"
                except Exception as e:
                    health.bert_model_status = f"error: {str(e)}"
                    self.logger.error(f"BERT health check failed: {e}")
            
            # LLM status
            if rag_engine:
                try:
                    if hasattr(rag_engine.llm, 'predict'):
                        health.llm_status = "healthy"
                    else:
                        health.llm_status = "mock"
                except Exception as e:
                    health.llm_status = f"error: {str(e)}"
                    self.logger.error(f"LLM health check failed: {e}")
            
            return health
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health.vector_store_status = "error"
            health.bert_model_status = "error"
            health.llm_status = "error"
            return health
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        gpu_info = {
            "available": False,
            "count": 0,
            "memory_total": 0,
            "memory_used": 0,
            "devices": []
        }
        
        try:
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["count"] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_allocated = torch.cuda.memory_allocated(i)
                    
                    gpu_info["devices"].append({
                        "id": i,
                        "name": device_props.name,
                        "memory_total": memory_total,
                        "memory_allocated": memory_allocated,
                        "memory_free": memory_total - memory_allocated
                    })
            
        except Exception as e:
            self.logger.error(f"GPU info check failed: {e}")
        
        return gpu_info


# ===============================================
# FILE: modules/export_manager.py
# ===============================================

import json
import csv
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import zipfile
import io

class ExportManager:
    """Handle data export in various formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_full_analysis(self, 
                           documents: List[Dict],
                           recommendations: List,
                           annotations: Dict,
                           matches: Dict) -> bytes:
        """Export complete analysis as ZIP file"""
        
        try:
            # Create in-memory ZIP file
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Export documents summary
                docs_data = self._prepare_documents_data(documents)
                docs_csv = pd.DataFrame(docs_data).to_csv(index=False)
                zip_file.writestr("documents.csv", docs_csv)
                
                # Export recommendations
                recs_data = self._prepare_recommendations_data(recommendations)
                recs_csv = pd.DataFrame(recs_data).to_csv(index=False)
                zip_file.writestr("recommendations.csv", recs_csv)
                
                # Export annotations
                annotations_json = json.dumps(annotations, indent=2)
                zip_file.writestr("annotations.json", annotations_json)
                
                # Export matches
                matches_data = self._prepare_matches_data(matches, recommendations)
                if matches_data:
                    matches_csv = pd.DataFrame(matches_data).to_csv(index=False)
                    zip_file.writestr("matches.csv", matches_csv)
                
                # Export metadata
                metadata = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_documents": len(documents),
                    "total_recommendations": len(recommendations),
                    "total_annotations": len(annotations),
                    "total_matches": len(matches)
                }
                metadata_json = json.dumps(metadata, indent=2)
                zip_file.writestr("metadata.json", metadata_json)
            
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise
    
    def _prepare_documents_data(self, documents: List[Dict]) -> List[Dict]:
        """Prepare documents data for export"""
        data = []
        for doc in documents:
            data.append({
                "filename": doc.get('filename', 'Unknown'),
                "document_type": doc.get('document_type', 'Unknown'),
                "upload_time": doc.get('upload_time', ''),
                "page_count": doc.get('metadata', {}).get('page_count', 'N/A'),
                "file_size_kb": round(doc.get('metadata', {}).get('file_size', 0) / 1024, 1),
                "content_length": len(doc.get('content', ''))
            })
        return data
    
    def _prepare_recommendations_data(self, recommendations: List) -> List[Dict]:
        """Prepare recommendations data for export"""
        data = []
        for rec in recommendations:
            data.append({
                "id": rec.id,
                "text": rec.text,
                "document_source": rec.document_source,
                "section_title": rec.section_title,
                "page_number": rec.page_number,
                "confidence_score": rec.confidence_score,
                "text_length": len(rec.text),
                "has_annotations": len(rec.annotations) > 0 if hasattr(rec, 'annotations') else False
            })
        return data
    
    def _prepare_matches_data(self, matches: Dict, recommendations: List) -> List[Dict]:
        """Prepare matches data for export"""
        data = []
        
        # Create lookup for recommendations
        rec_lookup = {rec.id: rec for rec in recommendations}
        
        for rec_index, result in matches.items():
            if isinstance(rec_index, int) and rec_index < len(recommendations):
                rec = recommendations[rec_index]
            else:
                continue
                
            for response in result.get('responses', []):
                data.append({
                    "recommendation_id": rec.id,
                    "recommendation_text": rec.text[:200] + "..." if len(rec.text) > 200 else rec.text,
                    "recommendation_source": rec.document_source,
                    "response_source": response.get('source', 'Unknown'),
                    "response_text": response.get('text', '')[:200] + "..." if len(response.get('text', '')) > 200 else response.get('text', ''),
                    "similarity_score": response.get('similarity_score', 0),
                    "combined_confidence": response.get('combined_confidence', 0),
                    "match_type": response.get('match_type', 'UNKNOWN'),
                    "shared_themes": len(response.get('concept_overlap', {}).get('shared_themes', [])),
                    "search_timestamp": result.get('search_time', '')
                })
        
        return data
    
    def export_framework_analysis(self, annotations: Dict) -> pd.DataFrame:
        """Export detailed framework analysis"""
        
        analysis_data = []
        
        for rec_id, result in annotations.items():
            rec = result['recommendation']
            
            for framework, themes in result['annotations'].items():
                for theme in themes:
                    analysis_data.append({
                        "recommendation_id": rec_id,
                        "recommendation_text": rec.text[:100] + "...",
                        "framework": framework,
                        "theme": theme['theme'],
                        "confidence": theme['confidence'],
                        "semantic_similarity": theme['semantic_similarity'],
                        "keyword_count": theme['keyword_count'],
                        "matched_keywords": ", ".join(theme['matched_keywords'])
                    })
        
        return pd.DataFrame(analysis_data)
