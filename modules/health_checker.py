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
