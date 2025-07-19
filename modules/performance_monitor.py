
# ===============================================
# FILE: modules/performance_monitor.py
# ===============================================

import time
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil

class PerformanceMonitor:
    """Monitor system performance and processing metrics"""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.active_operations = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def start_operation(self, operation_id: str, operation_type: str) -> str:
        """Start monitoring an operation"""
        with self._lock:
            start_time = time.time()
            self.active_operations[operation_id] = {
                'type': operation_type,
                'start_time': start_time,
                'start_memory': psutil.virtual_memory().used
            }
            return operation_id
    
    def end_operation(self, operation_id: str, 
                     items_processed: int = 1,
                     success: bool = True) -> Dict[str, Any]:
        """End monitoring an operation and record metrics"""
        with self._lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Operation {operation_id} not found")
                return {}
            
            operation = self.active_operations.pop(operation_id)
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            duration = end_time - operation['start_time']
            memory_delta = end_memory - operation['start_memory']
            
            metric = {
                'operation_type': operation['type'],
                'duration': duration,
                'items_processed': items_processed,
                'items_per_second': items_processed / duration if duration > 0 else 0,
                'memory_delta_mb': memory_delta / (1024 * 1024),
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            self.metrics[operation['type']].append(metric)
            return metric
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all operations"""
        summary = {}
        
        with self._lock:
            for op_type, metrics in self.metrics.items():
                if not metrics:
                    continue
                
                recent_metrics = [m for m in metrics if m['success']]
                
                if recent_metrics:
                    durations = [m['duration'] for m in recent_metrics]
                    throughputs = [m['items_per_second'] for m in recent_metrics]
                    memory_usage = [m['memory_delta_mb'] for m in recent_metrics]
                    
                    summary[op_type] = {
                        'total_operations': len(recent_metrics),
                        'avg_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations),
                        'avg_throughput': sum(throughputs) / len(throughputs),
                        'avg_memory_delta_mb': sum(memory_usage) / len(memory_usage),
                        'success_rate': len(recent_metrics) / len(list(metrics)),
                        'last_operation': recent_metrics[-1]['timestamp']
                    }
        
        return summary
    
    def get_recent_metrics(self, operation_type: str, 
                          minutes: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics for an operation type"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            if operation_type not in self.metrics:
                return []
            
            recent = []
            for metric in self.metrics[operation_type]:
                metric_time = datetime.fromisoformat(metric['timestamp'])
                if metric_time >= cutoff_time:
                    recent.append(metric)
            
            return recent
    
    def clear_metrics(self, operation_type: Optional[str] = None):
        """Clear metrics for specific operation type or all"""
        with self._lock:
            if operation_type:
                if operation_type in self.metrics:
                    self.metrics[operation_type].clear()
            else:
                self.metrics.clear()
