# ===============================================
# FILE: modules/async_processor.py
# ===============================================

import asyncio
import concurrent.futures
import threading
from typing import List, Dict, Any, Callable, Optional
import logging
from datetime import datetime

class AsyncProcessor:
    """Handle asynchronous processing tasks"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
    
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """Submit task for async processing"""
        try:
            future = self.executor.submit(func, *args, **kwargs)
            
            self.active_tasks[task_id] = {
                'future': future,
                'start_time': datetime.now(),
                'status': 'running'
            }
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error submitting task {task_id}: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of submitted task"""
        if task_id not in self.active_tasks:
            return {'status': 'not_found'}
        
        task = self.active_tasks[task_id]
        future = task['future']
        
        if future.done():
            if future.exception():
                task['status'] = 'error'
                task['error'] = str(future.exception())
            else:
                task['status'] = 'completed'
                task['result'] = future.result()
                task['end_time'] = datetime.now()
            
            # Clean up completed task
            del self.active_tasks[task_id]
        
        return {
            'status': task['status'],
            'start_time': task['start_time'].isoformat(),
            'duration': (datetime.now() - task['start_time']).total_seconds(),
            'error': task.get('error'),
            'result': task.get('result')
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel running task"""
        if task_id not in self.active_tasks:
            return False
        
        future = self.active_tasks[task_id]['future']
        success = future.cancel()
        
        if success:
            del self.active_tasks[task_id]
        
        return success
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks"""
        status = {}
        
        for task_id in list(self.active_tasks.keys()):
            status[task_id] = self.get_task_status(task_id)
        
        return status
    
    def cleanup(self):
        """Cleanup executor"""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
