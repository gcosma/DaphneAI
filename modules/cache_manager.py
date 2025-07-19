# ===============================================
# FILE: modules/cache_manager.py
# ===============================================

import pickle
import json
import logging
import threading
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

class CacheManager:
    """Manage caching for expensive operations"""
    
    def __init__(self, cache_dir: str = "./data/cache", max_age_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.memory_cache = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key from operation and parameters"""
        key_data = f"{operation}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def get(self, operation: str, **kwargs) -> Optional[Any]:
        """Get cached result"""
        cache_key = self._get_cache_key(operation, **kwargs)
        
        # Check memory cache first
        with self._lock:
            if cache_key in self.memory_cache:
                item = self.memory_cache[cache_key]
                if datetime.now() - item['timestamp'] < self.max_age:
                    return item['data']
                else:
                    del self.memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                # Check file age
                file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if datetime.now() - file_time < self.max_age:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Store in memory cache
                    with self._lock:
                        self.memory_cache[cache_key] = {
                            'data': data,
                            'timestamp': datetime.now()
                        }
                    
                    return data
                else:
                    cache_path.unlink()  # Remove expired file
            except Exception as e:
                self.logger.error(f"Error reading cache: {e}")
                if cache_path.exists():
                    cache_path.unlink()
        
        return None
    
    def set(self, operation: str, data: Any, **kwargs):
        """Store result in cache"""
        cache_key = self._get_cache_key(operation, **kwargs)
        
        # Store in memory cache
        with self._lock:
            self.memory_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
        
        # Store in disk cache
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Error writing cache: {e}")
    
    def invalidate(self, operation: str, **kwargs):
        """Invalidate specific cache entry"""
        cache_key = self._get_cache_key(operation, **kwargs)
        
        # Remove from memory cache
        with self._lock:
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
        
        # Remove from disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = datetime.now()
        
        # Clear expired memory cache
        with self._lock:
            expired_keys = [
                key for key, item in self.memory_cache.items()
                if current_time - item['timestamp'] >= self.max_age
            ]
            for key in expired_keys:
                del self.memory_cache[key]
        
        # Clear expired disk cache
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if current_time - file_time >= self.max_age:
                    cache_file.unlink()
            except Exception as e:
                self.logger.error(f"Error removing expired cache file {cache_file}: {e}")
    
    def clear_all(self):
        """Clear all cache"""
        # Clear memory cache
        with self._lock:
            self.memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.error(f"Error removing cache file {cache_file}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            memory_count = len(self.memory_cache)
        
        disk_files = list(self.cache_dir.glob("*.cache"))
        disk_count = len(disk_files)
        
        total_size = sum(f.stat().st_size for f in disk_files if f.exists()) / (1024 * 1024)  # MB
        
        return {
            'memory_entries': memory_count,
            'disk_entries': disk_count,
            'total_size_mb': round(total_size, 2),
            'cache_dir': str(self.cache_dir)
        }
