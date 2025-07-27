# modules/core_utils.py
# Enhanced core utilities with performance tracking

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

class PerformanceTracker:
    """Track application performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append({
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            
            del self.start_times[operation]
            return duration
        return 0.0
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation"""
        if operation in self.metrics and self.metrics[operation]:
            durations = [m['duration'] for m in self.metrics[operation]]
            return sum(durations) / len(durations)
        return 0.0
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get detailed stats for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = [m['duration'] for m in self.metrics[operation]]
        
        return {
            'count': len(durations),
            'average': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'total': sum(durations)
        }

# Global performance tracker
performance_tracker = PerformanceTracker()

def setup_logging(level=logging.INFO, log_file='app.log'):
    """Setup enhanced logging system"""
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== Application Started ===")
    logger.info(f"Logging level: {logging.getLevelName(level)}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def log_action(action: str, data: Dict[str, Any] = None, performance_data: Dict[str, float] = None):
    """Enhanced action logging with performance data"""
    logger = logging.getLogger(__name__)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'data': data or {},
        'performance': performance_data or {}
    }
    
    # Log to file in JSON format for better analysis
    logger.info(f"ACTION: {json.dumps(log_entry, default=str)}")
    
    return log_entry

def log_error(error: Exception, context: str = "", additional_data: Dict[str, Any] = None):
    """Enhanced error logging"""
    logger = logging.getLogger(__name__)
    
    error_entry = {
        'timestamp': datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'additional_data': additional_data or {}
    }
    
    logger.error(f"ERROR: {json.dumps(error_entry, default=str)}")
    
    return error_entry

def log_performance(operation: str, duration: float, additional_metrics: Dict[str, Any] = None):
    """Log performance metrics"""
    logger = logging.getLogger(__name__)
    
    perf_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'duration_seconds': duration,
        'additional_metrics': additional_metrics or {}
    }
    
    logger.info(f"PERFORMANCE: {json.dumps(perf_entry, default=str)}")
    
    return perf_entry

def timed_operation(operation_name: str):
    """Decorator for timing operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_tracker.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                duration = performance_tracker.end_timer(operation_name)
                log_performance(operation_name, duration)
                return result
            except Exception as e:
                performance_tracker.end_timer(operation_name)
                log_error(e, f"During {operation_name}")
                raise
        return wrapper
    return decorator

class SearchAnalytics:
    """Analytics for search operations"""
    
    def __init__(self):
        self.search_data = []
        self.user_patterns = {}
    
    def log_search(self, query: str, search_mode: str, results_count: int, 
                   search_time: float, filters: Dict[str, Any] = None):
        """Log a search operation"""
        
        search_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_length': len(query),
            'query_words': len(query.split()),
            'search_mode': search_mode,
            'results_count': results_count,
            'search_time': search_time,
            'filters': filters or {},
            'has_filters': bool(filters)
        }
        
        self.search_data.append(search_entry)
        self._update_user_patterns(search_entry)
        
        # Log for analytics
        log_action("search_performed", search_entry)
        
        return search_entry
    
    def _update_user_patterns(self, search_entry: Dict[str, Any]):
        """Update user behavior patterns"""
        mode = search_entry['search_mode']
        
        if mode not in self.user_patterns:
            self.user_patterns[mode] = {
                'usage_count': 0,
                'total_time': 0,
                'total_results': 0,
                'avg_query_length': 0
            }
        
        pattern = self.user_patterns[mode]
        pattern['usage_count'] += 1
        pattern['total_time'] += search_entry['search_time']
        pattern['total_results'] += search_entry['results_count']
        
        # Update average query length
        all_queries = [s for s in self.search_data if s['search_mode'] == mode]
        pattern['avg_query_length'] = sum(s['query_length'] for s in all_queries) / len(all_queries)
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get comprehensive search analytics"""
        if not self.search_data:
            return {}
        
        total_searches = len(self.search_data)
        
        # Time-based analytics
        total_search_time = sum(s['search_time'] for s in self.search_data)
        avg_search_time = total_search_time / total_searches
        
        # Results analytics
        total_results = sum(s['results_count'] for s in self.search_data)
        avg_results = total_results / total_searches
        
        # Query analytics
        query_lengths = [s['query_length'] for s in self.search_data]
        avg_query_length = sum(query_lengths) / len(query_lengths)
        
        # Mode popularity
        mode_counts = {}
        for search in self.search_data:
            mode = search['search_mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        return {
            'total_searches': total_searches,
            'total_search_time': total_search_time,
            'average_search_time': avg_search_time,
            'average_results_per_search': avg_results,
            'average_query_length': avg_query_length,
            'search_mode_popularity': mode_counts,
            'user_patterns': self.user_patterns,
            'performance_by_mode': self._get_performance_by_mode()
        }
    
    def _get_performance_by_mode(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by search mode"""
        mode_performance = {}
        
        for search in self.search_data:
            mode = search['search_mode']
            
            if mode not in mode_performance:
                mode_performance[mode] = {
                    'times': [],
                    'results': []
                }
            
            mode_performance[mode]['times'].append(search['search_time'])
            mode_performance[mode]['results'].append(search['results_count'])
        
        # Calculate averages
        performance_summary = {}
        for mode, data in mode_performance.items():
            performance_summary[mode] = {
                'avg_time': sum(data['times']) / len(data['times']),
                'avg_results': sum(data['results']) / len(data['results']),
                'min_time': min(data['times']),
                'max_time': max(data['times'])
            }
        
        return performance_summary

# Global analytics instance
search_analytics = SearchAnalytics()

def get_app_metrics() -> Dict[str, Any]:
    """Get overall application metrics"""
    
    return {
        'performance_metrics': {
            operation: performance_tracker.get_operation_stats(operation)
            for operation in performance_tracker.metrics.keys()
        },
        'search_analytics': search_analytics.get_search_analytics(),
        'timestamp': datetime.now().isoformat()
    }

def export_analytics_data(filename: str = None) -> str:
    """Export analytics data to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'search_analytics_{timestamp}.json'
    
    analytics_data = get_app_metrics()
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Analytics data exported to {filename}")
        
        return filename
        
    except Exception as e:
        log_error(e, "Failed to export analytics data")
        return ""

def cleanup_old_logs(days_to_keep: int = 7):
    """Clean up old log files"""
    try:
        log_files = Path('.').glob('*.log')
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in log_files:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                
        logger = logging.getLogger(__name__)
        logger.info(f"Cleaned up log files older than {days_to_keep} days")
        
    except Exception as e:
        log_error(e, "Failed to cleanup old logs")

def validate_search_query(query: str) -> Dict[str, Any]:
    """Validate and analyze search query"""
    if not query or not query.strip():
        return {
            'valid': False,
            'error': 'Query cannot be empty',
            'suggestions': ['Try entering some keywords', 'Use specific terms related to your search']
        }
    
    query = query.strip()
    words = query.split()
    
    validation_result = {
        'valid': True,
        'cleaned_query': query,
        'word_count': len(words),
        'character_count': len(query),
        'has_special_chars': bool(re.search(r'[^\w\s-"]', query)),
        'has_quotes': '"' in query,
        'suggested_improvements': []
    }
    
    # Provide suggestions for better searches
    if len(words) == 1 and len(query) < 3:
        validation_result['suggested_improvements'].append('Try using longer or more specific terms')
    
    if len(words) > 10:
        validation_result['suggested_improvements'].append('Consider using fewer, more specific keywords')
    
    if query.isupper():
        validation_result['suggested_improvements'].append('Consider using normal case instead of ALL CAPS')
        validation_result['cleaned_query'] = query.lower()
    
    return validation_result

def format_search_time(seconds: float) -> str:
    """Format search time for display"""
    if seconds < 0.001:
        return "<1ms"
    elif seconds < 1:
        return f"{seconds*1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"

def calculate_search_efficiency(results_count: int, search_time: float, query_complexity: int = 1) -> Dict[str, Any]:
    """Calculate search efficiency metrics"""
    
    # Results per second
    rps = results_count / max(search_time, 0.001)
    
    # Efficiency score (0-100)
    base_score = min(100, rps * 10)
    
    # Adjust for query complexity
    complexity_factor = 1.0 + (query_complexity - 1) * 0.1
    efficiency_score = base_score / complexity_factor
    
    # Classify efficiency
    if efficiency_score >= 80:
        efficiency_rating = "Excellent"
    elif efficiency_score >= 60:
        efficiency_rating = "Good"
    elif efficiency_score >= 40:
        efficiency_rating = "Fair"
    else:
        efficiency_rating = "Poor"
    
    return {
        'results_per_second': rps,
        'efficiency_score': efficiency_score,
        'efficiency_rating': efficiency_rating,
        'search_time_formatted': format_search_time(search_time)
    }
