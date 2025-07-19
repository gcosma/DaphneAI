# ===============================================
# FILE: modules/error_handler.py
# ===============================================

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from functools import wraps
import streamlit as st

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
    
    def handle_error(self, error: Exception, 
                    context: str = "",
                    user_message: Optional[str] = None,
                    show_details: bool = False) -> Dict[str, Any]:
        """Handle and log errors"""
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        # Log error
        self.logger.error(f"Error in {context}: {error}", exc_info=True)
        
        # Count errors
        error_key = f"{context}_{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Display to user
        if user_message:
            st.error(user_message)
        else:
            st.error(f"❌ An error occurred: {str(error)}")
        
        if show_details:
            with st.expander("🔍 Error Details"):
                st.code(error_info['traceback'])
        
        return error_info
    
    def with_error_handling(self, context: str, 
                          user_message: Optional[str] = None,
                          show_details: bool = False,
                          reraise: bool = False):
        """Decorator for automatic error handling"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(
                        e, 
                        context=context,
                        user_message=user_message,
                        show_details=show_details
                    )
                    if reraise:
                        raise
                    return None
            return wrapper
        return decorator
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'unique_error_types': len(self.error_counts),
            'error_breakdown': dict(sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        }
