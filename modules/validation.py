# ===============================================
# FILE: modules/validation.py
# ===============================================

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import magic
import hashlib

class DocumentValidator:
    """Validate documents and user inputs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_mime_types = {
            'application/pdf',
            'application/x-pdf'
        }
    
    def validate_pdf_file(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """Validate PDF file"""
        try:
            # Size check
            if len(file_content) > self.max_file_size:
                return False, f"File too large: {len(file_content)/1024/1024:.1f}MB > {self.max_file_size/1024/1024}MB"
            
            # Extension check
            if not filename.lower().endswith('.pdf'):
                return False, "File must have .pdf extension"
            
            # PDF signature check
            if not file_content.startswith(b'%PDF'):
                return False, "Invalid PDF file format"
            
            # Basic PDF structure check
            if b'%%EOF' not in file_content:
                return False, "PDF file appears corrupted"
            
            return True, "Valid PDF file"
            
        except Exception as e:
            self.logger.error(f"PDF validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_custom_framework(self, framework_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate custom framework structure"""
        try:
            # Check required fields
            if 'themes' not in framework_data:
                return False, "Framework must contain 'themes' field"
            
            themes = framework_data['themes']
            if not isinstance(themes, list):
                return False, "Themes must be a list"
            
            if len(themes) == 0:
                return False, "Framework must contain at least one theme"
            
            # Validate each theme
            for i, theme in enumerate(themes):
                if not isinstance(theme, dict):
                    return False, f"Theme {i+1} must be a dictionary"
                
                if 'name' not in theme:
                    return False, f"Theme {i+1} missing 'name' field"
                
                if 'keywords' not in theme:
                    return False, f"Theme {i+1} missing 'keywords' field"
                
                if not isinstance(theme['keywords'], list):
                    return False, f"Theme {i+1} keywords must be a list"
                
                if len(theme['keywords']) == 0:
                    return False, f"Theme {i+1} must have at least one keyword"
                
                # Validate keyword strings
                for j, keyword in enumerate(theme['keywords']):
                    if not isinstance(keyword, str) or not keyword.strip():
                        return False, f"Theme {i+1} keyword {j+1} must be a non-empty string"
            
            return True, f"Valid framework with {len(themes)} themes"
            
        except Exception as e:
            self.logger.error(f"Framework validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def sanitize_text_input(self, text: str, max_length: int = 10000) -> str:
        """Sanitize text input"""
        if not text:
            return ""
        
        # Convert to string and limit length
        text = str(text)[:max_length]
        
        # Remove potentially harmful content
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def validate_search_query(self, query: str) -> Tuple[bool, str]:
        """Validate search query"""
        if not query or not query.strip():
            return False, "Search query cannot be empty"
        
        query = query.strip()
        
        if len(query) < 3:
            return False, "Search query must be at least 3 characters"
        
        if len(query) > 1000:
            return False, "Search query too long (max 1000 characters)"
        
        # Check for potential injection attempts
        suspicious_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.*set'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains suspicious content"
        
        return True, "Valid search query"
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content"""
        return hashlib.sha256(file_content).hexdigest()
