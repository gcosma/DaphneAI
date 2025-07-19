# ===============================================
# FILE: modules/bert_annotator.py
# ===============================================

import torch
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from collections import Counter
import re
from dataclasses import asdict

try:
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from core_utils import AnnotationResult

class BERTConceptAnnotator:
    """BERT-based concept annotation system"""
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        self.frameworks = self._load_frameworks()
        self.config = {
            "base_similarity_threshold": 0.65,
            "context_window_size": 150,
            "max_themes_per_framework": 10,
            "min_keyword_length": 3
        }
        self.logger = logging.getLogger(__name__)
    
    def _get_device(self):
        """Get appropriate device"""
        if not TRANSFORMERS_AVAILABLE:
            return "cpu"
            
        try:
            if torch.cuda.is_available():
                torch.cuda.init()
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}, using CPU")
            return torch.device("cpu")
    
    def _load_frameworks(self) -> Dict[str, List[Dict]]:
        """Load annotation frameworks with fallbacks"""
        frameworks = {
            "I-SIRch": self._get_isirch_framework(),
            "House of Commons": self._get_house_of_commons_framework(),
            "Extended Analysis": self._get_extended_framework()
        }
        return frameworks
    
    def _get_isirch_framework(self):
        """I-SIRch framework themes"""
        return [
            {
                "name": "External - Policy factor",
                "keywords": ["policy factor", "policy", "regulation", "guideline", "standard", "legislation"]
            },
            {
                "name": "System - Organizational factors",
                "keywords": ["organizational", "institutional", "governance", "management", "leadership", "culture"]
            },
            {
                "name": "Technology - Technology and tools", 
                "keywords": ["technology", "tools", "software", "system", "digital", "electronic", "equipment"]
            },
            {
                "name": "Person - Staff characteristics",
                "keywords": ["staff", "personnel", "healthcare worker", "professional", "clinician", "competency"]
            },
            {
                "name": "Task - Task characteristics",
                "keywords": ["task", "procedure", "protocol", "workflow", "process", "activity", "operation"]
            }
        ]
    
    def _get_house_of_commons_framework(self):
        """House of Commons framework themes"""
        return [
            {
                "name": "Communication",
                "keywords": ["communication", "dismissed", "listened", "concerns not taken seriously", "dialogue"]
            },
            {
                "name": "Fragmented care",
                "keywords": ["fragmented care", "fragmented", "continuity", "coordination", "integrated", "seamless"]
            },
            {
                "name": "Workforce pressures", 
                "keywords": ["workforce pressures", "staffing", "workload", "burnout", "capacity", "understaffed"]
            },
            {
                "name": "Biases and stereotyping",
                "keywords": ["biases", "stereotyping", "discrimination", "prejudice", "assumptions", "stigma"]
            }
        ]
    
    def _get_extended_framework(self):
        """Extended analysis framework themes"""
        return [
            {
                "name": "Procedural and Process Failures",
                "keywords": ["procedure failure", "process breakdown", "protocol breach", "standard violation"]
            },
            {
                "name": "Medication safety",
                "keywords": ["medication safety", "drug error", "prescription", "medication error", "adverse reaction"]
            },
            {
                "name": "Resource allocation",
                "keywords": ["resource allocation", "resource constraints", "funding", "budget limitations"]
            }
        ]
    
    def initialize_bert_model(self) -> bool:
        """Initialize BERT model"""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available - using keyword-only matching")
            return False
            
        if self.model is not None:
            return True
            
        try:
            self.logger.info(f"Loading BERT model: {self.model_name} on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False,
                trust_remote_code=False
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=False,
                trust_remote_code=False
            )
            
            try:
                self.model = self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                self.logger.warning(f"Failed to move model to {self.device}: {e}")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
            
            self.logger.info(f"BERT model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading BERT model: {e}")
            return False
    
    def get_bert_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get BERT embedding"""
        if not self.initialize_bert_model():
            return None
            
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            if str(self.device) != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                
                if str(self.device) != "cpu":
                    embedding = embedding.cpu()
                
                return embedding.numpy()
                
        except Exception as e:
            self.logger.error(f"Error getting BERT embedding: {e}")
            return None
    
    def annotate_text(self, text: str, selected_frameworks: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
        """Annotate text with concepts"""
        if not text or not text.strip():
            return {}, {}
        
        if selected_frameworks is None:
            selected_frameworks = list(self.frameworks.keys())
        
        # Get document embedding if BERT is available
        document_embedding = self.get_bert_embedding(text)
        
        framework_results = {}
        highlighting_info = {}
        
        for framework_name in selected_frameworks:
            if framework_name not in self.frameworks:
                continue
                
            framework_themes = self.frameworks[framework_name]
            if not framework_themes:
                continue
            
            theme_matches = []
            
            for theme in framework_themes:
                keyword_matches = self._find_keyword_matches(text, theme["keywords"])
                
                if keyword_matches:
                    # Calculate similarity
                    if document_embedding is not None:
                        theme_description = theme["name"] + ": " + ", ".join(theme["keywords"])
                        theme_embedding = self.get_bert_embedding(theme_description)
                        
                        if theme_embedding is not None:
                            similarity = cosine_similarity(
                                [document_embedding], [theme_embedding]
                            )[0][0]
                        else:
                            similarity = 0.5  # Fallback
                    else:
                        # Keyword-only scoring
                        similarity = len(keyword_matches) / len(theme["keywords"])
                    
                    combined_score = self._calculate_combined_score(
                        similarity, len(keyword_matches), len(text.split())
                    )
                    
                    if combined_score >= self.config["base_similarity_threshold"]:
                        match_info = {
                            "theme": theme["name"],
                            "semantic_similarity": float(similarity),
                            "combined_score": float(combined_score),
                            "matched_keywords": keyword_matches,
                            "keyword_count": len(keyword_matches),
                            "confidence": float(combined_score)
                        }
                        theme_matches.append(match_info)
                        
                        theme_key = f"{framework_name}_{theme['name']}"
                        highlighting_info[theme_key] = self._find_keyword_positions(
                            text, keyword_matches
                        )
            
            theme_matches.sort(key=lambda x: x["combined_score"], reverse=True)
            framework_results[framework_name] = theme_matches[:self.config["max_themes_per_framework"]]
        
        return framework_results, highlighting_info
    
    def _find_keyword_matches(self, text: str, keywords: List[str]) -> List[str]:
        """Find keyword matches in text"""
        text_lower = text.lower()
        matches = []
        
        for keyword in keywords:
            if len(keyword) >= self.config["min_keyword_length"]:
                if keyword.lower() in text_lower:
                    matches.append(keyword)
        
        return list(set(matches))
    
    def _find_keyword_positions(self, text: str, keywords: List[str]) -> List[Tuple]:
        """Find positions of keywords in text for highlighting"""
        positions = []
        text_lower = text.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(keyword), keyword))
                start = pos + 1
        
        return sorted(positions)
    
    def _calculate_combined_score(self, semantic_similarity: float, keyword_count: int, text_length: int) -> float:
        """Calculate combined score"""
        keyword_boost = min(keyword_count * 0.1, 0.3)
        length_factor = min(text_length / 1000, 1.0)
        
        combined = (semantic_similarity * 0.7) + keyword_boost + (length_factor * 0.1)
        return min(combined, 1.0)
    
    def load_custom_framework(self, framework_file) -> Tuple[bool, str]:
        """Load custom taxonomy from file"""
        try:
            if framework_file.name.endswith('.json'):
                data = json.load(framework_file)
                if 'themes' in data:
                    self.frameworks["Custom"] = data['themes']
                else:
                    return False, "JSON file must contain 'themes' array"
                    
            elif framework_file.name.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(framework_file)
                taxonomy_data = self._csv_to_taxonomy(df)
                self.frameworks["Custom"] = taxonomy_data
                
            elif framework_file.name.endswith(('.xlsx', '.xls')):
                import pandas as pd
                df = pd.read_excel(framework_file)
                taxonomy_data = self._csv_to_taxonomy(df)
                self.frameworks["Custom"] = taxonomy_data
            else:
                return False, "Unsupported file format"
            
            return True, f"Loaded {len(self.frameworks['Custom'])} themes"
            
        except Exception as e:
            return False, f"Error loading taxonomy: {str(e)}"
    
    def _csv_to_taxonomy(self, df) -> List[Dict]:
        """Convert CSV/Excel to taxonomy format"""
        taxonomy = []
        
        if 'theme' in df.columns and 'keywords' in df.columns:
            for _, row in df.iterrows():
                theme_name = str(row['theme']).strip()
                keywords_str = str(row['keywords']).strip()
                keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                
                if theme_name and keywords:
                    taxonomy.append({
                        "name": theme_name,
                        "keywords": keywords
                    })
        
        return taxonomy
