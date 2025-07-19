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
