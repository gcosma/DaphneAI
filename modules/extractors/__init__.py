"""
Document extractors package.
"""

from .recommendation_extractor import (
    extract_recommendations,
    AdvancedRecommendationExtractor
)

__all__ = [
    'extract_recommendations',
    'AdvancedRecommendationExtractor'
]
