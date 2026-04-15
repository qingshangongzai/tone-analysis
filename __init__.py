"""
Tone Analysis - Image Tone Classification Library

MIT License
"""

from .tone_analysis import (
    ToneAnalyzer,
    ToneAnalysisResult,
    ToneKey,
    ToneRange,
    analyze_image,
    get_tone_name,
)

__version__ = "1.0.0"
__all__ = [
    "ToneAnalyzer",
    "ToneAnalysisResult",
    "ToneKey",
    "ToneRange",
    "analyze_image",
    "get_tone_name",
]
