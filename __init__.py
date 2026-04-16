"""
影调分析 - 图像影调分类库

MIT许可证
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
