"""
Tone Analysis - Image Tone Classification Library

A Python library for analyzing image tone based on histogram analysis.
Uses two-dimensional classification: peak position (key) and distribution spread (range).

MIT License
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


class ToneKey(str, Enum):
    """Tone key (brightness distribution)"""
    HIGH = "high"
    MID = "mid"
    LOW = "low"
    FULL = "full"


class ToneRange(str, Enum):
    """Tone range (contrast/spread)"""
    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


@dataclass
class ToneAnalysisResult:
    """Result of tone analysis"""
    mean: float
    median: float
    std: float
    min_val: int
    max_val: int
    shadows: float
    midtones: float
    highlights: float
    tone_key: ToneKey
    tone_range: ToneRange
    histogram: np.ndarray
    peak_position: float


class ToneAnalyzer:
    """
    Image tone analyzer based on histogram analysis.

    Classifies images into 10 tone types:
    - High-Long, High-Medium, High-Short
    - Mid-Long, Mid-Medium, Mid-Short
    - Low-Long, Low-Medium, Low-Short
    - Full-Long (special case)

    Based on two dimensions:
    1. Peak position: Determines key (high/mid/low)
    2. Distribution spread: Determines range (long/medium/short)
    """

    # Thresholds for key classification
    KEY_HIGH_MIN = 160
    KEY_LOW_MAX = 96

    # Thresholds for range classification
    RANGE_LONG = 180
    RANGE_MEDIUM = 100

    # Thresholds for full-tone detection
    MIN_ZONE_PERCENTAGE = 15
    MIN_RANGE_THRESHOLD = 30
    MAX_RANGE_THRESHOLD = 225
    U_SHAPE_RATIO = 0.7

    def analyze(self, image: np.ndarray) -> ToneAnalysisResult:
        """
        Analyze image tone.

        Args:
            image: RGB image array (H, W, 3) with values 0-255

        Returns:
            ToneAnalysisResult with tone classification and statistics
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

        gray = self._rgb_to_gray(image)

        mean = float(np.mean(gray))
        median = float(np.median(gray))
        std = float(np.std(gray))
        min_val = int(np.min(gray))
        max_val = int(np.max(gray))

        shadows = float(np.sum(gray < 64) / gray.size * 100)
        midtones = float(np.sum((gray >= 64) & (gray < 192)) / gray.size * 100)
        highlights = float(np.sum(gray >= 192) / gray.size * 100)

        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        peak_position = self._calc_peak_position(hist)

        tone_key, tone_range = self._classify_tone(
            peak_position, min_val, max_val, shadows, highlights, hist
        )

        return ToneAnalysisResult(
            mean=mean, median=median, std=std,
            min_val=min_val, max_val=max_val,
            shadows=shadows, midtones=midtones,
            highlights=highlights,
            tone_key=tone_key,
            tone_range=tone_range,
            histogram=hist,
            peak_position=peak_position
        )

    def _calc_peak_position(self, hist: np.ndarray) -> float:
        """Calculate peak position (histogram maximum position)."""
        return float(np.argmax(hist))

    def _classify_tone(self, peak: float, min_val: int, max_val: int,
                       shadows: float, highlights: float,
                       hist: np.ndarray) -> Tuple[ToneKey, ToneRange]:
        """
        Classify tone based on histogram characteristics.

        First checks for full-tone (U-shaped distribution),
        then classifies by peak position and spread.
        """
        spread = max_val - min_val

        # Check for full-tone (U-shaped distribution)
        if self._is_full_tone(hist, shadows, highlights, min_val, max_val):
            return ToneKey.FULL, ToneRange.LONG

        # Classify by peak position
        tone_key = self._get_tone_key(peak)

        # Classify by spread
        tone_range = self._get_tone_range(spread)

        return tone_key, tone_range

    def _is_full_tone(self, hist: np.ndarray, shadows: float,
                      highlights: float, min_val: int, max_val: int) -> bool:
        """
        Detect full-tone images (U-shaped histogram).

        Characteristics:
        - Significant pixels in both shadows and highlights
        - Full brightness range (near 0 to near 255)
        - Middle region has fewer pixels than edges
        """
        has_shadows = shadows > self.MIN_ZONE_PERCENTAGE
        has_highlights = highlights > self.MIN_ZONE_PERCENTAGE
        full_range = (min_val < self.MIN_RANGE_THRESHOLD) and (max_val > self.MAX_RANGE_THRESHOLD)

        # U-shape detection: edges have more pixels than middle
        mid_avg = np.mean(hist[64:192])
        edge_avg = np.mean(np.concatenate([hist[:32], hist[224:]]))

        return has_shadows and has_highlights and full_range and (mid_avg < edge_avg * self.U_SHAPE_RATIO)

    def _get_tone_key(self, peak: float) -> ToneKey:
        """Determine tone key from peak position."""
        if peak >= self.KEY_HIGH_MIN:
            return ToneKey.HIGH
        elif peak <= self.KEY_LOW_MAX:
            return ToneKey.LOW
        else:
            return ToneKey.MID

    def _get_tone_range(self, spread: int) -> ToneRange:
        """Determine tone range from brightness spread."""
        if spread >= self.RANGE_LONG:
            return ToneRange.LONG
        elif spread >= self.RANGE_MEDIUM:
            return ToneRange.MEDIUM
        return ToneRange.SHORT

    def _rgb_to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale using Rec. 709 standard."""
        return (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]).astype(np.uint8)


def get_tone_name(key: ToneKey, range_type: ToneRange) -> str:
    """Get human-readable tone name."""
    if key == ToneKey.FULL:
        return "Full-Long"
    return f"{key.value.title()}-{range_type.value.title()}"


def analyze_image(image_path: str) -> ToneAnalysisResult:
    """
    Convenience function to analyze an image file.

    Args:
        image_path: Path to image file

    Returns:
        ToneAnalysisResult
    """
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    analyzer = ToneAnalyzer()
    return analyzer.analyze(img_array)
