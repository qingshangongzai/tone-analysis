"""
影调分析 - 图像影调分类库

基于直方图分析的图像影调分析Python库。
使用二维分类法：峰值位置（调性）和分布范围（反差）。

MIT许可证
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


class ToneKey(str, Enum):
    """影调调性（亮度分布）"""
    HIGH = "high"
    MID = "mid"
    LOW = "low"
    FULL = "full"


class ToneRange(str, Enum):
    """影调范围（对比度/分布）"""
    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


@dataclass
class ToneAnalysisResult:
    """影调分析结果"""
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
    tone_key_confidence: float
    tone_range_confidence: float


class ToneAnalyzer:
    """
    基于直方图分析的图像影调分析器。

    将图像分为10种影调类型：
    - 高调长调、高调中调、高调短调
    - 中调长调、中调中调、中调短调
    - 低调长调、低调中调、低调短调
    - 全调长调（特殊情况）

    基于两个维度：
    1. 峰值位置：决定调性（高调/中调/低调）
    2. 分布范围：决定反差（长调/中调/短调）
    """

    # 调性分类阈值
    KEY_HIGH_MIN = 160
    KEY_LOW_MAX = 96

    # 全调检测阈值
    MIN_ZONE_PERCENTAGE = 15
    MIN_RANGE_THRESHOLD = 30
    MAX_RANGE_THRESHOLD = 225
    U_SHAPE_RATIO = 0.7

    # 边界缓冲值（用于置信度计算）
    KEY_BUFFER = 15  # 调性分类边界缓冲

    def analyze(self, image: np.ndarray) -> ToneAnalysisResult:
        """
        分析图像影调。

        参数:
            image: RGB图像数组 (H, W, 3)，数值范围0-255

        返回:
            ToneAnalysisResult，包含影调分类和统计信息
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"期望RGB图像形状为 (H, W, 3)，实际得到 {image.shape}")

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

        tone_key, tone_range, key_confidence, range_confidence = self._classify_tone(
            peak_position, min_val, max_val, shadows, midtones, highlights, hist
        )

        return ToneAnalysisResult(
            mean=mean, median=median, std=std,
            min_val=min_val, max_val=max_val,
            shadows=shadows, midtones=midtones,
            highlights=highlights,
            tone_key=tone_key,
            tone_range=tone_range,
            histogram=hist,
            peak_position=peak_position,
            tone_key_confidence=key_confidence,
            tone_range_confidence=range_confidence
        )

    def _calc_peak_position(self, hist: np.ndarray) -> float:
        """计算峰值位置（直方图最大值位置）。"""
        return float(np.argmax(hist))

    def _classify_tone(self, peak: float, min_val: int, max_val: int,
                       shadows: float, midtones: float, highlights: float,
                       hist: np.ndarray) -> Tuple[ToneKey, ToneRange, float, float]:
        """
        基于直方图特征进行影调分类。

        首先检测全调（U型分布），
        然后根据峰值位置和分布范围进行分类，并计算置信度。

        返回:
            元组 (tone_key, tone_range, key_confidence, range_confidence)
        """
        # 检测全调（U型分布）
        if self._is_full_tone(hist, shadows, highlights, min_val, max_val):
            return ToneKey.FULL, ToneRange.LONG, 1.0, 1.0

        # 根据峰值位置分类（含置信度）
        tone_key, key_confidence = self._get_tone_key(peak)

        # 根据区域分布占比分类（含置信度）
        tone_range, range_confidence = self._get_tone_range_by_distribution(
            shadows, midtones, highlights
        )

        return tone_key, tone_range, key_confidence, range_confidence

    def _is_full_tone(self, hist: np.ndarray, shadows: float,
                      highlights: float, min_val: int, max_val: int) -> bool:
        """
        检测全调图像（U型直方图）。

        特征：
        - 暗部和亮部都有显著像素
        - 完整的亮度范围（接近0到接近255）
        - 中间区域像素少于边缘区域
        """
        has_shadows = shadows > self.MIN_ZONE_PERCENTAGE
        has_highlights = highlights > self.MIN_ZONE_PERCENTAGE
        full_range = (min_val < self.MIN_RANGE_THRESHOLD) and (max_val > self.MAX_RANGE_THRESHOLD)

        # U型检测：边缘像素多于中间
        mid_avg = np.mean(hist[64:192])
        edge_avg = np.mean(np.concatenate([hist[:32], hist[224:]]))

        return has_shadows and has_highlights and full_range and (mid_avg < edge_avg * self.U_SHAPE_RATIO)

    def _get_tone_key(self, peak: float) -> Tuple[ToneKey, float]:
        """根据峰值位置确定影调调性，并返回置信度。

        参数:
            peak: 直方图峰值位置 (0-255)

        返回:
            元组 (tone_key, confidence)，置信度范围为0.5-1.0
        """
        # 高调区域
        if peak >= self.KEY_HIGH_MIN:
            distance = peak - self.KEY_HIGH_MIN
            confidence = 0.5 + 0.5 * min(distance / self.KEY_BUFFER, 1.0)
            return ToneKey.HIGH, confidence

        # 低调区域
        if peak <= self.KEY_LOW_MAX:
            distance = self.KEY_LOW_MAX - peak
            confidence = 0.5 + 0.5 * min(distance / self.KEY_BUFFER, 1.0)
            return ToneKey.LOW, confidence

        # 中调区域
        dist_to_low = peak - self.KEY_LOW_MAX
        dist_to_high = self.KEY_HIGH_MIN - peak

        if dist_to_low < self.KEY_BUFFER and dist_to_low < dist_to_high:
            # 接近低调边界
            confidence = 0.5 + 0.5 * (dist_to_low / self.KEY_BUFFER)
        elif dist_to_high < self.KEY_BUFFER:
            # 接近高调边界
            confidence = 0.5 + 0.5 * (dist_to_high / self.KEY_BUFFER)
        else:
            # 核心中调区域
            confidence = 1.0

        return ToneKey.MID, confidence

    def _get_tone_range_by_distribution(
        self, shadows: float, midtones: float, highlights: float
    ) -> Tuple[ToneRange, float]:
        """根据区域分布占比确定跨度及置信度

        基于影调分类的核心原则：
        - 长调：暗部、中间调、亮部都有明显分布（从黑到白都有）
        - 中调：缺失一端（只有两段有明显分布）
        - 短调：集中在窄范围内（只有一段占绝对主导）

        参数:
            shadows: 暗部占比 (%)
            midtones: 中间调占比 (%)
            highlights: 亮部占比 (%)

        返回:
            元组 (tone_range, confidence)，置信度范围为0.5-1.0
        """
        # 定义"明显分布"的阈值
        SIGNIFICANT_THRESHOLD = 0.5  # 占比超过0.5%认为有明显分布（仅过滤极端噪声）

        # 计算有多少个区域有明显分布
        significant_zones = 0
        if shadows >= SIGNIFICANT_THRESHOLD:
            significant_zones += 1
        if midtones >= SIGNIFICANT_THRESHOLD:
            significant_zones += 1
        if highlights >= SIGNIFICANT_THRESHOLD:
            significant_zones += 1

        # 长调：三个区域都有明显分布
        if significant_zones >= 3:
            # 置信度基于最小区域的占比
            min_ratio = min(shadows, midtones, highlights)
            confidence = min(1.0, 0.5 + min_ratio / 10.0)
            return ToneRange.LONG, confidence

        # 短调：只有一个区域有明显分布（集中度极高）
        if significant_zones == 1:
            max_ratio = max(shadows, midtones, highlights)
            confidence = min(1.0, 0.5 + (max_ratio - 80.0) / 30.0)
            return ToneRange.SHORT, confidence

        # 中调：两个区域有明显分布
        ratios = [r for r in [shadows, midtones, highlights] if r >= SIGNIFICANT_THRESHOLD]
        if len(ratios) == 2:
            # 置信度基于两个区域的均衡程度
            confidence = min(1.0, 0.5 + min(ratios) / max(ratios))
        else:
            confidence = 0.7

        return ToneRange.MEDIUM, confidence

    def _rgb_to_gray(self, image: np.ndarray) -> np.ndarray:
        """RGB 转灰度 (Rec. 709 标准 + sRGB Gamma 校正)"""
        # 归一化到 0-1
        rgb = image.astype(np.float32) / 255.0

        # sRGB Gamma 校正到线性空间
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

        # Rec. 709 系数计算线性亮度
        luminance_linear = 0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2]

        # 从线性空间转回 sRGB
        luminance = np.where(
            luminance_linear <= 0.0031308,
            luminance_linear * 12.92,
            1.055 * (luminance_linear ** (1.0 / 2.4)) - 0.055
        )

        # 转换到 0-255 并四舍五入
        return np.clip(np.round(luminance * 255), 0, 255).astype(np.uint8)


def get_tone_name(key: ToneKey, range_type: ToneRange) -> str:
    """获取可读性影调名称。"""
    if key == ToneKey.FULL:
        return "Full-Long"
    return f"{key.value.title()}-{range_type.value.title()}"


def analyze_image(image_path: str) -> ToneAnalysisResult:
    """
    分析图像文件的便捷函数。

    参数:
        image_path: 图像文件路径

    返回:
        ToneAnalysisResult
    """
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    analyzer = ToneAnalyzer()
    return analyzer.analyze(img_array)
