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

    # 调性分类阈值（与分区一致：暗部0-85，中间调86-170，亮部171-255）
    KEY_HIGH_MIN = 171  # 亮部起始
    KEY_LOW_MAX = 85    # 暗部结束

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

        shadows = float(np.sum(gray <= 85) / gray.size * 100)
        midtones = float(np.sum((gray >= 86) & (gray <= 170)) / gray.size * 100)
        highlights = float(np.sum(gray >= 171) / gray.size * 100)

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

    def _calc_peak_sharpness(self, hist: np.ndarray, peak: int, window: int = 10) -> float:
        """计算波峰尖锐度

        尖锐度反映波峰的集中程度，越尖锐的波峰说明像素堆积越集中，
        基调判定越确定。计算方式为峰值与周围平均值的比值。

        Args:
            hist: 直方图数据
            peak: 波峰位置
            window: 计算窗口大小，默认10

        Returns:
            float: 尖锐度系数 (1.0-5.0)，值越大越尖锐
        """
        peak_val = hist[int(peak)]

        # 获取窗口范围内的数据（排除峰值本身）
        start = max(0, int(peak) - window)
        end = min(len(hist), int(peak) + window + 1)
        surrounding = np.concatenate([hist[start:int(peak)], hist[int(peak)+1:end]])

        surrounding_avg = np.mean(surrounding)
        if surrounding_avg == 0:
            return 5.0

        # 归一化到 1.0-5.0 范围
        ratio = peak_val / surrounding_avg
        return min(5.0, max(1.0, ratio))

    def _calc_distribution_continuity(self, hist: np.ndarray, start: int, end: int) -> float:
        """计算指定区间的分布连续性

        连续性反映像素分布的连贯程度，连续性越高说明分布越自然。
        通过计算非零bin的占比来评估。

        Args:
            hist: 直方图数据
            start: 起始位置
            end: 结束位置

        Returns:
            float: 连续性系数 (0.0-1.0)，值越大连续性越好
        """
        region = hist[start:end]
        if np.sum(region) == 0:
            return 0.0

        # 非零bin占比
        return np.count_nonzero(region) / len(region)

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
        # 全长调判断（带置信度）
        is_full, full_confidence = self._is_full_tone(hist, shadows, highlights, min_val, max_val)
        if is_full:
            return ToneKey.FULL, ToneRange.LONG, full_confidence, full_confidence

        # 根据峰值位置分类（含置信度）
        tone_key, key_confidence = self._get_tone_key(peak, hist)

        # 根据区域分布占比分类（含置信度）
        tone_range, range_confidence = self._get_tone_range_by_distribution(
            shadows, midtones, highlights, hist
        )

        return tone_key, tone_range, key_confidence, range_confidence

    def _is_full_tone(self, hist: np.ndarray, shadows: float,
                      highlights: float, min_val: int, max_val: int) -> Tuple[bool, float]:
        """判断是否为全长调并返回置信度

        特征：两端都有明显像素，中间相对较少

        Args:
            hist: 直方图数据
            shadows: 暗部占比 (%)
            highlights: 亮部占比 (%)
            min_val: 最小亮度值
            max_val: 最大亮度值

        Returns:
            Tuple[bool, float]: (是否全长调, 置信度0.5-1.0)
        """
        # 基础条件检查
        has_shadows = shadows > self.MIN_ZONE_PERCENTAGE
        has_highlights = highlights > self.MIN_ZONE_PERCENTAGE
        full_range = (min_val < self.MIN_RANGE_THRESHOLD) and (max_val > self.MAX_RANGE_THRESHOLD)

        if not (has_shadows and has_highlights and full_range):
            return False, 0.0

        # U型分布判断（基于新分区：暗部0-85，中间调86-170，亮部171-255）
        mid_avg = np.mean(hist[86:171])
        edge_avg = np.mean(np.concatenate([hist[:43], hist[213:]]))

        if mid_avg >= edge_avg * self.U_SHAPE_RATIO:
            return False, 0.0

        # 计算置信度
        # 1. 两端占比因子：两端占比越高，置信度越高
        edge_factor = min(edge_ratio := (shadows + highlights) / 100.0, 0.5) / 0.5

        # 2. U型明显程度因子：中间相对两端越少，置信度越高
        u_factor = max(0.0, 1.0 - mid_avg / (edge_avg * self.U_SHAPE_RATIO))

        # 3. 范围完整度因子
        range_factor = min((max_val - min_val) / 240.0, 1.0)

        # 综合置信度
        confidence = 0.5 + 0.5 * (edge_factor * 0.4 + u_factor * 0.4 + range_factor * 0.2)

        return True, confidence

    def _get_tone_key(self, peak: float, hist: np.ndarray) -> Tuple[ToneKey, float]:
        """根据波峰位置确定基调及置信度

        结合波峰位置和波峰尖锐度评估基调置信度，
        波峰越尖锐，基调判定越确定。

        Args:
            peak: 波峰位置
            hist: 直方图数据

        Returns:
            Tuple[ToneKey, float]: 基调类型、置信度(0.5-1.0)
        """
        # 计算波峰尖锐度因子
        sharpness = self._calc_peak_sharpness(hist, int(peak))
        # 尖锐度 1.0-5.0 映射到 0.7-1.0 的置信度因子
        sharpness_factor = 0.7 + 0.3 * min(1.0, (sharpness - 1.0) / 4.0)

        # 高调区域
        if peak >= self.KEY_HIGH_MIN:
            distance = peak - self.KEY_HIGH_MIN
            position_confidence = 0.5 + 0.5 * min(distance / self.KEY_BUFFER, 1.0)
            confidence = position_confidence * sharpness_factor
            return ToneKey.HIGH, confidence

        # 低调区域
        if peak <= self.KEY_LOW_MAX:
            distance = self.KEY_LOW_MAX - peak
            position_confidence = 0.5 + 0.5 * min(distance / self.KEY_BUFFER, 1.0)
            confidence = position_confidence * sharpness_factor
            return ToneKey.LOW, confidence

        # 中调区域
        dist_to_low = peak - self.KEY_LOW_MAX
        dist_to_high = self.KEY_HIGH_MIN - peak

        if dist_to_low < self.KEY_BUFFER and dist_to_low < dist_to_high:
            # 靠近低调边界
            position_confidence = 0.5 + 0.5 * (dist_to_low / self.KEY_BUFFER)
        elif dist_to_high < self.KEY_BUFFER:
            # 靠近高调边界
            position_confidence = 0.5 + 0.5 * (dist_to_high / self.KEY_BUFFER)
        else:
            # 中调核心区
            position_confidence = 1.0

        confidence = position_confidence * sharpness_factor
        return ToneKey.MID, confidence

    def _get_tone_range_by_distribution(
        self, shadows: float, midtones: float, highlights: float, hist: np.ndarray
    ) -> Tuple[ToneRange, float]:
        """根据区域分布占比确定跨度及置信度

        基于影调分类的核心原则：
        - 长调：暗部、中间调、亮部都有明显分布（从黑到白都有）
        - 中调：缺失一端（只有两段有明显分布）
        - 短调：集中在窄范围内（只有一段占绝对主导）

        结合分布连续性评估，连续性越高，跨度判定越确定。

        Args:
            shadows: 暗部占比 (%)
            midtones: 中间调占比 (%)
            highlights: 亮部占比 (%)
            hist: 直方图数据

        Returns:
            Tuple[ToneRange, float]: 跨度类型、置信度(0.5-1.0)
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

        # 计算各区域的分布连续性
        shadow_continuity = self._calc_distribution_continuity(hist, 0, 86)
        midtone_continuity = self._calc_distribution_continuity(hist, 86, 171)
        highlight_continuity = self._calc_distribution_continuity(hist, 171, 256)

        # 长调：三个区域都有明显分布
        if significant_zones >= 3:
            # 基础置信度基于最小区域的占比
            min_ratio = min(shadows, midtones, highlights)
            base_confidence = 0.5 + min(min_ratio / 10.0, 0.5)
            # 连续性因子：三个区域连续性的平均值
            continuity_factor = (shadow_continuity + midtone_continuity + highlight_continuity) / 3.0
            # 连续性好的区域置信度更高（0.8-1.0范围调整）
            confidence = base_confidence * (0.8 + 0.2 * continuity_factor)
            return ToneRange.LONG, confidence

        # 短调：只有一个区域有明显分布（集中度极高）
        if significant_zones == 1:
            max_ratio = max(shadows, midtones, highlights)
            base_confidence = 0.5 + min((max_ratio - 80.0) / 30.0, 0.5)
            # 短调主要依赖主导区域的连续性
            if shadows >= SIGNIFICANT_THRESHOLD:
                continuity_factor = shadow_continuity
            elif midtones >= SIGNIFICANT_THRESHOLD:
                continuity_factor = midtone_continuity
            else:
                continuity_factor = highlight_continuity
            confidence = base_confidence * (0.8 + 0.2 * continuity_factor)
            return ToneRange.SHORT, confidence

        # 中调：两个区域有明显分布
        ratios = [r for r in [shadows, midtones, highlights] if r >= SIGNIFICANT_THRESHOLD]
        continuities = []
        if shadows >= SIGNIFICANT_THRESHOLD:
            continuities.append(shadow_continuity)
        if midtones >= SIGNIFICANT_THRESHOLD:
            continuities.append(midtone_continuity)
        if highlights >= SIGNIFICANT_THRESHOLD:
            continuities.append(highlight_continuity)

        if len(ratios) == 2:
            # 基础置信度基于两个区域的均衡程度
            base_confidence = 0.5 + min(min(ratios) / max(ratios), 0.5)
            # 连续性因子
            continuity_factor = sum(continuities) / len(continuities) if continuities else 1.0
            confidence = base_confidence * (0.8 + 0.2 * continuity_factor)
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
