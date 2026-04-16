"""
tone_analysis库的测试。
"""

import numpy as np
import pytest

from tone_analysis import ToneAnalyzer, ToneKey, ToneRange, get_tone_name


# 测试常量
BRIGHT_VALUE = 200
DARK_VALUE = 50
MID_VALUE = 128
HIGH_KEY_THRESHOLD = 180
LOW_KEY_THRESHOLD = 80
LONG_RANGE_THRESHOLD = 180
SHORT_RANGE_STD_THRESHOLD = 50
HIGHLIGHT_DOMINANT_THRESHOLD = 90
SHADOW_MINIMAL_THRESHOLD = 5
FULL_TONE_ZONE_THRESHOLD = 5
IMAGE_SIZE = (100, 100)


class TestToneAnalyzer:
    """ToneAnalyzer类的测试用例。"""

    def setup_method(self):
        """设置测试夹具。"""
        self.analyzer = ToneAnalyzer()

    def test_high_key_detection(self):
        """测试高调图像检测。"""
        bright_img = np.ones((*IMAGE_SIZE, 3), dtype=np.uint8) * BRIGHT_VALUE
        result = self.analyzer.analyze(bright_img)

        assert result.tone_key == ToneKey.HIGH
        assert result.mean > HIGH_KEY_THRESHOLD

    def test_low_key_detection(self):
        """测试低调图像检测。"""
        dark_img = np.ones((*IMAGE_SIZE, 3), dtype=np.uint8) * DARK_VALUE
        result = self.analyzer.analyze(dark_img)

        assert result.tone_key == ToneKey.LOW
        assert result.mean < LOW_KEY_THRESHOLD

    def test_mid_key_detection(self):
        """测试中调图像检测。"""
        mid_img = np.ones((*IMAGE_SIZE, 3), dtype=np.uint8) * MID_VALUE
        result = self.analyzer.analyze(mid_img)

        assert result.tone_key == ToneKey.MID

    def test_long_range_detection(self):
        """测试长调（高对比度）检测。"""
        img = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
        img[:50, :] = 255

        result = self.analyzer.analyze(img)

        assert result.tone_range == ToneRange.LONG
        assert result.max_val - result.min_val >= LONG_RANGE_THRESHOLD

    def test_short_range_detection(self):
        """测试短调（低对比度）检测。"""
        narrow_img = np.ones((*IMAGE_SIZE, 3), dtype=np.uint8) * MID_VALUE
        result = self.analyzer.analyze(narrow_img)

        assert result.tone_range == ToneRange.SHORT
        assert result.std < SHORT_RANGE_STD_THRESHOLD

    def test_zone_calculation(self):
        """测试暗部/中调/亮部区域计算。"""
        highlight_img = np.ones((*IMAGE_SIZE, 3), dtype=np.uint8) * 220
        result = self.analyzer.analyze(highlight_img)

        assert result.highlights > HIGHLIGHT_DOMINANT_THRESHOLD
        assert result.shadows < SHADOW_MINIMAL_THRESHOLD

    def test_histogram_shape(self):
        """测试直方图形状正确。"""
        img = np.random.randint(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8)
        result = self.analyzer.analyze(img)

        assert result.histogram.shape == (256,)
        assert np.sum(result.histogram) == IMAGE_SIZE[0] * IMAGE_SIZE[1]

    def test_peak_position_calculation(self):
        """测试峰值位置计算。"""
        img = np.ones((*IMAGE_SIZE, 3), dtype=np.uint8) * MID_VALUE
        result = self.analyzer.analyze(img)

        assert abs(result.peak_position - MID_VALUE) <= 1

    def test_full_tone_detection(self):
        """测试全调（U型）检测。"""
        img = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
        img[:25, :25] = 255
        img[75:, 75:] = 0
        img[25:75, 25:75] = MID_VALUE

        result = self.analyzer.analyze(img)

        assert result.shadows > FULL_TONE_ZONE_THRESHOLD
        assert result.highlights > FULL_TONE_ZONE_THRESHOLD


class TestHelperFunctions:
    """辅助函数测试。"""

    def test_get_tone_name(self):
        """测试影调名称生成。"""
        assert get_tone_name(ToneKey.HIGH, ToneRange.LONG) == "High-Long"
        assert get_tone_name(ToneKey.MID, ToneRange.MEDIUM) == "Mid-Medium"
        assert get_tone_name(ToneKey.LOW, ToneRange.SHORT) == "Low-Short"
        assert get_tone_name(ToneKey.FULL, ToneRange.LONG) == "Full-Long"


class TestEdgeCases:
    """边界情况和错误处理测试。"""

    def setup_method(self):
        """设置测试夹具。"""
        self.analyzer = ToneAnalyzer()

    def test_empty_image(self):
        """测试均匀图像。"""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = self.analyzer.analyze(img)

        assert result.min_val == 0
        assert result.max_val == 0
        assert result.std == 0

    def test_full_range_image(self):
        """测试完整0-255范围图像。"""
        img = np.random.randint(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8)
        result = self.analyzer.analyze(img)

        assert result.min_val >= 0
        assert result.max_val <= 255
        assert 0 <= result.mean <= 255

    def test_single_color_image(self):
        """测试单色图像。"""
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        result = self.analyzer.analyze(img)

        assert result.mean == 100
        assert result.std == 0
        assert result.tone_range == ToneRange.SHORT

    def test_invalid_image_shape(self):
        """测试无效图像形状引发ValueError。"""
        # 灰度图像（应为RGB）
        gray_img = np.ones((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="期望RGB图像"):
            self.analyzer.analyze(gray_img)

        # 通道数错误
        rgba_img = np.ones((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="期望RGB图像"):
            self.analyzer.analyze(rgba_img)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
