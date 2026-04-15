# 影调分析 (Tone Analysis)

基于直方图分析的图像影调分类库。

## 功能特性

- **10种影调类型**：完整的影调分类体系
  - 高长调、高中调、高短调
  - 中长调、中中调、中短调
  - 低长调、低中调、低短调
  - 全长调（U型直方图特殊情况）

- **双维度分析**：
  - **波峰位置**：决定基调（高/中/低调）
  - **分布跨度**：决定跨度（长/中/短调）

- **全长调检测**：自动识别U型直方图

## 安装

```bash
pip install numpy pillow
```

## 快速开始

```python
from tone_analysis import ToneAnalyzer, analyze_image, get_tone_name

# 分析图片文件
result = analyze_image("your_image.jpg")

# 获取影调名称
tone_name = get_tone_name(result.tone_key, result.tone_range)
print(f"影调: {tone_name}")  # 例如："High-Long"、"Mid-Short" 等

# 访问统计数据
print(f"平均明度: {result.mean:.1f}")
print(f"标准差: {result.std:.1f}")
print(f"暗部: {result.shadows:.1f}%")
print(f"中间调: {result.midtones:.1f}%")
print(f"亮部: {result.highlights:.1f}%")
```

## 算法原理

算法基于两个维度对图像进行分类：

### 1. 波峰位置（影调基调）
决定像素主要集中在哪个亮度区域：

| 基调 | 波峰位置 | 说明 |
|-----|---------|------|
| 高调 | ≥ 160 | 明亮图像 |
| 中调 | 96 - 160 | 均衡图像 |
| 低调 | ≤ 96 | 暗调图像 |

### 2. 分布跨度（影调跨度）
决定对比度/亮度范围：

| 跨度 | 分布范围 | 说明 |
|-----|---------|------|
| 长调 | ≥ 180 | 完整范围，高对比度 |
| 中调 | 100 - 180 | 中等范围 |
| 短调 | < 100 | 窄范围，低对比度 |

### 全长调检测
针对U型直方图的特殊情况：
- 暗部和亮部都有明显像素
- 完整亮度范围（0-255）
- 中间区域像素少于边缘

## 示例输出

```
影调: 高长调
平均明度: 206.9
标准差: 25.0
暗部: 0.0%
中间调: 26.4%
亮部: 73.6%
```

## 影调类型说明

| 影调类型 | 特征 | 典型应用 |
|---------|------|---------|
| **高长调** | 明亮且范围完整 | 高调摄影、明亮场景 |
| **高中调** | 明亮且范围中等 | 柔光照明、人像 |
| **高短调** | 明亮且范围窄 | 低对比度、空灵 |
| **中长调** | 均衡且范围完整 | 标准摄影 |
| **中中调** | 均衡且范围中等 | 自然光 |
| **中短调** | 均衡且范围窄 | 低对比度、平光 |
| **低长调** | 暗调且范围完整 | 低调摄影、戏剧性 |
| **低中调** | 暗调且范围中等 | 情绪化场景 |
| **低短调** | 暗调且范围窄 | 极少细节、剪影 |
| **全长调** | U型直方图 | 高对比度、类HDR |

## API 参考

### `ToneAnalyzer`

影调分析主类。

```python
analyzer = ToneAnalyzer()
result = analyzer.analyze(image_array)
```

### `ToneAnalysisResult`

包含分析结果的数据类：

- `mean`: 平均亮度 (0-255)
- `median`: 中位数亮度
- `std`: 标准差（对比度指标）
- `min_val`, `max_val`: 亮度范围
- `shadows`, `midtones`, `highlights`: 分区百分比
- `tone_key`: ToneKey 枚举 (HIGH/MID/LOW/FULL)
- `tone_range`: ToneRange 枚举 (LONG/MEDIUM/SHORT)
- `histogram`: 完整直方图数组 (256 bins)
- `peak_position`: 加权平均波峰位置

### 辅助函数

- `analyze_image(path)`: 直接分析图片文件
- `get_tone_name(key, range)`: 获取可读影调名称

## 作者

**青山公仔** (qingshangongzai@163.com)

GitHub: [@qingshangongzai](https://github.com/qingshangongzai)

代码：trae

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

---

# Tone Analysis

A Python library for analyzing image tone (high-key, low-key, etc.) based on histogram analysis.

## Features

- **10 Tone Types**: Complete classification system
  - High-Long, High-Medium, High-Short
  - Mid-Long, Mid-Medium, Mid-Short
  - Low-Long, Low-Medium, Low-Short
  - Full-Long (special case for U-shaped histograms)

- **Two-Dimensional Analysis**:
  - **Peak Position**: Determines the key (high/mid/low)
  - **Distribution Spread**: Determines the range (long/medium/short)

- **Full-Tone Detection**: Automatically identifies U-shaped histograms

## Installation

```bash
pip install numpy pillow
```

## Quick Start

```python
from tone_analysis import ToneAnalyzer, analyze_image, get_tone_name

# Analyze an image file
result = analyze_image("your_image.jpg")

# Get tone name
tone_name = get_tone_name(result.tone_key, result.tone_range)
print(f"Tone: {tone_name}")  # e.g., "High-Long", "Mid-Short", etc.

# Access statistics
print(f"Mean brightness: {result.mean:.1f}")
print(f"Standard deviation: {result.std:.1f}")
print(f"Shadows: {result.shadows:.1f}%")
print(f"Midtones: {result.midtones:.1f}%")
print(f"Highlights: {result.highlights:.1f}%")
```

## Algorithm

The algorithm classifies images based on two dimensions:

### 1. Peak Position (Tone Key)
Determines where most pixels are concentrated:

| Key | Peak Position | Description |
|-----|---------------|-------------|
| High | ≥ 160 | Bright images |
| Mid | 96 - 160 | Balanced images |
| Low | ≤ 96 | Dark images |

### 2. Distribution Spread (Tone Range)
Determines the contrast/brightness range:

| Range | Spread | Description |
|-------|--------|-------------|
| Long | ≥ 180 | Full range, high contrast |
| Medium | 100 - 180 | Moderate range |
| Short | < 100 | Narrow range, low contrast |

### Full-Tone Detection
Special case for images with U-shaped histograms:
- Significant pixels in both shadows and highlights
- Full brightness range (0-255)
- Middle region has fewer pixels than edges

## Example Output

```
Tone: High-Long
Mean brightness: 206.9
Standard deviation: 25.0
Shadows: 0.0%
Midtones: 26.4%
Highlights: 73.6%
```

## Tone Types Explained

| Tone Type | Characteristics | Typical Use |
|-----------|-----------------|-------------|
| **High-Long** | Bright with full range | High-key photography, airy scenes |
| **High-Medium** | Bright with moderate range | Soft lighting, portraits |
| **High-Short** | Bright with narrow range | Minimal contrast, ethereal |
| **Mid-Long** | Balanced with full range | Standard photography |
| **Mid-Medium** | Balanced with moderate range | Natural lighting |
| **Mid-Short** | Balanced with narrow range | Low contrast, flat lighting |
| **Low-Long** | Dark with full range | Low-key photography, dramatic |
| **Low-Medium** | Dark with moderate range | Moody scenes |
| **Low-Short** | Dark with narrow range | Minimal detail, silhouettes |
| **Full-Long** | U-shaped histogram | High contrast, HDR-like |

## API Reference

### `ToneAnalyzer`

Main class for tone analysis.

```python
analyzer = ToneAnalyzer()
result = analyzer.analyze(image_array)
```

### `ToneAnalysisResult`

Dataclass containing analysis results:

- `mean`: Average brightness (0-255)
- `median`: Median brightness
- `std`: Standard deviation (contrast measure)
- `min_val`, `max_val`: Brightness range
- `shadows`, `midtones`, `highlights`: Zone percentages
- `tone_key`: ToneKey enum (HIGH/MID/LOW/FULL)
- `tone_range`: ToneRange enum (LONG/MEDIUM/SHORT)
- `histogram`: Full histogram array (256 bins)
- `peak_position`: Weighted average peak position

### Helper Functions

- `analyze_image(path)`: Analyze image file directly
- `get_tone_name(key, range)`: Get human-readable tone name

## Author

**青山公仔** (qingshangongzai@163.com)

GitHub: [@qingshangongzai](https://github.com/qingshangongzai)

Code: trae

## License

MIT License - see [LICENSE](LICENSE) file for details.
