"""
tone_analysis库的基本使用示例。
"""

import numpy as np
from PIL import Image

from tone_analysis import ToneAnalyzer, analyze_image, get_tone_name


def example_1_analyze_file():
    """示例1：分析图像文件。"""
    print("=" * 50)
    print("示例1：分析图像文件")
    print("=" * 50)

    # 替换为您的图像路径
    image_path = "sample_image.jpg"

    try:
        result = analyze_image(image_path)

        print(f"影调: {get_tone_name(result.tone_key, result.tone_range)}")
        print(f"平均亮度: {result.mean:.1f}")
        print(f"中位数: {result.median:.1f}")
        print(f"标准差: {result.std:.1f}")
        print(f"范围: {result.min_val} - {result.max_val}")
        print(f"\n区域分布:")
        print(f"  暗部:   {result.shadows:>6.2f}%")
        print(f"  中调:  {result.midtones:>6.2f}%")
        print(f"  亮部:{result.highlights:>6.2f}%")

    except FileNotFoundError:
        print(f"图像未找到: {image_path}")
        print("请提供有效的图像路径。")


def example_2_analyze_array():
    """示例2：直接分析numpy数组。"""
    print("\n" + "=" * 50)
    print("示例2：分析Numpy数组")
    print("=" * 50)

    # 创建示例图像（明亮，高调）
    bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 200

    analyzer = ToneAnalyzer()
    result = analyzer.analyze(bright_image)

    print(f"影调: {get_tone_name(result.tone_key, result.tone_range)}")
    print(f"平均亮度: {result.mean:.1f}")

    # 创建暗调图像（低调）
    dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 50
    result = analyzer.analyze(dark_image)

    print(f"\n暗调图像影调: {get_tone_name(result.tone_key, result.tone_range)}")
    print(f"平均亮度: {result.mean:.1f}")


def example_3_batch_analysis():
    """示例3：批量分析多张图像。"""
    print("\n" + "=" * 50)
    print("示例3：批量分析")
    print("=" * 50)

    # 生成不同影调的示例图像
    images = {
        "明亮": np.ones((100, 100, 3), dtype=np.uint8) * 220,
        "暗调": np.ones((100, 100, 3), dtype=np.uint8) * 40,
        "中调": np.ones((100, 100, 3), dtype=np.uint8) * 128,
    }

    analyzer = ToneAnalyzer()

    print(f"{'图像':<10} {'影调':<15} {'平均':<8} {'标准差':<8}")
    print("-" * 45)

    for name, img in images.items():
        result = analyzer.analyze(img)
        tone = get_tone_name(result.tone_key, result.tone_range)
        print(f"{name:<10} {tone:<15} {result.mean:<8.1f} {result.std:<8.1f}")


def example_4_visualize_histogram():
    """示例4：可视化直方图（需要matplotlib）。"""
    print("\n" + "=" * 50)
    print("示例4：直方图可视化")
    print("=" * 50)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装matplotlib。跳过可视化。")
        print("安装命令: pip install matplotlib")
        return

    # 创建渐变示例图像
    gradient = np.linspace(0, 255, 256).reshape(1, 256, 1).repeat(100, axis=0).repeat(3, axis=2)
    gradient = gradient.astype(np.uint8)

    analyzer = ToneAnalyzer()
    result = analyzer.analyze(gradient)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 图像
    ax1.imshow(gradient)
    ax1.set_title("示例图像")
    ax1.axis('off')

    # 直方图
    ax2.bar(range(256), result.histogram, width=1, color='gray', alpha=0.7)
    ax2.axvline(result.mean, color='red', linestyle='--', label=f'平均值: {result.mean:.1f}')
    ax2.axvline(result.peak_position, color='blue', linestyle='--', label=f'峰值: {result.peak_position:.1f}')
    ax2.set_xlabel('亮度')
    ax2.set_ylabel('像素数量')
    ax2.set_title(f"直方图 - {get_tone_name(result.tone_key, result.tone_range)}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('histogram_example.png')
    print("直方图已保存至: histogram_example.png")


if __name__ == "__main__":
    example_1_analyze_file()
    example_2_analyze_array()
    example_3_batch_analysis()
    example_4_visualize_histogram()
