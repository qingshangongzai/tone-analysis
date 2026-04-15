"""
Basic usage example for tone_analysis library.
"""

import numpy as np
from PIL import Image

from tone_analysis import ToneAnalyzer, analyze_image, get_tone_name


def example_1_analyze_file():
    """Example 1: Analyze an image file."""
    print("=" * 50)
    print("Example 1: Analyze Image File")
    print("=" * 50)

    # Replace with your image path
    image_path = "sample_image.jpg"

    try:
        result = analyze_image(image_path)

        print(f"Tone: {get_tone_name(result.tone_key, result.tone_range)}")
        print(f"Mean brightness: {result.mean:.1f}")
        print(f"Median: {result.median:.1f}")
        print(f"Standard deviation: {result.std:.1f}")
        print(f"Range: {result.min_val} - {result.max_val}")
        print(f"\nZone distribution:")
        print(f"  Shadows:   {result.shadows:>6.2f}%")
        print(f"  Midtones:  {result.midtones:>6.2f}%")
        print(f"  Highlights:{result.highlights:>6.2f}%")

    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path.")


def example_2_analyze_array():
    """Example 2: Analyze a numpy array directly."""
    print("\n" + "=" * 50)
    print("Example 2: Analyze Numpy Array")
    print("=" * 50)

    # Create a sample image (bright, high-key)
    bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 200

    analyzer = ToneAnalyzer()
    result = analyzer.analyze(bright_image)

    print(f"Tone: {get_tone_name(result.tone_key, result.tone_range)}")
    print(f"Mean brightness: {result.mean:.1f}")

    # Create a dark image (low-key)
    dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 50
    result = analyzer.analyze(dark_image)

    print(f"\nDark image tone: {get_tone_name(result.tone_key, result.tone_range)}")
    print(f"Mean brightness: {result.mean:.1f}")


def example_3_batch_analysis():
    """Example 3: Batch analyze multiple images."""
    print("\n" + "=" * 50)
    print("Example 3: Batch Analysis")
    print("=" * 50)

    # Generate sample images with different tones
    images = {
        "Bright": np.ones((100, 100, 3), dtype=np.uint8) * 220,
        "Dark": np.ones((100, 100, 3), dtype=np.uint8) * 40,
        "Mid": np.ones((100, 100, 3), dtype=np.uint8) * 128,
    }

    analyzer = ToneAnalyzer()

    print(f"{'Image':<10} {'Tone':<15} {'Mean':<8} {'Std':<8}")
    print("-" * 45)

    for name, img in images.items():
        result = analyzer.analyze(img)
        tone = get_tone_name(result.tone_key, result.tone_range)
        print(f"{name:<10} {tone:<15} {result.mean:<8.1f} {result.std:<8.1f}")


def example_4_visualize_histogram():
    """Example 4: Visualize histogram (requires matplotlib)."""
    print("\n" + "=" * 50)
    print("Example 4: Histogram Visualization")
    print("=" * 50)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")
        return

    # Create a sample image with gradient
    gradient = np.linspace(0, 255, 256).reshape(1, 256, 1).repeat(100, axis=0).repeat(3, axis=2)
    gradient = gradient.astype(np.uint8)

    analyzer = ToneAnalyzer()
    result = analyzer.analyze(gradient)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Image
    ax1.imshow(gradient)
    ax1.set_title("Sample Image")
    ax1.axis('off')

    # Histogram
    ax2.bar(range(256), result.histogram, width=1, color='gray', alpha=0.7)
    ax2.axvline(result.mean, color='red', linestyle='--', label=f'Mean: {result.mean:.1f}')
    ax2.axvline(result.peak_position, color='blue', linestyle='--', label=f'Peak: {result.peak_position:.1f}')
    ax2.set_xlabel('Brightness')
    ax2.set_ylabel('Pixel Count')
    ax2.set_title(f"Histogram - {get_tone_name(result.tone_key, result.tone_range)}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('histogram_example.png')
    print("Histogram saved to: histogram_example.png")


if __name__ == "__main__":
    example_1_analyze_file()
    example_2_analyze_array()
    example_3_batch_analysis()
    example_4_visualize_histogram()
