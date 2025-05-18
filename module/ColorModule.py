from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from module.Detection import Detection

# === Type Aliases ===
RGBColor = List[int]
ColorCount = Tuple[RGBColor, int]
ColorPercentage = Tuple[RGBColor, int]
HexColorPercentage = Tuple[str, int]


class ColorModule:
    def __init__(self, base64_str: str, resize_to: Tuple[int, int] = (600, 600),detections:List[Detection]=[]) -> None:
        """Initialize and process base64 image."""
        self.detections: List[Detection] = detections
        self.original_image: NDArray[np.uint8] = self._load_base64_image(base64_str)
        self.image: NDArray[np.uint8] = self._preprocess_image(self.original_image, resize_to)

    @staticmethod
    def _load_base64_image(base64_str: str) -> NDArray[np.uint8]:
        """Decode a base64-encoded image to an RGB NumPy array."""
        try:
            decoded = base64.b64decode(base64_str)
            with Image.open(BytesIO(decoded)) as img:
                return np.array(img.convert("RGB"))
        except Exception as e:
            raise ValueError("Invalid base64 image data") from e

    @staticmethod
    def _resize_image(image: NDArray[np.uint8], size: Tuple[int, int]) -> NDArray[Any]:
        """Resize an image to the given size using OpenCV."""
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _remove_background(image: NDArray[np.uint8]) -> NDArray[Any]:
        """Remove image background using binary mask."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(image, image, mask=mask)

    def _preprocess_image(self, image: NDArray[np.uint8], size: Tuple[int, int]) -> NDArray[np.uint8]:
        """Resize image, mask out regions with class_label 'Image', and remove background."""
        resized = self._resize_image(image, size)

        for detection in self.detections:
            if detection.class_label.lower() == "imageview":
                x1 = int(detection.x)
                y1 = int(detection.y)
                x2 = int(detection.x + detection.width)
                y2 = int(detection.y + detection.height)

                # Clamp to image bounds
                x1 = max(0, min(x1, resized.shape[1]))
                x2 = max(0, min(x2, resized.shape[1]))
                y1 = max(0, min(y1, resized.shape[0]))
                y2 = max(0, min(y2, resized.shape[0]))

                # Apply black mask to the detected region
                resized[y1:y2, x1:x2] = 0

        return self._remove_background(resized)

    def extract_dominant_colors(self, color_number: int = 3) -> List[ColorCount]:
        """Extract dominant non-black colors from the processed image."""
        pixels = self.image.reshape(-1, 3)
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

        if pixels.size == 0:
            raise ValueError("No non-black pixels found for color extraction.")

        kmeans = KMeans(n_clusters=color_number, n_init=10, random_state=42)
        kmeans.fit(pixels)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)

        return [
            (kmeans.cluster_centers_[label].astype(int).tolist(), int(count))
            for label, count in zip(labels, counts)
        ]

    @staticmethod
    def calculate_percentages(colors: List[ColorCount]) -> List[ColorPercentage]:
        """Calculate and sort color usage percentages from counts."""
        total = sum(count for _, count in colors)
        if total == 0:
            return [(color, 0) for color, _ in colors]

        return sorted(
            [(color, round((count / total) * 100)) for color, count in colors],
            key=lambda x: x[1],
            reverse=True,
        )

    @staticmethod
    def plot_colors(colors: List[ColorPercentage]) -> None:
        """Display a horizontal bar chart of the dominant colors."""
        if not colors:
            print("No colors to plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 2))
        start = 0
        for rgb, percent in colors:
            color = np.array(rgb) / 255
            ax.barh(0, width=percent, left=start, color=color, edgecolor="black")
            start += percent

        ax.set_xlim(0, 100)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def srgb_to_linear(component: float) -> float:
        """Convert sRGB to linear light component."""
        c = component / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    def relative_luminance(self, r: int, g: int, b: int) -> float:
        """Calculate relative luminance of an RGB color."""
        R, G, B = map(self.srgb_to_linear, (r, g, b))
        return 0.2126 * R + 0.7152 * G + 0.0722 * B

    def contrast_ratio(self, color1: RGBColor, color2: RGBColor) -> float:
        """Calculate contrast ratio between two RGB colors."""
        L1 = self.relative_luminance(*color1)
        L2 = self.relative_luminance(*color2)
        lighter, darker = max(L1, L2), min(L1, L2)
        return round((lighter + 0.05) / (darker + 0.05), 2)