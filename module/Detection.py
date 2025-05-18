from typing import Tuple
from dataclasses import dataclass

@dataclass
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_label: str
    class_id: int
    detection_id: str

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def __repr__(self) -> str:
        return f"Detection({self.class_label}, id={self.detection_id[:6]}, conf={self.confidence:.2f})"

    def iou(self, other: 'Detection') -> float:
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2, y2 = min(self.x + self.width, other.x + other.width), min(self.y + self.height, other.y + other.height)
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = self.width * self.height
        area2 = other.width * other.height
        return inter_area / (area1 + area2 - inter_area + 1e-6)