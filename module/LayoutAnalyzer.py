import json
import statistics
from typing import List, Tuple
from dataclasses import dataclass
from functools import cached_property
from enum import Enum

from module.Detection import Detection


class AxisEnum(str, Enum):
    """
    Enumeration for axis directions used in layout analysis.
    """
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class LayoutAnalyzer:
    """
    Analyzes layout alignment of UI components based on detection bounding boxes.
    """
    def __init__(self, detections: List[Detection], tol_x: float = 10.0, tol_y: float = 10.0):
        """
        Initialize with detection predictions in JSON and tolerance values for alignment.
        """
        self.detections: List[Detection] = detections
        self.centers = [det.center for det in self.detections]
        self.tol_x = tol_x
        self.tol_y = tol_y

    def _cluster_1d(self, values: List[float], tol: float, min_cluster_size: int = 2) -> List[float]:
        """
        Clusters 1D values using simple distance-based grouping.
        """
        if not values:
            return []
        sorted_vals = sorted(values)
        clusters = [[sorted_vals[0]]]
        for v in sorted_vals[1:]:
            if abs(v - clusters[-1][-1]) <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [statistics.mean(cluster) for cluster in clusters if len(cluster) >= min_cluster_size]

    @cached_property
    def row_positions(self) -> List[float]:
        """
        Cached cluster positions for rows (Y-axis).
        """
        return self._cluster_1d([c[1] for c in self.centers], tol=self.tol_y)

    @cached_property
    def column_positions(self) -> List[float]:
        """
        Cached cluster positions for columns (X-axis).
        """
        return self._cluster_1d([c[0] for c in self.centers], tol=self.tol_x)

    def get_row_positions(self, tol: float | None = None) -> List[float]:
        """
        Return row (Y) cluster positions based on tolerance.
        """
        return self._cluster_1d([c[1] for c in self.centers], tol or self.tol_y)

    def get_column_positions(self, tol: float | None = None) -> List[float]:
        """
        Return column (X) cluster positions based on tolerance.
        """
        return self._cluster_1d([c[0] for c in self.centers], tol or self.tol_x)

    def generate_grid_with_skipped(
        self,
        tol_x: float | None = None,
        tol_y: float | None = None,
        debug: bool = False,
        allow_multi_assign: bool = False,
        allow_overlaps: bool = True
    ) -> Tuple[List[List[List[Detection]]], List[Detection]]:
        """
        Assign detections to a 2D grid based on proximity to clustered X/Y positions.
        Returns the grid and a list of detections that didn't match any cell.
        """
        tol_x = tol_x or self.tol_x
        tol_y = tol_y or self.tol_y

        row_pos = self.get_row_positions(tol_y)
        col_pos = self.get_column_positions(tol_x)
        grid: List[List[List[Detection]]] = [[[] for _ in col_pos] for _ in row_pos]

        skipped: List[Detection] = []

        for det in self.detections:
            cx, cy = det.center
            matched_rows = [i for i, y in enumerate(row_pos) if abs(cy - y) <= tol_y]
            matched_cols = [j for j, x in enumerate(col_pos) if abs(cx - x) <= tol_x]

            if not matched_rows or not matched_cols:
                skipped.append(det)
                if debug:
                    print(f"Skipping {det} (center=({cx:.1f},{cy:.1f})) - no grid match")
                continue

            targets = [(i, j) for i in matched_rows for j in matched_cols]
            if allow_multi_assign:
                for i, j in targets:
                    if allow_overlaps or not any(det.iou(existing) > 0.5 for existing in grid[i][j]):
                        grid[i][j].append(det)
                        if debug:
                            print(f"Assigning {det} to grid[{i}][{j}] (multi-assign)")
            else:
                i, j = targets[0]
                if allow_overlaps or not any(det.iou(existing) > 0.5 for existing in grid[i][j]):
                    grid[i][j].append(det)
                    if debug:
                        print(f"Assigning {det} to grid[{i}][{j}] (single-assign)")

        return grid, skipped

    def get_misaligned_and_skipped(
        self, axis: AxisEnum = AxisEnum.HORIZONTAL, tol: float | None = None
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Returns detections that are either misaligned or completely skipped (not aligned to any row/column).
        """
        tol = tol or (self.tol_y if axis == AxisEnum.HORIZONTAL else self.tol_x)
        axis_index = 1 if axis == AxisEnum.HORIZONTAL else 0
        positions = self._cluster_1d([c[axis_index] for c in self.centers], tol=tol)

        misaligned = []
        skipped = []

        for det in self.detections:
            matched = [pos for pos in positions if abs(det.center[axis_index] - pos) <= tol]
            if not matched:
                skipped.append(det)
            elif any(abs(det.center[axis_index] - pos) > tol / 2 for pos in matched):
                misaligned.append(det)

        return misaligned, skipped

    def calculate_misalignment_percentage(self, axis: AxisEnum = AxisEnum.HORIZONTAL, tol: float | None = None) -> float:
        """
        Calculate the percentage of detections that are either skipped or misaligned.
        """
        tol = tol or (self.tol_y if axis == AxisEnum.HORIZONTAL else self.tol_x)
        total = len(self.detections)
        if total == 0:
            return 0.0
        misaligned, skipped = self.get_misaligned_and_skipped(axis, tol)
        total_bad = len(misaligned) + len(skipped)
        return round((total_bad / total) * 100, 2)

    def _spacing_stats(self, positions: List[float]) -> float:
        """
        Compute average spacing between sorted cluster positions.
        """
        return statistics.mean([j - i for i, j in zip(positions, positions[1:])]) if len(positions) > 1 else 0.0

    def get_spacing_statistics(self) -> Tuple[float, float]:
        """
        Get average spacing for columns and rows respectively.
        """
        return (
            self._spacing_stats(self.get_column_positions()),
            self._spacing_stats(self.get_row_positions())
        )


def report_alignment_details(analyzer: LayoutAnalyzer, axis: AxisEnum = AxisEnum.HORIZONTAL, tol: float | None = None) -> None:
    """
    Print details about misaligned and skipped detections along the given axis.
    """
    misaligned, skipped = analyzer.get_misaligned_and_skipped(axis, tol)

    axis_label = "Y" if axis == AxisEnum.HORIZONTAL else "X"
    print(f"\n📊 {axis.name.title()} Alignment Report")
    print(f"  Misaligned: {len(misaligned)}")
    for det in misaligned:
        print(f"    - {det.class_label}#{det.detection_id[:6]} center={det.center}")

    print(f"  Skipped (No Match): {len(skipped)}")
    for det in skipped:
        print(f"    - {det.class_label}#{det.detection_id[:6]} center={det.center}")


if __name__ == "__main__":
    import os

    # Load example JSON file
    sample_path = "sample/layout_analyzer_sample.json"
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample file not found at: {sample_path}")

    with open(sample_path, "r") as f:
        detected_json = f.read()
        data = json.loads(detected_json)

    # Initialize LayoutAnalyzer with tolerances
    analyzer = LayoutAnalyzer([
            Detection(
                x=p['x'], y=p['y'], width=p['width'], height=p['height'],
                confidence=p['confidence'], class_label=p['class'],
                class_id=p['class_id'], detection_id=p['detection_id']
            ) for p in data.get("predictions", [])
        ], tol_x=20, tol_y=20)

    # Generate grid and collect skipped detections
    grid, skipped = analyzer.generate_grid_with_skipped(
        tol_x=20,
        tol_y=20,
        debug=True,
        allow_multi_assign=True,
        allow_overlaps=True
    )

    row_count = len(grid)
    col_count = len(grid[0]) if grid else 0

    print(f"\n🧩 Grid Summary: {row_count} rows x {col_count} columns")
    print(f"  Skipped in Grid: {len(skipped)}")

    # Calculate misalignment percentages
    h_score = analyzer.calculate_misalignment_percentage(AxisEnum.HORIZONTAL)
    v_score = analyzer.calculate_misalignment_percentage(AxisEnum.VERTICAL)

    print(f"\n📈 Alignment Scores:")
    print(f"\tHorizontal Misalignment: {h_score}%")
    print(f"\tVertical Misalignment: {v_score}%")

    # Spacing statistics (average gaps)
    col_spacing, row_spacing = analyzer.get_spacing_statistics()
    print(f"\n📐 Average Spacing:")
    print(f"\tColumns: {col_spacing:.2f}px")
    print(f"\tRows: {row_spacing:.2f}px")
