from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class EdgeResult:
    """Data structure to represent detected edge information."""
    edge_type: str  # "line" or "polyline"
    points: List[Tuple[int, int]]  # [[x,y], ...]
    score: float
    method: str     # "hough_straight" or "contour_curved"
    metadata: Dict