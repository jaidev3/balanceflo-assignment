import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from data_models import EdgeResult


def sample_edge_strength(edges: np.ndarray, p1: Tuple[int,int], p2: Tuple[int,int], samples: int = 50) -> float:
    """Average edge magnitude along the line in the Canny map."""
    x1, y1 = p1
    x2, y2 = p2
    vals = []
    for t in np.linspace(0, 1, samples):
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
            vals.append(edges[y, x])
    return float(np.mean(vals) if vals else 0.0)


def detect_straight_edge(img_bgr: np.ndarray,
                         edges: np.ndarray,
                         min_line_len_ratio: float = 0.2,
                         angle_tolerance_deg: float = 60.0,
                         user_facing_zone: float = 0.6,
                         boundary_margin: float = 0.05
                         ) -> Optional[EdgeResult]:
    """
    Use Probabilistic Hough to find near-horizontal long segments that represent
    the user-facing desk edge. Focus on bottom portion of image.
    Filters out edges too close to image boundaries.
    Returns best EdgeResult or None.
    """
    H, W = edges.shape[:2]
    min_line_len = max(20, int(min_line_len_ratio * W))

    # Calculate boundary margins to ignore image edge artifacts
    margin_x = int(W * boundary_margin)
    margin_y = int(H * boundary_margin)

    # Focus on user-facing zone (bottom portion of image)
    user_zone_start = int(H * (1 - user_facing_zone))
    user_zone_edges = edges.copy()
    user_zone_edges[:user_zone_start, :] = 0  # Zero out upper portion
    
    # Zero out boundary regions to ignore image edge artifacts
    user_zone_edges[:, :margin_x] = 0  # Left boundary
    user_zone_edges[:, W-margin_x:] = 0  # Right boundary
    user_zone_edges[:margin_y, :] = 0  # Top boundary
    user_zone_edges[H-margin_y:, :] = 0  # Bottom boundary

    lines = cv2.HoughLinesP(user_zone_edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=min_line_len, maxLineGap=30)
    if lines is None:
        return None

    # Scoring: prioritize user-facing characteristics
    best = None
    best_score = -1.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Additional boundary filtering: reject lines too close to image edges
        if (x1 <= margin_x or x1 >= W - margin_x or x2 <= margin_x or x2 >= W - margin_x or
            y1 <= margin_y or y1 >= H - margin_y or y2 <= margin_y or y2 >= H - margin_y):
            continue
        
        dx, dy = (x2 - x1), (y2 - y1)
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Allow most non-vertical lines (flexible for various camera angles)
        # Filter out near-vertical lines which are rarely desk edges
        if abs(angle) > angle_tolerance_deg and abs(angle) < (180 - angle_tolerance_deg):
            continue

        length = np.hypot(dx, dy)
        y_mean = (y1 + y2) / 2.0
        
        # New robust scoring: heavily rewards length, but only if line is in lower half
        # Lines in top half get massive penalty (camera angle independent)
        position_bonus = 1.0 if y_mean > H / 2 else 0.1
        
        # The new score is simpler and more robust to angles
        # Prioritizes longest, most prominent edges in lower half
        score = length * position_bonus

        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    if best is None:
        return None

    x1, y1, x2, y2 = best
    return EdgeResult(
        edge_type="line",
        points=[(int(x1), int(y1)), (int(x2), int(y2))],
        score=float(best_score),
        method="hough_straight",
        metadata={
            "angle_tolerance_deg": angle_tolerance_deg,
            "min_line_len_ratio": min_line_len_ratio,
            "user_facing_zone": user_facing_zone,
            "boundary_margin": boundary_margin,
            "flexible_angle_support": True,
            "robust_scoring": True,
            "boundary_filtering": True
        }
    )


def detect_curved_edge(img_bgr: np.ndarray,
                       edges: np.ndarray,
                       bottom_mask_ratio: float = 0.65,
                       approx_eps_ratio: float = 0.01,
                       resample_points: int = 40,
                       boundary_margin: float = 0.05) -> Optional[EdgeResult]:
    """
    Focus on bottom region to find curved user-facing desk edges.
    Extract the frontmost envelope (max y by x) representing the user-side edge.
    Filters out edges too close to image boundaries.
    Returns best EdgeResult or None.
    """
    H, W = edges.shape[:2]

    # Calculate boundary margins to ignore image edge artifacts
    margin_x = int(W * boundary_margin)
    margin_y = int(H * boundary_margin)

    # Enhanced mask for user-facing zone (bottom portion)
    mask = np.zeros_like(edges)
    y0 = int(H * (1 - bottom_mask_ratio))
    mask[y0:H, :] = 255
    
    # Zero out boundary regions to ignore image edge artifacts
    mask[:, :margin_x] = 0  # Left boundary
    mask[:, W-margin_x:] = 0  # Right boundary
    mask[:margin_y, :] = 0  # Top boundary
    mask[H-margin_y:, :] = 0  # Bottom boundary
    
    # Remove central bias assumption for variable camera angles
    # Use full width to accommodate side/angled views
    masked_edges = cv2.bitwise_and(edges, mask)

    # Enhanced morphology to better connect desk edge segments
    kernel_close = np.ones((7,7), np.uint8)  # Larger kernel for better connection
    kernel_dilate = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Enhanced scoring for user-facing desk edges
    best = None
    best_score = -1.0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out very small contours
        if w < W * 0.15 or h < 5:  # Must be reasonably wide and tall
            continue
        
        # Additional boundary filtering: reject contours too close to image edges
        if (x <= margin_x or x + w >= W - margin_x or 
            y <= margin_y or y + h >= H - margin_y):
            continue
            
        width_norm = w / float(W)
        height_norm = h / float(H)

        # Vertical position: prefer lower edges (user-facing)
        y_bottom = (y + h) / float(H)
        
        # Remove center bias - edges can be anywhere in frame for angled views
        # Focus on width coverage instead of central positioning
        width_coverage_score = min(1.0, width_norm * 2.0)
        
        # Aspect ratio: user-facing edges tend to be wider than tall
        aspect_ratio = w / max(h, 1)
        aspect_score = min(1.0, aspect_ratio / 5.0)  # Bonus for wide, shallow contours
        
        # Area coverage relative to user zone
        area = cv2.contourArea(cnt)
        area_norm = area / (W * H * bottom_mask_ratio)
        
        # Enhanced composite score for variable camera angles
        # Prioritize lower position and width coverage over central bias
        score = (0.35 * width_norm +           # Width coverage (increased weight)
                0.30 * y_bottom +              # Lower position (increased weight)
                0.20 * width_coverage_score +  # Width coverage bonus
                0.10 * aspect_score +          # Wide aspect ratio
                0.05 * min(1.0, area_norm * 10))  # Reasonable area

        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return None

    # Approximate contour to reduce noise
    perim = cv2.arcLength(best, closed=True)
    eps = max(2.0, approx_eps_ratio * perim)
    approx = cv2.approxPolyDP(best, eps, closed=False)

    # Extract the "user-facing envelope": for each x, take max y (closest to user)
    pts = approx.reshape(-1, 2)
    # Sort by x coordinate
    pts = pts[np.argsort(pts[:, 0])]

    # Group by x (pixel columns) and keep max y (frontmost point)
    envelope = {}
    for (x, y) in pts:
        if x not in envelope or y > envelope[x]:
            envelope[x] = y

    # Sort x coordinates and form ordered polyline
    xs_sorted = sorted(envelope.keys())
    poly = [(int(x), int(envelope[x])) for x in xs_sorted]
    
    # Ensure minimum width coverage for user-facing edge
    if len(poly) < 2 or (poly[-1][0] - poly[0][0]) < W * 0.2:
        return None

    # Downsample/resample to fixed number of points
    if len(poly) > resample_points:
        idxs = np.round(np.linspace(0, len(poly) - 1, resample_points)).astype(int)
        poly = [poly[i] for i in idxs]

    return EdgeResult(
        edge_type="polyline",
        points=poly,
        score=float(best_score),
        method="contour_curved",
        metadata={
            "bottom_mask_ratio": bottom_mask_ratio,
            "approx_eps_ratio": approx_eps_ratio,
            "resample_points": resample_points,
            "boundary_margin": boundary_margin,
            "center_bias_applied": False,
            "variable_angle_support": True,
            "boundary_filtering": True
        }
    )


def choose_best_edge(straight_res: Optional[EdgeResult],
                     curved_res: Optional[EdgeResult]) -> Optional[EdgeResult]:
    """
    Enhanced chooser for user-facing desk edges. Prioritizes edges that are
    more likely to represent the user-facing side of the desk.
    """
    if straight_res and not curved_res:
        return straight_res
    if curved_res and not straight_res:
        return curved_res
    if not straight_res and not curved_res:
        return None

    # Enhanced selection logic for user-facing edges
    straight_score = straight_res.score
    curved_score = curved_res.score
    
    # Apply user-facing bonuses
    # Straight edges are often more reliable for user-facing desk detection
    straight_bonus = 1.1 if straight_score > 0.3 else 1.0
    
    # Check if straight edge spans good width (user-facing characteristic)
    if straight_res.points and len(straight_res.points) >= 2:
        x1, x2 = straight_res.points[0][0], straight_res.points[1][0]
        width_span = abs(x2 - x1)
        # Bonus for edges that span significant width
        if 'user_facing_zone' in straight_res.metadata:
            straight_bonus *= 1.05
    
    # Curved edges get bonus if they show good central positioning
    curved_bonus = 1.0
    if curved_res.metadata.get('center_bias_applied', False):
        curved_bonus = 1.03
    
    adjusted_straight = straight_score * straight_bonus
    adjusted_curved = curved_score * curved_bonus
    
    # If straight is competitive (within 15%), prefer it for user-facing detection
    if adjusted_straight >= 0.85 * adjusted_curved:
        return straight_res
    return curved_res


def draw_edge_annotation(img_bgr: np.ndarray, result: EdgeResult) -> np.ndarray:
    """Draw edge annotation on the image."""
    out = img_bgr.copy()
    if result.edge_type == "line":
        (x1, y1), (x2, y2) = result.points
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 4)  # blue
    else:
        pts = np.array(result.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=(255, 0, 0), thickness=4)
    return out


def build_json(result: EdgeResult, img_shape: Tuple[int, int, int]) -> Dict:
    """Build JSON output from edge detection result."""
    H, W = img_shape[:2]
    return {
        "edge_type": result.edge_type,
        "points": result.points,
        "score": round(result.score, 5),
        "method": result.method,
        "image_size": {"width": W, "height": H},
        "metadata": result.metadata
    }