import io
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image
import streamlit as st


# =========================
# Utility data structures
# =========================
@dataclass
class EdgeResult:
    edge_type: str  # "line" or "polyline"
    points: List[Tuple[int, int]]  # [[x,y], ...]
    score: float
    method: str     # "hough_straight" or "contour_curved"
    metadata: Dict


# =========================
# Image helpers
# =========================
def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_rgb(cv_img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR to RGB (for Streamlit display)."""
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

def resize_max_width(img_bgr: np.ndarray, max_w: int = 1280) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w <= max_w:
        return img_bgr
    scale = max_w / float(w)
    new_size = (max_w, int(h * scale))
    return cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)


# =========================
# Preprocess
# =========================
def preprocess_for_edges(img_bgr: np.ndarray, use_clahe: bool = True):
    """
    Returns: gray, blurred, edges, debug dict
    """
    debug = {}
    # Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast normalize (CLAHE tends to help with uneven lighting)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    # Gentle blur to suppress noise but preserve edges decently
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # Canny - thresholds can be tuned via UI
    edges = cv2.Canny(blurred, 50, 150)

    debug["gray"] = gray
    debug["blurred"] = blurred
    debug["edges"] = edges
    return gray, blurred, edges, debug


# =========================
# Straight edge (Hough)
# =========================
def detect_straight_edge(img_bgr: np.ndarray,
                         edges: np.ndarray,
                         min_line_len_ratio: float = 0.2,
                         angle_tolerance_deg: float = 10.0,
                         user_facing_zone: float = 0.6
                         ) -> Optional[EdgeResult]:
    """
    Use Probabilistic Hough to find near-horizontal long segments that represent
    the user-facing desk edge. Focus on bottom portion of image.
    Returns best EdgeResult or None.
    """
    H, W = edges.shape[:2]
    min_line_len = max(20, int(min_line_len_ratio * W))

    # Focus on user-facing zone (bottom portion of image)
    user_zone_start = int(H * (1 - user_facing_zone))
    user_zone_edges = edges.copy()
    user_zone_edges[:user_zone_start, :] = 0  # Zero out upper portion

    lines = cv2.HoughLinesP(user_zone_edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=min_line_len, maxLineGap=30)
    if lines is None:
        return None

    # Scoring: prioritize user-facing characteristics
    best = None
    best_score = -1.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = (x2 - x1), (y2 - y1)
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Must be near-horizontal (user-facing desk edges are typically horizontal)
        if abs(angle) > angle_tolerance_deg:
            continue

        length = np.hypot(dx, dy)
        length_norm = length / float(W)

        # User-facing position score: prefer edges in lower 2/3 of image
        y_mean = (y1 + y2) / 2.0
        y_pos_norm = y_mean / float(H)
        
        # Boost score for edges in the "user zone" (bottom portion)
        user_zone_bonus = 1.0 if y_mean > user_zone_start else 0.5

        # Width coverage: user-facing edges often span significant width
        width_coverage = length / float(W)
        width_bonus = min(1.0, width_coverage * 2)  # Bonus for wider edges

        # Edge strength: sample points along the segment
        strength = sample_edge_strength(user_zone_edges, (x1, y1), (x2, y2))
        strength_norm = strength / 255.0

        # Horizontal continuity: check if edge extends across significant width
        x_span = abs(x2 - x1) / float(W)
        continuity_bonus = min(1.0, x_span * 1.5)

        # Enhanced composite score for user-facing edge detection
        score = (0.35 * length_norm + 
                0.25 * y_pos_norm + 
                0.15 * strength_norm + 
                0.15 * continuity_bonus +
                0.10 * width_bonus) * user_zone_bonus

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
            "user_facing_zone": user_facing_zone
        }
    )


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


# =========================
# Curved edge (Contours)
# =========================
def detect_curved_edge(img_bgr: np.ndarray,
                       edges: np.ndarray,
                       bottom_mask_ratio: float = 0.65,
                       approx_eps_ratio: float = 0.01,
                       resample_points: int = 40) -> Optional[EdgeResult]:
    """
    Focus on bottom region to find curved user-facing desk edges.
    Extract the frontmost envelope (max y by x) representing the user-side edge.
    Returns best EdgeResult or None.
    """
    H, W = edges.shape[:2]

    # Enhanced mask for user-facing zone (bottom portion)
    mask = np.zeros_like(edges)
    y0 = int(H * (1 - bottom_mask_ratio))
    mask[y0:H, :] = 255
    
    # Additional central bias - user typically sits in front of desk center
    center_bias_mask = np.zeros_like(edges)
    center_x = W // 2
    center_width = int(W * 0.8)  # Focus on central 80% of image width
    x_start = max(0, center_x - center_width // 2)
    x_end = min(W, center_x + center_width // 2)
    center_bias_mask[:, x_start:x_end] = 255
    
    # Combine masks
    combined_mask = cv2.bitwise_and(mask, center_bias_mask)
    masked_edges = cv2.bitwise_and(edges, combined_mask)

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
            
        width_norm = w / float(W)
        height_norm = h / float(H)

        # Vertical position: prefer lower edges (user-facing)
        y_bottom = (y + h) / float(H)
        
        # Horizontal span: user-facing edges often span significant width
        x_center = (x + w/2) / float(W)
        center_distance = abs(x_center - 0.5)  # Distance from image center
        center_score = 1.0 - center_distance  # Higher score for more centered edges
        
        # Aspect ratio: user-facing edges tend to be wider than tall
        aspect_ratio = w / max(h, 1)
        aspect_score = min(1.0, aspect_ratio / 5.0)  # Bonus for wide, shallow contours
        
        # Area coverage relative to user zone
        area = cv2.contourArea(cnt)
        area_norm = area / (W * H * bottom_mask_ratio)
        
        # Enhanced composite score for user-facing curved edges
        score = (0.30 * width_norm +           # Width coverage
                0.25 * y_bottom +              # Lower position
                0.20 * center_score +          # Central positioning  
                0.15 * aspect_score +          # Wide aspect ratio
                0.10 * min(1.0, area_norm * 10))  # Reasonable area

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
            "center_bias_applied": True
        }
    )


# =========================
# Choosing best edge
# =========================
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


# =========================
# Annotation & JSON
# =========================
def draw_edge_annotation(img_bgr: np.ndarray, result: EdgeResult) -> np.ndarray:
    out = img_bgr.copy()
    if result.edge_type == "line":
        (x1, y1), (x2, y2) = result.points
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 4)  # blue
    else:
        pts = np.array(result.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=(255, 0, 0), thickness=4)
    return out

def build_json(result: EdgeResult, img_shape: Tuple[int, int, int]) -> Dict:
    H, W = img_shape[:2]
    return {
        "edge_type": result.edge_type,
        "points": result.points,
        "score": round(result.score, 5),
        "method": result.method,
        "image_size": {"width": W, "height": H},
        "metadata": result.metadata
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Desk Edge Detector", page_icon="🧭", layout="wide")
st.title("🧭 User-Facing Desk Edge Detection")
st.caption("Precisely detect and retrieve coordinates of the desk edge facing the user (optimized for office work desks).")

with st.sidebar:
    st.header("User-Facing Edge Detection Settings")
    st.caption("Optimized for detecting desk edges facing the user")
    use_clahe = st.checkbox("Use CLAHE (contrast boost)", value=True)
    min_line_len_ratio = st.slider("Min line length (ratio of width)", 0.05, 0.6, 0.25, 0.05)
    angle_tol = st.slider("Horizontal angle tolerance (deg)", 2, 25, 8, 1)
    bottom_mask_ratio = st.slider("User-facing zone (bottom ratio)", 0.4, 0.9, 0.7, 0.05)
    approx_eps_ratio = st.slider("Contour smoothing (eps ratio)", 0.003, 0.05, 0.008, 0.001)
    resample_points = st.slider("Curve detail points", 8, 200, 35, 1)

    st.header("Debug Views")
    show_gray = st.checkbox("Show grayscale", value=True)
    show_edges = st.checkbox("Show Canny edges", value=True)
    show_hough_overlay = st.checkbox("Show Hough lines overlay", value=True)
    show_contour_overlay = st.checkbox("Show contour overlay", value=True)

uploaded = st.file_uploader("Upload an office desk image (PNG/JPG/WEBP)", type=["png","jpg","jpeg","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_bgr = pil_to_cv(image)
    img_bgr = resize_max_width(img_bgr, 1280)

    gray, blurred, edges, debug = preprocess_for_edges(img_bgr, use_clahe=use_clahe)

    # Straight edge candidate (user-facing)
    straight_res = detect_straight_edge(
        img_bgr, edges,
        min_line_len_ratio=min_line_len_ratio,
        angle_tolerance_deg=angle_tol,
        user_facing_zone=0.6  # Focus on bottom 60% for user-facing edges
    )

    # Curved edge candidate
    curved_res = detect_curved_edge(
        img_bgr, edges,
        bottom_mask_ratio=bottom_mask_ratio,
        approx_eps_ratio=approx_eps_ratio,
        resample_points=resample_points
    )

    result = choose_best_edge(straight_res, curved_res)

    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Annotated Result")
        if result:
            annotated = draw_edge_annotation(img_bgr, result)
            st.image(cv_to_rgb(annotated), caption=f"Detected edge ({result.method}, score={result.score:.3f})", use_container_width=True)

            # JSON
            out_json = build_json(result, img_bgr.shape)
            st.subheader("Retrieved Coordinates (JSON)")
            st.code(json.dumps(out_json, indent=2), language="json")

            # Downloads
            # annotated image
            _, buf_img = cv2.imencode(".png", annotated)
            st.download_button(
                "Download Annotated Image (PNG)",
                data=buf_img.tobytes(),
                file_name="annotated_edge.png",
                mime="image/png"
            )
            # json
            st.download_button(
                "Download Coordinates (JSON)",
                data=json.dumps(out_json).encode("utf-8"),
                file_name="edge_coordinates.json",
                mime="application/json"
            )
        else:
            st.warning("No user-facing desk edge detected. Try adjusting sidebar settings or ensure the image clearly shows the desk edge closest to where a user would sit.")

    with col2:
        st.subheader("Debug Visualizations")
        if show_gray:
            st.image(debug["gray"], caption="Grayscale", use_container_width=True, clamp=True)
        if show_edges:
            st.image(debug["edges"], caption="Canny Edges", use_container_width=True, clamp=True)

        # Hough overlay for debugging (user-facing zone)
        if show_hough_overlay:
            overlay = img_bgr.copy()
            H, W = img_bgr.shape[:2]
            
            # Create user-facing zone mask
            user_zone_start = int(H * (1 - 0.6))  # Match the user_facing_zone parameter
            user_zone_edges = edges.copy()
            user_zone_edges[:user_zone_start, :] = 0
            
            lines = cv2.HoughLinesP(user_zone_edges, 1, np.pi/180, 50,
                                    minLineLength=max(20, int(min_line_len_ratio * W)),
                                    maxLineGap=30)
            if lines is not None:
                for L in lines:
                    x1, y1, x2, y2 = L[0]
                    dx, dy = (x2 - x1), (y2 - y1)
                    angle = np.degrees(np.arctan2(dy, dx))
                    # Color code: green for near-horizontal (user-facing candidates)
                    color = (0, 255, 0) if abs(angle) <= angle_tol else (0, 200, 255)
                    cv2.line(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw user-facing zone boundary
            cv2.line(overlay, (0, user_zone_start), (W, user_zone_start), (255, 255, 0), 2)
            st.image(cv_to_rgb(overlay), caption="User-Facing Zone Hough Lines (Green=Candidates)", use_container_width=True)

        # Contour overlay for debugging (user-facing zone with center bias)
        if show_contour_overlay:
            H, W = edges.shape[:2]
            
            # Bottom mask for user-facing zone
            mask = np.zeros_like(edges)
            y0 = int(H * (1 - bottom_mask_ratio))
            mask[y0:H, :] = 255
            
            # Center bias mask (80% of width centered)
            center_bias_mask = np.zeros_like(edges)
            center_x = W // 2
            center_width = int(W * 0.8)
            x_start = max(0, center_x - center_width // 2)
            x_end = min(W, center_x + center_width // 2)
            center_bias_mask[:, x_start:x_end] = 255
            
            # Combine masks
            combined_mask = cv2.bitwise_and(mask, center_bias_mask)
            masked_edges = cv2.bitwise_and(edges, combined_mask)
            
            # Enhanced morphology
            kernel_close = np.ones((7,7), np.uint8)
            kernel_dilate = np.ones((3,3), np.uint8)
            closed = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
            
            overlay2 = img_bgr.copy()
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                # Color code contours by size (larger = more likely user-facing)
                for i, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w >= W * 0.15 and h >= 5:  # Significant contours
                        cv2.drawContours(overlay2, [cnt], -1, (0, 255, 0), 3)  # Green for candidates
                    else:
                        cv2.drawContours(overlay2, [cnt], -1, (100, 100, 100), 1)  # Gray for small ones
            
            # Draw zone boundaries
            cv2.line(overlay2, (0, y0), (W, y0), (255, 255, 0), 2)  # Bottom zone
            cv2.line(overlay2, (x_start, 0), (x_start, H), (255, 0, 255), 1)  # Center bias left
            cv2.line(overlay2, (x_end, 0), (x_end, H), (255, 0, 255), 1)  # Center bias right
            
            st.image(cv_to_rgb(overlay2), caption="User-Facing Contours (Green=Candidates, Yellow=Zone)", use_container_width=True)

else:
    st.info("Upload an image to begin. Use the sidebar to tune detection parameters.")
