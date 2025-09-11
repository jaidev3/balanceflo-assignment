import io
import json
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Import separated modules
from data_models import EdgeResult
from image_utils import pil_to_cv, cv_to_rgb, resize_max_width, preprocess_for_edges
from edge_detection_core import (
    detect_straight_edge,
    detect_curved_edge,
    choose_best_edge,
    draw_edge_annotation,
    build_json
)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Desk Edge Detector", page_icon="🧭", layout="wide")
st.title("🧭 Desk Edge Detection")
st.caption("AI-powered desk edge detection that works with people in the scene and various camera angles. Uses semantic segmentation to remove person/chair noise and flexible algorithms for robust detection.")

with st.sidebar:
    st.header("Edge Detection Settings")
    st.caption("Robust detection for various camera angles and scenes with people")
    
    st.subheader("🤖 AI Enhancement")
    use_segmentation = st.checkbox("Use AI person/chair removal", value=True, 
                                  help="Uses DeepLabv3 to remove people and chairs from the image before edge detection")
    
    st.subheader("📐 Edge Detection")
    use_clahe = st.checkbox("Use CLAHE (contrast boost)", value=True)
    resize_image = st.checkbox("Resize image to max width 1280px", value=True, 
                              help="Resize large images for faster processing. Uncheck to keep original size.")
    min_line_len_ratio = st.slider("Min line length (ratio of width)", 0.05, 0.6, 0.25, 0.05)
    angle_tol = st.slider("Angle tolerance (deg)", 10, 80, 60, 5, 
                         help="Higher values allow detection of diagonal desk edges from angled camera views")
    bottom_mask_ratio = st.slider("User-facing zone (bottom ratio)", 0.4, 0.9, 0.7, 0.05)
    approx_eps_ratio = st.slider("Contour smoothing (eps ratio)", 0.003, 0.05, 0.008, 0.001)
    resample_points = st.slider("Curve detail points", 8, 200, 35, 1)
    boundary_margin = st.slider("Boundary margin (ignore edge ratio)", 0.01, 0.15, 0.05, 0.01, 
                               help="Ignore edges within this ratio of image boundaries to filter out image frame artifacts")

    st.header("Debug Views")
    show_gray = st.checkbox("Show grayscale", value=True)
    show_edges = st.checkbox("Show Canny edges", value=True)
    show_hough_overlay = st.checkbox("Show Hough lines overlay", value=True)
    show_contour_overlay = st.checkbox("Show contour overlay", value=True)

uploaded = st.file_uploader("Upload an office desk image (PNG/JPG/WEBP)", type=["png","jpg","jpeg","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_bgr = pil_to_cv(image)
    if resize_image:
        img_bgr = resize_max_width(img_bgr, 1280)

    gray, blurred, edges, debug = preprocess_for_edges(img_bgr, use_clahe=use_clahe, use_segmentation=use_segmentation)

    # Run edge detection algorithms
    straight_res = detect_straight_edge(
        img_bgr, edges,
        min_line_len_ratio=min_line_len_ratio,
        angle_tolerance_deg=angle_tol,
        user_facing_zone=0.6,  # Focus on bottom 60% for user-facing edges
        boundary_margin=boundary_margin
    )

    st.write(f"Straight edge detection result: {straight_res}")

    curved_res = detect_curved_edge(
        img_bgr, edges,
        bottom_mask_ratio=bottom_mask_ratio,
        approx_eps_ratio=approx_eps_ratio,
        resample_points=resample_points,
        boundary_margin=boundary_margin
    )

    st.write(f"Curved edge detection result: {curved_res}")
    
    result = choose_best_edge(straight_res, curved_res)

    st.write(f"Best edge detection result: {result}")

    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Annotated Result")
        if result:
            annotated = draw_edge_annotation(img_bgr, result)
            st.image(cv_to_rgb(annotated), caption=f"Detected edge ({result.method}, score={result.score:.3f})", use_container_width=True)

            # JSON output
            out_json = build_json(result, img_bgr.shape)
            st.subheader("Retrieved Coordinates (JSON)")
            st.code(json.dumps(out_json, indent=2), language="json")

            # Download buttons
            _, buf_img = cv2.imencode(".png", annotated)
            st.download_button(
                "Download Annotated Image (PNG)",
                data=buf_img.tobytes(),
                file_name="annotated_edge.png",
                mime="image/png"
            )

            st.download_button(
                "Download Coordinates (JSON)",
                data=json.dumps(out_json).encode("utf-8"),
                file_name="edge_coordinates.json",
                mime="application/json"
            )
        else:
            st.warning("No desk edge detected. Try adjusting sidebar settings, enabling AI person/chair removal, or ensure the image clearly shows a desk edge.")

    with col2:
        st.subheader("Debug Visualizations")
        
        # Show segmentation results if enabled
        if use_segmentation and debug.get("segmentation_mask") is not None:
            st.image(cv_to_rgb(debug["processed_img"]), caption="AI-Cleaned Image (Person/Chair Removed)", use_container_width=True)
            mask_viz = (debug["segmentation_mask"] * 255).astype(np.uint8)
            st.image(mask_viz, caption="Person/Chair Detection Mask", use_container_width=True, clamp=True)
        
        if show_gray:
            st.image(debug["gray"], caption="Grayscale", use_container_width=True, clamp=True)
        if show_edges:
            st.image(debug["edges"], caption="Canny Edges", use_container_width=True, clamp=True)

        # Debug overlay for Hough lines
        if show_hough_overlay:
            overlay = img_bgr.copy()
            H, W = img_bgr.shape[:2]
            
            # Calculate boundary margins
            margin_x = int(W * boundary_margin)
            margin_y = int(H * boundary_margin)
            
            # Match the user_facing_zone parameter
            user_zone_start = int(H * (1 - 0.6))
            user_zone_edges = edges.copy()
            user_zone_edges[:user_zone_start, :] = 0
            
            # Apply boundary filtering
            user_zone_edges[:, :margin_x] = 0  # Left boundary
            user_zone_edges[:, W-margin_x:] = 0  # Right boundary
            user_zone_edges[:margin_y, :] = 0  # Top boundary
            user_zone_edges[H-margin_y:, :] = 0  # Bottom boundary
            
            lines = cv2.HoughLinesP(user_zone_edges, 1, np.pi/180, 50,
                                    minLineLength=max(20, int(min_line_len_ratio * W)),
                                    maxLineGap=30)
            if lines is not None:
                for L in lines:
                    x1, y1, x2, y2 = L[0]
                    dx, dy = (x2 - x1), (y2 - y1)
                    angle = np.degrees(np.arctan2(dy, dx))
                    # Green for candidates, orange for others
                    color = (0, 255, 0) if abs(angle) <= angle_tol else (0, 200, 255)
                    cv2.line(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw user zone boundary
            cv2.line(overlay, (0, user_zone_start), (W, user_zone_start), (255, 255, 0), 2)
            # Draw boundary margins
            cv2.rectangle(overlay, (0, 0), (margin_x, H), (255, 0, 255), 2)  # Left margin
            cv2.rectangle(overlay, (W-margin_x, 0), (W, H), (255, 0, 255), 2)  # Right margin
            cv2.rectangle(overlay, (0, 0), (W, margin_y), (255, 0, 255), 2)  # Top margin
            cv2.rectangle(overlay, (0, H-margin_y), (W, H), (255, 0, 255), 2)  # Bottom margin
            st.image(cv_to_rgb(overlay), caption="User-Facing Zone Hough Lines (Green=Candidates, Magenta=Boundary Margins)", use_container_width=True)

        # Debug overlay for contours
        if show_contour_overlay:
            H, W = edges.shape[:2]
            
            # Calculate boundary margins
            margin_x = int(W * boundary_margin)
            margin_y = int(H * boundary_margin)
            
            # Create masks for user-facing zone
            mask = np.zeros_like(edges)
            y0 = int(H * (1 - bottom_mask_ratio))
            mask[y0:H, :] = 255
            
            # Apply boundary filtering
            mask[:, :margin_x] = 0  # Left boundary
            mask[:, W-margin_x:] = 0  # Right boundary
            mask[:margin_y, :] = 0  # Top boundary
            mask[H-margin_y:, :] = 0  # Bottom boundary
            
            # Apply mask and morphology
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Morphological operations
            kernel_close = np.ones((7,7), np.uint8)
            kernel_dilate = np.ones((3,3), np.uint8)
            closed = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
            
            overlay2 = img_bgr.copy()
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                # Draw contours with different colors based on size
                for i, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w >= W * 0.15 and h >= 5:  # Significant contours
                        cv2.drawContours(overlay2, [cnt], -1, (0, 255, 0), 3)  # Green for candidates
                    else:
                        cv2.drawContours(overlay2, [cnt], -1, (100, 100, 100), 1)  # Gray for small ones
            
            # Draw zone boundaries
            cv2.line(overlay2, (0, y0), (W, y0), (255, 255, 0), 2)  # Bottom zone
            # Draw boundary margins
            cv2.rectangle(overlay2, (0, 0), (margin_x, H), (255, 0, 255), 2)  # Left margin
            cv2.rectangle(overlay2, (W-margin_x, 0), (W, H), (255, 0, 255), 2)  # Right margin
            cv2.rectangle(overlay2, (0, 0), (W, margin_y), (255, 0, 255), 2)  # Top margin
            cv2.rectangle(overlay2, (0, H-margin_y), (W, H), (255, 0, 255), 2)  # Bottom margin

            st.image(cv_to_rgb(overlay2), caption="User-Facing Contours (Green=Candidates, Yellow=Zone, Magenta=Boundary Margins)", use_container_width=True)

else:
    st.info("Upload an image to begin. Use the sidebar to tune detection parameters.")