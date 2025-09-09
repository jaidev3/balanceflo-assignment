import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional


def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_rgb(cv_img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR to RGB (for Streamlit display)."""
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def resize_max_width(img_bgr: np.ndarray, max_w: int = 1280) -> np.ndarray:
    """Resize image to maximum width while maintaining aspect ratio."""
    h, w = img_bgr.shape[:2]
    if w <= max_w:
        return img_bgr
    scale = max_w / float(w)
    new_size = (max_w, int(h * scale))
    return cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)


def preprocess_for_edges(img_bgr: np.ndarray, use_clahe: bool = True, 
                         use_segmentation: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Preprocess image for edge detection with optional semantic segmentation.
    
    Args:
        img_bgr: Input image in BGR format
        use_clahe: Whether to apply CLAHE contrast enhancement
        use_segmentation: Whether to apply semantic segmentation to remove person/chair noise
    
    Returns:
        gray: Grayscale image
        blurred: Blurred image
        edges: Canny edge detection result
        debug: Dictionary with debug information including segmentation mask
    """
    processed_img = img_bgr.copy()
    segmentation_mask = None
    
    # Apply semantic segmentation if enabled
    if use_segmentation:
        try:
            from semantic_segmentation import remove_person_chair_noise
            processed_img, segmentation_mask = remove_person_chair_noise(img_bgr, use_segmentation=True)
        except ImportError:
            print("Warning: Semantic segmentation not available. Install torch and torchvision.")
        except Exception as e:
            print(f"Warning: Semantic segmentation failed: {e}")
    
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    debug = {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "processed_img": processed_img,
        "segmentation_mask": segmentation_mask,
        "segmentation_enabled": use_segmentation
    }
    
    return gray, blurred, edges, debug