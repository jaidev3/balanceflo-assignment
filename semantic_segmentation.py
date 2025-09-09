import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np
import cv2
from typing import Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SemanticSegmentationMask:
    """
    Semantic segmentation module for removing person and chair noise from desk edge detection.
    Uses DeepLabv3 ResNet-101 pre-trained on COCO dataset.
    """
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),  # DeepLabv3 works well with 512x512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # COCO dataset class IDs
        self.PERSON_CLASS_ID = 15
        self.CHAIR_CLASS_ID = 62  # Updated to correct COCO chair class ID
        
    def _load_model(self):
        """Lazy loading of the segmentation model."""
        if self.model is None:
            try:
                self.model = deeplabv3_resnet101(pretrained=True)
                self.model.eval()
                self.model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load segmentation model: {e}")
                return False
        return True
    
    def get_person_chair_mask(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a binary mask for person and chair pixels in the image.
        
        Args:
            image_bgr: Input image in BGR format (OpenCV format)
            
        Returns:
            Binary mask where True indicates person/chair pixels, None if model fails
        """
        if not self._load_model():
            return None
            
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            original_shape = image_rgb.shape[:2]
            
            # Preprocess image
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)['out']
                predictions = output.argmax(1).byte().cpu().numpy().squeeze()
            
            # Create mask for person and chair classes
            mask_512 = np.isin(predictions, [self.PERSON_CLASS_ID, self.CHAIR_CLASS_ID])
            
            # Resize mask back to original image size
            mask_resized = cv2.resize(
                mask_512.astype(np.uint8), 
                (original_shape[1], original_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            return mask_resized
            
        except Exception as e:
            print(f"Warning: Segmentation failed: {e}")
            return None
    
    def apply_mask_to_image(self, image_bgr: np.ndarray, mask: np.ndarray, 
                           fill_value: int = 0) -> np.ndarray:
        """
        Apply the person/chair mask to remove those areas from the image.
        
        Args:
            image_bgr: Input image in BGR format
            mask: Binary mask where True indicates areas to remove
            fill_value: Value to fill masked areas with (default: 0 for black)
            
        Returns:
            Image with person/chair areas masked out
        """
        masked_image = image_bgr.copy()
        masked_image[mask] = fill_value
        return masked_image
    
    def get_cleaned_image(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Complete pipeline to get a cleaned image with person/chair areas removed.
        
        Args:
            image_bgr: Input image in BGR format
            
        Returns:
            Tuple of (cleaned_image, mask) where mask is None if segmentation failed
        """
        mask = self.get_person_chair_mask(image_bgr)
        
        if mask is None:
            # If segmentation fails, return original image
            return image_bgr.copy(), None
        
        cleaned_image = self.apply_mask_to_image(image_bgr, mask)
        return cleaned_image, mask


# Global instance for reuse
_segmentation_instance = None

def get_segmentation_instance() -> SemanticSegmentationMask:
    """Get or create a global segmentation instance for efficiency."""
    global _segmentation_instance
    if _segmentation_instance is None:
        _segmentation_instance = SemanticSegmentationMask()
    return _segmentation_instance


def remove_person_chair_noise(image_bgr: np.ndarray, 
                              use_segmentation: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to remove person and chair noise from an image.
    
    Args:
        image_bgr: Input image in BGR format
        use_segmentation: Whether to apply semantic segmentation
        
    Returns:
        Tuple of (processed_image, mask) where mask is None if segmentation disabled/failed
    """
    if not use_segmentation:
        return image_bgr.copy(), None
    
    segmenter = get_segmentation_instance()
    return segmenter.get_cleaned_image(image_bgr)