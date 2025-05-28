import os
import cv2
import numpy as np
from PIL import Image
import pydicom
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MedicalImagePreprocessor:
    """Preprocessor for medical images"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def preprocess_image(self, 
                        image_path: str,
                        apply_clahe: bool = True,
                        window: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Preprocess medical image
        
        Args:
            image_path: Path to image file
            apply_clahe: Whether to apply CLAHE enhancement
            window: Optional window settings (center, width) for DICOM
        
        Returns:
            Preprocessed image array
        """
        # Load image based on format
        if image_path.lower().endswith(('.dcm', '.dicom')):
            image = self._load_dicom(image_path, window)
        else:
            image = self._load_standard_image(image_path)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize
        image = cv2.resize(image, self.target_size)
        
        # Apply CLAHE if requested
        if apply_clahe:
            image = self._apply_clahe(image)
        
        return image
    
    def _load_dicom(self, 
                    path: str,
                    window: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Load and process DICOM image
        
        Args:
            path: Path to DICOM file
            window: Optional window settings (center, width)
        
        Returns:
            Processed image array
        """
        try:
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array
            
            # Apply windowing if provided or available in DICOM
            if window is not None:
                center, width = window
            elif hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                center = dcm.WindowCenter
                width = dcm.WindowWidth
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]
            else:
                center = None
                width = None
            
            if center is not None and width is not None:
                image = self._apply_windowing(image, center, width)
            
            # Normalize to [0, 255]
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {path}: {e}")
            return np.zeros(self.target_size, dtype=np.uint8)
    
    def _load_standard_image(self, path: str) -> np.ndarray:
        """Load standard image formats (PNG, JPG, etc.)
        
        Args:
            path: Path to image file
        
        Returns:
            Image array
        """
        try:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                # Fallback to PIL
                image = np.array(Image.open(path).convert('L'))
            return image
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            return np.zeros(self.target_size, dtype=np.uint8)
    
    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization
        
        Args:
            image: Input image
        
        Returns:
            Enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    @staticmethod
    def _apply_windowing(image: np.ndarray,
                        center: float,
                        width: float) -> np.ndarray:
        """Apply windowing to DICOM image
        
        Args:
            image: Input image
            center: Window center
            width: Window width
        
        Returns:
            Windowed image
        """
        min_value = center - width // 2
        max_value = center + width // 2
        return np.clip(image, min_value, max_value)

def prepare_dataset(input_dir: str,
                   output_dir: str,
                   target_size: Tuple[int, int] = (224, 224)):
    """Prepare dataset by preprocessing all images
    
    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save processed images
        target_size: Size to resize images to
    """
    os.makedirs(output_dir, exist_ok=True)
    preprocessor = MedicalImagePreprocessor(target_size)
    
    # Process all images
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.dicom')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            
            # Preprocess image
            try:
                image = preprocessor.preprocess_image(input_path)
                cv2.imwrite(output_path, image)
                logger.info(f"Processed {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess medical images")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=224, help="Target size")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process dataset
    prepare_dataset(args.input_dir, args.output_dir, (args.size, args.size)) 