import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Dict, Any
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from ..configs.config import DatasetConfig, HFDatasetConfig
import logging
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class MedicalImageTextDataset(Dataset):
    """Medical Image-Text Dataset that automatically loads from Hugging Face"""
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 512,
        img_size: int = 224,
        transform=None
    ):
        """
        Initialize the dataset
        
        Args:
            dataset_config: Dataset configuration
            tokenizer: Tokenizer for text processing
            split: Data split (train/val/test)
            max_length: Maximum text length
            img_size: Image size for resizing
            transform: Optional transform pipeline
        """
        self.config = dataset_config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_size = img_size
        self.transform = transform
        self.split = split
        
        # Load dataset from Hugging Face if configured
        if self.config.hf_config:
            self._load_from_huggingface(split)
        else:
            raise ValueError("Dataset must be configured with Hugging Face config")
            
        logger.info(f"Loaded {len(self.dataset)} samples for {split} split")
    
    def _load_from_huggingface(self, split: str):
        """Load dataset from Hugging Face Hub"""
        hf_config: HFDatasetConfig = self.config.hf_config
        
        # Map our split name to dataset split name
        dataset_split = hf_config.split_mapping.get(split)
        if not dataset_split:
            raise ValueError(f"Split {split} not found in dataset config")
            
        # Load dataset
        try:
            self.dataset = load_dataset(
                hf_config.name,
                name=hf_config.subset,
                split=dataset_split,
                cache_dir=self.config.data_dir
            )
        except Exception as e:
            logger.error(f"Error loading dataset {hf_config.name}: {str(e)}")
            raise
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text using tokenizer"""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0]
        }
    
    def _process_labels(self, item: Dict[str, Any]) -> torch.Tensor:
        """Process labels from dataset item"""
        if not self.config.label_columns:
            return None
            
        labels = []
        for col in self.config.label_columns:
            # Handle different label formats (0/1, True/False, etc.)
            val = item.get(col, 0)
            if isinstance(val, bool):
                val = 1 if val else 0
            elif isinstance(val, str):
                val = 1 if val.lower() in ['1', 'true', 'positive'] else 0
            labels.append(float(val))
            
        return torch.tensor(labels, dtype=torch.float32)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image"""
        try:
            # Handle both local paths and URLs
            if image_path.startswith(('http://', 'https://')):
                # If image is a URL (some HF datasets provide URLs)
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                # If image is a local path
                image = Image.open(image_path)
                
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item by index"""
        item = self.dataset[idx]
        
        # Get image
        image_data = item[self.config.hf_config.image_column]
        image = self._load_image(image_data)
        
        if self.transform:
            image = self.transform(image)
            
        # Get text
        text = item[self.config.hf_config.text_column]
        text_encodings = self._process_text(text)
        
        # Get labels
        labels = self._process_labels(item)
        
        output = {
            "image": image,
            **text_encodings,
        }
        
        if labels is not None:
            output["labels"] = labels
            
        return output
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.dataset)

def create_sample_dataset(output_path: str, num_samples: int = 1000):
    """Create a sample dataset CSV for testing
    
    Args:
        output_path: Path to save the CSV file
        num_samples: Number of samples to generate
    """
    # Sample data structure
    data = {
        'image_id': [f'img_{i:05d}' for i in range(num_samples)],
        'image_filename': [f'img_{i:05d}.jpg' for i in range(num_samples)],
        'patient_id': [f'patient_{i//10:04d}' for i in range(num_samples)],
        'study_id': [f'study_{i:05d}' for i in range(num_samples)],
        
        # Patient demographics
        'patient_age': np.random.randint(18, 90, num_samples),
        'patient_sex': np.random.choice(['M', 'F'], num_samples),
        
        # Image metadata
        'view_position': np.random.choice(['PA', 'AP', 'LAT'], num_samples),
        'modality': ['CR'] * num_samples,  # Chest Radiography
    }
    
    # Add pathology labels
    label_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    for col in label_columns:
        data[col] = np.random.choice([0, 1, -1], num_samples, p=[0.7, 0.2, 0.1])
    
    # Generate reports
    findings = []
    impressions = []
    
    for i in range(num_samples):
        sample_findings = []
        sample_impression = []
        
        # Generate based on labels
        if data['No Finding'][i] == 1:
            sample_findings.append("The lungs are clear without focal consolidation.")
            sample_impression.append("No acute cardiopulmonary abnormality.")
        else:
            if data['Atelectasis'][i] == 1:
                sample_findings.append("There is evidence of atelectasis in the lung bases.")
            if data['Cardiomegaly'][i] == 1:
                sample_findings.append("The cardiac silhouette appears enlarged.")
                sample_impression.append("Cardiomegaly.")
            if data['Pneumonia'][i] == 1:
                sample_findings.append("Patchy opacities consistent with pneumonia.")
                sample_impression.append("Findings consistent with pneumonia.")
        
        findings.append(" ".join(sample_findings) if sample_findings else "No significant findings.")
        impressions.append(" ".join(sample_impression) if sample_impression else "Normal study.")
    
    data['findings'] = findings
    data['impression'] = impressions
    data['report_text'] = [f"FINDINGS: {f}\n\nIMPRESSION: {i}" 
                          for f, i in zip(findings, impressions)]
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    return df 