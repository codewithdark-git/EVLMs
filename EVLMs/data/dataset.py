import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer
from ..configs.config import DatasetConfig
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MedicalImageTextDataset(Dataset):
    """Medical Image-Text Dataset for local data from a JSON file."""
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 512,
        transform=None
    ):
        self.config = dataset_config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.split = split

        json_path = os.path.join(self.config.data_dir, self.config.json_filename)
        with open(json_path, 'r') as f:
            all_data = json.load(f)['enhanced_reports']
        
        split_index = int(len(all_data) * (1 - self.config.val_split))
        if split == 'train':
            self.data = all_data[:split_index]
        elif split == 'val':
            self.data = all_data[split_index:]
        else:
            raise ValueError(f"Invalid split '{split}'. Only 'train' and 'val' are supported.")

        logger.info(f"Loaded {len(self.data)} samples for {split} split from {json_path}")

    def _process_text(self, report: dict) -> tuple[str, Dict[str, torch.Tensor]]:
        """Combine and process text from the radiology report."""
        full_report = (
            f"Findings: {report.get('findings', '')}\n"
            f"Impression: {report.get('impression', '')}\n"
            f"Recommendations: {report.get('recommendations', '')}"
        )
        
        encoding = self.tokenizer(
            full_report,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        processed_text = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }
        return full_report, processed_text

    def _process_labels(self, report_labels: list) -> torch.Tensor:
        """Process labels from the report."""
        label_tensor = torch.zeros(len(self.config.label_columns), dtype=torch.float32)
        for i, label_name in enumerate(self.config.label_columns):
            if label_name in report_labels:
                label_tensor[i] = 1.0
        return label_tensor

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from the filesystem."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return Image.new('RGB', (224, 224), (0, 0, 0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item by index."""
        item = self.data[idx]
        
        image = self._load_image(item['image_path'])
        
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        
        raw_text, text_encodings = self._process_text(item['radiology_report'])
        
        labels = self._process_labels(item['labels'])
        
        output = {
            "image": image,
            **text_encodings,
            "labels": labels,
            "text": raw_text
        }
        
        return output

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)