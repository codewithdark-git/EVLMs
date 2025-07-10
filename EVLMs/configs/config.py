from dataclasses import dataclass, field
import os
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for a local dataset from a single JSON file."""
    data_dir: str  # Path to the root of the dataset, where images are located.
    json_filename: str = "datasets.json"  # Name of the JSON file
    val_split: float = 0.2  # Fraction for the validation split
    label_columns: List[str] = field(default_factory=lambda: [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ])

@dataclass
class Config:
    # Model settings
    vision_model_name: str = 'microsoft/swin-base-patch4-window7-224'
    language_model_name: str = 'microsoft/DialoGPT-medium'
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    max_text_length: int = 512
    image_size: int = 224
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    
    # Dataset settings
    dataset: Optional[DatasetConfig] = None
    
    # Output settings
    output_dir: str = 'outputs'
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Experiment tracking
    use_wandb: bool = False
    experiment_name: Optional[str] = None
    
    @property
    def num_classes(self) -> int:
        """Number of classes based on the dataset's label columns."""
        if self.dataset:
            return len(self.dataset.label_columns)
        return 0

    def __post_init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"evlm_{self.vision_model_name.split('/')[-1]}"
        
        # Create data directory if it doesn't exist and is specified
        if self.dataset and not os.path.exists(self.dataset.data_dir):
            os.makedirs(self.dataset.data_dir, exist_ok=True)
