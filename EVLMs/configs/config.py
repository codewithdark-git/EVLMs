from dataclasses import dataclass
import os
from typing import Optional, List, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DatasetName(str, Enum):
    CHEXPERT = "chexpert"
    MIMIC_CXR = "mimic_cxr"
    NIH_CHEST = "nih_chest"
    CUSTOM = "custom"

@dataclass
class HFDatasetConfig:
    """Hugging Face Dataset Configuration"""
    name: str  # Dataset name on HF hub
    subset: Optional[str] = None  # Dataset subset/configuration
    split_mapping: Dict[str, str] = None  # Maps our splits to dataset splits
    image_column: str = "image"  # Column containing image data/path
    text_column: str = "text"  # Column containing text data
    label_columns: List[str] = None  # Columns containing labels
    
    def __post_init__(self):
        if self.split_mapping is None:
            self.split_mapping = {
                "train": "train",
                "val": "validation",
                "test": "test"
            }

@dataclass
class DatasetConfig:
    name: DatasetName
    data_dir: str  # Local cache directory
    train_csv: str
    val_csv: str
    test_csv: Optional[str] = None
    image_column: str = "image_filename"
    text_column: str = "report_text"
    label_columns: List[str] = None
    hf_config: Optional[HFDatasetConfig] = None
    
    def __post_init__(self):
        if self.label_columns is None:
            # Default medical condition labels
            self.label_columns = [
                'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                'Pneumonia', 'Pneumothorax', 'Support Devices'
            ]

@dataclass
class Config:
    # Model settings
    vision_model_name: str = 'microsoft/swin-base-patch4-window7-224'
    language_model_name: str = 'microsoft/DialoGPT-medium'
    num_classes: int = 14  # Number of medical conditions to classify
    
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
    dataset_name: DatasetName = DatasetName.CHEXPERT
    datasets: Dict[str, DatasetConfig] = None
    
    # Output settings
    output_dir: str = 'outputs'
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Experiment tracking
    use_wandb: bool = False
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        # Initialize dataset configurations
        self.datasets = {
            DatasetName.CHEXPERT: DatasetConfig(
                name=DatasetName.CHEXPERT,
                data_dir="data/CheXpert-v1.0",
                train_csv="train.csv",
                val_csv="valid.csv",
                image_column="Path",
                text_column="Report Impression",
                label_columns=[
                    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                    'Pleural Other', 'Fracture', 'Support Devices'
                ],
                hf_config=HFDatasetConfig(
                    name="danjacobellis/chexpert",
                    image_column="image",
                    text_column="report",
                    split_mapping={
                        "train": "train",
                        "val": "validation"
                    }
                )
            ),
            DatasetName.MIMIC_CXR: DatasetConfig(
                name=DatasetName.MIMIC_CXR,
                data_dir="data/MIMIC-CXR",
                train_csv="train/train.csv",
                val_csv="valid/valid.csv",
                test_csv="test/test.csv",
                image_column="dicom_id",
                text_column="report",
                label_columns=[
                    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                    'Airspace Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                    'Pleural Other', 'Fracture', 'Support Devices'
                ],
                hf_config=HFDatasetConfig(
                    name="medicalai/mimic-cxr",
                    image_column="image",
                    text_column="report",
                    split_mapping={
                        "train": "train",
                        "val": "validation",
                        "test": "test"
                    }
                )
            ),
            DatasetName.NIH_CHEST: DatasetConfig(
                name=DatasetName.NIH_CHEST,
                data_dir="data/NIH-Chest-Xray",
                train_csv="train_list.txt",
                val_csv="val_list.txt",
                test_csv="test_list.txt",
                image_column="Image Index",
                text_column="Finding Labels",
                label_columns=[
                    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 
                    'Hernia'
                ],
                hf_config=HFDatasetConfig(
                    name="alkzar90/NIH-Chest-X-ray14",
                    image_column="image",
                    text_column="finding",
                    split_mapping={
                        "train": "train",
                        "val": "validation",
                        "test": "test"
                    }
                )
            ),
            DatasetName.CUSTOM: DatasetConfig(
                name=DatasetName.CUSTOM,
                data_dir="data/custom",
                train_csv="train.csv",
                val_csv="val.csv",
                image_column="image_filename",
                text_column="report_text"
            )
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"evlm_{self.vision_model_name.split('/')[-1]}_{self.dataset_name}"
        
        # Create data directories if they don't exist
        dataset_config = self.datasets[self.dataset_name]
        os.makedirs(dataset_config.data_dir, exist_ok=True)

def get_config(dataset_name: str = DatasetName.CHEXPERT) -> Config:
    """Get configuration for specific dataset
    
    Args:
        dataset_name: Name of the dataset to use
    
    Returns:
        Configuration object
    """
    config = Config(dataset_name=dataset_name)
    
    # Update number of classes based on dataset
    config.num_classes = len(config.datasets[dataset_name].label_columns)
    
    return config