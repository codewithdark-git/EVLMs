# Working with Datasets

This guide explains how to work with different datasets in the EVLMs framework, including predefined medical datasets and custom datasets.

## Supported Datasets

### 1. CheXpert Dataset
- **Source**: Stanford ML Group
- **Size**: 224,316 chest radiographs of 65,240 patients
- **Labels**: 14 medical conditions
- **Features**: 
  - High-quality chest X-rays
  - Detailed radiology reports
  - Hierarchical label structure

```python
from EVLMs.configs.config import get_config, DatasetName

config = get_config(DatasetName.CHEXPERT)
```

### 2. MIMIC-CXR Dataset
- **Source**: PhysioNet
- **Size**: 377,110 chest X-rays with 227,827 imaging studies
- **Labels**: 14 medical conditions
- **Features**:
  - Free-text radiology reports
  - Multiple views per study
  - Temporal information

```python
config = get_config(DatasetName.MIMIC_CXR)
```

### 3. NIH Chest X-ray Dataset
- **Source**: National Institutes of Health
- **Size**: 112,120 X-ray images from 30,805 patients
- **Labels**: 14 disease categories
- **Features**:
  - Disease location information
  - Multiple disease labels
  - Demographic information

```python
config = get_config(DatasetName.NIH_CHEST)
```

## Dataset Structure

Each dataset should follow this structure:

```
dataset_root/
├── train.csv
├── val.csv
├── test.csv (optional)
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### CSV Format

The CSV files should contain the following columns:
- Image path/identifier
- Report text
- Labels (0/1 for each condition)

Example:
```csv
image_path,report_text,label1,label2,...
images/001.jpg,"Normal chest x-ray...",0,1,...
```

## Using Predefined Datasets

1. Basic Usage:
```python
from EVLMs.data.dataset import MedicalImageTextDataset
from transformers import AutoTokenizer

# Get dataset configuration
config = get_config(DatasetName.CHEXPERT)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)

# Create dataset
dataset = MedicalImageTextDataset(
    dataset_config=config.datasets[config.dataset_name],
    tokenizer=tokenizer,
    split="train"
)
```

2. With Custom Transforms:
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

dataset = MedicalImageTextDataset(
    dataset_config=config.datasets[config.dataset_name],
    tokenizer=tokenizer,
    split="train",
    transform=transforms
)
```

## Using Custom Datasets

1. Prepare Your Dataset:
   - Organize your images and labels as described above
   - Create CSV files with required columns

2. Create Custom Configuration:
```python
from EVLMs.configs.config import Config, DatasetConfig, DatasetName

custom_config = Config(
    dataset_name=DatasetName.CUSTOM,
    datasets={
        DatasetName.CUSTOM: DatasetConfig(
            name=DatasetName.CUSTOM,
            data_dir="path/to/your/data",
            train_csv="train.csv",
            val_csv="val.csv",
            image_column="image_path",
            text_column="report_text",
            label_columns=[
                'Condition1', 'Condition2', 'Condition3'
            ]
        )
    }
)
```

3. Use Custom Dataset:
```python
dataset = MedicalImageTextDataset(
    dataset_config=custom_config.datasets[DatasetName.CUSTOM],
    tokenizer=tokenizer,
    split="train"
)
```

## Using Hugging Face Datasets

The framework supports loading datasets directly from the Hugging Face Hub:

1. Configure HF Dataset:
```python
from EVLMs.configs.config import HFDatasetConfig

hf_config = HFDatasetConfig(
    name="your/dataset",
    image_column="image",
    text_column="text",
    split_mapping={
        "train": "train",
        "val": "validation"
    }
)
```

2. Use in Dataset Config:
```python
dataset_config = DatasetConfig(
    name=DatasetName.CUSTOM,
    data_dir="cache/dir",
    train_csv="train.csv",
    val_csv="val.csv",
    hf_config=hf_config
)
```

## Data Preprocessing

The framework automatically handles:
- Image resizing and augmentation
- Text tokenization
- Label processing
- Missing value handling

You can customize preprocessing by:
1. Modifying transforms
2. Adjusting tokenizer settings
3. Implementing custom preprocessing methods

## Best Practices

1. **Data Organization**:
   - Keep consistent directory structure
   - Use meaningful file names
   - Include metadata in CSV files

2. **Preprocessing**:
   - Normalize images appropriately
   - Clean text data
   - Handle missing values

3. **Performance**:
   - Use appropriate batch sizes
   - Enable data loading optimization
   - Cache preprocessed data when possible

4. **Validation**:
   - Check data distribution
   - Verify label balance
   - Monitor data quality

## Troubleshooting

Common issues and solutions:

1. **Data Loading Errors**:
   - Verify file paths
   - Check CSV format
   - Ensure all images exist

2. **Memory Issues**:
   - Reduce batch size
   - Use data streaming
   - Enable memory efficient loading

3. **Performance Issues**:
   - Increase num_workers
   - Use appropriate image size
   - Enable caching

## Next Steps

- Learn about [model architecture](model_architecture.md)
- Explore [training options](training.md)
- Check [visualization tools](visualization.md) 