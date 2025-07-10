# Working with Datasets

This guide explains how to structure and use your local dataset with the EVLMs framework. The system is designed to work with a specific JSON format that links images to their corresponding radiology reports and labels.

## Dataset Structure

To use the framework, you must organize your data in the following structure. The root directory should contain your `datasets.json` file and the images, which can be in any subdirectory.

```
/path/to/your/dataset/
├── datasets.json
└── ... any other folders with images ...
```

### `datasets.json` Format

The `datasets.json` file is the core of your dataset. It should be a JSON object with a single key, `"enhanced_reports"`, which holds a list of report objects. Each object in the list represents a single data point and must contain the following keys:

-   `"image_path"`: The absolute path to the image file.
-   `"labels"`: A list of strings, where each string is a finding or condition present in the image.
-   `"radiology_report"`: An object containing the text of the report, with the following keys:
    -   `"findings"`: The findings section of the report.
    -   `"impression"`: The impression or conclusion of the report.
    -   `"recommendations"`: Any recommendations for follow-up.

Here is an example of a single entry in the `"enhanced_reports"` list:

```json
{
  "image_path": "/path/to/your/dataset/images/00006763_000.png",
  "labels": [
    "No Finding"
  ],
  "radiology_report": {
    "findings": "The examination demonstrates normal cardiopulmonary findings...",
    "impression": "Normal chest radiograph with no acute cardiopulmonary findings.",
    "recommendations": "No acute intervention needed."
  }
}
```

## Using Your Dataset

The framework is designed to automatically load and process your `datasets.json` file. When you run the training script, you simply need to point to the root directory of your dataset.

### 1. Configuration

The `DatasetConfig` is now simplified to work with the JSON format. You only need to provide the path to your data directory.

```python
from EVLMs.configs.config import Config, DatasetConfig

# The configuration is handled automatically when you run main.py
# This is just for illustration
dataset_config = DatasetConfig(data_dir="path/to/your/dataset")
config = Config(dataset=dataset_config)
```

### 2. Loading the Dataset

The `MedicalImageTextDataset` class will automatically handle loading the `datasets.json` file, splitting it into training and validation sets, and processing the data.

```python
from EVLMs.data.dataset import MedicalImageTextDataset
from transformers import AutoTokenizer

# This is handled internally by the training script
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dataset = MedicalImageTextDataset(
    dataset_config=config.dataset,
    tokenizer=tokenizer,
    split="train"
)
```

### 3. Applying Custom Transforms

You can still apply custom image augmentations. The transforms are defined in `main.py` and passed to the `MedicalImageTextDataset`.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Example of transforms used in the project
transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# The trainer will pass these transforms to the dataset
dataset = MedicalImageTextDataset(
    dataset_config=config.dataset,
    tokenizer=tokenizer,
    split="train",
    transform=transforms
)
```

## Data Preprocessing

The framework automatically handles:
-   Parsing the `datasets.json` file.
-   Splitting the data into training and validation sets.
-   Combining the `findings`, `impression`, and `recommendations` fields into a single report for the language model.
-   One-hot encoding the `labels` for multi-label classification.
-   Applying image augmentations and normalization.
-   Tokenizing the text for the language model.

## Best Practices

1.  **Data Organization**: Ensure your `datasets.json` file is in the root of your dataset directory and that all `image_path` entries are correct.
2.  **Data Integrity**: Verify that your JSON is well-formed and that all required keys are present in each report object.
3.  **Performance**: For very large datasets, consider splitting your `datasets.json` into separate train and validation files for faster initialization, though this would require a minor code modification.

## Next Steps

-   Learn about the [model architecture](model_architecture.md).
-   Explore the [training options](training.md).
-   Check the [visualization tools](visualization.md). 