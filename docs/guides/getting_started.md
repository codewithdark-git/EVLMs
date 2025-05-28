# Getting Started with EVLMs

This guide will help you set up and start using the Explainable Vision-Language Models (EVLMs) framework.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ExplainableVisionLanguageModels-EVLMs.git
cd ExplainableVisionLanguageModels-EVLMs
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Training with a Predefined Dataset

```python
from EVLMs.configs.config import get_config, DatasetName
from EVLMs.data.dataset import MedicalImageTextDataset
from transformers import AutoTokenizer

# Get configuration for CheXpert dataset
config = get_config(DatasetName.CHEXPERT)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Create dataset
dataset = MedicalImageTextDataset(
    dataset_config=config.datasets[config.dataset_name],
    tokenizer=tokenizer,
    split="train"
)
```

### 2. Training with a Custom Dataset

```python
from EVLMs.configs.config import Config, DatasetConfig, DatasetName

# Create custom dataset configuration
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
                'Finding1', 'Finding2', 'Finding3'
            ]
        )
    }
)

# Create dataset
custom_dataset = MedicalImageTextDataset(
    dataset_config=custom_config.datasets[DatasetName.CUSTOM],
    tokenizer=tokenizer,
    split="train"
)
```

### 3. Running Training

```python
from EVLMs.models.medical_vlm import ExplainableMedicalVLM
from EVLMs.trainers.trainer import MedicalVLMTrainer

# Initialize model
model = ExplainableMedicalVLM(
    vision_model=config.vision_model_name,
    language_model=config.language_model_name,
    num_classes=config.num_classes
)

# Initialize trainer
trainer = MedicalVLMTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=config.learning_rate,
    num_epochs=config.num_epochs
)

# Start training
trainer.train()
```

### 4. Generating Explanations

```python
from EVLMs.visualization.explanation_generator import ExplanationGenerator

# Initialize explanation generator
explainer = ExplanationGenerator(model)

# Generate explanation for an image
explanation = explainer.explain(
    image_path="path/to/image.jpg",
    method="gradcam"  # or "attention", "integrated_gradients"
)

# Visualize explanation
explanation.show()
```

## Configuration

The framework uses a configuration system to manage model parameters, dataset settings, and training options. You can customize these settings in several ways:

1. Using predefined configurations:
```python
config = get_config(DatasetName.CHEXPERT)
```

2. Modifying existing configurations:
```python
config.batch_size = 16
config.learning_rate = 1e-4
```

3. Creating custom configurations:
```python
from EVLMs.configs.config import Config

custom_config = Config(
    vision_model_name="microsoft/swin-base-patch4-window7-224",
    language_model_name="microsoft/DialoGPT-medium",
    batch_size=8,
    learning_rate=5e-5
)
```

## Next Steps

- Learn more about [available datasets](datasets.md)
- Understand the [model architecture](model_architecture.md)
- Explore [training options](training.md)
- Check out [visualization tools](visualization.md)
- See [example notebooks](../examples/README.md) 