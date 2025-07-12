# API Reference

This section provides detailed documentation for the core modules and classes in the EVLMs framework.

## Core Modules

### Models
-   `ExplainableMedicalVLM`: The main model that integrates all components.
-   `MedicalVisionEncoder`: The vision encoder for processing images.
-   `MedicalLanguageDecoder`: The language decoder for generating reports.
-   `CrossModalAttention`: The attention mechanism for fusing vision and language features.

### Data
-   `MedicalImageTextDataset`: The dataset class for loading data from a local `datasets.json` file.
-   `DatasetConfig`: The configuration for specifying the dataset path.

### Training
-   `MedicalVLMTrainer`: The main trainer class that handles the training loop, optimization, and evaluation.

### Visualization
-   `ExplanationGenerator`: A class for generating visual explanations like GradCAM and attention maps.

### Configuration
-   `Config`: The main configuration class that holds settings for the model, trainer, and dataset.

## Class Reference

### ExplainableMedicalVLM

The main model class for medical image analysis and explanation.

```python
class ExplainableMedicalVLM(nn.Module):
    def __init__(
        self,
        vision_model: str,
        language_model: str,
        num_classes: int
    )
```

### MedicalImageTextDataset

Dataset class for loading medical images and text from a local `datasets.json` file.

```python
class MedicalImageTextDataset(Dataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        transform=None
    )
```

### MedicalVLMTrainer

Trainer class for model training.

```python
class MedicalVLMTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any,
        val_dataset: Any,
        # ... other training parameters
    )
```

## Configuration Reference

### Config

```python
@dataclass
class Config:
    # Model settings
    vision_model_name: str
    language_model_name: str
    
    # Training settings
    batch_size: int
    learning_rate: float
    num_epochs: int
    
    # Dataset settings
    dataset: Optional[DatasetConfig] = None
```

### DatasetConfig

```python
@dataclass
class DatasetConfig:
    data_dir: str
    json_filename: str = "datasets.json"
    val_split: float = 0.2
    label_columns: List[str] = field(default_factory=lambda: [...])
```

## Usage Example

### Training a Model

Training is initiated from the command line. See the [Getting Started](https://github.com/codewithdark-git/ExplainableVisionLanguageModels-EVLMs/blob/main/docs/guides/getting_started.md) guide for details.

```bash
python main.py --dataset_path /path/to/your/dataset
```

### Generating Explanations

```python
from EVLMs.models.medical_vlm import ExplainableMedicalVLM
from EVLMs.visualization.explanation_generator import ExplanationGenerator

# Load your trained model
model = ExplainableMedicalVLM(...)
model.load_state_dict(torch.load('path/to/your/best_model.pth'))

# Create explainer
explainer = ExplanationGenerator(model)

# Generate explanation
explanation = explainer.generate_visual_explanation(
    image_tensor, target_class=0
)
```

## Next Steps

-   Check the [Getting Started Guide](../guides/getting_started.md)
-   Learn about the [Model Architecture](../guides/model_architecture.md)
 