# API Reference

This section provides detailed documentation for all modules and classes in the EVLMs framework.

## Core Modules

### Models
- [ExplainableMedicalVLM](models/medical_vlm.md) - Main model class
- [VisionEncoder](models/vision_encoder.md) - Vision encoder implementation
- [LanguageDecoder](models/language_decoder.md) - Language decoder implementation
- [CrossModalAttention](models/cross_attention.md) - Cross-modal attention mechanism

### Data
- [MedicalImageTextDataset](data/dataset.md) - Dataset class
- [DatasetConfig](data/config.md) - Dataset configuration
- [Transforms](data/transforms.md) - Data augmentation and preprocessing

### Training
- [MedicalVLMTrainer](trainers/trainer.md) - Main trainer class
- [Losses](trainers/losses.md) - Loss functions
- [Metrics](trainers/metrics.md) - Evaluation metrics
- [Optimizers](trainers/optimizers.md) - Optimization utilities

### Visualization
- [ExplanationGenerator](visualization/explanation_generator.md) - Explanation generation
- [GradCAMExplainer](visualization/gradcam.md) - GradCAM implementation
- [AttentionVisualizer](visualization/attention.md) - Attention visualization
- [FeatureVisualizer](visualization/features.md) - Feature visualization

### Configuration
- [Config](config/config.md) - Configuration system
- [DatasetConfig](config/dataset_config.md) - Dataset configuration
- [ModelConfig](config/model_config.md) - Model configuration
- [TrainingConfig](config/training_config.md) - Training configuration

### Utilities
- [Logger](utils/logger.md) - Logging utilities
- [Metrics](utils/metrics.md) - Evaluation metrics
- [Visualization](utils/visualization.md) - Visualization utilities
- [IO](utils/io.md) - Input/output utilities

## Class Reference

### ExplainableMedicalVLM

Main model class for medical image analysis and explanation.

```python
class ExplainableMedicalVLM(nn.Module):
    def __init__(
        self,
        vision_model: str,
        language_model: str,
        num_classes: int
    )
```

**Parameters**:
- `vision_model` (str): Name or path of vision model
- `language_model` (str): Name or path of language model
- `num_classes` (int): Number of medical conditions

**Methods**:
- `forward()`: Forward pass
- `predict()`: Make predictions
- `explain()`: Generate explanations
- `save()`: Save model
- `load()`: Load model

### MedicalImageTextDataset

Dataset class for medical images and text.

```python
class MedicalImageTextDataset(Dataset):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        transform = None
    )
```

**Parameters**:
- `dataset_config`: Dataset configuration
- `tokenizer`: Text tokenizer
- `split`: Data split
- `transform`: Data transforms

**Methods**:
- `__len__()`: Get dataset size
- `__getitem__()`: Get dataset item
- `get_labels()`: Get label list
- `get_splits()`: Get data splits

### MedicalVLMTrainer

Trainer class for model training.

```python
class MedicalVLMTrainer:
    def __init__(
        self,
        model: ExplainableMedicalVLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs
    )
```

**Parameters**:
- `model`: Model instance
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `**kwargs`: Additional arguments

**Methods**:
- `train()`: Train model
- `evaluate()`: Evaluate model
- `save_checkpoint()`: Save checkpoint
- `load_checkpoint()`: Load checkpoint

### ExplanationGenerator

Class for generating model explanations.

```python
class ExplanationGenerator:
    def __init__(
        self,
        model: ExplainableMedicalVLM
    )
```

**Parameters**:
- `model`: Model instance

**Methods**:
- `explain()`: Generate explanation
- `gradcam()`: Generate GradCAM
- `attention()`: Get attention maps
- `feature_importance()`: Get feature importance

## Configuration Reference

### Config

```python
@dataclass
class Config:
    # Model settings
    vision_model_name: str
    language_model_name: str
    num_classes: int
    
    # Training settings
    batch_size: int
    learning_rate: float
    num_epochs: int
    
    # Dataset settings
    dataset_name: DatasetName
    datasets: Dict[str, DatasetConfig]
```

### DatasetConfig

```python
@dataclass
class DatasetConfig:
    name: DatasetName
    data_dir: str
    train_csv: str
    val_csv: str
    test_csv: Optional[str]
    image_column: str
    text_column: str
    label_columns: List[str]
```

## Usage Examples

### Training a Model

```python
from EVLMs.configs.config import get_config
from EVLMs.models.medical_vlm import ExplainableMedicalVLM
from EVLMs.trainers.trainer import MedicalVLMTrainer

# Get configuration
config = get_config()

# Create model
model = ExplainableMedicalVLM(
    vision_model=config.vision_model_name,
    language_model=config.language_model_name,
    num_classes=config.num_classes
)

# Create trainer
trainer = MedicalVLMTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

# Train model
trainer.train()
```

### Generating Explanations

```python
from EVLMs.visualization.explanation_generator import ExplanationGenerator

# Create explainer
explainer = ExplanationGenerator(model)

# Generate explanation
explanation = explainer.explain(
    image_path="image.jpg",
    method="gradcam"
)

# Show explanation
explanation.show()
```

## Next Steps

- Check [example notebooks](../examples/README.md)
- Read the [getting started guide](../guides/getting_started.md)
- Learn about [model architecture](../guides/model_architecture.md) 