# Getting Started with EVLMs

This guide will help you set up and start using the Explainable Vision-Language Models (EVLMs) framework.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ExplainableVisionLanguageModels-EVLMs.git
    cd ExplainableVisionLanguageModels-EVLMs
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Basic Usage

### 1. Prepare Your Dataset

Before you can start training, you must prepare your dataset as described in the [Working with Datasets](datasets.md) guide. You will need a root directory containing your `datasets.json` file and all the associated images.

### 2. Running Training

To train the model, run the `main.py` script and provide the path to your dataset directory using the `--dataset_path` argument.

```bash
python main.py --dataset_path /path/to/your/dataset
```

This single command will handle:
-   Loading the configuration.
-   Initializing the model, tokenizer, and trainer.
-   Creating the training and validation datasets from your `datasets.json` file.
-   Starting the training process.

### 3. Generating Explanations

After training, you can use the saved model to generate explanations for your images.

```python
from EVLMs.models.medical_vlm import ExplainableMedicalVLM
from EVLMs.visualization.explanation_generator import ExplanationGenerator

# Load your trained model
model = ExplainableMedicalVLM(
    vision_model='microsoft/swin-base-patch4-window7-224',
    language_model='microsoft/DialoGPT-medium',
    num_classes=14  # Make sure this matches your dataset
)
model.load_state_dict(torch.load('path/to/your/best_model.pth'))

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

The framework uses a configuration system based on dataclasses to manage model parameters, dataset settings, and training options. The core configurations are automatically handled when you run `main.py`.

If you need to customize settings like the batch size, learning rate, or model names, you can do so by modifying the `Config` and `DatasetConfig` classes in `EVLMs/configs/config.py`.

## Next Steps

-   Learn more about the [dataset format](datasets.md).
-   Understand the [model architecture](model_architecture.md).
-   Explore [training options](training.md).
-   Check out the [visualization tools](visualization.md).
 