# Explainable Vision-Language Models (EVLMs) Documentation

Welcome to the EVLMs documentation! This documentation provides comprehensive information about the Explainable Vision-Language Models project, which focuses on creating interpretable medical image analysis models that can both understand and explain their decisions.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](guides/getting_started.md)
3. [Model Architecture](guides/model_architecture.md)
4. [Datasets](guides/datasets.md)
5. [Training](guides/training.md)
6. [Visualization](guides/visualization.md)
7. [API Reference](api/README.md)
8. [Examples](examples/README.md)

## Introduction

EVLMs is a framework for training and deploying explainable vision-language models for medical image analysis. The project combines state-of-the-art vision transformers with advanced language models to create a system that can:

- Analyze medical images (X-rays, CT scans, etc.)
- Generate natural language descriptions
- Provide visual explanations for its decisions
- Support multiple medical conditions and modalities
- Handle various medical imaging datasets

### Key Features

- **Modular Architecture**: Easy to extend and customize components
- **Multiple Dataset Support**: Works with CheXpert, MIMIC-CXR, NIH Chest X-ray, and custom datasets
- **Explainability Tools**: Includes GradCAM, attention visualization, and more
- **Efficient Training**: Supports distributed training, mixed precision, and gradient accumulation
- **Comprehensive Metrics**: Tracks medical-specific metrics and general ML metrics
- **Easy Integration**: Uses HuggingFace datasets and models for simple setup

### Project Structure

```
EVLMs/
├── configs/          # Configuration files
├── data/            # Dataset handling
├── models/          # Model architectures
├── trainers/        # Training logic
├── utils/           # Utility functions
├── visualization/   # Visualization tools
└── docs/            # Documentation
    ├── api/         # API reference
    ├── guides/      # User guides
    └── examples/    # Example notebooks
```

## Quick Links

- [Getting Started Guide](guides/getting_started.md)
- [Dataset Guide](guides/datasets.md)
- [Training Guide](guides/training.md)
- [API Reference](api/README.md)
- [Example Notebooks](examples/README.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](guides/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details. 