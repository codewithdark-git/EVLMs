# Explainable Vision-Language Models (EVLMs) Documentation

Welcome to the EVLMs documentation! This documentation provides comprehensive information about the Explainable Vision-Language Models project, which focuses on creating interpretable medical image analysis models that can both understand and explain their decisions.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Getting Started](guides/getting_started.md)
3.  [Model Architecture](guides/model_architecture.md)
4.  [Datasets](guides/datasets.md)
5.  [Training](guides/training.md)
6.  [Visualization](guides/visualization.md)
7.  [API Reference](api/README.md)

## Introduction

EVLMs is a framework for training and deploying explainable vision-language models for medical image analysis. The project combines state-of-the-art vision transformers with advanced language models to create a system that can:

-   Analyze medical images (e.g., X-rays).
-   Generate natural language radiology reports.
-   Provide visual explanations for its decisions.
-   Support multi-label classification of medical conditions.

### Key Features

-   **Modular Architecture**: Easy to extend and customize components.
-   **Local Dataset Support**: Works with a local dataset structured in a specific JSON format.
-   **Explainability Tools**: Includes GradCAM, attention visualization, and more.
-   **Efficient Training**: Supports mixed precision and gradient accumulation.

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
    └── guides/      # User guides
```

## Quick Links

-   [Getting Started Guide](guides/getting_started.md)
-   [Dataset Guide](guides/datasets.md)
-   [Training Guide](guides/training.md)
-   [API Reference](api/README.md)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
 