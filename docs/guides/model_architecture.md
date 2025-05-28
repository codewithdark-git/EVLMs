# Model Architecture

This guide explains the architecture of the Explainable Vision-Language Models (EVLMs) framework, including its components, design choices, and how they work together.

## Overview

EVLMs combines vision and language models to create an explainable medical image analysis system. The architecture consists of four main components:

1. Vision Encoder
2. Language Decoder
3. Cross-Modal Attention
4. Explanation Generator

```
                                    ┌─────────────────┐
                                    │   Input Image   │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │ Vision Encoder  │
                                    │  (Swin-Base)    │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
┌─────────────────┐                │  Cross-Modal    │
│  Input Text/    │                │   Attention     │
│   Prompt        ├───────────────►│                 │
└─────────────────┘                └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │    Language     │
                                    │    Decoder      │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │    Outputs      │
                                    │ - Description   │
                                    │ - Labels        │
                                    │ - Explanations  │
                                    └─────────────────┘
```

## Components

### 1. Vision Encoder

The vision encoder uses the Swin Transformer architecture to process medical images:

```python
class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = 'microsoft/swin-base-patch4-window7-224'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, images):
        # Shape: [batch_size, num_patches, hidden_dim]
        features = self.model(images).last_hidden_state
        return features
```

Features:
- Hierarchical feature extraction
- Shifted window attention
- Relative position encoding
- Medical domain adaptation

### 2. Language Decoder

The language decoder is based on DialoGPT for generating natural language descriptions:

```python
class LanguageDecoder(nn.Module):
    def __init__(self, model_name: str = 'microsoft/DialoGPT-medium'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )
        return outputs
```

Features:
- Medical vocabulary integration
- Conditional text generation
- Attention-based decoding
- Medical knowledge incorporation

### 3. Cross-Modal Attention

The cross-modal attention mechanism connects vision and language:

```python
class CrossModalAttention(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = MultiheadAttention(
            vision_dim, text_dim, num_heads
        )
        
    def forward(self, vision_features, text_features, attention_mask):
        attended_features = self.attention(
            query=text_features,
            key=vision_features,
            value=vision_features,
            key_padding_mask=attention_mask
        )
        return attended_features
```

Features:
- Multi-head attention
- Feature alignment
- Modality fusion
- Attention visualization

### 4. Explanation Generator

The explanation generator provides visual and textual explanations:

```python
class ExplanationGenerator:
    def __init__(self, model):
        self.model = model
        self.grad_cam = GradCAM(model.vision_encoder)
        
    def explain(self, image, method="gradcam"):
        if method == "gradcam":
            return self.generate_gradcam(image)
        elif method == "attention":
            return self.generate_attention_map(image)
```

Features:
- GradCAM visualization
- Attention map generation
- Feature importance scoring
- Natural language explanations

## Model Integration

The components are integrated in the `ExplainableMedicalVLM` class:

```python
class ExplainableMedicalVLM(nn.Module):
    def __init__(
        self,
        vision_model: str,
        language_model: str,
        num_classes: int
    ):
        super().__init__()
        self.vision_encoder = VisionEncoder(vision_model)
        self.language_decoder = LanguageDecoder(language_model)
        self.cross_attention = CrossModalAttention(
            vision_dim=768,
            text_dim=768
        )
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, images, input_ids, attention_mask):
        # Vision encoding
        vision_features = self.vision_encoder(images)
        
        # Cross-modal attention
        attended_features = self.cross_attention(
            vision_features,
            self.language_decoder.get_embeddings(input_ids),
            attention_mask
        )
        
        # Language decoding
        text_outputs = self.language_decoder(
            input_ids,
            attention_mask,
            attended_features
        )
        
        # Classification
        classification = self.classifier(vision_features.mean(1))
        
        return {
            'text_outputs': text_outputs,
            'classification': classification,
            'attention_weights': self.cross_attention.weights
        }
```

## Training Process

The model is trained using multi-task learning:

1. **Classification Loss**: Binary cross-entropy for medical conditions
2. **Language Modeling Loss**: Teacher forcing for report generation
3. **Contrastive Loss**: Image-text alignment
4. **Explanation Loss**: Supervision for attention mechanisms

```python
def compute_loss(outputs, targets):
    losses = {
        'classification': F.binary_cross_entropy(
            outputs['classification'],
            targets['labels']
        ),
        'language': outputs['text_outputs'].loss,
        'contrastive': contrastive_loss(
            outputs['vision_features'],
            outputs['text_features']
        ),
        'explanation': attention_supervision_loss(
            outputs['attention_weights'],
            targets['attention_maps']
        )
    }
    return sum(losses.values())
```

## Inference Pipeline

During inference, the model:

1. Processes the input image
2. Generates textual description
3. Predicts medical conditions
4. Provides explanations

```python
def generate_report(image):
    # Encode image
    vision_features = model.vision_encoder(image)
    
    # Generate text
    generated = model.language_decoder.generate(
        encoder_hidden_states=vision_features
    )
    
    # Get explanations
    explanations = model.explain(image)
    
    return {
        'text': tokenizer.decode(generated[0]),
        'predictions': model.classify(vision_features),
        'explanations': explanations
    }
```

## Model Configuration

The model can be configured through the config system:

```python
config = Config(
    vision_model_name="microsoft/swin-base-patch4-window7-224",
    language_model_name="microsoft/DialoGPT-medium",
    num_classes=14,
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    num_hidden_layers=12
)
```

## Next Steps

- Learn about [training procedures](training.md)
- Explore [visualization tools](visualization.md)
- Check [example notebooks](../examples/README.md) 