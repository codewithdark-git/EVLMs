# Visualization and Explanation Tools

This guide covers the visualization and explanation tools available in the EVLMs framework for understanding model predictions and decisions.

## Overview

The framework provides several visualization methods:
1. GradCAM for visual explanations
2. Attention visualization
3. Feature importance maps
4. Natural language explanations

## GradCAM Visualization

GradCAM highlights regions in the image that influenced the model's decision:

```python
from EVLMs.visualization.explanation_generator import ExplanationGenerator

# Initialize explainer
explainer = ExplanationGenerator(model)

# Generate GradCAM visualization
explanation = explainer.explain(
    image_path="path/to/image.jpg",
    method="gradcam",
    target_class="Pneumonia"
)

# Display or save visualization
explanation.show()
explanation.save("explanation.png")
```

### Customizing GradCAM

```python
class GradCAMExplainer:
    def __init__(
        self,
        model,
        target_layer=None,
        use_relu=True,
        colormap=cv2.COLORMAP_JET
    ):
        self.model = model
        self.target_layer = target_layer or model.vision_encoder.layers[-1]
        self.use_relu = use_relu
        self.colormap = colormap
    
    def generate(self, image, class_idx):
        # Get gradients and activations
        gradients, activations = self._get_gradients_and_activations(
            image, class_idx
        )
        
        # Weight activations by gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = (weights * activations).sum(dim=1)
        
        # Apply ReLU if specified
        if self.use_relu:
            weighted_activations = F.relu(weighted_activations)
            
        return self._create_heatmap(weighted_activations)
```

## Attention Visualization

Visualize cross-modal attention patterns:

```python
def visualize_attention(
    self,
    image: torch.Tensor,
    text: str,
    layer_idx: int = -1
) -> plt.Figure:
    """Visualize attention between image and text."""
    
    # Get attention weights
    attention_weights = self.model.get_attention_weights(
        image, text, layer_idx
    )
    
    # Create attention map
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Plot attention heatmap
    sns.heatmap(
        attention_weights.cpu().numpy(),
        xticklabels=text.split(),
        yticklabels=False,
        cmap='viridis'
    )
    
    return fig
```

### Multi-head Attention Visualization

```python
def visualize_multihead_attention(
    self,
    image: torch.Tensor,
    text: str,
    num_heads: int = 8
) -> plt.Figure:
    """Visualize attention patterns for each attention head."""
    
    # Get attention weights for all heads
    attention_weights = self.model.get_multihead_attention(image, text)
    
    # Create subplot for each head
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, (ax, weights) in enumerate(zip(axes, attention_weights)):
        sns.heatmap(
            weights.cpu().numpy(),
            ax=ax,
            xticklabels=text.split(),
            yticklabels=False,
            cmap='viridis'
        )
        ax.set_title(f'Head {idx+1}')
    
    return fig
```

## Feature Importance Maps

Generate feature importance visualizations:

```python
class FeatureImportanceVisualizer:
    def __init__(self, model):
        self.model = model
        
    def generate_importance_map(
        self,
        image: torch.Tensor,
        method: str = "integrated_gradients",
        steps: int = 50
    ) -> torch.Tensor:
        """Generate feature importance map."""
        
        if method == "integrated_gradients":
            return self._integrated_gradients(image, steps)
        elif method == "smooth_grad":
            return self._smooth_grad(image)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _integrated_gradients(
        self,
        image: torch.Tensor,
        steps: int
    ) -> torch.Tensor:
        """Compute Integrated Gradients attribution."""
        baseline = torch.zeros_like(image)
        alphas = torch.linspace(0, 1, steps)
        
        gradient_sum = torch.zeros_like(image)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad_()
            
            outputs = self.model(interpolated)
            gradients = torch.autograd.grad(
                outputs,
                interpolated
            )[0]
            
            gradient_sum += gradients
            
        attribution = (image - baseline) * gradient_sum / steps
        return attribution
```

## Natural Language Explanations

Generate textual explanations for model decisions:

```python
class NaturalLanguageExplainer:
    def __init__(self, model):
        self.model = model
        
    def generate_explanation(
        self,
        image: torch.Tensor,
        prediction: str
    ) -> str:
        """Generate natural language explanation."""
        
        # Get model features
        features = self.model.extract_features(image)
        
        # Generate explanation using language model
        explanation = self.model.language_decoder.generate(
            encoder_hidden_states=features,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        
        return self.tokenizer.decode(explanation[0])
```

## Integrated Visualization

Combine multiple visualization methods:

```python
class IntegratedVisualizer:
    def __init__(self, model):
        self.model = model
        self.grad_cam = GradCAMExplainer(model)
        self.attention_vis = AttentionVisualizer(model)
        self.feature_vis = FeatureImportanceVisualizer(model)
        self.text_explainer = NaturalLanguageExplainer(model)
        
    def generate_complete_explanation(
        self,
        image_path: str,
        save_dir: str = None
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation."""
        
        # Load and preprocess image
        image = self.load_image(image_path)
        
        # Get model prediction
        prediction = self.model.predict(image)
        
        # Generate various explanations
        explanations = {
            'gradcam': self.grad_cam.generate(image),
            'attention': self.attention_vis.visualize_attention(image),
            'feature_importance': self.feature_vis.generate_importance_map(image),
            'text_explanation': self.text_explainer.generate_explanation(
                image, prediction
            )
        }
        
        # Save visualizations if directory provided
        if save_dir:
            self.save_explanations(explanations, save_dir)
            
        return explanations
```

## Visualization Utilities

Helper functions for visualization:

```python
def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """Overlay heatmap on image."""
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )
    
    # Apply colormap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    
    # Overlay
    return cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

def plot_attention_graph(
    attention_weights: torch.Tensor,
    words: List[str],
    threshold: float = 0.1
) -> plt.Figure:
    """Plot attention graph."""
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, word in enumerate(words):
        G.add_node(f"word_{i}", label=word)
    
    # Add edges based on attention weights
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            weight = attention_weights[i, j].item()
            if weight > threshold:
                G.add_edge(f"word_{i}", f"word_{j}", weight=weight)
    
    return nx.draw(
        G,
        with_labels=True,
        node_color='lightblue',
        node_size=1000,
        font_size=10
    )
```

## Best Practices

1. **Visualization Quality**:
   - Use appropriate color maps
   - Ensure proper scaling
   - Add legends and labels
   - Maintain aspect ratios

2. **Performance**:
   - Cache intermediate results
   - Use batch processing
   - Optimize image loading
   - Handle memory efficiently

3. **User Experience**:
   - Provide interactive controls
   - Add zoom capabilities
   - Include save options
   - Support multiple formats

4. **Interpretation**:
   - Add confidence scores
   - Show multiple views
   - Provide context
   - Include baselines

## Next Steps

- Check [example notebooks](../examples/README.md)
- Learn about [model deployment](deployment.md)
- Explore [API reference](../api/README.md) 