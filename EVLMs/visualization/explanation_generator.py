import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, GradCAM, LayerGradCam
import cv2
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ExplanationGenerator:
    """Generate visual and textual explanations"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Model to explain
            device: Device to run on
        """
        self.model = model
        self.device = device
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.gradcam = GradCAM(
            self.model,
            self.model.vision_encoder.backbone.layers[-1]
        )
    
    def generate_visual_explanation(self,
                                  image: torch.Tensor,
                                  target_class: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Generate visual saliency maps
        
        Args:
            image: Input image
            target_class: Target class for attribution
        
        Returns:
            Dictionary containing attributions
        """
        self.model.eval()
        
        # Integrated Gradients
        ig_attr = self.integrated_gradients.attribute(
            image.unsqueeze(0),
            target=target_class,
            n_steps=50
        )
        
        # GradCAM
        gradcam_attr = self.gradcam.attribute(
            image.unsqueeze(0),
            target=target_class
        )
        
        return {
            'integrated_gradients': ig_attr.squeeze().cpu(),
            'gradcam': gradcam_attr.squeeze().cpu(),
            'input_image': image.cpu()
        }
    
    def generate_attention_explanation(self,
                                     image: torch.Tensor,
                                     text_input: str) -> Dict[str, torch.Tensor]:
        """Generate attention-based explanations
        
        Args:
            image: Input image
            text_input: Input text
        
        Returns:
            Dictionary containing attention weights
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get attention weights from cross-modal attention
            visual_features = self.model.vision_encoder(image.unsqueeze(0))
            
            # Tokenize text
            text_tokens = self.model.language_decoder.tokenizer.encode(
                text_input, return_tensors='pt'
            ).to(self.device)
            text_features = self.model.language_decoder.language_model.get_input_embeddings()(
                text_tokens
            )
            
            # Get cross-attention weights
            cross_modal_outputs = self.model.cross_modal_attention(
                visual_features, text_features
            )
            
            attention_weights = cross_modal_outputs['cross_attention_weights']
        
        return {
            'visual_to_text': attention_weights['visual_to_text'].squeeze().cpu(),
            'text_to_visual': attention_weights['text_to_visual'].squeeze().cpu()
        }
    
    def visualize_explanation(self,
                            explanations: Dict[str, torch.Tensor],
                            save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of explanations
        
        Args:
            explanations: Dictionary of explanations
            save_path: Path to save visualization
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(explanations['input_image'].permute(1, 2, 0), cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Integrated Gradients
        ig_attr = explanations['integrated_gradients']
        axes[0, 1].imshow(ig_attr, cmap='RdBu_r')
        axes[0, 1].set_title('Integrated Gradients')
        axes[0, 1].axis('off')
        
        # GradCAM
        gradcam_attr = explanations['gradcam']
        axes[0, 2].imshow(gradcam_attr, cmap='jet', alpha=0.7)
        axes[0, 2].imshow(
            explanations['input_image'].squeeze(),
            cmap='gray',
            alpha=0.3
        )
        axes[0, 2].set_title('GradCAM Overlay')
        axes[0, 2].axis('off')
        
        # Attention visualizations
        if 'visual_to_text' in explanations:
            visual_att = explanations['visual_to_text']
            axes[1, 0].imshow(visual_att.mean(dim=0), cmap='Blues')
            axes[1, 0].set_title('Visual-to-Text Attention')
            axes[1, 0].axis('off')
        
        if 'text_to_visual' in explanations:
            text_att = explanations['text_to_visual']
            axes[1, 1].imshow(text_att.mean(dim=0), cmap='Reds')
            axes[1, 1].set_title('Text-to-Visual Attention')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def generate_counterfactual(self,
                               image: torch.Tensor,
                               original_prediction: torch.Tensor,
                               target_change: int,
                               num_samples: int = 10) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate counterfactual examples
        
        Args:
            image: Input image
            original_prediction: Original model prediction
            target_change: Desired change in prediction
            num_samples: Number of samples to generate
        
        Returns:
            Tuple of (perturbations, predictions)
        """
        perturbations = []
        predictions = []
        
        # Create multiple perturbations
        for i in range(num_samples):
            # Random perturbation
            noise = torch.randn_like(image) * 0.1
            perturbed_image = image + noise
            
            # Get prediction
            with torch.no_grad():
                pred = self.model(perturbed_image.unsqueeze(0), mode='classification')
                pred = torch.sigmoid(pred['logits'])
            
            perturbations.append(perturbed_image)
            predictions.append(pred)
        
        return perturbations, predictions 