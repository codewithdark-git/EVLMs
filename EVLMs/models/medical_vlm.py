import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .vision_encoder import MedicalVisionEncoder
from .language_decoder import MedicalLanguageDecoder
from .cross_attention import CrossModalAttention, ContrastiveLearning

class ExplainableMedicalVLM(nn.Module):
    """Complete explainable medical vision-language model"""
    
    def __init__(self,
                 vision_model: str = 'microsoft/swin-base-patch4-window7-224',
                 language_model: str = 'microsoft/DialoGPT-medium',
                 num_classes: int = 14,
                 hidden_dim: int = 512):
        """
        Args:
            vision_model: Name of vision model
            language_model: Name of language model
            num_classes: Number of medical conditions to classify
            hidden_dim: Hidden dimension for cross-attention
        """
        super().__init__()
        
        # Initialize components
        self.vision_encoder = MedicalVisionEncoder(vision_model)
        self.language_decoder = MedicalLanguageDecoder(language_model)
        
        vision_feature_dim = 768
        language_feature_dim = self.language_decoder.language_model.config.n_embd

        self.cross_modal_attention = CrossModalAttention(
            visual_dim=vision_feature_dim,
            text_dim=language_feature_dim,
            hidden_dim=hidden_dim
        )
        self.contrastive_learning = ContrastiveLearning()
        
        # Projections for contrastive learning
        self.vision_contrastive_proj = nn.Linear(vision_feature_dim, hidden_dim)
        self.text_contrastive_proj = nn.Linear(language_feature_dim, hidden_dim)

        # Classification head for multi-task learning
        self.classifier = nn.Sequential(
            nn.Linear(vision_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self,
                images: torch.Tensor,
                text_input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                mode: str = 'train') -> Dict[str, Any]:
        """
        Args:
            images: Input images (B, C, H, W)
            text_input_ids: Input text token IDs
            attention_mask: Attention mask for text
            labels: Labels for classification
            mode: Running mode ('train', 'classification', or 'explanation')
        
        Returns:
            Dictionary containing model outputs
        """
        batch_size = images.shape[0]
        
        # Vision encoding
        visual_features, visual_attention = self.vision_encoder(
            images, return_attention=True
        )
        
        # Global visual representation for classification
        visual_global = visual_features.mean(dim=1)  # [B, 768]
        
        if mode == 'classification':
            # Classification only
            logits = self.classifier(visual_global)
            return {'logits': logits}
        
        elif mode == 'explanation':
            # Generate explanation
            explanations = self.language_decoder.generate_explanation(visual_features)
            
            return {
                'explanations': explanations['explanations'],
                'visual_attention': visual_attention
            }
        
        elif mode == 'train':
            # Multi-task training
            
            # 1. Classification loss
            classification_logits = self.classifier(visual_global)
            classification_loss = nn.functional.binary_cross_entropy_with_logits(
                classification_logits, labels
            )
            
            # 2. Language generation loss
            language_outputs = self.language_decoder(
                visual_features,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                labels=text_input_ids  # Self-supervised
            )
            language_loss = language_outputs['loss']
            
            # 3. Cross-modal attention
            cross_modal_outputs = self.cross_modal_attention(
                visual_features=visual_features,
                text_features=language_outputs['hidden_states'][-1],
                attention_mask=attention_mask
            )
            
            # 4. Contrastive loss
            visual_for_contrastive = self.vision_contrastive_proj(visual_global)
            text_for_contrastive = self.text_contrastive_proj(
                cross_modal_outputs['text_output'].mean(dim=1)
            )
            contrastive_loss, similarity = self.contrastive_learning(
                visual_for_contrastive,
                text_for_contrastive
            )
            
            # Combined loss
            total_loss = (
                classification_loss +
                language_loss +
                0.1 * contrastive_loss  # Scale contrastive loss
            )
            
            return {
                'loss': total_loss,
                'classification_loss': classification_loss,
                'language_loss': language_loss,
                'contrastive_loss': contrastive_loss,
                'logits': classification_logits,
                'cross_attention_weights': cross_modal_outputs['cross_attention_weights']
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def explain(self, image: torch.Tensor) -> Dict[str, Any]:
        """Generate comprehensive explanation for an image
        
        Args:
            image: Input image (1, C, H, W)
        
        Returns:
            Dictionary containing explanation and visualizations
        """
        self.eval()
        with torch.no_grad():
            # Get model outputs
            classification_output = self.forward(
                image, mode='classification'
            )
            explanation_output = self.forward(
                image, mode='explanation'
            )
            
            # Get attention maps
            visual_features, visual_attention = self.vision_encoder(
                image, return_attention=True
            )
            
            # Combine outputs
            return {
                'prediction': torch.sigmoid(classification_output['logits']),
                'explanation': explanation_output['explanations'][0],
                'visual_attention': visual_attention,
                'cross_attention': explanation_output.get('cross_attention_weights', None)
            }