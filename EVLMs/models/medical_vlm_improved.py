import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
from collections import OrderedDict
import warnings

from .vision_encoder import MedicalVisionEncoder
from .language_decoder import MedicalLanguageDecoder
from .cross_attention import CrossModalAttention, ContrastiveLearning

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    vision_model: str = 'microsoft/swin-base-patch4-window7-224'
    language_model: str = 'microsoft/DialoGPT-medium'
    num_classes: int = 14
    hidden_dim: int = 512
    dropout_rate: float = 0.2
    use_gradient_checkpointing: bool = False
    enable_uncertainty: bool = True
    adaptive_loss_weights: bool = True

class AdaptiveLossWeighting(nn.Module):
    """Adaptive loss weighting using uncertainty-based approach"""
    
    def __init__(self, num_losses: int = 3):
        super().__init__()
        # Learnable log variance parameters
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss using uncertainty weighting
        
        Args:
            losses: Dictionary of individual losses
            
        Returns:
            weighted_loss: Combined weighted loss
            weights: Current loss weights
        """
        loss_values = list(losses.values())
        loss_names = list(losses.keys())
        
        weighted_losses = []
        weights = {}
        
        for i, (name, loss) in enumerate(zip(loss_names, loss_values)):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
            weights[f'{name}_weight'] = precision.item()
        
        total_loss = sum(weighted_losses)
        return total_loss, weights

class FeatureCache:
    """Cache for storing computed features to avoid redundant computation"""
    
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: torch.Tensor):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
        self.cache[key] = value.detach().clone()
    
    def clear(self):
        self.cache.clear()

class ExplainabilityModule(nn.Module):
    """Enhanced explainability methods"""
    
    def __init__(self, model_ref):
        super().__init__()
        self.model = model_ref
    
    def gradcam(self, image: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """Generate GradCAM visualization"""
        self.model.eval()
        image.requires_grad_(True)
        
        # Forward pass
        visual_features, attention_weights = self.model.vision_encoder(
            image, return_attention=True
        )
        
        # Get classification output
        visual_global = visual_features.mean(dim=1)
        logits = self.model.classifier(visual_global)
        
        if target_class is None:
            target_class = logits.argmax(dim=1)
        
        # Backward pass
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # Get gradients and features
        gradients = image.grad
        
        # Compute GradCAM
        weights = F.adaptive_avg_pool2d(gradients, 1)
        cam = torch.sum(weights * image, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)
        
        return cam
    
    def integrated_gradients(self, image: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Generate Integrated Gradients explanation"""
        baseline = torch.zeros_like(image)
        
        # Generate interpolated images
        alphas = torch.linspace(0, 1, steps).to(image.device)
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            visual_features, _ = self.model.vision_encoder(interpolated, return_attention=True)
            visual_global = visual_features.mean(dim=1)
            logits = self.model.classifier(visual_global)
            
            # Backward pass
            score = logits.max()
            score.backward(retain_graph=True)
            
            gradients.append(interpolated.grad.clone())
            interpolated.grad.zero_()
        
        # Compute integrated gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_grads = (image - baseline) * avg_gradients
        
        return integrated_grads
    
    def uncertainty_estimation(self, image: torch.Tensor, num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """Estimate prediction uncertainty using Monte Carlo Dropout"""
        self.model.train()  # Enable dropout
        
        predictions = []
        explanations = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.model.forward(image, mode='classification')
                predictions.append(torch.sigmoid(output['logits']))
        
        predictions = torch.stack(predictions)
        
        # Calculate uncertainty metrics
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)
        
        return {
            'mean_prediction': mean_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'prediction_std': predictions.std(dim=0)
        }

class ImprovedExplainableMedicalVLM(nn.Module):
    """Improved explainable medical vision-language model with enhanced features"""
    
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        
        self.config = config or ModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Feature cache for efficiency
        self.feature_cache = FeatureCache()
        
        # Initialize components with error handling
        try:
            self.vision_encoder = MedicalVisionEncoder(self.config.vision_model)
            self.language_decoder = MedicalLanguageDecoder(self.config.language_model)
        except Exception as e:
            self.logger.error(f"Failed to initialize encoders: {e}")
            raise
        
        # Dynamic feature dimensions
        self.vision_feature_dim = self._get_vision_feature_dim()
        self.language_feature_dim = self._get_language_feature_dim()
        
        # Cross-modal components
        self.cross_modal_attention = CrossModalAttention(
            visual_dim=self.vision_feature_dim,
            text_dim=self.language_feature_dim,
            hidden_dim=self.config.hidden_dim
        )
        self.contrastive_learning = ContrastiveLearning()
        
        # Projections with proper initialization
        self.vision_contrastive_proj = self._create_projection(
            self.vision_feature_dim, self.config.hidden_dim
        )
        self.text_contrastive_proj = self._create_projection(
            self.language_feature_dim, self.config.hidden_dim
        )
        
        # Enhanced classifier with uncertainty
        self.classifier = self._create_classifier()
        
        # Adaptive loss weighting
        if self.config.adaptive_loss_weights:
            self.adaptive_loss = AdaptiveLossWeighting(num_losses=3)
        
        # Explainability module
        self.explainer = ExplainabilityModule(self)
        
        # Enable gradient checkpointing if requested
        if self.config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _get_vision_feature_dim(self) -> int:
        """Dynamically determine vision feature dimension"""
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features, _ = self.vision_encoder(dummy_input, return_attention=True)
            return features.shape[-1]
        except Exception as e:
            self.logger.warning(f"Could not determine vision feature dim: {e}, using default 768")
            return 768
    
    def _get_language_feature_dim(self) -> int:
        """Dynamically determine language feature dimension"""
        try:
            return self.language_decoder.language_model.config.hidden_size
        except Exception as e:
            self.logger.warning(f"Could not determine language feature dim: {e}, using default 768")
            return 768
    
    def _create_projection(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create projection layer with proper initialization"""
        projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate)
        )
        
        # Xavier initialization
        for module in projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        return projection
    
    def _create_classifier(self) -> nn.Module:
        """Create enhanced classifier with uncertainty estimation"""
        layers = [
            nn.Linear(self.vision_feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, self.config.num_classes)
        ]
        
        classifier = nn.Sequential(*layers)
        
        # Initialize weights
        for module in classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        return classifier
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
            self.vision_encoder.gradient_checkpointing_enable()
        if hasattr(self.language_decoder, 'gradient_checkpointing_enable'):
            self.language_decoder.gradient_checkpointing_enable()
    
    def _validate_inputs(self, images: torch.Tensor, **kwargs):
        """Validate input tensors"""
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(images)}")
        
        if images.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got {images.dim()}D")
        
        if images.shape[1] not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {images.shape[1]}")
        
        # Check device compatibility
        if not images.is_cuda and next(self.parameters()).is_cuda:
            warnings.warn("Input tensor is on CPU but model is on CUDA")
    
    def _get_cached_features(self, images: torch.Tensor, cache_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached visual features if available"""
        if self.training:
            return None  # Don't use cache during training
        
        cached = self.feature_cache.get(cache_key)
        if cached is not None:
            return cached['features'], cached['attention']
        return None
    
    def _cache_features(self, cache_key: str, features: torch.Tensor, attention: torch.Tensor):
        """Cache visual features for reuse"""
        if not self.training:
            self.feature_cache.set(cache_key, {
                'features': features,
                'attention': attention
            })
    
    def forward(self,
                images: torch.Tensor,
                text_input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                mode: str = 'train',
                return_features: bool = False) -> Dict[str, Any]:
        """
        Enhanced forward pass with caching and error handling
        
        Args:
            images: Input images (B, C, H, W)
            text_input_ids: Input text token IDs
            attention_mask: Attention mask for text
            labels: Labels for classification
            mode: Running mode ('train', 'classification', 'explanation', 'full')
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing model outputs
        """
        try:
            # Input validation
            self._validate_inputs(images)
            batch_size = images.shape[0]
            
            # Create cache key for feature reuse
            cache_key = f"{images.shape}_{images.sum().item():.6f}"
            
            # Try to get cached features
            cached_features = self._get_cached_features(images, cache_key)
            
            if cached_features is not None:
                visual_features, visual_attention = cached_features
            else:
                # Vision encoding
                visual_features, visual_attention = self.vision_encoder(
                    images, return_attention=True
                )
                # Cache features for reuse
                self._cache_features(cache_key, visual_features, visual_attention)
            
            # Global visual representation
            visual_global = visual_features.mean(dim=1)  # [B, vision_feature_dim]
            
            outputs = {'visual_features': visual_features, 'visual_attention': visual_attention}
            
            if mode == 'classification':
                # Classification only
                logits = self.classifier(visual_global)
                outputs.update({
                    'logits': logits,
                    'predictions': torch.sigmoid(logits)
                })
                
                if self.config.enable_uncertainty:
                    uncertainty = self.explainer.uncertainty_estimation(images)
                    outputs.update(uncertainty)
            
            elif mode == 'explanation':
                # Generate explanation
                explanations = self.language_decoder.generate_explanation(visual_features)
                outputs.update({
                    'explanations': explanations['explanations'],
                    'visual_attention': visual_attention
                })
            
            elif mode == 'full':
                # Complete inference with all outputs
                logits = self.classifier(visual_global)
                explanations = self.language_decoder.generate_explanation(visual_features)
                
                outputs.update({
                    'logits': logits,
                    'predictions': torch.sigmoid(logits),
                    'explanations': explanations['explanations']
                })
                
                if self.config.enable_uncertainty:
                    uncertainty = self.explainer.uncertainty_estimation(images)
                    outputs.update(uncertainty)
            
            elif mode == 'train':
                # Multi-task training with enhanced loss computation
                
                # 1. Classification
                classification_logits = self.classifier(visual_global)
                if labels is not None:
                    classification_loss = F.binary_cross_entropy_with_logits(
                        classification_logits, labels.float()
                    )
                else:
                    classification_loss = torch.tensor(0.0, device=images.device)
                
                # 2. Language generation
                if text_input_ids is not None:
                    language_outputs = self.language_decoder(
                        visual_features,
                        text_input_ids=text_input_ids,
                        attention_mask=attention_mask,
                        labels=text_input_ids
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
                else:
                    language_loss = torch.tensor(0.0, device=images.device)
                    contrastive_loss = torch.tensor(0.0, device=images.device)
                    cross_modal_outputs = {'cross_attention_weights': None}
                
                # Adaptive loss combination
                losses = {
                    'classification': classification_loss,
                    'language': language_loss,
                    'contrastive': contrastive_loss
                }
                
                if self.config.adaptive_loss_weights and hasattr(self, 'adaptive_loss'):
                    total_loss, loss_weights = self.adaptive_loss(losses)
                    outputs.update(loss_weights)
                else:
                    # Fixed weighting as fallback
                    total_loss = classification_loss + language_loss + 0.1 * contrastive_loss
                
                outputs.update({
                    'loss': total_loss,
                    'classification_loss': classification_loss,
                    'language_loss': language_loss,
                    'contrastive_loss': contrastive_loss,
                    'logits': classification_logits,
                    'cross_attention_weights': cross_modal_outputs.get('cross_attention_weights')
                })
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            if return_features:
                outputs['intermediate_features'] = {
                    'visual_global': visual_global,
                    'visual_features': visual_features
                }
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            # Return minimal output to prevent training crash
            return {
                'loss': torch.tensor(float('inf'), device=images.device),
                'error': str(e)
            }
    
    def explain(self, image: torch.Tensor, method: str = 'attention') -> Dict[str, Any]:
        """
        Generate comprehensive explanation for an image
        
        Args:
            image: Input image (1, C, H, W)
            method: Explanation method ('attention', 'gradcam', 'integrated_gradients', 'all')
        
        Returns:
            Dictionary containing explanations and visualizations
        """
        self.eval()
        
        try:
            with torch.no_grad():
                # Get full model outputs efficiently
                outputs = self.forward(image, mode='full')
            
            explanations = {
                'prediction': outputs['predictions'],
                'explanation_text': outputs['explanations'][0] if outputs['explanations'] else "",
                'visual_attention': outputs['visual_attention']
            }
            
            # Add specific explanation methods
            if method in ['gradcam', 'all']:
                explanations['gradcam'] = self.explainer.gradcam(image)
            
            if method in ['integrated_gradients', 'all']:
                explanations['integrated_gradients'] = self.explainer.integrated_gradients(image)
            
            if method in ['uncertainty', 'all'] and self.config.enable_uncertainty:
                uncertainty = self.explainer.uncertainty_estimation(image)
                explanations.update(uncertainty)
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vision_feature_dim': self.vision_feature_dim,
            'language_feature_dim': self.language_feature_dim,
            'config': self.config,
            'cache_size': len(self.feature_cache.cache)
        }