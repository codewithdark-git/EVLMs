import torch
import torch.nn as nn
from typing import Dict, Tuple

class CrossModalAttention(nn.Module):
    """Cross-attention between visual and textual features"""
    
    def __init__(self,
                 visual_dim: int = 768,
                 text_dim: int = 768,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            visual_dim: Dimension of visual features
            text_dim: Dimension of text features
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Project to common dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projections
        self.visual_output = nn.Linear(hidden_dim, visual_dim)
        self.text_output = nn.Linear(hidden_dim, text_dim)
        
        # Layer normalization
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        
        # Gating mechanism
        self.visual_gate = nn.Sequential(
            nn.Linear(visual_dim + hidden_dim, visual_dim),
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim + hidden_dim, text_dim),
            nn.Sigmoid()
        )
    
    def forward(self,
                visual_features: torch.Tensor,
                text_features: torch.Tensor,
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: Visual features (B, N_v, D_v)
            text_features: Text features (B, N_t, D_t)
            attention_mask: Attention mask for text features
        
        Returns:
            Dictionary containing:
                visual_output: Updated visual features
                text_output: Updated text features
                cross_attention_weights: Cross-attention weights
        """
        batch_size = visual_features.shape[0]
        
        # Project to common space
        visual_proj = self.visual_proj(visual_features)
        text_proj = self.text_proj(text_features)
        
        # Visual attending to text
        visual_attended, visual_attention = self.cross_attention(
            query=visual_proj,
            key=text_proj,
            value=text_proj,
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        
        # Text attending to visual
        text_attended, text_attention = self.cross_attention(
            query=text_proj,
            key=visual_proj,
            value=visual_proj
        )
        
        # Gating mechanism
        visual_gate = self.visual_gate(
            torch.cat([visual_features, visual_attended], dim=-1)
        )
        text_gate = self.text_gate(
            torch.cat([text_features, text_attended], dim=-1)
        )
        
        # Project back and apply gating
        visual_output = self.visual_norm(
            visual_features + visual_gate * self.visual_output(visual_attended)
        )
        text_output = self.text_norm(
            text_features + text_gate * self.text_output(text_attended)
        )
        
        return {
            'visual_output': visual_output,
            'text_output': text_output,
            'cross_attention_weights': {
                'visual_to_text': visual_attention,
                'text_to_visual': text_attention
            }
        }

class ContrastiveLearning(nn.Module):
    """Contrastive learning for vision-text alignment"""
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for similarity scaling
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self,
                image_features: torch.Tensor,
                text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: Image features (B, D)
            text_features: Text features (B, D)
        
        Returns:
            loss: Contrastive loss
            similarity: Similarity matrix
        """
        # Normalize features
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Labels for contrastive loss (diagonal is positive pairs)
        labels = torch.arange(image_features.shape[0], device=image_features.device)
        
        # Symmetric loss
        loss_i2t = nn.functional.cross_entropy(similarity, labels)
        loss_t2i = nn.functional.cross_entropy(similarity.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss, similarity 