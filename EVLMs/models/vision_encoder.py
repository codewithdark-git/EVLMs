import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional

class MedicalVisionEncoder(nn.Module):
    """Vision encoder for medical images"""
    
    def __init__(self,
                 model_name: str = 'microsoft/swin-base-patch4-window7-224',
                 img_size: int = 224,
                 pretrained: bool = True):
        """
        Args:
            model_name: Name of the base vision model
            img_size: Input image size
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pre-trained vision transformer
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Keep spatial dimensions
        )
        
        # Medical-specific adaptations
        self.medical_adapter = nn.Sequential(
            nn.Conv2d(1024, 768, 1),  # Adapt to CLIP dimensions
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Multi-scale feature extraction
        self.multiscale_conv = nn.ModuleList([
            nn.Conv2d(768, 768, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])
        
        # Spatial attention for localization
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, 
                x: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input images (B, C, H, W)
            return_attention: Whether to return attention weights
        
        Returns:
            features: Visual features (B, N, D)
            attention_weights: Optional attention weights (B, N, N)
        """
        # Extract backbone features
        features = self.backbone.forward_features(x)  # [B, 49, 1024]
        
        # Apply medical adapter
        features = features.permute(0, 3, 1, 2)
        features = self.medical_adapter(features)
        
        # Multi-scale processing
        multiscale_features = []
        for conv in self.multiscale_conv:
            multiscale_features.append(conv(features))
        
        features = torch.stack(multiscale_features).mean(dim=0)
        
        # Flatten for attention
        features = features.flatten(2).transpose(1, 2)  # [B, 49, 768]
        
        # Self-attention for spatial relationships
        attended_features, attention_weights = self.spatial_attention(
            features, features, features
        )
        
        if return_attention:
            return attended_features, attention_weights
        return attended_features, None
    
    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for each image patch
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            patch_features: Features for each patch (B, N, D)
        """
        features = self.backbone.forward_features(x)
        return features
    
    def extract_global_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global image features
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            global_features: Global image features (B, D)
        """
        features, _ = self.forward(x)
        return features.mean(dim=1)  # Global average pooling 
