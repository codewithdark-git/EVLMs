# Completing the Dataset class and Training Pipeline



import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import os
import torch.nn as nn
import timm
from transformers import AutoModel
import torchvision.transforms as transforms

class MedicalVisionEncoder(nn.Module):
    def __init__(self, 
                 model_name='microsoft/swin-base-patch4-window7-224',
                 img_size=224,
                 num_classes=14,  # Common chest X-ray pathologies
                 pretrained=True):
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
            nn.LayerNorm(768),
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
        
    def forward(self, x, return_attention=False):
        # Extract backbone features
        features = self.backbone.forward_features(x)  # [B, 49, 1024] for 224x224
        
        # Apply medical adapter
        B, N, C = features.shape
        H = W = int(N**0.5)  # Assuming square patches
        features = features.transpose(1, 2).reshape(B, C, H, W)
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
        return attended_features

# Preprocessing pipeline for medical images
class MedicalImagePreprocessor:
    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # CLAHE for contrast enhancement
            self.apply_clahe,
            # Normalization for medical images
            transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale
        ])
    
    @staticmethod
    def apply_clahe(img):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        import cv2
        if len(img.shape) == 3:
            img = img.squeeze(0)
        img_np = (img.numpy() * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_np)
        return torch.from_numpy(enhanced).float() / 255.0
    
    def __call__(self, image):
        return self.transform(image)


### Advanced Vision Processing

import pydicom
import nibabel as nib
import SimpleITK as sitk

class DICOMProcessor:
    """Handle DICOM medical image processing"""
    
    @staticmethod
    def load_dicom(file_path):
        """Load and preprocess DICOM files"""
        ds = pydicom.dcmread(file_path)
        image = ds.pixel_array
        
        # Apply window/level settings if available
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            center = ds.WindowCenter
            width = ds.WindowWidth
            if isinstance(center, pydicom.multival.MultiValue):
                center = center[0]
            if isinstance(width, pydicom.multival.MultiValue):
                width = width[0]
            
            # Apply windowing
            image = np.clip(image, center - width//2, center + width//2)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        return image, ds
    
    @staticmethod
    def extract_metadata(ds):
        """Extract relevant DICOM metadata"""
        metadata = {
            'patient_age': getattr(ds, 'PatientAge', None),
            'patient_sex': getattr(ds, 'PatientSex', None),
            'modality': getattr(ds, 'Modality', None),
            'body_part': getattr(ds, 'BodyPartExamined', None),
            'view_position': getattr(ds, 'ViewPosition', None),
            'study_description': getattr(ds, 'StudyDescription', None)
        }
        return metadata

## 2. Language Decoder with Medical Knowledge

### Clinical Language Model

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model

class MedicalLanguageDecoder(nn.Module):
    def __init__(self, 
                 model_name='microsoft/DialoGPT-medium',
                 vocab_size=50257,
                 max_length=512):
        super().__init__()
        
        # Load pre-trained language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Vision-to-text projection
        self.vision_projection = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, self.language_model.config.n_embd)
        )
        
        # Medical knowledge embeddings
        self.medical_embeddings = self.load_medical_embeddings()
        
        # LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.language_model = get_peft_model(self.language_model, lora_config)
    
    def load_medical_embeddings(self):
        """Load pre-trained medical concept embeddings"""
        # This would load embeddings from medical knowledge bases
        # For now, we'll create a placeholder
        vocab_size = 10000  # Medical vocabulary size
        embed_dim = 768
        return nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, 
                visual_features,
                text_input_ids=None,
                attention_mask=None,
                labels=None):
        
        batch_size = visual_features.shape[0]
        
        # Project visual features to language space
        visual_tokens = self.vision_projection(visual_features)  # [B, 49, n_embd]
        
        if text_input_ids is not None:
            # Training mode - teacher forcing
            text_embeddings = self.language_model.get_input_embeddings()(text_input_ids)
            
            # Concatenate visual and text embeddings
            combined_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)
            
            # Create attention mask for combined input
            visual_attention = torch.ones(
                batch_size, visual_tokens.shape[1], 
                device=visual_tokens.device
            )
            combined_attention = torch.cat([visual_attention, attention_mask], dim=1)
            
            # Forward pass through language model
            outputs = self.language_model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention,
                labels=labels
            )
            
            return outputs
        else:
            # Inference mode - autoregressive generation
            return self.generate_explanation(visual_tokens)
    
    def generate_explanation(self, visual_tokens, max_length=200):
        """Generate clinical explanation from visual features"""
        batch_size = visual_tokens.shape[0]
        device = visual_tokens.device
        
        # Start with visual tokens
        current_embeddings = visual_tokens
        generated_ids = []
        
        for _ in range(max_length):
            # Forward pass
            outputs = self.language_model(inputs_embeds=current_embeddings)
            logits = outputs.logits
            
            # Get next token (greedy decoding for now)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids.append(next_token_id)
            
            # Check for end token
            if torch.all(next_token_id == self.tokenizer.eos_token_id):
                break
            
            # Get embedding for next token
            next_embedding = self.language_model.get_input_embeddings()(
                next_token_id.unsqueeze(1)
            )
            
            # Append to current embeddings
            current_embeddings = torch.cat([current_embeddings, next_embedding], dim=1)
        
        # Convert to text
        if generated_ids:
            generated_ids = torch.stack(generated_ids, dim=1)
            generated_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            generated_text = [""] * batch_size
        
        return generated_text

### Medical Knowledge Integration

import owlready2 as owl
import networkx as nx
from rdflib import Graph, Namespace

class MedicalKnowledgeGraph:
    """Integration with medical ontologies"""
    
    def __init__(self):
        self.snomed_graph = self.load_snomed_ct()
        self.radlex_terms = self.load_radlex()
        self.concept_embeddings = {}
    
    def load_snomed_ct(self):
        """Load SNOMED CT ontology (requires SNOMED CT files)"""
        # This would load actual SNOMED CT data
        # For demo purposes, we'll create a simple graph
        G = nx.DiGraph()
        
        # Example medical concepts
        concepts = {
            "pneumonia": "233604007",
            "consolidation": "196618008", 
            "pleural_effusion": "60046008",
            "cardiomegaly": "80891009",
            "atelectasis": "196610009"
        }
        
        for concept, code in concepts.items():
            G.add_node(code, label=concept)
        
        # Add relationships
        G.add_edge("233604007", "196618008", relation="causes")  # pneumonia causes consolidation
        
        return G
    
    def load_radlex(self):
        """Load RadLex radiological ontology"""
        return {
            "RID5": "lung",
            "RID6": "heart", 
            "RID39": "chest",
            "RID154": "pneumonia",
            "RID155": "consolidation"
        }
    
    def get_concept_embedding(self, concept):
        """Get embedding for medical concept"""
        if concept in self.concept_embeddings:
            return self.concept_embeddings[concept]
        
        # Generate embedding based on concept relationships
        # This is a simplified version - would use actual concept embeddings
        embedding = torch.randn(768)  # Placeholder
        self.concept_embeddings[concept] = embedding
        return embedding
    
    def validate_terminology(self, generated_text):
        """Validate medical terminology in generated text"""
        # Extract medical terms and check against ontologies
        terms = self.extract_medical_terms(generated_text)
        validated_terms = []
        
        for term in terms:
            if self.is_valid_medical_term(term):
                validated_terms.append(term)
        
        return validated_terms
    
    def extract_medical_terms(self, text):
        """Extract medical terms from text using NER"""
        # This would use a medical NER model
        # Placeholder implementation
        import re
        medical_patterns = [
            r'\b(?:pneumonia|consolidation|effusion|cardiomegaly)\b',
            r'\b(?:opacity|infiltrate|nodule|mass)\b'
        ]
        
        terms = []
        for pattern in medical_patterns:
            terms.extend(re.findall(pattern, text.lower()))
        
        return terms
    
    def is_valid_medical_term(self, term):
        """Check if term exists in medical ontologies"""
        return (term in [node[1]['label'] for node in self.snomed_graph.nodes(data=True)] or
                term in self.radlex_terms.values())


## 3. Cross-Modal Attention and Alignment

### Attention Mechanisms


class CrossModalAttention(nn.Module):
    """Cross-attention between visual and textual features"""
    
    def __init__(self, 
                 visual_dim=768,
                 text_dim=768,
                 hidden_dim=512,
                 num_heads=8):
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
            dropout=0.1,
            batch_first=True
        )
        
        # Output projections
        self.visual_output = nn.Linear(hidden_dim, visual_dim)
        self.text_output = nn.Linear(hidden_dim, text_dim)
        
        # Layer normalization
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
    
    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: [B, N_visual, visual_dim]
            text_features: [B, N_text, text_dim]
        """
        batch_size = visual_features.shape[0]
        
        # Project to common space
        visual_proj = self.visual_proj(visual_features)
        text_proj = self.text_proj(text_features)
        
        # Visual attending to text
        visual_attended, visual_attention = self.cross_attention(
            query=visual_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Text attending to visual
        text_attended, text_attention = self.cross_attention(
            query=text_proj,
            key=visual_proj,
            value=visual_proj
        )
        
        # Project back and residual connection
        visual_output = self.visual_norm(
            visual_features + self.visual_output(visual_attended)
        )
        text_output = self.text_norm(
            text_features + self.text_output(text_attended)
        )
        
        return visual_output, text_output, visual_attention, text_attention

class ContrastiveLearning(nn.Module):
    """Contrastive learning for image-text alignment"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: [B, D]
            text_features: [B, D]
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Labels for contrastive loss
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Symmetric loss
        loss_i2t = F.cross_entropy(similarity, labels)
        loss_t2i = F.cross_entropy(similarity.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss, similarity


## 4. Explanation Generation and Visualization

### Gradient-based Explanations


from captum.attr import (
    IntegratedGradients, GradCAM, LayerGradCam,
    visualization as viz
)
import matplotlib.pyplot as plt

class ExplanationGenerator:
    """Generate visual and textual explanations"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.gradcam = GradCAM(self.model, self.model.vision_encoder.backbone.layers[-1])
    
    def generate_visual_explanation(self, image, target_class=None):
        """Generate visual saliency maps"""
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
    
    def generate_attention_explanation(self, image, text_input):
        """Generate attention-based explanations"""
        self.model.eval()
        
        with torch.no_grad():
            # Get attention weights from cross-modal attention
            visual_features = self.model.vision_encoder(image.unsqueeze(0))
            
            # Tokenize text
            text_tokens = self.model.language_decoder.tokenizer.encode(
                text_input, return_tensors='pt'
            )
            text_features = self.model.language_decoder.language_model.get_input_embeddings()(
                text_tokens
            )
            
            # Get cross-attention weights
            cross_modal_layer = self.model.cross_modal_attention
            _, _, visual_attention, text_attention = cross_modal_layer(
                visual_features, text_features
            )
        
        return {
            'visual_attention': visual_attention.squeeze().cpu(),
            'text_attention': text_attention.squeeze().cpu()
        }
    
    def visualize_explanation(self, explanations, save_path=None):
        """Create visualization of explanations"""
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
        axes[0, 2].imshow(explanations['input_image'].squeeze(), cmap='gray', alpha=0.3)
        axes[0, 2].set_title('GradCAM Overlay')
        axes[0, 2].axis('off')
        
        # Attention visualizations (if available)
        if 'visual_attention' in explanations:
            visual_att = explanations['visual_attention']
            axes[1, 0].imshow(visual_att.mean(dim=0), cmap='Blues')
            axes[1, 0].set_title('Visual Attention')
            axes[1, 0].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class CounterfactualExplanation:
    """Generate counterfactual explanations"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_counterfactual(self, image, original_prediction, target_change):
        """
        Generate counterfactual by modifying image regions
        
        Args:
            image: Input medical image
            original_prediction: Original model prediction
            target_change: Desired change in prediction
        """
        # This would implement counterfactual generation
        # For now, we'll create a simple perturbation-based approach
        
        perturbations = []
        predictions = []
        
        # Create multiple perturbations
        for i in range(10):
            # Random perturbation
            noise = torch.randn_like(image) * 0.1
            perturbed_image = image + noise
            
            # Get prediction
            with torch.no_grad():
                pred = self.model(perturbed_image.unsqueeze(0))
            
            perturbations.append(perturbed_image)
            predictions.append(pred)
        
        return perturbations, predictions


## 5. Complete Model Integration

### Main Model Class


class ExplainableMedicalVLM(nn.Module):
    """Complete explainable medical vision-language model"""
    
    def __init__(self,
                 vision_model='swin_base_patch4_window7_224',
                 language_model='microsoft/DialoGPT-medium',
                 num_classes=14):
        super().__init__()
        
        # Initialize components
        self.vision_encoder = MedicalVisionEncoder(vision_model)
        self.language_decoder = MedicalLanguageDecoder(language_model)
        self.cross_modal_attention = CrossModalAttention()
        self.contrastive_learning = ContrastiveLearning()
        
        # Classification head for multi-task learning
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Knowledge graph integration
        self.knowledge_graph = MedicalKnowledgeGraph()
        
        # Explanation generator
        self.explanation_generator = ExplanationGenerator(self)
    
    def forward(self, 
                images,
                text_input_ids=None,
                attention_mask=None,
                labels=None,
                mode='train'):
        
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
            
            # Validate terminology
            validated_explanations = []
            for explanation in explanations:
                terms = self.knowledge_graph.validate_terminology(explanation)
                validated_explanations.append(explanation)
            
            return {
                'explanations': validated_explanations,
                'visual_attention': visual_attention
            }
        
        elif mode == 'train':
            # Multi-task training
            # 1. Classification loss
            classification_logits = self.classifier(visual_global)
            classification_loss = F.cross_entropy(classification_logits, labels)
            
            # 2. Language generation loss
            language_outputs = self.language_decoder(
                visual_features,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                labels=text_input_ids  # Self-supervised
            )
            language_loss = language_outputs.loss
            
            # 3. Contrastive loss
            text_features = language_outputs.hidden_states[-1].mean(dim=1)
            contrastive_loss, similarity = self.contrastive_learning(
                visual_global, text_features
            )
            
            # Combined loss
            total_loss = (
                classification_loss + 
                language_loss + 
                0.1 * contrastive_loss
            )
            
            return {
                'loss': total_loss,
                'classification_loss': classification_loss,
                'language_loss': language_loss,
                'contrastive_loss': contrastive_loss,
                'logits': classification_logits
            }
    
    def generate_full_explanation(self, image, return_attention=True):
        """Generate comprehensive explanation with visualizations"""
        self.eval()
        
        with torch.no_grad():
            # Get model outputs
            classification_output = self.forward(
                image.unsqueeze(0), mode='classification'
            )
            explanation_output = self.forward(
                image.unsqueeze(0), mode='explanation'
            )
            
            # Generate visual explanations
            visual_explanations = self.explanation_generator.generate_visual_explanation(
                image, target_class=classification_output['logits'].argmax()
            )
            
            # Generate attention explanations
            attention_explanations = self.explanation_generator.generate_attention_explanation(
                image, explanation_output['explanations'][0]
            )
            
            # Combine all explanations
            full_explanation = {
                'prediction': classification_output['logits'],
                'text_explanation': explanation_output['explanations'][0],
                'visual_saliency': visual_explanations,
                'attention_maps': attention_explanations if return_attention else None,
                'confidence': F.softmax(classification_output['logits'], dim=-1).max().item()
            }
        
        return full_explanation


## 6. Training Pipeline

### Data Loading and Preprocessing

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalImageTextDataset(Dataset):
    """Dataset for medical image-text pairs"""
    
    def __init__(self, 
                 csv_path,
                 image_dir,
                 tokenizer,
                 max_length=512,
                 img_size=224,
                 split='train'):
        
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Image preprocessing
        if split == 'train':
            self.image_transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
        else:
            self.image_transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, row['image_filename'])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback to PIL for DICOM or other formats
            image = np.array(Image.open(image_path).convert('L'))
        
        # Apply transforms
        transformed = self.image_transform(image=image)
        image_tensor = transformed['image']
        
        # If grayscale, add channel dimension
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Process text
        report_text = row['report_text'] if pd.notna(row['report_text']) else ""
        findings_text = row['findings'] if pd.notna(row['findings']) else ""
        impression_text = row['impression'] if pd.notna(row['impression']) else ""
        
        # Combine text sections
        full_text = f"Findings: {findings_text} Impression: {impression_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Labels for classification (if available)
        labels = []
        pathology_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]
        
        for col in pathology_columns:
            if col in row:
                labels.append(1.0 if row[col] == 1 else 0.0)
            else:
                labels.append(0.0)
        
        return {
            'image': image_tensor,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float),
            'text': full_text,
            'image_id': row.get('image_id', idx)
        }

class MedicalVLMTrainer:
    """Training pipeline for the medical VLM"""
    
    def __init__(self, 
                 model,
                 train_dataset,
                 val_dataset,
                 batch_size=8,
                 learning_rate=5e-5,
                 num_epochs=10,
                 device='cuda'):
        
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        classification_losses = 0
        language_losses = 0
        contrastive_losses = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                images=images,
                text_input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                mode='train'
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            classification_losses += outputs['classification_loss'].item()
            language_losses += outputs['language_loss'].item()
            contrastive_losses += outputs['contrastive_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = classification_losses / len(self.train_loader)
        avg_lang_loss = language_losses / len(self.train_loader)
        avg_cont_loss = contrastive_losses / len(self.train_loader)
        
        return {
            'avg_loss': avg_loss,
            'classification_loss': avg_class_loss,
            'language_loss': avg_lang_loss,
            'contrastive_loss': avg_cont_loss
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    images=images,
                    text_input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    mode='train'
                )
                
                total_loss += outputs['loss'].item()
                
                # Collect predictions for metrics
                predictions = torch.sigmoid(outputs['logits'])
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        # AUC-ROC for each class
        from sklearn.metrics import roc_auc_score
        auc_scores = []
        for i in range(all_labels.shape[1]):
            if len(torch.unique(all_labels[:, i])) > 1:  # Check if both classes present
                auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                auc_scores.append(auc)
        
        avg_auc = np.mean(auc_scores) if auc_scores else 0.0
        
        return {
            'avg_loss': avg_loss,
            'avg_auc': avg_auc,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self):
        """Complete training loop"""
        print("Starting training...")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            print(f"Val Loss: {val_metrics['avg_loss']:.4f}")
            print(f"Val AUC: {val_metrics['avg_auc']:.4f}")
            
            # Save best model
            if val_metrics['avg_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['avg_loss']
                torch.save(self.model.state_dict(), 'best_medical_vlm.pth')
                print("Saved new best model!")
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['avg_loss'],
                    'train_classification_loss': train_metrics['classification_loss'],
                    'train_language_loss': train_metrics['language_loss'],
                    'train_contrastive_loss': train_metrics['contrastive_loss'],
                    'val_loss': val_metrics['avg_loss'],
                    'val_auc': val_metrics['avg_auc']
                })

def create_sample_dataset():
    """Create a sample dataset CSV for demonstration"""
    
    # Sample data structure for medical imaging dataset
    sample_data = {
        'image_id': [f'img_{i:05d}' for i in range(1000)],
        'image_filename': [f'img_{i:05d}.jpg' for i in range(1000)],
        'patient_id': [f'patient_{i//10:04d}' for i in range(1000)],
        'study_id': [f'study_{i:05d}' for i in range(1000)],
        
        # Patient demographics
        'patient_age': np.random.randint(18, 90, 1000),
        'patient_sex': np.random.choice(['M', 'F'], 1000),
        
        # Image metadata
        'view_position': np.random.choice(['PA', 'AP', 'LAT'], 1000),
        'modality': ['CR'] * 1000,  # Chest Radiography
        
        # Pathology labels (14 classes from CheXpert/NIH datasets)
        'Atelectasis': np.random.choice([0, 1, -1], 1000, p=[0.7, 0.2, 0.1]),
        'Cardiomegaly': np.random.choice([0, 1, -1], 1000, p=[0.8, 0.15, 0.05]),
        'Consolidation': np.random.choice([0, 1, -1], 1000, p=[0.85, 0.1, 0.05]),
        'Edema': np.random.choice([0, 1, -1], 1000, p=[0.9, 0.08, 0.02]),
        'Enlarged Cardiomediastinum': np.random.choice([0, 1, -1], 1000, p=[0.95, 0.04, 0.01]),
        'Fracture': np.random.choice([0, 1, -1], 1000, p=[0.98, 0.015, 0.005]),
        'Lung Lesion': np.random.choice([0, 1, -1], 1000, p=[0.95, 0.04, 0.01]),
        'Lung Opacity': np.random.choice([0, 1, -1], 1000, p=[0.7, 0.25, 0.05]),
        'No Finding': np.random.choice([0, 1], 1000, p=[0.3, 0.7]),
        'Pleural Effusion': np.random.choice([0, 1, -1], 1000, p=[0.8, 0.15, 0.05]),
        'Pleural Other': np.random.choice([0, 1, -1], 1000, p=[0.95, 0.04, 0.01]),
        'Pneumonia': np.random.choice([0, 1, -1], 1000, p=[0.9, 0.08, 0.02]),
        'Pneumothorax': np.random.choice([0, 1, -1], 1000, p=[0.95, 0.04, 0.01]),
        'Support Devices': np.random.choice([0, 1, -1], 1000, p=[0.85, 0.12, 0.03]),
    }
    
    # Generate realistic medical reports
    def generate_report(row):
        findings = []
        impression = []
        
        if row['No Finding'] == 1:
            findings.append("The lungs are clear without focal consolidation, pleural effusion, or pneumothorax.")
            findings.append("The cardiac silhouette is normal in size and contour.")
            findings.append("The mediastinal and hilar contours are unremarkable.")
            impression.append("No acute cardiopulmonary abnormality.")
        else:
            # Generate findings based on positive labels
            if row['Atelectasis'] == 1:
                findings.append("There is evidence of atelectasis in the lung bases.")
            if row['Cardiomegaly'] == 1:
                findings.append("The cardiac silhouette appears enlarged.")
                impression.append("Cardiomegaly.")
            if row['Consolidation'] == 1:
                findings.append("Focal consolidation is noted in the right lower lobe.")
                impression.append("Right lower lobe consolidation, possibly infectious.")
            if row['Pleural Effusion'] == 1:
                findings.append("Small pleural effusion is present.")
                impression.append("Small pleural effusion.")
            if row['Pneumonia'] == 1:
                findings.append("Patchy opacities consistent with pneumonia.")
                impression.append("Findings consistent with pneumonia.")
            if row['Support Devices'] == 1:
                findings.append("Endotracheal tube and central venous catheter are in appropriate position.")
        
        if not findings:
            findings.append("No significant abnormalities detected.")
        if not impression:
            impression.append("Normal chest radiograph.")
        
        return " ".join(findings), " ".join(impression)
    
    # Generate reports
    reports_data = []
    for i in range(1000):
        row = {key: sample_data[key][i] for key in sample_data.keys()}
        findings, impression = generate_report(row)
        reports_data.append({
            'findings': findings,
            'impression': impression,
            'report_text': f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
        })
    
    # Add reports to main data
    for key in ['findings', 'impression', 'report_text']:
        sample_data[key] = [report[key] for report in reports_data]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    return df

# Example usage and dataset creation
if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample medical dataset...")
    sample_df = create_sample_dataset()
    
    # Save to CSV
    sample_df.to_csv('medical_image_dataset.csv', index=False)
    
    print("Dataset created with the following structure:")
    print(f"Shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")
    print("\nFirst few rows:")
    print(sample_df.head())
    
    print("\nSample report:")
    print(sample_df['report_text'].iloc[0])
    
    print("\nLabel distribution:")
    pathology_cols = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    for col in pathology_cols:
        positive_count = (sample_df[col] == 1).sum()
        print(f"{col}: {positive_count} positive cases ({positive_count/len(sample_df)*100:.1f}%)")

# Training script example
def train_medical_vlm():
    """Example training script"""
    
    # Initialize model
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = ExplainableMedicalVLM()
    
    # Create datasets
    train_dataset = MedicalImageTextDataset(
        csv_path='medical_image_dataset.csv',
        image_dir='./images/train',
        tokenizer=tokenizer,
        split='train'
    )
    
    val_dataset = MedicalImageTextDataset(
        csv_path='medical_image_dataset.csv',
        image_dir='./images/val',
        tokenizer=tokenizer,
        split='val'
    )
    
    # Initialize trainer
    trainer = MedicalVLMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        learning_rate=5e-5,
        num_epochs=10
    )
    
    # Start training
    trainer.train()

# Inference example
def inference_example():
    """Example inference with explanations"""
    
    # Load trained model
    model = ExplainableMedicalVLM()
    model.load_state_dict(torch.load('best_medical_vlm.pth'))
    model.eval()
    
    # Load sample image
    image_path = 'sample_chest_xray.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)
    
    # Generate explanation
    explanation = model.generate_full_explanation(image_tensor.squeeze(0))
    
    print("Prediction:", explanation['prediction'])
    print("Text Explanation:", explanation['text_explanation'])
    print("Confidence:", explanation['confidence'])
    
    # Visualize
    if explanation['visual_saliency']:
        model.explanation_generator.visualize_explanation(
            explanation['visual_saliency'],
            save_path='explanation_visualization.png'
        )