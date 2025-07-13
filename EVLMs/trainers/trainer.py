import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional

from EVLMs.utils.metrics import calculate_metrics, calculate_language_metrics
from EVLMs.utils.logger import log_metrics

from torch.amp import autocast, GradScaler

class MedicalVLMTrainer:
    def log_validation_predictions(self, num_samples: int = 5):
        """Log predictions for a few validation samples after each epoch."""
        self.model.eval()
        count = 0
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            labels = batch['labels']
            texts = batch['text']
            # Predict explanations
            with torch.no_grad():
                outputs = self.model(images=images, mode='explanation')
                explanations = outputs['explanations']
                # Predict labels
                logits = self.model(images=images, mode='train')['logits']
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            for i in range(len(images)):
                if count >= num_samples:
                    return
                label_str = ', '.join([str(int(x)) for x in labels[i].cpu().numpy()])
                pred_str = ', '.join([str(int(x)) for x in preds[i]])
                self.logger.info(f"[VAL SAMPLE {count+1}]\n  Ground Truth Labels: {label_str}\n  Predicted Labels:   {pred_str}\n  Ground Truth Text:  {texts[i]}\n  Generated Explanation: {explanations[i]}\n-------------------------")
                count += 1
    """Trainer for medical vision-language model"""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Any,
                 val_dataset: Any,
                 batch_size: int = 8,
                 learning_rate: float = 5e-5,
                 num_epochs: int = 10,
                 device: str = 'cuda',
                 output_dir: str = 'outputs',
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            device: Device to train on
            output_dir: Directory to save outputs
            logger: Logger instance
        """
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = GradScaler('cuda')
        
        # Create data loaders
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
            model.parameters(),
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
        self.global_step = 0

    def log_sample_generation(self):
        """Log a sample of generated text to the console."""
        self.model.eval()
        sample = next(iter(self.val_loader))
        image = sample['image'].to(self.device)
        true_text = sample['text'][0]

        with torch.no_grad():
            output = self.model.explain(image=image)
        # Convert logits to labels
        if 'prediction' in output:
            output['prediction'] = (torch.softmax(output['prediction'], dim=-1)).cpu().numpy()

        self.logger.info("--- Sample Generation ---")
        self.logger.info(f"Ground Truth: {true_text}")
        self.logger.info(f"Labels: {sample['labels'][0].cpu().numpy()}")
        self.logger.info(f"Generated Text: {output['explanation'][0]}")
        self.logger.info(f"Generated Labels: {output['prediction'][0]}")
        self.logger.info(f"Visual Attention: {output['visual_attention'][0]}")
        self.logger.info(f"Cross Attention: {output['cross_attention']}")
        self.logger.info("-------------------------")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        classification_losses = 0
        language_losses = 0
        contrastive_losses = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            with autocast('cuda'):
                outputs = self.model(
                    images=batch['image'],
                    text_input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    mode='train'
                )
                loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            classification_losses += outputs['classification_loss'].item()
            language_losses += outputs['language_loss'].item()
            contrastive_losses += outputs['contrastive_loss'].item()
            
            # Update progress bar
            self.global_step += 1
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log metrics
            if self.global_step % 100 == 0:
                metrics = {
                    'train/loss': loss.item(),
                    'train/classification_loss': outputs['classification_loss'].item(),
                    'train/language_loss': outputs['language_loss'].item(),
                    'train/contrastive_loss': outputs['contrastive_loss'].item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                }
                log_metrics(self.logger, metrics, self.global_step, prefix='Train')
        
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = classification_losses / len(self.train_loader)
        avg_lang_loss = language_losses / len(self.train_loader)
        avg_cont_loss = contrastive_losses / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'classification_loss': avg_class_loss,
            'language_loss': avg_lang_loss,
            'contrastive_loss': avg_cont_loss
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_generated_text = []
        all_reference_text = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                images=batch['image'],
                text_input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                mode='train'
            )
            
            total_loss += outputs['loss'].item()
            
            # Collect predictions
            predictions = torch.sigmoid(outputs['logits'])
            all_predictions.append(predictions.cpu())
            all_labels.append(batch['labels'].cpu())
            
            # Generate text
            explanations = self.model(
                images=batch['image'],
                mode='explanation'
            )['explanations']
            
            all_generated_text.extend(explanations)
            all_reference_text.extend(batch['text'])
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        metrics = calculate_metrics(all_predictions, all_labels)
        language_metrics = calculate_language_metrics(
            all_generated_text, all_reference_text
        )
        
        # Combine metrics
        metrics.update({
            'loss': total_loss / len(self.val_loader),
            **{f'language_{k}': v for k, v in language_metrics.items()}
        })
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float]):
        """Save only model weights to avoid disk errors"""
        checkpoint_path = os.path.join(
            self.output_dir,
            f'checkpoint_step_{self.global_step}.pt'
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Saved model weights to {checkpoint_path}")
    
    def train(self):
        """Complete training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self.logger.info(f"--- Epoch {epoch + 1}/{self.num_epochs} Summary ---")
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            self.logger.info(f"Val AUC: {val_metrics['mean_auc']:.4f}")
            for k, v in val_metrics.items():
                if 'language' in k:
                    self.logger.info(f"Val {k.replace('language_', '').upper()}: {v:.4f}")
            self.logger.info("---------------------------------")
            
            # Save if best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(val_metrics)
                self.logger.info("Saved new best model!")
            
            # Log to wandb
            try:
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()}
                })
            except:
                pass

            # Show a sample of generated text
            self.log_sample_generation() 