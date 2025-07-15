import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    ReduceLROnPlateau, 
    OneCycleLR
)
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import logging
import json
import time
import psutil
import gc
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np

from EVLMs.utils.metrics import calculate_metrics, calculate_language_metrics
from EVLMs.utils.logger import log_metrics
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    # Basic training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    
    # Optimizer settings
    optimizer_type: str = 'adamw'  # 'adamw', 'sgd', 'adam'
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.9
    
    # Scheduler settings
    scheduler_type: str = 'linear_warmup'  # 'linear_warmup', 'cosine', 'plateau', 'onecycle'
    warmup_ratio: float = 0.1
    lr_patience: int = 3
    lr_factor: float = 0.5
    
    # Memory and performance
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Monitoring and saving
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # Advanced features
    curriculum_learning: bool = False
    progressive_resizing: bool = False
    label_smoothing: float = 0.0
    
    # Validation settings
    validation_split: float = 0.2
    stratified_split: bool = True

class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        memory_info['cpu_memory_percent'] = process.memory_percent()
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info['gpu_memory_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return memory_info
    
    @staticmethod
    def cleanup_memory():
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 5, threshold: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.threshold
        else:
            return score > self.best_score + self.threshold

class MetricsTracker:
    """Enhanced metrics tracking"""
    
    def __init__(self, window_size: int = 100):
        self.metrics = defaultdict(list)
        self.window_size = window_size
        self.recent_metrics = defaultdict(lambda: deque(maxlen=window_size))
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics"""
        for key, value in metrics.items():
            self.metrics[key].append((step, value))
            self.recent_metrics[key].append(value)
    
    def get_recent_average(self, key: str) -> float:
        """Get recent average of a metric"""
        if key in self.recent_metrics and len(self.recent_metrics[key]) > 0:
            return sum(self.recent_metrics[key]) / len(self.recent_metrics[key])
        return 0.0
    
    def get_trend(self, key: str, window: int = 10) -> str:
        """Get trend of a metric (increasing/decreasing/stable)"""
        if key not in self.recent_metrics or len(self.recent_metrics[key]) < window:
            return "insufficient_data"
        
        recent_values = list(self.recent_metrics[key])[-window:]
        first_half = np.mean(recent_values[:window//2])
        second_half = np.mean(recent_values[window//2:])
        
        if second_half > first_half * 1.05:
            return "increasing"
        elif second_half < first_half * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def plot_metrics(self, save_path: str):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        key_metrics = ['train_loss', 'val_loss', 'val_auc', 'learning_rate']
        
        for i, metric in enumerate(key_metrics):
            if metric in self.metrics and len(self.metrics[metric]) > 0:
                steps, values = zip(*self.metrics[metric])
                axes[i].plot(steps, values)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel('Value')
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class ImprovedMedicalVLMTrainer:
    """Enhanced trainer with advanced features"""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Any,
                 val_dataset: Any,
                 config: TrainingConfig = None,
                 device: str = 'cuda',
                 output_dir: str = 'outputs',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced trainer
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            device: Device to train on
            output_dir: Directory to save outputs
            logger: Logger instance
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        self.device = device
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Initialize components
        self._setup_training()
        self._setup_monitoring()
        
        self.logger.info(f"Trainer initialized with config: {self.config}")
    
    def _setup_training(self):
        """Setup training components"""
        # Mixed precision
        self.use_mixed_precision = (
            self.config.use_mixed_precision and 
            self.device == 'cuda' and 
            torch.cuda.is_available()
        )
        
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.logger.info("Using mixed precision training (FP16)")
        else:
            self.scaler = None
            self.logger.info("Using full precision training (FP32)")
        
        # Data loaders with optimized settings
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            persistent_workers=True if self.config.dataloader_num_workers > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.dataloader_num_workers > 0 else False
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Gradient checkpointing
        if self.config.use_gradient_checkpointing:
            if hasattr(self.model, '_enable_gradient_checkpointing'):
                self.model._enable_gradient_checkpointing()
            self.logger.info("Gradient checkpointing enabled")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        # Separate parameters for different learning rates
        vision_params = []
        language_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'vision_encoder' in name:
                vision_params.append(param)
            elif 'language_decoder' in name:
                language_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': vision_params, 'lr': self.config.learning_rate * 0.1},  # Lower LR for vision
            {'params': language_params, 'lr': self.config.learning_rate * 0.5},  # Medium LR for language
            {'params': other_params, 'lr': self.config.learning_rate}  # Full LR for other components
        ]
        
        if self.config.optimizer_type.lower() == 'adamw':
            optimizer = AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.adam_beta1, self.config.adam_beta2)
            )
        elif self.config.optimizer_type.lower() == 'sgd':
            optimizer = SGD(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.sgd_momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        self.logger.info(f"Created {self.config.optimizer_type} optimizer with {len(param_groups)} parameter groups")
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        
        if self.config.scheduler_type == 'linear_warmup':
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.config.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                verbose=True
            )
        elif self.config.scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_ratio
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")
        
        self.logger.info(f"Created {self.config.scheduler_type} scheduler")
        return scheduler
    
    def _setup_monitoring(self):
        """Setup monitoring components"""
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            threshold=self.config.early_stopping_threshold
        )
        self.memory_manager = MemoryManager()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.training_start_time = None
    
    def _efficient_validation(self) -> Dict[str, float]:
        """Memory-efficient validation with batched processing"""
        self.model.eval()
        
        # Use smaller batches for validation to save memory
        val_batch_size = min(self.config.batch_size, 4)
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_generated_text = []
        all_reference_text = []
        
        # Process in chunks to avoid memory issues
        chunk_size = 100
        current_chunk = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision if enabled
                if self.use_mixed_precision:
                    with autocast(device_type=self.device):
                        outputs = self.model(
                            images=batch['image'],
                            text_input_ids=batch.get('input_ids'),
                            attention_mask=batch.get('attention_mask'),
                            labels=batch.get('labels'),
                            mode='train'
                        )
                        
                        # Generate explanations less frequently to save time
                        if batch_idx % 10 == 0:  # Only every 10th batch
                            explanations = self.model(
                                images=batch['image'],
                                mode='explanation'
                            )['explanations']
                        else:
                            explanations = [''] * len(batch['image'])
                else:
                    outputs = self.model(
                        images=batch['image'],
                        text_input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels'),
                        mode='train'
                    )
                    
                    if batch_idx % 10 == 0:
                        explanations = self.model(
                            images=batch['image'],
                            mode='explanation'
                        )['explanations']
                    else:
                        explanations = [''] * len(batch['image'])
                
                # Accumulate results
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                
                if 'logits' in outputs:
                    predictions = torch.sigmoid(outputs['logits']).cpu()
                    all_predictions.append(predictions)
                
                if 'labels' in batch and batch['labels'] is not None:
                    all_labels.append(batch['labels'].cpu())
                
                all_generated_text.extend(explanations)
                if 'text' in batch:
                    all_reference_text.extend(batch['text'])
                
                # Memory cleanup every chunk
                current_chunk += 1
                if current_chunk % chunk_size == 0:
                    self.memory_manager.cleanup_memory()
        
        # Calculate metrics
        metrics = {'loss': total_loss / len(val_loader)}
        
        if all_predictions and all_labels:
            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)
            
            classification_metrics = calculate_metrics(all_predictions, all_labels)
            metrics.update(classification_metrics)
        
        if all_generated_text and all_reference_text:
            language_metrics = calculate_language_metrics(
                all_generated_text, all_reference_text
            )
            metrics.update({f'language_{k}': v for k, v in language_metrics.items()})
        
        # Add memory usage
        memory_info = self.memory_manager.get_memory_usage()
        metrics.update({f'memory_{k}': v for k, v in memory_info.items()})
        
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Enhanced training epoch with better monitoring"""
        self.model.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Training Epoch",
            leave=False
        )
        
        # Gradient accumulation
        self.optimizer.zero_grad()
        accumulation_step = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            if self.use_mixed_precision:
                with autocast(device_type=self.device):
                    outputs = self.model(
                        images=batch['image'],
                        text_input_ids=batch.get('input_ids'),
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels'),
                        mode='train'
                    )
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    images=batch['image'],
                    text_input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels'),
                    mode='train'
                )
                loss = outputs['loss'] / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Accumulate metrics
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    epoch_metrics[key] += value.item()
            
            accumulation_step += 1
            
            # Optimizer step
            if (accumulation_step == self.config.gradient_accumulation_steps or 
                batch_idx == len(self.train_loader) - 1):
                
                if self.use_mixed_precision:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                # Scheduler step (except for ReduceLROnPlateau)
                if not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                accumulation_step = 0
                self.global_step += 1
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f"{outputs['loss'].item():.4f}",
                    'lr': f"{current_lr:.2e}",
                    'step': self.global_step
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    train_metrics = {
                        'train_loss': outputs['loss'].item(),
                        'learning_rate': current_lr,
                        'global_step': self.global_step
                    }
                    
                    # Add component losses if available
                    for key in ['classification_loss', 'language_loss', 'contrastive_loss']:
                        if key in outputs:
                            train_metrics[f'train_{key}'] = outputs[key].item()
                    
                    self.metrics_tracker.update(train_metrics, self.global_step)
                    
                    # Log to wandb
                    try:
                        wandb.log(train_metrics, step=self.global_step)
                    except:
                        pass
                
                # Validation
                if self.global_step % self.config.eval_steps == 0:
                    val_metrics = self._efficient_validation()
                    
                    # Update metrics tracker
                    val_metrics_prefixed = {f'val_{k}': v for k, v in val_metrics.items()}
                    self.metrics_tracker.update(val_metrics_prefixed, self.global_step)
                    
                    # Scheduler step for ReduceLROnPlateau
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    
                    # Early stopping check
                    if self.early_stopping(val_metrics['loss']):
                        self.logger.info("Early stopping triggered")
                        return self._finalize_epoch_metrics(epoch_metrics, num_batches)
                    
                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint(val_metrics, is_best=True)
                    
                    if 'mean_auc' in val_metrics and val_metrics['mean_auc'] > self.best_val_auc:
                        self.best_val_auc = val_metrics['mean_auc']
                
                # Regular checkpoint saving
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint({}, is_best=False)
                
                # Memory cleanup
                if self.global_step % 100 == 0:
                    self.memory_manager.cleanup_memory()
            
            num_batches += 1
        
        return self._finalize_epoch_metrics(epoch_metrics, num_batches)
    
    def _finalize_epoch_metrics(self, epoch_metrics: Dict, num_batches: int) -> Dict[str, float]:
        """Finalize epoch metrics"""
        return {k: v / num_batches for k, v in epoch_metrics.items()}
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Enhanced checkpoint saving with metadata"""
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_val_auc': self.best_val_auc,
            'config': asdict(self.config),
            'metrics': metrics
        }
        
        if self.use_mixed_precision:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            checkpoint_path = os.path.join(self.output_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint_step_{self.global_step}.pt')
        
        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        checkpoint_files = []
        for file in os.listdir(self.output_dir):
            if file.startswith('checkpoint_step_') and file.endswith('.pt'):
                step = int(file.split('_')[2].split('.')[0])
                checkpoint_files.append((step, file))
        
        # Keep only the most recent checkpoints
        checkpoint_files.sort(reverse=True)
        for step, file in checkpoint_files[self.config.save_total_limit:]:
            file_path = os.path.join(self.output_dir, file)
            os.remove(file_path)
            self.logger.info(f"Removed old checkpoint: {file}")
    
    def train(self):
        """Enhanced training loop"""
        self.logger.info("Starting enhanced training...")
        self.training_start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                self.logger.info(f"{'='*50}")
                
                # Training
                train_metrics = self.train_epoch()
                
                # Final validation for the epoch
                val_metrics = self._efficient_validation()
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Log epoch summary
                self.logger.info(f"\nEpoch {epoch + 1} Summary:")
                self.logger.info(f"  Time: {epoch_time:.2f}s")
                self.logger.info(f"  Train Loss: {train_metrics.get('loss', 0):.4f}")
                self.logger.info(f"  Val Loss: {val_metrics.get('loss', 0):.4f}")
                if 'mean_auc' in val_metrics:
                    self.logger.info(f"  Val AUC: {val_metrics['mean_auc']:.4f}")
                
                # Memory usage
                memory_info = self.memory_manager.get_memory_usage()
                self.logger.info(f"  GPU Memory: {memory_info.get('gpu_memory_allocated_mb', 0):.1f}MB")
                
                # Learning rate info
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"  Learning Rate: {current_lr:.2e}")
                
                # Trends
                loss_trend = self.metrics_tracker.get_trend('val_loss')
                self.logger.info(f"  Loss Trend: {loss_trend}")
                
                # Early stopping check
                if self.early_stopping.early_stop:
                    self.logger.info("Early stopping triggered!")
                    break
                
                # Memory cleanup
                self.memory_manager.cleanup_memory()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self._finalize_training()
    
    def _finalize_training(self):
        """Finalize training with summary and cleanup"""
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("Training Complete!")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Total Time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        self.logger.info(f"Total Steps: {self.global_step}")
        self.logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best Val AUC: {self.best_val_auc:.4f}")
        
        # Save final metrics plot
        plot_path = os.path.join(self.output_dir, 'training_metrics.png')
        self.metrics_tracker.plot_metrics(plot_path)
        self.logger.info(f"Saved metrics plot to {plot_path}")
        
        # Save training summary
        summary = {
            'total_time_seconds': total_time,
            'total_steps': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_val_auc': self.best_val_auc,
            'final_memory_usage': self.memory_manager.get_memory_usage(),
            'config': asdict(self.config)
        }
        
        with open(os.path.join(self.output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Final cleanup
        self.memory_manager.cleanup_memory()
        
        # Clear model cache if available
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        
        self.logger.info(f"Resumed from step {self.global_step}")
