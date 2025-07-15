#!/usr/bin/env python3
"""
Example usage of the improved EVLMs framework
This script demonstrates how to use the enhanced medical VLM with all improvements
"""

import torch
import logging
import argparse
from pathlib import Path

# Import improved components
from EVLMs.models.medical_vlm_improved import ImprovedExplainableMedicalVLM, ModelConfig
from EVLMs.trainers.trainer_improved import ImprovedMedicalVLMTrainer, TrainingConfig
from EVLMs.data.dataset import MedicalImageTextDataset
from EVLMs.configs.config import DatasetConfig

def setup_logging(output_dir: Path):
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_model_config(args) -> ModelConfig:
    """Create model configuration based on arguments"""
    return ModelConfig(
        vision_model=args.vision_model,
        language_model=args.language_model,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        enable_uncertainty=args.enable_uncertainty,
        adaptive_loss_weights=args.adaptive_loss_weights
    )

def create_training_config(args) -> TrainingConfig:
    """Create training configuration based on arguments"""
    return TrainingConfig(
        # Basic settings
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimizer settings
        optimizer_type=args.optimizer_type,
        weight_decay=args.weight_decay,
        
        # Scheduler settings
        scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        
        # Memory and performance
        use_mixed_precision=args.use_mixed_precision,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        
        # Monitoring
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        
        # Early stopping
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )

def demonstrate_model_features(model, sample_image, logger):
    """Demonstrate the enhanced model features"""
    logger.info("=== Demonstrating Enhanced Model Features ===")
    
    # 1. Model information
    model_info = model.get_model_info()
    logger.info(f"Model Parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"Vision Feature Dim: {model_info['vision_feature_dim']}")
    logger.info(f"Language Feature Dim: {model_info['language_feature_dim']}")
    
    # 2. Different forward modes
    logger.info("\n--- Testing Different Forward Modes ---")
    
    # Classification only
    with torch.no_grad():
        class_output = model(sample_image, mode='classification')
        logger.info(f"Classification output shape: {class_output['logits'].shape}")
        
        # Full inference
        full_output = model(sample_image, mode='full')
        logger.info(f"Full inference keys: {list(full_output.keys())}")
        
        # Explanation generation
        explanation_output = model(sample_image, mode='explanation')
        logger.info(f"Generated explanation length: {len(explanation_output['explanations'][0])}")
    
    # 3. Enhanced explainability
    logger.info("\n--- Testing Enhanced Explainability ---")
    
    # Attention-based explanation
    attention_explanation = model.explain(sample_image, method='attention')
    logger.info(f"Attention explanation keys: {list(attention_explanation.keys())}")
    
    # GradCAM explanation
    try:
        gradcam_explanation = model.explain(sample_image, method='gradcam')
        logger.info(f"GradCAM shape: {gradcam_explanation['gradcam'].shape}")
    except Exception as e:
        logger.warning(f"GradCAM failed: {e}")
    
    # Comprehensive explanation
    all_explanations = model.explain(sample_image, method='all')
    logger.info(f"All explanations keys: {list(all_explanations.keys())}")
    
    # 4. Feature caching demonstration
    logger.info("\n--- Testing Feature Caching ---")
    cache_size_before = model.feature_cache.cache.__len__()
    
    # Multiple forward passes with same input
    for i in range(3):
        with torch.no_grad():
            _ = model(sample_image, mode='classification')
    
    cache_size_after = model.feature_cache.cache.__len__()
    logger.info(f"Cache size before: {cache_size_before}, after: {cache_size_after}")
    
    # Clear cache
    model.clear_cache()
    logger.info(f"Cache cleared, size now: {len(model.feature_cache.cache)}")

def run_training_example(args):
    """Run a complete training example with the improved framework"""
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("=== Starting Improved EVLMs Training Example ===")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create configurations
    model_config = create_model_config(args)
    training_config = create_training_config(args)
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    # Create model
    logger.info("Creating improved model...")
    model = ImprovedExplainableMedicalVLM(model_config)
    model = model.to(device)
    
    # Create sample data for demonstration
    sample_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Demonstrate model features
    demonstrate_model_features(model, sample_image, logger)
    
    # Load datasets (placeholder - replace with actual dataset loading)
    logger.info("Loading datasets...")
    try:
        # This would be your actual dataset loading code
        dataset_config = DatasetConfig(data_dir=args.dataset_path)
        
        # For demonstration, we'll create dummy datasets
        # Replace this with actual MedicalImageTextDataset loading
        train_dataset = None  # MedicalImageTextDataset(dataset_config, split='train')
        val_dataset = None    # MedicalImageTextDataset(dataset_config, split='val')
        
        if train_dataset is None:
            logger.warning("No actual datasets provided, skipping training")
            logger.info("To run actual training, provide --dataset_path argument")
            return
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        logger.info("Running in demo mode without actual training")
        return
    
    # Create trainer
    logger.info("Creating improved trainer...")
    trainer = ImprovedMedicalVLMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        device=device,
        output_dir=str(output_dir),
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)
    
    # Start training
    logger.info("Starting training with improved framework...")
    trainer.train()
    
    logger.info("Training completed successfully!")

def run_inference_example(args):
    """Run inference example with the improved model"""
    logger = logging.getLogger(__name__)
    logger.info("=== Running Inference Example ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_config = create_model_config(args)
    model = ImprovedExplainableMedicalVLM(model_config)
    
    if args.model_checkpoint:
        logger.info(f"Loading model from {args.model_checkpoint}")
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Create sample input
    sample_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Run different types of inference
    logger.info("Running classification inference...")
    with torch.no_grad():
        class_result = model(sample_image, mode='classification')
        predictions = torch.sigmoid(class_result['logits'])
        logger.info(f"Predictions: {predictions.cpu().numpy()}")
    
    logger.info("Running explanation generation...")
    explanations = model.explain(sample_image, method='all')
    logger.info(f"Generated explanation: {explanations.get('explanation_text', 'N/A')}")
    
    if 'epistemic_uncertainty' in explanations:
        uncertainty = explanations['epistemic_uncertainty'].mean().item()
        logger.info(f"Average uncertainty: {uncertainty:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Improved EVLMs Framework Example")
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'inference', 'demo'], default='demo',
                       help='Mode to run: train, inference, or demo')
    
    # Model configuration
    parser.add_argument('--vision_model', default='microsoft/swin-base-patch4-window7-224',
                       help='Vision model name')
    parser.add_argument('--language_model', default='microsoft/DialoGPT-medium',
                       help='Language model name')
    parser.add_argument('--num_classes', type=int, default=14,
                       help='Number of classification classes')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for cross-attention')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Optimizer settings
    parser.add_argument('--optimizer_type', choices=['adamw', 'sgd', 'adam'], default='adamw',
                       help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # Scheduler settings
    parser.add_argument('--scheduler_type', 
                       choices=['linear_warmup', 'cosine', 'plateau', 'onecycle'], 
                       default='linear_warmup',
                       help='Scheduler type')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    
    # Advanced features
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing')
    parser.add_argument('--enable_uncertainty', action='store_true', default=True,
                       help='Enable uncertainty estimation')
    parser.add_argument('--adaptive_loss_weights', action='store_true', default=True,
                       help='Use adaptive loss weighting')
    
    # Training monitoring
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Save steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Logging steps')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--early_stopping_threshold', type=float, default=1e-4,
                       help='Early stopping threshold')
    
    # Other settings
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Paths
    parser.add_argument('--dataset_path', type=str,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', default='outputs/improved_training',
                       help='Output directory')
    parser.add_argument('--model_checkpoint', type=str,
                       help='Path to model checkpoint for inference')
    parser.add_argument('--resume_from_checkpoint', type=str,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training_example(args)
    elif args.mode == 'inference':
        run_inference_example(args)
    else:  # demo mode
        # Create a minimal demo
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(output_dir)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_config = create_model_config(args)
        model = ImprovedExplainableMedicalVLM(model_config)
        model = model.to(device)
        
        sample_image = torch.randn(1, 3, 224, 224).to(device)
        demonstrate_model_features(model, sample_image, logger)
        
        logger.info("Demo completed! Check the logs for detailed output.")
        logger.info("To run actual training, use: python example_improved_usage.py --mode train --dataset_path /path/to/your/dataset")

if __name__ == "__main__":
    main()