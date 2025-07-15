#!/usr/bin/env python3
"""
Example configurations for different GPU memory scenarios using the improved EVLMs framework
This demonstrates how to configure the improved trainer for various hardware setups
"""

from EVLMs.models.medical_vlm_improved import ModelConfig
from EVLMs.trainers.trainer_improved import TrainingConfig

# Configuration for high-end GPUs (24GB+ VRAM)
high_memory_config = TrainingConfig(
    # Large batch sizes for maximum throughput
    batch_size=16,
    gradient_accumulation_steps=1,
    
    # Aggressive learning settings
    learning_rate=1e-4,
    optimizer_type='adamw',
    scheduler_type='cosine',
    warmup_ratio=0.1,
    
    # Full precision for maximum accuracy
    use_mixed_precision=False,
    use_gradient_checkpointing=False,
    
    # Frequent evaluation for better monitoring
    eval_steps=250,
    save_steps=500,
    logging_steps=50,
    
    # Performance optimizations
    dataloader_num_workers=8,
    pin_memory=True,
    
    # Advanced features
    early_stopping_patience=3,
    early_stopping_threshold=1e-5,
    
    output_dir="outputs/high_memory_training",
    experiment_name="high_memory_setup"
)

# Configuration for mid-range GPUs (12-16GB VRAM)
medium_memory_config = TrainingConfig(
    # Moderate batch sizes with gradient accumulation
    batch_size=8,
    gradient_accumulation_steps=2,  # Effective batch size = 16
    
    # Conservative learning settings
    learning_rate=5e-5,
    optimizer_type='adamw',
    scheduler_type='linear_warmup',
    warmup_ratio=0.15,
    
    # Mixed precision for memory efficiency
    use_mixed_precision=True,
    use_gradient_checkpointing=False,
    
    # Balanced evaluation frequency
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    
    # Moderate performance settings
    dataloader_num_workers=4,
    pin_memory=True,
    
    # Standard early stopping
    early_stopping_patience=5,
    early_stopping_threshold=1e-4,
    
    output_dir="outputs/medium_memory_training",
    experiment_name="medium_memory_setup"
)

# Configuration for low-end GPUs (8-12GB VRAM)
low_memory_config = TrainingConfig(
    # Small batch sizes with high gradient accumulation
    batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    
    # Conservative learning settings
    learning_rate=3e-5,
    optimizer_type='adamw',
    scheduler_type='linear_warmup',
    warmup_ratio=0.2,
    
    # Memory-saving features
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    
    # Less frequent evaluation to save memory
    eval_steps=1000,
    save_steps=2000,
    logging_steps=200,
    
    # Reduced performance settings
    dataloader_num_workers=2,
    pin_memory=False,
    
    # Patient early stopping
    early_stopping_patience=7,
    early_stopping_threshold=5e-4,
    
    output_dir="outputs/low_memory_training",
    experiment_name="low_memory_setup"
)

# Configuration for very limited memory (4-8GB VRAM)
minimal_memory_config = TrainingConfig(
    # Very small batch sizes with maximum gradient accumulation
    batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 16
    
    # Very conservative learning settings
    learning_rate=1e-5,
    optimizer_type='adamw',
    scheduler_type='plateau',
    lr_patience=3,
    lr_factor=0.5,
    
    # All memory-saving features enabled
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    
    # Minimal evaluation frequency
    eval_steps=2000,
    save_steps=4000,
    logging_steps=500,
    
    # Minimal performance settings
    dataloader_num_workers=1,
    pin_memory=False,
    
    # Very patient early stopping
    early_stopping_patience=10,
    early_stopping_threshold=1e-3,
    
    output_dir="outputs/minimal_memory_training",
    experiment_name="minimal_memory_setup"
)

def get_config_for_gpu_memory(gpu_memory_gb: float) -> TrainingConfig:
    """
    Get recommended configuration based on available GPU memory.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        TrainingConfig: Recommended configuration for the given memory
    """
    if gpu_memory_gb >= 24:  # High-end GPUs (A100, etc.)
        return high_memory_config
    elif gpu_memory_gb >= 12:  # Mid-high GPUs (RTX 3080 Ti, etc.)
        return medium_memory_config
    elif gpu_memory_gb >= 8:   # Mid-range GPUs (RTX 3070, etc.)
        return low_memory_config
    else:  # Low-end GPUs or limited memory
        return minimal_memory_config

def print_config_summary(config: TrainingConfig, config_name: str):
    """Print a summary of the configuration"""
    print(f"\n=== {config_name} ===")
    print(f"Actual batch size: {config.batch_size}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Mixed precision: {config.use_mixed_precision}")
    print(f"Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"Optimizer: {config.optimizer_type}")
    print(f"Scheduler: {config.scheduler_type}")
    print(f"Data workers: {config.dataloader_num_workers}")
    print(f"Early stopping patience: {config.early_stopping_patience}")

def demonstrate_memory_configurations():
    """Demonstrate different memory configurations"""
    print("EVLMs Improved Framework - Memory Configuration Examples")
    print("=" * 60)
    
    configs = [
        (high_memory_config, "High Memory (24GB+ VRAM)"),
        (medium_memory_config, "Medium Memory (12-16GB VRAM)"),
        (low_memory_config, "Low Memory (8-12GB VRAM)"),
        (minimal_memory_config, "Minimal Memory (4-8GB VRAM)")
    ]
    
    for config, name in configs:
        print_config_summary(config, name)
    
    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("=" * 60)
    
    print("\n1. Automatic configuration based on GPU memory:")
    print("```python")
    print("import torch")
    print("gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9")
    print("config = get_config_for_gpu_memory(gpu_memory_gb)")
    print("```")
    
    print("\n2. Manual configuration for specific setup:")
    print("```python")
    print("from EVLMs.trainers.trainer_improved import ImprovedMedicalVLMTrainer")
    print("trainer = ImprovedMedicalVLMTrainer(")
    print("    model=model,")
    print("    train_dataset=train_dataset,")
    print("    val_dataset=val_dataset,")
    print("    config=low_memory_config,  # Choose appropriate config")
    print("    device='cuda',")
    print("    output_dir='outputs/training'")
    print(")")
    print("trainer.train()")
    print("```")
    
    print("\n3. Custom configuration:")
    print("```python")
    print("custom_config = TrainingConfig(")
    print("    batch_size=6,")
    print("    gradient_accumulation_steps=3,")
    print("    learning_rate=2e-5,")
    print("    use_mixed_precision=True,")
    print("    use_gradient_checkpointing=True,")
    print("    scheduler_type='cosine',")
    print("    early_stopping_patience=5")
    print(")")
    print("```")

def create_model_configs():
    """Create different model configurations for various scenarios"""
    
    # Lightweight model configuration
    lightweight_model_config = ModelConfig(
        vision_model='microsoft/swin-tiny-patch4-window7-224',
        language_model='microsoft/DialoGPT-small',
        num_classes=14,
        hidden_dim=256,
        dropout_rate=0.1,
        use_gradient_checkpointing=True,
        enable_uncertainty=False,  # Disable for speed
        adaptive_loss_weights=False
    )
    
    # Standard model configuration
    standard_model_config = ModelConfig(
        vision_model='microsoft/swin-base-patch4-window7-224',
        language_model='microsoft/DialoGPT-medium',
        num_classes=14,
        hidden_dim=512,
        dropout_rate=0.2,
        use_gradient_checkpointing=False,
        enable_uncertainty=True,
        adaptive_loss_weights=True
    )
    
    # High-performance model configuration
    performance_model_config = ModelConfig(
        vision_model='microsoft/swin-large-patch4-window7-224',
        language_model='microsoft/DialoGPT-large',
        num_classes=14,
        hidden_dim=768,
        dropout_rate=0.3,
        use_gradient_checkpointing=False,
        enable_uncertainty=True,
        adaptive_loss_weights=True
    )
    
    return {
        'lightweight': lightweight_model_config,
        'standard': standard_model_config,
        'performance': performance_model_config
    }

def main():
    """Main function to demonstrate configurations"""
    print("Starting EVLMs Configuration Demonstration...")
    
    # Show memory configurations
    demonstrate_memory_configurations()
    
    # Show model configurations
    print("\n" + "=" * 60)
    print("Model Configuration Examples")
    print("=" * 60)
    
    model_configs = create_model_configs()
    
    for name, config in model_configs.items():
        print(f"\n--- {name.title()} Model Configuration ---")
        print(f"Vision Model: {config.vision_model}")
        print(f"Language Model: {config.language_model}")
        print(f"Hidden Dim: {config.hidden_dim}")
        print(f"Dropout Rate: {config.dropout_rate}")
        print(f"Gradient Checkpointing: {config.use_gradient_checkpointing}")
        print(f"Uncertainty Estimation: {config.enable_uncertainty}")
        print(f"Adaptive Loss Weights: {config.adaptive_loss_weights}")
    
    print("\n" + "=" * 60)
    print("Complete Training Example")
    print("=" * 60)
    
    print("""
# Complete example for medium memory GPU
from EVLMs.models.medical_vlm_improved import ImprovedExplainableMedicalVLM
from EVLMs.trainers.trainer_improved import ImprovedMedicalVLMTrainer

# Create model
model = ImprovedExplainableMedicalVLM(standard_model_config)

# Create trainer with appropriate memory configuration
trainer = ImprovedMedicalVLMTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=medium_memory_config,
    device='cuda',
    output_dir='outputs/training'
)

# Start training
trainer.train()
    """)

if __name__ == "__main__":
    main()