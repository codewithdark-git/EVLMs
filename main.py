import os
import torch
import argparse
import wandb
from EVLMs.models.medical_vlm import ExplainableMedicalVLM
from EVLMs.data.dataset import MedicalImageTextDataset
from EVLMs.trainers.trainer import MedicalVLMTrainer
from EVLMs.configs.config import Config, DatasetConfig
from EVLMs.utils.logger import setup_logging
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size: int, is_training: bool = True):
    """Get image transforms for training/validation"""
    if is_training:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set.")

def main():
    parser = argparse.ArgumentParser(description="Train an Explainable Medical VLM.")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the root directory of the dataset.")
    args = parser.parse_args()

    # Create dataset and base configuration
    dataset_config = DatasetConfig(data_dir=args.dataset_path)
    config = Config(dataset=dataset_config)
    
    # Setup logging
    logger = setup_logging(config.output_dir)
    logger.info("Starting EVLMs training...")
    
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project="evlms",
            config=config.__dict__,
            name=config.experiment_name
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets with transforms
    train_dataset = MedicalImageTextDataset(
        dataset_config=config.dataset,
        tokenizer=tokenizer,
        split="train",
        max_length=config.max_text_length,
        transform=get_transforms(config.image_size, is_training=True)
    )
    
    val_dataset = MedicalImageTextDataset(
        dataset_config=config.dataset,
        tokenizer=tokenizer,
        split="val",
        max_length=config.max_text_length,
        transform=get_transforms(config.image_size, is_training=False)
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = ExplainableMedicalVLM(
        vision_model=config.vision_model_name,
        language_model=config.language_model_name,
        num_classes=config.num_classes
    ).to(device)

    # Log model parameters (improved formatting)
    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    vision_params = count_parameters(model.vision_encoder)
    language_params = count_parameters(model.language_decoder)
    total_params = count_parameters(model)

    logger.info("\n========== Model Parameters ==========")
    logger.info(f"Vision Encoder Parameters:    {vision_params:,} ({vision_params / 1e6:.2f}M)")
    logger.info(f"Language Decoder Parameters: {language_params:,} ({language_params / 1e6:.2f}M)")
    logger.info(f"Total Trainable Parameters:  {total_params:,} ({total_params / 1e6:.2f}M)")
    logger.info("======================================\n")
    
    # Initialize trainer
    trainer = MedicalVLMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=config.learning_rate,
        batch_size=4,  # Reduced batch size
        num_epochs=10,  # Reduced epochs for quick test
        device=device,
        output_dir=config.output_dir,
        logger=logger
    )
    
    # Start training
    trainer.train()
    
    # Close wandb
    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
