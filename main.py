import os
import torch
from torch.utils.data import DataLoader
import wandb
from EVLMs.models.medical_vlm import ExplainableMedicalVLM
from EVLMs.data.dataset import MedicalImageTextDataset
from EVLMs.trainers.trainer import MedicalVLMTrainer
from EVLMs.configs.config import get_config, DatasetName
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

def main():
    # Load configuration
    config = get_config(DatasetName.CHEXPERT)
    
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
        dataset_config=config.datasets[config.dataset_name],
        tokenizer=tokenizer,
        split="train",
        max_length=config.max_text_length,
        img_size=config.image_size,
        transform=get_transforms(config.image_size, is_training=True)
    )
    
    val_dataset = MedicalImageTextDataset(
        dataset_config=config.datasets[config.dataset_name],
        tokenizer=tokenizer,
        split="val",
        max_length=config.max_text_length,
        img_size=config.image_size,
        transform=get_transforms(config.image_size, is_training=False)
    )
    
    # # Create data loaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = ExplainableMedicalVLM(
        vision_model=config.vision_model_name,
        language_model=config.language_model_name,
        num_classes=config.num_classes
    ).to(device)
    
    # Initialize trainer
    trainer = MedicalVLMTrainer(
        model=model,
        train_dataset=train_dataset,  # FIX: pass dataset, not DataLoader
        val_dataset=val_dataset,      # FIX: pass dataset, not DataLoader
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
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