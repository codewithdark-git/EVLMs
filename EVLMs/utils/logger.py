import logging
import os
from datetime import datetime

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logger
    logger = logging.getLogger('EVLMs')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    log_file = os.path.join(
        output_dir, 
        f'evlm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_metrics(logger: logging.Logger, metrics: dict, step: int, prefix: str = ''):
    """Log metrics to logger and wandb if available"""
    
    # Format metrics string
    metrics_str = ' '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    logger.info(f'{prefix} Step {step}: {metrics_str}')
    
    # Log to wandb if available
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass 