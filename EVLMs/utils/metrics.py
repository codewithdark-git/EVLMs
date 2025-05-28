import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from typing import Dict, List, Tuple

def calculate_metrics(predictions: torch.Tensor, 
                     labels: torch.Tensor, 
                     threshold: float = 0.5) -> Dict[str, float]:
    """Calculate classification metrics
    
    Args:
        predictions: Model predictions (B, num_classes)
        labels: Ground truth labels (B, num_classes)
        threshold: Classification threshold
    
    Returns:
        Dictionary containing metrics
    """
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    
    # Move to CPU and convert to numpy
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Calculate metrics
    metrics = {}
    
    # Binary predictions
    binary_preds = (predictions > threshold).astype(np.int32)
    
    # Accuracy
    metrics['accuracy'] = np.mean(binary_preds == labels)
    
    # Per-class metrics
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:  # Only if both classes present
            # AUC-ROC
            metrics[f'auc_class_{i}'] = roc_auc_score(labels[:, i], predictions[:, i])
            
            # Average Precision
            metrics[f'ap_class_{i}'] = average_precision_score(labels[:, i], predictions[:, i])
    
    # Mean metrics across classes
    metrics['mean_auc'] = np.mean([v for k, v in metrics.items() if k.startswith('auc_')])
    metrics['mean_ap'] = np.mean([v for k, v in metrics.items() if k.startswith('ap_')])
    
    return metrics

def calculate_language_metrics(predictions: List[str], 
                             references: List[str]) -> Dict[str, float]:
    """Calculate language generation metrics
    
    Args:
        predictions: List of generated text
        references: List of reference text
    
    Returns:
        Dictionary containing metrics
    """
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
        import nltk
        nltk.download('punkt', quiet=True)
    except ImportError:
        print("Please install rouge-score and nltk for language metrics")
        return {}
    
    metrics = {}
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(pred, ref) for pred, ref in zip(predictions, references)]
    
    metrics['rouge1_f'] = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    metrics['rouge2_f'] = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    metrics['rougeL_f'] = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    
    # BLEU scores
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        try:
            bleu = sentence_bleu([ref_tokens], pred_tokens)
            bleu_scores.append(bleu)
        except:
            continue
    
    if bleu_scores:
        metrics['bleu'] = np.mean(bleu_scores)
    
    return metrics 