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

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from typing import List, Dict

def calculate_language_metrics(generated: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate language generation metrics (BLEU and ROUGE)."""
    
    # Use a smoothing function to avoid zero scores for short sentences
    chencherry = SmoothingFunction().method1

    bleu_scores = {
        'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []
    }
    
    for gen, ref in zip(generated, references):
        ref_tokens = [ref.split()]
        gen_tokens = gen.split()
        
        bleu_scores['bleu1'].append(sentence_bleu(ref_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=chencherry))
        bleu_scores['bleu2'].append(sentence_bleu(ref_tokens, gen_tokens, weights=(0, 1, 0, 0), smoothing_function=chencherry))
        bleu_scores['bleu3'].append(sentence_bleu(ref_tokens, gen_tokens, weights=(0, 0, 1, 0), smoothing_function=chencherry))
        bleu_scores['bleu4'].append(sentence_bleu(ref_tokens, gen_tokens, weights=(0, 0, 0, 1), smoothing_function=chencherry))

    # Average BLEU scores
    avg_bleu = {k: sum(v) / len(v) for k, v in bleu_scores.items()}
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rougeL': []}
    
    for gen, ref in zip(generated, references):
        scores = scorer.score(ref, gen)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
    avg_rouge = {k: sum(v) / len(v) for k, v in rouge_scores.items()}
    
    return {**avg_bleu, **avg_rouge} 