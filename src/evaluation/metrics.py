"""
Evaluation metrics for model performance assessment.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None
) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None
) -> Dict[str, Any]:
    """
    Complete model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        
    Returns:
        Dictionary with metrics, confusion matrix, and report
    """
    metrics = calculate_metrics(y_true, y_pred)
    cm = get_confusion_matrix(y_true, y_pred)
    report = get_classification_report(y_true, y_pred, target_names)
    
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    
    return results
