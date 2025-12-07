"""
Visualization utilities for model evaluation and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = 'Confusion Matrix',
    save_path: str = None,
    figsize: tuple = (10, 8)
):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_metrics_comparison(
    results: dict,
    save_path: str = None,
    figsize: tuple = (12, 6)
):
    """
    Plot comparison of metrics across models.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        save_path: Path to save figure
        figsize: Figure size
    """
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    fig, axes = plt.subplots(1, len(metrics_names), figsize=figsize)
    
    for idx, metric in enumerate(metrics_names):
        values = [results[model]['metrics'][metric] for model in models]
        axes[idx].bar(models, values, color=sns.color_palette("husl", len(models)))
        axes[idx].set_title(metric.replace('_', ' ').title(), fontweight='bold')
        axes[idx].set_ylim([0, 1])
        axes[idx].set_ylabel('Score')
        axes[idx].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison saved to {save_path}")
    
    plt.close()


def plot_training_history(
    history: dict,
    title: str = 'Training History',
    save_path: str = None,
    figsize: tuple = (12, 5)
):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    if 'loss' in history:
        ax1.plot(history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Model Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
    
    plt.close()


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    feature_idx: int = 0,
    feature_name: str = 'Feature',
    save_path: str = None,
    figsize: tuple = (10, 6)
):
    """
    Plot distribution of a specific feature across classes.
    
    Args:
        features: Feature matrix
        labels: Class labels
        class_names: List of class names
        feature_idx: Index of feature to plot
        feature_name: Name of feature
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.hist(
            features[mask, feature_idx],
            alpha=0.6,
            label=class_name,
            bins=30
        )
    
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of {feature_name} across Genres', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature distribution saved to {save_path}")
    
    plt.close()
