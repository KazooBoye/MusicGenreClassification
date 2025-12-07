"""
Utility functions for model training and management.
"""

import yaml
import json
import numpy as np
import logging
from pathlib import Path
from typing import Any, Dict
import joblib

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_model(model: Any, save_path: str):
    """
    Save model to file.
    
    Args:
        model: Model object to save
        save_path: Path to save model
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(model_path: str) -> Any:
    """
    Load model from file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model object
    """
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def save_results(results: Dict[str, Any], save_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        save_path: Path to save results
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    serializable_results = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {save_path}")


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Results dictionary
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def setup_logging(log_file: str = None, level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        handlers=handlers
    )


def aggregate_segment_predictions(predictions: np.ndarray, method: str = 'vote') -> int:
    """
    Aggregate predictions from multiple segments.
    
    Args:
        predictions: Array of predictions for each segment
        method: Aggregation method ('vote', 'mean')
        
    Returns:
        Final prediction
    """
    if method == 'vote':
        return np.bincount(predictions).argmax()
    elif method == 'mean':
        return int(np.round(np.mean(predictions)))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
