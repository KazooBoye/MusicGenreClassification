"""
SVM model training script for music genre classification.
Loads preprocessed features from data/features/ directory.
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.utils import load_config, save_model, save_results, setup_logging
from src.data.dataset_loader import DatasetLoader
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import plot_confusion_matrix

logger = logging.getLogger(__name__)


def load_preprocessed_features(config):
    """Load features from data/features/ directory."""
    features_dir = Path(config['paths']['data_features'])
    
    features_path = features_dir / 'features.npy'
    labels_path = features_dir / 'labels.npy'
    
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Preprocessed features not found in {features_dir}.\n"
            f"Please run: python src/data/extract_all_features.py"
        )
    
    X = np.load(features_path)
    y = np.load(labels_path)
    
    logger.info(f"Loaded preprocessed features from {features_dir}")
    return X, y


def main():
    config = load_config('config.yaml')
    
    setup_logging(
        log_file=f"{config['paths']['logs']}/train_svm.log",
        level=config['logging']['level']
    )
    
    logger.info("Starting SVM model training")
    
    try:
        X, y = load_preprocessed_features(config)
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n{e}")
        return
    
    dataset_loader = DatasetLoader(config['paths']['data_raw'])
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=y if config['training']['stratify'] else None
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info("Training SVM model")
    model = SVC(
        C=config['models']['svm']['C'],
        gamma=config['models']['svm']['gamma'],
        kernel=config['models']['svm']['kernel'],
        max_iter=config['models']['svm']['max_iter'],
        probability=True,
        random_state=config['training']['random_state']
    )
    
    model.fit(X_train, y_train)
    
    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    
    results = evaluate_model(y_test, y_pred, dataset_loader.genres)
    
    logger.info(f"\nClassification Report:\n{results['classification_report']}")
    
    model_path = f"{config['paths']['models']}/svm.pkl"
    scaler_path = f"{config['paths']['models']}/svm_scaler.pkl"
    results_path = f"{config['paths']['outputs']}/svm_results.json"
    cm_plot_path = f"{config['paths']['plots']}/svm_confusion_matrix.png"
    
    save_model(model, model_path)
    save_model(scaler, scaler_path)
    save_results(results, results_path)
    
    plot_confusion_matrix(
        np.array(results['confusion_matrix']),
        dataset_loader.genres,
        title='SVM Confusion Matrix',
        save_path=cm_plot_path
    )
    
    logger.info("SVM training completed successfully")
    print(f"\nTraining complete!")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1-Score: {results['metrics']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
