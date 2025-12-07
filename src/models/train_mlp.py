"""
MLP model training script using PyTorch.
Loads preprocessed features from data/features/ directory.
"""

import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.utils import load_config, save_model, save_results, setup_logging
from src.data.dataset_loader import DatasetLoader
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import plot_confusion_matrix, plot_training_history

logger = logging.getLogger(__name__)


class MLPModel(nn.Module):
    """Multi-Layer Perceptron for genre classification."""
    
    def __init__(self, input_size, hidden_layers, num_classes, dropout=0.3):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), correct / total


def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), correct / total


def main():
    config = load_config('config.yaml')
    
    setup_logging(
        log_file=f"{config['paths']['logs']}/train_mlp.log",
        level=config['logging']['level']
    )
    
    logger.info("Starting MLP model training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        X, y = load_preprocessed_features(config)
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n{e}")
        return
    
    dataset_loader = DatasetLoader(config['paths']['data_raw'])
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=config['training']['test_size'] + config['training']['val_size'],
        random_state=config['training']['random_state'],
        stratify=y if config['training']['stratify'] else None
    )
    
    val_ratio = config['training']['val_size'] / (config['training']['test_size'] + config['training']['val_size'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio,
        random_state=config['training']['random_state'],
        stratify=y_temp if config['training']['stratify'] else None
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['models']['mlp']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['models']['mlp']['batch_size']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['models']['mlp']['batch_size']
    )
    
    model = MLPModel(
        input_size=X_train.shape[1],
        hidden_layers=config['models']['mlp']['hidden_layers'],
        num_classes=dataset_loader.get_num_classes(),
        dropout=config['models']['mlp']['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['models']['mlp']['learning_rate']
    )
    
    logger.info("Training MLP model")
    
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['models']['mlp']['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{config['models']['mlp']['epochs']} - "
                   f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{config['paths']['models']}/mlp_best.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= config['models']['mlp']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(f"{config['paths']['models']}/mlp_best.pth"))
    
    logger.info("Evaluating model")
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(targets.numpy())
    
    results = evaluate_model(np.array(y_true), np.array(y_pred), dataset_loader.genres)
    
    logger.info(f"\nClassification Report:\n{results['classification_report']}")
    
    model_path = f"{config['paths']['models']}/mlp.pth"
    scaler_path = f"{config['paths']['models']}/mlp_scaler.pkl"
    results_path = f"{config['paths']['outputs']}/mlp_results.json"
    cm_plot_path = f"{config['paths']['plots']}/mlp_confusion_matrix.png"
    history_plot_path = f"{config['paths']['plots']}/mlp_training_history.png"
    
    torch.save(model.state_dict(), model_path)
    save_model(scaler, scaler_path)
    save_results(results, results_path)
    
    plot_confusion_matrix(
        np.array(results['confusion_matrix']),
        dataset_loader.genres,
        title='MLP Confusion Matrix',
        save_path=cm_plot_path
    )
    
    plot_training_history(
        history,
        title='MLP Training History',
        save_path=history_plot_path
    )
    
    logger.info("MLP training completed successfully")
    print(f"\nTraining complete!")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1-Score: {results['metrics']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
