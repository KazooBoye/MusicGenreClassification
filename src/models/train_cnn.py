"""
CNN model training script for music genre classification.
Uses mel-spectrograms as input from preprocessed data.
"""

import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.cnn_model import CNNModel
from src.models.utils import load_config, save_results, setup_logging
from src.data.dataset_loader import DatasetLoader
from src.features.extract_melspectrogram import extract_melspectrogram_normalized, pad_melspectrogram
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import plot_confusion_matrix, plot_training_history

logger = logging.getLogger(__name__)


class MelSpectrogramDataset(Dataset):
    """Dataset for mel-spectrograms."""
    
    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.spectrograms[idx]), torch.LongTensor([self.labels[idx]])[0]


def load_preprocessed_segments(config):
    """Load preprocessed audio segments from data/processed/."""
    processed_dir = Path(config['paths']['data_processed'])
    
    if not processed_dir.exists() or not any(processed_dir.iterdir()):
        raise FileNotFoundError(
            f"Preprocessed audio not found in {processed_dir}.\n"
            f"Please run: python src/data/preprocess_dataset.py"
        )
    
    segment_files = list(processed_dir.rglob('*.npy'))
    
    if len(segment_files) == 0:
        raise FileNotFoundError(
            f"No preprocessed segments found in {processed_dir}.\n"
            f"Please run: python src/data/preprocess_dataset.py"
        )
    
    logger.info(f"Found {len(segment_files)} preprocessed segments")
    return segment_files


def extract_melspec_from_segment(segment_path, config, target_length):
    """Extract mel-spectrogram from preprocessed segment."""
    audio_segment = np.load(segment_path)
    sr = config['audio']['sample_rate']
    
    melspec = extract_melspectrogram_normalized(
        audio_segment, sr,
        n_mels=config['features']['melspectrogram']['n_mels'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        fmin=config['features']['melspectrogram']['fmin'],
        fmax=config['features']['melspectrogram']['fmax']
    )
    
    if target_length:
        melspec = pad_melspectrogram(melspec, target_length)
    
    return melspec


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
        log_file=f"{config['paths']['logs']}/train_cnn.log",
        level=config['logging']['level']
    )
    
    logger.info("Starting CNN model training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        segment_files = load_preprocessed_segments(config)
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n{e}")
        return
    
    dataset_loader = DatasetLoader(config['paths']['data_raw'])
    
    sample_sr = config['audio']['sample_rate']
    segment_length = config['audio']['segment_length']
    hop_length = config['audio']['hop_length']
    target_length = int((segment_length * sample_sr) / hop_length) + 1
    
    logger.info(f"Extracting mel-spectrograms from {len(segment_files)} segments")
    
    X = []
    y = []
    
    for segment_path in tqdm(segment_files, desc="Processing segments"):
        try:
            genre = segment_path.parent.name
            label = dataset_loader.genre_to_idx[genre]
            
            melspec = extract_melspec_from_segment(segment_path, config, target_length)
            X.append(melspec)
            y.append(label)
            
        except Exception as e:
            logger.warning(f"Error processing {segment_path}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Mel-spectrogram shape: {X.shape}")
    
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
    
    train_dataset = MelSpectrogramDataset(X_train, y_train)
    val_dataset = MelSpectrogramDataset(X_val, y_val)
    test_dataset = MelSpectrogramDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['models']['cnn']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['models']['cnn']['batch_size'],
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['models']['cnn']['batch_size'],
        num_workers=0
    )
    
    model = CNNModel(
        num_classes=dataset_loader.get_num_classes(),
        conv_channels=config['models']['cnn']['conv_channels'],
        kernel_size=config['models']['cnn']['kernel_size'],
        pool_size=config['models']['cnn']['pool_size'],
        dense_units=config['models']['cnn']['dense_units'],
        dropout=config['models']['cnn']['dropout']
    )
    
    input_shape = (1, X_train.shape[1], X_train.shape[2])
    model.build_fc_layers(input_shape, dataset_loader.get_num_classes())
    model = model.to(device)
    
    logger.info(f"Model architecture:\n{model}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['models']['cnn']['learning_rate']
    )
    
    logger.info("Training CNN model")
    
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['models']['cnn']['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{config['models']['cnn']['epochs']} - "
                   f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{config['paths']['models']}/cnn_best.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= config['models']['cnn']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(f"{config['paths']['models']}/cnn_best.pth"))
    
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
    
    model_path = f"{config['paths']['models']}/cnn.pth"
    results_path = f"{config['paths']['outputs']}/cnn_results.json"
    cm_plot_path = f"{config['paths']['plots']}/cnn_confusion_matrix.png"
    history_plot_path = f"{config['paths']['plots']}/cnn_training_history.png"
    
    torch.save(model.state_dict(), model_path)
    save_results(results, results_path)
    
    config_save = {
        'input_shape': input_shape,
        'num_classes': dataset_loader.get_num_classes(),
        'conv_channels': config['models']['cnn']['conv_channels'],
        'kernel_size': config['models']['cnn']['kernel_size'],
        'pool_size': config['models']['cnn']['pool_size'],
        'dense_units': config['models']['cnn']['dense_units'],
        'dropout': config['models']['cnn']['dropout']
    }
    save_results(config_save, f"{config['paths']['models']}/cnn_config.json")
    
    plot_confusion_matrix(
        np.array(results['confusion_matrix']),
        dataset_loader.genres,
        title='CNN Confusion Matrix',
        save_path=cm_plot_path
    )
    
    plot_training_history(
        history,
        title='CNN Training History',
        save_path=history_plot_path
    )
    
    logger.info("CNN training completed successfully")


if __name__ == '__main__':
    main()
