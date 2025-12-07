"""
Inference script for music genre classification.
Supports all trained models: KNN, SVM, Random Forest, MLP, GMM, CNN.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import torch
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.utils import load_config, load_model, setup_logging
from src.models.cnn_model import CNNModel
from src.data.dataset_loader import DatasetLoader
from src.data.preprocess_audio import preprocess_audio
from src.data.noise_reduction import noise_reduction_pipeline
from src.features.extract_mfcc_features import extract_mfcc_features
from src.features.extract_spectral_features import extract_all_spectral_features
from src.features.extract_melspectrogram import extract_melspectrogram_normalized, pad_melspectrogram

logger = logging.getLogger(__name__)


def extract_classical_features(audio_file, config, apply_noise_reduction=True):
    """Extract features for classical ML models."""
    segments, sr = preprocess_audio(
        audio_file,
        sr=config['audio']['sample_rate'],
        segment_length=config['audio']['segment_length'],
        mono=config['audio']['mono']
    )
    
    if apply_noise_reduction and config['noise_reduction']['enable']:
        segments = [
            noise_reduction_pipeline(
                segment, sr,
                enable=True,
                highpass_cutoff=config['noise_reduction']['highpass_cutoff'],
                gate_threshold=config['noise_reduction']['spectral_gate_threshold']
            )
            for segment in segments
        ]
    
    features_list = []
    for segment in segments:
        mfcc_features = extract_mfcc_features(
            segment, sr,
            n_mfcc=config['features']['mfcc']['n_mfcc'],
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            n_mels=config['features']['mfcc']['n_mels']
        )
        
        spectral_features = extract_all_spectral_features(
            segment, sr,
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            n_chroma=config['features']['chroma']['n_chroma'],
            n_bands=config['features']['spectral']['n_bands']
        )
        
        combined_features = np.concatenate([mfcc_features, spectral_features])
        features_list.append(combined_features)
    
    return np.mean(features_list, axis=0)


def extract_cnn_features(audio_file, config, apply_noise_reduction=True):
    """Extract mel-spectrograms for CNN model."""
    segments, sr = preprocess_audio(
        audio_file,
        sr=config['audio']['sample_rate'],
        segment_length=config['audio']['segment_length'],
        mono=config['audio']['mono']
    )
    
    if apply_noise_reduction and config['noise_reduction']['enable']:
        segments = [
            noise_reduction_pipeline(
                segment, sr,
                enable=True,
                highpass_cutoff=config['noise_reduction']['highpass_cutoff'],
                gate_threshold=config['noise_reduction']['spectral_gate_threshold']
            )
            for segment in segments
        ]
    
    sample_sr = config['audio']['sample_rate']
    segment_length = config['audio']['segment_length']
    hop_length = config['audio']['hop_length']
    target_length = int((segment_length * sample_sr) / hop_length) + 1
    
    melspecs = []
    for segment in segments:
        melspec = extract_melspectrogram_normalized(
            segment, sr,
            n_mels=config['features']['melspectrogram']['n_mels'],
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            fmin=config['features']['melspectrogram']['fmin'],
            fmax=config['features']['melspectrogram']['fmax']
        )
        melspec = pad_melspectrogram(melspec, target_length)
        melspecs.append(melspec)
    
    return np.array(melspecs)


def predict_classical(model_name, audio_file, config):
    """Predict using classical ML models (KNN, SVM, RF, GMM)."""
    model_path = f"{config['paths']['models']}/{model_name}.pkl"
    scaler_path = f"{config['paths']['models']}/{model_name}_scaler.pkl"
    
    model = load_model(model_path)
    scaler = load_model(scaler_path)
    
    features = extract_classical_features(audio_file, config)
    features = scaler.transform(features.reshape(1, -1))
    
    prediction = model.predict(features)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
    else:
        probabilities = None
    
    return prediction, probabilities


def predict_mlp(audio_file, config):
    """Predict using MLP model."""
    model_path = f"{config['paths']['models']}/mlp.pth"
    scaler_path = f"{config['paths']['models']}/mlp_scaler.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = load_model(scaler_path)
    features = extract_classical_features(audio_file, config)
    features = scaler.transform(features.reshape(1, -1))
    
    dataset_loader = DatasetLoader(config['paths']['data_raw'])
    num_classes = dataset_loader.get_num_classes()
    
    from src.models.train_mlp import MLPModel
    model = MLPModel(
        input_size=features.shape[1],
        hidden_layers=config['models']['mlp']['hidden_layers'],
        num_classes=num_classes,
        dropout=config['models']['mlp']['dropout']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = torch.FloatTensor(features).to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        prediction = outputs.argmax(dim=1).cpu().numpy()[0]
    
    return prediction, probabilities


def predict_cnn(audio_file, config):
    """Predict using CNN model."""
    model_path = f"{config['paths']['models']}/cnn.pth"
    model_config_path = f"{config['paths']['models']}/cnn_config.json"
    
    import json
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    melspecs = extract_cnn_features(audio_file, config)
    
    model = CNNModel(
        num_classes=model_config['num_classes'],
        conv_channels=model_config['conv_channels'],
        kernel_size=model_config['kernel_size'],
        pool_size=model_config['pool_size'],
        dense_units=model_config['dense_units'],
        dropout=model_config['dropout']
    )
    model.build_fc_layers(tuple(model_config['input_shape']), model_config['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for melspec in melspecs:
            inputs = torch.FloatTensor(melspec).unsqueeze(0).to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = outputs.argmax(dim=1).cpu().numpy()[0]
            all_predictions.append(prediction)
            all_probabilities.append(probabilities)
    
    final_prediction = np.bincount(all_predictions).argmax()
    final_probabilities = np.mean(all_probabilities, axis=0)
    
    return final_prediction, final_probabilities


def main():
    parser = argparse.ArgumentParser(description='Music Genre Classification Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['knn', 'svm', 'rf', 'mlp', 'gmm', 'cnn'],
                       help='Model to use for prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(level=config['logging']['level'])
    
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Processing audio: {args.audio}")
    
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        return
    
    dataset_loader = DatasetLoader(config['paths']['data_raw'])
    
    if args.model in ['knn', 'svm', 'rf', 'gmm']:
        prediction, probabilities = predict_classical(args.model, args.audio, config)
    elif args.model == 'mlp':
        prediction, probabilities = predict_mlp(args.audio, config)
    elif args.model == 'cnn':
        prediction, probabilities = predict_cnn(args.audio, config)
    
    predicted_genre = dataset_loader.idx_to_genre.get(prediction, f"Class {prediction}")
    
    logger.info(f"\nPredicted Genre: {predicted_genre}")
    
    if probabilities is not None:
        logger.info("\nGenre Probabilities:")
        for idx, prob in enumerate(probabilities):
            genre = dataset_loader.idx_to_genre.get(idx, f"Class {idx}")
            logger.info(f"  {genre}: {prob:.4f}")
    
    print(f"\nPredicted Genre: {predicted_genre}")
    if probabilities is not None:
        print("\nGenre Probabilities:")
        for idx, prob in enumerate(probabilities):
            genre = dataset_loader.idx_to_genre.get(idx, f"Class {idx}")
            print(f"  {genre}: {prob:.4f}")


if __name__ == '__main__':
    main()
