"""
Gradio UI for music genre classification inference.
Provides interactive interface for model demonstration.
"""

import sys
import numpy as np
from pathlib import Path
import torch
import gradio as gr
import json
import tempfile

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.utils import load_config, load_model
from src.models.cnn_model import CNNModel
from src.models.train_gmm import GMMClassifier
from src.data.dataset_loader import DatasetLoader
from src.data.preprocess_audio import preprocess_audio
from src.data.noise_reduction import noise_reduction_pipeline
from src.features.extract_mfcc_features import extract_mfcc_features
from src.features.extract_spectral_features import extract_all_spectral_features
from src.features.extract_melspectrogram import extract_melspectrogram_normalized, pad_melspectrogram


class GenreClassifierUI:
    def __init__(self, config_path='../../config.yaml'):
        self.config = load_config(config_path)
        
        # Get project root directory (2 levels up from src/inference)
        self.project_root = Path(__file__).parent.parent.parent
        
        # Convert relative paths in config to absolute paths
        data_raw_path = self.project_root / self.config['paths']['data_raw']
        self.dataset_loader = DatasetLoader(str(data_raw_path))
        
        self.models_path = self.project_root / self.config['paths']['models']
        self.models = {}
        self.scalers = {}
        
    def load_models(self):
        """Preload all models for faster inference."""
        model_names = ['knn', 'svm', 'rf', 'gmm']
        for name in model_names:
            try:
                model_path = self.models_path / f"{name}.pkl"
                scaler_path = self.models_path / f"{name}_scaler.pkl"
                self.models[name] = load_model(str(model_path))
                self.scalers[name] = load_model(str(scaler_path))
                print(f"Successfully loaded {name.upper()} model")
            except Exception as e:
                print(f"Warning: Could not load {name} model: {e}")
                import traceback
                traceback.print_exc()
        
        # Load MLP
        try:
            self.load_mlp_model()
        except Exception as e:
            print(f"Warning: Could not load MLP model: {e}")
            
        # Load CNN
        try:
            self.load_cnn_model()
        except Exception as e:
            print(f"Warning: Could not load CNN model: {e}")
    
    def load_mlp_model(self):
        """Load MLP model."""
        model_path = self.models_path / "mlp.pth"
        scaler_path = self.models_path / "mlp_scaler.pkl"
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalers['mlp'] = load_model(scaler_path)
        
        from src.models.train_mlp import MLPModel
        num_classes = self.dataset_loader.get_num_classes()
        
        # Get input size from scaler
        input_size = 126  # Default from feature extraction
        
        model = MLPModel(
            input_size=input_size,
            hidden_layers=self.config['models']['mlp']['hidden_layers'],
            num_classes=num_classes,
            dropout=self.config['models']['mlp']['dropout']
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        self.models['mlp'] = model
        self.device = device
        
    def load_cnn_model(self):
        """Load CNN model."""
        model_path = self.models_path / "cnn.pth"
        model_config_path = self.models_path / "cnn_config.json"
        
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        self.models['cnn'] = model
        self.cnn_config = model_config
        self.device = device
        
    def extract_classical_features(self, audio_file):
        """Extract features for classical ML models."""
        segments, sr = preprocess_audio(
            audio_file,
            sr=self.config['audio']['sample_rate'],
            segment_length=self.config['audio']['segment_length'],
            mono=self.config['audio']['mono']
        )
        
        if self.config['noise_reduction']['enable']:
            segments = [
                noise_reduction_pipeline(
                    segment, sr,
                    enable=True,
                    highpass_cutoff=self.config['noise_reduction']['highpass_cutoff'],
                    gate_threshold=self.config['noise_reduction']['spectral_gate_threshold']
                )
                for segment in segments
            ]
        
        features_list = []
        for segment in segments:
            mfcc_features = extract_mfcc_features(
                segment, sr,
                n_mfcc=self.config['features']['mfcc']['n_mfcc'],
                n_fft=self.config['audio']['n_fft'],
                hop_length=self.config['audio']['hop_length'],
                n_mels=self.config['features']['mfcc']['n_mels']
            )
            
            spectral_features = extract_all_spectral_features(
                segment, sr,
                n_fft=self.config['audio']['n_fft'],
                hop_length=self.config['audio']['hop_length'],
                n_chroma=self.config['features']['chroma']['n_chroma'],
                n_bands=self.config['features']['spectral']['n_bands']
            )
            
            combined_features = np.concatenate([mfcc_features, spectral_features])
            features_list.append(combined_features)
        
        return np.mean(features_list, axis=0)
    
    def extract_cnn_features(self, audio_file):
        """Extract mel-spectrograms for CNN model."""
        segments, sr = preprocess_audio(
            audio_file,
            sr=self.config['audio']['sample_rate'],
            segment_length=self.config['audio']['segment_length'],
            mono=self.config['audio']['mono']
        )
        
        if self.config['noise_reduction']['enable']:
            segments = [
                noise_reduction_pipeline(
                    segment, sr,
                    enable=True,
                    highpass_cutoff=self.config['noise_reduction']['highpass_cutoff'],
                    gate_threshold=self.config['noise_reduction']['spectral_gate_threshold']
                )
                for segment in segments
            ]
        
        sample_sr = self.config['audio']['sample_rate']
        segment_length = self.config['audio']['segment_length']
        hop_length = self.config['audio']['hop_length']
        target_length = int((segment_length * sample_sr) / hop_length) + 1
        
        melspecs = []
        for segment in segments:
            melspec = extract_melspectrogram_normalized(
                segment, sr,
                n_mels=self.config['features']['melspectrogram']['n_mels'],
                n_fft=self.config['audio']['n_fft'],
                hop_length=self.config['audio']['hop_length'],
                fmin=self.config['features']['melspectrogram']['fmin'],
                fmax=self.config['features']['melspectrogram']['fmax']
            )
            melspec = pad_melspectrogram(melspec, target_length)
            melspecs.append(melspec)
        
        return np.array(melspecs)
    
    def predict(self, audio_file, model_name):
        """Make prediction using selected model."""
        if audio_file is None:
            return "Please provide an audio file or record audio.", {}
        
        try:
            if model_name in ['knn', 'svm', 'rf', 'gmm']:
                return self.predict_classical(audio_file, model_name)
            elif model_name == 'mlp':
                return self.predict_mlp(audio_file)
            elif model_name == 'cnn':
                return self.predict_cnn(audio_file)
            else:
                return f"Model {model_name} not supported.", {}
        except Exception as e:
            return f"Error during prediction: {str(e)}", {}
    
    def predict_classical(self, audio_file, model_name):
        """Predict using classical ML models."""
        model = self.models.get(model_name)
        scaler = self.scalers.get(model_name)
        
        if model is None or scaler is None:
            return f"Model {model_name} not loaded.", {}
        
        features = self.extract_classical_features(audio_file)
        features = scaler.transform(features.reshape(1, -1))
        
        prediction = model.predict(features)[0]
        predicted_genre = self.dataset_loader.idx_to_genre[prediction]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            prob_dict = {
                self.dataset_loader.idx_to_genre[idx]: float(prob)
                for idx, prob in enumerate(probabilities)
            }
        else:
            prob_dict = {predicted_genre: 1.0}
        
        result_text = f"Predicted Genre: {predicted_genre}\n\n"
        result_text += f"Model: {model_name.upper()}\n"
        result_text += f"Confidence: {max(prob_dict.values()):.2%}"
        
        return result_text, prob_dict
    
    def predict_mlp(self, audio_file):
        """Predict using MLP model."""
        model = self.models.get('mlp')
        scaler = self.scalers.get('mlp')
        
        if model is None or scaler is None:
            return "MLP model not loaded.", {}
        
        features = self.extract_classical_features(audio_file)
        features = scaler.transform(features.reshape(1, -1))
        
        with torch.no_grad():
            inputs = torch.FloatTensor(features).to(self.device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = outputs.argmax(dim=1).cpu().numpy()[0]
        
        predicted_genre = self.dataset_loader.idx_to_genre[prediction]
        prob_dict = {
            self.dataset_loader.idx_to_genre[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }
        
        result_text = f"Predicted Genre: {predicted_genre}\n\n"
        result_text += f"Model: MLP (Deep Learning)\n"
        result_text += f"Confidence: {max(prob_dict.values()):.2%}"
        
        return result_text, prob_dict
    
    def predict_cnn(self, audio_file):
        """Predict using CNN model."""
        model = self.models.get('cnn')
        
        if model is None:
            return "CNN model not loaded.", {}
        
        melspecs = self.extract_cnn_features(audio_file)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for melspec in melspecs:
                inputs = torch.FloatTensor(melspec).unsqueeze(0).to(self.device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                prediction = outputs.argmax(dim=1).cpu().numpy()[0]
                all_predictions.append(prediction)
                all_probabilities.append(probabilities)
        
        final_prediction = np.bincount(all_predictions).argmax()
        final_probabilities = np.mean(all_probabilities, axis=0)
        
        predicted_genre = self.dataset_loader.idx_to_genre[final_prediction]
        prob_dict = {
            self.dataset_loader.idx_to_genre[idx]: float(prob)
            for idx, prob in enumerate(final_probabilities)
        }
        
        result_text = f"Predicted Genre: {predicted_genre}\n\n"
        result_text += f"Model: CNN (Deep Learning)\n"
        result_text += f"Processed {len(melspecs)} segments\n"
        result_text += f"Confidence: {max(prob_dict.values()):.2%}"
        
        return result_text, prob_dict


def create_ui():
    """Create Gradio interface."""
    classifier = GenreClassifierUI()
    
    print("Loading models...")
    classifier.load_models()
    print("Models loaded successfully!")
    
    with gr.Blocks(title="Music Genre Classification", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Music Genre Classification
            
            Classify music into 10 genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
            
            Choose a model and either upload an audio file or record from your microphone.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=['knn', 'svm', 'rf', 'gmm', 'mlp', 'cnn'],
                    value='cnn',
                    label="Select Model",
                    info="Choose which model to use for prediction"
                )
                
                gr.Markdown("### Input Audio")
                
                with gr.Tabs():
                    with gr.TabItem("Upload File"):
                        audio_file = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload"]
                        )
                        
                    with gr.TabItem("Record Audio"):
                        audio_record = gr.Audio(
                            label="Record Audio",
                            type="filepath",
                            sources=["microphone"]
                        )
                
                predict_button = gr.Button("Classify Genre", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ### Model Information
                    
                    - **KNN**: K-Nearest Neighbors (distance-based)
                    - **SVM**: Support Vector Machine (kernel-based)
                    - **RF**: Random Forest (ensemble)
                    - **GMM**: Gaussian Mixture Model (probabilistic)
                    - **MLP**: Multi-Layer Perceptron (neural network)
                    - **CNN**: Convolutional Neural Network (deep learning)
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                
                result_text = gr.Textbox(
                    label="Prediction",
                    lines=6,
                    interactive=False
                )
                
                probability_plot = gr.Label(
                    label="Genre Probabilities",
                    num_top_classes=10
                )
                
                gr.Markdown(
                    """
                    ### How to Use
                    
                    1. Select a model from the dropdown menu
                    2. Either upload an audio file or record audio using your microphone
                    3. Click "Classify Genre" to get the prediction
                    4. View the predicted genre and confidence scores
                    
                    **Supported formats**: WAV, MP3, OGG, FLAC
                    
                    **Note**: Longer audio files will be automatically segmented into 3-second chunks for analysis.
                    """
                )
        
        def classify_audio(audio_file_input, audio_record_input, model_name):
            # Use whichever input is provided
            audio_input = audio_file_input if audio_file_input is not None else audio_record_input
            return classifier.predict(audio_input, model_name)
        
        predict_button.click(
            fn=classify_audio,
            inputs=[audio_file, audio_record, model_dropdown],
            outputs=[result_text, probability_plot]
        )
        
        gr.Markdown(
            """
            ---
            
            **Project**: Music Genre Classification using Machine Learning and Deep Learning
            
            **Dataset**: GTZAN Genre Collection (10 genres, 999 tracks)
            """
        )
    
    return demo


def main():
    demo = create_ui()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == '__main__':
    main()
