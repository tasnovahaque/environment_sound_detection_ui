import os
import logging
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the class names to match your model
CLASSES = ['Bus', 'Metro', 'Metro_Station', 'Park', 'Restaurant', 'Shopping_Mall', 'University']

def preprocess_audio(audio_path, max_length=16000*10):  # 10 seconds at 16kHz
    """Preprocess audio file for Wav2Vec2 model using librosa"""
    try:
        logger.info(f"Preprocessing audio file: {audio_path}")
        
        # Load audio using librosa (handles various formats)
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        # Ensure correct shape for the model
        waveform = np.asarray(waveform)
        
        # Truncate or pad to max_length
        if len(waveform) > max_length:
            logger.info(f"Truncating audio from {len(waveform)} to {max_length} samples")
            waveform = waveform[:max_length]
        elif len(waveform) < max_length:
            logger.info(f"Padding audio from {len(waveform)} to {max_length} samples")
            padding = np.zeros(max_length - len(waveform))
            waveform = np.concatenate((waveform, padding))
        
        logger.info(f"Preprocessed audio shape: {waveform.shape}")
        return waveform, sample_rate
    except Exception as e:
        logger.exception(f"Error preprocessing audio: {e}")
        return None, None

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Test direct audio preprocessing without conversion')
    parser.add_argument('--file', required=True, help='Path to the audio file to test')
    
    args = parser.parse_args()
    
    # Test preprocessing
    file_path = args.file
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    # Get file information
    file_size = os.path.getsize(file_path) / 1024  # KB
    file_extension = os.path.splitext(file_path)[1]
    
    logger.info(f"Testing file: {file_path} ({file_size:.2f} KB, {file_extension})")
    
    # Preprocess audio
    waveform, sample_rate = preprocess_audio(file_path)
    
    if waveform is None:
        logger.error("Preprocessing failed")
        sys.exit(1)
    
    # Create a simple feature extractor (won't actually load a model)
    try:
        # Try to create a feature extractor to verify format compatibility
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
        # Get features
        inputs = feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding="max_length",
            max_length=160000  # 10 seconds at 16kHz
        )
        
        logger.info(f"Features created successfully: {inputs.input_values.shape}")
        print(f"✅ Successfully processed {file_path}")
        print(f"   Audio duration: {len(waveform)/sample_rate:.2f} seconds")
        print(f"   Ready for model input with shape: {inputs.input_values.shape}")
        
    except Exception as e:
        logger.exception(f"Error creating features: {e}")
        print(f"❌ Failed to process {file_path}: {str(e)}")
        sys.exit(1) 