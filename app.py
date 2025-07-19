import os
import torch
import numpy as np
import librosa
import soundfile as sf
import time
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import tempfile
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac'}
app.config['MODEL_PATH'] = 'wav2vec_finetuned'  # Path to your model

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the class names
CLASSES = ['Bus', 'Metro', 'Metro_Station', 'Park', 'Restaurant', 'Shopping_Mall', 'University']

# Global variables for models
model = None
feature_extractor = None
device = None

def load_model(model_path):
    """Load the Wav2Vec2 model and feature extractor"""
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        
        # Load model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully. Using device: {device}")
        
        return model, feature_extractor, device
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        return None, None, None

def preprocess_audio(audio_path, feature_extractor, max_length=16000*10):  # 10 seconds at 16kHz
    """Preprocess audio file for Wav2Vec2 model"""
    try:
        logger.info(f"Preprocessing audio file: {audio_path}")
        
        # Load audio using librosa (handles various formats)
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        # Ensure correct shape for the model
        waveform = np.asarray(waveform)
        
        # Truncate or pad to max_length
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            padding = np.zeros(max_length - len(waveform))
            waveform = np.concatenate((waveform, padding))
        
        # Get features
        inputs = feature_extractor(
            waveform, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding="max_length",
            max_length=max_length
        )
        
        return inputs
    except Exception as e:
        logger.exception(f"Error preprocessing audio: {e}")
        return None

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_audio_length(file_path, min_seconds=10):
    """Check if audio file is at least 10 seconds long and trim if longer than needed"""
    try:
        # Try using librosa first which handles more formats
        try:
            logger.info(f"Validating audio length using librosa: {file_path}")
            y, sr = librosa.load(file_path, sr=None)
            duration_seconds = librosa.get_duration(y=y, sr=sr)
            logger.info(f"Audio duration (librosa): {duration_seconds:.2f} seconds")
        except Exception as e:
            logger.warning(f"Librosa validation failed, trying pydub: {e}")
            # Fall back to pydub if librosa fails
            audio = AudioSegment.from_file(file_path)
            duration_seconds = len(audio) / 1000  # pydub uses milliseconds
            logger.info(f"Audio duration (pydub): {duration_seconds:.2f} seconds")
        
        if duration_seconds < min_seconds:
            logger.warning(f"Audio is too short: {duration_seconds:.2f} seconds (minimum: {min_seconds})")
            # Audio is too short
            return False, None
        
        logger.info(f"Audio passes length validation: {duration_seconds:.2f} seconds")
        
        # If audio is longer than 10 seconds, trim it to exactly 10 seconds
        if duration_seconds > min_seconds:
            logger.info(f"Trimming audio from {duration_seconds:.2f} to {min_seconds} seconds")
            
            # Use librosa for trimming if we have the data already
            if 'y' in locals() and 'sr' in locals():
                # Calculate samples to keep (convert seconds to samples)
                samples_to_keep = int(min_seconds * sr)
                trimmed_y = y[:samples_to_keep]
                
                # Create a temporary file to store the trimmed audio
                temp_trimmed = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_trimmed_path = temp_trimmed.name
                temp_trimmed.close()
                
                # Save trimmed audio
                import soundfile as sf
                sf.write(temp_trimmed_path, trimmed_y, sr)
                
                logger.info(f"Trimmed audio saved to: {temp_trimmed_path}")
                return True, temp_trimmed_path
            else:
                # Use pydub for trimming if we don't have librosa data
                trimmed_audio = audio[:min_seconds * 1000]  # Convert to milliseconds
                
                # Create a temporary file to store the trimmed audio
                temp_trimmed = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_trimmed_path = temp_trimmed.name
                temp_trimmed.close()
                
                # Export the trimmed audio
                trimmed_audio.export(temp_trimmed_path, format='wav')
                
                logger.info(f"Trimmed audio saved to: {temp_trimmed_path}")
                return True, temp_trimmed_path
        
        # Audio is within acceptable range (exactly 10 seconds or very close)
        logger.info(f"Audio is within acceptable range, no trimming needed")
        return True, None
    except Exception as e:
        logger.exception(f"Error validating audio length: {e}")
        return False, None

def convert_to_wav(file_path):
    """Convert any audio file to WAV format"""
    try:
        logger.info(f"Converting audio file to WAV: {file_path}")
        
        # Create a temporary file to store the converted WAV
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(temp_wav_path), exist_ok=True)

        # Convert to WAV
        audio = AudioSegment.from_file(file_path)
        audio.export(temp_wav_path, format="wav")
        
        logger.info(f"Successfully converted audio to WAV: {temp_wav_path}")
        return temp_wav_path
    except Exception as e:
        logger.exception(f"Error converting audio: {e}")
        # Clean up the temp file if it exists
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except:
                pass
        return None

def classify_audio(file_path):
    """Classify the audio using Wav2Vec2 model"""
    try:
        logger.info(f"Starting classification of file: {file_path}")
        
        # Check if model is loaded
        global model, feature_extractor, device
        if model is None or feature_extractor is None or device is None:
            logger.info("Model not loaded, attempting to load...")
            model, feature_extractor, device = load_model(app.config['MODEL_PATH'])
            if model is None:
                return None, "Error loading model"
        
        # Preprocess audio for model directly without conversion
        inputs = preprocess_audio(file_path, feature_extractor)
        
        if inputs is None:
            return None, "Error preprocessing audio"
        
        # Move inputs to device
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
        
        # Run inference
        with torch.no_grad():
            if attention_mask is not None:
                outputs = model(input_values=input_values, attention_mask=attention_mask)
            else:
                outputs = model(input_values=input_values)
        
        # Get predictions
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        prediction = torch.argmax(probs).item()
        
        # Get class name and confidence
        predicted_class = CLASSES[prediction]
        confidence = probs[prediction].item()
        
        # Get all probabilities
        all_probs = {class_name: float(prob) for class_name, prob in zip(CLASSES, probs.cpu().numpy())}
        
        logger.info(f"Classification result: {predicted_class} with {confidence:.2%} confidence")
        
        # Create results dictionary
        result = {
            'class': predicted_class,
            'confidence': confidence,
            'all_probs': all_probs,
            'message': "Classification based on Wav2Vec2 model finetuned on environmental audio."
        }
        
        return result, None
    except Exception as e:
        logger.exception(f"Error classifying audio: {e}")
        return None, f"Error classifying audio: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            logger.info(f"Successfully saved uploaded file: {file_path}")
            
            # Validate audio length directly
            logger.info(f"Validating audio length: {file_path}")
            valid, trimmed_path = validate_audio_length(file_path)
            
            logger.info(f"Validation result: valid={valid}, trimmed_path={trimmed_path}")
            
            if not valid:
                logger.warning(f"Audio validation failed for {file_path}")
                flash('Audio must be at least 10 seconds long', 'error')
                if os.path.exists(file_path):
                    os.remove(file_path)
                return redirect(url_for('index'))
            
            # Use trimmed path if available, otherwise use original file
            classification_path = trimmed_path if trimmed_path else file_path
            logger.info(f"Using file for classification: {classification_path}")
            
            # Classify the audio
            result, error = classify_audio(classification_path)
            
            # Clean up files
            if trimmed_path and os.path.exists(trimmed_path):
                os.unlink(trimmed_path)
                logger.info(f"Cleaned up trimmed file: {trimmed_path}")
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up original file: {file_path}")
            
            if error:
                logger.error(f"Classification error: {error}")
                flash(error, 'error')
                return redirect(url_for('index'))
            
            logger.info(f"Classification successful: {result['class']} with {result['confidence']:.2%} confidence")
            return render_template('result.html', result=result)
        except Exception as e:
            logger.exception(f"Error in upload_file: {e}")
            flash(f"An error occurred: {str(e)}", 'error')
            # Cleanup any left files
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            if 'trimmed_path' in locals() and trimmed_path and os.path.exists(trimmed_path):
                os.unlink(trimmed_path)
            return redirect(url_for('index'))
    
    flash('File type not allowed', 'error')
    return redirect(url_for('index'))

@app.route('/record', methods=['POST'])
def record_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio data received'}), 400
    
    audio_file = request.files['audio']
    
    # Save the recorded audio
    try:
        filename = secure_filename(f"recording_{int(time.time())}.wav")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(file_path)
        
        logger.info(f"Successfully saved recorded audio: {file_path}")
        
        # Validate audio length directly
        logger.info(f"Validating audio length: {file_path}")
        valid, trimmed_path = validate_audio_length(file_path, min_seconds=9.5)  # Slightly less than 10s for browser recordings
        
        logger.info(f"Validation result: valid={valid}, trimmed_path={trimmed_path}")
        
        if not valid:
            logger.warning(f"Audio validation failed for {file_path}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Audio must be at least 10 seconds long'}), 400
        
        # Use trimmed path if available, otherwise use original file
        classification_path = trimmed_path if trimmed_path else file_path
        logger.info(f"Using file for classification: {classification_path}")
        
        # Classify the audio
        result, error = classify_audio(classification_path)
        
        # Clean up
        if trimmed_path and os.path.exists(trimmed_path):
            os.unlink(trimmed_path)
            logger.info(f"Cleaned up trimmed file: {trimmed_path}")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up original file: {file_path}")
        
        if error:
            logger.error(f"Classification error: {error}")
            return jsonify({'error': error}), 500
        
        logger.info(f"Classification successful: {result['class']} with {result['confidence']:.2%} confidence")
        return jsonify(result)
    except Exception as e:
        # If anything goes wrong, make sure we don't leave files behind
        logger.exception(f"Error in record_audio: {e}")
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            if 'trimmed_path' in locals() and trimmed_path and os.path.exists(trimmed_path):
                os.unlink(trimmed_path)
        except Exception as cleanup_error:
            logger.exception(f"Error during cleanup: {cleanup_error}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/api/classify', methods=['POST'])
def api_classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Validate audio length directly
        valid, trimmed_path = validate_audio_length(file_path)
        if not valid:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Audio must be at least 10 seconds long'}), 400
        
        # Use trimmed path if available, otherwise use original file
        classification_path = trimmed_path if trimmed_path else file_path
        
        # Classify the audio
        result, error = classify_audio(classification_path)
        
        # Clean up files
        if trimmed_path and os.path.exists(trimmed_path):
            os.unlink(trimmed_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result)
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/health')
def health_check():
    """Health check endpoint with system status"""
    global model, feature_extractor, device
    status = {
        "status": "ok",
        "timestamp": time.time(),
        "model_loaded": model is not None,
        "feature_extractor_loaded": feature_extractor is not None,
        "device": str(device) if device else None,
        "classes": CLASSES
    }
    return jsonify(status)

if __name__ == '__main__':
    # Load the model at startup
    logger.info("Starting Audio Environment Classifier with Wav2Vec2")
    
    # Initialize model
    model, feature_extractor, device = load_model(app.config['MODEL_PATH'])
    if model is None:
        logger.warning("Could not load model at startup. Will try again when classifying audio.")
    else:
        logger.info(f"Model loaded successfully. Using device: {device}")
    
    # Inform the user
    flash("Using Wav2Vec2 model for audio classification. This model has been finetuned on environmental audio data.", "info")
    
    # Run the app
    logger.info("Starting Flask application...")
    app.run(debug=True)