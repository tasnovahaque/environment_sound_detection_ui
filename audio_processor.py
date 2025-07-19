import os
import logging
from pydub import AudioSegment
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format."""
    try:
        logger.info(f"Converting audio file to WAV: {input_path} -> {output_path}")
        
        # Create category directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to WAV
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        
        logger.info(f"Successfully converted audio to WAV: {output_path}")
        return output_path
    except Exception as e:
        logger.exception(f"Error converting audio: {e}")
        return None

def split_audio(audio_path, output_dir, chunk_length_ms=10000, overlap_ms=0):
    """Split audio file into chunks of specified length with optional overlap"""
    try:
        logger.info(f"Splitting audio: {audio_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the audio
        audio = AudioSegment.from_file(audio_path)
        
        # Get the base filename without extension
        base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
        
        # Calculate number of chunks
        audio_length_ms = len(audio)
        
        if audio_length_ms < chunk_length_ms:
            logger.warning(f"Audio file {audio_path} is shorter than the chunk length, skipping split")
            # Just copy the file to the output directory
            output_path = os.path.join(output_dir, f"{base_name}.wav")
            audio.export(output_path, format="wav")
            return 1
            
        # Calculate step size and number of chunks
        step_ms = chunk_length_ms - overlap_ms
        num_chunks = max(1, int((audio_length_ms - overlap_ms) / step_ms))
        
        logger.info(f"Creating {num_chunks} chunks of {chunk_length_ms}ms each with {overlap_ms}ms overlap")
        
        # Extract chunks
        for i in range(num_chunks):
            start_ms = i * step_ms
            end_ms = min(start_ms + chunk_length_ms, audio_length_ms)
            
            # Create chunk
            chunk = audio[start_ms:end_ms]
            
            # Save chunk
            chunk_path = os.path.join(output_dir, f"{base_name}_chunk{i+1:03d}.wav")
            chunk.export(chunk_path, format="wav")
            
            logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            
        return num_chunks
    except Exception as e:
        logger.exception(f"Error splitting audio {audio_path}: {e}")
        return 0

def process_dataset(input_dir, wav_dir, chunks_dir):
    """Process all audio files in the dataset."""
    # Create output directories
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Count total files for progress tracking
    total_files = 0
    for root, _, files in os.walk(input_dir):
        total_files += len([f for f in files if f.endswith(('.mp3', '.m4a', '.wav', '.aac'))])

    print(f"Processing {total_files} audio files...")

    # Process each file
    processed_files = 0
    total_chunks = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp3', '.m4a', '.wav', '.aac')):
                # Get paths
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                wav_path = os.path.join(wav_dir, relative_path.rsplit('.', 1)[0] + '.wav')

                # Ensure the directory exists
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)

                try:
                    # Convert to WAV
                    converted_file = convert_to_wav(input_path, wav_path)
                    
                    if converted_file is None:
                        print(f"Failed to convert {input_path}")
                        continue

                    # Split into chunks
                    chunk_output_dir = os.path.join(chunks_dir, relative_path.rsplit('.', 1)[0])
                    num_chunks = split_audio(converted_file, chunk_output_dir)
                    total_chunks += num_chunks

                    processed_files += 1
                    print(f"Processed {processed_files}/{total_files}: {relative_path} -> {num_chunks} chunks")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

    print(f"Completed processing {processed_files} files into {total_chunks} chunks.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process audio files for the classifier dataset')
    parser.add_argument('--input', required=True, help='Directory containing input audio files')
    parser.add_argument('--wav_output', required=True, help='Directory to store converted WAV files')
    parser.add_argument('--chunks_output', required=True, help='Directory to store audio chunks')
    parser.add_argument('--chunk_length', type=int, default=10000, help='Length of each chunk in milliseconds (default: 10000)')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between chunks in milliseconds (default: 0)')
    
    args = parser.parse_args()
    
    # Process the dataset
    process_dataset(args.input, args.wav_output, args.chunks_output) 