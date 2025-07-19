from pydub import AudioSegment
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_conversion(input_file, output_file):
    """Test converting an audio file to WAV format"""
    try:
        logger.info(f"Testing conversion: {input_file} -> {output_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert to WAV
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        
        logger.info(f"Conversion successful: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Create test directory
    os.makedirs("test_output", exist_ok=True)
    
    if len(sys.argv) > 1:
        # Test with provided file
        input_file = sys.argv[1]
        output_file = os.path.join("test_output", os.path.basename(input_file).split('.')[0] + ".wav")
        
        success = test_audio_conversion(input_file, output_file)
        
        if success:
            print(f"✅ Conversion successful: {output_file}")
        else:
            print(f"❌ Conversion failed. Check logs for details.")
    else:
        print("Please provide an audio file path to test. Usage:")
        print("python test_audio_conversion.py path/to/audio/file.mp3") 