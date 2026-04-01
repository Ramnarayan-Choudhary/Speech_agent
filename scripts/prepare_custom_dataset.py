"""Format custom datasets for training."""

import csv
from pathlib import Path
import soundfile as sf
import numpy as np
from loguru import logger


def prepare_custom_dataset(audio_dir: str, text_file: str, output_dir: str):
    """Prepare custom dataset in format compatible with training pipeline."""
    
    audio_path = Path(audio_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read text file
    with open(text_file, 'r') as f:
        reader = csv.DictReader(f)
        records = list(reader)
    
    logger.info(f"Processing {len(records)} audio-text pairs...")
    
    # Create dataset index
    dataset_index = []
    for i, record in enumerate(records):
        audio_file = record.get('audio') or record.get('filename') or f"audio_{i}.wav"
        text = record.get('text') or record.get('sentence')
        
        audio_full_path = audio_path / audio_file
        
        if not audio_full_path.exists():
            logger.warning(f"Audio file not found: {audio_full_path}")
            continue
        
        # Verify audio
        try:
            audio, sr = sf.read(audio_full_path)
            duration = len(audio) / sr
            
            dataset_index.append({
                'audio_path': str(audio_full_path),
                'text': text,
                'duration': duration,
                'sample_rate': sr
            })
        except Exception as e:
            logger.warning(f"Error reading {audio_full_path}: {e}")
            continue
    
    logger.info(f"✅ Prepared {len(dataset_index)} samples")
    logger.info(f"Output directory: {output_path}")
    
    return dataset_index


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--text-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    args = parser.parse_args()
    
    prepare_custom_dataset(args.audio_dir, args.text_file, args.output_dir)
