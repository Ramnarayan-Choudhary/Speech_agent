"""Inference pipeline for multilingual ASR."""

import torch
from transformers import pipeline
from loguru import logger


class WhisperInference:
    """Whisper inference wrapper for transcription."""
    
    def __init__(self, model_id: str):
        """Initialize inference pipeline."""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        
        logger.info(f"Loading model {model_id} on {self.device}")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=self.device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file."""
        logger.info(f"Transcribing: {audio_path}")
        result = self.pipe(audio_path)
        logger.info(f"Result: {result['text']}")
        return result
    
    def transcribe_batch(self, audio_paths: list) -> list:
        """Transcribe multiple audio files."""
        logger.info(f"Transcribing {len(audio_paths)} files...")
        results = [self.transcribe(path) for path in audio_paths]
        return results


if __name__ == "__main__":
    # Example usage
    model = WhisperInference("openai/whisper-small")
    print("Inference module loaded successfully")
