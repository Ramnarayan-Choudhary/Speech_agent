"""End-to-end test for inference module."""

import pytest
from src.speech_to_text_finetune.inference import WhisperInference


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="Skipping model loading test on CPU-only machine (too slow without GPU)"
)
def test_whisper_inference_smoke():
    """Test WhisperInference class instantiation and method presence.
    
    Only runs when GPU is available since model loading is very slow on CPU.
    """
    inference = WhisperInference("openai/whisper-tiny")

    assert hasattr(inference, "transcribe")
    assert hasattr(inference, "transcribe_batch")
    assert hasattr(inference, "pipe")


def test_whisper_inference_class_exists():
    """Verify WhisperInference class has expected interface."""
    assert hasattr(WhisperInference, "transcribe")
    assert hasattr(WhisperInference, "transcribe_batch")
