import pytest
from speech_to_text_finetune.inference import WhisperInference


def test_whisper_inference_smoke():
    # This test assumes model docker execution and availability is provided in CI.
    # It should only ensure API works end-to-end for the class logic.
    inference = WhisperInference(model_path="openai/whisper-tiny")

    # A small dummy audio generated in-memory can be used in a real test environment.
    # For now we assert class instantiation and method presence.
    assert hasattr(inference, "transcribe")
