import pytest
from speech_to_text_finetune.data_process import DataCollatorSpeechSeq2SeqWithPadding
from transformers import WhisperProcessor
import torch


def test_data_collator_shapes():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    features = [
        {
            "input_features": torch.randn(80, 80),
            "labels": torch.tensor([1, 2, 3, 4])
        },
        {
            "input_features": torch.randn(80, 80),
            "labels": torch.tensor([1, 2, 3])
        }
    ]

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    batch = collator(features)

    assert batch["input_features"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    assert (batch["labels"] == -100).sum().item() >= 1
