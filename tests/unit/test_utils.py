"""Tests for utility functions."""

import pytest
from src.speech_to_text_finetune.utils import create_model_card, ModelCard


def test_create_model_card_returns_model_card():
    """Test that create_model_card returns a ModelCard with .save() method."""
    baseline_eval = {"eval_wer": 45.0, "eval_cer": 12.0}
    ft_eval = {"eval_wer": 35.0, "eval_cer": 8.0}

    card = create_model_card(
        model_id="openai/whisper-small",
        dataset_id="mozilla-foundation/common_voice_17_0",
        language="marathi",
        baseline_eval=baseline_eval,
        ft_eval=ft_eval,
    )

    assert isinstance(card, ModelCard)
    assert hasattr(card, "save")
    assert "marathi" in card.content.lower()
    assert "LoRA" in card.content


def test_model_card_save(tmp_path):
    """Test that ModelCard.save() writes content to disk."""
    card = ModelCard("# Test Model Card\nSome content here.")
    output_path = str(tmp_path / "subdir" / "README.md")
    card.save(output_path)

    with open(output_path, "r") as f:
        content = f.read()

    assert "Test Model Card" in content


def test_create_model_card_zero_baseline():
    """Test that create_model_card handles zero baseline (no division by zero)."""
    baseline_eval = {"eval_wer": 0, "eval_cer": 0}
    ft_eval = {"eval_wer": 10.0, "eval_cer": 5.0}

    card = create_model_card(
        model_id="openai/whisper-small",
        dataset_id="test-dataset",
        language="hindi",
        baseline_eval=baseline_eval,
        ft_eval=ft_eval,
    )

    assert isinstance(card, ModelCard)
    assert "hindi" in card.content.lower()
