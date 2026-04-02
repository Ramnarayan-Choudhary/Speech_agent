"""Tests for data processing pipeline."""

import pytest
from src.speech_to_text_finetune.data_process import (
    DataCollatorSpeechSeq2SeqWithPadding,
    load_subset_of_dataset,
)
from datasets import Dataset


def test_load_subset_of_dataset_all():
    """Test that n_samples=-1 returns full dataset."""
    data = Dataset.from_dict({"col": [1, 2, 3, 4, 5]})
    result = load_subset_of_dataset(data, -1)
    assert len(result) == 5


def test_load_subset_of_dataset_partial():
    """Test that n_samples=3 returns 3 items."""
    data = Dataset.from_dict({"col": [1, 2, 3, 4, 5]})
    result = load_subset_of_dataset(data, 3)
    assert len(result) == 3


def test_load_subset_of_dataset_exceeds_length():
    """Test that n_samples > dataset length returns full dataset."""
    data = Dataset.from_dict({"col": [1, 2, 3]})
    result = load_subset_of_dataset(data, 100)
    assert len(result) == 3
