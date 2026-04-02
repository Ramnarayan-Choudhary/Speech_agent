"""Dataset loading and processing pipeline.

Supports both:
  1. HuggingFace Hub datasets (e.g., "mozilla-foundation/common_voice_17_0")
  2. Local datasets saved via Dataset.save_to_disk() (e.g., "data/processed/marathi")
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from datasets import load_dataset, DatasetDict, Audio, Dataset, load_from_disk
from transformers import WhisperProcessor
import logging

logger = logging.getLogger(__name__)


def load_dataset_from_local(local_path: str) -> Tuple[DatasetDict, str]:
    """Load a pre-processed HuggingFace DatasetDict from disk.
    
    Args:
        local_path: Path to a directory saved via DatasetDict.save_to_disk()
    
    Returns:
        Tuple of (DatasetDict, proc_path)
    """
    logger.info(f"Loading local dataset from: {local_path}")
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local dataset not found: {local_path}")
    
    dataset = load_from_disk(local_path)
    
    if isinstance(dataset, Dataset):
        # Single dataset — split it
        split_data = dataset.train_test_split(test_size=0.2, seed=42)
        dataset = DatasetDict({
            "train": split_data["train"],
            "test": split_data["test"],
        })
    
    logger.info(f"Loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")
    return dataset, local_path


def load_dataset_from_dataset_id(
    dataset_id: str,
    language_id: str,
) -> Tuple[DatasetDict, str]:
    """Load dataset from HuggingFace Hub."""
    logger.info(f"Loading {dataset_id} for language {language_id}")
    
    # Get HF token from environment if available
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    
    dataset = load_dataset(
        dataset_id,
        language_id,
        split="train",
        streaming=False,
        token=hf_token,
        trust_remote_code=True,
    )
    
    # Cast audio to 16kHz (Whisper requirement)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split into train/test
    split_data = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict = DatasetDict({
        "train": split_data["train"],
        "test": split_data["test"]
    })
    
    return dataset_dict, f"./processed/{language_id}"


def load_subset_of_dataset(dataset: Dataset, n_samples: int) -> Dataset:
    """Load subset of dataset."""
    if n_samples == -1:
        return dataset
    return dataset.select(range(min(n_samples, len(dataset))))


def process_dataset(
    dataset: DatasetDict,
    processor: WhisperProcessor,
    batch_size: int = 16,
    proc_dataset_path: str = "processed"
) -> DatasetDict:
    """Process audio and text into model-ready format."""
    logger.info(f"Processing dataset {proc_dataset_path}")
    
    def preprocess_data(sample):
        audio = sample["audio"]
        sample["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        sample["labels"] = processor.tokenizer(sample["sentence"]).input_ids
        return sample
    
    dataset = dataset.map(
        preprocess_data,
        remove_columns=dataset["train"].column_names,
        batched=False,
        batch_size=batch_size
    )
    
    return dataset


class DataCollatorSpeechSeq2SeqWithPadding:
    """Collate speech data for training."""
    
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
    
    def __call__(self, batch):
        input_features = [{"input_features": item["input_features"]} for item in batch]
        batch_features = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch_labels = self.processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
        
        return {
            "input_features": batch_features["input_features"],
            "labels": batch_labels["input_ids"],
        }
