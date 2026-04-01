"""Dataset loading and processing pipeline."""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from datasets import load_dataset, DatasetDict, Audio, Dataset, load_from_disk
from transformers import WhisperProcessor
import logging

logger = logging.getLogger(__name__)


def load_dataset_from_dataset_id(
    dataset_id: str,
    language_id: str,
) -> Tuple[DatasetDict, str]:
    """Load dataset from HuggingFace."""
    logger.info(f"Loading {dataset_id} for language {language_id}")
    
    dataset = load_dataset(
        dataset_id,
        language_id,
        split="train",
        trust_remote_code=True,
        streaming=False
    )
    
    # Split into train/test
    split_data = dataset.train_test_split(test_size=0.2)
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
    
    # Save processed dataset
    Path(proc_dataset_path).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(proc_dataset_path)
    logger.info(f"Saved processed dataset to {proc_dataset_path}")
    
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
