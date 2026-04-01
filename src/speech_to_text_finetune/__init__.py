"""
Speech-to-Text Fine-tuning library for Indic languages.

Based on MultiLingual OpenAI Whisper model fine-tuning,
implementing LoRA (Low-Rank Adaptation) for efficient training.
"""

__version__ = "0.1.0"
__author__ = "Ramnarayan Choudhary"
__email__ = "choudharyramnarayan123@gmail.com"

from .config import Config, TrainingConfig, LoRAConfig
from .data_process import load_dataset_from_dataset_id, process_dataset
from .finetune_whisper import run_finetuning
from .inference import WhisperInference

__all__ = [
    "Config",
    "TrainingConfig",
    "LoRAConfig",
    "load_dataset_from_dataset_id",
    "process_dataset",
    "run_finetuning",
    "WhisperInference",
]
