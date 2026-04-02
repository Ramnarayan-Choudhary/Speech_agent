"""Tests for configuration loading and validation."""

import pytest
import yaml
from src.speech_to_text_finetune.config import load_config, Config, LoRAConfig, TrainingConfig


def test_config_load_success(tmp_path):
    """Test loading a valid config YAML file."""
    config_data = {
        "model_id": "openai/whisper-small",
        "dataset_id": "mozilla-foundation/common_voice_17_0",
        "language": "marathi",
        "language_code": "mr",
        "repo_name": "whisper-marathi-lora-small",
        "n_train_samples": 500,
        "n_test_samples": 100,
        "lora_config": {
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
        "training_hp": {
            "push_to_hub": False,
            "hub_private_repo": True,
            "max_steps": 100,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "warmup_steps": 50,
            "gradient_checkpointing": True,
            "fp16": False,
            "eval_strategy": "steps",
            "eval_steps": 50,
            "per_device_eval_batch_size": 4,
            "predict_with_generate": True,
            "generation_max_length": 128,
            "save_steps": 50,
            "logging_steps": 10,
            "load_best_model_at_end": True,
            "save_total_limit": 2,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
        },
    }

    file_path = tmp_path / "test_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)

    cfg = load_config(str(file_path))

    assert cfg.model_id == "openai/whisper-small"
    assert cfg.language == "marathi"
    assert cfg.lora_config.use_lora is True
    assert cfg.lora_config.lora_rank == 8
    assert cfg.training_hp.max_steps == 100
    assert cfg.n_train_samples == 500


def test_config_with_extra_training_fields(tmp_path):
    """Test that extra fields in training_hp are allowed (forward-compatibility)."""
    config_data = {
        "model_id": "openai/whisper-small",
        "dataset_id": "mozilla-foundation/common_voice_17_0",
        "language": "marathi",
        "language_code": "mr",
        "repo_name": "whisper-marathi-lora-small",
        "lora_config": {
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
        "training_hp": {
            "push_to_hub": False,
            "hub_private_repo": True,
            "max_steps": 100,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "warmup_steps": 50,
            "gradient_checkpointing": True,
            "fp16": False,
            "eval_strategy": "steps",
            "eval_steps": 50,
            "per_device_eval_batch_size": 4,
            "predict_with_generate": True,
            "generation_max_length": 128,
            "save_steps": 50,
            "logging_steps": 10,
            "load_best_model_at_end": True,
            "save_total_limit": 2,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
            "remove_unused_columns": False,
            "label_names": ["labels"],
        },
    }

    file_path = tmp_path / "test_config_extra.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)

    cfg = load_config(str(file_path))
    assert cfg.training_hp.remove_unused_columns is False
    assert cfg.training_hp.label_names == ["labels"]


def test_load_actual_gpu_config():
    """Test loading actual GPU config from repo."""
    cfg = load_config("example_configs/marathi/config_lora_gpu.yaml")
    assert cfg.model_id == "openai/whisper-small"
    assert cfg.language == "marathi"
    assert cfg.lora_config.use_lora is True


def test_lora_config_defaults():
    """Test LoRAConfig default values."""
    lora = LoRAConfig()
    assert lora.use_lora is True
    assert lora.lora_rank == 8
    assert lora.lora_alpha == 16
    assert lora.lora_dropout == 0.05
    assert lora.target_modules == ["q_proj", "v_proj"]
    assert lora.bias == "none"
