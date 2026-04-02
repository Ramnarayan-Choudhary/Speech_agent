"""Integration test for training pipeline config loading."""

import pytest
from src.speech_to_text_finetune.config import load_config
import yaml


def test_training_pipeline_config_loads(tmp_path):
    """Test that a valid training config can be loaded and has expected structure."""
    config_data = {
        "model_id": "openai/whisper-tiny",
        "dataset_id": "mozilla-foundation/common_voice_17_0",
        "language": "marathi",
        "language_code": "mr",
        "repo_name": "whisper-marathi-test",
        "n_train_samples": 10,
        "n_test_samples": 5,
        "lora_config": {
            "use_lora": False,
            "lora_rank": 4,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
        "training_hp": {
            "push_to_hub": False,
            "hub_private_repo": False,
            "max_steps": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "warmup_steps": 2,
            "gradient_checkpointing": True,
            "fp16": False,
            "eval_strategy": "steps",
            "eval_steps": 5,
            "per_device_eval_batch_size": 1,
            "predict_with_generate": True,
            "generation_max_length": 128,
            "save_steps": 5,
            "logging_steps": 1,
            "load_best_model_at_end": True,
            "save_total_limit": 1,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
        },
    }

    config_path = tmp_path / "integration_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(str(config_path))

    assert config is not None
    assert config.model_id == "openai/whisper-tiny"
    assert config.lora_config.use_lora is False
    assert config.training_hp.max_steps == 5
    assert config.n_train_samples == 10


def test_all_gpu_configs_load():
    """Test that all actual GPU config files load without errors."""
    configs = [
        "example_configs/marathi/config_lora_gpu.yaml",
        "example_configs/gujarati/config_lora_gpu.yaml",
        "example_configs/hindi/config_lora_gpu.yaml",
    ]

    for config_path in configs:
        cfg = load_config(config_path)
        assert cfg.model_id == "openai/whisper-small"
        assert cfg.lora_config.use_lora is True


def test_all_cpu_configs_load():
    """Test that all actual CPU config files also load without errors."""
    configs = [
        "example_configs/marathi/config_lora_cpu.yaml",
    ]

    for config_path in configs:
        cfg = load_config(config_path)
        assert cfg.lora_config.use_lora is True
