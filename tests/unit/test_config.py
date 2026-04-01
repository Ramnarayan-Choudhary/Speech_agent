import pytest
from speech_to_text_finetune.config import Config


def test_config_load_success(tmp_path):
    yaml_content = """
model_name_and_path: openai/whisper-small
language: mr
lora_config:
  use_lora: true
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj]
training:
  num_train_epochs: 1
  train_batch_size: 4
  eval_batch_size: 4
  learning_rate: 1e-4
  weight_decay: 0.01
  gradient_accumulation_steps: 2
  reservoir: 0.1
  seed: 42
  output_dir: ./checkpoints
  logging_dir: ./logs
"""

    file_path = tmp_path / "test_config.yaml"
    file_path.write_text(yaml_content)

    cfg = Config.from_yaml(str(file_path))

    assert cfg.model_name_and_path == "openai/whisper-small"
    assert cfg.language == "mr"
    assert cfg.lora_config.use_lora
    assert cfg.training.num_train_epochs == 1

def test_config_invalid_language():
    yaml_content = """
model_name_and_path: openai/whisper-small
language: unsupported
lora_config:
  use_lora: true
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj]
training:
  num_train_epochs: 1
  train_batch_size: 4
  eval_batch_size: 4
  learning_rate: 1e-4
  weight_decay: 0.01
  gradient_accumulation_steps: 2
  reservoir: 0.1
  seed: 42
  output_dir: ./checkpoints
  logging_dir: ./logs
"""

    file_path = tmp_path / "test_config_invalid.yaml"
    file_path.write_text(yaml_content)

    with pytest.raises(ValueError):
        Config.from_yaml(str(file_path))
