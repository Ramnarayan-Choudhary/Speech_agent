import pytest
from speech_to_text_finetune.config import Config
from speech_to_text_finetune.finetune_whisper import train


def test_training_pipeline_config(tmp_path):
    # this is a smoke test to ensure the train function can accept config and run minimal steps
    yaml_content = """
model_name_and_path: openai/whisper-tiny
language: mr
lora_config:
  use_lora: false
  lora_rank: 4
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules: [q_proj, v_proj]
training:
  num_train_epochs: 1
  train_batch_size: 1
  eval_batch_size: 1
  learning_rate: 1e-4
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  reservoir: 0.1
  seed: 42
  output_dir: ./checkpoints
  logging_dir: ./logs
"""

    config_path = tmp_path / "integration_config.yaml"
    config_path.write_text(yaml_content)

    config = Config.from_yaml(str(config_path))

    # Just ensure that it runs without crashing in dry run mode (no actual training loop)
    result = train(config)

    assert result is not None
