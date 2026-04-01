"""Configuration schema for Whisper fine-tuning with LoRA."""

import yaml
from pydantic import BaseModel, Field
from typing import Optional, List


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""
    
    use_lora: bool = Field(default=True, description="Enable LoRA")
    lora_rank: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA Alpha")
    lora_dropout: float = Field(default=0.05, description="Dropout ratio")
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj"],
        description="Target modules for adaptation"
    )
    bias: str = Field(default="none", description="Bias adaptation")


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    
    push_to_hub: bool
    hub_private_repo: bool
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    gradient_checkpointing: bool
    fp16: bool
    eval_strategy: str
    eval_steps: int
    per_device_eval_batch_size: int
    predict_with_generate: bool
    generation_max_length: int
    save_steps: int
    logging_steps: int
    load_best_model_at_end: bool
    save_total_limit: int
    metric_for_best_model: str
    greater_is_better: bool


class Config(BaseModel):
    """Main configuration for fine-tuning."""
    
    model_id: str = Field(description="Whisper model ID")
    dataset_id: str = Field(description="Dataset ID")
    language: str = Field(description="Language name")
    language_code: Optional[str] = Field(description="Language code (e.g., 'hi', 'mr')")
    repo_name: str = Field(description="Repository name")
    n_train_samples: int = Field(default=-1, description="Number of training samples (-1 for all)")
    n_test_samples: int = Field(default=-1, description="Number of test samples (-1 for all)")
    lora_config: LoRAConfig = Field(description="LoRA configuration")
    training_hp: TrainingConfig = Field(description="Training hyperparameters")


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    
    # Handle nested LoRA config
    if "lora_config" in config_dict:
        config_dict["lora_config"] = LoRAConfig(**config_dict["lora_config"])
    
    if "training_hp" in config_dict:
        config_dict["training_hp"] = TrainingConfig(**config_dict["training_hp"])
    
    return Config(**config_dict)


# Constants
PROC_DATASET_DIR = "processed_version"
