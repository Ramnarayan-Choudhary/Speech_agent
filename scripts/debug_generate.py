import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import get_peft_model, LoraConfig
from src.speech_to_text_finetune.config import load_config
from src.speech_to_text_finetune.data_process import (
    load_dataset_from_local,
    process_dataset,
    DataCollatorSpeechSeq2SeqWithPadding
)
import evaluate
from functools import partial
import numpy as np

model_id = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_id, use_cache=False)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_cfg)

processor = WhisperProcessor.from_pretrained(model_id, language="mr", task="transcribe")
collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

# Mock input data
import numpy as np
mock_feature = {"input_features": np.random.randn(80, 3000).tolist()}
mock_label = {"labels": [1, 2, 3, 4]}

batch = collator([
    {"input_features": mock_feature["input_features"], "labels": mock_label["labels"]},
    {"input_features": mock_feature["input_features"], "labels": mock_label["labels"]}
])

print("Batch keys:", batch.keys())
print("input_features shape:", batch["input_features"].shape)
print("labels shape:", batch["labels"].shape)

# How trainer calls compute_loss:
try:
    print("\nTesting forward pass for loss...")
    inputs = {"input_features": torch.tensor(batch["input_features"]), "labels": torch.tensor(batch["labels"])}
    outputs = model(**inputs)
    print("Forward pass SUCCESS!")
except Exception as e:
    print("Forward pass FAILED:", type(e).__name__, e)

try:
    print("\nTesting generation pass...")
    outputs = model.generate(
        input_features=torch.tensor(batch["input_features"]),
        max_length=50
    )
    print("Generate SUCCESS! Output shape:", outputs.shape)
except Exception as e:
    print("Generate FAILED:", type(e).__name__, e)
