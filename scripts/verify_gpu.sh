#!/bin/bash
# Step 1: Verify GPU + CUDA setup on GPU node
# Run this INSIDE your gpu-03 srun session

set -e

echo "========================================="
echo "  STEP 1: GPU & CUDA VERIFICATION"
echo "========================================="

cd /home/ramnarayan.ramniwas/MS_projects/Speech_agent

echo ""
echo "1a. Checking nvidia-smi..."
nvidia-smi

echo ""
echo "1b. Checking PyTorch CUDA..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'Torch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "1c. Checking all imports..."
python -c "
import torch, transformers, peft, datasets, evaluate, pydantic, yaml, loguru
from src.speech_to_text_finetune.config import load_config
from src.speech_to_text_finetune.utils import create_model_card, ModelCard
from src.speech_to_text_finetune.data_process import DataCollatorSpeechSeq2SeqWithPadding
from src.speech_to_text_finetune.inference import WhisperInference
cfg = load_config('example_configs/marathi/config_lora_gpu.yaml')
print(f'Config loaded: model={cfg.model_id}, lora_rank={cfg.lora_config.lora_rank}')
print('✅ ALL IMPORTS AND CONFIG OK!')
"

echo ""
echo "========================================="
echo "  ✅ GPU SETUP VERIFIED!"
echo "  Now run: bash scripts/run_training.sh"
echo "========================================="
