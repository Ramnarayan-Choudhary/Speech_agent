#!/bin/bash
source /home/ramnarayan.ramniwas/MS_projects/speech/Speech_agent/venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
cd /home/ramnarayan.ramniwas/MS_projects/speech/Speech_agent

echo "=== Node Information ==="
echo "Python: $(which python)"
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available())"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo ""

echo "=== [1/2] Training Marathi LoRA ==="
echo "Started Marathi at: $(date)"
python -u src/speech_to_text_finetune/finetune_whisper.py --config example_configs/marathi/config_indicspeech.yaml

echo ""
echo "=== [2/2] Training Gujarati LoRA ==="
echo "Started Gujarati at: $(date)"
python -u src/speech_to_text_finetune/finetune_whisper.py --config example_configs/gujarati/config_indicspeech.yaml

echo ""
echo "=========================================================="
echo "  ✅ ALL TRAINING COMPLETE: $(date)"
echo "=========================================================="
