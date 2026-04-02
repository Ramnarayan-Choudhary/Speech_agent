#!/bin/bash
#SBATCH --job-name=speech_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=training_progress.log

source /home/ramnarayan.ramniwas/MS_projects/Speech_agent/venv/bin/activate
export PYTHONPATH=/home/ramnarayan.ramniwas/MS_projects/Speech_agent:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
cd /home/ramnarayan.ramniwas/MS_projects/Speech_agent

# Load the WANDB_API_KEY
if [ -f .env ]; then
  source .env
  export WANDB_API_KEY
fi

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
