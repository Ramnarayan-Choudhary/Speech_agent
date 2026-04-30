#!/bin/bash
#SBATCH --job-name=speech_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=eval_run_combined.log

source /home/ramnarayan.ramniwas/MS_projects/speech/Speech_agent/venv/bin/activate
export PYTHONPATH=/home/ramnarayan.ramniwas/MS_projects/speech/Speech_agent:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
cd /home/ramnarayan.ramniwas/MS_projects/speech/Speech_agent

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

echo "=== [1/2] Running Full Pipeline Evaluation (STT, Routing, BLEU) ==="
echo "Started at: $(date)"
# Run full pipeline eval with 20 samples to get reliable numbers quickly
python scripts/evaluate_full_pipeline.py --num_samples 20

echo ""
echo "=== [2/2] Running TTS Performance Evaluation (ASR-WER, RTF) ==="
echo "Started at: $(date)"
# Run the TTS evaluation script
python scripts/evaluate_tts_quality.py

echo ""
echo "=========================================================="
echo "  ✅ ALL EVALUATIONS COMPLETE: $(date)"
echo "=========================================================="
