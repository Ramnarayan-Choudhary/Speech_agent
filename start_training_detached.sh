#!/bin/bash
# Script to run IndicSpeech LoRA Training completely immune to SSH disconnections.

LOG_FILE="/home/ramnarayan.ramniwas/MS_projects/Speech_agent/training_progress.log"

echo "==========================================================" > $LOG_FILE
echo "  🚀 STARTING DETACHED SPEECH AGENT TRAINING PIPELINE" >> $LOG_FILE
echo "  Resilient against WiFi/Power SSH Disconnections" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE

# Create a clean worker script on the fly so we avoid nasty bash quote nesting
cat << 'EOF' > run_worker.sh
#!/bin/bash
source /home/ramnarayan.ramniwas/MS_projects/Speech_agent/venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
cd /home/ramnarayan.ramniwas/MS_projects/Speech_agent

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
EOF

chmod +x run_worker.sh

# Run the worker inside srun detached
srun --partition=gpu --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 --mem=32G ./run_worker.sh >> $LOG_FILE 2>&1 &

echo "Detached training started successfully!"
echo "PID: $!"
