# 🚀 Deployment Guide

This guide explains how to deploy and run the Indic Cognitive Speech Agent.

---

## Option 1: Gradio UI (Recommended)

The easiest way to use the system is through the Gradio web interface.

### Local Machine

```bash
cd Speech_agent
source venv/bin/activate
python app.py
# → Opens at http://localhost:7860
# → Also generates a public share link (share=True)
```

### SLURM Cluster (MBZUAI)

#### Interactive Mode

```bash
# Request a GPU node
srun --partition=gpu --gres=gpu:1 --mem=32G --time=04:00:00 --pty bash

# Inside the node
cd /path/to/Speech_agent
source venv/bin/activate
python app.py
# Note the public share link from Gradio output
```

#### Batch Mode (Background)

```bash
# Use the provided worker script
bash run_worker.sh

# Or submit via sbatch
sbatch train.sbatch
```

#### Port Forwarding (Alternative to share link)

```bash
# From your local machine
ssh -L 7860:gpu-node:7860 user@cluster.mbzuai.ac.ae

# Then open http://localhost:7860 in your browser
```

---

## Option 2: Command-Line Interface

### Single Audio File

```bash
python src/speech_agent.py path/to/audio.wav
```

### With Environment Setup

```bash
# Ensure .env has your API key
echo 'WANDB_API_KEY=your_key' > .env

# Run with automatic language detection
python src/speech_agent.py demo_marathi.wav
```

---

## Option 3: Python API

```python
from src.speech_agent import CognitiveSpeechAgent

# Initialize (loads Whisper + VAD + LLM)
agent = CognitiveSpeechAgent()

# Optionally load LoRA adapters
agent.load_language_adapter("artifacts/whisper-marathi-lora-indicspeech/checkpoint-1000", "mr")

# Process audio
result = agent.process_audio("audio.wav")
print(result["raw_stt"])
print(result["agent_final_response"])
```

### TTS Only

```python
from src.tts_engine import IndicTTSEngine

engine = IndicTTSEngine()
result = engine.synthesize(
    text="नमस्कार, हे मराठी भाषण आहे.",
    language="mr",
    voice_preset="clear_female",
    output_path="output.wav",
)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WANDB_API_KEY` | Optional | Enables LLM translation via W&B Inference |
| `HUGGINGFACE_TOKEN` | Optional | For downloading gated datasets |
| `GRADIO_TEMP_DIR` | Auto-set | Temp directory for Gradio uploads |

These are set in `.env` and loaded automatically by `app.py`.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 8 GB (STT only) | 16+ GB (STT + TTS) |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 10 GB (models) | 20 GB (models + data) |

**Note**: The TTS model (`ai4bharat/indic-parler-tts`) requires ~4 GB additional VRAM. If GPU memory is limited, use `--skip_tts` in the evaluation script and run STT-only mode.

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - The full system (Whisper + TTS) needs ~12 GB VRAM
   - Solution: Run STT and TTS separately, or use a smaller Whisper model

2. **Gradio `/tmp` Permission Error (SLURM)**
   - `app.py` already handles this by setting `GRADIO_TEMP_DIR` to `.gradio/`
   - If it persists: `export TMPDIR=/path/to/Speech_agent/.gradio`

3. **LLM Not Working**
   - Check that `WANDB_API_KEY` is set in `.env`
   - The `openai` Python package must be installed
   - The system works without LLM — raw transcription is still produced

4. **TTS Import Error**
   - Install: `pip install git+https://github.com/huggingface/parler-tts.git`
   - The `parler_tts` package is not on PyPI; install from GitHub

5. **Slow Inference on CPU**
   - Expected: CPU inference is 5-10x slower than GPU
   - Solution: Use a GPU node or Google Colab
