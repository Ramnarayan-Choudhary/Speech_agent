# 🏗️ Speech Agent: Infrastructure & Architecture Guide

**Document Purpose**: Complete technical architecture, data pipeline, GPU resources, and deployment strategy for Indic multilingual ASR project.

---

## 📐 System Architecture

### 3-Tier Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE & DEMO TIER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Gradio UI  │  │  FastAPI     │  │  Colab Notebook      │   │
│  │  localhost   │  │  (Docker)    │  │  (Web-based)         │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│         ↓                  ↓                      ↓               │
└─────────────────────────────────────────────────────────────────┘
         │                   │                      │
         └───────────────────┴──────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   INFERENCE ENGINE TIER         │
        │  ┌─────────────────────────┐   │
        │  │  WhisperInference Class │   │
        │  │  (CPU/GPU support)      │   │
        │  │  Batch processing       │   │
        │  └─────────────────────────┘   │
        └────────────────┬────────────────┘
                         │
    ┌────────────────────┴────────────────────┐
    │     MODEL STORAGE TIER                   │
    │  ┌──────────────────────────────────┐   │
    │  │  GPU Memory (0.6GB per model)    │   │
    │  │  OR Disk Cache (GGUF quantized)  │   │
    │  └──────────────────────────────────┘   │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────┴────────────────────────┐
    │    CHECKPOINT & TRAINING TIER               │
    │  ┌──────────────────────────────────────┐   │
    │  │ Local: ./checkpoints/                │   │
    │  │ - marathi_lora/                      │   │
    │  │ - gujarati_lora/                     │   │
    │  │ - hindi_lora/                        │   │
    │  │                                      │   │
    │  │ Google Drive:                        │   │
    │  │ - Backup checkpoints (~1.5GB)        │   │
    │  │ - Training logs                      │   │
    │  └──────────────────────────────────────┘   │
    └────────────────────┬────────────────────────┘
                         │
    ┌────────────────────┴────────────────────────┐
    │   DATA STORAGE TIER                         │
    │  ┌──────────────────────────────────────┐   │
    │  │ Local (./data/):                     │   │
    │  │ - raw/ (processed CV 17.0)           │   │
    │  │ - processed/ (model-ready data)      │   │
    │  │ - FLEURS/ (evaluation set)           │   │
    │  │                                      │   │
    │  │ Google Drive:                        │   │
    │  │ - Cache for repeated downloads       │   │
    │  │ - Backup of processed datasets       │   │
    │  │                                      │   │
    │  │ HuggingFace Hub:                     │   │
    │  │ - Final models (public share)        │   │
    │  │ - Dataset cards & documentation      │   │
    │  └──────────────────────────────────────┘   │
    └────────────────────────────────────────────┘
```

---

## 💾 Data Pipeline - Detailed

### Stage 1: Dataset Acquisition

**Common Voice 17.0 Download**
```
Source: mozilla-foundation/common_voice_17_0 (HF Hub)
├── Marathi (mr)
│   ├── Train: ~1,500 samples
│   ├── Test: ~400 samples  
│   ├── Size: ~12GB total
│   └── Duration: 50+ hours audio
├── Gujarati (gu)
│   ├── Train: ~1,200 samples
│   ├── Size: ~10GB total
│   └── Duration: 40+ hours audio
└── Hindi (hi)
    ├── Train: ~1,800 samples
    ├── Size: ~14GB total
    └── Duration: 60+ hours audio
```

**Script**: `scripts/download_cv_datasets.py`
```bash
python scripts/download_cv_datasets.py \
    --language marathi \
    --output data/raw/marathi \
    --sample-size 500  # Limit for faster iteration
```

**Storage**: `data/raw/{language}/`
**Disk Usage**: ~300MB for subset (500 samples × 3 languages)

---

### Stage 2: Data Preprocessing

**Processing Pipeline**:
```
Raw Audio (16kHz WAV)
    ↓
[Librosa] Audio normalization & resampling
    ↓
[Tokenizer] Text tokenization with special tokens
    ↓
[Processor] Feature extraction (Mel-spectrograms)
    ↓
[Format] PyTorch TensorDataset
    ↓
Cached Dataset (Parquet)
```

**Implementation**: `src/speech_to_text_finetune/data_process.py`

**Code Example**:
```python
def process_dataset(dataset, processor, batch_size=16):
    def preprocess_data(sample):
        audio = sample["audio"]
        sample["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        sample["labels"] = processor.tokenizer(sample["sentence"]).input_ids
        return sample
    
    return dataset.map(preprocess_data, batch_size=batch_size, num_proc=4)
```

**Output**: `data/processed/{language}/`
**Disk Usage**: ~150MB per language (cached tensors)

---

### Stage 3: Training Data Distribution

**Train/Test Split**: 80/20
```
Marathi:
├── Train: 400 samples → ~4000 steps (batch_size=16, grad_accum=2)
└── Test: 100 samples → Validation every 500 steps

Gujarati:
├── Train: 400 samples
└── Test: 100 samples

Hindi:
├── Train: 400 samples
└── Test: 100 samples
```

---

## 🖥️ GPU Resource Requirements

### Hardware Specifications

| Component | Config | Details |
|-----------|--------|---------|
| **GPU Type** | Preferred | NVIDIA A100 (80GB) |
| **Fallback** | Acceptable | V100 (32GB) or T4 (16GB) |
| **CPU** | Minimum | 4 cores, 8GB RAM |
| **Storage** | Minimum | 50GB SSD |

### Memory Breakdown

**Full Fine-tuning (No LoRA)**:
```
Model weights:      2.4GB (fp32) / 1.2GB (fp16)
Optimizer states:   2.4GB (Adam with momentum)
Gradients:          1.2GB
Batch data:         0.8GB (batch_size=16)
────────────────────────────
Total:             ~6.8GB (fp16)
```

**LoRA Fine-tuning**:
```
Model weights:      2.4GB (frozen)
LoRA adapters:      ~50MB per module
Optimizer states:   0.2GB (only adapters)
Gradients:          ~50MB
Batch data:         0.8GB
────────────────────────────
Total:             ~0.6GB ✅ 90% reduction!
```

### GPU Utilization Estimates

| Task | GPU Required | Duration | Cost (Cloud) |
|------|--------------|----------|-------------|
| Data prep | Optional | 30 min | $0 |
| Model training (1 lang, 3 epochs) | Required | 4-6 hrs | $15-20 |
| All 3 models (parallel) | 3×GPU | 6-8 hrs total | $45-60 |
| Evaluation | Optional | 1-2 hrs | $0-5 |
| **Total** | | **~15-20 hrs** | **$60-85** |

### CPU-Only Alternative

For those without GPU access, fall back to OpenAI Whisper-tiny:
```
Model size: 39M parameters (vs 244M for small)
Memory: ~400MB with LoRA
Training time: 2-3x slower
Accuracy: ~10-15% higher WER (acceptable for MVP)
```

---

## ☁️ Cloud Infrastructure Options

### Option 1: Google Colab (Recommended)

**Pros**:
- Free GPU access (T4 or V100)
- Colab Pro: A100 GPUs
- Pre-installed ML libraries
- Easy notebook sharing
- Integrated Drive storage

**Cons**:
- Session timeout (12 hours)
- Limited storage (100GB free)
- Restart required regularly

**Setup**:
```python
# In Colab cell
from google.colab import drive
drive.mount('/content/drive')

# Then train
%cd /tmp
!git clone <repo>
%cd Speech_agent
!python src/speech_to_text_finetune/finetune_whisper.py \
    --config example_configs/marathi/config_lora_gpu.yaml
```

**Cost**: $0 (free tier) or $10/month (Pro with A100)

---

### Option 2: MBZUAI Lab GPU (If Available)

**Pros**:
- On-campus A100 GPUs
- No timeout limits
- Fast NVMe storage
- Network bandwidth

**Cons**:
- Limited availability
- May need scheduling
- Lab-specific setup

**Setup**:
```bash
# SSH into lab
ssh -i ~/.ssh/mbzuai_key user@lab-gpu-node

# Run training on permanent storage
python finetune_whisper.py --config example_configs/marathi/config_lora_gpu.yaml
```

---

### Option 3: Kaggle Notebooks (Free GPU)

**Pros**:
- 2 free GPUs (P100)
- 20GB RAM
- Free tier available
- Good for competition-style projects

**Cons**:
- 9-hour session limit
- 20GB output limit
- Less community for ASR

---

### Option 4: AWS EC2 (Self-pay)

**Instance Type**: `g4dn.2xlarge` (1×T4 GPU, 8 vCPU, 32GB RAM)
```bash
# Estimated cost: $0.53/hour

# Spot instance: $0.15/hour (75% savings, but interruptible)
```

---

## 📊 Cost Comparison Matrix

| Platform | GPU Type | Cost/Hour | Monthly (24/7) | Training Cost |
|----------|----------|-----------|----------------|---------------|
| Colab Free | T4 | $0 | $0 | $0 |
| Colab Pro | A100 | $1.87 | $1350 | $50-100 |
| MBZUAI Lab | A100 | $0 | $0 | $0 |
| Kaggle | P100 | $0 | $0 | $0 |
| AWS (on-demand) | T4 | $0.35 | $252 | $10-15 |
| AWS (spot) | T4 | $0.10 | $72 | $3-5 |

**Recommendation**: Use Colab Free (T4) for MVP, upgrade to Colab Pro (A100) for production.

---

## 🔄 Model Storage & Versioning

### Local Storage

```
./checkpoints/
├── marathi_lora/              # Best Marathi model
│   ├── adapter_config.json
│   ├── adapter_model.bin      # LoRA weights only (~10MB)
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin      # Full model
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.model
│   └── training_args.bin
├── gujarati_lora/
└── hindi_lora/

./models/
├── marathi.gguf               # Quantized (CPU inference)
├── gujarati.gguf
└── hindi.gguf
```

### Google Drive Backup

```
MyDrive/
└── speech_agent_workspace/
    ├── models/
    │   ├── marathi_ckpt_*.pt  # Full checkpoints
    │   ├── gujarati_ckpt_*.pt
    │   └── hindi_ckpt_*.pt
    ├── logs/
    │   ├── marathi_training.log
    │   ├── gujarati_training.log
    │   └── hindi_training.log
    └── datasets/
        ├── common_voice_mr_cache/
        ├── common_voice_gu_cache/
        └── common_voice_hi_cache/

Total estimated: ~50GB
```

### Hugging Face Hub

```
huggingface.co/
├── whisper-marathi-lora-small/
│   ├── README.md             # Model card
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer.*
├── whisper-gujarati-lora-small/
└── whisper-hindi-lora-small/
└── speech-agent-datasets/
    ├── common_voice_marathi_subset
    ├── common_voice_gujarati_subset
    └── common_voice_hindi_subset

Access: https://huggingface.co/<username>/whisper-marathi-lora-small
```

### Model Versioning Strategy

```
v1.0: Initial training (500 samples, 3 epochs)
├── Metrics: WER=15.3%, CER=4.8%
├── Checkpoint: marathi_lora/checkpoint-5000

v1.1: Data augmentation (added 200 more samples)
├── Metrics: WER=14.1%, CER=4.2%
├── Changes: +data augmentation + warmup_steps 500→1000

v1.2: Hyperparameter tuning (lora_rank 8→16)
├── Metrics: WER=12.8%, CER=3.9%
├── Changes: rank=16, alpha=32, learning_rate 1e-4→1e-3

v2.0: Production release (final models)
└── Metrics: WER<12%, ready for deployment
```

**Branching Strategy**:
```bash
git checkout -b training/marathi-v1
# Make training modifications
git commit -m "Marathi training v1: WER=15.3%"
git push origin training/marathi-v1

git checkout main
git merge training/marathi-v1 --no-ff
git tag v1.0-marathi
```

---

## 📡 Inference Deployment Architecture

### Local Deployment

```
┌─────────────────────────────────┐
│   Gradio Web Interface          │
│   localhost:7860                │
├─────────────────────────────────┤
│   WhisperInference Engine       │
│   (CPU or GPU)                  │
├─────────────────────────────────┤
│   Cached Models                 │
│   ~/models/whisper-mr-lora/     │
└─────────────────────────────────┘
```

**Usage**:
```bash
python -m streamlit run demo.py
# Or
jupyter notebook demo_inference.ipynb
```

---

### Server Deployment (FastAPI)

```
┌────────────────────────────────────────┐
│         Load Balancer (nginx)          │
│         localhost:8000                 │
├────────────────────────────────────────┤
│    FastAPI Application Server          │
│  ┌──────────────────────────────────┐  │
│  │ POST /transcribe                 │  │
│  │ GET /health                      │  │
│  │ GET /models                      │  │
│  │ POST /evaluate                   │  │
│  └──────────────────────────────────┘  │
├────────────────────────────────────────┤
│    WhisperInference (3 instances)      │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Marathi  │ │ Gujarati │ │ Hindi  │ │
│  └──────────┘ └──────────┘ └────────┘ │
├────────────────────────────────────────┤
│   Model Cache (Redis or Local)         │
│   Memory: ~2GB (all 3 models)          │
└────────────────────────────────────────┘
```

**FastAPI Code**:
```python
from fastapi import FastAPI, UploadFile
from src.speech_to_text_finetune.inference import WhisperInference

app = FastAPI()
inference_mr = WhisperInference("marathi_lora")
inference_gu = WhisperInference("gujarati_lora")
inference_hi = WhisperInference("hindi_lora")

@app.post("/transcribe")
async def transcribe(audio: UploadFile, language: str = "marathi"):
    if language == "marathi":
        result = inference_mr.transcribe(audio.file)
    elif language == "gujarati":
        result = inference_gu.transcribe(audio.file)
    else:
        result = inference_hi.transcribe(audio.file)
    
    return {"text": result["text"], "language": language}

@app.get("/health")
async def health():
    return {"status": "ok", "models": ["mr", "gu", "hi"]}
```

**Deployment**:
```bash
# Docker
docker build -t speech-agent-api .
docker run -p 8000:8000 speech-agent-api

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY checkpoints/ checkpoints/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### HuggingFace Spaces Deployment

Deploy directly to HF Spaces with Gradio:

```python
# app.py
import gradio as gr
from transformers import pipeline

mr_pipe = pipeline("automatic-speech-recognition", 
                   model="<user>/whisper-marathi-lora")

interface = gr.Interface(
    fn=mr_pipe,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Indic Speech-to-Text",
    theme="soft"
)

interface.launch()
```

**Steps**:
1. Create repo: https://huggingface.co/spaces/new
2. Select "Gradio"
3. Upload `app.py` and `requirements.txt`
4. HF Spaces automatically deploys 🚀

**URL**: `https://huggingface.co/spaces/<user>/indic-asr`

---

## 🔐 Security & Authentication

### API Key Management

```python
# .env file (add to .gitignore)
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
OPENAI_API_KEY=sk_xxxxxxxxxxxxx
DATABASE_URL=postgresql://user:pass@localhost/db
```

```python
# Load in code
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
```

### API Rate Limiting

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.util import get_remote_address

@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe(request: Request, audio: UploadFile):
    # Limited to 10 requests per minute
    ...
```

---

## 📈 Performance Monitoring

### Metrics to Track

```python
# During training
training_loss = []
validation_wer = []
validation_cer = []
inference_time = []

# Logging with Weights & Biases
import wandb

wandb.init(project="indic-asr")
wandb.log({
    "epoch": epoch,
    "training_loss": loss,
    "validation_wer": wer,
    "validation_cer": cer,
    "inference_time_ms": inference_time * 1000
})
```

### Prometheus Metrics (Production)

```python
from prometheus_client import Counter, Histogram, Gauge

transcription_counter = Counter(
    'transcriptions_total',
    'Total transcriptions',
    ['language', 'status']
)

inference_duration = Histogram(
    'inference_duration_seconds',
    'Inference latency',
    ['language']
)

model_memory = Gauge(
    'model_memory_bytes',
    'Model memory usage',
    ['language']
)
```

---

## 🚨 Troubleshooting Guide

### Issue: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions** (in order):
1. Reduce batch size: `per_device_train_batch_size: 16 → 8`
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Use Whisper-tiny instead of small
4. Reduce max_steps or num_epochs

### Issue: Slow Data Loading

**Symptoms**: GPU utilization <30%, training slow

**Solution**:
```python
# Use multiple workers
DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,  # Increase based on CPU cores
    pin_memory=True
)
```

### Issue: Model Not Improving

**Symptoms**: Validation loss plateaus

**Solutions**:
1. Check learning rate: try 1e-4 or 1e-5
2. Increase warmup_steps: 500 → 1000
3. Collect more diverse data
4. Try different LoRA rank: 8 → 16

---

## 📋 Pre-Deployment Checklist

- [ ] All tests passing: `pytest tests/ -v`
- [ ] Models stored in both local and HF Hub
- [ ] API health check working
- [ ] Load testing completed (1000+ requests/min)
- [ ] Error handling tested (corrupt audio, timeout, etc.)
- [ ] Documentation complete
- [ ] Security review done (no credentials in code)
- [ ] Backup strategy in place
- [ ] Monitoring dashboards up

---

## 🔗 Resource Links

- PyTorch Distributed Training: https://pytorch.org/docs/stable/distributed.html
- HuggingFace Accelerate: https://huggingface.co/docs/accelerate
- PEFT LoRA Guide: https://huggingface.co/docs/peft
- FastAPI Tutorial: https://fastapi.tiangolo.com/
- Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-XX  
**Status**: Ready for Implementation
