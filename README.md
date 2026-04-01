# 🎙️ Indic Multilingual Speech-to-Text (STT) Project

## 🎯 Project Overview

A **4-week MBZUAI course project** (60% grade weight) for efficient multilingual speech-to-text systems in **Indic languages** using:
- **Whisper-small** (244M params) from OpenAI
- **LoRA fine-tuning** (60% memory savings via PEFT)
- **Agentic learning** (continuous improvement loop)
- **Multilingual support** (Marathi, Gujarati, Hindi)

### 🗣️ Supported Languages
- **Marathi** (मराठी) - Primary focus
- **Gujarati** (ગુજરાતી) - Secondary  
- **Hindi** (हिंदी) - Optional

## 👥 Project Details
- **Course**: Intro to Speech Processing at MBZUAI
- **Grade Weight**: 60% of overall score
- **Timeline**: 4 weeks
- **Developer**: Ramnarayan Choudhary

## 🎯 Key Objectives

1. ✅ **Efficient Fine-Tuning**: Implement LoRA to achieve 3-5x speedup, 60% memory reduction
2. ✅ **Multilingual System**: Language-agnostic pipeline supporting 3+ Indic languages
3. ✅ **Agentic Intelligence**: Build feedback loop for continuous model improvement
4. ✅ **Benchmarked Evaluation**: WER/CER metrics on FLEURS multilingual benchmark
5. ✅ **Production Ready**: Containerized, documented, deployable

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Base ASR Model** | OpenAI Whisper-small (244M parameters) |
| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) via PEFT |
| **Training Data** | Common Voice 17.0 + IndicVoices dataset |
| **Evaluation** | FLEURS benchmark (multilingual) |
| **GPU Resources** | Google Colab Pro (V100/A100) or MBZUAI Lab |
| **Storage Backend** | Google Drive + HuggingFace Hub |
| **Inference UI** | Gradio app + FastAPI (optional)
| **ML Framework** | PyTorch 2.0+, Transformers 4.30+, Accelerate |
| **Language** | Python 3.10+ |

## 📊 Performance Targets

```
Model: Whisper-small
Training: LoRA adaptation with rank=8, alpha=16
Memory (LoRA): ~600MB vs 2.4GB (full fine-tuning)
Training Time: 4-6 hours per language (Google Colab A100)
Speedup: ~30% faster than full fine-tuning
Target WER: <15% for Marathi on FLEURS
Expected Improvement: ~3-4% WER reduction vs baseline
```

## 🚀 Quick Start

### Prerequisites
```bash
python --version  # Python 3.10+
git clone https://github.com/Ramnarayan-Choudhary/Speech_agent.git
cd Speech_agent
pip install -r requirements.txt
export HUGGINGFACE_TOKEN="your_hf_token_here"
```

### Training on Google Colab (Recommended)

1. Open [Colab Training Notebook](notebooks/colab_indic_asr_training.ipynb)
2. Click "Open in Colab"
3. Toggle GPU: Runtime → Change runtime type → GPU (V100 or A100)
4. Mount Google Drive
5. Run all cells

### Local Training

```bash
# Download dataset
python scripts/download_cv_datasets.py --language mr --output data/raw/marathi

# Train model with LoRA
python src/speech_to_text_finetune/finetune_whisper.py \
    --config example_configs/marathi/config_lora_gpu.yaml

# Evaluate model
python src/speech_to_text_finetune/evaluate_whisper_fleurs.py \
    --model_id checkpoints/marathi_lora \
    --language mr_in
```

## 📁 Repository Structure

```
Speech_agent/
├── README.md                    ← Project overview
├── IMPLEMENTATION_PLAN.md       ← 4-week detailed roadmap
├── INFRASTRUCTURE.md            ← Cloud architecture & data strategy
├── requirements.txt             ← Python dependencies
├── setup.py                     ← Package installation
├── environment.yml              ← Conda environment file
├── .gitignore                   ← Git ignore patterns
├── LICENSE                      ← MIT License
│
├── src/speech_to_text_finetune/
│   ├── __init__.py
│   ├── config.py                # Pydantic config schema (LoRA params)
│   ├── data_process.py          # Dataset loading & processing
│   ├── finetune_whisper.py      # LoRA training script (PEFT integration)
│   ├── evaluate_whisper_fleurs.py # WER/CER evaluation
│   ├── inference.py             # Inference pipeline + language detection
│   └── utils.py                 # Helper functions
│
├── example_configs/
│   ├── marathi/
│   │   ├── config_lora_cpu.yaml   # CPU training config
│   │   ├── config_lora_gpu.yaml   # GPU training config  
│   │   └── README.md              # Config documentation
│   ├── gujarati/
│   ├── hindi/
│
├── notebooks/
│   ├── colab_indic_asr_training.ipynb    # Main training (Colab)
│   ├── colab_data_preparation.ipynb      # Data prep helper
│   ├── colab_evaluation.ipynb            # Evaluation runner
│   └── local_inference_demo.ipynb        # Local testing
│
├── scripts/
│   ├── download_cv_datasets.py           # Download Common Voice
│   ├── download_fleurs.py                # Download FLEURS eval data
│   ├── setup_google_drive.py             # Google Drive setup
│   ├── backup_to_drive.py                # Auto-backup checkpoints
│   ├── prepare_custom_dataset.py         # Custom data formatter
│   └── convert_to_gguf.py                # Model quantization
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── training-guide.md
│   ├── evaluation-guide.md
│   └── deployment.md
│
└── data/
    ├── raw/{marathi,gujarati,hindi}/
    └── processed/{marathi,gujarati,hindi}/
```

## 🔑 Key Features

### ⚡ LoRA Fine-Tuning (Parameter-Efficient)
- Memory: 2.4GB → 600MB (75% reduction)
- Speed: 3-5x faster training
- Quality: 95-98% of full fine-tuning
- Easy: Drop-in replacement

### 🗣️ Multilingual Router
- Automatic language detection from audio
- Route to language-specific fine-tuned model
- Fallback to base Whisper if needed

### 🤖 Agentic Learning Loop
User Audio → STT → NLP → TTS → User Feedback → Continuous Improvement

## 📈 Evaluation Framework

| Metric | Dataset | Target |
|--------|---------|--------|
| **WER** | FLEURS | <15% |
| **CER** | FLEURS | <8% |
| **Latency** | Colab GPU | <2 seconds |
| **Memory** | Training | <2GB |

## 🖥️ Hardware Requirements

| Scenario | RAM | GPU | Training Time |
|----------|-----|-----|----------------|
| **CPU Only** | 8GB | None | 4-6 hours |
| **Colab Free** | 16GB | T4 | 2-3 hours |
| **Colab Pro (V100)** | 16GB | V100 | 30-45 min |
| **Colab Pro (A100)** | 24GB | A100 | 15-20 min |
| **MBZUAI GPU** | 24GB | A100 | 5-10 min |

## 📋 Implementation Timeline

```
Week 1: Foundation & Prototyping
Week 2: Data Preparation & LoRA Integration
Week 3: GPU Training & Multi-Language Models
Week 4: Agentic System & Deployment
```

See `IMPLEMENTATION_PLAN.md` for detailed breakdown.

## 🔐 License

MIT License - See LICENSE file

## 📞 Contact & Support

- **Email**: choudharyramnarayan123@gmail.com
- **GitHub**: https://github.com/Ramnarayan-Choudhary
- **Course**: Intro to Speech Processing (MBZUAI)

---

**Status**: 🚀 Active Development | **Last Updated**: April 2026
