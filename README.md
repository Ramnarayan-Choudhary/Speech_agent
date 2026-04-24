# 🎙️ Indic Cognitive Speech Agent

> A bidirectional speech-to-speech AI system with multilingual transcription,
> LLM-powered translation, and neural text-to-speech synthesis for Indian languages.

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Audio Input │ ──► │ Silero VAD│ ──► │ Whisper LID  │ ──► │ Whisper STT  │ ──► │  Qwen3-14B   │
│  (mic/file)  │     │ (denoise) │     │ (route lang) │     │  + LoRA      │     │  (W&B API)   │
└──────────────┘     └───────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                                        │
                                                              ┌─────────────────────────▼─────┐
                                                              │ [CLEANED]: corrected text     │
                                                              │ [ENGLISH]: English translation │
                                                              └─────────────────────────┬─────┘
                                                                                        │
                                                              ┌─────────────────────────▼─────┐
                                                              │ Indic Parler-TTS (AI4Bharat)  │
                                                              │ → Synthesized speech output    │
                                                              └───────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Base STT Model** | `openai/whisper-large-v3-turbo` (809M params) |
| **Fine-tuning** | LoRA via PEFT (rank=8, alpha=16) |
| **Language Detection** | Domain-constrained decoder-logit routing (en, mr, gu) |
| **Voice Activity Detection** | Silero VAD (silence filtering) |
| **LLM Translation** | Qwen3-14B via Weights & Biases Inference API |
| **Text-to-Speech** | AI4Bharat Indic Parler-TTS (21+ languages) |
| **Training Data** | Common Voice 17.0 + IndicSpeech |
| **Evaluation** | FLEURS benchmark (WER/CER/BLEU) |
| **UI** | Gradio Blocks (STT + TTS tabs) |
| **Cluster** | MBZUAI SLURM (A100 GPUs) |
| **Tracing** | Weights & Biases Weave |
| **Framework** | PyTorch 2.x, Transformers, PEFT, Accelerate |

### 🗣️ Supported Languages
- **Marathi** (मराठी) — LoRA fine-tuned
- **Gujarati** (ગુજરાતી) — LoRA fine-tuned
- **English** — Base Whisper model
- **Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia, Assamese** — TTS support

## 🚀 Quick Start

### Prerequisites

```bash
python --version   # Python 3.10+
git clone https://github.com/Ramnarayan-Choudhary/Speech_agent.git
cd Speech_agent
pip install -r requirements.txt
```

### Run the Gradio Demo

```bash
# Set your W&B API key for LLM translation (optional but recommended)
echo 'WANDB_API_KEY=your_key_here' > .env

# Launch the full UI (STT + TTS tabs)
python app.py
# → Opens at http://localhost:7860 with a public share link
```

### Run on SLURM Cluster

```bash
# Submit as a batch job
sbatch train.sbatch

# Or use the worker script
bash run_worker.sh
```

### Run the Evaluation Suite

```bash
# Quick evaluation (5 samples, ~5 min)
python scripts/evaluate_full_pipeline.py --num_samples 5 --skip_tts

# Full evaluation (50 samples, ~30 min)
python scripts/evaluate_full_pipeline.py --num_samples 50

# Results saved to: eval_results/full_pipeline_report.json
```

### Interactive Demo Notebook

```bash
# Open in VS Code or Jupyter (uses # %% cell markers)
code notebooks/full_pipeline_demo.py
```

## 📁 Repository Structure

```
Speech_agent/
├── app.py                           ← Gradio UI (STT + TTS tabs)
├── requirements.txt                 ← Python dependencies
├── .env                             ← API keys (WANDB_API_KEY)
│
├── src/
│   ├── speech_agent.py              ← CognitiveSpeechAgent (main brain)
│   ├── tts_engine.py                ← IndicTTSEngine (Parler-TTS wrapper)
│   └── speech_to_text_finetune/
│       ├── config.py                ← Pydantic config (LoRA params)
│       ├── data_process.py          ← Dataset loading & preprocessing
│       ├── finetune_whisper.py      ← LoRA fine-tuning script
│       ├── evaluate_whisper_fleurs.py ← Basic FLEURS evaluator
│       ├── inference.py             ← Standalone inference pipeline
│       └── utils.py                 ← WER/CER metrics, model cards
│
├── scripts/
│   ├── evaluate_full_pipeline.py    ← Comprehensive 5-dim evaluation suite
│   ├── convert_indicspeech.py       ← Dataset conversion for IndicSpeech
│   ├── download_cv_datasets.py      ← Download Common Voice data
│   ├── download_fleurs.py           ← Download FLEURS eval data
│   └── ...
│
├── notebooks/
│   ├── full_pipeline_demo.py        ← Full agentic loop demo (9 sections)
│   ├── colab_indic_asr_training.ipynb ← Colab training notebook
│   ├── colab_data_preparation.ipynb   ← Data prep helper
│   ├── colab_evaluation.ipynb         ← Evaluation runner
│   └── local_inference_demo.ipynb     ← Local testing
│
├── example_configs/
│   ├── marathi/                     ← CPU/GPU/dryrun YAML configs
│   ├── gujarati/
│   └── hindi/
│
├── artifacts/
│   ├── whisper-marathi-lora-indicspeech/  ← Trained LoRA weights
│   └── whisper-gujarati-lora-indicspeech/ ← Trained LoRA weights
│
├── eval_results/                    ← JSON evaluation reports
├── docs/                           ← Guides (getting-started, training, eval, deployment)
├── tests/                          ← Unit, integration, e2e tests
│
├── IMPLEMENTATION_PLAN.md           ← 4-week project roadmap
├── INFRASTRUCTURE.md                ← Cloud & cluster architecture
├── VERIFICATION_REPORT.md           ← Component verification audit
├── START_HERE.md                    ← Quick reference guide
└── README.md                        ← This file
```

## 🔑 Key Features

### 🧠 Level 1: Cognitive STT with Auto-Routing
- **Domain-constrained LID**: Restricts Whisper's language detection to `{en, mr, gu}` for high accuracy
- **LoRA adapter switching**: Automatically swaps fine-tuned adapters based on detected language
- **Silero VAD**: Pre-filters silence and noise before transcription

### 🤖 Level 2: LLM Translation & Cleaning
- **Phonetic error correction**: LLM fixes acoustic model misinterpretations
- **Bidirectional translation**: Native → English (with structured output format)
- **RAG context grounding**: Optional context injection for domain-specific disambiguation

### 🔊 Level 3: Multilingual TTS
- **Indic Parler-TTS**: Neural TTS for 21+ Indian languages
- **Voice presets**: Control speaker gender, pace, and quality via description prompts
- **Full loop**: Audio in → transcribe → translate → synthesize → audio out

### ⚡ LoRA Fine-Tuning
- **Memory**: 2.4GB → 600MB (75% reduction)
- **Speed**: 3-5x faster training
- **Quality**: 95-98% of full fine-tuning performance

## 📊 Evaluation Framework

The comprehensive evaluation suite (`scripts/evaluate_full_pipeline.py`) measures:

| Dimension | Metric | Method |
|-----------|--------|--------|
| **STT Accuracy** | WER, CER | FLEURS test set comparison |
| **Language Routing** | Accuracy % | Labeled FLEURS → detected language |
| **LLM Translation** | BLEU | English output vs FLEURS English ref |
| **TTS Health** | Pass/Fail + duration | Synthesize per-language samples |
| **Latency** | Avg/Median/P95 seconds | End-to-end timing profile |

Run `python scripts/evaluate_full_pipeline.py --help` for all options.

## 👥 Project Details

- **Course**: Intro to Speech Processing at MBZUAI
- **Developer**: Ramnarayan Choudhary
- **Email**: choudharyramnarayan123@gmail.com

## 🔐 License

MIT License — See [LICENSE](LICENSE)

---

**Status**: 🚀 Active Development | **Last Updated**: April 2026
