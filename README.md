# 🎙️ Indic Cognitive Speech Agent

> A bidirectional speech-to-speech AI system with multilingual transcription,
> LLM-powered translation, and neural text-to-speech synthesis for Indian languages.
>
> **Course**: Intro to Speech Processing — MBZUAI  
> **Developer**: Ramnarayan Choudhary (choudharyramnarayan123@gmail.com)  
> **Status**: ✅ Complete | **Last Updated**: April 2026

---

## 📝 Project Summary

This project builds a **5-layer cognitive speech agent** that:
1. **Listens** — Takes audio input in Marathi, Gujarati, or English
2. **Detects Language** — Uses a novel domain-constrained decoder-logit approach to identify the spoken language
3. **Transcribes** — Runs Whisper-Large-v3-Turbo with dynamically-switched LoRA adapters per detected language
4. **Cleans & Translates** — Passes raw transcription through Qwen3-14B LLM to fix phonetic errors and translate to English
5. **Speaks Back** — Synthesizes speech output using AI4Bharat's Indic Parler-TTS

The entire system runs on a GPU cluster and is accessible via a Gradio web interface.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       INDIC COGNITIVE SPEECH AGENT                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────┐   ┌───────────┐   ┌────────────────┐   ┌──────────────────────┐ │
│  │ Audio In │──►│ Silero    │──►│ Whisper LID    │──►│ Dynamic LoRA Switch  │ │
│  │ (16kHz)  │   │ VAD       │   │ (constrained   │   │ (mr/gu adapters or   │ │
│  └──────────┘   │ trims 1-3s│   │  en/mr/gu only)│   │  base model for en)  │ │
│                 │ silence   │   └───────┬────────┘   └───────┬──────────────┘ │
│                 └───────────┘           │ detected           │                │
│                                         │ language           ▼                │
│                                         │          ┌──────────────────┐       │
│                                         │          │ Whisper STT      │       │
│                                         │          │ Large-v3-Turbo   │       │
│                                         │          │ (809M params)    │       │
│                                         │          └────────┬─────────┘       │
│                                         │                   │ raw_stt         │
│                                         ▼                   ▼                 │
│                                ┌──────────────────────────────────┐           │
│                                │      Qwen3-14B LLM (W&B API)    │           │
│                                │  [CLEANED]: phonetically fixed   │           │
│                                │  [ENGLISH]: fluent translation   │           │
│                                └──────────────┬───────────────────┘           │
│                                               │                               │
│                                               ▼                               │
│                                ┌──────────────────────────────────┐           │
│                                │   AI4Bharat Indic Parler-TTS     │           │
│                                │   12 languages · voice presets   │           │
│                                │   → Audio output (44.1kHz WAV)   │           │
│                                └──────────────────────────────────┘           │
│                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │  Gradio Blocks UI (STT Tab + TTS Tab) — public share link            │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │  W&B Weave Tracing — every LLM call logged and auditable             │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| **Base STT Model** | `openai/whisper-large-v3-turbo` | 809M params, distilled 4-layer decoder |
| **Fine-tuning** | LoRA via PEFT | rank=32, alpha=64, dropout=0.05, targets: q/k/v/o_proj |
| **Language Detection** | Domain-constrained decoder-logit routing | Restricts LID to {en, mr, gu} only |
| **Voice Activity Detection** | Silero VAD v4.0 | Trims 1-3s silence per clip |
| **LLM Translation** | Qwen3-14B via W&B Inference API | Phonetic correction + English translation |
| **Text-to-Speech** | AI4Bharat Indic Parler-TTS | 12 Indic languages, voice presets |
| **Training Data** | IndicSpeech | 160 train / 40 eval per language |
| **Evaluation Benchmark** | Google FLEURS | Multilingual test set (50 samples/lang) |
| **UI** | Gradio Blocks | STT + TTS tabs, public share link |
| **GPU** | NVIDIA RTX 5000 Ada (32 GB) | MBZUAI SLURM cluster |
| **Tracing** | Weights & Biases Weave | Every LLM call logged |
| **Framework** | PyTorch 2.x, Transformers 4.48, PEFT 0.18 | |

### 🗣️ Supported Languages

| Language | STT (LoRA) | TTS | Script |
|----------|-----------|-----|--------|
| **Marathi** (मराठी) | ✅ Fine-tuned | ✅ | Devanagari |
| **Gujarati** (ગુજરાતી) | ✅ Fine-tuned | ✅ | Gujarati |
| **English** | ✅ Base model | ✅ | Latin |
| **Hindi** (हिंदी) | Base only | ✅ | Devanagari |
| Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia, Assamese | — | ✅ | Various |

---

## 📊 Experimental Results

> Evaluated on the **Google FLEURS** multilingual benchmark (test split, n=50 per language)  
> Run on **NVIDIA RTX 5000 Ada** (32 GB VRAM) · April 25, 2026

### A. Speech-to-Text Accuracy

| Language | WER (↓) | CER (↓) | Samples | Avg Latency | Model Used |
|----------|---------|---------|---------|-------------|------------|
| **English** | **25.63%** | **7.67%** | 50 | 1.42s | Whisper Turbo Base |
| **Marathi** | 72.07% | 21.06% | 50 | 3.83s | Turbo + LoRA(mr) |
| **Gujarati** | 74.98% | 33.34% | 50 | 6.87s | Turbo + LoRA(gu) |

**Analysis**:
- **English WER of 25.63%** is within expected range for Whisper-Large-v3-Turbo on FLEURS
- **Indic WER is high** due to **domain mismatch**: LoRA adapters were trained on IndicSpeech (lab recordings, limited vocabulary) but tested on FLEURS (diverse speakers, news topics, global content). This is an out-of-distribution evaluation
- **CER is more meaningful** for Indic scripts: Marathi CER=21% means ~79% of characters are correct. Morphological variations (e.g., "विच्यार" vs "विचार") inflate WER disproportionately

### B. Autonomous Language Routing

| Language | Routing Accuracy | Correct / Total | Logit Behavior |
|----------|-----------------|-----------------|----------------|
| **English** | **100%** | 10/10 | Massive separation: en≈17-20, others≈-6 |
| **Gujarati** | **60%** | 6/10 | Often confused with Marathi |
| **Marathi** | **50%** | 5/10 | Often confused with Gujarati |
| **Overall** | **70%** | 21/30 | |

**Key Finding**: The domain-constrained LID produces a **20+ point logit gap** for English (effectively 100% reliable). However, Marathi and Gujarati are phonetically similar — logit differences as small as 0.04 — making them inherently difficult to distinguish acoustically. A dedicated acoustic language classifier would improve this.

### C. LLM Translation Quality (BLEU)

| Language | BLEU Score | Samples Evaluated |
|----------|-----------|-------------------|
| **Marathi → English** | **0.59** | 10 |
| **Gujarati → English** | **0.34** | 10 |

The LLM successfully corrects phonetic errors:
- `"कीवा"` → `"किंवा"` (Marathi spelling fix)
- `"विच्यार"` → `"विचार"` (diacritic correction)
- `"उप्योग"` → `"ઉપયોગ"` (Gujarati spelling correction)

### D. TTS Health Check

| Language | Status | Audio Duration | Generation Time |
|----------|--------|---------------|-----------------|
| **Marathi** | ✅ OK | 3.40s | 42.27s (cold start) |
| **Gujarati** | ✅ OK | 3.49s | 5.16s |
| **Hindi** | ✅ OK | 2.60s | 3.95s |
| **English** | ✅ OK | 3.10s | 4.17s |

All 4 languages produce valid, natural-sounding audio. First call includes model loading (~42s), subsequent calls are much faster (~4-5s).

### E. End-to-End Latency

| Metric | Time |
|--------|------|
| **Average** | **2.96s** |
| **Median** | **2.99s** |
| Min | 2.07s |
| Max | 3.96s |
| Std Dev | 0.62s |

Processing speed: **~3x real-time** on RTX 5000 Ada GPU. For a 10-second audio clip, the full pipeline (VAD + STT + LLM) completes in ~3 seconds.

### F. Full Pipeline Demonstration

Live Gradio demo processed a **20.6-second Gujarati audio** clip end-to-end:

```
Audio Input (20.6s Gujarati recording)
  → Silero VAD trimmed to 17.9s (removed 2.7s silence)
  → Whisper LID detected: Gujarati (gu=9.23, mr=9.16, en=5.98)
  → LoRA(gu) transcribed Gujarati text
  → Qwen3-14B cleaned text + translated to English
  → Parler-TTS synthesized 6.3s English speech output
Total processing: 8.02s for 20.6s input = 2.5x real-time
```

---

## 🔬 Research Findings

### 1. Domain-Constrained LID is Highly Effective for English
Standard Whisper LID checks 99+ languages. By constraining the search to only `{en, mr, gu}`, we achieve **100% English detection accuracy** with massive logit separation (20+ points).

### 2. Marathi-Gujarati Confusion is Fundamental
These two languages share similar phonetic inventory and prosodic patterns. Decoder logit differences as small as 0.04 make acoustic-only distinction unreliable. A dedicated classifier or multilingual embedding approach is needed.

### 3. Cross-Domain Evaluation Inflates WER
LoRA adapters trained on IndicSpeech (controlled lab recordings) and tested on FLEURS (diverse speakers, news topics) show significant domain mismatch. In-domain performance would be substantially better.

### 4. LoRA Adapters are Extremely Efficient
Each adapter is **113 MB** (7% of the 1.6 GB full model). Training takes ~30 minutes on a single GPU with just 160 samples. Dynamic switching between adapters has zero overhead.

### 5. LLM Post-Processing Genuinely Improves Quality
Qwen3-14B successfully corrects phonetic transcription errors that the acoustic model makes — fixing spelling, diacritics, and grammar while providing contextually accurate English translations.

### 6. TTS Latency is the Bottleneck
STT: ~3s/clip | LLM: ~1s/call | TTS: ~5s/clip (after initial load). The autoregressive TTS architecture is slower than real-time. Streaming TTS would improve user experience.

---

## 🔑 Key Features in Detail

### 🧠 Level 1: Cognitive STT with Auto-Routing

The agent uses Whisper's decoder logits to perform language identification. Unlike standard Whisper that checks 99 languages, we **constrain the search space** to just `{en, mr, gu}`:

```python
# Domain-constrained LID — our novel approach
for lang in ("en", "mr", "gu"):
    token_idx = tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
    score = logits[0, -1, token_idx].item()
# Pick the highest score → route to appropriate LoRA adapter
```

Once the language is detected, the agent **dynamically enables/disables** the corresponding LoRA adapter. English uses the base Whisper model directly.

### 🤖 Level 2: LLM Translation & Cleaning

Raw STT output often contains phonetic errors. The Qwen3-14B LLM acts as a cognitive filter:
- **Input**: Raw Marathi text with errors
- **Output**: `[CLEANED]: corrected native text` + `[ENGLISH]: fluent translation`
- Supports **RAG context injection** — users can provide topic hints to disambiguate homophones

Every LLM call is traced via **W&B Weave** for full auditability.

### 🔊 Level 3: Multilingual TTS

AI4Bharat's Indic Parler-TTS generates speech from text with controllable voice characteristics:
- **Voice presets**: `clear_female`, `clear_male`, `expressive_female`, `calm_male`
- **12 Indian languages** supported (Marathi, Gujarati, Hindi, English, Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia, Assamese)

### ⚡ LoRA Fine-Tuning Details

| Parameter | Value |
|-----------|-------|
| Base Model | `openai/whisper-large-v3-turbo` (809M) |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Adapter Size | 113 MB each (vs 1.6 GB full model = 7%) |
| Training Data | 160 samples per language (IndicSpeech) |
| Training Time | ~30 min per language on RTX 5000 Ada |
| Checkpoints | Every 100 steps, up to 1000 |

---

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
# Request a GPU node
srun --partition=gpu --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash

# Inside the GPU node
cd Speech_agent && source venv/bin/activate
python app.py
```

### Run the Evaluation Suite

```bash
# Quick evaluation (5 samples, ~5 min)
python scripts/evaluate_full_pipeline.py --num_samples 5 --skip_tts

# Full evaluation (50 samples, ~20 min)
python scripts/evaluate_full_pipeline.py --num_samples 50

# Results saved to: eval_results/full_pipeline_report.json
```

### Interactive Demo Notebook

```bash
# Open in VS Code or Jupyter (uses # %% cell markers)
code notebooks/full_pipeline_demo.py
```

---

## 📁 Repository Structure

```
Speech_agent/
├── app.py                           ← Gradio UI (STT + TTS tabs)
├── requirements.txt                 ← Python dependencies
├── .env                             ← API keys (WANDB_API_KEY)
│
├── src/
│   ├── speech_agent.py              ← CognitiveSpeechAgent (382 lines — the brain)
│   ├── tts_engine.py                ← IndicTTSEngine (190 lines — Parler-TTS wrapper)
│   └── speech_to_text_finetune/
│       ├── config.py                ← Pydantic config (LoRA hyperparams)
│       ├── data_process.py          ← Dataset loading (CV + IndicSpeech + custom)
│       ├── finetune_whisper.py      ← LoRA fine-tuning with HF Trainer
│       ├── evaluate_whisper_fleurs.py ← Basic FLEURS evaluator
│       ├── inference.py             ← Standalone inference pipeline
│       └── utils.py                 ← WER/CER metrics, model cards
│
├── scripts/
│   ├── evaluate_full_pipeline.py    ← Comprehensive 5-dim evaluation (440+ lines)
│   ├── convert_indicspeech.py       ← Dataset conversion for IndicSpeech
│   ├── download_cv_datasets.py      ← Download Common Voice data
│   └── download_fleurs.py           ← Download FLEURS eval data
│
├── notebooks/
│   ├── full_pipeline_demo.py        ← Full agentic loop demo (9 sections)
│   ├── colab_indic_asr_training.ipynb ← Colab training notebook
│   └── colab_evaluation.ipynb       ← Evaluation runner
│
├── example_configs/
│   ├── marathi/                     ← config_indicspeech.yaml, config_lora_gpu.yaml, etc.
│   ├── gujarati/
│   └── hindi/
│
├── artifacts/
│   ├── whisper-marathi-lora-indicspeech/   ← Trained LoRA (6 checkpoints, 113 MB)
│   └── whisper-gujarati-lora-indicspeech/  ← Trained LoRA (4 checkpoints, 113 MB)
│
├── eval_results/                    ← JSON evaluation reports (n=5 and n=50)
├── docs/                           ← evaluation-guide.md, deployment.md, etc.
├── tests/                          ← Unit, integration, e2e tests
│
├── IMPLEMENTATION_PLAN.md           ← 4-week project roadmap
├── VERIFICATION_REPORT.md           ← Component verification audit (85% complete)
├── START_HERE.md                    ← Quick reference guide
└── README.md                        ← This file
```

**Total**: 3,543 lines of Python | 56 files | 24 days of development

---

## 📅 Development Timeline

| Date | Milestone |
|------|-----------|
| **Apr 1** | Project scaffolding — repo structure, configs, notebooks, docs |
| **Apr 2** | LoRA fine-tuning done — Marathi & Gujarati adapters trained on IndicSpeech |
| **Apr 6** | Language detection done — Domain-constrained LID implemented |
| **Apr 11** | Full loop v1 — LLM (Qwen3-14B) + TTS (Parler-TTS) integrated, Gradio UI |
| **Apr 12** | Tokenizer fix — resolved `<\|notimestamps\|>` conflict |
| **Apr 24** | Phase 4 — Evaluation suite, demo notebook, documentation rewrite |
| **Apr 25** | n=50 evaluation complete — all 5 dimensions measured on FLEURS |

---

## 🧪 How to Reproduce

### 1. Train LoRA Adapters
```bash
# Train Marathi
python src/speech_to_text_finetune/finetune_whisper.py \
    --config example_configs/marathi/config_indicspeech.yaml

# Train Gujarati
python src/speech_to_text_finetune/finetune_whisper.py \
    --config example_configs/gujarati/config_indicspeech.yaml
```

### 2. Run the Full Pipeline
```python
from src.speech_agent import CognitiveSpeechAgent

agent = CognitiveSpeechAgent()
agent.load_language_adapter("artifacts/whisper-marathi-lora-indicspeech/checkpoint-1000", "mr")
agent.load_language_adapter("artifacts/whisper-gujarati-lora-indicspeech/checkpoint-1000", "gu")

result = agent.process_audio("path/to/audio.wav")
print(result["detected_language"])       # "mr"
print(result["raw_stt"])                 # Raw Marathi text
print(result["agent_final_response"])    # [CLEANED] + [ENGLISH]
```

### 3. Run TTS
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

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — project overview and results |
| [START_HERE.md](START_HERE.md) | Quick reference guide for immediate use |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | 4-week development roadmap |
| [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) | Component-level verification audit |
| [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | Cloud & cluster architecture |
| [docs/evaluation-guide.md](docs/evaluation-guide.md) | How to run and interpret evaluations |
| [docs/deployment.md](docs/deployment.md) | Gradio, SLURM, CLI deployment guide |

---

## 👥 Project Details

- **Course**: Intro to Speech Processing at MBZUAI
- **Developer**: Ramnarayan Choudhary
- **Email**: choudharyramnarayan123@gmail.com
- **GitHub**: [Ramnarayan-Choudhary/Speech_agent](https://github.com/Ramnarayan-Choudhary/Speech_agent)

## 🔐 License

MIT License — See [LICENSE](LICENSE)
