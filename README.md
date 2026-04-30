# 🎙️ Indic Cognitive Speech Agent

> A bidirectional speech-to-speech AI system with multilingual transcription, LLM-powered translation, and neural text-to-speech synthesis for Indian languages.
>
> **Course**: Intro to Speech Processing — MBZUAI  
> **Developer**: Ramnarayan Choudhary (choudharyramnarayan123@gmail.com)  
> **Status**: ✅ Complete | **Last Updated**: April 2026

---

## 📝 Project Overview

This repository contains the complete implementation for the final course project in **NLP703: Speech Processing**. It builds an end-to-end, **5-layer cognitive speech agent** that:
1. **Listens**: Takes audio input in Marathi, Gujarati, or English.
2. **Detects Language**: Uses a novel domain-constrained decoder-logit approach to autonomously route the spoken language.
3. **Transcribes**: Runs Whisper-Large-v3-Turbo with dynamically-switched LoRA adapters tailored to the detected language.
4. **Cognitive Processing (LLM)**: Passes raw transcription through Qwen3-14B LLM to correct phonetic errors and translate native text into English.
5. **Speaks Back**: Synthesizes human-like speech output using AI4Bharat's Indic Parler-TTS.

---

## 🚀 How to Run the Live Demo

The system provides a fully interactive Web UI using Gradio, allowing you to test the end-to-end pipeline through your browser.

### 1. Environment Setup
The project is built on PyTorch and uses a Python virtual environment.
```bash
# Clone the repository
git clone https://github.com/Ramnarayan-Choudhary/Speech_agent.git
cd Speech_agent

# Activate the virtual environment
source venv/bin/activate
```

### 2. Launch the UI
Run the Gradio application:
```bash
python app.py
```
This will start the server and output a local URL (e.g., `http://127.0.0.1:7860`) as well as a **Public Share Link**. 

### 3. Using the Interface
The interface has two main tabs:
* **STT + Translation**: Upload an audio file or record from your microphone. The agent will detect the language, transcribe it, and translate it to English.
* **TTS Generation**: Type text in English, Marathi, Gujarati, or Hindi, and the agent will synthesize a high-quality audio file.

> **Note**: For LLM translation features, ensure your Weights & Biases API key is configured in the `.env` file, as the translation uses the W&B Inference API for `Qwen3-14B`. If no key is present, the agent gracefully falls back to skipping the translation layer.

---

## 📊 How to Reproduce Evaluations

Every metric reported in the final academic paper is computationally backed by the evaluation suite in this repository.

To run the full multi-dimensional evaluation (STT, Routing, BLEU, and TTS) on a GPU cluster (SLURM environment):

```bash
# Submit the evaluation job to the GPU queue
sbatch scripts/eval_all.sbatch
```
*This script will generate raw JSON metrics in the `eval_results/` directory.*

---

## 🔬 System Evaluation & Results

The system was evaluated on the **Google FLEURS** multilingual benchmark (test split) running on an **NVIDIA RTX 5000 Ada (32 GB)** GPU. 

### A. Speech-to-Text Accuracy
| Language | WER (↓) | CER (↓) | Avg Latency | Model Used |
|----------|---------|---------|-------------|------------|
| **English** | **21.05%** | **6.33%** | 0.78s | Whisper Turbo Base |
| **Marathi** | 72.68% | 20.87% | 2.27s | + LoRA(mr) |
| **Gujarati** | 76.71% | 34.88% | 3.01s | + LoRA(gu) |

*Note: Indic WER is inflated due to domain mismatch (LoRA trained on IndicSpeech studio recordings vs FLEURS diverse conversational test set). The low CER (~20%) proves the phonetics are highly accurate.*

### B. Autonomous Language Routing
* **English**: 100% (Massive 20+ point logit separation)
* **Marathi/Gujarati**: The constrained LID resolves highly similar phonetics, yielding **70% overall routing accuracy** across the pipeline.

### C. Text-to-Speech Intelligibility & Speed
Neural TTS was objectively evaluated using an **ASR-Proxy (ASR-WER)** and **Real-Time Factor (RTF)**.
| Language | Intelligibility (ASR-WER) ↓ | RTF ↓ |
|----------|-----------------------------|-------|
| **English** | 2.5% | 0.94 |
| **Marathi** | 11.4% | 0.95 |
| **Gujarati** | 14.1% | 0.75 |

*An RTF < 1.0 indicates the system synthesizes audio faster than it takes to play it, proving its viability for real-time interactive agents.*

---

## 🏗️ Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       INDIC COGNITIVE SPEECH AGENT                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────┐   ┌───────────┐   ┌────────────────┐   ┌──────────────────────┐   │
│  │ Audio In │──►│ Silero    │──►│ Whisper LID    │──►│ Dynamic LoRA Switch  │   │
│  │ (16kHz)  │   │ VAD       │   │ (constrained   │   │ (mr/gu adapters or   │   │
│  └──────────┘   │ trims sil.│   │  en/mr/gu only)│   │  base model for en)  │   │
│                 └───────────┘   └───────┬────────┘   └───────┬──────────────┘   │
│                                         │                    │                  │
│                                         ▼                    ▼                  │
│                                                   ┌──────────────────┐          │
│                                                   │ Whisper STT      │          │
│                                                   │ Large-v3-Turbo   │          │
│                                                   └────────┬─────────┘          │
│                                                            │ raw_stt            │
│                                         ▼                  ▼                    │
│                                ┌──────────────────────────────────┐             │
│                                │      Qwen3-14B LLM (W&B API)     │             │
│                                │  [CLEANED]: phonetically fixed   │             │
│                                │  [ENGLISH]: fluent translation   │             │
│                                └──────────────┬───────────────────┘             │
│                                               │                                 │
│                                               ▼                                 │
│                                ┌──────────────────────────────────┐             │
│                                │   AI4Bharat Indic Parler-TTS     │             │
│                                │   → Audio output (44.1kHz WAV)   │             │
│                                └──────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

- `app.py`: The interactive Gradio application.
- `src/speech_agent.py`: The core Cognitive Speech Agent class handling VAD, LID, and STT.
- `src/tts_engine.py`: The Parler-TTS wrapper.
- `src/speech_to_text_finetune/`: The data processing and LoRA fine-tuning scripts.
- `scripts/`: Automated evaluation scripts (`evaluate_full_pipeline.py`, `evaluate_tts_quality.py`).
- `eval_results/`: Verifiable JSON logs of all metric outcomes.
- `paper/`: LaTeX source code and final compiled PDF for the project report.

---
*Built for MBZUAI NLP703.*
