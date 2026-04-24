# 🚀 START HERE: QUICK REFERENCE GUIDE

**Status**: ✅ Full Agentic Loop Operational (Weeks 1-3 Complete)  
**Ready to Execute**: YES — Evaluation & submission phase  
**Location**: MBZUAI SLURM Cluster

---

## ✅ Current System Status

### All Core Components Working
```
✅ src/speech_agent.py              (CognitiveSpeechAgent — main brain)
✅ src/tts_engine.py                (IndicTTSEngine — Parler-TTS wrapper)
✅ src/speech_to_text_finetune/     (LoRA fine-tuning pipeline)
✅ app.py                           (Gradio UI — STT + TTS tabs)
✅ scripts/evaluate_full_pipeline.py (5-dimension evaluation suite)
✅ notebooks/full_pipeline_demo.py  (Interactive demo walkthrough)
✅ artifacts/whisper-*-lora-*/      (Trained LoRA weights)
✅ example_configs/                 (YAML configs for 3 languages)
✅ requirements.txt                 (All dependencies)
```

### Architecture
```
Audio → Silero VAD → Whisper LID → Whisper STT + LoRA → Qwen3-14B LLM → Parler-TTS → Audio
```

---

## 📋 WHAT TO DO NOW

### 1. Launch the Gradio Demo
```bash
cd /path/to/Speech_agent
source venv/bin/activate
python app.py
# → Opens at http://localhost:7860 with a public share link
```

### 2. Run the Evaluation Suite
```bash
# Quick run (5 samples per language, ~5 min)
python scripts/evaluate_full_pipeline.py --num_samples 5 --skip_tts

# Medium run (20 samples, all components)
python scripts/evaluate_full_pipeline.py --num_samples 20

# Full run (50 samples per language)
python scripts/evaluate_full_pipeline.py --num_samples 50

# Results: eval_results/full_pipeline_report.json
```

### 3. Walk Through the Demo Notebook
```bash
# Open in VS Code (uses # %% cell markers for interactive execution)
code notebooks/full_pipeline_demo.py

# Or run as a script
python notebooks/full_pipeline_demo.py
```

---

## 🔗 CRITICAL FILES & THEIR LOCATIONS

| What You Need | Where It Is |
|---------------|-------------|
| **Main Agent Brain** | `src/speech_agent.py` — CognitiveSpeechAgent class |
| **TTS Engine** | `src/tts_engine.py` — IndicTTSEngine (Parler-TTS) |
| **Gradio UI** | `app.py` — STT + TTS tabs |
| **Evaluation Suite** | `scripts/evaluate_full_pipeline.py` — 5-dim eval |
| **Demo Notebook** | `notebooks/full_pipeline_demo.py` — Full walkthrough |
| **Fine-tuning script** | `src/speech_to_text_finetune/finetune_whisper.py` |
| **Config system** | `src/speech_to_text_finetune/config.py` |
| **Data pipeline** | `src/speech_to_text_finetune/data_process.py` |
| **LoRA weights (Marathi)** | `artifacts/whisper-marathi-lora-indicspeech/` |
| **LoRA weights (Gujarati)** | `artifacts/whisper-gujarati-lora-indicspeech/` |
| **Eval results** | `eval_results/full_pipeline_report.json` |
| **Architecture guide** | `INFRASTRUCTURE.md` |
| **Verification audit** | `VERIFICATION_REPORT.md` |

---

## 🎯 PROJECT PHASE STATUS

| Phase | Week | Focus | Status |
|-------|------|-------|--------|
| **Phase 0** | Setup | Environment & Dependencies | ✅ Complete |
| **Phase 1** | Data | Preparation & Exploration | ✅ Complete |
| **Phase 2** | Training | LoRA Fine-tuning | ✅ Complete |
| **Phase 3** | Agentic | Router + LLM + TTS + Gradio | ✅ Complete |
| **Phase 4** | Eval | Comprehensive evaluation + docs | ✅ Complete |
| **Phase 5** | Submit | Presentation + demo video | ⏳ Remaining |

---

## 💡 KEY FEATURES IMPLEMENTED

### 🧠 Level 1: Cognitive STT with Auto-Routing
- ✅ Domain-constrained LID (en, mr, gu)
- ✅ LoRA adapter auto-switching by detected language
- ✅ Silero VAD for silence removal
- ✅ Whisper-Large-v3-Turbo (809M params)

### 🤖 Level 2: LLM Translation & Cleaning
- ✅ Qwen3-14B via W&B Inference API
- ✅ Phonetic error correction
- ✅ Bidirectional translation (native ↔ English)
- ✅ RAG context grounding (optional)

### 🔊 Level 3: Multilingual TTS
- ✅ AI4Bharat Indic Parler-TTS (21+ languages)
- ✅ Voice presets (clear_female, clear_male, etc.)
- ✅ Full loop: Audio → transcribe → translate → synthesize → Audio

### 📊 Evaluation
- ✅ STT Accuracy (WER/CER on FLEURS)
- ✅ Language Routing Accuracy
- ✅ LLM Translation Quality (BLEU)
- ✅ TTS Health Check
- ✅ Latency Profiling

---

## ❓ Common Questions

**Q: How do I launch the full UI?**  
A: `python app.py` → opens Gradio at http://localhost:7860 with STT + TTS tabs.

**Q: How do I get quantitative evaluation numbers?**  
A: `python scripts/evaluate_full_pipeline.py --num_samples 20` → JSON report in `eval_results/`.

**Q: What if I don't have the W&B API key?**  
A: The system still works! LLM translation will be skipped but STT + TTS operate independently.

**Q: How do I run on SLURM?**  
A: Use `sbatch train.sbatch` or `bash run_worker.sh` for interactive mode.

**Q: What's the interactive demo notebook?**  
A: `notebooks/full_pipeline_demo.py` — open in VS Code and run cells with `# %%` markers.

---

## 🟢 STATUS: GO/NO-GO DECISION

**DECISION: 🟢 GO — SYSTEM FULLY OPERATIONAL**

All agentic components implemented and tested. Ready for evaluation run and final submission.

**Start with**: Run the evaluation suite → Collect numbers → Create presentation

**Good luck! 🚀**

---

**Updated**: April 24, 2026  
**Repository**: github.com/Ramnarayan-Choudhary/Speech_agent  
**Status**: ✅ FULL AGENTIC LOOP OPERATIONAL
