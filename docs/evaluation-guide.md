# 📊 Evaluation Guide

This guide explains how to evaluate the Indic Cognitive Speech Agent at every level.

---

## Quick Start

```bash
# Run the comprehensive 5-dimension evaluation suite
python scripts/evaluate_full_pipeline.py --num_samples 20
```

Results are saved to `eval_results/full_pipeline_report.json`.

---

## Comprehensive Evaluation Suite

The `scripts/evaluate_full_pipeline.py` script evaluates **5 dimensions** of the full agentic pipeline:

### A. STT Accuracy (WER / CER)

Measures how accurately Whisper transcribes speech using the [FLEURS](https://huggingface.co/datasets/google/fleurs) multilingual benchmark.

- **WER** (Word Error Rate): Fraction of words incorrectly transcribed
- **CER** (Character Error Rate): Fraction of characters incorrectly transcribed
- Run per-language (mr_in, gu_in, en_us)

### B. Language Routing Accuracy

Tests whether the domain-constrained language detection correctly identifies the spoken language from audio.

- Feeds labeled FLEURS audio (known language)
- Checks if `detected_language` matches the expected language code
- Reports per-language and overall accuracy

### C. LLM Translation Quality (BLEU)

Evaluates the quality of English translations produced by the Qwen3-14B LLM layer.

- Extracts `[ENGLISH]:` line from the agent's response
- Compares against FLEURS English reference translations
- Computes SacreBLEU score
- Requires `WANDB_API_KEY` in `.env`

### D. TTS Health Check

Validates that the Indic Parler-TTS engine generates valid audio for each language.

- Synthesizes a sample sentence per language (mr, gu, hi, en)
- Checks: audio array non-empty, duration > 0.3s, correct sample rate
- Reports generation time per language

### E. Latency Profiling

Profiles end-to-end processing time on a small sample.

- Measures total time from audio input to agent output
- Reports avg, median, min, max, and standard deviation

---

## Command-Line Options

```bash
python scripts/evaluate_full_pipeline.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_samples` | 20 | FLEURS samples per language |
| `--languages` | mr_in gu_in en_us | FLEURS language codes |
| `--skip_tts` | false | Skip TTS health check |
| `--skip_llm` | false | Skip LLM translation eval |
| `--skip_routing` | false | Skip language routing eval |
| `--output_dir` | eval_results/ | Output directory for JSON |
| `--base_model` | openai/whisper-large-v3-turbo | Whisper model ID |

---

## Interpreting the JSON Report

The report (`eval_results/full_pipeline_report.json`) has this structure:

```json
{
  "timestamp": "2026-04-24T22:00:00",
  "config": { ... },
  "stt_accuracy": {
    "mr_in": {"wer": 0.18, "cer": 0.06, "num_samples": 20, "avg_latency_s": 3.5},
    "gu_in": {"wer": 0.22, "cer": 0.08, ...},
    "en_us": {"wer": 0.08, "cer": 0.03, ...}
  },
  "routing_accuracy": {
    "overall": 0.95,
    "per_lang": { ... }
  },
  "translation_quality": {
    "mr_in": {"bleu": 32.5, ...},
    ...
  },
  "tts_health": {
    "mr": {"ok": true, "duration_s": 3.2, ...},
    ...
  },
  "latency": {
    "avg_total_s": 4.2,
    "median_total_s": 3.8,
    ...
  }
}
```

**Key thresholds** (targets for the course):
- WER < 20% for Indic languages
- WER < 10% for English
- Routing accuracy > 90%
- BLEU > 20 for translations
- TTS all languages: ok = true

---

## Running Individual Evaluations

### STT-Only (Basic FLEURS)

```bash
python src/speech_to_text_finetune/evaluate_whisper_fleurs.py \
    --model_id openai/whisper-large-v3-turbo \
    --language mr_in \
    --output eval_results/stt_marathi.json
```

### TTS-Only (Quick Check)

```python
from src.tts_engine import IndicTTSEngine
engine = IndicTTSEngine()
result = engine.synthesize("नमस्कार", language="mr", output_path="test.wav")
print(f"Generated {result['duration_s']}s of audio")
```

---

## Tips

- Use `--num_samples 5` for quick debugging runs (~5 min)
- Use `--num_samples 50` for publication-quality numbers (~30 min)
- Add `--skip_tts --skip_llm` to focus only on STT accuracy
- The script gracefully handles missing API keys and packages
