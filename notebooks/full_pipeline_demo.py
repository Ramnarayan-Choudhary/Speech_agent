# %% [markdown]
# # 🎙️ Indic Cognitive Speech Agent — Full Pipeline Demo
#
# This notebook walks through every stage of the **bidirectional speech-to-speech**
# agentic system we have built:
#
# ```
# ┌─────────┐     ┌───────────┐     ┌──────────────┐     ┌───────────┐     ┌─────────┐
# │  Audio  │ ──► │ Silero VAD│ ──► │ Whisper LID  │ ──► │ Whisper   │ ──► │   LLM   │
# │  Input  │     │ (denoise) │     │ (route lang) │     │ STT+LoRA  │     │ (Qwen3) │
# └─────────┘     └───────────┘     └──────────────┘     └───────────┘     └────┬────┘
#                                                                                │
#                                                                 ┌──────────────▼──────┐
#                                                                 │ [CLEANED]: मराठी... │
#                                                                 │ [ENGLISH]: English  │
#                                                                 └──────────┬──────────┘
#                                                                            │
#                                                                 ┌──────────▼──────────┐
#                                                                 │ Indic Parler-TTS    │
#                                                                 │ (AI4Bharat)         │
#                                                                 └─────────────────────┘
# ```
#
# **Components:**
# - **STT**: `openai/whisper-large-v3-turbo` (809M params) with optional LoRA adapters
# - **Language ID**: Domain-constrained decoder-logit probability routing
# - **VAD**: Silero VAD for silence filtering
# - **LLM**: Qwen3-14B via Weights & Biases Inference API
# - **TTS**: AI4Bharat Indic Parler-TTS (multilingual)
#
# ---

# %% [markdown]
# ## 1. Environment Setup

# %%
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import torch
print(f"PyTorch version:  {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

import transformers
print(f"Transformers:     {transformers.__version__}")

# %% [markdown]
# ## 2. Initialize the Cognitive Speech Agent
#
# The `CognitiveSpeechAgent` class is the brain of our system. It:
# 1. Loads Whisper-Large-v3-Turbo as the base model
# 2. Optionally loads LoRA adapters for Marathi and Gujarati
# 3. Initializes Silero VAD for silence removal
# 4. Connects to the W&B LLM inference API (if API key is set)

# %%
from src.speech_agent import CognitiveSpeechAgent

print("⏳ Initializing the Cognitive Speech Agent...")
print("   (This loads Whisper + VAD + LLM client — may take 30-60s on first run)\n")

agent = CognitiveSpeechAgent()

# Load LoRA adapters if available
mr_adapter = os.path.join(PROJECT_ROOT, "artifacts", "whisper-marathi-lora-indicspeech", "checkpoint-1000")
gu_adapter = os.path.join(PROJECT_ROOT, "artifacts", "whisper-gujarati-lora-indicspeech", "checkpoint-1000")

if os.path.exists(mr_adapter):
    agent.load_language_adapter(mr_adapter, "mr")
    print("✅ Marathi LoRA adapter loaded")
else:
    print("ℹ️  No Marathi LoRA adapter found (using base model)")

if os.path.exists(gu_adapter):
    agent.load_language_adapter(gu_adapter, "gu")
    print("✅ Gujarati LoRA adapter loaded")
else:
    print("ℹ️  No Gujarati LoRA adapter found (using base model)")

print(f"\n📊 Agent Status:")
print(f"   Device:          {agent.device}")
print(f"   VAD enabled:     {agent.vad_model is not None}")
print(f"   LLM enabled:     {agent.llm_client is not None}")
print(f"   Adapters loaded: {list(agent.adapters_loaded.keys()) or 'None'}")

# %% [markdown]
# ## 3. Language Detection Demo
#
# The agent uses Whisper's decoder logits to perform **domain-constrained**
# language identification. Rather than detecting from 100+ languages, we
# restrict the search space to just `{en, mr, gu}` and pick the highest
# probability token.

# %%
import librosa
import numpy as np

def demo_language_detection(audio_path: str):
    """Show detailed language detection probabilities."""
    print(f"\n🎤 Audio file: {os.path.basename(audio_path)}")

    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    print(f"   Duration: {duration:.1f}s")

    input_features = agent.processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(agent.device, dtype=agent.torch_dtype)

    detected = agent.detect_language(input_features)
    lang_names = {"en": "English 🇬🇧", "mr": "Marathi 🇮🇳", "gu": "Gujarati 🇮🇳"}
    print(f"   🏆 Detected: {lang_names.get(detected, detected)}")
    return detected

# Test with available demo files
demo_files = {
    "Marathi": os.path.join(PROJECT_ROOT, "demo_marathi.wav"),
    "Gujarati": os.path.join(PROJECT_ROOT, "demo_gujarati.wav"),
}

for name, path in demo_files.items():
    if os.path.exists(path):
        demo_language_detection(path)
    else:
        print(f"  ⚠️  Demo file not found: {path}")

# %% [markdown]
# ## 4. Speech-to-Text Demo
#
# The `process_audio()` method runs the full agentic loop:
# 1. Load and preprocess audio (16kHz, mono, float32)
# 2. Apply Silero VAD to remove silence
# 3. Detect language from decoder logits
# 4. Route to LoRA adapter or base model
# 5. Generate transcription
# 6. Pass through LLM for cleaning and translation

# %%
def demo_stt(audio_path: str, explicit_lang: str = "Autonomous"):
    """Run full agent pipeline on a single audio file."""
    print(f"\n{'='*60}")
    print(f"  🎙️  STT Demo: {os.path.basename(audio_path)}")
    print(f"  Routing mode: {explicit_lang}")
    print(f"{'='*60}")

    result = agent.process_audio(audio_path, explicit_lang=explicit_lang)

    print(f"\n  1. Detected Language:  {result['detected_language']}")
    print(f"  2. STT Source:         {result['stt_source']}")
    print(f"  3. Raw Transcription:  {result['raw_stt']}")
    print(f"  4. LLM Enabled:       {result['llm_enabled']}")
    print(f"\n  5. Agent Final Response:")
    print(f"     {result['agent_final_response']}")
    print(f"{'='*60}")
    return result

# Run demos
for name, path in demo_files.items():
    if os.path.exists(path):
        demo_stt(path)

# %% [markdown]
# ## 5. LLM Translation Demo
#
# The LLM layer uses **Qwen3-14B** (via W&B Inference) to:
# 1. Clean phonetic errors in the native-language transcription
# 2. Translate the cleaned text to fluent English
#
# Output format:
# ```
# [CLEANED]: <corrected native text>
# [ENGLISH]: <English translation>
# ```

# %%
if agent.llm_client is not None:
    # Direct LLM call demo
    sample_texts = {
        "mr": "नमस्कार माझे नाव रामनारायण आहे आणि मी एमबीझेडयूएआय मध्ये शिकतो",
        "gu": "નમસ્તે મારું નામ રામનારાયણ છે અને હું એમબીઝેડયુએઆઈ માં ભણું છું",
    }

    for lang, text in sample_texts.items():
        print(f"\n{'─'*50}")
        print(f"  LLM Translation [{lang}]")
        print(f"  Input:  {text}")
        response = agent.llm_translation_agent(raw_text=text, language=lang)
        print(f"  Output: {response}")
        print(f"{'─'*50}")
else:
    print("⚠️  LLM client not initialized (WANDB_API_KEY not set).")
    print("   Set it in .env to enable LLM polishing and translation.")

# %% [markdown]
# ## 6. Text-to-Speech Demo
#
# The TTS engine uses **AI4Bharat's Indic Parler-TTS** to synthesize speech
# in 21+ Indian languages. It accepts a voice description prompt to control
# speaker characteristics (gender, pace, quality).

# %%
from src.tts_engine import IndicTTSEngine, LANG_META, VOICE_PRESETS

print("⏳ Initializing TTS engine (lazy load on first call)...\n")
tts_engine = IndicTTSEngine()

# Synthesize samples
tts_demos = {
    "mr": "नमस्कार, हे मराठी भाषेतील संश्लेषित भाषण आहे.",
    "gu": "નમસ્તે, આ ગુજરાતી ભાષામાં સંશ્લેષિત ભાષણ છે.",
    "en": "Hello, this is synthesized speech in English language.",
}

for lang, text in tts_demos.items():
    print(f"\n  🔊 Synthesizing [{lang}]: '{text[:50]}...'")
    try:
        result = tts_engine.synthesize(
            text=text,
            language=lang,
            voice_preset="clear_female",
        )
        print(f"     ✅ Generated {result['duration_s']}s of audio at {result['sample_rate']}Hz")
    except Exception as e:
        print(f"     ❌ TTS failed: {e}")

print(f"\n  Supported languages: {', '.join(m['name'] for m in LANG_META.values())}")
print(f"  Voice presets: {', '.join(VOICE_PRESETS.keys())}")

# %% [markdown]
# ## 7. Full Loop Demo: Audio → Agent → TTS
#
# This demonstrates the complete bidirectional cycle:
# 1. Input speech audio (Marathi/Gujarati)
# 2. Agent transcribes and translates to English
# 3. TTS synthesizes the English translation as speech

# %%
import soundfile as sf
import re

def full_loop_demo(audio_path: str):
    """Complete cycle: Audio → STT → LLM → TTS → Audio"""
    print(f"\n{'='*60}")
    print(f"  🔄 FULL LOOP: {os.path.basename(audio_path)}")
    print(f"{'='*60}")

    # Step 1: Agent processes audio (STT + LLM)
    result = agent.process_audio(audio_path)

    print(f"\n  Step 1 — Language:      {result['detected_language']}")
    print(f"  Step 2 — Raw STT:       {result['raw_stt'][:80]}...")
    print(f"  Step 3 — Agent Output:  {result['agent_final_response'][:120]}...")

    # Step 2: Extract English translation for TTS
    english_text = ""
    for line in result.get("agent_final_response", "").split("\n"):
        if line.strip().upper().startswith("[ENGLISH]"):
            english_text = re.sub(
                r"^\[ENGLISH\]\s*:?\s*", "", line.strip(), flags=re.IGNORECASE
            ).strip()
            break

    if not english_text:
        english_text = result.get("raw_stt", "No text available")
        print(f"  (No [ENGLISH] tag found, using raw STT for TTS)")

    print(f"  Step 4 — TTS Input:     '{english_text[:80]}...'")

    # Step 3: Synthesize speech from translation
    try:
        tts_result = tts_engine.synthesize(
            text=english_text,
            language="en",
            voice_preset="clear_female",
            output_path=os.path.join(PROJECT_ROOT, "demo_full_loop_output.wav"),
        )
        print(f"  Step 5 — TTS Output:    {tts_result['duration_s']}s audio → demo_full_loop_output.wav")
        print(f"\n  ✅ Full loop complete!")
    except Exception as e:
        print(f"  ❌ TTS step failed: {e}")

    print(f"{'='*60}")


# Run the full loop on available demo files
for name, path in demo_files.items():
    if os.path.exists(path):
        full_loop_demo(path)
        break  # One demo is enough to show the concept

# %% [markdown]
# ## 8. Run the Evaluation Suite
#
# The comprehensive evaluation script measures 5 dimensions:
# - **A.** STT Accuracy (WER/CER)
# - **B.** Language Routing Accuracy
# - **C.** LLM Translation Quality (BLEU)
# - **D.** TTS Health Check
# - **E.** Latency Profiling
#
# For a quick test, use `--num_samples 5`. For a full evaluation, use 50+.

# %%
print("To run the full evaluation suite, execute this from the terminal:\n")
print("  # Quick run (5 samples, skip expensive components):")
print("  python scripts/evaluate_full_pipeline.py --num_samples 5 --skip_tts --skip_llm\n")
print("  # Medium run (20 samples, all components):")
print("  python scripts/evaluate_full_pipeline.py --num_samples 20\n")
print("  # Full run (50 samples):")
print("  python scripts/evaluate_full_pipeline.py --num_samples 50\n")
print("  Results will be saved to: eval_results/full_pipeline_report.json")

# %% [markdown]
# ## 9. Summary
#
# This notebook demonstrated the complete **Indic Cognitive Speech Agent**:
#
# | Component | Technology | Status |
# |-----------|-----------|--------|
# | **STT** | Whisper-Large-v3-Turbo + LoRA | ✅ Working |
# | **Language ID** | Domain-constrained decoder logits | ✅ Working |
# | **VAD** | Silero VAD | ✅ Working |
# | **LLM** | Qwen3-14B via W&B Inference | ✅ Working |
# | **TTS** | AI4Bharat Indic Parler-TTS | ✅ Working |
# | **UI** | Gradio (STT + TTS tabs) | ✅ Working |
#
# To launch the full Gradio UI:
# ```bash
# python app.py
# ```
#
# ---
# *Built for MBZUAI Speech Processing Course by Ramnarayan Choudhary*
