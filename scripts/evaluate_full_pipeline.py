#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for the Indic Cognitive Speech Agent.

Evaluates 5 dimensions of the full agentic pipeline:
  A. STT Accuracy        – WER/CER of raw transcription on FLEURS
  B. Language Routing     – % of audio correctly routed to the right language
  C. LLM Translation     – BLEU score of English translations
  D. TTS Health Check    – Validates TTS generates non-empty, correct audio
  E. Latency Profiling   – End-to-end timing breakdown per stage

Usage:
    # Quick run (5 samples per language)
    python scripts/evaluate_full_pipeline.py --num_samples 5

    # Full run (50 samples per language)
    python scripts/evaluate_full_pipeline.py --num_samples 50

    # Skip expensive components
    python scripts/evaluate_full_pipeline.py --skip_tts --skip_llm
"""

import argparse
import json
import os
import re
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ===========================================================================
# Helpers
# ===========================================================================

def _safe_import(module_name: str):
    """Try to import a module; return None on failure."""
    try:
        return __import__(module_name)
    except ImportError:
        return None


def _load_fleurs_subset(lang_code: str, split: str = "test", num_samples: int = 20):
    """
    Load a slice of Google FLEURS for a given language.

    FLEURS lang codes: mr_in, gu_in, hi_in, en_us
    Returns list of dicts with keys: audio_path, transcription, language
    """
    from datasets import load_dataset, Audio

    logger.info(f"Loading FLEURS [{lang_code}] split={split} (n={num_samples})...")
    ds = load_dataset(
        "google/fleurs",
        lang_code,
        split=split,
        streaming=False,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Take a subset
    n = min(num_samples, len(ds))
    ds = ds.select(range(n))

    samples = []
    for item in ds:
        samples.append({
            "audio_array": item["audio"]["array"],
            "sampling_rate": item["audio"]["sampling_rate"],
            "transcription": item["transcription"],
            "language": lang_code,
        })
    logger.info(f"  → Loaded {len(samples)} samples for [{lang_code}]")
    return samples


def _save_audio_to_tmp(audio_array: np.ndarray, sr: int = 16000) -> str:
    """Write audio array to a temporary WAV file and return its path."""
    import soundfile as sf

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(PROJECT_ROOT / ".gradio"))
    sf.write(tmp.name, audio_array, sr, subtype="PCM_16")
    return tmp.name


# ===========================================================================
# A.  STT Accuracy (WER / CER)
# ===========================================================================

def evaluate_stt_accuracy(agent, lang_samples: dict, results: dict):
    """
    Evaluate raw STT accuracy (WER / CER) per language.

    Args:
        agent: CognitiveSpeechAgent instance
        lang_samples: {"mr_in": [samples], "gu_in": [...], ...}
        results: dict to populate in-place
    """
    import jiwer

    logger.info("=" * 60)
    logger.info("  [A] STT ACCURACY EVALUATION (WER / CER)")
    logger.info("=" * 60)

    # Map FLEURS codes → Whisper codes
    fleurs_to_whisper = {"mr_in": "mr", "gu_in": "gu", "hi_in": "hi", "en_us": "en"}

    stt_results = {}

    for fleurs_code, samples in lang_samples.items():
        whisper_lang = fleurs_to_whisper.get(fleurs_code, "unknown")
        predictions = []
        references = []
        total_time = 0.0

        logger.info(f"\n  Evaluating STT for [{fleurs_code}] ({len(samples)} samples)...")

        for i, sample in enumerate(samples):
            audio_path = _save_audio_to_tmp(sample["audio_array"], sample["sampling_rate"])
            try:
                t0 = time.perf_counter()
                result = agent.process_audio(audio_path, explicit_lang=whisper_lang)
                elapsed = time.perf_counter() - t0
                total_time += elapsed

                raw_text = result.get("raw_stt", "")
                predictions.append(raw_text)
                references.append(sample["transcription"])

                if (i + 1) % 5 == 0:
                    logger.info(f"    [{fleurs_code}] {i+1}/{len(samples)} done ({elapsed:.1f}s)")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        # Compute metrics
        if predictions and references:
            # Filter out empty predictions/references
            valid = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
            if valid:
                preds, refs = zip(*valid)
                wer = jiwer.wer(list(refs), list(preds))
                cer = jiwer.cer(list(refs), list(preds))
            else:
                wer, cer = 1.0, 1.0
        else:
            wer, cer = 1.0, 1.0

        avg_latency = total_time / max(len(samples), 1)

        stt_results[fleurs_code] = {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "num_samples": len(samples),
            "avg_latency_s": round(avg_latency, 3),
        }
        logger.info(
            f"  ✅ [{fleurs_code}] WER={wer:.2%}  CER={cer:.2%}  "
            f"avg_latency={avg_latency:.2f}s  (n={len(samples)})"
        )

    results["stt_accuracy"] = stt_results


# ===========================================================================
# B.  Language Routing Accuracy
# ===========================================================================

def evaluate_routing_accuracy(agent, lang_samples: dict, results: dict):
    """
    Test whether the agent correctly detects the language from audio.
    """
    logger.info("=" * 60)
    logger.info("  [B] LANGUAGE ROUTING ACCURACY")
    logger.info("=" * 60)

    fleurs_to_whisper = {"mr_in": "mr", "gu_in": "gu", "hi_in": "hi", "en_us": "en"}
    per_lang = {}
    total_correct = 0
    total_samples = 0

    for fleurs_code, samples in lang_samples.items():
        expected = fleurs_to_whisper.get(fleurs_code, "unknown")
        correct = 0

        logger.info(f"\n  Routing test [{fleurs_code}] → expected={expected} ({len(samples)} samples)...")

        for i, sample in enumerate(samples):
            audio_path = _save_audio_to_tmp(sample["audio_array"], sample["sampling_rate"])
            try:
                result = agent.process_audio(audio_path, explicit_lang="Autonomous")
                detected = result.get("detected_language", "unknown")
                if detected == expected:
                    correct += 1
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        accuracy = correct / max(len(samples), 1)
        per_lang[fleurs_code] = {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": len(samples),
        }
        total_correct += correct
        total_samples += len(samples)
        logger.info(f"  ✅ [{fleurs_code}] Routing accuracy: {accuracy:.2%} ({correct}/{len(samples)})")

    overall = total_correct / max(total_samples, 1)
    results["routing_accuracy"] = {
        "overall": round(overall, 4),
        "per_lang": per_lang,
    }
    logger.info(f"\n  🏆 Overall routing accuracy: {overall:.2%} ({total_correct}/{total_samples})")


# ===========================================================================
# C.  LLM Translation Quality (BLEU)
# ===========================================================================

def evaluate_llm_translation(agent, lang_samples: dict, en_references: dict, results: dict):
    """
    Evaluate LLM translation quality using BLEU against FLEURS English refs.

    Args:
        agent: CognitiveSpeechAgent with LLM enabled
        lang_samples: Indic FLEURS samples keyed by lang code
        en_references: dict mapping index → English reference text (from FLEURS en_us)
        results: output dict
    """
    logger.info("=" * 60)
    logger.info("  [C] LLM TRANSLATION QUALITY (BLEU)")
    logger.info("=" * 60)

    if agent.llm_client is None:
        logger.warning("  ⚠️  LLM client not available. Skipping translation evaluation.")
        results["translation_quality"] = {"status": "skipped", "reason": "LLM client not initialized"}
        return

    evaluate_mod = _safe_import("evaluate")
    if evaluate_mod is None:
        logger.warning("  ⚠️  `evaluate` package not available. Skipping BLEU.")
        results["translation_quality"] = {"status": "skipped", "reason": "evaluate package missing"}
        return

    sacrebleu = evaluate_mod.load("sacrebleu")
    fleurs_to_whisper = {"mr_in": "mr", "gu_in": "gu", "hi_in": "hi"}
    translation_results = {}

    for fleurs_code, samples in lang_samples.items():
        if fleurs_code == "en_us":
            continue  # Skip English (no translation needed)

        whisper_lang = fleurs_to_whisper.get(fleurs_code, "unknown")
        predictions = []
        references = []

        logger.info(f"\n  BLEU evaluation [{fleurs_code}] ({len(samples)} samples)...")

        for i, sample in enumerate(samples):
            audio_path = _save_audio_to_tmp(sample["audio_array"], sample["sampling_rate"])
            try:
                result = agent.process_audio(audio_path, explicit_lang=whisper_lang)
                response = result.get("agent_final_response", "")

                # Extract [ENGLISH]: line
                english_translation = ""
                for line in response.split("\n"):
                    line_stripped = line.strip()
                    if line_stripped.upper().startswith("[ENGLISH]"):
                        english_translation = re.sub(
                            r"^\[ENGLISH\]\s*:?\s*", "", line_stripped, flags=re.IGNORECASE
                        ).strip()
                        break

                if english_translation and i in en_references:
                    predictions.append(english_translation)
                    references.append(en_references[i])

            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        if predictions and references:
            bleu_result = sacrebleu.compute(
                predictions=predictions,
                references=[[r] for r in references],
            )
            bleu_score = bleu_result["score"]
        else:
            bleu_score = 0.0

        translation_results[fleurs_code] = {
            "bleu": round(bleu_score, 2),
            "num_evaluated": len(predictions),
            "num_total": len(samples),
        }
        logger.info(f"  ✅ [{fleurs_code}] BLEU={bleu_score:.2f} (evaluated {len(predictions)}/{len(samples)})")

    results["translation_quality"] = translation_results


# ===========================================================================
# D.  TTS Health Check
# ===========================================================================

def evaluate_tts_health(results: dict):
    """
    Validate that TTS produces non-empty audio for each supported language.
    """
    logger.info("=" * 60)
    logger.info("  [D] TTS HEALTH CHECK")
    logger.info("=" * 60)

    try:
        from src.tts_engine import IndicTTSEngine, LANG_META
    except ImportError:
        logger.warning("  ⚠️  TTS engine not importable. Skipping.")
        results["tts_health"] = {"status": "skipped", "reason": "import failed"}
        return

    # Test only our primary languages
    test_langs = {
        "mr": "नमस्कार, हे मराठी भाषेतील चाचणी वाक्य आहे.",
        "gu": "નમસ્તે, આ ગુજરાતી ભાષામાં પરીક્ષણ વાક્ય છે.",
        "hi": "नमस्ते, यह हिंदी भाषा में परीक्षण वाक्य है.",
        "en": "Hello, this is a test sentence in English language.",
    }

    tts_engine = IndicTTSEngine()
    tts_results = {}

    for lang_code, text in test_langs.items():
        logger.info(f"  Testing TTS [{lang_code}]: '{text[:40]}...'")
        try:
            t0 = time.perf_counter()
            result = tts_engine.synthesize(text=text, language=lang_code)
            elapsed = time.perf_counter() - t0

            audio = result.get("audio")
            duration = result.get("duration_s", 0)
            sr = result.get("sample_rate", 0)

            ok = audio is not None and len(audio) > 0 and duration > 0.3
            tts_results[lang_code] = {
                "ok": ok,
                "duration_s": round(duration, 2),
                "sample_rate": sr,
                "generation_time_s": round(elapsed, 2),
                "audio_length": len(audio) if audio is not None else 0,
            }
            status = "✅" if ok else "❌"
            logger.info(f"  {status} [{lang_code}] dur={duration:.1f}s gen_time={elapsed:.1f}s")

        except Exception as e:
            tts_results[lang_code] = {
                "ok": False,
                "error": str(e),
            }
            logger.error(f"  ❌ [{lang_code}] TTS failed: {e}")

    results["tts_health"] = tts_results


# ===========================================================================
# E.  Latency Profiling
# ===========================================================================

def evaluate_latency(agent, lang_samples: dict, results: dict, num_profile: int = 5):
    """
    Profile end-to-end latency breakdown on a small sample.
    """
    logger.info("=" * 60)
    logger.info("  [E] LATENCY PROFILING")
    logger.info("=" * 60)

    fleurs_to_whisper = {"mr_in": "mr", "gu_in": "gu", "hi_in": "hi", "en_us": "en"}
    timings = []

    # Pick a few samples from the first available language
    first_lang = list(lang_samples.keys())[0]
    subset = lang_samples[first_lang][:num_profile]
    whisper_lang = fleurs_to_whisper.get(first_lang, "Autonomous")

    logger.info(f"  Profiling {len(subset)} samples from [{first_lang}]...")

    for i, sample in enumerate(subset):
        audio_path = _save_audio_to_tmp(sample["audio_array"], sample["sampling_rate"])
        try:
            t_total_start = time.perf_counter()
            result = agent.process_audio(audio_path, explicit_lang=whisper_lang)
            t_total = time.perf_counter() - t_total_start

            timings.append(t_total)
            logger.info(f"    Sample {i+1}/{len(subset)}: {t_total:.2f}s")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    if timings:
        latency_results = {
            "language_profiled": first_lang,
            "num_samples": len(timings),
            "avg_total_s": round(np.mean(timings), 3),
            "median_total_s": round(np.median(timings), 3),
            "min_total_s": round(np.min(timings), 3),
            "max_total_s": round(np.max(timings), 3),
            "std_total_s": round(np.std(timings), 3),
        }
    else:
        latency_results = {"status": "no samples processed"}

    results["latency"] = latency_results
    logger.info(
        f"\n  🏆 Avg latency: {latency_results.get('avg_total_s', '?')}s  "
        f"Median: {latency_results.get('median_total_s', '?')}s"
    )


# ===========================================================================
# Report Generation
# ===========================================================================

def print_summary_table(results: dict):
    """Print a clean human-readable summary to stdout."""

    print("\n" + "=" * 70)
    print("  📊  FULL PIPELINE EVALUATION REPORT")
    print("=" * 70)

    # A. STT
    stt = results.get("stt_accuracy", {})
    if stt:
        print("\n  [A] STT Accuracy (WER / CER)")
        print("  " + "-" * 50)
        print(f"  {'Language':<12} {'WER':>8} {'CER':>8} {'Samples':>8} {'Avg Lat':>10}")
        print("  " + "-" * 50)
        for lang, m in stt.items():
            print(
                f"  {lang:<12} {m['wer']:>7.2%} {m['cer']:>7.2%} "
                f"{m['num_samples']:>8} {m['avg_latency_s']:>9.2f}s"
            )

    # B. Routing
    routing = results.get("routing_accuracy", {})
    if routing and "per_lang" in routing:
        print(f"\n  [B] Language Routing Accuracy: {routing['overall']:.2%} overall")
        print("  " + "-" * 50)
        for lang, m in routing["per_lang"].items():
            print(f"  {lang:<12} {m['accuracy']:>7.2%}  ({m['correct']}/{m['total']})")

    # C. Translation
    trans = results.get("translation_quality", {})
    if trans and "status" not in trans:
        print("\n  [C] LLM Translation Quality (BLEU)")
        print("  " + "-" * 50)
        for lang, m in trans.items():
            print(f"  {lang:<12} BLEU={m['bleu']:>6.2f}  (evaluated {m['num_evaluated']}/{m['num_total']})")
    elif isinstance(trans, dict) and trans.get("status") == "skipped":
        print(f"\n  [C] LLM Translation: SKIPPED ({trans.get('reason', '')})")

    # D. TTS
    tts = results.get("tts_health", {})
    if tts and "status" not in tts:
        print("\n  [D] TTS Health Check")
        print("  " + "-" * 50)
        for lang, m in tts.items():
            status = "✅ OK" if m.get("ok") else "❌ FAIL"
            dur = m.get("duration_s", "?")
            gen = m.get("generation_time_s", "?")
            print(f"  {lang:<12} {status:<8}  dur={dur}s  gen_time={gen}s")
    elif isinstance(tts, dict) and tts.get("status") == "skipped":
        print(f"\n  [D] TTS Health Check: SKIPPED ({tts.get('reason', '')})")

    # E. Latency
    lat = results.get("latency", {})
    if lat and "avg_total_s" in lat:
        print(f"\n  [E] Latency Profile ({lat['language_profiled']}, n={lat['num_samples']})")
        print("  " + "-" * 50)
        print(f"  Avg:    {lat['avg_total_s']:.3f}s")
        print(f"  Median: {lat['median_total_s']:.3f}s")
        print(f"  Min:    {lat['min_total_s']:.3f}s")
        print(f"  Max:    {lat['max_total_s']:.3f}s")
        print(f"  Std:    {lat['std_total_s']:.3f}s")

    print("\n" + "=" * 70)
    print(f"  Report saved at: {results.get('output_path', 'N/A')}")
    print("=" * 70 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation Suite for the Indic Cognitive Speech Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num_samples", type=int, default=20,
        help="Number of FLEURS samples per language (default: 20)"
    )
    parser.add_argument(
        "--languages", nargs="+", default=["mr_in", "gu_in", "en_us"],
        help="FLEURS language codes to evaluate (default: mr_in gu_in en_us)"
    )
    parser.add_argument(
        "--skip_tts", action="store_true",
        help="Skip TTS health check evaluation"
    )
    parser.add_argument(
        "--skip_llm", action="store_true",
        help="Skip LLM translation quality evaluation"
    )
    parser.add_argument(
        "--skip_routing", action="store_true",
        help="Skip language routing evaluation (saves time — reuses STT pass)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results",
        help="Directory to write JSON report (default: eval_results/)"
    )
    parser.add_argument(
        "--base_model", type=str, default="openai/whisper-large-v3-turbo",
        help="Base Whisper model ID"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results container
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": args.base_model,
            "num_samples": args.num_samples,
            "languages": args.languages,
            "skip_tts": args.skip_tts,
            "skip_llm": args.skip_llm,
            "skip_routing": args.skip_routing,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
    }

    # -------------------------------------------------------------------
    # 1. Load FLEURS data
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  LOADING FLEURS EVALUATION DATA")
    logger.info("=" * 60)

    lang_samples = {}
    for lang in args.languages:
        lang_samples[lang] = _load_fleurs_subset(lang, split="test", num_samples=args.num_samples)

    # Load English references for BLEU (parallel to Indic samples)
    en_references = {}
    if not args.skip_llm:
        try:
            en_samples = _load_fleurs_subset("en_us", split="test", num_samples=args.num_samples)
            en_references = {i: s["transcription"] for i, s in enumerate(en_samples)}
            logger.info(f"  Loaded {len(en_references)} English reference sentences for BLEU.")
        except Exception as e:
            logger.warning(f"  Could not load English references: {e}")

    # -------------------------------------------------------------------
    # 2. Initialize Agent
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  INITIALIZING COGNITIVE SPEECH AGENT")
    logger.info("=" * 60)

    from src.speech_agent import CognitiveSpeechAgent

    agent = CognitiveSpeechAgent(base_model_id=args.base_model)

    # Load LoRA adapters if available
    mr_adapter = PROJECT_ROOT / "artifacts" / "whisper-marathi-lora-indicspeech" / "checkpoint-1000"
    gu_adapter = PROJECT_ROOT / "artifacts" / "whisper-gujarati-lora-indicspeech" / "checkpoint-1000"

    if mr_adapter.exists():
        agent.load_language_adapter(str(mr_adapter), "mr")
    if gu_adapter.exists():
        agent.load_language_adapter(str(gu_adapter), "gu")

    # -------------------------------------------------------------------
    # 3. Run evaluations
    # -------------------------------------------------------------------

    # A. STT Accuracy
    evaluate_stt_accuracy(agent, lang_samples, results)

    # B. Language Routing
    if not args.skip_routing:
        # Use a small subset for routing to save time
        routing_n = min(10, args.num_samples)
        routing_samples = {
            lang: samples[:routing_n] for lang, samples in lang_samples.items()
        }
        evaluate_routing_accuracy(agent, routing_samples, results)
    else:
        results["routing_accuracy"] = {"status": "skipped"}

    # C. LLM Translation Quality
    if not args.skip_llm:
        # Use a small subset for LLM eval (expensive)
        llm_n = min(10, args.num_samples)
        llm_samples = {
            lang: samples[:llm_n]
            for lang, samples in lang_samples.items()
            if lang != "en_us"
        }
        evaluate_llm_translation(agent, llm_samples, en_references, results)
    else:
        results["translation_quality"] = {"status": "skipped", "reason": "user flag --skip_llm"}

    # D. TTS Health
    if not args.skip_tts:
        evaluate_tts_health(results)
    else:
        results["tts_health"] = {"status": "skipped", "reason": "user flag --skip_tts"}

    # E. Latency Profiling
    evaluate_latency(agent, lang_samples, results)

    # -------------------------------------------------------------------
    # 4. Save report
    # -------------------------------------------------------------------
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"full_pipeline_report_{timestamp_str}.json"
    # Also save as latest
    latest_path = output_dir / "full_pipeline_report.json"

    results["output_path"] = str(report_path)

    for path in [report_path, latest_path]:
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    logger.info(f"  Report saved to: {report_path}")
    logger.info(f"  Latest symlink:  {latest_path}")

    # Print summary
    print_summary_table(results)

    return results


if __name__ == "__main__":
    main()
