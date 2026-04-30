import os
import time
import json
import torch
import soundfile as sf
from jiwer import wer, cer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from tts_engine import IndicTTSEngine

# Test sentences for evaluation
TEST_CASES = [
    {"lang": "en", "text": "The quick brown fox jumps over the lazy dog."},
    {"lang": "en", "text": "We are evaluating the text to speech engine for intelligibility."},
    {"lang": "mr", "text": "नमस्कार, मी मराठी बोलतो. आज हवामान खूप छान आहे."},
    {"lang": "mr", "text": "हे एक चाचणी वाक्य आहे जे आपण तपासण्यासाठी वापरत आहोत."},
    {"lang": "gu", "text": "નમસ્તે, હું ગુજરાતી બોલું છું. આજે હવામાન ખૂબ સારું છે."},
    {"lang": "gu", "text": "આ એક પરીક્ષણ વાક્ય છે જેનો ઉપયોગ આપણે ચકાસણી માટે કરી રહ્યા છીએ."},
]

def load_asr_model():
    """Load Whisper Turbo for ASR evaluation."""
    print("Loading ASR Model (whisper-large-v3-turbo)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=False,
    )
    return pipe

def evaluate():
    print("Initializing TTS Engine...")
    tts = IndicTTSEngine()
    tts.load()
    
    pipe = load_asr_model()
    
    results = []
    total_audio_duration = 0
    total_processing_time = 0
    
    os.makedirs("eval_results", exist_ok=True)
    
    print("\n--- Starting Evaluation ---")
    for i, test in enumerate(TEST_CASES):
        lang = test["lang"]
        text = test["text"]
        print(f"\n[{lang.upper()}] Text: {text}")
        
        # 1. Generate Audio
        t0 = time.perf_counter()
        result_dict = tts.synthesize(text, language=lang)
        audio_array = result_dict["audio"]
        sr = result_dict["sample_rate"]
        t1 = time.perf_counter()
        
        processing_time = t1 - t0
        audio_duration = len(audio_array) / sr
        rtf = processing_time / audio_duration
        
        total_processing_time += processing_time
        total_audio_duration += audio_duration
        
        # Save audio temp
        temp_audio = f"eval_results/temp_tts_{i}.wav"
        sf.write(temp_audio, audio_array, sr)
        
        # 2. ASR Transcription
        lang_map = {"en": "english", "mr": "marathi", "gu": "gujarati"}
        result = pipe(temp_audio, generate_kwargs={"language": lang_map[lang]})
        transcription = result["text"].strip()
        
        # 3. Calculate metrics
        # Normalize a bit for WER
        ref = text.lower()
        hyp = transcription.lower()
        w_err = wer(ref, hyp)
        c_err = cer(ref, hyp)
        
        print(f"Transcribed: {transcription}")
        print(f"WER: {w_err:.3f} | CER: {c_err:.3f} | RTF: {rtf:.3f}")
        
        results.append({
            "language": lang,
            "original_text": text,
            "transcribed_text": transcription,
            "wer": w_err,
            "cer": c_err,
            "processing_time": processing_time,
            "audio_duration": audio_duration,
            "rtf": rtf
        })
        
        # Cleanup
        os.remove(temp_audio)

    # Aggregate
    overall_rtf = total_processing_time / total_audio_duration
    avg_wer = sum(r["wer"] for r in results) / len(results)
    avg_cer = sum(r["cer"] for r in results) / len(results)
    
    summary = {
        "overall_rtf": overall_rtf,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "total_samples": len(TEST_CASES),
        "details": results
    }
    
    with open("eval_results/tts_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    print(f"\n=== Final TTS Evaluation ===")
    print(f"Overall RTF: {overall_rtf:.3f}")
    print(f"Average WER: {avg_wer:.3f}")
    print(f"Average CER: {avg_cer:.3f}")
    print("Detailed report saved to eval_results/tts_eval_report.json")

if __name__ == "__main__":
    evaluate()
