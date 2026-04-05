import librosa
import torch
import warnings
warnings.filterwarnings('ignore')
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading on {device}")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

# Load Marathi adapter
model = PeftModel.from_pretrained(
    base_model,
    "artifacts/whisper-marathi-lora-indicspeech/checkpoint-500",
    adapter_name="mr"
)
model.load_adapter("artifacts/whisper-gujarati-lora-indicspeech/checkpoint-500", adapter_name="gu")

def test_inference(audio_path, lang):
    audio, sr = librosa.load(audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    model.set_adapter(lang)
    print(f"\n--- Testing {lang.upper()} on {audio_path} ---")
    
    # Method 1: Just task and language
    out1 = model.generate(input_features, max_new_tokens=128, language=lang, task="transcribe")
    print("Method 1 (lang/task kwargs):", processor.batch_decode(out1, skip_special_tokens=True)[0])
    
    # Method 2: forced_decoder_ids
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
    out2 = model.generate(input_features, max_new_tokens=128, forced_decoder_ids=forced_decoder_ids)
    print("Method 2 (forced_decoder_ids):", processor.batch_decode(out2, skip_special_tokens=True)[0])
    
    # Method 3: No kwargs (Base fallback mimicking)
    out3 = model.generate(input_features, max_new_tokens=128)
    print("Method 3 (No Kwargs):", processor.batch_decode(out3, skip_special_tokens=True)[0])

test_inference("demo_marathi.wav", "mr")
test_inference("demo_gujarati.wav", "gu")
