import librosa
import torch
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda")

for lang_name, f in [("Marathi", "demo_marathi.wav"), ("Gujarati", "demo_gujarati.wav")]:
    audio, sr = librosa.load(f, sr=16000)
    feat = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")

    # Native base generation
    model.config.forced_decoder_ids = None
    out = model.generate(feat, max_new_tokens=3)
    dec = processor.tokenizer.decode(out[0])
    print(f"\n{lang_name} Audio Base Decode: {dec}")

    # Proper Logits evaluation method!
    # Whisper starts with <|startoftranscript|> which is token 50258
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to("cuda")
    with torch.no_grad():
        logits = model(feat, decoder_input_ids=decoder_input_ids).logits
    
    # The language tokens are in the range 50259 to 50357
    lang_logits = logits[0, -1, 50259:50358]
    best_lang_idx = lang_logits.argmax().item() + 50259
    best_lang = processor.tokenizer.decode([best_lang_idx])
    print(f"{lang_name} Highest Prob Lang Token: {best_lang}")
