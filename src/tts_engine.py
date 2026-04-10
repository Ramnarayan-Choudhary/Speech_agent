"""
Indic Text-to-Speech Engine
Wraps AI4Bharat's Indic Parler-TTS for multilingual speech synthesis.
Supports: Marathi, Gujarati, Hindi, English, and 17+ more Indic languages.
"""

import os
import time
import numpy as np
import torch
import soundfile as sf
from loguru import logger


# Language metadata for TTS
LANG_META = {
    "mr": {"name": "Marathi", "script": "Devanagari", "sample": "नमस्कार, मी मराठी बोलतो. आज हवामान खूप छान आहे."},
    "gu": {"name": "Gujarati", "script": "Gujarati", "sample": "નમસ્તે, હું ગુજરાતી બોલું છું. આજે હવામાન ખૂબ સારું છે."},
    "hi": {"name": "Hindi", "script": "Devanagari", "sample": "नमस्ते, मैं हिंदी बोलता हूं. आज मौसम बहुत अच्छा है."},
    "en": {"name": "English", "script": "Latin", "sample": "Hello, I speak English. The weather is very nice today."},
    "bn": {"name": "Bengali", "script": "Bengali", "sample": "নমস্কার, আমি বাংলায় কথা বলি।"},
    "ta": {"name": "Tamil", "script": "Tamil", "sample": "வணக்கம், நான் தமிழ் பேசுகிறேன்."},
    "te": {"name": "Telugu", "script": "Telugu", "sample": "నమస్కారం, నేను తెలుగు మాట్లాడతాను."},
    "kn": {"name": "Kannada", "script": "Kannada", "sample": "ನಮಸ್ಕಾರ, ನಾನು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇನೆ."},
    "ml": {"name": "Malayalam", "script": "Malayalam", "sample": "നമസ്കാരം, ഞാൻ മലയാളം സംസാരിക്കുന്നു."},
    "pa": {"name": "Punjabi", "script": "Gurmukhi", "sample": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ ਪੰਜਾਬੀ ਬੋਲਦਾ ਹਾਂ."},
    "od": {"name": "Odia", "script": "Odia", "sample": "ନମସ୍କାର, ମୁଁ ଓଡ଼ିଆ କହେ."},
    "as": {"name": "Assamese", "script": "Bengali", "sample": "নমস্কাৰ, মই অসমীয়া কওঁ।"},
}

# Default voice descriptions for Parler-TTS style control
VOICE_PRESETS = {
    "clear_female": "A female speaker with a clear, pleasant voice, moderate speaking rate, and no background noise.",
    "clear_male": "A male speaker with a deep, clear voice, moderate speaking rate, and no background noise.",
    "expressive_female": "A female speaker with an expressive, warm voice, slightly fast speaking rate, and high quality audio.",
    "calm_male": "A male speaker with a calm, soothing voice, slow speaking rate, and very clean audio.",
}


class IndicTTSEngine:
    """
    Text-to-Speech engine using AI4Bharat's Indic Parler-TTS.
    Generates natural speech from text in 21+ Indian languages.
    """

    def __init__(self, model_id: str = "ai4bharat/indic-parler-tts"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = None
        self.tokenizer = None
        self.model_id = model_id
        self._loaded = False

    def load(self):
        """Lazy-load the TTS model into GPU memory."""
        if self._loaded:
            return

        load_start = time.perf_counter()
        logger.info(f"Loading Indic Parler-TTS model [{self.model_id}]...")

        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                padding_side="left",
            )

            self._loaded = True
            logger.info(
                f"Indic Parler-TTS loaded in {time.perf_counter() - load_start:.2f}s "
                f"on {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def synthesize(
        self,
        text: str,
        language: str = "mr",
        voice_preset: str = "clear_female",
        output_path: str = None,
    ) -> dict:
        """
        Convert text to speech audio.

        Args:
            text: The text to synthesize into speech.
            language: Language code (mr, gu, hi, en, etc.)
            voice_preset: Key from VOICE_PRESETS or a custom description string.
            output_path: Optional path to save the WAV file.

        Returns:
            dict with 'audio' (numpy array), 'sample_rate', 'output_path', 'duration_s'
        """
        self.load()

        gen_start = time.perf_counter()
        lang_info = LANG_META.get(language, {"name": language})
        logger.info(f"🔊 TTS: Generating {lang_info.get('name', language)} speech...")

        # Resolve voice description
        description = VOICE_PRESETS.get(voice_preset, voice_preset)

        # Tokenize description and prompt SEPARATELY to avoid pad/eos collision
        desc_tokens = self.tokenizer(description, return_tensors="pt")
        prompt_tokens = self.tokenizer(text, return_tensors="pt")

        input_ids = desc_tokens.input_ids.to(self.device)
        attention_mask = desc_tokens.attention_mask.to(self.device)
        prompt_input_ids = prompt_tokens.input_ids.to(self.device)
        prompt_attention_mask = prompt_tokens.attention_mask.to(self.device)

        # Generate audio
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
            )

        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
        sample_rate = self.model.config.sampling_rate
        duration_s = len(audio_arr) / sample_rate

        logger.info(
            f"TTS generated {duration_s:.1f}s of audio in "
            f"{time.perf_counter() - gen_start:.2f}s"
        )

        # Normalize audio to prevent clipping
        peak = max(abs(audio_arr.max()), abs(audio_arr.min()))
        if peak > 0:
            audio_arr = audio_arr / peak * 0.95

        # Save if path provided
        if output_path:
            sf.write(output_path, audio_arr, sample_rate)
            logger.info(f"Audio saved to {output_path}")

        return {
            "audio": audio_arr,
            "sample_rate": sample_rate,
            "output_path": output_path,
            "duration_s": round(duration_s, 2),
            "language": language,
            "language_name": lang_info.get("name", language),
        }

    def get_supported_languages(self) -> list:
        """Return list of supported language codes and names."""
        return [
            {"code": code, **meta} for code, meta in LANG_META.items()
        ]


if __name__ == "__main__":
    engine = IndicTTSEngine()

    # Test Marathi
    result = engine.synthesize(
        text="नमस्कार, हे मराठी भाषेतील चाचणी आहे.",
        language="mr",
        output_path="test_tts_marathi.wav",
    )
    print(f"Marathi: {result['duration_s']}s saved to {result['output_path']}")

    # Test Gujarati
    result = engine.synthesize(
        text="નમસ્તે, આ ગુજરાતી ભાષામાં પરીક્ષણ છે.",
        language="gu",
        output_path="test_tts_gujarati.wav",
    )
    print(f"Gujarati: {result['duration_s']}s saved to {result['output_path']}")
