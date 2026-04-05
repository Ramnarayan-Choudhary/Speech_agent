import os
import torch
import weave
from openai import OpenAI
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from loguru import logger
import soundfile as sf
import librosa

class CognitiveSpeechAgent:
    """
    A Level 1 & Level 2 Cognitive Speech Agent.
    - Level 1: Auto-Routing LoRA weights based on Language Detection.
    - Level 2: Self-Reflecting Translation via W&B Inference LLMs.
    """
    def __init__(self, base_model_id: str = "openai/whisper-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Cognitive Agent on {self.device}...")
        
        # Load Base Whisper Brain
        self.processor = WhisperProcessor.from_pretrained(base_model_id)
        self.base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id).to(self.device)
        self.base_model.config.forced_decoder_ids = None
        self.base_model.config.suppress_tokens = []
        
        # Load LoRA Adapters into memory without swapping the base model out!
        self.adapters_loaded = {}
        self.model = None

        # Level 2 LLM Setup (Weights & Biases Inference)
        if "WANDB_API_KEY" not in os.environ:
            logger.warning("WANDB_API_KEY not found. LLM translation will fail.")
        else:
            weave.init("personal_rc/speech_agent")
            self.llm_client = OpenAI(
                base_url='https://api.inference.wandb.ai/v1',
                api_key=os.environ['WANDB_API_KEY'],
            )
            self.llm_model_name = "OpenPipe/Qwen3-14B-Instruct"

    def load_language_adapter(self, adapter_path: str, lang_code: str):
        """Loads a tuned LoRA completely invisibly onto the base brain."""
        try:
            if self.model is None:
                self.model = PeftModel.from_pretrained(self.base_model, adapter_path, adapter_name=lang_code)
            else:
                self.model.load_adapter(adapter_path, adapter_name=lang_code)
            self.adapters_loaded[lang_code] = True
            logger.info(f"Loaded LoRA Adapter for [{lang_code}] from {adapter_path}")
        except Exception as e:
            logger.error(f"Failed to load adapter for {lang_code}: {e}")

    def detect_language(self, input_features) -> str:
        """Domain-Constrained Probability LID using the Base Whisper Model."""
        if self.model is not None:
            self.model.disable_adapter_layers()
        
        with torch.no_grad():
            decoder_input_ids = torch.tensor([[self.base_model.config.decoder_start_token_id]]).to(self.device)
            logits = self.base_model(input_features, decoder_input_ids=decoder_input_ids).logits
        
        if self.model is not None:
            self.model.enable_adapter_layers()
            
        target_langs = ["en", "mr", "gu"]
        best_lang = "unknown"
        best_prob = -float("inf")
        
        for lang in target_langs:
            token_idx = self.processor.tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
            logit_score = logits[0, -1, token_idx].item()
            logger.info(f"Base Model Domain LID - [{lang}]: {logit_score:.2f}")
            if logit_score > best_prob:
                best_prob = logit_score
                best_lang = lang
                
        logger.info(f"🏆 Constrained LID Routing WINNER: {best_lang}")
        return best_lang

    @weave.op() # We track the LLM translation magically through W&B!
    def llm_translation_agent(self, raw_text: str, language: str) -> str:
        """Level 2 Cognitive Pass: Uses W&B traced LLMs to clean STT and translate to English."""
        logger.info(f"Agents reviewing {language} transcript via LLM...")
        
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.inference.wandb.ai/v1",
            api_key=os.environ.get("WANDB_API_KEY")
        )
        
        # If it's pure English directly from the base Whisper, we just do a grammar check.
        if language == "en":
            system_prompt = "You are a linguistic editor. The user provides a raw English speech-to-text transcript. Fix any minor phonetic typos or grammar mistakes. DO NOT translate, just return the polished English. Output format: \n[CLEANED]: {fixed english}"
        else:
            system_prompt = f"You are a linguistic editor. Review the raw {language} speech-to-text transcript. Fix grammar mistakes, phonetic STT errors. Provide the clean native text, and a perfect English translation. Output MUST rigidly follow this format: \n[CLEANED]: {{fixed native text}} \n[ENGLISH]: {{english translation}}"
            
        response = client.chat.completions.create(
            model="OpenPipe/Qwen3-14B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def process_audio(self, audio_path: str, explicit_lang: str = "Autonomous") -> dict:
        """The Main Agentic Loop."""
        logger.info("====================================")
        logger.info(f"🎙️ Processing Audio: {os.path.basename(audio_path)}")
        
        # 1. Load Audio
        self.current_audio_path = audio_path
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

        # 2. LEVEL 1: Automatic Cognitive Routing
        if explicit_lang != "Autonomous":
            detected_lang = explicit_lang
            logger.info(f"User Explicitly Forced Language Routing: {detected_lang}")
        else:
            detected_lang = self.detect_language(input_features)
        
        # Route to exact LoRA adapter if we have trained it!
        if detected_lang in self.adapters_loaded:
            logger.info(f"Language [{detected_lang}] autonomously detected! Swapping to tuned LoRA Brain...")
            if self.model is not None:
                self.model.set_adapter(detected_lang)
            active_model = self.model
        else:
            logger.warning(f"Language [{detected_lang}] detected. No specialized LoRA found. Falling back to Standard Base Model.")
            if self.model is not None:
                self.model.disable_adapter_layers()
            active_model = self.base_model
            
        # 3. Transcribe precise text without breaking constraint parameters
        with torch.no_grad():
            pred_ids = active_model.generate(
                input_features, 
                max_new_tokens=128,
                language=detected_lang if detected_lang != "unknown" else None,
                task="transcribe"
            )
            transcription = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip().strip()
        
        logger.info(f"RAW STT Output: {transcription}")

        # 4. LEVEL 2: Weave LLM Translating
        if hasattr(self, "llm_client"):
            refined_output = self.llm_translation_agent(raw_text=transcription, language=detected_lang)
        else:
            refined_output = "W&B LLM Integration not configured."
            
        logger.success("====================================")
        return {
            "detected_language": detected_lang,
            "raw_stt": transcription,
            "agent_final_response": refined_output
        }

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv("/home/ramnarayan.ramniwas/MS_projects/Speech_agent/.env")
    
    agent = CognitiveSpeechAgent()
    
    # Let's load the two LoRAs we just spent 2 hours tuning!
    mr_adapter = os.path.join("artifacts", "whisper-marathi-lora-indicspeech", "checkpoint-500")
    gu_adapter = os.path.join("artifacts", "whisper-gujarati-lora-indicspeech", "checkpoint-500")
    
    if os.path.exists(mr_adapter):
        agent.load_language_adapter(mr_adapter, "mr")
    if os.path.exists(gu_adapter):
        agent.load_language_adapter(gu_adapter, "gu")
        
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            result = agent.process_audio(audio_file)
            print("\n>>> AGENT OUTPUT <<<\n")
            print(result["agent_final_response"])
        else:
            print(f"File not found: {audio_file}")
    else:
        print("Provide an audio file to test: python src/speech_agent.py my_audio.wav")
