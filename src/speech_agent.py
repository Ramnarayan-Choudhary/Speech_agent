import os
import time

import librosa
import torch
from loguru import logger
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---------------------------------------------------------------------------
# Graceful optional imports — the app runs for STT even without these
# ---------------------------------------------------------------------------
try:
    import weave
except ImportError:
    weave = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# Safe decorator: use weave.op() when available, identity otherwise
def _identity_decorator(*_args, **_kwargs):
    """No-op decorator used when weave is not installed."""
    def wrapper(func):
        return func
    return wrapper


weave_op = weave.op if weave is not None else _identity_decorator

SUPPORTED_LANGUAGE_CODES = ("en", "mr", "gu")
LANG_NAMES = {"en": "English", "mr": "Marathi", "gu": "Gujarati", "unknown": "Unknown"}


class CognitiveSpeechAgent:
    """
    A Level 1 & Level 2 Cognitive Speech Agent.
    - Level 1: Auto-Routing LoRA weights based on Language Detection.
    - Level 2: Self-Reflecting Translation via W&B Inference LLMs.
    """
    def __init__(self, base_model_id: str = "openai/whisper-large-v3-turbo"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Initializing Cognitive Agent on {self.device}...")

        if self.device == "cpu":
            logger.warning(
                "CUDA is not available. Whisper-small inference will be slow on CPU."
            )

        # Load Base Whisper Brain
        model_load_start = time.perf_counter()
        try:
            self.processor = WhisperProcessor.from_pretrained(base_model_id)
            self.base_model = WhisperForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            ).to(self.device)
        except OSError as exc:
            raise OSError(
                "Unable to load the Whisper base model. If you are offline, set "
                "WHISPER_BASE_MODEL_ID to a local model directory or pre-populate "
                "the Hugging Face cache."
            ) from exc

        logger.info(
            f"Base Whisper model loaded in {time.perf_counter() - model_load_start:.2f}s"
        )
        self.base_model.config.forced_decoder_ids = None
        self.base_model.config.suppress_tokens = []

        # LoRA adapter registry (deprecated for Turbo run, keeping structure safe)
        self.adapters_loaded = {}
        self.model = None

        # Phase 3 VAD: Initialize Silero VAD
        self.vad_model = None
        try:
            vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.vad_model = vad_model.to(self.device)
            self.get_speech_timestamps = utils[0]
            self.collect_chunks = utils[4]
            logger.info("Silero VAD initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load Silero VAD: {e}")

        # Level 2 LLM Setup (Weights & Biases Inference) — entirely optional
        self.llm_client = None
        self.llm_model_name = None
        wandb_api_key = os.environ.get("WANDB_API_KEY")

        if not wandb_api_key:
            logger.info(
                "WANDB_API_KEY not found. LLM polishing/translation will be skipped. "
                "Set WANDB_API_KEY in .env to enable it."
            )
        elif OpenAI is None:
            logger.warning(
                "openai package not installed. LLM polishing/translation disabled."
            )
        else:
            llm_init_start = time.perf_counter()
            # Initialize Weave tracing if available
            if weave is not None:
                try:
                    weave.init("personal_rc/speech_agent")
                except Exception as exc:
                    logger.warning(
                        f"Weave initialization failed, continuing without tracing: {exc}"
                    )

            self.llm_client = OpenAI(
                base_url="https://api.inference.wandb.ai/v1",
                api_key=wandb_api_key,
            )
            self.llm_model_name = "OpenPipe/Qwen3-14B-Instruct"
            logger.info(
                f"LLM layer initialized in {time.perf_counter() - llm_init_start:.2f}s"
            )

    def load_language_adapter(self, adapter_path: str, lang_code: str):
        """Loads a tuned LoRA completely invisibly onto the base brain."""
        if PeftModel is None:
            logger.warning("PEFT is not installed. Adapter loading is unavailable.")
            return

        try:
            adapter_load_start = time.perf_counter()
            if self.model is None:
                self.model = PeftModel.from_pretrained(
                    self.base_model, adapter_path, adapter_name=lang_code
                )
            else:
                self.model.load_adapter(adapter_path, adapter_name=lang_code)
            self.adapters_loaded[lang_code] = True
            logger.info(
                f"Loaded LoRA Adapter for [{lang_code}] from {adapter_path} "
                f"in {time.perf_counter() - adapter_load_start:.2f}s"
            )
        except Exception as e:
            logger.error(f"Failed to load adapter for {lang_code}: {e}")

    def detect_language(self, input_features) -> str:
        """Domain-Constrained Probability LID using the Base Whisper Model."""
        if self.model is not None:
            self.model.disable_adapter_layers()

        with torch.no_grad():
            decoder_input_ids = torch.tensor(
                [[self.base_model.config.decoder_start_token_id]]
            ).to(self.device)
            logits = self.base_model(
                input_features, decoder_input_ids=decoder_input_ids
            ).logits

        if self.model is not None:
            self.model.enable_adapter_layers()

        best_lang = "unknown"
        best_prob = -float("inf")

        for lang in SUPPORTED_LANGUAGE_CODES:
            token_idx = self.processor.tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
            logit_score = logits[0, -1, token_idx].item()
            logger.info(f"Base Model Domain LID - [{lang}]: {logit_score:.2f}")
            if logit_score > best_prob:
                best_prob = logit_score
                best_lang = lang

        logger.info(f"🏆 Constrained LID Routing WINNER: {best_lang}")
        return best_lang

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove Qwen3 <think>...</think> reasoning artifacts from output."""
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    @weave_op()
    def llm_translation_agent(self, raw_text: str, language: str, context: str = "") -> str:
        """Level 2 Cognitive Pass: Uses W&B traced LLMs to clean STT and translate."""
        if self.llm_client is None:
            return "LLM polishing is disabled. Raw transcription is available above."

        lang_name = LANG_NAMES.get(language, language)
        logger.info(f"LLM reviewing [{lang_name}] transcript...")
        
        # Phase 3: RAG Context Injection
        context_prompt = ""
        if context.strip():
            context_prompt = f"The user provided the following contextual background regarding this audio: '{context}'. Use this to accurately resolve ambiguous specific nouns or regional terms.\n\n"

        sys_msg = (
            f"You are a professional audio translator and cognitive filter. "
            f"The user spoke in {lang_name}. We used an acoustic STT model to generate raw text.\n"
            f"Sometimes acoustic models misinterpret names or add trailing repetitions.\n\n{context_prompt}"
            f"Your exact constraints:\n"
            f"1. Fix any phonetic errors in the native language text (output as [CLEANED]: text).\n"
            f"2. Translate the finalized text to fluent English (output as [ENGLISH]: text).\n"
            f"3. Output strictly these two lines and ABSOLUTELY NOTHING ELSE. No reasoning tags, no chatty text."
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"/no_think\n{raw_text}"},
                ],
                temperature=0.7,
                top_p=0.8,
                extra_body={"enable_thinking": False},
            )
            result = response.choices[0].message.content
            # Strip any Qwen3 thinking artifacts
            result = self._strip_thinking_tags(result)
            logger.info(f"LLM response: {result[:200]}")
            return result
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return f"LLM processing failed: {exc}"

    def _transcribe(self, model, input_features, lang: str) -> str:
        """Run transcription with a given model and return text."""
        pred_ids = model.generate(
            input_features,
            max_new_tokens=440,
            num_beams=5,
            language=lang if lang != "unknown" else None,
            task="transcribe",
        )
        return (
            self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            .strip()
        )

    def process_audio(self, audio_path: str, explicit_lang: str = "Autonomous", context: str = "") -> dict:
        """The Main Agentic Loop."""
        self.current_context = context
        request_start = time.perf_counter()
        logger.info("====================================")
        logger.info(f"🎙️ Processing Audio: {os.path.basename(audio_path)}")

        # 1. Load Audio
        self.current_audio_path = audio_path
        preprocess_start = time.perf_counter()
        audio, sr = librosa.load(audio_path, sr=16000)
        duration_s = len(audio) / sr
        logger.info(f"Audio loaded: {duration_s:.1f}s, {len(audio)} samples at {sr}Hz")

        # Phase 3 VAD: Filter Out Silent Frames
        if getattr(self, "vad_model", None) is not None:
            vad_start = time.perf_counter()
            # Silero expects torch tensor on the same device as the model
            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            try:
                speech_timestamps = self.get_speech_timestamps(
                    audio_tensor, self.vad_model, sampling_rate=16000
                )
                if speech_timestamps:
                    # collect_chunks typically returns dict or tensors; we ensure it's CPU numpy
                    audio_tensor = self.collect_chunks(speech_timestamps, audio_tensor)
                    audio = audio_tensor.cpu().numpy()
                    logger.info(f"Silero VAD trimmed audio to {len(audio)/16000:.1f}s (took {time.perf_counter() - vad_start:.2f}s)")
                else:
                    logger.warning("Silero VAD found no speech! Proceeding with raw audio.")
            except Exception as e:
                logger.error(f"Silero VAD failed during inference: {e}")

        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.torch_dtype)
        logger.info(
            f"Audio preprocessing took {time.perf_counter() - preprocess_start:.2f}s"
        )

        # 2. LEVEL 1: Automatic Cognitive Routing
        routing_start = time.perf_counter()
        if explicit_lang in SUPPORTED_LANGUAGE_CODES:
            detected_lang = explicit_lang
            logger.info(f"User Explicitly Forced Language Routing: {detected_lang}")
        elif explicit_lang != "Autonomous":
            detected_lang = self.detect_language(input_features)
        else:
            detected_lang = self.detect_language(input_features)
        logger.info(
            f"Language routing took {time.perf_counter() - routing_start:.2f}s"
        )

        # 3. TRANSCRIPTION: SOTA LoRA Pipeline
        transcription_start = time.perf_counter()

        with torch.no_grad():
            if detected_lang in self.adapters_loaded and self.model is not None:
                self.model.set_adapter(detected_lang)
                logger.info(f"Language [{detected_lang}] autonomously detected! Swapping to tuned LoRA matrix...")
                transcription = self._transcribe(self.model, input_features, detected_lang)
                stt_source = f"Turbo LoRA ({detected_lang})"
            else:
                if self.model is not None:
                    self.model.disable_adapter_layers()
                logger.warning(f"Language [{detected_lang}] fallback to Standard Base Model.")
                transcription = self._transcribe(self.base_model, input_features, detected_lang)
                stt_source = "Turbo Base"

        logger.info(
            f"Transcription took {time.perf_counter() - transcription_start:.2f}s "
            f"(selected: {stt_source})"
        )
        logger.info(f"FINAL STT Output [{stt_source}]: {transcription}")

        # 4. LEVEL 2: LLM Translation / Polish
        if self.llm_client is not None:
            context = getattr(self, "current_context", "")
            refined_output = self.llm_translation_agent(
                raw_text=transcription, language=detected_lang, context=context
            )
        else:
            refined_output = "LLM polishing is disabled. Raw transcription is available above."

        total_time = time.perf_counter() - request_start
        logger.success("====================================")
        logger.info(f"Total request time: {total_time:.2f}s")

        return {
            "detected_language": detected_lang,
            "raw_stt": transcription,
            "stt_source": stt_source,
            "llm_enabled": self.llm_client is not None,
            "agent_final_response": refined_output,
        }


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    # Load .env from THIS project's root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(project_root, ".env"))

    agent = CognitiveSpeechAgent()

    # Load the two LoRA adapters (Phase 4: Checkpoint 1000)
    mr_adapter = os.path.join(
        project_root, "artifacts", "whisper-marathi-lora-indicspeech", "checkpoint-1000"
    )
    gu_adapter = os.path.join(
        project_root, "artifacts", "whisper-gujarati-lora-indicspeech", "checkpoint-1000"
    )

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
