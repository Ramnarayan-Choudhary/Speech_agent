import os
import gradio as gr
from dotenv import load_dotenv
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# We must be in the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))
os.environ["PYTHONPATH"] = ROOT_DIR

# SLURM fix: redirect temp dirs to project-local writable directory
# (avoids PermissionError on /tmp/gradio/ in SLURM environments)
_gradio_tmp = os.path.join(ROOT_DIR, ".gradio")
os.makedirs(_gradio_tmp, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", _gradio_tmp)
os.environ.setdefault("TMPDIR", _gradio_tmp)
os.environ.setdefault("TMP", _gradio_tmp)
os.environ.setdefault("TEMP", _gradio_tmp)

# Import the brain
try:
    from src.speech_agent import CognitiveSpeechAgent
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.speech_agent import CognitiveSpeechAgent

# 1. Initialize the global Cognitive Agent (Loads Whisper Base Brain into GPU once)
print("Initializing Cognitive Agent Backend...")
agent = CognitiveSpeechAgent()

# 2. Phase 4: Dynamic LoRA Matrix Loading for Turbo
mr_adapter = os.path.join(ROOT_DIR, "artifacts", "whisper-marathi-lora-indicspeech", "checkpoint-1000")
gu_adapter = os.path.join(ROOT_DIR, "artifacts", "whisper-gujarati-lora-indicspeech", "checkpoint-1000")

if os.path.exists(mr_adapter):
    agent.load_language_adapter(mr_adapter, "mr")
if os.path.exists(gu_adapter):
    agent.load_language_adapter(gu_adapter, "gu")

import soundfile as sf
import time

def process_audio_ui(audio_input_data, routing_mode, ui_context=""):
    """The Gradio Callback Function."""
    if audio_input_data is None:
        logger.error("audio_input_data was None!")
        return "Not Detected", "No audio provided.", "Awaiting audio input..."
        
    try:
        sr, y = audio_input_data
        
        # --- Audio Normalization ---
        # Gradio sends raw numpy arrays (often int16/int32 at 48kHz).
        # Whisper expects 16kHz float32 audio in [-1, 1] range.
        import numpy as np
        
        # Convert integer audio to float32 [-1, 1]
        if np.issubdtype(y.dtype, np.integer):
            max_val = np.iinfo(y.dtype).max
            y = y.astype(np.float32) / max_val
        else:
            y = y.astype(np.float32)
        
        # If stereo, convert to mono
        if y.ndim > 1:
            y = y.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Save normalized audio
        local_audio_path = os.path.join(ROOT_DIR, f"safe_upload_{int(time.time())}.wav")
        sf.write(local_audio_path, y, sr, subtype='PCM_16')
        
        # Run the speech agent pipeline
        results = agent.process_audio(local_audio_path, explicit_lang=routing_mode, context=ui_context)
        
        detected = results.get("detected_language", "Unknown")
        raw_text = results.get("raw_stt", "")
        stt_source = results.get("stt_source", "")
        agent_resp = results.get("agent_final_response", "")
        
        # Cleanup the temp file
        if os.path.exists(local_audio_path):
            os.remove(local_audio_path)
            
        # Format language nicely
        lang_map = {"en": "English (Base Model) 🇬🇧", "mr": "Marathi (मराठी) 🇮🇳", "gu": "Gujarati (ગુજરાતી) 🇮🇳", "unknown": "Unknown Base / English"}
        formatted_lang = lang_map.get(detected, detected)
        
        # Show which model produced the STT
        raw_text_display = f"[{stt_source} Model]\n{raw_text}" if stt_source else raw_text
        
        return formatted_lang, raw_text_display, agent_resp
    except Exception as e:
        logger.error(f"Gradio Error: {e}")
        return "Error", f"Crash Log:\n{str(e)}", "Please check your microphone and try again."

# 3. Initialize TTS Engine (Lazy Load — only loads when first TTS request arrives)
from src.tts_engine import IndicTTSEngine, LANG_META, VOICE_PRESETS
tts_engine = IndicTTSEngine()

def process_tts_ui(text_input, language, voice_style):
    """The Gradio TTS Callback Function."""
    if not text_input or not text_input.strip():
        return None, "Please enter some text to synthesize."

    try:
        output_path = os.path.join(ROOT_DIR, f"tts_output_{int(time.time())}.wav")
        result = tts_engine.synthesize(
            text=text_input.strip(),
            language=language,
            voice_preset=voice_style,
            output_path=output_path,
        )
        status = (
            f"✅ Generated {result['duration_s']}s of {result['language_name']} speech "
            f"using [{voice_style}] voice."
        )
        return output_path, status
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return None, f"❌ TTS Failed: {str(e)}"


# 4. Build the Beautiful Gradio Interface with Tabs
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ Indic Cognitive Speech Agent")
    gr.Markdown("""
    ### Full-Loop Multilingual Speech AI
    A bidirectional speech system powered by `Whisper-Large-v3-Turbo` (809M) + LoRA fine-tuning and `Indic Parler-TTS` (AI4Bharat).
    **Mode 1:** Voice → Text (Speech Recognition + LLM Translation)  
    **Mode 2:** Text → Voice (Indic Text-to-Speech Synthesis)
    """)

    with gr.Tabs():
        # =================== TAB 1: SPEECH-TO-TEXT ===================
        with gr.TabItem("🎤 Speech → Text (STT)"):
            gr.Markdown("Upload or record audio in **Marathi**, **Gujarati**, or **English**. The AI will transcribe, translate, and polish the output.")
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Input your Speech here")
                    with gr.Row():
                        routing_mode = gr.Radio(choices=["Autonomous", "mr", "gu"], value="Autonomous", label="Intelligence Routing Mode", scale=1)
                        context_box = gr.Textbox(label="RAG Context Grounding (Optional)", placeholder="E.g., 'Discussing agriculture rain damage in Gujarat'", scale=2)
                    submit_btn = gr.Button("🧠 Process Audio", variant="primary")

                with gr.Column():
                    lang_output = gr.Textbox(label="1. Final Language Routed", placeholder="Waiting for Speech...")
                    raw_output = gr.Textbox(label="2. Raw STT Output (Local GPU)", placeholder="Transcription will appear here...")
                    agent_output = gr.Textbox(label="3. Cognitive Translation (Weave + Qwen3-14B)", lines=4, placeholder="LLM logic will load here...")

            submit_btn.click(
                fn=process_audio_ui,
                inputs=[audio_input, routing_mode, context_box],
                outputs=[lang_output, raw_output, agent_output]
            )

        # =================== TAB 2: TEXT-TO-SPEECH ===================
        with gr.TabItem("🔊 Text → Speech (TTS)"):
            gr.Markdown("Type text in any supported **Indic language** and generate natural-sounding speech using AI4Bharat's Indic Parler-TTS.")
            with gr.Row():
                with gr.Column():
                    tts_text = gr.Textbox(
                        label="Enter Text to Speak",
                        placeholder="E.g., નમસ્તે, આ ગુજરાતી ભાષામાં ટેક્સ્ટ છે.",
                        lines=4,
                    )
                    with gr.Row():
                        tts_lang = gr.Dropdown(
                            choices=[(f"{v['name']} ({k})", k) for k, v in LANG_META.items()],
                            value="mr",
                            label="Language",
                            scale=1,
                        )
                        tts_voice = gr.Dropdown(
                            choices=[(k.replace("_", " ").title(), k) for k in VOICE_PRESETS.keys()],
                            value="clear_female",
                            label="Voice Style",
                            scale=1,
                        )
                    tts_btn = gr.Button("🔊 Generate Speech", variant="primary")

                with gr.Column():
                    tts_audio_output = gr.Audio(label="Generated Speech", type="filepath")
                    tts_status = gr.Textbox(label="Status", placeholder="Awaiting text input...")

            tts_btn.click(
                fn=process_tts_ui,
                inputs=[tts_text, tts_lang, tts_voice],
                outputs=[tts_audio_output, tts_status],
            )

            # Quick demo buttons
            gr.Markdown("#### Quick Demo Samples")
            with gr.Row():
                for code, meta in [("mr", LANG_META["mr"]), ("gu", LANG_META["gu"]), ("hi", LANG_META.get("hi", {}))]:
                    sample_text = meta.get("sample", "")
                    if sample_text:
                        gr.Button(f"Try {meta['name']}", size="sm").click(
                            fn=lambda t=sample_text, l=code: (t, l),
                            outputs=[tts_text, tts_lang],
                        )

    gr.Markdown("*Powered by MBZUAI SLURM Cluster, Weights & Biases Inference, and AI4Bharat.*")

if __name__ == "__main__":
    logger.info("Spinning up Public UI Server...")
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)

