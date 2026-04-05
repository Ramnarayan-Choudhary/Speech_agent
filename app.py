import os
import gradio as gr
from dotenv import load_dotenv
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# We must be in the root directory
load_dotenv(".env")
os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

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

# 2. Attach our Trained Adapters dynamically (if they exist)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
mr_adapter = os.path.join(ROOT_DIR, "artifacts", "whisper-marathi-lora-indicspeech", "checkpoint-500")
gu_adapter = os.path.join(ROOT_DIR, "artifacts", "whisper-gujarati-lora-indicspeech", "checkpoint-500")

# Load Marathi
if os.path.exists(mr_adapter):
    logger.info("Found Marathi Adapter. Loading...")
    agent.load_language_adapter(mr_adapter, "mr")

# Load Gujarati
if os.path.exists(gu_adapter):
    logger.info("Found Gujarati Adapter. Loading...")
    agent.load_language_adapter(gu_adapter, "gu")

import soundfile as sf
import time

def process_audio_ui(audio_input_data, routing_mode):
    """The Gradio Callback Function."""
    if audio_input_data is None:
        logger.error("audio_input_data was None!")
        return "Not Detected", "No audio provided.", "Awaiting audio input..."
        
    try:
        sr, y = audio_input_data
        
        # Save it to our local project directory to completely bypass SLURM /tmp node restrictions
        local_audio_path = os.path.join(ROOT_DIR, f"safe_upload_{int(time.time())}.wav")
        sf.write(local_audio_path, y, sr)
        
        # Run our phase 3 pipeline seamlessly
        results = agent.process_audio(local_audio_path, explicit_lang=routing_mode)
        
        detected = results.get("detected_language", "Unknown")
        raw_text = results.get("raw_stt", "")
        agent_resp = results.get("agent_final_response", "")
        
        # Cleanup the custom temp file
        if os.path.exists(local_audio_path):
            os.remove(local_audio_path)
            
        # Format language nicely
        lang_map = {"en": "English (Base Model) 🇬🇧", "mr": "Marathi (मराठी) 🇮🇳", "gu": "Gujarati (ગુજરાતી) 🇮🇳", "unknown": "Unknown Base / English"}
        formatted_lang = lang_map.get(detected, detected)
        
        return formatted_lang, raw_text, agent_resp
    except Exception as e:
        logger.error(f"Gradio Error: {e}")
        return "Error", f"Crash Log:\n{str(e)}", "Please check your microphone and try again."

# 3. Build the Beautiful Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ Indic Cognitive Speech Agent")
    gr.Markdown("""
    ### Multilingual Cognitive Auto-Routing AI
    Speak into your microphone in **Marathi** or **Gujarati**. 
    The AI will autonomously index your spoken language, hot-swap the specific LoRA adapters onto the GPU, extract raw text, and beam it to a W&B Weave 14-Billion parameter LLM to provide error corrections and fluent English translations.
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Input your Speech here")
            routing_mode = gr.Radio(choices=["Autonomous", "mr", "gu"], value="Autonomous", label="Intelligence Routing Mode")
            submit_btn = gr.Button("🧠 Process Audio", variant="primary")
            
        with gr.Column():
            lang_output = gr.Textbox(label="1. Final Language Routed", placeholder="Waiting for Speech...")
            raw_output = gr.Textbox(label="2. Raw STT Output (Local GPU)", placeholder="Transcription will appear here...")
            agent_output = gr.Textbox(label="3. Cognitive Translation (Weave + Qwen3-14B)", lines=4, placeholder="LLM logic will load here...")

    # Bind the logic securely to just the button, giving the browser time to buffer the audio!
    submit_btn.click(
        fn=process_audio_ui,
        inputs=[audio_input, routing_mode],
        outputs=[lang_output, raw_output, agent_output]
    )
    
    gr.Markdown("*Powered by MBZUAI SLURM Cluster & Weights & Biases Inference.*")

if __name__ == "__main__":
    # share=True ensures you get a public URL like https://xyz.gradio.live to access it securely!
    logger.info("Spinning up Public UI Server...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
