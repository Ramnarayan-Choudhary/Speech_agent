# Deployment Guide

1. Convert model to GGUF (for whisper.cpp or edge deployment):
   ```bash
python scripts/convert_to_gguf.py --model checkpoints/latest --output models/whisper-indic.gguf
```
2. Deploy via Hugging Face:
   ```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import upload_folder
upload_folder(
    folder_path="models/whisper-mr-finetuned",
    path_in_repo=".",
    repo_id="<user>/whisper-mr-finetuned"
)
PY
```
3. Use inference pipeline:
   ```bash
python -m speech_to_text_finetune.inference --model_path models/whisper-mr-finetuned
```
