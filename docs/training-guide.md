# Training Guide

1. Prepare datasets using `notebooks/colab_data_preparation.ipynb`.
2. Load language config from `example_configs/{language}/config_lora_gpu.yaml`.
3. Train using: 
   ```bash
python -m speech_to_text_finetune.finetune_whisper --config example_configs/marathi/config_lora_gpu.yaml
```
4. Monitor logs in `logs/` and checkpoints in `checkpoints/`.
