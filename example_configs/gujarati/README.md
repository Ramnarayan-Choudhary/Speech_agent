# Gujarati Configuration Guide

## Overview
Configurations for fine-tuning Whisper models on Gujarati (ગુજરાતી) language data from Common Voice 17.0.

## Files

### `config_lora_cpu.yaml`
- **Model**: Whisper-tiny (39M params)
- **Environment**: Local CPU or free Colab tier
- **Training Time**: ~2-4 hours
- **Dataset Size**: 500 samples (for quick iteration)
- **Batch Size**: 2 (limited by memory)
- **Use Case**: Prototyping, debugging

### `config_lora_gpu.yaml`
- **Model**: Whisper-small (244M params)
- **Environment**: Colab Pro (V100/A100) or MBZUAI GPU
- **Training Time**: 5-8 hours (V100), 2-3 hours (A100)
- **Dataset Size**: All available (~10k samples)
- **Batch Size**: 16
- **Use Case**: Production training

## LoRA Parameters Explanation

```yaml
lora_rank: 8              # Low-rank (8) reduces params by 90%
lora_alpha: 16            # Scaling factor (typically 2x rank)
lora_dropout: 0.05        # Dropout in LoRA layers
target_modules:           # Modules to adapt (q_proj, v_proj)
  - "q_proj"              # Query projection
  - "v_proj"              # Value projection
```

## Key Differences

| Parameter | CPU | GPU |
|-----------|-----|-----|
| Model Size | tiny (39M) | small (244M) |
| Batch Size | 2 | 16 |
| Samples | 500 | all |
| FP16 | No | Yes |
|Max Steps | 100 | 5000 |
| Eval Steps | 50 | 500 |

## Expected Results

### CPU Config
- **Training Time**: 2-4 hours
- **WER**: ~20-25% (high due to small dataset)
- **Final Model Size**: ~20MB (LoRA adapter)

### GPU Config
- **Training Time**: 5-8 hours (V100), 2-3 hours (A100)
- **WER**: ~16-18% (good performance)
- **Final Model Size**: ~20MB (LoRA adapter)

## Usage

```bash
# CPU training
python src/speech_to_text_finetune/finetune_whisper.py \
    --config example_configs/gujarati/config_lora_cpu.yaml

# GPU training (Colab)
# Use notebook: notebooks/colab_indic_asr_training.ipynb
```

## Notes

- LoRA adapters are ~20MB (vs 1.5GB for full model)
- Inference requires base model + adapter
- Merging adapter with base model creates full model
- All configs use language="gujarati" for Whisper setup
