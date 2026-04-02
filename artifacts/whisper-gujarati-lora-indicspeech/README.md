
# Gujarati Whisper Model (LoRA Fine-tuned)

Fine-tuned Whisper model for gujarati speech recognition using LoRA adaptation.

## Model Details

- **Base Model**: openai/whisper-small
- **Language**: gujarati
- **Training Data**: None
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: PyTorch + Hugging Face Transformers

## Performance

### Baseline (Pre-trained Whisper)
- WER: 1.22%
- CER: 0.84%

### Fine-tuned (LoRA)
- WER: 0.47%
- CER: 0.25%

### Improvement
- WER: 0.75% absolute (61.7% relative)
- CER: 0.59% absolute (69.8% relative)

## Usage

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="<model_path>")
result = pipe("audio.wav")
print(result["text"])
```

## Training Details

- **Framework**: Seq2Seq Trainer
- **Optimization**: LoRA with rank=8, alpha=16
- **Memory**: ~600MB (vs 2.4GB for full fine-tuning)
- **Training Time**: See implementation logs

## Limitations

- Trained only on gujarati data
- Best performance on Common Voice audio
- May struggle with accented speech outside training distribution

---

*Created as part of MBZUAI Speech Processing course project*
