
# Marathi Whisper Model (LoRA Fine-tuned)

Fine-tuned Whisper model for marathi speech recognition using LoRA adaptation.

## Model Details

- **Base Model**: openai/whisper-small
- **Language**: marathi
- **Training Data**: None
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: PyTorch + Hugging Face Transformers

## Performance

### Baseline (Pre-trained Whisper)
- WER: 0.61%
- CER: 0.35%

### Fine-tuned (LoRA)
- WER: 0.43%
- CER: 0.22%

### Improvement
- WER: 0.18% absolute (29.5% relative)
- CER: 0.13% absolute (36.1% relative)

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

- Trained only on marathi data
- Best performance on Common Voice audio
- May struggle with accented speech outside training distribution

---

*Created as part of MBZUAI Speech Processing course project*
