"""Utility functions for training and evaluation."""

import numpy as np
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from loguru import logger


def compute_wer_cer_metrics(
    pred,
    processor,
    wer,
    cer,
    normalizer,
):
    """Compute WER and CER metrics."""
    
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
        
    label_ids = pred.label_ids
    
    # Replace -100 with pad_token_id explicitly to safely mask padding
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    pred_str = [normalizer(text) for text in pred_str]
    label_str = [normalizer(text) for text in label_str]
    
    wer_score = wer.compute(predictions=pred_str, references=label_str)
    cer_score = cer.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer_score, "cer": cer_score}


class ModelCard:
    """Simple model card wrapper with save capability."""
    
    def __init__(self, content: str):
        self.content = content
    
    def save(self, path: str):
        """Save model card to file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.content)
        logger.info(f"Model card saved to {path}")


def create_model_card(
    model_id: str,
    dataset_id: str,
    language: str,
    baseline_eval: dict,
    ft_eval: dict,
) -> ModelCard:
    """Create model card for HuggingFace Hub."""
    
    baseline_wer = baseline_eval.get('eval_wer', 0)
    baseline_cer = baseline_eval.get('eval_cer', 0)
    ft_wer = ft_eval.get('eval_wer', 0)
    ft_cer = ft_eval.get('eval_cer', 0)
    
    improvement_wer = baseline_wer - ft_wer
    improvement_cer = baseline_cer - ft_cer
    
    # Avoid division by zero
    relative_wer = (improvement_wer / baseline_wer * 100) if baseline_wer else 0
    relative_cer = (improvement_cer / baseline_cer * 100) if baseline_cer else 0
    
    card_content = f"""
# {language.capitalize()} Whisper Model (LoRA Fine-tuned)

Fine-tuned Whisper model for {language} speech recognition using LoRA adaptation.

## Model Details

- **Base Model**: {model_id}
- **Language**: {language}
- **Training Data**: {dataset_id}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: PyTorch + Hugging Face Transformers

## Performance

### Baseline (Pre-trained Whisper)
- WER: {baseline_wer:.2f}%
- CER: {baseline_cer:.2f}%

### Fine-tuned (LoRA)
- WER: {ft_wer:.2f}%
- CER: {ft_cer:.2f}%

### Improvement
- WER: {improvement_wer:.2f}% absolute ({relative_wer:.1f}% relative)
- CER: {improvement_cer:.2f}% absolute ({relative_cer:.1f}% relative)

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

- Trained only on {language} data
- Best performance on Common Voice audio
- May struggle with accented speech outside training distribution

---

*Created as part of MBZUAI Speech Processing course project*
"""
    
    return ModelCard(card_content)


def get_hf_username() -> str:
    """Get HuggingFace username."""
    try:
        from huggingface_hub import whoami
        info = whoami()
        return info["name"]
    except Exception as e:
        logger.warning(f"Could not get HF username: {e}")
        return "anonymous"
