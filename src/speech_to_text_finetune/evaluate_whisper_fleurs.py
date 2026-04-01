"""Evaluation script using FLEURS benchmark."""

import torch
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from loguru import logger


def evaluate_on_fleurs(
    model_id: str,
    language_id: str,
    output_file: str = "eval_results.json"
):
    """Evaluate model on FLEURS benchmark."""
    
    logger.info(f"Loading model: {model_id}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        processor=processor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Load FLEURS
    logger.info(f"Loading FLEURS for {language_id}...")
    fleurs = load_dataset("google/fleurs", language_id)
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []
    
    logger.info("Running inference...")
    for sample in tqdm(fleurs["test"]):
        pred = pipe(sample["audio"])["text"]
        ref = sample["transcription"]
        
        predictions.append(pred)
        references.append(ref)
    
    wer_score = wer_metric.compute(predictions=predictions, references=references)
    cer_score = cer_metric.compute(predictions=predictions, references=references)
    
    results = {
        "model_id": model_id,
        "language": language_id,
        "WER": wer_score,
        "CER": cer_score,
        "num_samples": len(predictions)
    }
    
    logger.info(f"\n=== Evaluation Results ===")
    logger.info(f"WER: {wer_score:.2f}%")
    logger.info(f"CER: {cer_score:.2f}%")
    logger.info(f"Samples: {len(predictions)}")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    import sys
    
    model_id = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-small"
    language = sys.argv[2] if len(sys.argv) > 2 else "hi_in"
    
    evaluate_on_fleurs(model_id, language)
