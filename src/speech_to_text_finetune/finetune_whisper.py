"""LoRA-enabled Whisper fine-tuning script."""

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from functools import partial
import evaluate
from loguru import logger

import sys
import os

# Support both direct execution and module execution
if __name__ == "__main__" or __package__ is None:
    # Add project root to path for direct execution
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.speech_to_text_finetune.config import load_config
    from src.speech_to_text_finetune.data_process import (
        DataCollatorSpeechSeq2SeqWithPadding,
        load_dataset_from_dataset_id,
        load_dataset_from_local,
        process_dataset,
        load_subset_of_dataset,
    )
    from src.speech_to_text_finetune.utils import compute_wer_cer_metrics, create_model_card, get_hf_username
else:
    from .config import load_config
    from .data_process import (
        DataCollatorSpeechSeq2SeqWithPadding,
        load_dataset_from_dataset_id,
        load_dataset_from_local,
        process_dataset,
        load_subset_of_dataset,
    )
    from .utils import compute_wer_cer_metrics, create_model_card, get_hf_username


try:
    from peft import get_peft_model, LoraConfig as PEFTLoRAConfig
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    logger.warning("PEFT not installed. LoRA will not be available.")


def run_finetuning(config_path: str = "config.yaml"):
    """Fine-tune Whisper with LoRA."""
    cfg = load_config(config_path)
    language_id = TO_LANGUAGE_CODE.get(cfg.language.lower())
    
    if not language_id:
        raise ValueError(f"Language {cfg.language} not supported")
    
    # Setup
    local_output_dir = f"./artifacts/{cfg.repo_name}"
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Using device: {device}")
    
    # Load model & processor
    processor = WhisperProcessor.from_pretrained(
        cfg.model_id, language=cfg.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)
    model.config.use_cache = False
    model.generate = partial(
        model.generate, language=cfg.language.lower(), task="transcribe", use_cache=True
    )
    
    # Apply LoRA if configured
    if cfg.lora_config.use_lora and HAS_PEFT:
        logger.info("Applying LoRA adaptation...")
        lora_cfg = PEFTLoRAConfig(
            r=cfg.lora_config.lora_rank,
            lora_alpha=cfg.lora_config.lora_alpha,
            target_modules=cfg.lora_config.target_modules,
            lora_dropout=cfg.lora_config.lora_dropout,
            bias=cfg.lora_config.bias
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    
    # Data loading — prefer local dataset if specified
    if cfg.local_dataset_path:
        logger.info(f"Loading local dataset from: {cfg.local_dataset_path}")
        dataset, proc_path = load_dataset_from_local(cfg.local_dataset_path)
    else:
        dataset, proc_path = load_dataset_from_dataset_id(cfg.dataset_id, language_id)
    dataset["train"] = load_subset_of_dataset(dataset["train"], cfg.n_train_samples)
    dataset["test"] = load_subset_of_dataset(dataset["test"], cfg.n_test_samples)
    
    # Process dataset
    dataset = process_dataset(
        dataset,
        processor,
        batch_size=cfg.training_hp.per_device_train_batch_size,
        proc_dataset_path=proc_path
    )
    
    # Training
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    training_args = Seq2SeqTrainingArguments(
        output_dir=local_output_dir,
        report_to=["wandb"],
        run_name=f"{cfg.language}-whisper-lora",
        **cfg.training_hp.model_dump()
    )
    
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=partial(
            compute_wer_cer_metrics,
            processor=processor,
            wer=wer,
            cer=cer,
            normalizer=BasicTextNormalizer(),
        ),
        processing_class=processor.feature_extractor,
    )
    
    processor.save_pretrained(training_args.output_dir)
    
    # Baseline evaluation
    logger.info("Evaluating baseline model...")
    baseline_eval = trainer.evaluate()
    logger.info(f"Baseline WER: {baseline_eval['eval_wer']:.2f}%")
    
    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Evaluate
    logger.info("Evaluating fine-tuned model...")
    eval_results = trainer.evaluate()
    logger.info(f"Fine-tuned WER: {eval_results['eval_wer']:.2f}%")
    
    # Save results
    model_card = create_model_card(
        model_id=cfg.model_id,
        dataset_id=cfg.dataset_id,
        language=cfg.language,
        baseline_eval=baseline_eval,
        ft_eval=eval_results,
    )
    model_card.save(f"{local_output_dir}/README.md")
    
    logger.info(f"✅ Training complete! Model saved to {local_output_dir}")
    return baseline_eval, eval_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA")
    parser.add_argument("--config", type=str, default="example_configs/marathi/config_lora_gpu.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    run_finetuning(args.config)
