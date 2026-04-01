"""Download Common Voice datasets for Indic languages."""

import argparse
from datasets import load_dataset
from pathlib import Path
from loguru import logger


def download_cv_dataset(language: str, output_dir: str, sample_size: int = None):
    """Download Common Voice dataset for a specific language."""
    
    logger.info(f"Downloading Common Voice for {language}...")
    
    # Map language to code
    language_code_map = {
        "marathi": "mr",
        "gujarati": "gu",
        "hindi": "hi",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "bengali": "bn",
    }
    
    lang_code = language_code_map.get(language.lower(), language.lower())
    
    # Download
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        lang_code,
        split="train",
        cache_dir=output_dir
    )
    
    logger.info(f"Downloaded {len(dataset)} samples")
    
    if sample_size:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        logger.info(f"Using {len(dataset)} samples")
    
    # Save info
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "info.txt", "w") as f:
        f.write(f"Language: {language}\n")
        f.write(f"Language Code: {lang_code}\n")
        f.write(f"Total Samples: {len(dataset)}\n")
        f.write(f"Dataset ID: mozilla-foundation/common_voice_17_0\n")
    
    logger.info(f"Dataset info saved to {output_path}/info.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="Language to download")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size")
    
    args = parser.parse_args()
    
    download_cv_dataset(args.language, args.output, args.sample_size)
