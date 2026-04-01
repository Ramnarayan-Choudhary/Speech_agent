"""Download FLEURS evaluation dataset."""

import argparse
from datasets import load_dataset
from pathlib import Path
from loguru import logger


def download_fleurs(language_id: str = "hi_in", output_dir: str = "."):
    """Download FLEURS dataset for evaluation."""
    
    logger.info(f"Downloading FLEURS for {language_id}...")
    
    dataset = load_dataset(
        "google/fleurs",
        language_id,
        cache_dir=output_dir
    )
    
    logger.info(f"Downloaded FLEURS splits: {list(dataset.keys())}")
    for split, data in dataset.items():
        logger.info(f"  {split}: {len(data)} samples")
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="hi_in", help="Language ID (e.g., hi_in, mr_in)")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    download_fleurs(args.language, args.output)
