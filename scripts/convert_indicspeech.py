"""Convert IndicSpeech 2022 Kaldi-format data to HuggingFace Dataset.

Usage:
    python scripts/convert_indicspeech.py --language gujarati
    python scripts/convert_indicspeech.py --language marathi
    python scripts/convert_indicspeech.py --language all
"""

import os
import sys
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Audio

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "indicspeech2022")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

LANGUAGE_MAP = {
    "gujarati": {
        "subdir": "gujarati/gujarati_dictation_speechdata_sample",
        "code": "gu",
    },
    "marathi": {
        "subdir": "marathi/marathi_dictation_speechdata_sample",
        "code": "mr",
    },
}


def parse_kaldi_text(text_path: str) -> dict:
    """Parse Kaldi text file into {utt_id: transcription}."""
    transcriptions = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcriptions[utt_id.strip()] = text.strip()
    return transcriptions


def convert_language(language: str) -> DatasetDict:
    """Convert a single language from Kaldi format to HuggingFace DatasetDict."""
    info = LANGUAGE_MAP[language]
    data_dir = os.path.join(RAW_DATA_DIR, info["subdir"])
    wav_dir = os.path.join(data_dir, "wav")
    text_path = os.path.join(data_dir, "text")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    print(f"\n{'='*60}")
    print(f"  Converting {language.upper()} IndicSpeech 2022")
    print(f"{'='*60}")

    # Parse transcriptions
    transcriptions = parse_kaldi_text(text_path)
    print(f"  Transcriptions loaded: {len(transcriptions)}")

    # Match with existing WAV files
    records = []
    missing = 0
    for utt_id, text in transcriptions.items():
        wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
        if os.path.exists(wav_path):
            records.append({
                "audio": wav_path,        # absolute path for Audio feature
                "sentence": text,
                "utt_id": utt_id,
                "language": language,
            })
        else:
            missing += 1

    if missing > 0:
        print(f"  Warning: {missing} utterances missing WAV files")
    print(f"  Valid audio-text pairs: {len(records)}")

    if len(records) == 0:
        raise ValueError(f"No valid records found for {language}")

    # Create HuggingFace Dataset
    ds = Dataset.from_dict({
        "audio": [r["audio"] for r in records],
        "sentence": [r["sentence"] for r in records],
        "utt_id": [r["utt_id"] for r in records],
        "language": [r["language"] for r in records],
    })

    # Cast audio column to Audio feature (handles loading + resampling to 16kHz)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Shuffle and split (80% train, 20% test)
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=0.2, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "test": split["test"],
    })

    # Save to disk
    output_dir = os.path.join(PROCESSED_DIR, language)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)

    print(f"  Train samples: {len(dataset_dict['train'])}")
    print(f"  Test samples:  {len(dataset_dict['test'])}")
    print(f"  Saved to:      {output_dir}")

    # Verify a sample
    sample = dataset_dict["train"][0]
    print(f"\n  Sample verification:")
    print(f"    Text:       {sample['sentence'][:60]}...")
    print(f"    Audio rate: {sample['audio']['sampling_rate']} Hz")
    print(f"    Audio len:  {len(sample['audio']['array'])} samples "
          f"({len(sample['audio']['array'])/16000:.1f}s)")

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Convert IndicSpeech 2022 to HuggingFace Dataset")
    parser.add_argument("--language", type=str, default="all",
                        choices=["gujarati", "marathi", "all"],
                        help="Language to convert")
    args = parser.parse_args()

    languages = list(LANGUAGE_MAP.keys()) if args.language == "all" else [args.language]

    results = {}
    for lang in languages:
        dd = convert_language(lang)
        results[lang] = {
            "train": len(dd["train"]),
            "test": len(dd["test"]),
        }

    print(f"\n{'='*60}")
    print("  CONVERSION SUMMARY")
    print(f"{'='*60}")
    for lang, counts in results.items():
        print(f"  {lang.capitalize():12s}: {counts['train']:4d} train + {counts['test']:3d} test")
    print(f"  Output dir:   {PROCESSED_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
