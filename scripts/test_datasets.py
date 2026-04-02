"""Test which datasets are available and compatible with current setup."""
from datasets import load_dataset

# Try Google FLEURS (open, no gating)
print("=== Trying FLEURS (mr_in) ===")
try:
    ds = load_dataset("google/fleurs", "mr_in", split="train")
    n = len(ds)
    cols = ds.column_names
    text = ds[0].get("transcription", "N/A")[:80]
    print(f"FLEURS Marathi train: {n} samples")
    print(f"Columns: {cols}")
    print(f"Sample text: {text}")
except Exception as e:
    print(f"FLEURS failed: {e}")

print()

# Try Common Voice 13.0
print("=== Trying Common Voice 13.0 (mr) ===")
try:
    ds = load_dataset("mozilla-foundation/common_voice_13_0", "mr", split="train[:5]")
    n = len(ds)
    cols = ds.column_names
    print(f"CV 13.0 mr: {n} samples, columns: {cols}")
except Exception as e:
    print(f"CV 13.0 failed: {e}")

print()

# Try Common Voice 16.1
print("=== Trying Common Voice 16.1 (mr) ===")
try:
    ds = load_dataset("mozilla-foundation/common_voice_16_1", "mr", split="train[:5]")
    n = len(ds)
    cols = ds.column_names
    print(f"CV 16.1 mr: {n} samples, columns: {cols}")
except Exception as e:
    print(f"CV 16.1 failed: {e}")
