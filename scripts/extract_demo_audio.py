import os
import soundfile as sf
import argparse
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="gujarati")
    args = parser.parse_args()
    
    dataset_path = f"/home/ramnarayan.ramniwas/MS_projects/Speech_agent/data/processed/{args.lang}"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    ds = load_from_disk(dataset_path)
    sample = ds["test"][0] # Get the first test sample
    
    out_name = f"demo_{args.lang}.wav"
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    
    sf.write(out_name, audio_array, sampling_rate)
    
    print(f"Extracted {out_name} !")
    print(f"Ground Truth Transcription: {sample['sentence']}")

if __name__ == "__main__":
    main()
