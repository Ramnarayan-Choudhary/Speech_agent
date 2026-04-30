#!/usr/bin/env python3
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_FILE = PROJECT_ROOT / "paper" / "main.tex"
FULL_EVAL = PROJECT_ROOT / "eval_results" / "full_pipeline_report.json"
TTS_EVAL = PROJECT_ROOT / "eval_results" / "tts_eval_report.json"

def main():
    if not FULL_EVAL.exists() or not TTS_EVAL.exists():
        print("Error: Missing eval reports.")
        return

    with open(FULL_EVAL) as f:
        full_data = json.load(f)
    
    with open(TTS_EVAL) as f:
        tts_data = json.load(f)

    with open(PAPER_FILE) as f:
        content = f.read()

    # 1. Update STT Table (Table 1)
    # \textbf{Lang} & \textbf{WER} & \textbf{CER} & \textbf{Lat.} & \textbf{Model} \\
    # English  & \best{25.6} & \best{7.7}  & 1.4\,s & Base \\
    # Marathi  & 72.1 & 21.1 & 3.8\,s & +LoRA(mr) \\
    # Gujarati & 75.0 & 33.3 & 6.9\,s & +LoRA(gu) \\
    
    en_stt = full_data["stt_accuracy"]["en_us"]
    mr_stt = full_data["stt_accuracy"]["mr_in"]
    gu_stt = full_data["stt_accuracy"]["gu_in"]
    
    # Simple regex replace for STT values
    def replace_stt_row(lang, wer, cer, lat):
        nonlocal content
        # Finds a row starting with the lang name and updates the next 3 columns
        pattern = re.compile(rf"({lang}\s*&\s*)(.*?)\s*&\s*(.*?)\s*&\s*(.*?)\s*(&\s*.*?\\\\)")
        # Make best if English
        if lang == "English":
            wer_str = rf"\best{{{wer*100:.1f}}}"
            cer_str = rf"\best{{{cer*100:.1f}}}"
        else:
            wer_str = f"{wer*100:.1f}"
            cer_str = f"{cer*100:.1f}"
        
        replacement = rf"\g<1>{wer_str} & {cer_str} & {lat:.1f}\,s \g<5>"
        content = pattern.sub(replacement, content)

    replace_stt_row("English", en_stt["wer"], en_stt["cer"], en_stt["avg_latency_s"])
    replace_stt_row("Marathi", mr_stt["wer"], mr_stt["cer"], mr_stt["avg_latency_s"])
    replace_stt_row("Gujarati", gu_stt["wer"], gu_stt["cer"], gu_stt["avg_latency_s"])

    # 2. Update Routing Table
    routing = full_data["routing_accuracy"]["per_lang"]
    en_rout = routing["en_us"]["accuracy"]
    mr_rout = routing["mr_in"]["accuracy"]
    gu_rout = routing["gu_in"]["accuracy"]
    overall_rout = full_data["routing_accuracy"]["overall"]
    
    def replace_rout_row(lang, acc, total):
        nonlocal content
        pattern = re.compile(rf"({lang}\s*&\s*)(.*?)\s*&\s*(.*?)\s*(&\s*.*?\\\\)")
        if lang == "English":
            acc_str = rf"\best{{{acc*100:.0f}\%}}"
        else:
            acc_str = f"{acc*100:.0f}\%"
        correct = int(acc * total)
        replacement = rf"\g<1>{acc_str} & {correct}/{total} \g<4>"
        content = pattern.sub(replacement, content)

    # Assuming 20 samples per language
    samples = 20
    replace_rout_row("English", en_rout, samples)
    replace_rout_row("Marathi", mr_rout, samples)
    replace_rout_row("Gujarati", gu_rout, samples)
    
    content = re.sub(r"(Overall\s*&\s*).*?(\s*&.*?\\\\)", rf"\g<1>{overall_rout*100:.0f}\% & {int(overall_rout*samples*3)}/{samples*3}\g<2>", content)
    
    # 3. Update BLEU
    mr_bleu = full_data["translation_quality"].get("mr_in", {}).get("bleu", 0)
    gu_bleu = full_data["translation_quality"].get("gu_in", {}).get("bleu", 0)
    
    content = re.sub(r"(Marathi \$\\to\$ English\s*&\s*).*?(\s*&.*?\\\\)", rf"\g<1>\\best{{{mr_bleu:.2f}}}\g<2>", content)
    content = re.sub(r"(Gujarati \$\\to\$ English\s*&\s*).*?(\s*&.*?\\\\)", rf"\g<1>{gu_bleu:.2f}\g<2>", content)

    # 4. Update TTS Evaluation Table
    tts_details = tts_data["details"]
    en_tts = next(d for d in tts_details if d["language"] == "en")
    mr_tts = next(d for d in tts_details if d["language"] == "mr")
    gu_tts = next(d for d in tts_details if d["language"] == "gu")
    
    def replace_tts_row(lang, wer, rtf):
        nonlocal content
        pattern = re.compile(rf"({lang}\s*&\s*)(.*?)\s*&\s*(.*?)\s*(&\s*.*?\\\\)")
        replacement = rf"\g<1>{wer*100:.1f}\% & {rtf:.2f} \g<4>"
        content = pattern.sub(replacement, content)

    replace_tts_row("English", en_tts["wer"], en_tts["rtf"])
    replace_tts_row("Marathi", mr_tts["wer"], mr_tts["rtf"])
    replace_tts_row("Gujarati", gu_tts["wer"], gu_tts["rtf"])

    with open(PAPER_FILE, "w") as f:
        f.write(content)
        
    print("Updated paper metrics successfully!")

if __name__ == "__main__":
    main()
