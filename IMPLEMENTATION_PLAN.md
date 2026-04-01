# 🗓️ Speech Agent: 4-Week Implementation Plan

**Course**: Intro to Speech Processing (MBZUAI)  
**Grade Weight**: 60% of final score  
**Timeline**: 4 weeks (20 business days)  
**Goal**: Production-ready Indic multilingual ASR with LoRA fine-tuning + agentic learning

---

## 📊 Project Phases Overview

| Phase | Week | Focus | Deliverable |
|-------|------|-------|-------------|
| **Phase 0** | Setup | Environment & Dependencies | Dev environment ready |
| **Phase 1** | Data | Preparation & Exploration | Balanced datasets (500 samples/lang) |
| **Phase 2** | Training | LoRA Fine-tuning | 3 trained models (mr, gu, hi) |
| **Phase 3** | Evaluation | Benchmarking & Metrics | WER <15%, CER <5% |
| **Phase 4** | Deployment | Inference & Demo | Gradio app + Colab notebook |

---

## ⏰ Week-by-Week Breakdown

### **WEEK 1: Environment Setup & Data Preparation**

#### **Day 1: Project Initialization & Environment Setup**
**Goal**: Get local dev environment fully operational

**Tasks**:
- [ ] Clone repo: `git clone <repo>`
- [ ] Create virtual environment: `python -m venv venv && source venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt` (~10 min)
- [ ] Verify PyTorch GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Setup Hugging Face authentication: `huggingface-cli login`
- [ ] Create directories: `mkdir -p data/raw/{marathi,gujarati,hindi} data/processed/{marathi,gujarati,hindi}`

**Deliverable**: Dev environment ready, GPU confirmed working
**Time Estimate**: 1-2 hours

---

#### **Day 2: Dataset Download - Marathi (Primary)**
**Goal**: Download & explore Marathi Common Voice dataset

**Tasks**:
- [ ] Download Common Voice 17.0 (Marathi): 
  ```bash
  python scripts/download_cv_datasets.py --language marathi --output data/raw/marathi --sample-size 500
  ```
  **Expected**: ~500 samples (~30 mins, ~2GB)

- [ ] Explore dataset structure:
  ```bash
  python -c "from datasets import load_dataset; ds = load_dataset('mozilla-foundation/common_voice_17_0', 'mr', split='train'); print(ds[0].keys()); print(f'Total samples: {len(ds)}')"
  ```

- [ ] Check audio quality: Listen to 5-10 samples manually

**Deliverable**: Marathi dataset ready in `data/raw/marathi/`
**Time Estimate**: 1-2 hours (mostly download time)

---

#### **Day 3: Dataset Download - Gujarati & Hindi**
**Goal**: Download remaining language datasets

**Tasks**:
- [ ] Download Gujarati:
  ```bash
  python scripts/download_cv_datasets.py --language gujarati --output data/raw/gujarati --sample-size 500
  ```

- [ ] Download Hindi:
  ```bash
  python scripts/download_cv_datasets.py --language hindi --output data/raw/hindi --sample-size 500
  ```

- [ ] Verify all datasets loaded successfully

**Deliverable**: All 3 language datasets ready
**Time Estimate**: 2-3 hours

---

#### **Day 4: Data Exploration & Analysis**
**Goal**: Understand dataset characteristics

**Tasks**:
- [ ] Run Jupyter notebook for data exploration:
  ```bash
  jupyter notebook notebooks/colab_data_preparation.ipynb
  ```

- [ ] Statistics per language:
  - Total samples
  - Average audio duration (seconds)
  - Text length distribution
  - Gender/age distribution (if available)

- [ ] Create dataset summary report:
  ```
  Marathi: 500 samples, avg duration 5.2s, text length 32.5 words
  Gujarati: 500 samples, avg duration 4.8s, text length 28.1 words
  Hindi: 500 samples, avg duration 5.5s, text length 35.2 words
  ```

**Deliverable**: Data analysis report + sample statistics
**Time Estimate**: 2-3 hours

---

#### **Day 5: Configuration Validation & Test Run**
**Goal**: Validate entire pipeline with minimum data

**Tasks**:
- [ ] Load config file:
  ```bash
  python -c "from src.speech_to_text_finetune.config import load_config; cfg = load_config('example_configs/marathi/config_lora_gpu.yaml'); print(cfg)"
  ```

- [ ] Test data loading pipeline:
  ```bash
  python -c "from src.speech_to_text_finetune.data_process import load_dataset_from_dataset_id; ds, path = load_dataset_from_dataset_id('mozilla-foundation/common_voice_17_0', 'mr_IN')"
  ```

- [ ] Run unit tests:
  ```bash
  pytest tests/unit/ -v
  ```

- [ ] Do a dry-run of training with 10 samples:
  ```bash
  # Modify config: n_train_samples=10, max_steps=5, eval_steps=1
  python src/speech_to_text_finetune/finetune_whisper.py --config example_configs/marathi/config_lora_gpu.yaml
  ```

**Deliverable**: All tests passing, 1 training step completed
**Time Estimate**: 1-2 hours

**END OF WEEK 1**: ✅ Data ready, pipeline validated, environment operational

---

### **WEEK 2: Marathi LoRA Fine-tuning (Primary)**

#### **Day 6: Marathi Model Training - Hyperparameter Tuning**
**Goal**: Train Whisper-small + LoRA on Marathi data

**Setup**:
- GPU: Google Colab A100 or MBZUAI Lab
- Model: openai/whisper-small (244M params)
- Method: LoRA (rank=8, alpha=16)
- Expected runtime: 2-4 hours per epoch

**Tasks**:
- [ ] Open Colab notebook: `notebooks/colab_indic_asr_training.ipynb`
- [ ] Mount Google Drive for checkpoint storage
- [ ] Modify config (if needed):
  ```yaml
  model_id: openai/whisper-small
  language: marathi
  lora_config:
    use_lora: true
    lora_rank: 8
  training_hp:
    num_train_epochs: 3
    max_steps: 5000
    per_device_train_batch_size: 16
  ```

- [ ] Train for 3 epochs:
  ```bash
  python src/speech_to_text_finetune/finetune_whisper.py --config example_configs/marathi/config_lora_gpu.yaml
  ```

- [ ] Monitor:
  - Training loss should decrease
  - Validation WER should improve
  - Memory usage should stay ~0.6GB (LoRA advantage)

**Expected Results**:
- Training loss: ~2.5 → 1.2 over 3 epochs
- Validation WER: Baseline 45% → ~35%

**Deliverable**: Trained model checkpoint in `checkpoints/marathi_lora/`
**Time Estimate**: 4-6 hours (mostly GPU compute)

---

#### **Day 7: Marathi Model Evaluation - FLEURS Benchmark**
**Goal**: Evaluate model on FLEURS multilingual test set

**Tasks**:
- [ ] Run evaluation notebook:
  ```bash
  jupyter notebook notebooks/colab_evaluation.ipynb
  ```

- [ ] Execute evaluation script:
  ```bash
  python src/speech_to_text_finetune/evaluate_whisper_fleurs.py \
    --model_id checkpoints/marathi_lora \
    --language mr_IN \
    --output results/marathi_metrics.json
  ```

- [ ] Collect metrics:
  - WER (Word Error Rate)
  - CER (Character Error Rate)
  - Inference time per sample
  - Model size

**Expected Results**:
- WER: 12-18% on FLEURS test set
- CER: 4-6%
- Inference time: 1-2s per audiosample

**Deliverable**: Evaluation report with metrics + sample transcriptions
**Time Estimate**: 2-3 hours

---

#### **Day 8: Gujarati Model Training**
**Goal**: Repeat training process for Gujarati language

**Tasks**:
- [ ] Prepare config: `example_configs/gujarati/config_lora_gpu.yaml`
- [ ] Train model:
  ```bash
  python src/speech_to_text_finetune/finetune_whisper.py --config example_configs/gujarati/config_lora_gpu.yaml
  ```

- [ ] Training loop: Same as Marathi (3 epochs, ~4-6 hours)

**Deliverable**: Gujarati trained model in `checkpoints/gujarati_lora/`
**Time Estimate**: 5-7 hours

---

#### **Day 9: Gujarati Model Evaluation**
**Goal**: Evaluate Gujarati model

**Tasks**:
- [ ] Run evaluation:
  ```bash
  python src/speech_to_text_finetune/evaluate_whisper_fleurs.py \
    --model_id checkpoints/gujarati_lora \
    --language gu_IN
  ```

**Deliverable**: Gujarati metrics + comparison table (Marathi vs Gujarati)
**Time Estimate**: 2-3 hours

---

#### **Day 10: Hindi Model Training & Evaluation**
**Goal**: Train and evaluate Hindi model

**Tasks**:
- [ ] Train: `python src/speech_to_text_finetune/finetune_whisper.py --config example_configs/hindi/config_lora_gpu.yaml`
- [ ] Evaluate: `python src/speech_to_text_finetune/evaluate_whisper_fleurs.py --model_id checkpoints/hindi_lora`

**Deliverable**: All 3 models trained + evaluated with metrics table
**Time Estimate**: 7-9 hours

**END OF WEEK 2**: ✅ All 3 models trained & evaluated on FLEURS

---

### **WEEK 3: Optimization & Agentic Learning**

#### **Day 11: Performance Analysis & Baseline Comparison**
**Goal**: Analyze results vs baseline Whisper

**Tasks**:
- [ ] Compare LoRA fine-tuned vs baseline (no LoRA):
  ```
  Model          | LoRA | WER (%) | CER (%) | Size  | Memory | Speed
  Marathi        | No   | 18.5    | 6.2     | 244MB | 2.4GB  | 1x
  Marathi        | Yes  | 15.3    | 4.8     | 244MB | 0.6GB  | 1.3x
  ```

- [ ] Document improvements: ~15-20% WER reduction, 60% memory savings

**Deliverable**: Performance report + comparison table
**Time Estimate**: 1-2 hours

---

#### **Day 12: Agentic Learning Loop - Setup**
**Goal**: Implement feedback collection & analysis

**Tasks**:
- [ ] Create feedback collection mechanism:
  - User input interface (transcription feedback)
  - Confidence score tracking
  - Error categorization (acoustic vs linguistic)

- [ ] Setup performance monitoring:
  - WER tracking over time
  - Confusion matrix per language pair
  - Common error patterns

**Deliverable**: Feedback collection module + monitoring dashboard
**Time Estimate**: 3-4 hours

---

#### **Day 13: Agentic Learning Loop - Implementation**
**Goal**: Auto-retraining trigger logic

**Tasks**:
- [ ] Implement auto-retrain logic:
  ```python
  if avg_wer_last_100 > baseline_wer * 1.1:
      # Collect hard examples
      # Retrain with new data
      # Evaluate improvement
  ```

- [ ] Create data collection pipeline for hard examples
- [ ] Setup incremental training script

**Deliverable**: Auto-retraining mechanism + documentation
**Time Estimate**: 4-5 hours

---

#### **Day 14: Model Quantization & Optimization**
**Goal**: Prepare models for edge deployment

**Tasks**:
- [ ] Convert to GGUF format:
  ```bash
  python scripts/convert_to_gguf.py \
    --model checkpoints/marathi_lora \
    --output models/marathi.gguf
  ```

- [ ] Benchmark quantized vs non-quantized:
  - Model size
  - Inference speed
  - Accuracy loss

**Deliverable**: Quantized models + performance comparison
**Time Estimate**: 2-3 hours

---

#### **Day 15: Model Curation & Best Checkpoints**
**Goal**: Select and document best models

**Tasks**:
- [ ] Analyze all checkpoints per language
- [ ] Select best based on FLEURS WER
- [ ] Create model cards with full metadata:
  ```markdown
  # Whisper-Marathi-LoRA-Small
  - Base: openai/whisper-small
  - Training Data: Common Voice 17.0 (500 samples)
  - LoRA Config: rank=8, alpha=16
  - FLEURS WER: 14.2%
  - Parameters: 244M (only ~10M trainable with LoRA)
  - Training Time: 6 hours (A100)
  ```

**Deliverable**: Model cards + selection report
**Time Estimate**: 1-2 hours

**END OF WEEK 3**: ✅ Models optimized, agentic loop integrated

---

### **WEEK 4: Deployment & Demo**

#### **Day 16: Inference Pipeline Setup**
**Goal**: Create production inference APIs

**Tasks**:
- [ ] Implement REST API:
  ```python
  from fastapi import FastAPI
  from src.speech_to_text_finetune.inference import WhisperInference
  
  app = FastAPI()
  inference = WhisperInference("checkpoints/marathi_lora")
  
  @app.post("/transcribe")
  async def transcribe(audio: UploadFile):
      result = inference.transcribe(audio.file)
      return {"text": result["text"]}
  ```

- [ ] Setup Docker container for API
- [ ] Test with sample audiofiles

**Deliverable**: Production-ready API + Docker image
**Time Estimate**: 3-4 hours

---

#### **Day 17: Gradio Web Interface**
**Goal**: Create user-friendly demo app

**Tasks**:
- [ ] Build Gradio interface:
  ```python
  import gradio as gr
  from src.speech_to_text_finetune.inference import WhisperInference
  
  demo = gr.Interface(
      fn=inference.transcribe,
      inputs=gr.Audio(type="filepath"),
      outputs="text",
      examples=[...],
      title="Indic Speech-to-Text"
  )
  demo.launch()
  ```

- [ ] Add language selector
- [ ] Display confidence scores
- [ ] Test locally: `python demo.py`

**Deliverable**: Working Gradio app
**Time Estimate**: 2-3 hours

---

#### **Day 18: Hugging Face Hub Deployment**
**Goal**: Push models to HF Hub for sharing

**Tasks**:
- [ ] Create HF Hub organization/repos
- [ ] Upload models:
  ```bash
  huggingface-cli repo create whisper-marathi-lora
  python -c "from huggingface_hub import upload_folder; upload_folder('checkpoints/marathi_lora', 'whisper-marathi-lora')"
  ```

- [ ] Add model cards (README.md)
- [ ] Add usage examples
- [ ] Test loading from Hub:
  ```python
  from transformers import pipeline
  pipe = pipeline("automatic-speech-recognition", model="<user>/whisper-marathi-lora")
  ```

**Deliverable**: Models on HF Hub + usage documentation
**Time Estimate**: 1-2 hours

---

#### **Day 19: Documentation & Reporting**
**Goal**: Create comprehensive project documentation

**Tasks**:
- [ ] Update README with results:
  ```markdown
  ## Results
  
  | Model | FLEURS WER | CER | Training Time | Memory |
  |-------|-----------|-----|---------------|--------|
  | Marathi | 14.2% | 4.5% | 6h | 0.6GB |
  | Gujarati | 16.1% | 5.2% | 6h | 0.6GB |
  | Hindi | 13.8% | 4.1% | 6h | 0.6GB |
  ```

- [ ] Create deployment guide
- [ ] Document agentic learning approach
- [ ] Add troubleshooting section

- [ ] Create presentation slides:
  - System architecture diagram
  - Performance metrics
  - Deployment options
  - Future improvements

**Deliverable**: Complete documentation + presentation
**Time Estimate**: 3-4 hours

---

#### **Day 20: Final Testing & Submission**
**Goal**: Quality assurance & project completion

**Tasks**:
- [ ] Run full test suite:
  ```bash
  pytest tests/ -v --cov=src/
  ```

- [ ] Verify all links in README/GitHub
- [ ] Test all 3 models end-to-end
- [ ] Clean up artifacts (remove large model files from local repo)
- [ ] Create final git commit:
  ```bash
  git add -A
  git commit -m "Final submission: Indic ASR with LoRA finetuning + agentic learning"
  git push origin main
  ```

- [ ] Create release tag:
  ```bash
  git tag -a v1.0 -m "Course project submission"
  git push origin v1.0
  ```

**Deliverable**: Production-ready GitHub repo + release
**Time Estimate**: 2-3 hours

**END OF WEEK 4**: ✅ Full project complete, deployed, documented

---

## 📋 Success Criteria

### **Minimum Targets (60% grade)**
- ✅ All 3 Indic languages supported (Marathi, Gujarati, Hindi)
- ✅ LoRA fine-tuning implemented (60% memory savings demonstrated)
- ✅ WER < 20% on FLEURS benchmark
- ✅ Training & inference code working end-to-end
- ✅ GitHub repository with full documentation

### **Medium Targets (80% grade)**
- ✅ WER < 15% on all languages
- ✅ Agentic learning loop partially implemented
- ✅ Deployment guide + local inference working
- ✅ Comparison vs baseline with metrics
- ✅ Model optimization (quantization) explored

### **Advanced Targets (95%+ grade)**
- ✅ WER < 12% on Marathi (state-of-the-art)
- ✅ Full agentic learning with auto-retraining
- ✅ Production API + Docker container
- ✅ Models on HF Hub with proper documentation
- ✅ Comprehensive research report on multilingual ASR

---

## 🚀 Day-by-Day Quick Reference

```
WEEK 1 (Setup & Data)
Day 1:  Environment setup
Day 2:  Marathi dataset download
Day 3:  Gujarati + Hindi download
Day 4:  Data exploration & analysis
Day 5:  Config validation & dry run

WEEK 2 (Training & Evaluation)
Day 6:  Marathi training (3 epochs)
Day 7:  Marathi evaluation (FLEURS)
Day 8:  Gujarati training
Day 9:  Gujarati evaluation
Day 10: Hindi training + evaluation

WEEK 3 (Optimization & Agenting)
Day 11: Performance analysis vs baseline
Day 12: Agentic loop setup
Day 13: Auto-retraining implementation
Day 14: Model quantization
Day 15: Best checkpoint selection

WEEK 4 (Deployment)
Day 16: FastAPI inference server
Day 17: Gradio web interface
Day 18: HF Hub deployment
Day 19: Documentation & presentation
Day 20: Final testing & submission
```

---

## 💾 Checkpoint Savepoints

Save your work at these points for easy rollback:

```bash
# After Week 1
git commit -m "Week 1: Data preparation complete"
git tag week-1-complete

# After Week 2
git commit -m "Week 2: All models trained and evaluated"
git tag week-2-complete

# After Week 3
git commit -m "Week 3: Optimization and agentic learning"
git tag week-3-complete

# Final
git commit -m "Week 4: Deployment and submission ready"
git tag v1.0-final
```

---

## 📞 Support & Debugging

**Common Issues**:

1. **CUDA Out of Memory**
   - Solution: Reduce batch_size in config (16 → 8 or 4)
   - Or: Use Whisper-tiny instead (39M params)

2. **Dataset Loading Fails**
   - Solution: Check internet connection, HF token, dataset availability
   - Fallback: Use cached version or subset

3. **Model Diverges During Training**
   - Solution: Reduce learning_rate (1e-3 → 1e-4)
   - Increase warmup_steps (500 → 1000)

4. **Slow Inference**
   - Solution: Convert to GGUF for faster inference
   - Or: Use lower precision (FP16 instead of FP32)

---

## 🎯 Final Notes

- This plan is flexible—adapt based on actual results
- If any milestone takes longer, adjust subsequent tasks
- Regular git commits ensure you don't lose work
- Keep a learning journal of key insights
- Document any custom modifications for future reference

**Good luck! 🚀**
