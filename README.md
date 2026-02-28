# LLM Fine-Tuning Demo

This repo contains a single script, `fine_tune_llama2.py`, that supports:

- LoRA fine-tuning
- single-prompt inference
- interactive inference shell
- teacher/student comparison and dataset generation
- size reporting

## 1) Environment Setup

Use one of the dependency files:

- macOS / Apple Silicon:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-mac.txt
```

- CUDA GPU (RunPod, etc):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-cuda.txt
```

## 2) Core Ways to Run

### Train Only

```bash
python fine_tune_llama2.py --mode train
```

### Prompt Once (non-interactive)

```bash
python fine_tune_llama2.py --mode prompt --prompt "What is a large language model?"
```

### Interactive Shell

```bash
python fine_tune_llama2.py --interactive
```

Type prompts and use `exit` / `quit` to stop.

## 3) Demo-Friendly Inference

Use cleaner decoding/formatting for presentations:

```bash
python fine_tune_llama2.py --interactive --demo-mode --max-new-tokens 220
```

Manual decoding controls:

```bash
python fine_tune_llama2.py \
  --mode prompt \
  --prompt "Explain attention in transformers." \
  --temperature 0.2 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --max-new-tokens 220
```

## 4) Distillation / Local JSONL Training

If you have local JSONL with `prompt` and `text` fields:

```bash
python fine_tune_llama2.py \
  --mode train \
  --dataset-jsonl teacher_sft_data.jsonl \
  --max-steps-train 600 \
  --learning-rate-train 8e-5 \
  --num-train-epochs-override 2
```

## 5) Teacher Dataset Generation (HF teacher)

```bash
python fine_tune_llama2.py --teacher-generate-dataset
```

Optional custom path:

```bash
python fine_tune_llama2.py --teacher-generate-dataset my_teacher_data.jsonl
```

## 6) Evaluate Teacher vs Student

```bash
python fine_tune_llama2.py \
  --evaluate \
  --prompt "Should a startup use RAG or fine-tuning for customer support?"
```

## 7) Size Reporting

```bash
python fine_tune_llama2.py --report-size
```

## 8) Override Student Base Model

```bash
python fine_tune_llama2.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --interactive
```

## Notes

- On macOS, CUDA is not available; script uses MPS/CPU paths.
- For HF downloads/rate limits, set a token:

```bash
export HF_TOKEN="your_token_here"
```

- In zsh, always use normal ASCII quotes (`"` or `'`) for prompts, not smart quotes.
