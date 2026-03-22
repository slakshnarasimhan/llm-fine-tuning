# LLM Fine-Tuning to Phone-Ready GGUF

This repo provides a step-by-step pipeline to:

1. generate a teacher dataset,
2. fine-tune a student LLM with LoRA,
3. merge LoRA into a full Hugging Face model,
4. convert to GGUF,
5. quantize for edge/mobile-friendly inference,
6. test with `llama.cpp`.

The intended execution order is numeric scripts `1` to `5`, with `fine_tune_llama2.py` run after step `1` to produce the LoRA adapter used by step `2`.

## Prerequisites

## 1) Python environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

- macOS / Apple Silicon:

```bash
pip install -r requirements-mac.txt
```

- CUDA GPU:

```bash
pip install -r requirements-cuda.txt
```

Install OpenAI SDK (required by `1.generate-teacher-response.py`):

```bash
pip install openai
```

## 2) Environment variables

Set your OpenAI API key for teacher-data generation:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Optional but recommended for Hugging Face model downloads:

```bash
export HF_TOKEN="your_huggingface_token"
```

## 3) `llama.cpp` checkout and build

Steps `3`, `4`, and `5` assume a local folder named `llama.cpp` at repo root with built binaries and conversion tools.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build -j
```

Expected paths after build:

- `llama.cpp/convert_hf_to_gguf.py`
- `llama.cpp/build/bin/llama-quantize`
- `llama.cpp/build/bin/llama-cli`

## Run the full pipeline (in order)

Run all commands from repository root.

## Step 1 - Generate teacher dataset

```bash
python 1.generate-teacher-response.py
```

Output:

- `teacher_dataset_250.jsonl`

## Step 2 - Fine-tune the student model (LoRA adapter)

Use the dataset from step 1:

```bash
python fine_tune_llama2.py \
  --mode train \
  --dataset-jsonl teacher_dataset_250.jsonl
```

Expected output directory (consumed by step 3):

- `llama2-7b-chat-finetune`

Note: on non-CUDA machines, `fine_tune_llama2.py` auto-switches to TinyLlama defaults for feasibility.

## Step 3 - Merge LoRA adapter into base model

```bash
python 2.merge_lora.py
```

This script expects:

- base model: `NousResearch/Llama-2-7b-chat-hf`
- adapter dir: `llama2-7b-chat-finetune`

Output:

- `llama2-7b-merged/`

## Step 4 - Convert merged HF model to GGUF (f16)

```bash
bash 3.generate-gguf.sh
```

Input:

- `llama2-7b-merged/`

Output:

- `llama2-7b-merged-f16.gguf`

## Step 5 - Quantize GGUF for edge/mobile usage

```bash
bash 4.quantize.sh
```

Input:

- `llama2-7b-merged-f16.gguf`

Output:

- `llama2-7b-merged-q2_k.gguf`

## Step 6 - Test the quantized model

```bash
bash 5.test.sh
```

This runs `llama-cli` with:

- model: `llama2-7b-merged-q2_k.gguf`
- context length: `1024`
- output tokens: `128`
- CPU execution (`-ngl 0`)

## Quick sanity checklist

Before running step `n`, verify step `n-1` output exists:

- before step 2: `teacher_dataset_250.jsonl`
- before step 3: `llama2-7b-chat-finetune/`
- before step 4: `llama2-7b-merged/`
- before step 5: `llama2-7b-merged-f16.gguf`
- before step 6: `llama2-7b-merged-q2_k.gguf`

## Notes

- Use normal ASCII quotes in zsh commands (`"` or `'`), not smart quotes.
- If model download/auth errors occur, check `OPENAI_API_KEY` and `HF_TOKEN`.
- The current quantization script uses `Q2_K` (very small, lower quality). For better quality on stronger phones, try a higher-bit quantization in `4.quantize.sh`.
