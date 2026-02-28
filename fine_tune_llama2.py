import inspect
import importlib.util
import json
import os
from pathlib import Path
import textwrap
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Environment setup:
# - macOS / Apple Silicon: pip install -r requirements-mac.txt
# - CUDA GPU (RunPod, etc): pip install -r requirements-cuda.txt

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "Llama-2-7b-chat-finetune"
teacher_model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Hardware detection
has_cuda = torch.cuda.is_available()
has_mps = torch.backends.mps.is_available()
device_label = "cuda" if has_cuda else "mps" if has_mps else "cpu"

# Mac/CPU-friendly defaults
if not has_cuda:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    new_model = "tinyllama-chat-finetune-local"
    print(f"Running on {device_label}. Switching to smaller model: {model_name}")

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = has_cuda

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 2

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per device for training
per_device_train_batch_size = 1 if not has_cuda else 4

# Batch size per device for evaluation
per_device_eval_batch_size = 1 if not has_cuda else 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 8 if not has_cuda else 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 1e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit" if has_cuda else "adamw_torch"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = 400 if not has_cuda else -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

# Use TensorBoard only if it is installed in the environment.
report_to = "tensorboard" if importlib.util.find_spec("tensorboard") else "none"

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 512 if not has_cuda else None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Let Transformers choose an appropriate device map
device_map = "auto"

# Generation defaults (can be overridden via CLI)
gen_max_new_tokens = 200
gen_temperature = 0.7
gen_top_p = 0.9
gen_repetition_penalty = 1.1
demo_mode = False
dataset_jsonl_path = None

def build_bnb_config():
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    local_use_4bit = use_4bit and has_cuda
    if use_4bit and not has_cuda:
        print("CUDA is not available. Disabling 4-bit quantization (use_4bit=False).")
    if local_use_4bit and not hasattr(torch.nn.Module, "set_submodule"):
        print(
            "Current torch build does not support nn.Module.set_submodule; "
            "disabling 4-bit quantization for compatibility."
        )
        print(
            f"Detected torch version: {torch.__version__}. "
            "Upgrade torch to a newer build (recommended) to re-enable 4-bit."
        )
        local_use_4bit = False

    if compute_dtype == torch.float16 and local_use_4bit and has_cuda:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    if not local_use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )


def load_tokenizer(tokenizer_source):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _has_local_tokenizer_files(model_dir):
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        return False
    required = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    return any((path / name).exists() for name in required)


def _normalize_prompt(prompt):
    # Replace smart quotes pasted from docs/slides with plain quotes.
    return prompt.replace("\u201c", '"').replace("\u201d", '"').strip()


def _extract_assistant_text(generated_text):
    marker = "[/INST]"
    if marker in generated_text:
        return generated_text.split(marker, 1)[1].strip()
    return generated_text.strip()


def _format_model_input(tokenizer_obj, prompt):
    clean_prompt = _normalize_prompt(prompt)
    if hasattr(tokenizer_obj, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": clean_prompt}]
            return tokenizer_obj.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"<s>[INST] {clean_prompt} [/INST]"


def _format_training_example(tokenizer_obj, prompt, answer):
    prompt = _normalize_prompt(prompt)
    answer = answer.strip()
    if hasattr(tokenizer_obj, "apply_chat_template"):
        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            return tokenizer_obj.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass
    return f"<s>[INST] {prompt} [/INST] {answer}"


def _prepare_training_dataset(dataset, tokenizer_obj):
    column_names = set(dataset.column_names)
    if "prompt" in column_names and "text" in column_names:
        # Distillation-style data: convert prompt+answer into single chat-formatted text.
        return dataset.map(
            lambda ex: {"text": _format_training_example(tokenizer_obj, ex["prompt"], ex["text"])},
            desc="Formatting prompt+answer training samples",
        )
    return dataset


def _generate_text(pipe, tokenizer_obj, prompt):
    model_input = _format_model_input(tokenizer_obj, prompt)
    generated = pipe(model_input)[0]["generated_text"]
    if generated.startswith(model_input):
        return generated[len(model_input):].strip()
    return _extract_assistant_text(generated)


def _build_generation_pipe(model_obj, tokenizer_obj):
    do_sample = gen_temperature > 0
    pipe_kwargs = {
        "task": "text-generation",
        "model": model_obj,
        "tokenizer": tokenizer_obj,
        "max_new_tokens": gen_max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": gen_repetition_penalty,
    }
    if do_sample:
        pipe_kwargs["temperature"] = gen_temperature
        pipe_kwargs["top_p"] = gen_top_p
    return pipeline(
        **pipe_kwargs
    )


def _wrap_text_preserving_newlines(text, width=100):
    wrapped_lines = []
    for line in text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.append(textwrap.fill(line, width=width))
    return "\n".join(wrapped_lines).strip()


def _print_response_block(title, text):
    print(f"\n{title}")
    print("-" * len(title))
    formatted = _wrap_text_preserving_newlines(text, width=100 if demo_mode else 110)
    print(formatted)


def load_model_and_tokenizer_for_inference(base_model_id, adapter_dir=None):
    base_kwargs = {"device_map": device_map}
    if has_cuda:
        base_kwargs["torch_dtype"] = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **base_kwargs)

    if adapter_dir and os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        if _has_local_tokenizer_files(adapter_dir):
            tokenizer = load_tokenizer(adapter_dir)
        else:
            print(f"No tokenizer files found in {adapter_dir}. Using base tokenizer: {base_model_id}")
            tokenizer = load_tokenizer(base_model_id)
    else:
        if adapter_dir:
            print(f"Adapter directory not found: {adapter_dir}. Using base model only.")
        model = base_model
        tokenizer = load_tokenizer(base_model_id)
    return model, tokenizer


def load_base_model_for_training():
    bnb_config = build_bnb_config()
    model_kwargs = {"device_map": device_map}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def build_training_arguments():
    training_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optim": optim,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "fp16": fp16,
        "bf16": bf16,
        "max_grad_norm": max_grad_norm,
        "max_steps": max_steps,
        "warmup_ratio": warmup_ratio,
        "group_by_length": group_by_length,
        "lr_scheduler_type": lr_scheduler_type,
        "dataloader_pin_memory": False,
        "report_to": report_to,
    }
    supported_args = inspect.signature(TrainingArguments.__init__).parameters
    filtered_training_kwargs = {k: v for k, v in training_kwargs.items() if k in supported_args}
    unsupported_keys = sorted(set(training_kwargs) - set(filtered_training_kwargs))
    if unsupported_keys:
        print(f"Ignoring unsupported TrainingArguments keys: {', '.join(unsupported_keys)}")
    return TrainingArguments(**filtered_training_kwargs)


def build_trainer(model, tokenizer, dataset):
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_arguments = build_training_arguments()
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": training_arguments,
    }
    sft_signature = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in sft_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in sft_signature:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in sft_signature:
        trainer_kwargs["dataset_text_field"] = "text"
    if "formatting_func" in sft_signature:
        trainer_kwargs["formatting_func"] = lambda example: example["text"]
    if "max_seq_length" in sft_signature:
        trainer_kwargs["max_seq_length"] = max_seq_length
    if "packing" in sft_signature:
        trainer_kwargs["packing"] = packing
    return SFTTrainer(**trainer_kwargs)


def train_and_save():
    if dataset_jsonl_path:
        dataset = load_dataset("json", data_files=dataset_jsonl_path, split="train")
        print(f"Using local JSONL dataset: {dataset_jsonl_path}")
    else:
        dataset = load_dataset(dataset_name, split="train")
    model = load_base_model_for_training()
    tokenizer = load_tokenizer(model_name)
    dataset = _prepare_training_dataset(dataset, tokenizer)
    trainer = build_trainer(model, tokenizer, dataset)
    trainer.train()
    trainer.model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)
    print(f"Saved adapter and tokenizer to: {new_model}")


def load_finetuned_for_inference():
    return load_model_and_tokenizer_for_inference(model_name, adapter_dir=new_model)


def run_prompt(prompt):
    logging.set_verbosity(logging.CRITICAL)
    model, tokenizer = load_finetuned_for_inference()
    pipe = _build_generation_pipe(model, tokenizer)
    answer = _generate_text(pipe, tokenizer, prompt)
    _print_response_block("Assistant", answer)


def run_interactive():
    logging.set_verbosity(logging.CRITICAL)
    model, tokenizer = load_finetuned_for_inference()
    pipe = _build_generation_pipe(model, tokenizer)
    print("Interactive mode. Type a prompt and press Enter. Type 'exit' or 'quit' to stop.")
    while True:
        prompt = input("\nPrompt> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("Exiting interactive mode.")
            break
        answer = _generate_text(pipe, tokenizer, prompt)
        _print_response_block("Assistant", answer)


def generate_response(model_obj, tokenizer_obj, prompt):
    pipe = _build_generation_pipe(model_obj, tokenizer_obj)
    return _generate_text(pipe, tokenizer_obj, prompt)


def teacher_generate_dataset(output_path):
    prompts = [
        "Explain large language models in simple terms.",
        "What is transfer learning and why is it useful?",
        "How does attention work in transformers?",
        "Give three practical uses of generative AI.",
        "Explain overfitting and two ways to reduce it.",
        "What is the difference between precision and recall?",
        "How do embeddings help in semantic search?",
        "When should I use LoRA fine-tuning?",
        "Explain gradient descent for beginners.",
        "What are tokenizers and why do they matter?",
    ]
    print(f"Generating teacher dataset from: {teacher_model_name}")
    teacher_model, teacher_tokenizer = load_model_and_tokenizer_for_inference(teacher_model_name)
    lines = []
    for idx, q in enumerate(prompts, start=1):
        answer = generate_response(teacher_model, teacher_tokenizer, q)
        lines.append({"prompt": q, "text": answer})
        print(f"Generated {idx}/{len(prompts)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Saved teacher-generated dataset to: {output_path}")
    del teacher_model
    del teacher_tokenizer
    gc.collect()


def evaluate_models(prompt):
    print("\n=== Three-way comparison ===")
    print(f"Prompt: {prompt}")
    print(f"Teacher model: {teacher_model_name}")
    print(f"Student base model: {model_name}")
    print(f"Student adapter dir: {new_model}\n")

    teacher_m, teacher_t = load_model_and_tokenizer_for_inference(teacher_model_name)
    teacher_out = generate_response(teacher_m, teacher_t, prompt)
    print("[Teacher output]")
    print(teacher_out)
    del teacher_m
    del teacher_t
    gc.collect()

    base_m, base_t = load_model_and_tokenizer_for_inference(model_name)
    base_out = generate_response(base_m, base_t, prompt)
    print("\n[Student base output]")
    print(base_out)
    del base_m
    del base_t
    gc.collect()

    tuned_m, tuned_t = load_model_and_tokenizer_for_inference(model_name, adapter_dir=new_model)
    tuned_out = generate_response(tuned_m, tuned_t, prompt)
    print("\n[Student tuned output]")
    print(tuned_out)
    del tuned_m
    del tuned_t
    gc.collect()
    print("\n============================\n")


def _format_size(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _get_num_params_from_config(model_id):
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(cfg, "num_parameters"):
            num_params = cfg.num_parameters()
            if isinstance(num_params, int) and num_params > 0:
                return num_params
        return None
    except Exception as exc:
        print(f"Could not read config for {model_id}: {exc}")
        return None


def _dir_size_bytes(path_obj):
    if not path_obj.exists():
        return 0
    total = 0
    for f in path_obj.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def report_size():
    print("\n=== Model Size Report ===")
    print(f"Current base model: {model_name}")
    print(f"Fine-tuned adapter dir: {new_model}")

    base_params = _get_num_params_from_config(model_name)
    if base_params is not None:
        print(f"Base model params (from config): {base_params:,}")
    else:
        print("Base model params (from config): unavailable")

    original_reference = "NousResearch/Llama-2-7b-chat-hf"
    if model_name != original_reference:
        ref_params = _get_num_params_from_config(original_reference)
        if ref_params is not None and base_params is not None and base_params > 0:
            ratio = ref_params / base_params
            print(f"Reference model params ({original_reference}): {ref_params:,}")
            print(f"Reference/base param ratio: {ratio:.2f}x")
        elif ref_params is not None:
            print(f"Reference model params ({original_reference}): {ref_params:,}")

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_models_root = hf_home / "hub"
    cache_size = 0
    if hub_models_root.exists():
        repo_key = model_name.replace("/", "--")
        for p in hub_models_root.glob(f"models--{repo_key}*"):
            cache_size += _dir_size_bytes(p)
    if cache_size > 0:
        print(f"Base model cache size (local): {_format_size(cache_size)}")
    else:
        print("Base model cache size (local): not found in HF cache yet")

    adapter_path = Path(new_model)
    if adapter_path.exists():
        adapter_size = _dir_size_bytes(adapter_path)
        print(f"Adapter folder size (local): {_format_size(adapter_size)}")
        if cache_size > 0:
            pct = (adapter_size / cache_size) * 100
            print(f"Adapter as % of base cache size: {pct:.4f}%")
    else:
        print("Adapter folder size (local): not found")

    print("=========================\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and/or run prompt on fine-tuned model.",
        epilog=(
            "Dependency setup: use requirements-mac.txt on macOS/Apple Silicon, "
            "or requirements-cuda.txt on CUDA GPUs."
        ),
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override student base model (example: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "prompt", "all"],
        default="prompt",
        help="train: fine-tune only, prompt: inference only, all: train then prompt",
    )
    parser.add_argument(
        "--prompt",
        default="What is a large language model?",
        help="Prompt text for inference mode",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive chat loop (inference only)",
    )
    parser.add_argument(
        "--report-size",
        action="store_true",
        help="Print parameter and disk-size comparison report",
    )
    parser.add_argument(
        "--teacher-generate-dataset",
        nargs="?",
        const="teacher_sft_data.jsonl",
        default=None,
        help="Generate teacher outputs dataset JSONL (optional output path)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compare teacher vs student-base vs student-tuned on the same prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (set 0 for greedy decoding)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p value",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty to reduce repetitive outputs",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of newly generated tokens",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Use cleaner deterministic settings and tighter formatting for demos",
    )
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help="Path to local JSONL training data (supports prompt+text columns)",
    )
    parser.add_argument(
        "--max-steps-train",
        type=int,
        default=None,
        help="Override training max_steps",
    )
    parser.add_argument(
        "--learning-rate-train",
        type=float,
        default=None,
        help="Override training learning rate",
    )
    parser.add_argument(
        "--num-train-epochs-override",
        type=float,
        default=None,
        help="Override number of training epochs",
    )
    args = parser.parse_args()

    global model_name, demo_mode, dataset_jsonl_path
    global gen_temperature, gen_top_p, gen_repetition_penalty, gen_max_new_tokens
    global max_steps, learning_rate, num_train_epochs
    if args.base_model:
        model_name = args.base_model
        print(f"Using overridden base model: {model_name}")
    demo_mode = args.demo_mode

    gen_temperature = args.temperature
    gen_top_p = args.top_p
    gen_repetition_penalty = args.repetition_penalty
    gen_max_new_tokens = args.max_new_tokens
    if demo_mode:
        gen_temperature = 0.2
        gen_top_p = 0.9
        gen_repetition_penalty = 1.15
        print(
            "Demo mode enabled: temperature=0.2, top_p=0.9, "
            "repetition_penalty=1.15"
        )
    dataset_jsonl_path = args.dataset_jsonl
    if args.max_steps_train is not None:
        max_steps = args.max_steps_train
    if args.learning_rate_train is not None:
        learning_rate = args.learning_rate_train
    if args.num_train_epochs_override is not None:
        num_train_epochs = args.num_train_epochs_override

    ran_special_flow = False
    if args.report_size:
        report_size()
    if args.teacher_generate_dataset:
        teacher_generate_dataset(args.teacher_generate_dataset)
        ran_special_flow = True
    if args.evaluate:
        evaluate_models(args.prompt)
        ran_special_flow = True

    if args.mode in ("train", "all"):
        train_and_save()
    if args.interactive:
        run_interactive()
    elif args.mode in ("prompt", "all") and not (ran_special_flow and args.mode == "prompt"):
        run_prompt(args.prompt)


if __name__ == "__main__":
    main()


