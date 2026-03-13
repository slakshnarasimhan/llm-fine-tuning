from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = "tinyllama-chat-finetune-local"
out_dir = "tinyllama-merged"

base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base, adapter_dir)
merged = model.merge_and_unload()
merged.save_pretrained(out_dir)
tok = AutoTokenizer.from_pretrained(base_model, use_fast=False)
tok.save_pretrained(out_dir)

print("Saved merged model to", out_dir)
