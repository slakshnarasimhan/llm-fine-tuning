from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

base_model = "NousResearch/Llama-2-7b-chat-hf"
adapter_dir = "Llama-2-7b-chat-finetune"
out_dir = "llama2-7b-merged-aligned"

# load tokenizer first to get authoritative vocab size
tok = AutoTokenizer.from_pretrained(base_model, use_fast=False)
vocab_n = len(tok)

base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

# align model embeddings/config to tokenizer size
model.resize_token_embeddings(vocab_n)
model.config.vocab_size = vocab_n

os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)

print("tokenizer_len:", vocab_n)
print("embed_rows:", model.get_input_embeddings().weight.shape[0])
print("saved:", out_dir)

