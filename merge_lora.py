from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "tinyllama-chat-finetune-local"

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="cpu"
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(model, adapter_path)

print("Merging adapter...")

model = model.merge_and_unload()

print("Saving merged model...")

model.save_pretrained("tinyllama-merged")
