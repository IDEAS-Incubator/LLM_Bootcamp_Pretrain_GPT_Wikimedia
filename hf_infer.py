import os
import torch
import json
import tiktoken
import importlib.util
from huggingface_hub import hf_hub_download

# === CONFIG ===
REPO_ID = "faizack/bayes_mini"

# === Step 1: Download necessary files ===
config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
model_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
modeling_path = hf_hub_download(repo_id=REPO_ID, filename="modeling_gpt2_custom.py")

# === Step 2: Dynamically import modeling_gpt2_custom.py ===
spec = importlib.util.spec_from_file_location("modeling_gpt2_custom", modeling_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GPTModel = mod.GPTModel  # Now you can use GPTModel

# === Step 3: Load config ===
with open(config_path, "r") as f:
    config = json.load(f)

model_config = {
    "vocab_size": config["vocab_size"],
    "context_length": config["n_positions"],
    "emb_dim": config["n_embd"],
    "n_heads": config["n_head"],
    "n_layers": config["n_layer"],
    "drop_rate": config["dropout"],
    "qkv_bias": config["qkv_bias"],
}

# === Step 4: Load tokenizer ===
tokenizer = tiktoken.get_encoding("gpt2")
prompt = "The rise of artificial intelligence"
input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

# === Step 5: Load model ===
model = GPTModel(model_config)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


# === Step 6: Generate ===
def generate(model, idx, max_new_tokens=50):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model_config["context_length"] :]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


output = generate(model, input_ids)
print(tokenizer.decode(output[0].tolist()))
