import os
import json
import torch
import shutil
from huggingface_hub import create_repo, upload_folder, hf_hub_download

from transformer import GPTModel
from config import MODEL_CONFIGS


# === CONFIGURATION === #
HF_USERNAME = "faizack"
MODEL_NAME = "bayes_mini"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# Path to the model file and export directory
# Adjust these paths as necessary
MODEL_FILE = r"C:\Users\Faijan\Downloads\code\LLM_Bootcamp_Pretrain_GPT_Wikimedia\models\gpt2_original_Foundation better_quality_2025-05-16_05-39-43\final_model_2025-05-16_05-39-43.pth"
EXPORT_DIR = r"C:\Users\Faijan\Downloads\gpt2-custom-minis"
SOURCE_MODEL_FILE = "transformer.py"


# === FUNCTION TO SAVE LOCALLY === #
def save_model_locally(export_dir: str, model_file: str):
    model_config = MODEL_CONFIGS["gpt2_original"]
    model = GPTModel(model_config)
    state = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    # Save config.json
    hf_config = {
        "architectures": ["GPTModel"],
        "model_type": "gpt2",
        "vocab_size": model_config["vocab_size"],
        "n_positions": model_config["context_length"],
        "n_embd": model_config["emb_dim"],
        "n_layer": model_config["n_layers"],
        "n_head": model_config["n_heads"],
        "dropout": model_config["drop_rate"],
        "qkv_bias": model_config["qkv_bias"],
    }
    with open(os.path.join(export_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=4)

    # Save model
    torch.save(model.state_dict(), os.path.join(export_dir, "pytorch_model.bin"))

    # Save model class
    shutil.copy(SOURCE_MODEL_FILE, os.path.join(export_dir, "modeling_gpt2_custom.py"))

    print(f"✅ Model + config saved locally to: {export_dir}")
    return model_config


# === FUNCTION TO SAVE tokenizer_config.json === #
def save_tokenizer_config(export_dir: str):
    tokenizer_config = {
        "tokenizer_class": "TiktokenTokenizer",
        "auto_map": {"AutoTokenizer": "tiktoken.TiktokenTokenizer"},
        "tiktoken_encoding": "gpt2",
        "add_bos_token": False,
        "add_eos_token": False,
    }
    with open(os.path.join(export_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=4)
    print("✅ tokenizer_config.json saved")


# === FUNCTION TO SAVE README.md === #
def save_readme(export_dir: str):
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            """# bayes_mini

`bayes_mini` is a custom GPT-2 (124M) language model trained from scratch on ~20 GB of English Wikipedia data.

## Architecture

- Based on GPT-2 small (124M parameters)
- 12 layers, 12 attention heads
- Hidden size: 768
- Context length: 1024
- Vocabulary size: 50257
- Dropout: 0.1

## Training Configuration

- Dataset: Cleaned English Wikipedia (~20 GB)
- Architecture: GPT-2 Small (124M parameters)
- Optimizer settings: `Foundation better_quality`
- Hardware: NVIDIA GeForce RTX 4060 (8 GB VRAM)
- Epochs: 50
- Batch size: 4 (gradient accumulation steps: 8 -> effective batch size: 32)
- Learning rate: 2e-4
- Warmup steps: 2000
- Weight decay: 0.01


## Install required packages
```bash
pip install torch transformers tiktoken huggingface_hub
```

## Example Usage

```python
import os
import torch
import json
import tiktoken
import importlib.util
from huggingface_hub import hf_hub_download

# === CONFIG ===
REPO_ID = "faizack/bayes_mini_custom"

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

```
"""
        )
    print("✅ README.md saved")


# === FUNCTION TO PUSH TO HF ===
def push_to_huggingface(repo_id: str, export_dir: str):
    create_repo(repo_id, private=False, exist_ok=True)
    upload_folder(folder_path=export_dir, repo_id=repo_id)
    print(f"✅ Model pushed to: https://huggingface.co/{repo_id}")


# === FUNCTION TO LOAD FROM HF ===
def load_from_hf(repo_id: str):
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")

    with open(config_path) as f:
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

    from modeling_gpt2_custom import GPTModel

    model = GPTModel(model_config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, tokenizer


# === MAIN ===
if __name__ == "__main__":
    config = save_model_locally(EXPORT_DIR, MODEL_FILE)
    save_tokenizer_config(EXPORT_DIR)
    save_readme(EXPORT_DIR)
    push_to_huggingface(REPO_ID, EXPORT_DIR)
