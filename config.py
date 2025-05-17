# Different model configurations for  comparison
MODEL_CONFIGS = {
    "gpt2_original": {
        # Original GPT-2 (124M parameters) configuration
        "vocab_size": 50257,  # GPT-2 vocabulary size
        "context_length": 1024,  # Original GPT-2 context window
        "emb_dim": 768,  # Embedding dimension for GPT-2 small
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of transformer layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias enabled
    },
    # "small": {
    #     "vocab_size": 50257,  # Vocabulary size
    #     "context_length": 128,  # Shorter context length for faster training
    #     "emb_dim": 256,  # Smaller embedding dimension
    #     "n_heads": 4,  # Fewer attention heads
    #     "n_layers": 4,  # Fewer layers
    #     "drop_rate": 0.1,  # Dropout rate
    #     "qkv_bias": False,  # Query-key-value bias
    # },
    # "medium": {
    #     "vocab_size": 50257,
    #     "context_length": 256,  # Medium context length
    #     "emb_dim": 512,
    #     "n_heads": 8,
    #     "n_layers": 6,
    #     "drop_rate": 0.1,
    #     "qkv_bias": True,
    # },
    # # Add more model configs as needed
}

# Different training settings for faster/slower training comparisons
TRAINING_SETTINGS = {
    # "foundation": {
    #     "learning_rate": 3e-4,
    #     "num_epochs": 10,
    #     "batch_size": 4,
    #     "weight_decay": 0.01,
    #     "warmup_steps": 1000,
    #     "gradient_accumulation_steps": 4,
    #     "max_grad_norm": 1.0,
    # },
    # "Foundation better_quality": {
    #     "learning_rate": 2e-4,  # Slightly lower for more stable training
    #     "num_epochs": 50,  # Train for more epochs to improve generalization
    #     "batch_size": 4,  # Reduce if GPU memory is limited (increase if possible)
    #     "weight_decay": 0.01,
    #     "warmup_steps": 2000,  # Longer warmup for better convergence
    #     "gradient_accumulation_steps": 8,  # Effective batch size = batch_size * grad_accum_steps
    #     "max_grad_norm": 1.0,
    # },
    "fast": {
        "learning_rate": 1e-3,  # Higher learning rate for faster convergence
        "num_epochs": 2,  # Fewer epochs
        "batch_size": 4,  # Larger batch size if memory allows
        "weight_decay": 0.1,
    },
    # "slow": {
    #     "learning_rate": 5e-4,
    #     "num_epochs": 4,
    #     "batch_size": 2,
    #     "weight_decay": 0.01,
    # },
    # Add more training settings as needed
}

#  Folder CONFIG
DATAFOLDER = "dataset"
MODEL_DIR = "models"


# listing various data sources to train LLM Model
DATA_SOURCE = [
    f"{DATAFOLDER}/wiki_1K_Lines.txt",
    # f"{DATAFOLDER}/wiki_1M.txt",
    # f"{DATAFOLDER}/wiki_10M.txt",
    # f"{DATAFOLDER}/wikipedia_data.txt",  # 20 Gb Data set
]
