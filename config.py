# Different model configurations for  comparison
MODEL_CONFIGS = {
    "small": {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 128,  # Shorter context length for faster training
        "emb_dim": 256,  # Smaller embedding dimension
        "n_heads": 4,  # Fewer attention heads
        "n_layers": 4,  # Fewer layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-key-value bias
    },
    "medium": {
        "vocab_size": 50257,
        "context_length": 256,  # Medium context length
        "emb_dim": 512,
        "n_heads": 8,
        "n_layers": 6,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    # Add more model configs as needed
}

# Different training settings for faster/slower training comparisons
TRAINING_SETTINGS = {
    "fast": {
        "learning_rate": 1e-3,  # Higher learning rate for faster convergence
        "num_epochs": 2,  # Fewer epochs
        "batch_size": 4,  # Larger batch size if memory allows
        "weight_decay": 0.1,
    },
    "slow": {
        "learning_rate": 5e-4,
        "num_epochs": 4,
        "batch_size": 2,
        "weight_decay": 0.01,
    },
    # Add more training settings as needed
}

#  Folder CONFIG
DATAFOLDER = "dataset"
MODEL_DIR = "models"
