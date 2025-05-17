import torch


def text_to_token_ids(text, tokenizer):
    """Encode text to a 1×T tensor of token IDs."""
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # (1, T)


def token_ids_to_text(token_ids, tokenizer):
    """Decode a 1×T or T tensor of token IDs back to text."""
    flat = token_ids.squeeze(0)  # remove batch dimension if present
    return tokenizer.decode(flat.tolist())
