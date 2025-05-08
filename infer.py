import torch
from transformer import GPTModel, generate_text_simple
import tiktoken


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def load_model_and_infer(gpt_config, model_path, input_text, device, max_gen=20):
    # Load the saved model
    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Prepare the input
    encoded_input = text_to_token_ids(input_text, tokenizer).to(device)

    # Generate text
    with torch.no_grad():
        generated_ids = generate_text_simple(
            model=model,
            idx=encoded_input,
            max_new_tokens=max_gen,
            context_size=model.pos_emb.weight.shape[0],
        )

    # Decode the output back to text
    generated_text = token_ids_to_text(generated_ids, tokenizer)
    return generated_text


if __name__ == "__main__":
    # Define argument parser
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Inference for GPT Model")
    parser.add_argument(
        "--input", type=str, required=True, help="Input text for inference"
    )
    parser.add_argument(
        "--model_config", type=str, default="small", help="Model size: small or medium"
    )

    args = parser.parse_args()

    # Define different model configurations
    from config import MODEL_CONFIGS

    # Get model config
    if args.model_config not in MODEL_CONFIGS:
        raise ValueError(
            f"Invalid model_config '{args.model_config}'. Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    GPT_CONFIG = MODEL_CONFIGS[args.model_config]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_text = args.input  # Take input from command line

    # List of different models trained with same config but different settings
    model_names = {
        "small": [
            "model_small_slow_wiki_1K_Lines.txt.pth",
            "model_small_fast_wiki_1K_Lines.txt.pth",
        ],
        "medium": [
            "model_medium_slow_wiki_1K_Lines.txt.pth",
            "model_medium_fast_wiki_1K_Lines.txt.pth",
        ],
    }

    # Loop through all models and run inference
    for model_name in model_names[args.model_config]:
        model_path = f"models/{model_name}"
        print("=" * 50)
        print(f"🔍 Responding Model Name: {model_name}")
        generated_text = load_model_and_infer(
            GPT_CONFIG, model_path, input_text, device
        )
        print("Generated Text:", generated_text)
