import os
import torch
import tiktoken
from transformer import GPTModel, generate_text_simple
from encode_decode import text_to_token_ids, token_ids_to_text


def load_model_and_infer(
    gpt_config,
    model_path,
    input_text,
    device,
    max_gen=20,
    temperature=1.0,
    stream=False,
):
    # --- Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = GPTModel(gpt_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --- Tokenize input
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_input = text_to_token_ids(input_text, tokenizer).to(device)

    # --- Generate tokens
    result = generate_text_simple(
        model=model,
        idx=encoded_input,
        max_new_tokens=max_gen,
        context_size=model.pos_emb.weight.shape[0],
        temperature=temperature,
        stream=stream,
        tokenizer=tokenizer,
    )

    if stream:
        # Print each decoded token as it comes
        print(input_text, end="", flush=True)
        for token_str in result:  # result is a generator
            print(token_str, end="", flush=True)
        print()  # Final newline
        return None
    else:
        # result is a tensor of token IDs
        return token_ids_to_text(result, tokenizer)


if __name__ == "__main__":
    import argparse
    from config import MODEL_CONFIGS

    parser = argparse.ArgumentParser(description="Inference for GPT Model")
    parser.add_argument("--input", type=str, required=True, help="Input text")
    parser.add_argument(
        "--model_config", type=str, default="gpt2_original", help="Key from config.py"
    )
    parser.add_argument("--max_gen", type=int, default=50, help="Tokens to generate")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--stream", action="store_true", help="Stream output tokens")

    args = parser.parse_args()
    if args.model_config not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model_config '{args.model_config}'")

    GPT_CONFIG = MODEL_CONFIGS[args.model_config]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # map config → paths
    model_names = {
        "gpt2_original": [
            "gpt2_original_Foundation better_quality_2025-05-16_05-39-43/final_model_2025-05-16_05-39-43.pth"
        ],
        # "small": ["small_fast_2025-05-17_13-31-59/final_model_2025-05-17_13-31-59.pth"],
    }

    for model_rel_path in model_names[args.model_config]:
        full_path = os.path.join("models", model_rel_path)
        print("=" * 50)
        print(f"🔍 Responding Model Name: {model_rel_path}")
        if args.stream:
            load_model_and_infer(
                GPT_CONFIG,
                full_path,
                args.input,
                device,
                max_gen=args.max_gen,
                temperature=args.temperature,
                stream=True,
            )
        else:
            out = load_model_and_infer(
                GPT_CONFIG,
                full_path,
                args.input,
                device,
                max_gen=args.max_gen,
                temperature=args.temperature,
                stream=False,
            )
            print("Generated Text:", out)
