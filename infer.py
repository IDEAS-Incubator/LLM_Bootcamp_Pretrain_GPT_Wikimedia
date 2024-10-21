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


def load_model_and_infer(gpt_config, model_path, input_text, device,max_gen=20):
    # Load the saved model
    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.to(device)
    model.eval()

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Prepare the input
    encoded_input = text_to_token_ids(input_text, tokenizer).to(device)

    # Generate text
    with torch.no_grad():
        generated_ids = generate_text_simple(
            model=model, idx=encoded_input, max_new_tokens=max_gen, context_size=model.pos_emb.weight.shape[0]
        )

    # Decode the output back to text
    generated_text = token_ids_to_text(generated_ids, tokenizer)
    return generated_text


if __name__ == "__main__":
        # Define argument parser
    import argparse
    parser = argparse.ArgumentParser(description="Inference for GPT Model")
    parser.add_argument('--input', type=str, required=True, help='Input text for inference')

    args = parser.parse_args()
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model.pth"  # Path to the saved model
    # input_text = "who invented wiki"
    input_text = args.input  # Take input from command line

    # Perform inference
    generated_text = load_model_and_infer(GPT_CONFIG_124M, model_path, input_text, device)
    print("Generated Text:", generated_text)
