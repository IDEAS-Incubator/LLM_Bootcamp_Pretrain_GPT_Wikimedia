import matplotlib.pyplot as plt
import os
import torch
import tiktoken


# Import from local files
from transformer import GPTModel, create_dataloader_v1, generate_text_simple
from gpt_dataset import get_data
from loguru import logger
import time

logger.add("logs/file_1.log", rotation="500 MB")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0

    # Handle empty data_loader
    if isinstance(data_loader, list) and len(data_loader) == 0:
        return float("nan")
    elif hasattr(data_loader, "__len__") and len(data_loader) == 0:
        return float("nan")

    # Determine number of batches to process
    if hasattr(data_loader, "__len__"):
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
    else:
        # If data_loader doesn't have a length, use the provided num_batches or default to 1
        if num_batches is None:
            num_batches = 1

    # Handle different types of data_loader
    if isinstance(data_loader, list):
        # Process list of pre-loaded batches
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            # Check if batch is already unpacked
            if isinstance(batch, tuple) and len(batch) == 2:
                input_batch, target_batch = batch
            else:
                input_batch, target_batch = batch

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    else:
        # Process DataLoader
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        logger.debug(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Check if train_loader is a DataLoader or a list of batches
        if hasattr(train_loader, "__iter__") and not hasattr(train_loader, "__len__"):
            # It's an iterable but not a DataLoader, use as is
            batch_iterator = train_loader
        elif isinstance(train_loader, list):
            # It's a list of batches
            batch_iterator = train_loader
        else:
            # It's a DataLoader, use as before
            batch_iterator = train_loader

        for batch in batch_iterator:
            # Check if we're dealing with a pre-unpacked batch (list) or a DataLoader batch (tuple)
            if isinstance(batch, tuple) and len(batch) == 2:
                input_batch, target_batch = batch
            else:
                # If it's already unpacked, use directly
                input_batch, target_batch = batch

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings, filename="dataset/wiki_1K_Lines.txt"):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Download data if necessary
    ##############################

    # Use a much smaller dataset for faster training
    text_data = get_data(
        file_path=filename,
        stream_mode=True,
        max_lines=10000,  # Limit to 1000 lines
        max_articles=5000,  # Limit to 50 articles if downloading
    )

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(
        device
    )  # no assignment model = model.to(device) necessary for nn.Module classes

    # Use a faster optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        betas=(0.9, 0.95),  # Slightly modified betas for faster convergence
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Create a single dataloader with more efficient settings
    full_dataloader = create_dataloader_v1(
        text_data,
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"]
        // 2,  # Use smaller stride for more efficient training
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    # Calculate the split point (90% for training, 10% for validation)
    total_batches = len(full_dataloader)
    train_batches = int(0.9 * total_batches)

    # Preload batches to avoid generator overhead
    logger.info(f"Preloading {train_batches} training batches")
    train_loader = list(full_dataloader)[:train_batches]
    # Small validation set
    val_samples = min(
        10, total_batches - train_batches
    )  # Just use 10 batches for validation
    val_loader = (
        list(full_dataloader)[train_batches : train_batches + val_samples]
        if train_batches < total_batches
        else []
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    # More frequent evaluation for faster feedback
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=10,  # Evaluate more frequently
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # Start timing
    start_time = time.time()

    from config import (
        MODEL_CONFIGS,
        TRAINING_SETTINGS,
        DATAFOLDER,
        MODEL_DIR,
    )  # model and training setting

    # listing various data sources to train LLM Model
    datasources = [
        f"{DATAFOLDER}/wiki_1K_Lines.txt",
        f"{DATAFOLDER}/wiki_1M.txt",
        f"{DATAFOLDER}/wiki_10M.txt",
        f"{DATAFOLDER}/wikipedia_data.txt",  # 20 Gb Data set
    ]

    # Create the Model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    for config_name, GPT_CONFIG in MODEL_CONFIGS.items():
        for setting_name, TRAIN_SETTINGS in TRAINING_SETTINGS.items():
            for datapath in datasources:

                print(
                    f"\n🔧 Training Model: {config_name}, Setting: {setting_name}, Data: {datapath}"
                )

                ###########################
                # Initiate training
                ###########################

                train_losses, val_losses, tokens_seen, model = main(
                    GPT_CONFIG, TRAIN_SETTINGS, datapath
                )

                ###########################
                # After training
                ###########################

                # Plot results
                epochs_tensor = torch.linspace(
                    0, TRAIN_SETTINGS["num_epochs"], len(train_losses)
                )
                plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
                plt.title(f"{config_name} + {setting_name} - {datapath}")
                plt.savefig(f"result/loss_{config_name}_{setting_name}_{datapath}.pdf")
                plt.clf()

                # Save and load model
                model_path = (
                    f"{MODEL_DIR}/model_{config_name}_{setting_name}_{datapath}.pth"
                )
                torch.save(model.state_dict(), model_path)
                model = GPTModel(GPT_CONFIG)
                # model.load_state_dict(torch.load("model.pth"), weights_only=True)

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Training completed in: {elapsed_time / 60:.2f} minutes")
