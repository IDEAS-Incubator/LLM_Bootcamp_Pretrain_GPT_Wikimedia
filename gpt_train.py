import matplotlib.pyplot as plt
import os
import torch
import tiktoken


# Import from local files
from transformer import GPTModel, create_dataloader_v1, generate_text_simple
from encode_decode import text_to_token_ids, token_ids_to_text
from gpt_dataset import get_data
from loguru import logger
import time
from datetime import datetime


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
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=1.0,
            stream=False,
            tokenizer=tokenizer,
        )

        if isinstance(token_ids, torch.Tensor):  # Check if we got back a tensor
            decoded_text = token_ids_to_text(token_ids, tokenizer)
            logger.debug(decoded_text.replace("\n", " "))
        else:
            logger.warning(
                "Expected tensor from generate_text_simple, but got generator or None."
            )

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
    checkpoint_dir="models/model_checkpoints",  # New parameter for checkpoint directory
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_checkpoint_dir = os.path.join(checkpoint_dir, timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

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

        # Save model checkpoint after each epoch with timestamp
        epoch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_path = os.path.join(
            run_checkpoint_dir, f"model_epoch_{epoch+1}_{epoch_timestamp}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_losses[-1] if train_losses else None,
                "val_loss": val_losses[-1] if val_losses else None,
                "tokens_seen": tokens_seen,
                "timestamp": epoch_timestamp,
            },
            checkpoint_path,
        )
        logger.info(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")

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


def main(
    gpt_config,
    settings,
    filename="dataset/wiki_1K_Lines.txt",
    config_name=None,
    setting_name=None,
):
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

    # Create checkpoint directory based on model config and settings
    checkpoint_dir = os.path.join(
        f"{MODEL_DIR}/model_checkpoints", f"{config_name}_{setting_name}"
    )

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
        checkpoint_dir=checkpoint_dir,  # Pass the checkpoint directory
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # Start timing
    start_time = time.time()

    from config import (
        MODEL_CONFIGS,
        TRAINING_SETTINGS,
        MODEL_DIR,
        DATA_SOURCE,
    )  # model and training setting

    # Create the Model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    for config_name, GPT_CONFIG in MODEL_CONFIGS.items():
        for setting_name, TRAIN_SETTINGS in TRAINING_SETTINGS.items():
            for datapath in DATA_SOURCE:
                # Create timestamp for this training run
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Create a unique run directory
                run_dir = os.path.join(
                    MODEL_DIR, f"{config_name}_{setting_name}_{timestamp}"
                )
                os.makedirs(run_dir, exist_ok=True)

                # Set up logging for this run
                log_file = os.path.join(run_dir, f"training_{timestamp}.log")
                logger.add(log_file, rotation="500 MB")

                logger.info(
                    f"\n🔧 Training Model: {config_name}, Setting: {setting_name}, Data: {datapath}"
                )

                ###########################
                # Initiate training
                ###########################

                train_losses, val_losses, tokens_seen, model = main(
                    GPT_CONFIG, TRAIN_SETTINGS, datapath, config_name, setting_name
                )

                ###########################
                # After training
                ###########################

                # Plot results
                epochs_tensor = torch.linspace(
                    0, TRAIN_SETTINGS["num_epochs"], len(train_losses)
                )

                # Save plot in the run directory
                plot_path = os.path.join(run_dir, f"loss_plot_{timestamp}.pdf")
                plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
                plt.title(f"{config_name} + {setting_name} - {datapath}")
                plt.savefig(plot_path)
                plt.clf()

                # Save final model in the run directory
                model_path = os.path.join(run_dir, f"final_model_{timestamp}.pth")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config_name": config_name,
                        "setting_name": setting_name,
                        "datapath": datapath,
                        "final_train_loss": train_losses[-1] if train_losses else None,
                        "final_val_loss": val_losses[-1] if val_losses else None,
                        "total_tokens_seen": tokens_seen[-1] if tokens_seen else None,
                        "timestamp": timestamp,
                    },
                    model_path,
                )
                logger.info(f"Saved final model to {model_path}")

                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f"Training completed in: {elapsed_time / 60:.2f} minutes")
