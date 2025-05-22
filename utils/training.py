import torch
from tiktoken import Encoding
from torch import Tensor, nn, optim
from torch import device
from torch.utils.data import DataLoader

from utils.model import GPTModel, generate_text_simple
from utils.text_tokenizer import text_to_token_ids, token_ids_to_text


def calc_loss_batch(
        input_batch: Tensor,
        target_batch: Tensor,
        model: nn.Module,
        # Allow transferring the data on a GPU.
        device: device,
) -> Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits: Tensor = model(input_batch)
    loss = nn.functional.cross_entropy(
        input=logits.flatten(0, 1),
        target=target_batch.flatten(),
    )
    return loss


def calc_loss_loader(
        data_loader: DataLoader,
        model: nn.Module,
        device: device,
        num_batches: int | None = None,
) -> float:
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # Iterate over all batches if no number is specified.
        num_batches = len(data_loader)
    else:
        # Ensure the number of batches does not exceed the number of batches in the data loader.
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        input_batch: Tensor
        target_batch: Tensor

        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # Sum loss for each batch.
            total_loss += loss.item()
        else:
            break

    # Return average loss over all batches.
    return total_loss / num_batches


def train_model_simple(
        model: GPTModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: device,
        num_epochs: int,
        eval_freq: int,
        eval_iter: int,
        start_context: str,
        tokenizer: Encoding,
) -> tuple[list[float], list[float], list[int]]:
    # Initialize lists to track losses and tokens seen.
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            input_batch: Tensor
            target_batch: Tensor

            # Reset loss gradients from previous batch iteration.
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
            )
            # Calculate loss gradients.
            loss.backward()
            # Updates model weights using loss gradients.
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step.
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss: {train_loss:.3f}, "
                    f"Val loss: {val_loss:.3f}"
                )

        # Print a sample text after each epoch.
        generate_and_print_sample(
            model, tokenizer, device, start_context,
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: device,
        eval_iter: int,
) -> tuple[float, float]:
    # Disable dropout during evaluation for stable, reproducible results.
    model.eval()
    # Disable gradient tracking, which is not required during evaluation to reduce the computational overhead.
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
        )

    model.train()

    return train_loss, val_loss


def generate_and_print_sample(
        model: GPTModel,
        tokenizer: Encoding,
        device: device,
        start_context: str,
):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    # Compact print format.
    print(decoded_text.replace("\n", " "))
    model.train()
