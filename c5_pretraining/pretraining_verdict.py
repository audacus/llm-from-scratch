import tiktoken
import torch

from data.config import GPT_CONFIG_124M
from data.the_verdict import get_the_verdict
from utils.dataset import create_dataloader_v1
from utils.model import GPTModel, get_device
from utils.training import calc_loss_loader, train_model_simple

config = dict(
    GPT_CONFIG_124M,
    **{
        'context_length': 256,
    }
)

model = GPTModel(config)
tokenizer = tiktoken.get_encoding("gpt2")

# Calculate the training and validation set losses on `The Verdict`.
text_data = get_the_verdict()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.9
split_idx = int(train_ratio * total_characters)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    text=train_data,
    batch_size=2,
    max_length=config["context_length"],
    stride=config["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)
val_loader = create_dataloader_v1(
    text=val_data,
    batch_size=2,
    max_length=config["context_length"],
    stride=config["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

device = get_device()
model.to(device)

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device,
    )
    var_loss = calc_loss_loader(
        val_loader, model, device,
    )

print("\nTraining loss:", train_loss)
print("Validation loss:", var_loss, "\n")

# Everything in action.
torch.manual_seed(123)
model = GPTModel(config)
model.to(device)
optimizer = torch.optim.AdamW(
    # Pass all trainable weight parameters of the model.
    params=model.parameters(),
    lr=0.0004,
    weight_decay=0.1,
)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)

