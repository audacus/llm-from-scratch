import tiktoken
import torch

from dataset import create_dataloader_v1
from the_verdict import get_the_verdict

raw_text = get_the_verdict()

# Work with more realistic numbers.
vocab_size = tiktoken.get_encoding("gpt2").max_token_value + 1
output_dim = 256

# Create embedding layer for tokens.
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    text=raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False,
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInput shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("\nToken embeddings shape:\n", token_embeddings.shape)

# Create embedding layer for positions (same dimensions as token embedding layer).
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# Placeholder vector containing a sequence of numbers 0, 1, ..., `input_length - 1`.
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("\nPos embeddings shape:\n", pos_embeddings.shape)

# Add the position embeddings to each row of the token embeddings.
input_embeddings = token_embeddings + pos_embeddings
print("\nInput embeddings shape:\n", input_embeddings.shape)
