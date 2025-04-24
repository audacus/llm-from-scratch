import torch

input_ids = torch.tensor([2, 3, 5, 1])

# BPE tokenizer: 50'257 words
vocab_size = 6
# GPT-3: 12'288 dimensions
output_dim = 3

# Set the random seed manually for reproducibility purposes.
torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=output_dim,
)
# Print the embedding layer's underlying weight matrix.
print("\nEmbedding weights:\n", embedding_layer.weight)

# Apply it to a token ID -> obtain embedding vector:
print("\nEmbedding for single ID `[3]`:\n", embedding_layer(torch.tensor([3])))

# Apply it to all token IDs.
print("\nEmbeddings for all IDs `[2, 3, 5, 1]`:\n", embedding_layer(input_ids))
