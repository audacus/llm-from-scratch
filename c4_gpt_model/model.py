import torch

from data.config import GPT_CONFIG_124M, get_config
from data.model import batch
from utils.model import GPTModel, TransformerBlock

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters: {total_params:,}")

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

total_params_gpt2 = (
        total_params - sum(p.numel() for p in model.out_head.parameters())
)
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

# Exercise 4.1: Calculate the number of parameters in the feed forward and attention modules.
block = TransformerBlock(cfg=GPT_CONFIG_124M)
feed_forward_params = sum(
    p.numel() for p in block.ff.parameters()
)
print(f"Total number of parameters in the feed forward module: {feed_forward_params:,}")
attention_params = sum(
    p.numel() for p in block.att.parameters()
)
print(f"Total number of parameters in the attention module: {attention_params:,}")

# Calculate memory requirements.
# Calculate the total size in bytes (assuming float32, 4 bytes per parameter).
total_size_bytes = total_params * 4
# Convert to megabytes.
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

# Exercise 4.2: Initialize a `GPT-2 medium`, `GPT-2 large` and `GPT-2 XL`.
# GPT-2 medium.
gpt2_medium = GPTModel(get_config("gpt2-medium"))
total_params = sum(p.numel() for p in gpt2_medium.parameters())
print(f"\nTotal number of parameters `GPT-2 medium`: {total_params:,}")
# GPT-2 large.
gpt2_large = GPTModel(get_config("gpt2-large"))
total_params = sum(p.numel() for p in gpt2_large.parameters())
print(f"Total number of parameters `GPT-2 large`: {total_params:,}")
# GPT-2 XL.
gpt2_xl = GPTModel(get_config("gpt2-xl"))
total_params = sum(p.numel() for p in gpt2_xl.parameters())
print(f"Total number of parameters `GPT-2 XL`: {total_params:,}")
