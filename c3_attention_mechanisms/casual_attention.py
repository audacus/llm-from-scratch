import torch

from data.attention import inputs
from utils.attention import SelfAttentionV2, CasualAttention, MultiHeadAttentionWrapper, MultiHeadAttention

d_in = inputs.shape[1]
d_out = 2

# Compute attention weights.
torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores: torch.Tensor = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print("Attention weights:\n", attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("\nDiagonal mask:\n", mask_simple)

# Apply mask to attention weights.
masked_simple = attn_weights * mask_simple
print("\nMasked simple:\n", masked_simple)

# Normalize with the sum of the row.
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("\nMasked simple normalized:\n", masked_simple_norm)

# More efficiently mask the attention weights with `-inf` (minus infinity) values.
# `e^-inf` approaches zero.
# First fill with ones, then replace with `-inf`.
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("\nMasked scores:\n", masked)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print("\nAttention weights:\n", attn_weights)

# Masking additional attention weights with dropout.
torch.manual_seed(123)
# 50% dropout rate.
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
# The remaining weights are scaled up by the dropout factor, e.g. 0.5, to maintain overall balance of the weights.
# 1 * 0.5 = 2
print("\nDropout example:\n", dropout(example))

# Apply to attention weights.
torch.manual_seed(123)
print("\nDropped-out weights:\n", dropout(attn_weights))

# Create batch by duplicating inputs.
batch = torch.stack((inputs, inputs), dim=0)
print("\nBatch shape:", batch.shape)

# Casual attention class usage.
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("\n`context_vecs.shape`:", context_vecs.shape)

# Multi-head attention wrapper.
torch.manual_seed(123)
# Number of tokens.
context_length = batch.shape[1]
d_in, d_out = 3, 2
# Exercise 3.2: resulting context vectors with only 2 dimensions.
# d_out = 1

mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2,
)
context_vecs = mha(batch)
print("\n`context_vecs`:\n", context_vecs)
print("\n`context_vecs.shape`:", context_vecs.shape)

# Illustrate batched matrix multiplication.
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])
# Batched matrix multiplication between the tensor itself and a view of the tensor with transposed the last two dimensions `num_tokens` and `head_dim`,
print(a @ a.transpose(2, 3))

# Compute the matrix multiplication for each head separately.
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("\nFirst head:\n", first_res)
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)

# Use the multi-head attention class.
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(
    d_in, d_out, context_length, 0.0, num_heads=2,
)
context_vecs = mha(batch)
print("\n`context_vecs`:\n", context_vecs)
print("\n`context_vecs.shape`:", context_vecs.shape)

# Exercise 3.3: Initializing GPT-2 size attention modules.
context_length = 1024
d_in, d_out = 768, 768
num_heads = 12
mha = MultiHeadAttention(
    d_in, d_out, context_length, 0.0, num_heads,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("\nParameter count:", count_parameters(mha))
