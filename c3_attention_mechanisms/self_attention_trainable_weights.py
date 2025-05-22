import torch

from data.attention import inputs
from utils.attention import SelfAttentionV1, SelfAttentionV2

# Calculate context vector `z^2` for `x^2` only.
# The second element.
x_2 = inputs[1]
# The input embedding size -> `d_in=3`
d_in = inputs.shape[1]
# The output embedding size -> `d_out=2`
d_out = 2

# Initialize three weight matrices W_q, W_k, W_v.
# `W` corresponds to `weight parameters` -> values that are optimized during training of the neural network.
# Not to be confused with `attention weights`.
# - Weight parameters: fundamental, learned coefficients that define the neural network's connections
# - Attention weights: dynamic, context-specific values.
torch.manual_seed(123)
# Set `requires_grad=False` to prevent clutter in the outputs.
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# Compute the query, key and value vectors.
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("Query vector of `x^2`:", query_2)

# Calculate the key and value vectors for all input elements -> needed for the attention weights in respect of the `query^2` vector.
keys = inputs @ W_key
values = inputs @ W_value
print("`keys.shape`:", keys.shape)
print("`values.shape`:", values.shape)

# Calculate the attention score `w_22`.
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
# Un-normalized attention score.
print("\nAttention score `w_22`:", attn_score_22)

attn_scores_2 = query_2 @ keys.T
print("All attention scores for `q^2`:", attn_scores_2)

# Calculate attention weights through scaling the attention scores by dividing them by the square root of the square root of the embedding dimension of the keys.
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print("\nAttention weights for `q^2`:", attn_weights_2)

# Compute the context vector as weighted sum over the value vectors.
# Use the attention weights as weighting factor that weighs the respective importance of each value vector.
context_vec_2 = attn_weights_2 @ values
print("\nContext vector `z^2` for `q^2`:", context_vec_2)

# Use compact class.
# V1
torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, d_out)
print("\nContext vectors v1:\n", sa_v1(inputs))

# V2
torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
print("\nContext vectors v2:\n", sa_v2(inputs))

# Assign the weights from v2 to v1.
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
print("\nContext vectors v1 with weights from v2:\n", sa_v1(inputs))
