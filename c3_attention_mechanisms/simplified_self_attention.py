import torch

from data.attention import inputs

# Calculate dot product of the query (x^n) with every other token.
# Second token serves as query.
query = inputs[1]

# Attention scores for the second input (`inputs[1]`)
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    x_i: torch.Tensor
    attn_scores_2[i] = torch.dot(x_i, query)
print("\nAttention scores:", attn_scores_2)

# Demonstrate dot product calculation.
print("\n`torch.dot`:\n", torch.dot(inputs[0], query))
res = 0.0
for index, element in enumerate(inputs[0]):
    res += inputs[0][index] * query[index]
print("\n`res`:\n", res)

# Normalize attention scores.
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("\nAttention weights (normalized):\n", attn_weights_2_tmp)
print("\nSum:", attn_weights_2_tmp.sum())


# Naive `softmax` function implementation.
def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("\nAttention weights (naive `softmax`):\n", attn_weights_2_naive)
print("\nSum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("\nAttention weights (`torch.softmax`):\n", attn_weights_2)
print("\nSum:", attn_weights_2.sum())

# Calculate the context vector.
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("\nContext vector:\n", context_vec_2)

# Calculate all attention weights.
# Slow approach using `for` loops.
# attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
# for i, x_i in enumerate(inputs):
#     x_i: torch.Tensor
#     for j, x_j in enumerate(inputs):
#         x_j: torch.Tensor
#         attn_scores[i, j] = torch.dot(x_i, x_j)

# Fast approach using matrix multiplication.
attn_scores = inputs @ inputs.T
print("\nAttention scores:\n", attn_scores)

# Normalize.
attn_weights = torch.softmax(attn_scores, dim=-1)
print("\nAttention weights:\n", attn_weights)

# Verify that the rows sum to 1.
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("\nRow 2 sum:", row_2_sum)
print("All row sums:\n", attn_weights.sum(dim=-1))

# Compute all context vectors.
all_context_vecs = attn_weights @ inputs
print("\nAll context vectors:\n", all_context_vecs)
print("\nPrevious 2nd context vector:\n", context_vec_2)
