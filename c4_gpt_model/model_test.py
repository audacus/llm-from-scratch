import torch
from matplotlib import pyplot as plt
from torch import nn

from data.config import GPT_CONFIG_124M
from data.model import batch
from utils.dummy_model import DummyGPTModel
from utils.model import LayerNorm, GELU, FeedForward, TransformerBlock
from utils.neural_network import ExampleDeepNeuralNetwork

print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

# Example from Figure 4.5.
torch.manual_seed(123)
# Create two training examples with five dimensions (features) each.
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# Examine mean and variance.
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# Apply layer normalization.
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layouer outputs:\n", out_norm)
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# Use layer norm class.
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# Compare GELU and ReLU activation function.
gelu, relu = GELU(), nn.ReLU()
# Create 100 sample data points in the range -3 to 3.
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

# Create plots.
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
# TODO: Uncomment to show plots.
# plt.show()

# Use feed forward network.
ffn = FeedForward(GPT_CONFIG_124M)
# Create sample input with batch dimension 2.
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

# Neural network without shortcut connections.
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False,
)


def print_gradients(model: nn.Module, x: torch.Tensor) -> None:
    # Forward pass.
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target and output are.
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate the gradients.
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


print("\nWeights without shortcut connections:")
print_gradients(model_without_shortcut, sample_input)

# Model with shortcut connections.
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True,
)

print("\nWeights with shortcut connections:")
print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
# Create sample input of shape [batch_size, num_tokens, emb_dim].
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("\nInput shape:", x.shape)
print("Output shape:", output.shape)
