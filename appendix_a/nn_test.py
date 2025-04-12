import torch

from neural_network import NeuralNetwork

model = NeuralNetwork(50, 3)

# Inspecting model
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

print(model.layers[0].weight.shape)

# Forward pass
torch.manual_seed(123)
X = torch.rand((1, 50))
# out = model(X)
# print(out)

# Do not keep track of the gradients
with torch.no_grad():
    # Compute class-membership probabilities using `softmax`
    out = torch.softmax(model(X), dim=1)
print(out)
