import torch
import torch.nn.functional as F
from torch.autograd import grad

# True label
y = torch.tensor([1.0])

# Input feature
x1 = torch.tensor([1.1])
# Weight parameter
w1 = torch.tensor([2.2], requires_grad=True)
# Bias unit
b = torch.tensor([0.0], requires_grad=True)

# Net input
z = x1 * w1 + b
# Activation and output
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)
print(f"loss: {loss}")

# Manually using the `grad` function
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)
print(f"grad_L_w1: {grad_L_w1}")
print(f"grad_L_b: {grad_L_b}")

# Automatically compute gradients for all leaf nodes
loss.backward()
print(f"w1.grad: {w1.grad}")
print(f"b.grad: {b.grad}")