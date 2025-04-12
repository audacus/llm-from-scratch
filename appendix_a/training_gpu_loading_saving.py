import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from appendix_a.neural_network import NeuralNetwork
from appendix_a.toy_dataset import ToyDataset


def compute_accuracy(model: torch.nn.Module, dataloader: DataLoader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features: Tensor
        labels: Tensor

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        # A tensor of `True/False` values depending on whether the labels match.
        compare = labels == predictions
        # Add the number of `True` (correct) values.
        correct += torch.sum(compare)
        total_examples += len(compare)

    # The fraction of correct prediction.
    # `.item()` returns the value of the tensor as a Python float.
    return (correct / total_examples).item()


print("cuda:", torch.cuda.is_available())
print("mps:", torch.backends.mps.is_available())
print("cpu:", torch.cpu.is_available())

# Define device following priorities: cuda > mps > cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("Device:", device)

# Train data
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5],
]).to(device)
y_train = torch.tensor([0, 0, 0, 1, 1]).to(device)

# Test data
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_text = torch.tensor([0, 1])

# Data sets
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_text)

# Print number of rows
print("Length of training dataset:", len(train_ds))

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    # Drop last batch to prevent disturbing the convergence.
    drop_last=True,
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    # Drop last batch to prevent disturbing the convergence.
    drop_last=True,
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Training batch {idx + 1}:", x, y)

for idx, (x, y) in enumerate(test_loader):
    print(f"Test batch {idx + 1}:", x, y)

# Typical training loop.
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# Transfer model onto device.
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        features: Tensor
        labels: Tensor
        # Transfer data onto the device
        features, labels = features.to(device), labels.to(device)

        logits: Tensor = model(features)
        loss = F.cross_entropy(logits, labels)

        # Reset gradients from previous round to prevent unintended gradient accumulation.
        optimizer.zero_grad()
        # Compute the gradients of the loss given the model parameters.
        loss.backward()
        # The optimizer uses the gradients to update the model parameters.
        optimizer.step()

        ### LOGGING
        print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train loss: {loss:.2f}")

    model.eval()
    # Insert optional model evaluation code here...

# Use the trained model to make predictions.
model.eval()
with torch.no_grad():
    outputs = model(X_train)
print("Training outputs:")
print(outputs)

# Obtain class membership probabilities.
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print("Membership probabilities:")
print(probas)

# Get class labels.
# `dim=1` -> return highest value in each row.
# `dim=0` -> return highest value in each column.
# predictions = torch.argmax(probas, dim=1)
predictions = torch.argmax(outputs, dim=1)
print("Predictions:", predictions)

# Compare predictions with the training dataset.
print("Compare predictions to training dataset:", predictions == y_train)
print(f"Correct predictions: {torch.sum(predictions == y_train)}")

print("Accuracy:", compute_accuracy(model, train_loader))

# Save state.
torch.save(model.state_dict(), "model.pth")

# Load state.
model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("model.pth"))
