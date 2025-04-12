import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # Output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x) -> torch.nn.Sequential:
        # Return logits (outputs of the last layer)
        return self.layers(x)
