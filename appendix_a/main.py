from neural_network import NeuralNetwork

model = NeuralNetwork(50, 3)

print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)