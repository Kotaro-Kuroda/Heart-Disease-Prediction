from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.input_features = input_features
        self.num_classes = num_classes

        self.network = nn.Sequential()
        self.network.add_module('fc1', nn.Linear(input_features, 8))
        self.network.add_module('relu1', nn.ReLU(inplace=False))
        self.network.add_module('fc2', nn.Linear(8, 8))
        self.network.add_module('relu2', nn.ReLU(inplace=False))
        self.network.add_module('fc3', nn.Linear(8, 16))
        self.network.add_module('relu3', nn.ReLU(inplace=False))
        self.network.add_module('fc4', nn.Linear(16, 8))
        self.network.add_module('relu4', nn.ReLU(inplace=False))
        self.network.add_module('fc5', nn.Linear(8, 4))
        self.network.add_module('relu5', nn.ReLU(inplace=False))
        self.network.add_module('fc6', nn.Linear(4, num_classes))

    def forward(self, x):
        logits = self.network(x)
        return logits
