from torch import nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, bitstring_length, hidden_size=512):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(bitstring_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
