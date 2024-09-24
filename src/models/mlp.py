from torch import nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, bitstring_length):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(bitstring_length, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
