from torch import nn, zeros

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size=2, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(1, hidden_size, num_layers, batch_first=True, nonlinearity='sigmoid')
        self.fc = nn.Linear(hidden_size, 1)  # Fully connected layer for classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0) 
        out = self.fc(out)
        return self.sigmoid(out)