import torch.nn as nn
from activations import silu


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = silu(x_fc1) * x_fc2
        return self.fc3(x)