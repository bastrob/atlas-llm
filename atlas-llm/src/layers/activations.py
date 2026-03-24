import torch


def sigmoid(x):
    return torch.where(
            x >= 0,
            1 / (1 + torch.exp(-x)),
            torch.exp(x) / (1 + torch.exp(x))
        )

def silu(x):
    return x * sigmoid(x)