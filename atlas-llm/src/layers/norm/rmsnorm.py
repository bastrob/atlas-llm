
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8) -> None:
        """
        Args:
            dim (int): hidden dimension size
            eps (float): numerical stability constant
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        returns: same shape as input
        """
        # Calculate RMS accross the last dimension(s)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize
        x_norm = x * rms * self.weight
        return x_norm