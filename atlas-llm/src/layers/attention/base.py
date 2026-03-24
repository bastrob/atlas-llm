
import torch.nn as nn


class AttentionBase(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        """
        Args:
            dim (int) hidden dimension
            num_heads (int) attention heads
        """
        self.dim = dim
        self.num_heads = num_heads

    def forward(self, x, mask=None, kv_cache=None, pos_emb=None):
        """
        Forward pass signature for all attention types.
        Subclasses must implement this.
        """
        raise NotImplementedError