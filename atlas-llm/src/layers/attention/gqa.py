import torch.nn as nn

from .base import AttentionBase


class GQA(AttentionBase):
    def __init__(self, dim, num_heads, num_kv_groups):
        """
        Grouped Query Attention module with optional RoPE and KV cache support.
        """
        super().__init__(dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.group_size = num_heads // num_kv_groups

        # Linear projections for queries, keys, values, and output
        self.wk = nn.Linear(dim, num_kv_groups * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_groups * self.head_dim, bias=False)

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    

