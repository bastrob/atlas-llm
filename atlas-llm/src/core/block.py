
import torch.nn as nn
from layers.attention import MHA
from layers.ffn import FeedForward
from layers.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = MHA(dim, num_heads)

        self.ffn_norm =  RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)
    
    def forward(self, x, mask=None, kv_cache=None, pos_emb=None):
        residual = x
        x = self.attn_norm(x)
        attn_out, next_cache = self.attn(
            x,
            mask=mask,
            kv_cache=kv_cache,
            pos_emb=pos_emb
        )
        x = attn_out + residual
        x = self.ffn(self.ffn_norm(x)) + x

        return x, next_cache
