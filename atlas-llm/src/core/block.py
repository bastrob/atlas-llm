
import torch.nn as nn

from src.layers.attention import MHA
from src.layers.ffn import FeedForward
from src.layers.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn_norm = RMSNorm(cfg["emb_dim"])
        self.attn = MHA(cfg["emb_dim"], cfg["n_heads"])

        self.ffn_norm =  RMSNorm(cfg["emb_dim"])
        self.ffn = FeedForward(cfg["emb_dim"], cfg["hidden_dim"])
    
    def forward(self, x, mask=None, cache=None, pos_emb=None):
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, next_cache = self.attn(
            x_norm,
            mask=mask,
            cache=cache,
            pos_emb=pos_emb
        )
        x = attn_out + residual
        x = self.ffn(self.ffn_norm(x)) + x

        return x, next_cache
