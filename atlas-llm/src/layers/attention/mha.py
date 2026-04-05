import torch
import torch.nn as nn

from .base import AttentionBase


class MHA(AttentionBase):
    def __init__(self, dim, num_heads):
        """
        Multi-Head Attention module with optional RoPE and KV cache support.
        """
        super().__init__(dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Linear projections for queries, keys, values, and output
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def apply_rope(self, x, cos, sin):
        """
        Apply Rotary Positional Embedding (RoPE) to Q or K.

        Args:
            x: (B, H, S, HD) tensor
            cos: (1, 1, S, HD) precomputed cosine values
            sin: (1, 1, S, HD) precomputed sine values
        Returns:
            x with RoPE applied (B, H, S, HD)
        """
        HD = x.shape[-1]
        assert HD % 2 == 0, "Head dimension must be even"

        x1 = x[..., :HD // 2]  # even indices
        x2 = x[..., HD // 2:] # odd indices

         # Apply the rotary transformation
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.to(dtype=x.dtype)

    def _reshape_heads(self, x):
        """
        Reshape (B, S, D) -> (B, H, S, HD) for multi-head attention.
        """
        B, S, D = x.shape
        H = self.num_heads
        HD = D // H
        return x.view(B, S, H, HD).transpose(1, 2) # (B, H, S, HD)
    
    def forward(self, x, mask=None, cache=None, pos_emb=None):
        """
        Forward pass of MHA.

        Args:
            x: (B, S, D) input embeddings
            mask: optional attention mask (0 = masked)
            cache: optional dict for autoregressive inference
            pos_emb: tuple (cos, sin) for RoPE

        Returns:
            output: (B, S, D) attention output
            next_cache: updated KV cache tuple (k, v)
        """
        B, S, D = x.shape

        #1. Project Q, K, V
        q = self.wq(x)
        new_k = self.wk(x)
        new_v = self.wv(x)

        #2. Reshape for multi-head
        q = self._reshape_heads(q)
        new_k = self._reshape_heads(new_k)
        new_v = self._reshape_heads(new_v)

        #3. Apply RoPE to Q and K if provided
        if pos_emb is not None:
            cos, sin = pos_emb
            q = self.apply_rope(q, cos, sin)
            new_k = self.apply_rope(new_k, cos, sin)

        #4. Use KV cache if provided
        if cache is not None:
            prev_k, prev_v = cache
            k = torch.cat([prev_k, new_k], dim=2)
            v = torch.cat([prev_v, new_v], dim=2)
        else:
            k, v = new_k, new_v

        #5. Store updated KV cache values
        next_cache = (k, v)

        #6. Compute attention logits (attn unnormalized): (Q @ K^T) / sqrt(head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) # (B, H, S, HD) @ (B, H, HD, S) -> (B, H, S, S)
        attn_logits = attn_logits / (self.head_dim ** 0.5)

        #7. Apply attention mask if provided
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.bool(), float('-inf'))

        #8. Softmax to get attention weights
        attn_weights = torch.softmax(attn_logits, dim=-1)

        #8. Multiply attention weights by values
        context_vec = torch.matmul(attn_weights, v)

        #9. Merge heads
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(B, S, D)
        #10. Output linear projection
        context_vec = self.wo(context_vec)

        return context_vec, next_cache