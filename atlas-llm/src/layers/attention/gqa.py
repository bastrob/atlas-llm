import torch
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
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        # Linear projections for queries, keys, values, and output
        self.wk = nn.Linear(dim, num_kv_groups * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_groups * self.head_dim, bias=False)

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def apply_rope(self, x, cos, sin):
        """
        Apply Rotary Positional Embedding (RoPE) to Q or K.

        Args:
            x: (B, ?, S, HD) tensor
            cos: (1, 1, S, HD) precomputed cosine values
            sin: (1, 1, S, HD) precomputed sine values
        Returns:
            x with RoPE applied (B, ?, S, HD)
        """
        HD = x.shape[-1]
        assert HD % 2 == 0, "Head dimension must be even"

        # Split x into first half and second halh
        x1 = x[..., :HD // 2]
        x2 = x[..., HD // 2:]

         # Apply the rotary transformation
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.to(dtype=x.dtype)



    def forward(self, x, mask=None, cache=None, pos_emb=None):
        """
        Forward pass of GQA.

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

        # 1. Project Q, K, V
        queries = self.wq(x) # (B, S, D)
        keys = self.wk(x) # (B, S, num_kv_groups * head_dim)
        values = self.wv(x) # (B, S, num_kv_groups * head_dim)

        # 2. Reshape
        queries = queries.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, HD)
        keys = keys.view(B, S, self.num_kv_groups, self.head_dim).transpose(1, 2) # (B, G, S, HD)
        values = values.view(B, S, self.num_kv_groups, self.head_dim).transpose(1, 2) # (B, G, S, HD)

        # 3. Apply RoPE to Q and K if provided
        if pos_emb:
            cos, sin = pos_emb
            queries = self.apply_rope(queries, cos, sin)
            keys = self.apply_rope(keys, cos, sin)

        # 4. Use KV cache if provided
        if cache:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys], dim=2)
            values = torch.cat([prev_v, values], dim=2)

        # 5. Store updated KV cache values
        next_cache = (keys, values)

        # 6. Expand keys and values (group dim) to match the number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1) # (B, (G*GS)=H, S, HD)
        values = values.repeat_interleave(self.group_size, dim=1) 

        # 7. Compute attention logits (attn unnormalized): (Q @ K^T) / sqrt(head_dim)
        attn_scores = queries @ keys.transpose(2, 3) # (B, H, S, S)

        # 8. Apply attention mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))

        # 9. Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # 10. Multiply attention weights by values
        context_vec = (attn_weights @ values).transpose(1,2) # (B, S, H, HD)

        # 11. Merge heads
        context_vec = context_vec.reshape(B, S, D)

        # 12. Output linear projection
        context_vec = self.wo(context_vec)

        return context_vec, next_cache

