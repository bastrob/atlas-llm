from typing import Tuple

import torch

from .base import RopeBase


class QwenRopeEmbedding(RopeBase):
    """
    Dynamic Rotary Positional Embedding (RoPE).

    Implements Qwen-style dynamic scaling of RoPE frequencies to improve
    long-context stability beyond the training context length.

    Key idea:
        - Base RoPE frequencies are fixed (inv_freq)
        - Frequencies are scaled at runtime depending on sequence length
        - Allows extrapolation beyond training context without precomputing full tables
    """
    def __init__(self, head_dim, theta_base, context_length):
        super().__init__()

        # Maximum context length used as reference for scaling
        self.context_length = context_length

        # Base inverse frequencies for rotary embeddings (fixed per model config)
        self.inv_freq: torch.Tensor

        # Compute theta_i = 1.0 / (base ^ (2i / d))
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
            )

    def forward(self, pos_start: int, pos_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE cos/sin embeddings for a given position range.

        Args:
            pos_start: start position in sequence (inclusive)
            pos_end: end position in sequence (exclusive)

        Returns:
            cos, sin: tensors shaped (1, 1, S, head_dim)
                      ready for broadcasting in attention:
                      (batch, heads, seq_len, head_dim)
        """

        # Determine dynamic scaling factor based on sequence length
        # If within training range → no scaling
        # If beyond → frequencies are compressed
        if pos_end <= self.context_length:
            scale = 1.0
        else:
            scale = pos_end / self.context_length

        # Create position indices for current sequence chunk
        positions = torch.arange(pos_start, pos_end, device=self.inv_freq.device).float()

        # Apply frequency scaling (core Qwen idea)
        scaled_inv_freq = self.inv_freq / scale

        # Compute outer product: position × frequency
        angles = torch.einsum("i,j->ij", positions, scaled_inv_freq) # Shape: (context_length, head_dim // 2)
        
        # Expand to full head dimension by duplicating pairs
        angles = angles.repeat_interleave(2, dim=-1) # Shape: (context_length, head_dim)

        # Convert angles into rotation components
        cos = angles.cos()
        sin = angles.sin()

        # Reshape for broadcasting with (B, H, S, HD)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, HD)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, S, HD)

        return cos, sin