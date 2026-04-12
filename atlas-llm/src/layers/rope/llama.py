import torch

from .base import RopeBase


class LLamaRopeEmbedding(RopeBase):
    def __init__(self, head_dim, theta_base, context_length, freq_config):
        super().__init__()

        self.cos: torch.Tensor
        self.sin: torch.Tensor

        # 1. Compute theta_i = 1.0 / (base ^ (2i / d))
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Frequency adjustments
        if freq_config is not None:
            low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
            high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

            wavelen = 2 * torch.pi / inv_freq

            inv_freq_llama = torch.where(
                wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
            )

            smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
                freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
            )

            smoothed_inv_freq = (
                (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
            )

            is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            inv_freq = inv_freq_llama

        # 2. Generate position indices 'm'
        positions = torch.arange(context_length, dtype=torch.float32) 

        # 3. Outer product: m * theta_i
        angles = torch.einsum("i,j->ij", positions, inv_freq) # Shape: (context_length, head_dim // 2)
        angles = angles.repeat_interleave(2, dim=-1) # Shape: (context_length, head_dim)
        
        cos = angles.cos()
        sin = angles.sin()

        # 4. Register buffers
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
            )

    def forward(self, pos_start: int, pos_end: int):
        """
        Returns RoPE embeddings for positions [pos_start, pos_end]

        Output shape:
            cos, sin: (1, 1, S, head_dim)
        """
        # Adjust sin and cos shapes
        cos = self.cos[pos_start:pos_end, :] # (S, D)
        sin = self.sin[pos_start:pos_end, :] # (S, D)

        # reshape for broadcasting with (B, H, S, HD)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, HD)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, S, HD)

        return cos, sin