import torch
import torch.nn as nn

from src.layers.attention import RopeEmbedding
from src.layers.norm import RMSNorm
from src.layers.utils import KVCache

from .block import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.tok_embs = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for i in range(cfg["n_layers"])])
        self.norm = RMSNorm(cfg["emb_dim"])
        self.output = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
        self.rope = RopeEmbedding(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
            )
        self.current_pos = 0 # Track current position in KV cache


    def forward(self, 
                input_ids: torch.LongTensor | None = None, 
                cache: KVCache | None = None
                ):
        
        if input_ids is None:
            raise ValueError("You must specify input_ids")

        input_embeds: torch.Tensor = self.tok_embs(input_ids)
        num_tokens: int = input_embeds.shape[1]

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            mask = torch.triu(
                torch.ones(pos_end, pos_end, dtype=torch.bool, device=input_embeds.device), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0
            pos_end = num_tokens
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=input_embeds.device), diagonal=1
            )
        
        hidden_states = input_embeds

        # Shape (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        mask = mask[None, None, :, :]
        
        # --- call RoPE ONCE here ---
        cos, sin = self.rope(pos_start, pos_end)

        for i, block in enumerate(self.blocks):
            blk_cache = cache.get(i) if cache else None
            hidden_states, blk_next_cache = block(hidden_states, mask=mask, cache=blk_cache, pos_emb=(cos, sin))

            if cache:
                cache.update(i, blk_next_cache)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary
        logits = self.output(hidden_states) # (B, S, vocab_size)

        return logits

    def reset_kv_cache(self):
        self.current_pos = 0