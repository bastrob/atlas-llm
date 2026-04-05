import torch
from src.core import TransformerModel
from src.layers.utils import KVCache

LLAMA32_CONFIG_MOCK = {
    "vocab_size": 512,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 128,                 # Embedding dimension
    "n_heads": 2,                   # Number of attention heads
    "n_layers": 1,                  # Number of layers
    "hidden_dim": 64,              # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 4,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           
    "context_length": 131_072,      
    "emb_dim": 2048,                 
    "n_heads": 32,                   
    "n_layers": 16,                  
    "hidden_dim": 8192,              
    "n_kv_groups": 8,                
    "rope_base": 500_000.0,         
    "rope_freq": {                   
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

def generate(model: TransformerModel, idx: torch.Tensor, max_new_tokens: int=1, context_size: int | None = None, use_cache: bool=True):
    model.eval()
    ctx_len = context_size or model.cfg["context_length"]

    with torch.no_grad():
        if use_cache:
            cache = KVCache(n_layers=model.cfg["n_layers"])
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], cache=cache)

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, cache=cache)
        
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:])
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
    
    return idx


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama = TransformerModel(LLAMA32_CONFIG_MOCK)
    llama = llama.to(device)
    x = torch.LongTensor([[1, 2, 3, 4, 5]])
    x = x.to(device)

    result = generate(llama, idx=x, max_new_tokens=5)
    print(result)

if __name__ == "__main__":
    main()