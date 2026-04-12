from .llama import LLamaRopeEmbedding
from .qwen import QwenRopeEmbedding

ROPE_REGISTRY = {
    "llama": LLamaRopeEmbedding,
    "qwen": QwenRopeEmbedding
}

class RopeFactory:
    @staticmethod
    def build(config):
        cls = ROPE_REGISTRY[config["rope_type"]]
        return cls.from_cfg(config)