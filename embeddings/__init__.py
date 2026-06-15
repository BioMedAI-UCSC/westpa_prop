"""Low-dimensional embeddings (learned progress coordinates).

All share BaseEmbedding: fit / transform / fit_transform / save / load.

    linear_embedding.LinearEmbedding   method="pca" | "tica"
    cvae_embedding.CVAEEmbedding       convolutional VAE on contact maps (Paper 2)
    modern_embedding.VAMPNetEmbedding  VAMPnet (deeptime)
"""

REGISTRY = {
    "pca": ("embeddings.linear_embedding", "LinearEmbedding", {"method": "pca"}),
    "tica": ("embeddings.linear_embedding", "LinearEmbedding", {"method": "tica"}),
    "cvae": ("embeddings.cvae_embedding", "CVAEEmbedding", {}),
    "vampnet": ("embeddings.modern_embedding", "VAMPNetEmbedding", {}),
}


def build(name, **kwargs):
    import importlib
    mod, cls, defaults = REGISTRY[name]
    obj = getattr(importlib.import_module(mod), cls)
    return obj(**{**defaults, **kwargs})
