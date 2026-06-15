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


def method_kwargs(method, n_components=2, lag=1, epochs=30):
    kw = {"n_components": n_components}
    if method in ("tica", "vampnet"):
        kw["lag"] = lag
    if method in ("cvae", "vampnet"):
        kw["epochs"] = epochs
    return kw


def default_feature_mode(method):
    return "contact_map" if method == "cvae" else "vector"


def _cls(name):
    import importlib
    mod, cls, _ = REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)


def save_model(emb, name, path):
    """Save an embedding atomically with a .meta sidecar recording its method."""
    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    emb.save(tmp)
    os.replace(tmp, path)
    with open(path + ".meta", "w") as f:
        json.dump({"method": name}, f)


def load_model(path):
    import json
    with open(path + ".meta") as f:
        name = json.load(f)["method"]
    return _cls(name).load(path)
