import numpy as np


class BaseEmbedding:
    def __init__(self, n_components=2):
        self.n_components = int(n_components)

    def fit(self, X, lengths=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, lengths=None):
        return self.fit(X, lengths).transform(X)

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    @staticmethod
    def safe_device(preferred=None):
        """Return a usable torch device. Falls back to CPU if CUDA is present but
        unusable (e.g. PyTorch built without this GPU's compute capability)."""
        import torch
        if preferred and preferred != "cuda":
            return preferred
        if not torch.cuda.is_available():
            return "cpu"
        try:
            (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).cpu()
            conv = torch.nn.Conv2d(1, 1, 3, padding=1).cuda()
            conv(torch.zeros(1, 1, 8, 8, device="cuda")).cpu()
            return "cuda"
        except Exception:
            return "cpu"

    @staticmethod
    def flatten(X):
        X = np.asarray(X, dtype=np.float32)
        return X.reshape(X.shape[0], -1)

    @staticmethod
    def split_lengths(X, lengths):
        """Yield contiguous sub-arrays per trajectory length (for lagged methods)."""
        if lengths is None:
            return [X]
        out, s = [], 0
        for n in lengths:
            out.append(X[s:s + n])
            s += n
        return out
