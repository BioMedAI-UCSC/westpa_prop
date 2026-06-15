import pickle

import numpy as np

from embeddings.base_embedding import BaseEmbedding


class LinearEmbedding(BaseEmbedding):
    """PCA or TICA over flattened features.

    method="pca": sklearn PCA. method="tica": deeptime TICA at `lag` (uses
    `lengths` to respect trajectory boundaries when building lagged pairs).
    """

    def __init__(self, n_components=2, method="pca", lag=1, scale=True):
        super().__init__(n_components)
        if method not in ("pca", "tica"):
            raise ValueError(method)
        self.method = method
        self.lag = int(lag)
        self.scale = scale
        self.scaler = None
        self.model = None

    def fit(self, X, lengths=None):
        Xs = self._prep_fit(self.flatten(X))
        if self.method == "pca":
            from sklearn.decomposition import PCA
            self.model = PCA(n_components=self.n_components).fit(Xs)
        else:
            from deeptime.decomposition import TICA
            trajs = self.split_lengths(Xs, lengths)
            est = TICA(lagtime=self.lag, dim=self.n_components)
            for t in trajs:
                if len(t) > self.lag:
                    est.partial_fit((t[:-self.lag], t[self.lag:]))
            self.model = est.fetch_model()
        return self

    def transform(self, X):
        Xs = self.flatten(X)
        Xs = self.scaler.transform(Xs) if self.scaler is not None else Xs
        Z = self.model.transform(Xs)
        return np.asarray(Z[:, :self.n_components], dtype=np.float32)

    def _prep_fit(self, X):
        if self.scale:
            self.scaler = self.fit_scaler(X)
            return self.scaler.transform(X)
        return X

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"method": self.method, "lag": self.lag,
                         "n_components": self.n_components, "scaler": self.scaler,
                         "model": self.model}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(n_components=d["n_components"], method=d["method"], lag=d["lag"])
        obj.scaler, obj.model = d["scaler"], d["model"]
        return obj
