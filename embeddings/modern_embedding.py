import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from embeddings.base_embedding import BaseEmbedding


class VAMPNetEmbedding(BaseEmbedding):
    """VAMPnet (deeptime): a nonlinear, kinetically-meaningful learned coordinate.

    Trains on time-lagged pairs built from trajectory `lengths`. A modern
    alternative to the CVAE for the learned-PC comparison.
    """

    def __init__(self, n_components=2, lag=1, hidden=(64, 64), epochs=40,
                 lr=1e-3, batch=256, scale=True, device=None, seed=0):
        super().__init__(n_components)
        self.lag = int(lag)
        self.hidden = tuple(hidden)
        self.epochs, self.lr, self.batch = int(epochs), float(lr), int(batch)
        self.scale, self.seed, self.device = scale, int(seed), device
        self.scaler = None
        self.model = None

    def _dev(self):
        return self.safe_device(self.device)

    def fit(self, X, lengths=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from deeptime.decomposition.deep import VAMPNet
        from deeptime.util.data import TrajectoriesDataset
        from deeptime.util.torch import MLP

        torch.manual_seed(self.seed)
        X = self.flatten(X)
        if self.scale:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        trajs = [t.astype(np.float32) for t in self.split_lengths(X, lengths)
                 if len(t) > self.lag]
        ds = TrajectoriesDataset.from_numpy(lagtime=self.lag, data=trajs)

        units = [X.shape[1], *self.hidden, self.n_components]
        lobe = MLP(units, nonlinearity=nn.ELU, output_nonlinearity=None).to(self._dev())
        vnet = VAMPNet(lobe=lobe, learning_rate=self.lr, device=self._dev())
        loader = DataLoader(ds, batch_size=self.batch, shuffle=True)
        self.model = vnet.fit(loader, n_epochs=self.epochs).fetch_model()
        return self

    def transform(self, X):
        X = self.flatten(X)
        X = self.scaler.transform(X) if self.scaler is not None else X
        Z = self.model.transform(X.astype(np.float32))
        return np.asarray(Z[:, :self.n_components], dtype=np.float32)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"cfg": {k: getattr(self, k) for k in
                                 ("n_components", "lag", "hidden", "epochs",
                                  "lr", "batch", "scale", "seed")},
                         "scaler": self.scaler, "model": self.model}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(**d["cfg"])
        obj.scaler, obj.model = d["scaler"], d["model"]
        return obj
