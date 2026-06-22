import os

import numpy as np

from computation.base_computation import BaseComputation
from features.interface_featurizer import InterfaceFeaturizer
from embeddings import load_model


class LearnedPCoord(BaseComputation):
    """Progress coordinate = embedding(interface features).

    Loads the current embedding model from `model_path`, hot-reloading when the
    file changes (the plugin retrains and rewrites it between iterations). Until a
    model exists, falls back to the first `n_components` standardized features so
    w_init / iteration 1 still produce a valid pcoord.

    Featurizer kwargs (mode/scheme/contact_cutoff_angstrom/...) are passed through.
    """

    requires_positions = True
    requires_energy = False

    def __init__(self, topology_path, selection_a, selection_b, model_path,
                 n_components=2, feature_mode="vector", **feat_kwargs):
        self.model_path = os.path.expandvars(model_path)
        self.n_components = int(n_components)
        self.featurizer = InterfaceFeaturizer(topology_path, selection_a, selection_b,
                                              mode=feature_mode, **feat_kwargs)
        self._model = None
        self._mtime = None

    def _refresh_model(self):
        if not os.path.isfile(self.model_path):
            self._model = None
            return
        mt = os.path.getmtime(self.model_path)
        if self._model is None or mt != self._mtime:
            self._model = load_model(self.model_path)
            self._mtime = mt

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        X = self.featurizer.calculate(data)
        self._refresh_model()
        if self._model is not None:
            Z = np.asarray(self._model.transform(X), dtype=np.float32)
        else:
            Z = self._fallback(X)
        Z = Z[:, :self.n_components]
        if not np.all(np.isfinite(Z)):
            raise ValueError("non-finite learned pcoord")
        return Z if Z.shape[0] > 1 else Z[0]

    def _fallback(self, X):
        Xf = X.reshape(X.shape[0], -1).astype(np.float32)
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
        Z = (Xf - mu) / sd
        if Z.shape[1] < self.n_components:
            Z = np.pad(Z, ((0, 0), (0, self.n_components - Z.shape[1])))
        return Z[:, :self.n_components]
