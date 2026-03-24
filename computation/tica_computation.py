import pickle
import numpy as np

from computation.base_computation import BaseComputation


class TICAComputation(BaseComputation):

    def __init__(self, model_path, components=None):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "tica_model")
        self.tica_model = model.tica_model
        self.components = components  # None -> all components

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        n_atoms = data.shape[1]
        a, b = np.triu_indices(n_atoms, k=1)
        distances = np.linalg.norm(data[:, a] - data[:, b], axis=2) / 10.0
        result = self.tica_model.transform(distances)
        if self.components is not None:
            result = result[:, self.components]
        return result.astype(np.float32)
