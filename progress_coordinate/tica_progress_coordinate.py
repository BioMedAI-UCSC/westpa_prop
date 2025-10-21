from .base_progress_coordinate import BaseProgressCoordinate
import numpy as np
import pickle


class TICAProgressCoordinate(BaseProgressCoordinate):
    
    def __init__(self, model_path, components=[0]):
        super().__init__()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            assert hasattr(model, "tica_model")
        self.tica_model = model.tica_model
        assert isinstance(components, list)
        self.components = components
    
    def calculate(self, data):
        self._validate_data_shape(data, expected_ndim=3)
        n_atoms = data.shape[1]
        pairs_a, pairs_b = np.triu_indices(n_atoms, k=1)
        distances = np.linalg.norm(data[:, pairs_a] - data[:, pairs_b], axis=2)
        distances /= 10
        tica_comps = self.tica_model.transform(distances)
        return tica_comps[:, self.components]
