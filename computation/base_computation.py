import numpy as np


class BaseComputation:
    """
    Pure computation over position data.

    Subclasses implement calculate(data) and nothing else.
    Storage, granularity, and WESTPA pcoord concerns are handled
    by the layers that wrap this class.

    data convention: (n_frames, n_atoms, 3) float32, Angstrom.
    """

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        raise NotImplementedError

    def _validate_input(self, data: np.ndarray):
        if data.ndim != 3:
            raise ValueError(f"Expected (n_frames, n_atoms, 3), got shape {data.shape}")
