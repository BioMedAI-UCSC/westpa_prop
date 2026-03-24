import numpy as np
from computation.base_computation import BaseComputation

VALID_COMPONENTS = {"energy_k", "energy_u", "times"}


class OpenMMEnergyComputation(BaseComputation):
    """
    Returns per-frame OpenMM energy data as a progress coordinate.

    energy_data dict keys:
        energy_k : list or array of shape (n_frames,)  — kinetic energy  (kJ/mol)
        energy_u : list or array of shape (n_frames,)  — potential energy (kJ/mol)
        times    : list or array of shape (n_frames,)  — simulation time  (ps)

    Returns an ndarray of shape (n_frames, len(components)).

    Config example (west.cfg):
        recorded_calculators:
          - name:        energy
            storage:     west_h5
            granularity: per_frame
            computation:
              class:      computation.openmm_energy_computation.OpenMMEnergyComputation
              components: [energy_u]          # or [energy_k, energy_u, times]
    """

    requires_positions = False
    requires_energy    = True

    def __init__(self, components: list[str] | None = None):
        if components is None:
            components = ["energy_u"]

        unknown = set(components) - VALID_COMPONENTS
        if unknown:
            raise ValueError(
                f"Unknown energy component(s): {unknown}. "
                f"Valid options are: {VALID_COMPONENTS}"
            )
        if not components:
            raise ValueError("components must contain at least one entry.")

        self.components = components

    def calculate(self, positions: np.ndarray, energy_data: dict) -> np.ndarray:
        """
        Parameters
        ----------
        positions : np.ndarray
            Ignored. Present to conform to the unified BaseComputation signature.
        energy_data : dict
            Must contain keys matching self.components. Expected keys:
                energy_k, energy_u, times

        Returns
        -------
        np.ndarray, shape (n_frames, len(self.components)), dtype float64
        """
        self._validate_input(positions, energy_data)

        columns = [np.asarray(energy_data[c], dtype=np.float64) for c in self.components]

        lengths = [col.shape[0] for col in columns]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Energy component arrays have inconsistent lengths: "
                f"{dict(zip(self.components, lengths))}"
            )

        return np.column_stack(columns) if len(columns) > 1 else columns[0].reshape(-1, 1)

    def _validate_input(self, positions, energy_data):
        if not isinstance(energy_data, dict):
            raise TypeError(
                f"OpenMMEnergyComputation expects energy_data as a dict, got {type(energy_data)}"
            )
        missing = set(self.components) - set(energy_data.keys())
        if missing:
            raise KeyError(f"energy_data dict is missing required keys: {missing}")
