import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


class RMSDComputation(BaseComputation):

    def __init__(self, reference_pdb_path, reference_xml_path=None, atom_selection="name CA", components=None):
        if reference_xml_path is not None:
            ref = mdtraj.load(reference_xml_path, top=reference_pdb_path)
        else:
            ref = mdtraj.load(reference_pdb_path)

        self.atom_indices = ref.topology.select(atom_selection)
        if len(self.atom_indices) == 0:
            raise ValueError(f"No atoms matched selection '{atom_selection}'")

        self.reference  = ref.atom_slice(self.atom_indices)
        self.topology   = self.reference.topology
        self.components = components

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        data_nm = (data[:, self.atom_indices, :] / 10.0).astype(np.float32)
        traj = mdtraj.Trajectory(data_nm, self.topology)
        result = (mdtraj.rmsd(traj, self.reference) * 10.0).reshape(-1, 1).astype(np.float32)
        if self.components is not None:
            result = result[:, self.components]
        return result
