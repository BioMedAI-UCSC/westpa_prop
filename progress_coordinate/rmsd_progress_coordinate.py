from .base_progress_coordinate import BaseProgressCoordinate
import numpy as np
import mdtraj


class RMSDProgressCoordinate(BaseProgressCoordinate):
    
    def __init__(self, reference_pdb_path=None, components=[0]):
        super().__init__()
        self.reference_pdb_path = reference_pdb_path
        self.reference_traj = None
        
        if reference_pdb_path is not None:
            self.reference_traj = mdtraj.load(reference_pdb_path)
        
        assert isinstance(components, list)
        self.components = components
    
    def calculate(self, data):
        self._validate_data_shape(data, expected_ndim=3)
        
        data_nm = data / 10.0
        
        n_atoms = data.shape[1]
        topology = mdtraj.Topology()
        chain = topology.add_chain()
        for i in range(n_atoms):
            residue = topology.add_residue(f"RES", chain)
            topology.add_atom(f"CA", mdtraj.element.carbon, residue)
        
        traj = mdtraj.Trajectory(data_nm, topology)
        
        if self.reference_traj is None:
            self.reference_traj = traj[0]
        
        rmsd_values = mdtraj.rmsd(traj, self.reference_traj, frame=0)
        
        rmsd_values = rmsd_values * 10.0
        
        rmsd_array = rmsd_values.reshape(-1, 1)
        
        if len(self.components) > 1:
            rmsd_array = np.tile(rmsd_array, (1, len(self.components)))
        
        return rmsd_array[:, self.components]
