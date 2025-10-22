from progress_coordinate.base_progress_coordinate import BaseProgressCoordinate
import numpy as np
import mdtraj


class RMSDProgressCoordinate(BaseProgressCoordinate):

    def __init__(self, reference_pdb_path=None, reference_xml_path=None, atom_selection="name CA", components=[0]):
        super().__init__()
        self.reference_pdb_path = reference_pdb_path
        self.reference_xml_path = reference_xml_path
        self.atom_selection = atom_selection
        self.reference_traj = None
        self.atom_indices = None
        
        if reference_pdb_path is not None:
            if reference_xml_path is not None:
                full_traj = mdtraj.load(reference_xml_path, top=reference_pdb_path)
            else:
                full_traj = mdtraj.load(reference_pdb_path)
            
            self.atom_indices = full_traj.topology.select(atom_selection)
            
            if len(self.atom_indices) == 0:
                raise ValueError(f"No atoms found with selection: {atom_selection}")

            self.reference_traj = full_traj.atom_slice(self.atom_indices)
        
        assert isinstance(components, list)
        self.components = components

    
    def calculate(self, data):
        self._validate_data_shape(data, expected_ndim=3)
        
        data_nm = data / 10.0
        
        if self.reference_traj is not None:
            traj = mdtraj.Trajectory(data_nm, self.reference_traj.topology)
        else:
            n_atoms = data.shape[1]
            topology = mdtraj.Topology()
            chain = topology.add_chain()
            for i in range(n_atoms):
                residue = topology.add_residue(f"RES", chain)
                topology.add_atom(f"CA", mdtraj.element.carbon, residue)
            
            traj = mdtraj.Trajectory(data_nm, topology)
            
            if self.reference_traj is None:
                self.reference_traj = traj[0]
                self.atom_indices = None
        
        rmsd_values = mdtraj.rmsd(traj, self.reference_traj)
        rmsd_values = rmsd_values * 10.0
        rmsd_array = rmsd_values.reshape(-1, 1)
        
        if len(self.components) > 1:
            rmsd_array = np.tile(rmsd_array, (1, len(self.components)))
        
        return rmsd_array[:, self.components]
