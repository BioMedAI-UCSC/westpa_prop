import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from propagators.openmm_propagator import OpenMMPropagator 
from file_system.md_store.save_dcd import FullDCDReporter 

from openmm.app import NoCutoff, HBonds

import mdtraj
import numpy as np
import os


class OpenMMImplicitPropagator(OpenMMPropagator):
    
    def __init__(self, rc=None):
        super(OpenMMImplicitPropagator, self).__init__(rc)
        if not self.implicit_solvent:
            raise ValueError("OpenMMImplicitPropagator requires implicit_solvent=True")
        self.nonbondedMethod = NoCutoff
    
    def _create_system(self):
        return self.forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=self.nonbondedMethod,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=self.hydrogenMass
        )
    
    def _setup_reporters(self, simulation, segment_outdir):
        if self.save_format == 'dcd':
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            simulation.reporters.clear()
            simulation.reporters.append(FullDCDReporter(dcd_path, self.save_steps))
    
    def _calculate_pcoord(self, segment_outdir, initial_pos):
        if self.save_format == 'dcd':
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            md_top = mdtraj.Topology.from_openmm(self.pdb.topology)
            traj = mdtraj.load_dcd(dcd_path, top=md_top)
            all_positions = np.concatenate([initial_pos * 10, traj.xyz * 10])
        else:
            npz_path = os.path.join(segment_outdir, 'seg.npz')
            positions = np.load(npz_path)['positions']
            all_positions = np.concatenate([initial_pos * 10, positions * 10])
        
        ca_indices = [atom.index for atom in self.pdb.topology.atoms() if atom.name == 'CA']
        ca_positions = all_positions[:, ca_indices, :]
        return self.pcoord_calculator.calculate(ca_positions)
