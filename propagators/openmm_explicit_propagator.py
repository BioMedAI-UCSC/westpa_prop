import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from file_system.md_store.save_dcd import get_solute_indices, SoluteDCDReporter
from propagators.openmm_propagator import OpenMMPropagator 

from openmm.app import PME, HBonds
from openmm import MonteCarloBarostat
from openmm.unit import atmospheres, kelvin

import mdtraj
import numpy as np
import os


class OpenMMExplicitPropagator(OpenMMPropagator):
    
    def __init__(self, rc=None):
        super(OpenMMExplicitPropagator, self).__init__(rc)
        if self.implicit_solvent:
            raise ValueError("OpenMMExplicitPropagator requires implicit_solvent=False")
        self.nonbondedMethod = PME
        
        md_top_full = mdtraj.Topology.from_openmm(self.pdb.topology)
        self.solute_atom_indices = get_solute_indices(md_top_full)
        self.md_top_solute = md_top_full.subset(self.solute_atom_indices)
    
    def _create_system(self):
        system = self.forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=self.nonbondedMethod,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=self.hydrogenMass
        )
        system.addForce(MonteCarloBarostat(
            self.pressure * atmospheres,
            self.temperature * kelvin,
            self.barostatInterval
        ))
        return system
    
    def _setup_reporters(self, simulation, segment_outdir):
        if self.save_format == 'dcd':
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            simulation.reporters.clear()
            simulation.reporters.append(
                SoluteDCDReporter(dcd_path, self.save_steps, self.solute_atom_indices,
                                enforcePeriodicBox=False, append=False)
            )
    
    def _calculate_pcoord(self, segment_outdir, initial_pos):
        initial_pos_solute = initial_pos[:, self.solute_atom_indices, :]
        
        if self.save_format == 'dcd':
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            traj = mdtraj.load_dcd(dcd_path, top=self.md_top_solute)
            all_positions = np.concatenate([initial_pos_solute * 10, traj.xyz * 10])
        else:
            npz_path = os.path.join(segment_outdir, 'seg.npz')
            positions = np.load(npz_path)['positions']
            positions_solute = positions[:, self.solute_atom_indices, :]
            all_positions = np.concatenate([initial_pos_solute * 10, positions_solute * 10])
        
        ca_indices = [atom.index for atom in self.md_top_solute.atoms if atom.name == 'CA']
        ca_positions = all_positions[:, ca_indices, :]
        return self.pcoord_calculator.calculate(ca_positions)
