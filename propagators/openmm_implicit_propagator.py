import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from propagators.openmm_propagator import OpenMMPropagator 
from file_system.md_store.save_dcd import FullDCDReporter 

from openmm.app import NoCutoff, HBonds, Modeller

import pdbfixer
import mdtraj
import numpy as np
import os


class OpenMMImplicitPropagator(OpenMMPropagator):
    
    def __init__(self, rc=None):
        super(OpenMMImplicitPropagator, self).__init__(rc)
        if not self.implicit_solvent:
            raise ValueError("OpenMMImplicitPropagator requires implicit_solvent=True")

        fixer = self._create_and_configure_fixer()
        modeller = self._create_modeller(fixer)
        self._finalize_system(modeller)

    def _create_and_configure_fixer(self):
        fixer = pdbfixer.PDBFixer(self.topology_path)
        
        self._find_missing_components(fixer)
        self._print_diagnostics(fixer)
        self._remove_terminal_missing_residues(fixer)
        self._fix_nonstandard_residues(fixer)
        self._add_missing_components(fixer)
        
        return fixer
    
    def _find_missing_components(self, fixer):
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
    
    def _print_diagnostics(self, fixer):
        print(f"Missing residues: {fixer.missingResidues}")
        print(f"Missing terminals: {fixer.missingTerminals}")
        print(f"Missing atoms: {fixer.missingAtoms}")
    
    def _remove_terminal_missing_residues(self, fixer):
        chains = list(fixer.topology.chains())
        keys = list(fixer.missingResidues.keys())
        
        for key in keys:
            chain = chains[key[0]]
            chain_length = len(list(chain.residues()))
            
            if key[1] == 0 or key[1] == chain_length:
                del fixer.missingResidues[key]
        
        self._verify_terminal_removal(fixer, chains, keys)
    
    def _verify_terminal_removal(self, fixer, chains, original_keys):
        for key in original_keys:
            chain = chains[key[0]]
            chain_length = len(list(chain.residues()))
            
            assert key[1] != 0 or key[1] != chain_length, \
                "Terminal residues are not removed."
    
    def _fix_nonstandard_residues(self, fixer):
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
    
    def _add_missing_components(self, fixer, ph=7.0):
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)
    
    def _create_modeller(self, fixer):
        modeller = Modeller(fixer.topology, fixer.positions)
        self._prepare_implicit_solvent(modeller)
        
        return modeller
    
    def _prepare_implicit_solvent(self, modeller):
        modeller.deleteWater()
        
        ions_to_delete = [
            res for res in modeller.topology.residues() 
            if res.name in ('NA', 'CL')
        ]
        modeller.delete(ions_to_delete)
    
    def _finalize_system(self, modeller):
        self.pdb.topology = modeller.getTopology()
        self.pdb.positions = modeller.getPositions()
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
