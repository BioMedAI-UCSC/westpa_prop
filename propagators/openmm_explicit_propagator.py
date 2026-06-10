import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from file_system.md_store.save_dcd import get_solute_indices, SoluteDCDReporter
from propagators.openmm_propagator import OpenMMPropagator 

from openmm.app import PME, HBonds
from openmm import MonteCarloBarostat
from openmm.unit import atmospheres, kelvin, nanometer, kilojoule_per_mole

from westpa.core.states import BasisState

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
        # Cα indices INTO the solute subset (chain-A then chain-B order). The
        # interface-RMSD pcoord reference (run_pair_we) is Cα-only, built from
        # this SAME solvated PDB, so these line up 1:1 with the reference atoms.
        self.solute_ca_indices = [a.index for a in self.md_top_solute.atoms
                                  if a.name == 'CA']
        # Structure-integrity diagnostics (mirror cgml_propagator): per-chain
        # Cα groups (indices INTO the Cα array) + the docked-initial Cα coords.
        # Lets us compare "atomistic stays folded AND dissociates" (rmsd_init
        # small, pcoord grows) against the CG path's collapse (rmsd_init → 30 Å,
        # pcoord = 0). Per-chain superposed RMSD is invariant to inter-chain
        # separation, so it isolates intra-chain integrity from dissociation.
        ca_atoms = [a for a in self.md_top_solute.atoms if a.name == 'CA']
        chain_ids = sorted({a.residue.chain.index for a in ca_atoms})
        self._ca_chain_groups = [
            np.array([k for k, a in enumerate(ca_atoms)
                      if a.residue.chain.index == c], dtype=int)
            for c in chain_ids]
        init_full = np.asarray(self.pdb.positions.value_in_unit(nanometer))
        self._init_ca = (init_full[self.solute_atom_indices][self.solute_ca_indices]
                         * 10.0).astype(np.float64)   # (n_ca, 3) Å, docked

    @staticmethod
    def _kabsch_rmsd(P, Q):
        """Superposed (rotation+translation-removed) RMSD between two (N,3)
        point sets via Kabsch. Å in, Å out."""
        if P.shape[0] < 3:
            return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))
        Pc = P - P.mean(0); Qc = Q - Q.mean(0)
        V, S, Wt = np.linalg.svd(Pc.T @ Qc)
        d = np.sign(np.linalg.det(Wt.T @ V.T))
        Pr = Pc @ (Wt.T @ np.diag([1.0, 1.0, d]) @ V.T).T
        return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))

    def _save_struct_metrics(self, segment_outdir, ca_positions):
        """rmsd_from_init = max over chains of per-chain superposed Cα RMSD vs
        the docked pose; rmsd_from_segstart = whole-Cα superposed RMSD of the
        final frame vs this segment's start."""
        final = ca_positions[-1].astype(np.float64)
        start = ca_positions[0].astype(np.float64)
        rmsd_init = max(self._kabsch_rmsd(final[idx], self._init_ca[idx])
                        for idx in self._ca_chain_groups)
        rmsd_seg = self._kabsch_rmsd(final, start)
        np.savez(os.path.join(segment_outdir, "struct_rmsd.npz"),
                 rmsd_from_init=np.float32(rmsd_init),
                 rmsd_from_segstart=np.float32(rmsd_seg))

    def get_pcoord(self, state):
        # Base OpenMMPropagator.get_pcoord feeds the FULL solvated system
        # (~40k atoms) to the pcoord calculator, but propagate()'s
        # _calculate_pcoord feeds solute-Cα only. For an interface-RMSD pcoord
        # whose reference is Cα-only those two paths MUST agree — so override
        # the basis-state pcoord here to also extract solute-Cα.
        if isinstance(state, BasisState):
            simulation = self._create_simulation(seg_id=0)
            simulation.context.setPositions(self.pdb.positions)
            simulation.minimizeEnergy(maxIterations=self.minimize_iters)
            st = simulation.context.getState(getPositions=True, getEnergy=True)
            pos = st.getPositions(asNumpy=True).value_in_unit(nanometer)
            ca = pos[self.solute_atom_indices][self.solute_ca_indices]
            ca = ca[np.newaxis, :, :] * 10.0   # (1, n_ca, 3) Å
            energy_u = st.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            energy_data = {"energy_k": [0.0], "energy_u": [energy_u], "times": [0.0]}
            state.pcoord = self.pcoord_calculator.calculate(ca, energy_data).reshape((-1, 1))
            return
        raise NotImplementedError

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
    
    def _calculate_pcoord(self, segment_outdir, initial_pos, energy_data):
        initial_pos_solute = initial_pos[:, self.solute_atom_indices, :]
        
        if self.save_format == 'dcd':
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            traj = mdtraj.load_dcd(dcd_path, top=self.md_top_solute)
            all_positions = np.concatenate([initial_pos_solute * 10, traj.xyz * 10])
        else:
            npz_path = os.path.join(segment_outdir, 'seg.npz')
            # seg.npz now stores SOLUTE-only positions (protein incl H, no water —
            # see propagate()), already in solute order, so do NOT re-slice.
            positions_solute = np.load(npz_path)['positions']
            all_positions = np.concatenate([initial_pos_solute * 10, positions_solute * 10])
        
        ca_positions = all_positions[:, self.solute_ca_indices, :]
        self._save_struct_metrics(segment_outdir, ca_positions)
        return self.pcoord_calculator.calculate(ca_positions, energy_data)
