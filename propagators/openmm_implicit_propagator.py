import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from propagators.openmm_propagator import OpenMMPropagator
from file_system.md_store.save_dcd import FullDCDReporter

from openmm.app import NoCutoff, HBonds, Modeller, Simulation, PDBFile
from openmm import LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, femtosecond

import pdbfixer
import mdtraj
import numpy as np
import os


# Modified RNA residues → closest standard analog for AMBER14
_RNA_MOD_MAP = {
    'PSU': 'U', 'H2U': 'U', '5MU': 'U', 'OMU': 'U',
    '5MC': 'C', 'OMC': 'C',
    '1MG': 'G', 'M2G': 'G', 'OMG': 'G', 'YG':  'G', 'YYG': 'G', 'QUO': 'G',
    'MA6': 'A', '6MA': 'A', 'T6A': 'A', 'MIA': 'A', 'OMA': 'A', 'I': 'A',
}


# Atoms that belong to a 5'-phosphate cap on 5'-terminal RNA residues.
# AMBER14 X5 templates (A5/G5/C5/U5) have no phosphate; the cap must be
# stripped so the residue matches the correct terminal template.
_RNA_5PRIME_PHOSPHATE = {'P', 'OP1', 'OP2', 'OP3'}
# AMBER14 5'-terminal templates exist for both RNA and DNA; both need the
# phosphate cap stripped to match the X5 template.
_RNA_NAMES = {'A', 'G', 'C', 'U', 'DA', 'DG', 'DC', 'DT'}


class OpenMMImplicitPropagator(OpenMMPropagator):

    def __init__(self, rc=None):
        super(OpenMMImplicitPropagator, self).__init__(rc)
        if not self.implicit_solvent:
            raise ValueError("OpenMMImplicitPropagator requires implicit_solvent=True")

        fixer = self._create_and_configure_fixer()
        modeller = self._create_modeller(fixer)
        self._finalize_system(modeller)
        self._minimize_basis_state()
        self._dump_simulated_topology()

    def _create_and_configure_fixer(self):
        fixer = pdbfixer.PDBFixer(self.topology_path)

        self._rename_rna_modifications(fixer)
        self._find_missing_components(fixer)
        self._print_diagnostics(fixer)
        self._remove_terminal_missing_residues(fixer)
        self._fix_nonstandard_residues(fixer)
        self._add_missing_components(fixer)

        return fixer

    def _rename_rna_modifications(self, fixer):
        """Rename modified RNA residues to their standard analogs before pdbfixer
        adds atoms/hydrogens, so the downstream AMBER14 templates can match them."""
        for res in fixer.topology.residues():
            if res.name in _RNA_MOD_MAP:
                res.name = _RNA_MOD_MAP[res.name]

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
        self._rename_his_for_amber14(modeller)
        self._prepare_implicit_solvent(modeller)
        self._fix_rna_5prime_termini(modeller)

        return modeller

    def _rename_his_for_amber14(self, modeller):
        """Rename HIS residues to the correct AMBER14 protonation variant.

        pdbfixer leaves histidine named 'HIS', but AMBER14 requires 'HIE'
        (ε-protonated), 'HID' (δ-protonated), or 'HIP' (doubly protonated).
        Detect the protonation state from the atoms added by pdbfixer and rename
        accordingly.  HIS residues that cannot be classified are renamed to HIE
        (neutral, ε-protonated — the most common form at pH 7).
        """
        for res in modeller.topology.residues():
            if res.name != 'HIS':
                continue
            atom_names = {a.name for a in res.atoms()}
            has_hd1 = 'HD1' in atom_names
            has_he2 = 'HE2' in atom_names
            if has_hd1 and has_he2:
                res.name = 'HIP'
            elif has_hd1:
                res.name = 'HID'
            else:
                res.name = 'HIE'

    def _prepare_implicit_solvent(self, modeller):
        modeller.deleteWater()
        self._remove_unsupported_residues(modeller)

    def _remove_unsupported_residues(self, modeller):
        """Delete any residue that has no template in the force field.

        After pdbfixer processing and HIS renaming, the only remaining
        unresolvable residues are metals, ions, and non-standard ligands.
        Checking against ff._templates is more robust than a hardcoded blocklist.
        """
        resolvable = set()
        for name in self.forcefield._templates:
            resolvable.add(name)
            for suf in ('5', '3', 'N'):
                if name.endswith(suf) and len(name) > 1:
                    resolvable.add(name[:-1])
            for pre in ('N', 'C'):
                if name.startswith(pre) and len(name) > 1:
                    resolvable.add(name[1:])

        to_delete = [r for r in modeller.topology.residues()
                     if r.name not in resolvable]
        if to_delete:
            names = {r.name for r in to_delete}
            print(f"Removing unsupported residues: {names}")
            modeller.delete(to_delete)

    def _fix_rna_5prime_termini(self, modeller):
        """Strip the 5'-phosphate cap (P/OP1/OP2/OP3) from the first residue of
        each RNA chain so it matches the AMBER14 X5 terminal template.

        PDB structures store the 5'-end with a phosphate group that AMBER14 X5
        templates do not include.  pdbfixer may also add OP3 as a 'missing' atom,
        producing a P+OP1+OP2+OP3+HO5' combination that matches no template.
        Removing P/OP1/OP2/OP3 leaves O5'+HO5'+rest, which exactly matches X5.
        """
        atoms_to_delete = []
        for chain in modeller.topology.chains():
            residues = list(chain.residues())
            if not residues or residues[0].name not in _RNA_NAMES:
                continue
            for atom in residues[0].atoms():
                if atom.name in _RNA_5PRIME_PHOSPHATE:
                    atoms_to_delete.append(atom)
        if atoms_to_delete:
            modeller.delete(atoms_to_delete)
    
    def _dump_simulated_topology(self):
        sim_root = Path(__file__).resolve().parents[1]
        out_path = sim_root / "simulated_topology.pdb"
        with open(out_path, "w") as f:
            PDBFile.writeFile(self.pdb.topology, self.pdb.positions, f, keepIds=True)
        print(f"Wrote simulated topology to {out_path}")

    def _minimize_basis_state(self):
        # Randomized structures can have severe inter-chain clashes (translations up
        # to 100 Å) that cause NaN coordinates within the first MD steps.  Minimizing
        # to convergence once at init removes those clashes so all new trajectories
        # start from a physically reasonable configuration.
        system = self._create_system()
        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction / picosecond,
            self.timestep * femtosecond,
        )
        platform, properties = self._get_platform(seg_id=0)
        sim = Simulation(self.pdb.topology, system, integrator, platform, properties)
        sim.context.setPositions(self.pdb.positions)
        sim.minimizeEnergy()
        self.pdb.positions = sim.context.getState(getPositions=True).getPositions()

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
    
    def _calculate_pcoord(self, segment_outdir, initial_pos, energy_data):
        if self.save_format == 'dcd':
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            md_top = mdtraj.Topology.from_openmm(self.pdb.topology)
            traj = mdtraj.load_dcd(dcd_path, top=md_top)
            all_positions = np.concatenate([initial_pos * 10, traj.xyz * 10])
        else:
            npz_path = os.path.join(segment_outdir, 'seg.npz')
            positions = np.load(npz_path)['positions']
            all_positions = np.concatenate([initial_pos * 10, positions * 10])

        return self.pcoord_calculator.calculate(all_positions, energy_data).reshape((-1, 1))
