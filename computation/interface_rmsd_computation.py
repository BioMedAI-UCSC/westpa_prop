import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


class InterfaceRMSDComputation(BaseComputation):
    """CAPRI-style binding-pose RMSD to a reference complex (validation only).

    mode="ligand":    superpose on selection_a (receptor), RMSD over selection_b
                      (ligand) -> L-RMSD; captures rigid-body pose, unlike a
                      globally-superposed RMSD.
    mode="interface": superpose on reference interface residues (both chains,
                      heavy-atom contact within interface_cutoff_angstrom), RMSD
                      over those same atoms -> i-RMSD.

    Returns (n_frames, 1) Angstrom. Needs a bound reference; never used as a pcoord.
    """

    requires_positions = True
    requires_energy = False

    def __init__(self, reference_pdb_path, selection_a, selection_b, mode="ligand",
                 atom_selection="name CA", interface_cutoff_angstrom=10.0,
                 reference_xml_path=None):
        ref = (mdtraj.load(reference_xml_path, top=reference_pdb_path)
               if reference_xml_path else mdtraj.load(reference_pdb_path))[0]
        self.ref = ref
        top = ref.topology
        if mode == "ligand":
            self.align_idx = top.select(f"({selection_a}) and ({atom_selection})")
            self.rmsd_idx = top.select(f"({selection_b}) and ({atom_selection})")
        elif mode == "interface":
            res = self._interface_residues(ref, selection_a, selection_b,
                                           interface_cutoff_angstrom)
            self.align_idx = self.rmsd_idx = top.select(
                f"({res}) and ({atom_selection})")
        else:
            raise ValueError(f"bad mode {mode!r}")
        if len(self.align_idx) == 0 or len(self.rmsd_idx) == 0:
            raise ValueError("empty align/rmsd selection")
        self.ref_rmsd_xyz = ref.xyz[0, self.rmsd_idx, :]

    def _interface_residues(self, ref, sel_a, sel_b, cutoff_A):
        top = ref.topology
        a = top.select(f"({sel_a}) and not element H")
        b = top.select(f"({sel_b}) and not element H")
        xyz = ref.xyz[0] * 10.0
        d = np.sqrt(((xyz[a][:, None] - xyz[b][None]) ** 2).sum(-1))
        ia, ib = np.where(d < cutoff_A)
        resids = {top.atom(int(a[i])).residue.index for i in ia}
        resids |= {top.atom(int(b[j])).residue.index for j in ib}
        if not resids:
            raise ValueError("no interface residues found in reference")
        return "resid " + " ".join(str(r) for r in sorted(resids))

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        traj = mdtraj.Trajectory(data / 10.0, self.ref.topology)
        traj.superpose(self.ref, atom_indices=self.align_idx)
        diff = traj.xyz[:, self.rmsd_idx, :] - self.ref_rmsd_xyz[None]
        out = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1)) * 10.0
        return out.reshape(-1, 1).astype(np.float32)
