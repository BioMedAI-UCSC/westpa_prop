from openmm.unit import nanometer
import numpy as np
import mdtraj


class BaseDCDReporter:
    
    def __init__(self, file, reportInterval, enforcePeriodicBox=False, append=False):
        self._file = file
        self._reportInterval = int(reportInterval)
        self._enforce = enforcePeriodicBox
        self._append = append
        self._dcd = None
    
    def describeNextReport(self, simulation):
        return (self._reportInterval, True, False, False, False, self._enforce)
    
    def _ensure_open(self, natoms):
        if self._dcd is None:
            self._dcd = mdtraj.formats.DCDTrajectoryFile(
                self._file, 'a' if self._append else 'w', force_overwrite=True
            )
    
    def _get_positions(self, state):
        raise NotImplementedError
    
    def report(self, simulation, state):
        xyz = self._get_positions(state)
        self._ensure_open(xyz.shape[1])
        self._dcd.write(xyz)
    
    def __del__(self):
        try:
            if self._dcd is not None:
                self._dcd.close()
        except Exception:
            pass


class FullDCDReporter(BaseDCDReporter):
    
    def _get_positions(self, state):
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        return pos_nm[np.newaxis, :, :]


class SoluteDCDReporter(BaseDCDReporter):
    
    def __init__(self, file, reportInterval, atom_indices, enforcePeriodicBox=False, append=False):
        super().__init__(file, reportInterval, enforcePeriodicBox, append)
        self._atom_indices = np.asarray(atom_indices, dtype=int)
    
    def _get_positions(self, state):
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        sel = pos_nm[self._atom_indices]
        return sel[np.newaxis, :, :]


def write_dcd_from_positions(filepath, positions, topology=None):
    if positions.ndim == 2:
        positions = positions[np.newaxis, :, :]
    
    with mdtraj.formats.DCDTrajectoryFile(filepath, 'w', force_overwrite=True) as dcd:
        dcd.write(positions)


def get_solute_indices(topology):
    solvent_resnames = {
        "HOH", "WAT", "TIP3", "TIP4", "TIP5", "SPC", "SOL",
        "NA", "K", "CL", "CL-", "BR", "I", "LI", "RB", "CS",
        "MG", "CA", "ZN", "MN", "FE", "CU", "SR", "CD"
    }
    
    solute_indices = []
    for atom in topology.atoms:
        r = atom.residue
        is_water = getattr(r, "is_water", False)
        is_mono = (len(list(r.atoms)) == 1)
        is_solvent = (r.name.upper() in solvent_resnames)
        
        if not (is_water or is_mono or is_solvent):
            solute_indices.append(atom.index)
    
    if len(solute_indices) == 0:
        raise RuntimeError("No solute atoms found")
    
    return np.array(solute_indices, dtype=int)
