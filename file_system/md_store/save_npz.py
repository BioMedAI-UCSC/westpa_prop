import numpy as np
import os

def save_openmm_npz(output_dir, times, forces, energy_k, energy_u, positions=None):
    data = {
        'times': np.array(times),
        'forces': np.array(forces),
        'energy_k': np.array(energy_k),
        'energy_u': np.array(energy_u)
    }
    if positions is not None:
        data['positions'] = np.array(positions)
    
    np.savez(os.path.join(output_dir, 'seg.npz'), **data)


def save_cg_npz(output_dir, epot, ekin, temp, time, pos, vel, forces=None):
    """Save a CG segment. `forces`, when provided, are the EXACT per-frame
    CG forces from the torchmd integrator at the saved positions (velocity-
    Verlet leaves system.forces = compute(system.pos) after each step, so
    forces[t] ↔ pos[t] with no recomputation). Needed for MS-CG force
    matching (V_θ supervision). torchmd force units: kcal/mol/Å."""
    data = dict(
        epot=np.array(epot),
        ekin=np.array(ekin),
        temp=np.array(temp),
        time=np.array(time),
        pos=np.array(pos),
        vel=np.array(vel),
    )
    if forces is not None:
        data['forces'] = np.array(forces)
    np.savez(os.path.join(output_dir, 'seg.npz'), **data)
