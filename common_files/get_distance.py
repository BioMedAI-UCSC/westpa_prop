import mdtraj
import numpy as np
import json
import os

def calculate_rmsd(base_traj, traj):
    ca_atoms = base_traj.topology.select("name CA")
    distances = mdtraj.rmsd(traj, base_traj, atom_indices=ca_atoms)
    return np.array(distances)

def build_distance_array_rmsd(base_traj, parent_traj, traj):
    dist_parent = calculate_rmsd(base_traj, parent_traj)
    dist_traj = calculate_rmsd(base_traj, traj)
    dist = np.append(dist_parent, dist_traj)
    d_arr = np.asarray(dist)
    return d_arr

def read_native_contact_pairs(filename):
     return json.load(open(filename, mode="r", encoding="utf-8"))

def calculate_percent_native(traj, native_contact_pairs, cutoff_dist):
    percent_native = np.sum(mdtraj.compute_distances(traj, native_contact_pairs)<cutoff_dist,axis=1)/len(native_contact_pairs)
    return np.array(percent_native)

def build_distance_array_percent_native(parent_traj, traj, contact_pairs_info):
    dist_parent = calculate_percent_native(parent_traj, contact_pairs_info["native_pairs"], contact_pairs_info["cutoff"])
    dist_traj = calculate_percent_native(traj, contact_pairs_info["native_pairs"], contact_pairs_info["cutoff"])
    dist = np.append(dist_parent, dist_traj)
    d_arr = np.asarray(dist)
    return d_arr

def traj_from_numpy(top_traj, seg_npz):
    seg_npz_data = np.load(seg_npz)
    n_frames = len(seg_npz_data["times"])
    unitcell_lengths = np.repeat(top_traj.unitcell_lengths, n_frames, axis=0)
    unitcell_angles = np.repeat(top_traj.unitcell_angles, n_frames, axis=0)
    return mdtraj.Trajectory(seg_npz_data["positions"], top_traj.topology, seg_npz_data["times"], unitcell_lengths, unitcell_angles)

if __name__ == "__main__":
    # Generate the WESTPA distances file for a segment

    # # For RMSD
    base_traj = mdtraj.load(os.path.join(os.environ["WEST_SIM_ROOT"], "bstates/bstate.xml"),
                            top=os.path.join(os.environ["WEST_SIM_ROOT"], "bstates/bstate.pdb"))
    parent_traj = mdtraj.load('parent.xml', top='bstate.pdb')
    traj = traj_from_numpy(parent_traj, "seg.npz")

    d_arr = build_distance_array_rmsd(base_traj, parent_traj, traj)

    # # For Percent Native
    # contact_pairs_info = read_native_contact_pairs(os.path.join(os.environ["WEST_SIM_ROOT"], "contact_pairs.json"))
    # parent_traj = mdtraj.load('parent.xml', top='bstate.pdb')
    # traj = traj_from_numpy(parent_traj, "seg.npz")

    # d_arr = build_distance_array_percent_native(parent_traj, traj, contact_pairs_info)

    # # For RMSD + Q
    # base_traj = mdtraj.load(os.path.join(os.environ["WEST_SIM_ROOT"], "bstates/bstate.xml"),
    #                         top=os.path.join(os.environ["WEST_SIM_ROOT"], "bstates/bstate.pdb"))
    # parent_traj = mdtraj.load('parent.xml', top='bstate.pdb')
    # traj = traj_from_numpy(parent_traj, "seg.npz")

    # contact_pairs_info = read_native_contact_pairs(os.path.join(os.environ["WEST_SIM_ROOT"], "contact_pairs.json"))

    # d_arr = np.transpose(np.array([
    #     build_distance_array_rmsd(base_traj, parent_traj, traj),
    #     build_distance_array_percent_native(parent_traj, traj, contact_pairs_info)
    # ]))

    print("Saving:", d_arr)
    np.savetxt("dist.dat", d_arr)