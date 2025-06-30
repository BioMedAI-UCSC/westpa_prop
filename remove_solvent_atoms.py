#!/usr/bin/env python

import mdtraj
import numpy as np
import glob
import os
import sys

# This script goes through all the finished OpenMM segments and removes the water atoms.
# It is in theory resumable as long as it didn't die exactly between the os.replace calls.

if not len(sys.argv) == 2:
    print(f"Usage: {__file__} reference_topology.pdb")
    sys.exit(1)

ref_topology_path = sys.argv[1]

# ref_topology_path = "/mnt/secondary/argon/benchmark_CA_only/chignolin/starting_pos_0/processed/starting_pos_0_processed.pdb"
ref_topology = mdtraj.load(ref_topology_path).topology
subset_atom_slice = ref_topology.select("not resname HOH")
subset_topology = ref_topology.subset(subset_atom_slice)

print("Ref:   ", ref_topology)
print("Subset:", subset_topology)

num_processed = 0
for seg in glob.glob("traj_segs/*/*"):
    seg_npz_path = os.path.join(seg, "seg.npz")
    seg_dcd_path = os.path.join(seg, "seg.dcd")

    if not os.path.exists(seg_npz_path):
        print("Skipping incomplete segment:", seg)
        continue
    seg_npz = np.load(seg_npz_path)
    if seg_npz["forces"].shape[1] == subset_topology.n_atoms:
        # Already processed
        continue
    assert seg_npz["forces"].shape[1] == ref_topology.n_atoms

    seg_dcd = mdtraj.load_dcd(seg_dcd_path, top=ref_topology)
    seg_dcd = seg_dcd.atom_slice(subset_atom_slice)

    seg_npz = dict(seg_npz)
    seg_npz["forces"] = seg_npz["forces"][:,subset_atom_slice,:]
    seg_dcd.save_dcd(seg_dcd_path + ".fix.dcd")
    np.savez(seg_npz_path + ".fix.npz", **seg_npz)
    os.replace(seg_dcd_path + ".fix.dcd", seg_dcd_path)
    os.replace(seg_npz_path + ".fix.npz", seg_npz_path)
    num_processed += 1

print(f"Processed {num_processed} segments")
