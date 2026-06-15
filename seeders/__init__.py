"""Basis-state seeders for protein-protein association WE.

Two entry points:
    randomize_seeder.py - VALIDATION mode: perturb a known complex apart.
    blind_seeder.py     - BLIND mode: random encounter geometries, no native pose.

Both share seed_common.py and emit a multi-state bstates.txt + bstates/*.pdb
that w_init can consume directly:

    w_init --bstates-from bstates.txt --segs-per-state N
"""
