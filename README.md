# WESTPA Propagator Extensions

This repository provides custom propagator implementations for the WESTPA (Weighted Ensemble Simulation Tool with Parallel Analysis) framework:

1. **CGSchNet Propagator** - A coarse-grained molecular dynamics propagator using neural network potentials through TorchMD
2. **OpenMM Propagator** - An all-atom molecular dynamics propagator using OpenMM

Both propagators use TICA (Time-lagged Independent Component Analysis) for progress coordinate calculation.

**Note:** This code does not work with WESTPA's parallel work managers!

## Prerequisites

### Common Requirements
- Python 3.6+
- WESTPA
- NumPy

### CGSchNet Requirements
- PyTorch
- TorchMD
- CGSchNet (path specified in west.cfg)

### OpenMM Requirements
```bash
pip install openmm mdtraj
```

## Setting Up Your System

### Basis States
1. For CGSchNet: Configure basis states in `bstates/bstates.txt`
2. For OpenMM: Provide a properly formatted PDB file at `bstates/bstate.pdb`

### TICA Model
Ensure your TICA model is properly configured in your configuration files or set the environment variable:

```bash
# For example, using the default model location:
export TICA_MODEL_PATH="/media/DATA_18_TB_1/andy/benchmark_cache/chignolin_300K.tica"
```

> **Note**: The default TICA model for this project is located at `/media/DATA_18_TB_1/andy/benchmark_cache/chignolin_300K.tica`. If using a different model, update the path accordingly.

## Running Simulations

### Using CGSchNet (Default)
```bash
w_init --bstate-file bstates/bstates.txt && w_run
```

### Using OpenMM
```bash
# Initialize with OpenMM configuration
w_init -r west_openmm.cfg --bstate-file bstates/bstates.txt 

# Run with OpenMM configuration
w_run -r west_openmm.cfg
```

## Configuration Files

- `west.cfg` - Configuration for CGSchNet-based simulations
- `west_openmm.cfg` - Configuration for OpenMM-based simulations

## Customizing Your Simulations

### CGSchNet Configuration
Edit the `west.cfg` file to modify parameters in the `cg_prop` section.

### OpenMM Configuration
Edit the `west_openmm.cfg` file to modify:
- Force fields in the `forcefield` list
- Temperature, timestep, and other simulation parameters
- Progress coordinate calculation
