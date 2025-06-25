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
Configure basis states in the config topology in the west.cfg 

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
w_init -r west_cg.cfg --bstate-file bstates/bstates.txt && w_run
```

### Using OpenMM
```bash
# Initialize with OpenMM configuration
w_init -r west_openmm.cfg --bstate-file bstates/bstates.txt 

# Run with OpenMM configuration
w_run -r west_openmm.cfg
```

### Run in Parallel
```
w_run -r west_openmm.cfg --work-manager processes --n-workers $NUM_WORKERS
```

## Configuration Files

- `west_cg.cfg` - Configuration for CGSchNet-based simulations
- `west_openmm.cfg` - Configuration for OpenMM-based simulations

## Customizing Your Simulations

### CGSchNet Configuration
Edit the `west_cg.cfg` file to modify parameters in the `cg_prop` section.

### OpenMM Configuration
Edit the `west_openmm.cfg` file to modify:
- Force fields in the `forcefield` list
- Temperature, timestep, and other simulation parameters
- Progress coordinate calculation


Here’s a revised and detailed section to add to your `README.md`, specifically for running **OpenMM-based WESTPA simulations on Delta**:

---

## Running on Delta (OpenMM Only)

To run WESTPA simulations using the OpenMM propagator on [Delta](https://docs.delta.ncsa.illinois.edu/), follow these steps:

### 1. Clone the Repository

Clone this repository directly onto Delta (e.g., in your `/home` or project directory):

### 2. Configure `west_openmm.cfg`

Update the `west_openmm.cfg` file with the correct paths to your:

* **Topology and coordinate files**
* **Basis state files**
* **TICA model path**

Make sure all paths reflect the Delta filesystem (e.g., `/scratch`, `/work/hdd`, or `/projects`).

### 3. Use `/work/hdd` for `traj_segs`

Output data such as `traj_segs` should be written to `/work/hdd` (a high-throughput scratch filesystem on Delta).
This is **automatically set** by the Slurm script (`run_delta.slurm`), but double-check that all file paths point to `/work/hdd` or another performance-optimized location.

### 4. Edit `env_mpi.sh`

Modify the `env_mpi.sh` script to your account details.

### 5. Configure `run_delta.slurm`

This Slurm submission script launches your WESTPA job across GPU nodes. Adjust the following parameters:

```bash
#SBATCH -A bbpa-delta-gpu           # Allocation/project name (use your group's project code)
#SBATCH --partition=gpuA40x4        # GPU partition on Delta
#SBATCH -t 38:00:00                 # Wall time (HH:MM:SS)
#SBATCH --cpus-per-task=2           # Number of CPU threads per task (match OMP_NUM_THREADS)
#SBATCH -N 45                       # Total number of nodes to use
#SBATCH --ntasks-per-node=4         # Number of MPI tasks per node (use 1 per GPU)
#SBATCH --gpus-per-task=1           # GPUs per task
#SBATCH --gpus-per-node=4           # Total GPUs per node
#SBATCH --job-name=westpadeltabba   # Job name
#SBATCH --output=slurm.out          # Output log file
#SBATCH --mail-user=awaghili@ucsc.edu # Email for notifications
```

### 6. Initialize the Simulation

Run the initialization command with your OpenMM config:

```bash
w_init -r west_openmm.cfg --bstate-file bstates/bstates.txt
```

### 7. Submit the Job

Submit the simulation using Slurm:

```bash
sbatch run_delta.slurm
```

### 8. Monitor the Job

Use standard Slurm tools to check status:

```bash
squeue -u $USER
scontrol show job <JOBID>
```

---

