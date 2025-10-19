# Weighted Ensemble Simulation Framework (WESTPA + OpenMM / CGML)

This repository provides a modular framework for running **Weighted Ensemble (WE)** molecular dynamics simulations using **OpenMM**, **CGML**, and **TICA**-based progress coordinates.  
It integrates with **WESTPA** for trajectory management and adaptive sampling across multiple simulation backends (implicit, explicit, and coarse-grained models).

---

## 🧱 Repository Structure

# Weighted Ensemble Simulation Framework (WESTPA + OpenMM / CGML)

This repository provides a modular framework for running **Weighted Ensemble (WE)** molecular dynamics simulations using **OpenMM**, **CGML**, and **TICA**-based progress coordinates.  
It integrates with **WESTPA** for trajectory management and adaptive sampling across multiple simulation backends (implicit, explicit, and coarse-grained models).

---

## 🧱 Repository Structure
```yaml
west.cfg
propagators/
  ├── base_propagator.py
  ├── openmm_propagator.py
  ├── openmm_implicit_propagator.py
  ├── openmm_explicit_propagator.py
  └── cg_propagator.py
file_system/
  └── md_store/
      ├── save_dcd.py
      └── save_npz.py
progress_coordinate/
  ├── base_progress_coordinate.py
  └── tica_progress_coordinate.py
```
## ⚙️ Directory Overview

### **`propagators/`**
Contains classes responsible for advancing simulation trajectories (segments) during WESTPA propagation.

- **`base_propagator.py`** — Abstract base class defining a common interface for all propagators. Handles initialization, segment input/output, and interface with WESTPA.
- **`openmm_propagator.py`** — Shared utilities for OpenMM-based propagators. Defines setup of integrators, reporters, and checkpoint handling.
- **`openmm_implicit_propagator.py`** — Implements implicit-solvent OpenMM simulations. Uses GBn2 forcefields for fast, solvent-free sampling.
- **`openmm_explicit_propagator.py`** — Implements explicit-solvent OpenMM simulations. Includes barostat control, solvent boxes, and hydrogen constraints.
- **`cg_propagator.py`** — Integrates with CGML (coarse-grained machine-learned forcefields). Interfaces with pre-trained CGSchNet models.

### **`file_system/md_store/`**
Handles storage and serialization of trajectory and metadata.

- **`save_dcd.py`** — Writes simulation trajectories to **DCD format** for high-performance I/O and downstream analysis with visualization tools.
- **`save_npz.py`** — Writes trajectories as **NumPy NPZ** files, efficient for machine learning pipelines or progress coordinate extraction.

### **`progress_coordinate/`**
Defines the metrics (progress coordinates) that guide WESTPA’s resampling and binning.

- **`base_progress_coordinate.py`** — Abstract interface for progress coordinate calculators.
- **`tica_progress_coordinate.py`** — Implements **TICA (Time-lagged Independent Component Analysis)** progress coordinate. Loads pre-trained models and computes slow collective modes for binning and resampling.

---

## ⚙️ WESTPA Configuration (`west.cfg`)

The `west.cfg` file is the **master configuration file** controlling WESTPA’s runtime behavior, binning scheme, and propagator settings.

### **Main Sections**

#### `west.system`
Defines system dimensionality and binning behavior.

- `pcoord_ndim`: Number of dimensions in the progress coordinate (e.g., 2 for TICA [IC1, IC2]).
- `pcoord_len`: Number of data points per segment.
- `bins`: Defines adaptive bin mapper (e.g., `MABBinMapper`).
- `bin_target_counts`: Number of walkers per bin.

#### `west.propagation`
Controls how WESTPA propagates trajectories.

- `max_total_iterations`: Number of WE iterations to run.
- `max_run_wallclock`: Maximum allowed wall-clock time.
- `propagator`: Python path to the propagator class.
- `block_size`: Number of trajectories per propagation call.
- `gen_istates`: Whether to generate initial states automatically.

#### `west.data`
Controls how data is stored.

- `west_data_file`: HDF5 file for WE data (`west.h5`).
- `datasets`: Datasets to be tracked (e.g., `pcoord`).
- `data_refs`: File path patterns for trajectories and initial states.

#### `west.openmm` or `west.cg_prop`
Propagator-specific configuration (depends on simulation backend).  
Defines physical and computational parameters.

---

## 🧩 Example Configurations

### **Implicit Solvent (OpenMM)**

```yaml
propagator: propagators.openmm_implicit_propagator.OpenMMImplicitPropagator
implicit_solvent: true
num_gpus: 1
forcefield:
  - amber14-all.xml
  - implicit/gbn2.xml
temperature: 300.0
timestep: 4.0
steps: 1000
save_steps: 100
pcoord_calculator:
  class: progress_coordinate.tica_progress_coordinate.TICAProgressCoordinate
  model_path: /path/to/model.tica
  components: [0, 1]
```

### **Explicit Solvent (OpenMM)**
```yaml
propagator: propagators.openmm_explicit_propagator.OpenMMExplicitPropagator
implicit_solvent: false
forcefield:
  - amber14-all.xml
  - amber14/tip3pfb.xml
pressure: 1.0
barostatInterval: 25
```


### **Coarse-Grained Machine Learning (CGML)**
```yaml
propagator: propagators.cg_propagator.CGMLPropagator
cg_prop:
  cgschnet_path: /path/to/cgschnet/scripts
  model_path: /path/to/model
  topology_path: /path/to/topology.pdb
  pcoord_calculator:
    class: progress_coordinate.tica_progress_coordinate.TICAProgressCoordinate
    model_path: /path/to/model.tica
    components: [0, 1]
  timestep: 1
  steps: 1000
  save_steps: 100
```

## 🚀 Running the Simulation
#### Initialization 
```yaml
w_init --bstate "basis,1.0" --segs-per-state 1
```

#### Run Locally
```yaml
w_run
```

#### Run with MPI
```yaml
srun --mpi=pmi2 -n $SLURM_NTASKS w_run --work-manager mpi --n-workers=$N_WORKERS
```

## 🖥️ SLURM Deployment
For HPC execution:

- NERSC: `run_nersc.slurm`
- Environment setup: `env_mpi_nersc`

- DELTA: `run_delta.slurm`
- Environment setup: `env_mpi_delta`

Both scripts handle:

- MPI-based parallelism.
- WESTPA work manager configuration.
- GPU environment variables and paths.

## 📂 Output Structure
Simulation outputs are organized as:
```yaml
traj_segs/
  ├── 000001/
  │   ├── 000000/
  │   │   ├── seg.xml
  │   │   └── seg.npz   # or seg.dcd
  ├── 000002/
  │   └── ...

```
### Trajectory Formats
- NPZ: NumPy archive (xyz positions)
- DCD: Standard molecular dynamics trajectory file 

## 🔧 Modifying Parameters
You can modify the following parameters directly in `west.cfg` to tune the simulation:
| Parameter              | Description                           | Typical Range      |
| ---------------------- | ------------------------------------- | ------------------ |
| `pcoord_ndim`          | Dimensionality of progress coordinate | 1–3                |
| `nbins`                | Adaptive bin resolution               | `[7,7]` to `[9,9]` |
| `bin_target_counts`    | Number of walkers per bin             | 3–6                |
| `steps`                | MD steps per segment                  | 500–5000           |
| `save_steps`           | Frame save interval                   | 50–200             |
| `temperature`          | Simulation temperature (K)            | 300–310            |
| `timestep`             | Integration timestep (fs)             | 1–4                |
| `max_total_iterations` | Number of WE iterations               | 100–100000         |

## 🧠 Notes
- Implicit vs Explicit: Use `implicit_solvent`: true for GBn2 simulations and false for explicit solvent.
- Progress Coordinate: Ensure `model_path` points to a valid .tica model trained on your system.
- Forcefields: Always match implicit/explicit parameter sets to avoid inconsistent energy calculations.
- Trajectory Storage: Choose `.npz` for lightweight coarse-grained data and `.dcd` for atomistic output.


## 📜 References
- WESTPA Documentation: https://westpa.github.io/westpa/
- OpenMM: http://openmm.org
