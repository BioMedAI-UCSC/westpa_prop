import os
import sys
import random
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from openmm.app import PDBFile, ForceField, Simulation
from openmm import Platform, LangevinMiddleIntegrator, XmlSerializer
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole
from westpa.core.states import BasisState
from westpa.core.segment import Segment

from file_system.md_store.save_npz import save_openmm_npz
from propagators.base_propagator import BasePropagator


# Module-level caches keyed by PID survive across WESTPA tasks within the same
# forked worker process. Instance attributes (self._cached_sim) do NOT, because
# WESTPA pickles the propagator into the task queue and reconstructs it on the
# receiving end — every task call would otherwise rebuild the OpenMM Context.
_WORKER_SIM_CACHE = {}    # pid -> Simulation (1 per worker, bound to 1 GPU)
_ANNOUNCED_PIDS   = set() # pids that have already printed platform info


class OpenMMPropagator(BasePropagator):

    def __init__(self, rc=None):
        super().__init__(rc)

    def _load_config(self):
        config = self.rc.config["west"]["openmm"]

        self.temperature         = float(config.get("temperature", 300.0))
        self.timestep            = float(config.get("timestep", 2.0))
        self.friction            = float(config.get("friction", 1.0))
        self.pressure            = float(config.get("pressure", 1.0))
        self.barostatInterval    = int(config.get("barostatInterval", 25))
        self.constraintTolerance = float(config.get("constraintTolerance", 1e-6))
        self.hydrogenMass        = float(config.get("hydrogenMass", 1.5))
        self.implicit_solvent    = config.get("implicit_solvent", False)
        self.steps               = config["steps"]
        self.save_steps          = config["save_steps"]
        self.save_format         = self._get_save_format(["west", "openmm"])

        # Set num_gpus WITHOUT touching the CUDA Platform here. _load_config
        # runs in the parent process before WESTPA forks its workers; any CUDA
        # init in the parent — including merely fetching the CUDA Platform
        # object — can leave forked children unable to create CUDA contexts
        # (CUDA_ERROR_NOT_INITIALIZED). Auto-detection (num_gpus: -1) is only
        # honored when explicitly requested, and only then are we forced to
        # touch CUDA pre-fork.
        self.num_gpus = int(config.get("num_gpus", 1))
        if self.num_gpus == -1:
            platform      = Platform.getPlatformByName("CUDA")
            default       = platform.getPropertyDefaultValue("CudaDeviceIndex")
            self.num_gpus = default.count(",") + 1 if "," in default else 1

        self.gpu_precision    = config.get("gpu_precision", "single")
        self.topology_path    = os.path.expandvars(config["topology_path"])
        self.forcefield_files = config["forcefield"]
        self.pdb              = PDBFile(self.topology_path)
        self.forcefield       = ForceField(*self.forcefield_files)
        self.nonbondedMethod  = None

    def _get_pcoord_config(self):
        return self.rc.config["west"]["openmm"].get("pcoord_calculator")

    def _get_recorded_configs(self):
        return self.rc.config["west"]["openmm"].get("recorded_calculators", [])

    def get_pcoord(self, state):
        if isinstance(state, BasisState):
            # CPU-only here. get_pcoord runs in the parent w_init process,
            # before WESTPA forks workers. Any CUDA init in the parent leaves
            # forked children unable to create CUDA contexts (CUDA_ERROR_NOT_
            # INITIALIZED). Use CPU for this one-shot pcoord eval.
            simulation = self._create_cpu_simulation()
            simulation.context.setPositions(self.pdb.positions)
            simulation.minimizeEnergy()
            openmm_state = simulation.context.getState(getPositions=True, getEnergy=True)
            
            positions = openmm_state.getPositions(asNumpy=True).value_in_unit(nanometer)
            positions = positions[np.newaxis, :, :] * 10.0  # (1, n_atoms, 3) Angstrom

            energy_u = openmm_state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            energy_data = {"energy_k": [0.0], "energy_u": [energy_u], "times": [0.0]}

            state.pcoord = self.pcoord_calculator.calculate(positions, energy_data).reshape((-1, 1))
            return
        raise NotImplementedError

    def _get_next_gpu_index(self, segment_id):
        # Bind each WESTPA worker to one GPU. The work manager dispatches
        # segments from a shared queue, so seg_id mod num_gpus does NOT pin
        # workers — multiple workers routinely collide on a single device.
        # Process work manager: exports WM_PROCESS_INDEX per forked worker
        # (westpa/work_managers/processes.py:132 + environment.py:24,43).
        # Thread work manager: names threads "worker-{i}" (threads.py:65).
        # Both: Worker N -> GPU N.
        wm_idx = os.environ.get('WM_PROCESS_INDEX')
        if wm_idx is not None:
            return int(wm_idx) % self.num_gpus
        import threading
        tname = threading.current_thread().name
        if tname.startswith('worker-'):
            try:
                return int(tname.split('-', 1)[1]) % self.num_gpus
            except (ValueError, IndexError):
                pass
        return segment_id % self.num_gpus

    def _get_platform(self, seg_id):
        # Fail loud when CUDA was requested but missing. The previous version
        # silently fell back to CPU, which meant configs asking for `num_gpus`
        # would run entirely on CPU with the GPUs idle and no indication in
        # any log. If you really want CPU, set num_gpus: 0 in the config.
        if self.num_gpus > 0:
            try:
                platform = Platform.getPlatformByName("CUDA")
            except Exception as e:
                avail = [Platform.getPlatform(i).getName()
                         for i in range(Platform.getNumPlatforms())]
                raise RuntimeError(
                    f"OpenMM CUDA platform requested (num_gpus={self.num_gpus}) "
                    f"but unavailable: {e}. Available platforms: {avail}. "
                    f"Install openmm with a CUDA build (e.g. "
                    f"`mamba install -c conda-forge openmm cuda-version=12.9`) "
                    f"or set num_gpus: 0 in west.cfg to run on CPU."
                )
            gpu_index  = self._get_next_gpu_index(seg_id)
            properties = {"CudaDeviceIndex": str(gpu_index),
                          "Precision":       self.gpu_precision}
        else:
            platform   = Platform.getPlatformByName("CPU")
            properties = {}
        return platform, properties

    def _create_system(self):
        raise NotImplementedError

    def _create_cpu_simulation(self):
        # Force-CPU simulation for parent-process work (basis state pcoord,
        # one-time minimization). Bypasses _get_platform / _get_next_gpu_index
        # so the parent never initializes CUDA — see _minimize_basis_state's
        # comment for why. Worker processes still get CUDA via _create_simulation.
        system     = self._create_system()
        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction / picosecond,
            self.timestep * femtosecond,
        )
        integrator.setConstraintTolerance(self.constraintTolerance)
        integrator.setRandomNumberSeed(random.randint(1, 1_000_000))
        platform   = Platform.getPlatformByName("CPU")
        print(f"[propagator] parent pid={os.getpid()} platform=CPU (one-shot setup)",
              flush=True)
        return Simulation(self.pdb.topology, system, integrator, platform)

    def _create_simulation(self, seg_id):
        platform, properties = self._get_platform(seg_id)
        # Announce platform once per worker so it's unmistakable in w_run.log
        # whether MD is on GPU or CPU. Key on (pid, tid) so the threads work
        # manager (all workers share one pid, distinct tids) and the processes
        # work manager (distinct pids, shared tid) both announce each worker.
        import threading
        wid = (os.getpid(), threading.get_ident())
        if wid not in _ANNOUNCED_PIDS:
            _ANNOUNCED_PIDS.add(wid)
            wm_idx = os.environ.get('WM_PROCESS_INDEX', threading.current_thread().name)
            pname  = platform.getName()
            extra  = (f" CudaDeviceIndex={properties.get('CudaDeviceIndex','?')}"
                      f" Precision={properties.get('Precision','-')}") if pname == "CUDA" else ""
            print(f"[propagator] worker={wm_idx} pid={wid[0]} tid={wid[1]} "
                  f"platform={pname}{extra}", flush=True)
        system     = self._create_system()
        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction / picosecond,
            self.timestep * femtosecond,
        )
        integrator.setConstraintTolerance(self.constraintTolerance)
        integrator.setRandomNumberSeed(random.randint(1, 1_000_000))
        return Simulation(self.pdb.topology, system, integrator, platform, properties)

    def _init_segment_state(self, simulation, segment):
        if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
            parent_outdir = self._get_parent_outdir(segment)
            with open(os.path.join(parent_outdir, "seg.xml")) as f:
                simulation.context.setState(XmlSerializer.deserialize(f.read()))

        elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
            simulation.context.setPositions(self.pdb.positions)
            simulation.minimizeEnergy()
            simulation.context.setVelocitiesToTemperature(self.temperature)

        else:
            raise ValueError(f"Unsupported initpoint_type: {segment.initpoint_type}")

        return self._get_state_and_energy(simulation)

    def _get_state_and_energy(self, simulation):
        state    = simulation.context.getState(getPositions=True, getEnergy=True)
        pos      = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        energy_u = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        energy_k = state.getKineticEnergy().value_in_unit(kilojoule_per_mole)
        time     = state.getTime().value_in_unit(picosecond)
        return np.array([pos]), {"energy_k": energy_k, "energy_u": energy_u, "times": time}


    def _setup_reporters(self, simulation, segment_outdir):
        raise NotImplementedError

    def _run_simulation(self, simulation):
        assert self.steps % self.save_steps == 0
        times, forces, energy_k, energy_u, positions_list = [], [], [], [], []

        for _ in range(self.steps // self.save_steps):
            simulation.step(self.save_steps)
            state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
            positions_list.append(state.getPositions(asNumpy=True).value_in_unit(nanometer))
            forces.append(state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole / nanometer))
            times.append(state.getTime().value_in_unit(picosecond))
            energy_k.append(state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
            energy_u.append(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))

        return times, forces, energy_k, energy_u, positions_list

    def _save_final_state(self, simulation, segment_outdir):
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getForces=True,
            getEnergy=True, enforcePeriodicBox=False,
        )
        with open(os.path.join(segment_outdir, "seg.xml"), "w") as f:
            f.write(XmlSerializer.serialize(state))

    def _calculate_pcoord(self, segment_outdir, initial_pos, energy_data):
        raise NotImplementedError

    def propagate(self, segments):
        starttime = time.time()
        # Cache the Simulation per worker via module-level dict keyed by
        # (pid, tid). Per-instance caching (self._cached_sim) does NOT work
        # because WESTPA's process work manager pickles bound methods into the
        # task queue and reconstructs the propagator on the receiving end
        # every task call, so per-instance state is lost between segments.
        # Keying on (pid, tid) makes the cache correct for BOTH the processes
        # work manager (distinct pids, one tid each) AND the threads work
        # manager (one shared pid, distinct tids per worker). With this cache
        # the CUDA context + kernel compile (~10s) is paid once per worker,
        # then every subsequent segment reuses it.
        import threading
        wid = (os.getpid(), threading.get_ident())
        if wid not in _WORKER_SIM_CACHE:
            _WORKER_SIM_CACHE[wid] = self._create_simulation(segments[0].seg_id)
        simulation = _WORKER_SIM_CACHE[wid]

        for segment in segments:
            simulation.integrator.setRandomNumberSeed(random.randint(1, 1_000_000))
            segment_outdir = self._get_segment_outdir(segment)
            os.makedirs(segment_outdir, exist_ok=True)

            initial_pos, initial_energy = self._init_segment_state(simulation, segment)
            self._setup_reporters(simulation, segment_outdir)
            times, forces, energy_k, energy_u, positions_list = self._run_simulation(simulation)

            energy_data = {
                "energy_k": [initial_energy["energy_k"]] + energy_k,
                "energy_u": [initial_energy["energy_u"]] + energy_u,
                "times":    [initial_energy["times"]]    + times,
            }

            if self.save_format == "npz":
                save_openmm_npz(segment_outdir, times, forces, energy_k, energy_u, positions_list)
            else:
                save_openmm_npz(segment_outdir, times, forces, energy_k, energy_u)

            self._save_final_state(simulation, segment_outdir)
            segment.pcoord = self._calculate_pcoord(segment_outdir, initial_pos, energy_data)

            all_positions_ang = np.concatenate(
                [initial_pos * 10.0, np.array(positions_list) * 10.0], axis=0
            )
            self._run_recorded(
                positions=all_positions_ang,
                energy_data=energy_data,
                segment_outdir=segment_outdir,
                n_iter=segment.n_iter,
                seg_id=segment.seg_id,
            )

            self._finalize_segment(segment, starttime)

        self._print_completion(len(segments), time.time() - starttime)
        return segments
