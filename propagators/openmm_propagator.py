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
from openmm import Platform, LangevinMiddleIntegrator, XmlSerializer, Vec3
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole
from westpa.core.states import BasisState
from westpa.core.segment import Segment

from file_system.md_store.save_npz import save_openmm_npz
from propagators.base_propagator import BasePropagator


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
        # Cap on L-BFGS minimization iterations (0 = OpenMM default = until
        # convergence). For large freshly-solvated systems the default tolerance
        # can take 1000s of iters to reach everywhere → bound it (the random-
        # orientation separated association start needs only light clash relief
        # before constrained Langevin MD takes over). Applied to BOTH the
        # basis-state get_pcoord and every NEWTRAJ segment.
        self.minimize_iters      = int(config.get("minimize_iters", 0))
        self.implicit_solvent    = config.get("implicit_solvent", False)
        self.steps               = config["steps"]
        self.save_steps          = config["save_steps"]
        self.save_format         = self._get_save_format(["west", "openmm"])

        try:
            platform      = Platform.getPlatformByName("CUDA")
            self.num_gpus = int(config.get("num_gpus", 1))
            if self.num_gpus == -1:
                default       = platform.getPropertyDefaultValue("CudaDeviceIndex")
                self.num_gpus = default.count(",") + 1 if "," in default else 1
        except Exception:
            self.num_gpus = 1

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
            simulation = self._create_simulation(seg_id=0)
            simulation.context.setPositions(self.pdb.positions)
            simulation.minimizeEnergy(maxIterations=self.minimize_iters)
            openmm_state = simulation.context.getState(getPositions=True, getEnergy=True)
            
            positions = openmm_state.getPositions(asNumpy=True).value_in_unit(nanometer)
            positions = positions[np.newaxis, :, :] * 10.0  # (1, n_atoms, 3) Angstrom

            energy_u = openmm_state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            energy_data = {"energy_k": [0.0], "energy_u": [energy_u], "times": [0.0]}

            state.pcoord = self.pcoord_calculator.calculate(positions, energy_data).reshape((-1, 1))
            return
        raise NotImplementedError

    def _get_next_gpu_index(self, segment_id):
        return segment_id % self.num_gpus

    def _get_platform(self, seg_id):
        try:
            platform   = Platform.getPlatformByName("CUDA")
            gpu_index  = self._get_next_gpu_index(seg_id)
            properties = {"CudaDeviceIndex": str(gpu_index), "Precision": self.gpu_precision}
        except Exception:
            platform   = Platform.getPlatformByName("CPU")
            properties = {}
        return platform, properties

    def _create_system(self):
        raise NotImplementedError

    def _create_simulation(self, seg_id):
        platform, properties = self._get_platform(seg_id)
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
            npz = os.path.join(parent_outdir, "seg_restart.npz")
            if os.path.exists(npz):                       # new binary restart
                d = np.load(npz)
                box = d["box"].astype(float)
                simulation.context.setPeriodicBoxVectors(
                    Vec3(*box[0]) * nanometer, Vec3(*box[1]) * nanometer, Vec3(*box[2]) * nanometer)
                simulation.context.setPositions(d["positions"].astype(float) * nanometer)
                simulation.context.setVelocities(d["velocities"].astype(float) * (nanometer / picosecond))
            else:                                          # legacy XML restart (back-compat)
                with open(os.path.join(parent_outdir, "seg.xml")) as f:
                    simulation.context.setState(XmlSerializer.deserialize(f.read()))

        elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
            simulation.context.setPositions(self.pdb.positions)
            simulation.minimizeEnergy(maxIterations=self.minimize_iters)
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
        # Restart state (positions + velocities + box) for the NEXT iteration's
        # parent. DEFAULT = binary float32 `seg_restart.npz` (~10x smaller than the
        # full-system XML; portable across GPUs, unlike OpenMM .chk). Set
        # WCMD_RESTART_FORMAT=xml to keep writing legacy seg.xml. The xml READ path
        # (_init_segment_state) is always kept, so runs with old seg.xml parents
        # resume seamlessly and only new iterations switch to npz.
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, enforcePeriodicBox=False,
        )
        if os.environ.get("WCMD_RESTART_FORMAT", "npz").lower() == "xml":
            with open(os.path.join(segment_outdir, "seg.xml"), "w") as f:
                f.write(XmlSerializer.serialize(state))
        else:
            np.savez(
                os.path.join(segment_outdir, "seg_restart.npz"),
                positions=state.getPositions(asNumpy=True).value_in_unit(nanometer).astype(np.float32),
                velocities=state.getVelocities(asNumpy=True).value_in_unit(nanometer / picosecond).astype(np.float32),
                box=state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometer).astype(np.float32),
            )

    def _save_cg_seg(self, segment_outdir, positions_list, forces):
        """On-the-fly Cα-bead CG projection → cg_seg.npz, alongside the full-system
        seg.npz (kept for re-mapping flexibility). Bead pos = residue Cα; bead force
        = Σ residue atoms incl H (exact — full forces are in memory). Defensive:
        a projection error logs and is skipped, never breaking WE propagation."""
        try:
            from wcmd.training.cg_projector import build_ca_bead_map, project_to_ca_beads
            if getattr(self, "_ca_map", None) is None:
                self._ca_map = build_ca_bead_map(PDBFile(self.topology_path).topology)
                root = Path(segment_outdir).parents[2]   # sim_root/traj_segs/iter/seg
                np.savez(os.path.join(root, "cg_map.npz"), **self._ca_map)
            cm = self._ca_map
            cg_pos, cg_force = [], []
            for p, f in zip(positions_list, forces):
                cp, cf = project_to_ca_beads(np.asarray(p) * 10.0, np.asarray(f), cm)
                cg_pos.append(cp); cg_force.append(cf)
            np.savez(os.path.join(segment_outdir, "cg_seg.npz"),
                     cg_pos=np.stack(cg_pos), cg_force=np.stack(cg_force))
        except Exception as e:
            print(f"[cg] projection skipped ({type(e).__name__}: {e})", flush=True)

    def _calculate_pcoord(self, segment_outdir, initial_pos, energy_data):
        raise NotImplementedError

    def propagate(self, segments):
        starttime  = time.time()
        simulation = self._create_simulation(segments[0].seg_id)

        for segment in segments:
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
                # Save SOLUTE-only (protein incl H, no water/ions) — water is ~94%
                # of the system and is not needed for V_θ force-matching (the
                # protein-atom forces already contain the water-mediated part).
                # ~16× smaller seg.npz; pcoord (_calculate_pcoord) reads these
                # solute-order arrays directly. seg.xml (restart) keeps the full
                # system. Implicit/other propagators without solute indices save full.
                sol = getattr(self, "solute_atom_indices", None)
                if sol is not None:
                    save_openmm_npz(segment_outdir, times, [f[sol] for f in forces],
                                    energy_k, energy_u, [p[sol] for p in positions_list])
                else:
                    save_openmm_npz(segment_outdir, times, forces, energy_k, energy_u, positions_list)

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
