import torch
import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

import numpy as np
import time

import sys
import os
import json

from torchmd.forces import Forces
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmd.integrator import maxwell_boltzmann
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper

CGSCHNET_SCRIPTS = "/home/argon/openmm/cgschnet/cgschnet/cgschnet/scripts/"
if not CGSCHNET_SCRIPTS in sys.path:
    sys.path.append(CGSCHNET_SCRIPTS)

import simulate

# https://westpa.readthedocs.io/en/stable/documentation/core/westpa.core.propagators.html
# https://github.com/westpa/westpa/blob/b3afe209fcffc6238c1d2ec700059c7e30f2adca/src/westpa/core/propagators/executable.py#L688


MODEL_PATH = "/home/argon/Stuff/harmonic_net_2025.04.06/model_high_density_benchmark_CA_only_2025.04.03_mix1_s100_CA_lj_bondNull_angleNull_dihedralNull_cutoff2_seq6_harBAD_termBAD100__wd0_plateaulr5en4_0.1_0_1en3_1en7_bs4_chunk120"
TOPOLOGY_PATH = "/mnt/secondary/argon/benchmark_inputs_2025.04.03/chignolin_start_0.pdb"


class TestPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(TestPropagator, self).__init__(rc)

        device = "cuda"
        # input_path = os.path.expandvars('$WEST_SIM_ROOT/bstates/bstate0.pdb')
        input_path = TOPOLOGY_PATH
        use_box = False
        replicas = 1
        temperature = 300

        checkpoint_path = MODEL_PATH

        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint-best.pth")
        checkpoint_dir = os.path.dirname(checkpoint_path)

        prior_path = os.path.join(checkpoint_dir, "priors.yaml")
        assert os.path.exists(prior_path)
        prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")

        with open(f"{prior_params_path}", 'r') as file:
            prior_params = json.load(file)
        
        self.model = simulate.load_model(checkpoint_path, device, verbose=False)
        mol, embeddings = simulate.load_molecule(prior_path, prior_params, input_path, use_box=use_box, verbose=False)

        calcs = []
        calcs.append(simulate.build_calc(self.model, mol, embeddings, use_box=use_box, replicas=replicas,
                                temperature=temperature, device=device))
        
        forceterms = prior_params["forceterms"]
        exclusions = prior_params["exclusions"]

        #FIXME: There should be as many mols as replicas
        system, forces = simulate.make_system([mol], prior_path, calcs, device, forceterms, exclusions, replicas, temperature=temperature, new_ff=True)
        self.md_system = system
        self.md_forces = forces

        self.timestep = 1
        langevin_gamma = 1
        self.temperature = temperature

        self.integrator = Integrator(
            system,
            forces,
            self.timestep,
            device,
            gamma=langevin_gamma,
            T=self.temperature,
        )
        self.wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)

        self.steps = 1000
        self.save_steps = 100

        self.mol = mol


    def get_pcoord(self, state):
        print(state)

        # I belive we get a BasisState when it wants us to start from a file (or really whatever basis_state.auxref represents)
        # and an InitialState when it wants us to generate something?
        if isinstance(state, BasisState):
            print("state.auxref", state.auxref)
            return [0.0, 0.5, 1.0]
        # elif isinstance(state, InitialState):
        else:
            raise NotImplementedError
        raise NotImplementedError
    
    def gen_istate(self, basis_state, initial_state):
        raise NotImplementedError
    
    def propagate(self, segments):
        # print("self.basis_states", self.basis_states)
        # print("self.initial_states", self.initial_states)
        # # print(dir(segments[0]))
        # print(segments[0].data, segments[0].seg_id, segments[0].n_iter)
        # print(segments[0].parent_id, segments[0].initial_state_id)
        # print((self.rc.config['west', 'data', 'data_refs', 'segment'].format(segment=segments[0])))
        # print((self.rc.config['west', 'data', 'data_refs', 'basis_state']))

        # print("##", self.num_active, len(segments))

        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']
        
        

        replicas = 1
        

        for segment in segments:
            starttime = time.time()

            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                print("Seg continue")
                # From the executable propagator
                parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
                parent_outdir = (segment_pattern.format(segment=parent))
                parent_outdir = os.path.expandvars(parent_outdir)

                parent_traj = np.load(os.path.join(parent_outdir, "seg.npz"))
                velocities = np.expand_dims(parent_traj["vel"][-1], 0)
                velocities = torch.as_tensor(velocities)
                coords = np.expand_dims(parent_traj["pos"][-1], -1)
                # print(velocities.shape)
                # print(coords.shape)

            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                print("Seg from bstate")
                # Could also be InitialState.ISTATE_TYPE_START but we don't support that mode
                initial_state = self.initial_states[segment.initial_state_id]
                assert initial_state.istate_type == InitialState.ISTATE_TYPE_BASIS
                # TODO: Pay attention to what the basis state was
                # basis_state = self.basis_states[initial_state.basis_state_id]

                velocities = maxwell_boltzmann(self.md_forces.par.masses, self.temperature, replicas)
                coords = self.mol.coords.copy()


            segment_outdir = (segment_pattern.format(segment=segment))
            segment_outdir = os.path.expandvars(segment_outdir)
            os.makedirs(segment_outdir, exist_ok=True)

            # coords, velocities, ?box?
            # "Positions shape must be (natoms, 3, 1) or (natoms, 3, nreplicas)"
            # "Velocities shape must be (nreplicas, natoms, 3)"


            #TODO: Set seed

            #FIXME: Would probably make more sense to set these directly that deal with the weird shapes
            self.md_system.set_velocities(velocities)
            self.md_system.set_positions(coords)

            trajEpot = []
            trajEkin = []
            trajTemp = []
            trajTime = []
            trajPos  = []
            trajVel  = []
            for i in range(1, int(self.steps / self.save_steps) + 1):
                Ekin, Epot, T = self.integrator.step(niter=self.save_steps)
                self.wrapper.wrap(self.md_system.pos, self.md_system.box)
                currpos = self.md_system.pos.detach().cpu().numpy()
                currvel = self.md_system.vel.detach().cpu().numpy()

                trajEkin.append(Ekin)
                trajEpot.append(Epot)
                trajTemp.append(T)
                trajTime.append(np.repeat(i*self.timestep, replicas))
                trajPos.append(currpos)
                trajVel.append(currvel)

            i = 0 #FIXME: This assumes no replicas
            np.savez(os.path.join(segment_outdir, "seg.npz"),
                epot = np.array([f[i] for f in trajEpot]),
                ekin = np.array([f[i] for f in trajEkin]),
                temp = np.array([f[i] for f in trajTemp]),
                time = np.array([f[i] for f in trajTime]),
                pos  = np.array([f[i] for f in trajPos]),
                vel  = np.array([f[i] for f in trajVel]),
            )

            segment.status = Segment.SEG_STATUS_COMPLETE

            segment.walltime = time.time() - starttime
            # segment.cputime = rusage.ru_utime
        
        return segments
