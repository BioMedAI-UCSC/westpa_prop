from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

from file_system.md_store.save_npz import save_cg_npz
from file_system.md_store.save_dcd import write_dcd_from_positions

import torch
import numpy as np
import time
import sys
import os
import json

from torchmd.integrator import maxwell_boltzmann


class CGMLPropagator(BasePropagator):
    
    def __init__(self, rc=None):
        super(CGPropagator, self).__init__(rc)
    
    def _load_config(self):
        device = "cuda"
        
        cgschnet_path = self.rc.config.require(['west', 'cg_prop', 'cgschnet_path'])
        if cgschnet_path not in sys.path:
            sys.path.append(cgschnet_path)
        import simulate
        
        checkpoint_path = self.rc.config.require(['west', 'cg_prop', 'model_path'])
        topology_path = self.rc.config.require(['west', 'cg_prop', 'topology_path'])
        
        self.replicas = self.rc.config.get_typed(['west', 'propagation', 'block_size'], int, 1)
        use_box = self.rc.config.get_typed(['west', 'cg_prop', 'use_box'], bool, False)
        self.temperature = self.rc.config.get_typed(['west', 'cg_prop', 'temperature'], int, 300)
        self.steps = self.rc.config.require(['west', 'cg_prop', 'steps'], int)
        self.save_steps = self.rc.config.require(['west', 'cg_prop', 'save_steps'], int)
        self.timestep = self.rc.config.require(['west', 'cg_prop', 'timestep'], int)
        self.save_format = self._get_save_format(['west', 'cg_prop'])
        
        assert not use_box
        
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint-best.pth")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        prior_path = os.path.join(checkpoint_dir, "priors.yaml")
        prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")
        
        with open(prior_params_path, 'r') as f:
            prior_params = json.load(f)
        
        self.model = simulate.load_model(checkpoint_path, device, verbose=False)
        mol, embeddings = simulate.load_molecule(prior_path, prior_params, topology_path,
                                                 use_box=use_box, verbose=False)
        
        calcs = [simulate.build_calc(self.model, mol, embeddings, use_box=use_box,
                                     replicas=self.replicas, temperature=self.temperature,
                                     device=device)]
        
        forceterms = prior_params["forceterms"]
        exclusions = prior_params["exclusions"]
        
        system, forces = simulate.make_system([mol], prior_path, calcs, device, forceterms,
                                              exclusions, self.replicas,
                                              temperature=self.temperature, new_ff=True)
        self.md_system = system
        self.md_forces = forces
        
        from torchmd.integrator import Integrator
        from torchmd.wrapper import Wrapper
        
        self.integrator = Integrator(system, forces, self.timestep, device,
                                    gamma=1, T=self.temperature)
        self.wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)
        self.mol = mol
    
    def _get_pcoord_config(self):
        return self.rc.config['west', 'cg_prop'].get('pcoord_calculator')
    
    def get_pcoord(self, state):
        if isinstance(state, BasisState):
            state.pcoord = self.pcoord_calculator.calculate(
                np.transpose(self.mol.coords, (2, 0, 1))
            )
            return
        elif isinstance(state, InitialState):
            raise NotImplementedError
        raise NotImplementedError
    
    def propagate(self, segments):
        starttime = time.time()
        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']
        
        assert len(segments) <= self.replicas
        
        parent_pos = []
        for i, segment in enumerate(segments):
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                parent_outdir = self._get_parent_outdir(segment)
                parent_traj = np.load(os.path.join(parent_outdir, "seg.npz"))
                coords = torch.as_tensor(parent_traj["pos"][-1])
                velocities = torch.as_tensor(parent_traj["vel"][-1])
            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                initial_state = self.initial_states[segment.initial_state_id]
                assert initial_state.istate_type == InitialState.ISTATE_TYPE_BASIS
                coords = torch.as_tensor(self.mol.coords.reshape(-1, 3))
                velocities = maxwell_boltzmann(self.md_forces.par.masses, self.temperature, 1)[0]
            
            self.md_system.pos[i][:] = coords
            self.md_system.vel[i][:] = velocities
            parent_pos.append(coords.numpy())
        
        Epot = self.md_forces.compute(self.md_system.pos, self.md_system.box, self.md_system.forces)
        
        trajEpot, trajEkin, trajTemp, trajTime, trajPos, trajVel = [], [], [], [], [], []
        
        for i in range(1, int(self.steps / self.save_steps) + 1):
            Ekin, Epot, T = self.integrator.step(niter=self.save_steps)
            self.wrapper.wrap(self.md_system.pos, self.md_system.box)
            currpos = self.md_system.pos.detach().cpu().numpy()
            currvel = self.md_system.vel.detach().cpu().numpy()
            
            trajEkin.append(Ekin)
            trajEpot.append(Epot)
            trajTemp.append(T)
            trajTime.append(np.repeat(i * self.timestep, self.replicas))
            trajPos.append(currpos)
            trajVel.append(currvel)
        
        for i, segment in enumerate(segments):
            segment_outdir = self._get_segment_outdir(segment)
            os.makedirs(segment_outdir, exist_ok=True)
            
            if self.save_format == 'npz':
                save_cg_npz(
                    segment_outdir,
                    epot=[f[i] for f in trajEpot],
                    ekin=[f[i] for f in trajEkin],
                    temp=[f[i] for f in trajTemp],
                    time=[f[i] for f in trajTime],
                    pos=[f[i] for f in trajPos],
                    vel=[f[i] for f in trajVel]
                )
            else:
                dcd_path = os.path.join(segment_outdir, 'seg.dcd')
                positions_nm = np.array([f[i] for f in trajPos]) / 10.0
                write_dcd_from_positions(dcd_path, positions_nm)
            
            pcoord_pos = np.array([parent_pos[i]] + [f[i] for f in trajPos])
            segment.pcoord = self.pcoord_calculator.calculate(pcoord_pos)
            self._finalize_segment(segment, starttime)
        
        self._print_completion(len(segments), time.time() - starttime)
        return segments
