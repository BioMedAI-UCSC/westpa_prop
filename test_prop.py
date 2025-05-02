import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

import numpy as np
import time

# https://westpa.readthedocs.io/en/stable/documentation/core/westpa.core.propagators.html
# https://github.com/westpa/westpa/blob/b3afe209fcffc6238c1d2ec700059c7e30f2adca/src/westpa/core/propagators/executable.py#L688

class TestPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(TestPropagator, self).__init__(rc)
        print(self.rc.config)
        self.num_active = 0

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
        print("self.basis_states", self.basis_states)
        print("self.initial_states", self.initial_states)
        # print(dir(segments[0]))
        print(segments[0].data, segments[0].seg_id, segments[0].n_iter)
        print(segments[0].parent_id, segments[0].initial_state_id)
        print((self.rc.config['west', 'data', 'data_refs', 'segment'].format(segment=segments[0])))
        print((self.rc.config['west', 'data', 'data_refs', 'basis_state']))

        print("##", self.num_active, len(segments))
        self.num_active += 1
        

        for segment in segments:
            starttime = time.time()

            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                pass
            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                # Based on the executable probagator it looks like this branch is responsible for initializing from a basis state                
                initial_state = self.initial_states[segment.initial_state_id]
                print("initial_state.istate_type", initial_state.istate_type)
                # Could also be InitialState.ISTATE_TYPE_START
                assert initial_state.istate_type == InitialState.ISTATE_TYPE_BASIS
                basis_state = self.basis_states[initial_state.basis_state_id]
                print("basis_state", basis_state, basis_state.auxref)


            segment.status = Segment.SEG_STATUS_COMPLETE

            segment.walltime = time.time() - starttime
            # segment.cputime = rusage.ru_utime
        raise NotImplementedError