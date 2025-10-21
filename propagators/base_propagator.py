import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.segment import Segment
import time
import os
from datetime import datetime


class BasePropagator(WESTPropagator):
    
    def __init__(self, rc=None):
        super(BasePropagator, self).__init__(rc)
        self._load_config()
        self._init_pcoord_calculator()
    
    def _load_config(self):
        raise NotImplementedError
    
    def _get_save_format(self, config_path):
        config = self.rc.config
        for key in config_path:
            config = config.get(key, {})
        return config.get('save_format', 'dcd').lower()
    
    def _init_pcoord_calculator(self):
        pcoord_config = self._get_pcoord_config()
        if pcoord_config:
            pcoord_config = dict(pcoord_config)
            class_path = pcoord_config.pop("class")
            calculator_class = westpa.core.extloader.get_object(class_path)
            self.pcoord_calculator = calculator_class(**pcoord_config)
    
    def _get_pcoord_config(self):
        raise NotImplementedError
    
    def get_pcoord(self, state):
        raise NotImplementedError
    
    def propagate(self, segments):
        raise NotImplementedError
    
    def _get_segment_outdir(self, segment):
        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']
        return os.path.expandvars(segment_pattern.format(segment=segment))
    
    def _get_parent_outdir(self, segment):
        parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
        return self._get_segment_outdir(parent)
    
    def _finalize_segment(self, segment, starttime):
        segment.status = Segment.SEG_STATUS_COMPLETE
        segment.walltime = time.time() - starttime
    
    def _print_completion(self, num_segments, elapsed_time):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}: Finished {num_segments} segments in {elapsed_time:0.2f}s")
