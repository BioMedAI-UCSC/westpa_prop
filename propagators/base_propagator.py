import os
import time
from datetime import datetime

import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.segment import Segment

from computation.recorded_computation import RecordedComputation
from file_system.recorded_io import persist


class BasePropagator(WESTPropagator):

    def __init__(self, rc=None):
        super().__init__(rc)
        self._load_config()
        self._init_pcoord_calculator()
        self._init_recorded_calculators()

    def _load_config(self):
        raise NotImplementedError

    def _get_save_format(self, config_path):
        config = self.rc.config
        for key in config_path:
            config = config.get(key, {})
        return config.get("save_format", "dcd").lower()

    def _init_pcoord_calculator(self):
        cfg = self._get_pcoord_config()
        if cfg:
            cfg = dict(cfg)
            class_path = cfg.pop("class")
            self.pcoord_calculator = westpa.core.extloader.get_object(class_path)(**cfg)

    def _get_pcoord_config(self):
        raise NotImplementedError

    def _init_recorded_calculators(self):
        """
        Instantiate RecordedComputation wrappers from config.

        Each entry in recorded_calculators must have:
            name, storage, granularity  — RecordedComputation fields
            computation.class           — BaseComputation subclass to instantiate
            computation.*               — kwargs forwarded to the computation

        Computations with requires_positions=False are skipped online;
        run scripts/compute_recorded.py for those.

        west.cfg layout:
            recorded_calculators:
              - name:        tica_full
                storage:     west_h5
                granularity: per_frame
                computation:
                  class:      computation.tica_computation.TICAComputation
                  model_path: /path/to/model.tica
                  components: [0, 1, 2, 3]
        """
        self.recorded_calculators = []

        for cfg in self._get_recorded_configs():
            cfg             = dict(cfg)
            comp_cfg        = dict(cfg.pop("computation"))
            class_path      = comp_cfg.pop("class")
            computation     = westpa.core.extloader.get_object(class_path)(**comp_cfg)
            recorded        = RecordedComputation(computation=computation, **cfg)

            if not getattr(recorded.computation, "requires_positions", True):
                print(
                    f"[recorded] '{recorded.name}' requires_positions=False — "
                    f"skipping online, run compute_recorded.py instead."
                )
                continue

            self.recorded_calculators.append(recorded)

    def _get_recorded_configs(self):
        raise NotImplementedError

    def _run_recorded(self, positions: "np.ndarray", segment_outdir: str, n_iter: int, seg_id: int):
        """
        Called inside propagate() after pcoord is set.
        positions: (n_frames+1, n_atoms, 3) Angstrom, includes parent frame.
        """
        if not self.recorded_calculators:
            return

        west_h5_path = os.path.expandvars(
            self.rc.config.get(["west", "data", "west_data_file"], "west.h5")
        )

        for recorded in self.recorded_calculators:
            try:
                values = recorded.calculate(positions)
                persist(recorded, values, west_h5_path, segment_outdir, n_iter, seg_id)
            except Exception as e:
                print(f"[recorded] WARNING: '{recorded.name}' failed iter={n_iter} seg={seg_id}: {e}")

    def get_pcoord(self, state):
        raise NotImplementedError

    def propagate(self, segments):
        raise NotImplementedError

    def _get_segment_outdir(self, segment):
        pattern = self.rc.config["west", "data", "data_refs", "segment"]
        return os.path.expandvars(pattern.format(segment=segment))

    def _get_parent_outdir(self, segment):
        parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
        return self._get_segment_outdir(parent)

    def _finalize_segment(self, segment, starttime):
        segment.status = Segment.SEG_STATUS_COMPLETE
        segment.walltime = time.time() - starttime

    def _print_completion(self, num_segments, elapsed_time):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}: Finished {num_segments} segments in {elapsed_time:0.2f}s")
