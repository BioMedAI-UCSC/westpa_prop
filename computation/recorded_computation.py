from enum import Enum

from computation.base_computation import BaseComputation


class Storage(str, Enum):
    WEST_H5 = "west_h5"
    NPZ     = "npz"
    BOTH    = "both"


class Granularity(str, Enum):
    PER_FRAME   = "per_frame"
    PER_SEGMENT = "per_segment"


class RecordedComputation:
    """
    Wraps a BaseComputation and declares where and how its output is saved.

    The wrapped computation is instantiated from config identically to a
    pcoord_calculator. RecordedComputation adds:
        name        — key used in west.h5 dataset / seg.npz
        storage     — Storage enum: west_h5 | npz | both
        granularity — Granularity enum: per_frame | per_segment

    west.cfg usage:
        recorded_calculators:
          - name:        tica_full
            storage:     west_h5
            granularity: per_frame
            computation:
              class:      computation.tica_computation.TICAComputation
              model_path: /path/to/model.tica
              components: [0, 1, 2, 3]
    """

    def __init__(
        self,
        name: str,
        storage: str,
        granularity: str,
        computation: BaseComputation,
    ):
        self.name        = name
        self.storage     = Storage(storage)
        self.granularity = Granularity(granularity)
        self.computation = computation

        if not isinstance(computation, BaseComputation):
            raise TypeError(
                f"computation must be a BaseComputation instance, got {type(computation)}"
            )

    def calculate(self, data):
        return self.computation.calculate(data)

    @property
    def requires_positions(self) -> bool:
        return getattr(self.computation, "requires_positions", True)

    @property
    def write_west_h5(self) -> bool:
        return self.storage in (Storage.WEST_H5, Storage.BOTH)

    @property
    def write_npz(self) -> bool:
        return self.storage in (Storage.NPZ, Storage.BOTH)
