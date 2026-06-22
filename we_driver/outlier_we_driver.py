import logging
import operator

import numpy as np
import westpa
from westpa.core.binning.mab_driver import MABDriver

log = logging.getLogger(__name__)


class OutlierWEDriver(MABDriver):
    """MAB driver + Paper-2 outlier splitting.

    After standard MAB/WE resampling, identifies novel (high Local-Outlier-Factor)
    walkers in latent pcoord space and, within each bin, performs count- and
    weight-preserving swaps: split the most-novel walker, merge the two least-novel.
    This steers trajectory resources toward newly discovered interface geometries
    without defining a target state.

    Config (west.drivers.outlier):
        enabled, n_neighbors, n_swaps_per_bin, min_weight
    """

    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)
        cfg = (westpa.rc.config.get(['west', 'drivers', 'outlier'], {})) or {}
        self.o_enabled = bool(cfg.get('enabled', True))
        self.o_neighbors = int(cfg.get('n_neighbors', 20))
        self.o_swaps = int(cfg.get('n_swaps_per_bin', 1))
        self.o_min_weight = float(cfg.get('min_weight', 1e-12))

    def _run_we(self):
        super()._run_we()
        if not self.o_enabled:
            return
        try:
            self._apply_outlier_bias()
        except Exception as e:
            log.warning(f'outlier bias skipped: {e!r}')

    def _latent(self, seg):
        return np.asarray(seg.pcoord[0], dtype=np.float64).ravel()

    def _apply_outlier_bias(self):
        bins = [b for b in self.next_iter_binning if len(b) > 0]
        segs = [s for b in bins for s in b]
        if len(segs) < self.o_neighbors + 2:
            return
        from sklearn.neighbors import LocalOutlierFactor
        Z = np.array([self._latent(s) for s in segs])
        lof = LocalOutlierFactor(n_neighbors=min(self.o_neighbors, len(segs) - 1))
        lof.fit(Z)
        score = {id(s): float(v) for s, v in zip(segs, -lof.negative_outlier_factor_)}

        n_swaps = 0
        for bin in bins:
            for _ in range(self.o_swaps):
                if not self._swap_once(bin, score):
                    break
                n_swaps += 1
        if n_swaps:
            log.info(f'outlier bias: {n_swaps} split/merge swaps across {len(bins)} bins')

    def _swap_once(self, bin, score):
        members = sorted(bin, key=lambda s: score.get(id(s), 0.0))
        if len(members) < 3:
            return False
        splittable = [m for m in reversed(members) if m.weight >= 2 * self.o_min_weight]
        if not splittable:
            return False
        high = splittable[0]
        lows = [m for m in members if m is not high][:2]
        if len(lows) < 2:
            return False

        bin.remove(high)
        kids = self._split_walker(high, 2, bin)
        bin.update(kids)
        for k in kids:
            score[id(k)] = score.get(id(high), 0.0)

        lows = sorted(lows, key=operator.attrgetter('weight'))
        bin.difference_update(lows)
        merged, _ = self._merge_walkers(lows, None, bin)
        bin.add(merged)
        score[id(merged)] = min(score.get(id(lows[0]), 0.0), score.get(id(lows[1]), 0.0))
        return True
