import logging
import os

from features.interface_featurizer import InterfaceFeaturizer
from embeddings import build, save_model, method_kwargs, default_feature_mode
from analysis.frame_loader import load_segments, featurize_segments

log = logging.getLogger(__name__)


class LearnedPCPlugin:
    """Retrain the learned-PC embedding on-the-fly during a WE run.

    Every `retrain_period` iterations (after `warmup`), pools interface features
    from the last `window` iterations and refits the embedding, rewriting
    `model_path` atomically so propagator workers hot-reload it next iteration.

    Config (west.plugins entry):
        plugin, model_path, topology, selection_a, selection_b, method,
        n_components, feature_mode, retrain_period, warmup, window,
        iter_stride, seg_stride, lag, epochs, priority
    """

    def __init__(self, sim_manager, plugin_config):
        self.sim_manager = sim_manager
        c = plugin_config
        self.method = c.get('method', 'tica')
        self.model_path = os.path.expandvars(c['model_path'])
        self.topology = os.path.expandvars(c['topology'])
        self.sel_a, self.sel_b = c['selection_a'], c['selection_b']
        self.n_components = int(c.get('n_components', 2))
        self.feature_mode = c.get('feature_mode', default_feature_mode(self.method))
        self.retrain_period = int(c.get('retrain_period', 5))
        self.warmup = int(c.get('warmup', 5))
        self.window = int(c.get('window', 20))
        self.iter_stride = int(c.get('iter_stride', 1))
        self.seg_stride = int(c.get('seg_stride', 1))
        self.lag = int(c.get('lag', 1))
        self.epochs = int(c.get('epochs', 30))
        self.sim_root = os.environ.get('WEST_SIM_ROOT', os.getcwd())
        self.featurizer = InterfaceFeaturizer(self.topology, self.sel_a, self.sel_b,
                                              mode=self.feature_mode)
        sim_manager.register_callback(sim_manager.finalize_iteration,
                                      self.retrain, c.get('priority', 1))

    def retrain(self, *args, **kwargs):
        n = self.sim_manager.n_iter
        if n is None or n < self.warmup or n % self.retrain_period != 0:
            return
        start = max(1, n - self.window + 1)
        segs, _, lens = load_segments(self.sim_root, self.topology,
                                      iter_stride=self.iter_stride,
                                      seg_stride=self.seg_stride,
                                      min_iter=start, max_iter=n)
        if not segs:
            log.warning(f'[learned-pc] iter {n}: no frames found, skip retrain')
            return
        X = featurize_segments(self.featurizer, segs)
        emb = build(self.method, **method_kwargs(self.method, self.n_components,
                                                 self.lag, self.epochs))
        emb.fit(X, lengths=lens)
        save_model(emb, self.method, self.model_path)
        log.info(f'[learned-pc] iter {n}: retrained {self.method} on {X.shape[0]} '
                 f'frames (iters {start}-{n}) -> {self.model_path}')
