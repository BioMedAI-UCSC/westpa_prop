import os
import numpy as np
import h5py


def write_to_west_h5(west_h5_path: str, n_iter: int, seg_id: int, name: str, values: np.ndarray):
    """
    Write values into west.h5 as an aux dataset alongside pcoord.

    Layout: /iterations/iter_{n_iter:08d}/{name}[seg_id, n_frames, n_dims]
    Dataset is created on first write, sized to (n_segs, n_frames, n_dims).
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[np.newaxis, :]

    with h5py.File(west_h5_path, "a") as f:
        iter_grp        = f[f"iterations/iter_{n_iter:08d}"]
        n_segs          = iter_grp["seg_index"].shape[0]
        n_frames, n_dims = values.shape

        if name not in iter_grp:
            iter_grp.create_dataset(
                name,
                shape=(n_segs, n_frames, n_dims),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

        iter_grp[name][seg_id] = values


def read_from_west_h5(west_h5_path: str, n_iter: int, name: str, seg_id: int = None) -> np.ndarray:
    with h5py.File(west_h5_path, "r") as f:
        ds = f[f"iterations/iter_{n_iter:08d}/{name}"]
        return ds[seg_id][()] if seg_id is not None else ds[()]


def write_to_npz(segment_outdir: str, name: str, values: np.ndarray):
    """Append or overwrite a key in the segment's seg.npz, preserving existing keys."""
    npz_path = os.path.join(segment_outdir, "seg.npz")
    existing = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    existing[name] = np.asarray(values, dtype=np.float32)
    np.savez(npz_path, **existing)


def read_from_npz(segment_outdir: str, name: str) -> np.ndarray:
    npz_path = os.path.join(segment_outdir, "seg.npz")
    data = np.load(npz_path)
    if name not in data:
        raise KeyError(f"'{name}' not in {npz_path}. Available: {list(data.keys())}")
    return data[name]


def persist(recorded: "RecordedComputation", values: np.ndarray, west_h5_path: str, segment_outdir: str, n_iter: int, seg_id: int):
    """Route results to storage targets declared on the RecordedComputation."""
    if recorded.write_west_h5:
        write_to_west_h5(west_h5_path, n_iter, seg_id, recorded.name, values)
    if recorded.write_npz:
        write_to_npz(segment_outdir, recorded.name, values)
