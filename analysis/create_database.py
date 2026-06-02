from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import duckdb
import h5py
import numpy as np


@dataclass
class Config:
    h5_path: Path = Path("west.h5")
    db_path: Path = Path("west_db.duckdb")
    table_name: str = "segment_data"
    iterations_group: str = "iterations"
    iter_name_prefix: str = "iter_"
    datasets: Optional[list[str]] = None   # None = ingest all datasets
    max_iters: Optional[int] = None        # None = ingest all iterations
    batch_size: int = 1000
    drop_existing: bool = True
    quiet: bool = False


def setup_logging(quiet: bool) -> logging.Logger:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )
    return logging.getLogger(__name__)


def to_python_value(x: np.ndarray) -> object:
    arr = np.squeeze(np.asarray(x))
    return arr.item() if arr.ndim == 0 else arr.tolist()


def iter_segments(
    h5_file: h5py.File,
    cfg: Config,
    log: logging.Logger,
) -> Generator[tuple[int, int, str, str], None, None]:
    """Yield (iteration, segment_index, dataset_name, values_json). segment_index is -1 for scalar datasets."""
    iterations_group = h5_file[cfg.iterations_group]
    iter_names = sorted(iterations_group.keys())

    if cfg.max_iters is not None:
        iter_names = iter_names[: cfg.max_iters]

    for iter_name in iter_names:
        iter_group = iterations_group[iter_name]
        try:
            iter_num = int(iter_name.removeprefix(cfg.iter_name_prefix))
        except ValueError:
            log.warning("Skipping non-numeric iteration group: %s", iter_name)
            continue

        for dset_name, obj in iter_group.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            if cfg.datasets is not None and dset_name not in cfg.datasets:
                continue

            data = obj[()]

            if np.ndim(data) == 0:
                yield iter_num, -1, dset_name, json.dumps(to_python_value(data))
                log.info("iter %d: scalar dataset %s", iter_num, dset_name)
                continue

            n_seg = data.shape[0]
            log.info("iter %d: dataset %-30s  segments=%d", iter_num, dset_name, n_seg)
            for seg_idx in range(n_seg):
                yield iter_num, seg_idx, dset_name, json.dumps(to_python_value(data[seg_idx]))


def init_table(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    if cfg.drop_existing:
        con.execute(f"DROP TABLE IF EXISTS {cfg.table_name}")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {cfg.table_name} (
            iteration   INTEGER,
            segment     INTEGER,
            dataset     TEXT,
            values_json JSON
        )
    """)


def insert_batched(
    con: duckdb.DuckDBPyConnection,
    table: str,
    rows: Generator[tuple, None, None],
    batch_size: int,
    log: logging.Logger,
) -> int:
    """Insert rows in batches to avoid loading the full dataset into memory. Returns total row count."""
    sql = f"INSERT INTO {table} VALUES (?, ?, ?, ?)"
    total = 0
    batch: list[tuple] = []

    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            con.executemany(sql, batch)
            total += len(batch)
            log.debug("Flushed %d rows (total: %d)", len(batch), total)
            batch.clear()

    if batch:
        con.executemany(sql, batch)
        total += len(batch)

    return total


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest an HDF5 file into DuckDB (one row per iteration/segment/dataset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--h5",          dest="h5_path",           default="west.h5",        help="Path to the input HDF5 file")
    p.add_argument("--db",          dest="db_path",           default="west.db", help="Path to the output DuckDB database")
    p.add_argument("--table",       dest="table_name",        default="segment_data",   help="DuckDB table name")
    p.add_argument("--iter-group",  dest="iterations_group",  default="iterations",     help="Top-level HDF5 group containing iterations")
    p.add_argument("--iter-prefix", dest="iter_name_prefix",  default="iter_",          help="Prefix stripped from iteration group names to get the integer index")
    p.add_argument("--datasets",    dest="datasets",          nargs="+", default=None,  help="Dataset names to ingest (default: all)")
    p.add_argument("--max-iters",   dest="max_iters",         type=int, default=None,   help="Maximum number of iterations to process")
    p.add_argument("--batch-size",  dest="batch_size",        type=int, default=1000,   help="Number of rows per INSERT batch")
    p.add_argument("--no-drop",     dest="drop_existing",     action="store_false",     help="Do not drop existing table; append instead")
    p.add_argument("--quiet",       dest="quiet",             action="store_true",      help="Suppress per-dataset progress messages")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        h5_path=Path(args.h5_path),
        db_path=Path(args.db_path),
        table_name=args.table_name,
        iterations_group=args.iterations_group,
        iter_name_prefix=args.iter_name_prefix,
        datasets=args.datasets,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        drop_existing=args.drop_existing,
        quiet=args.quiet,
    )


def run(cfg: Config) -> int:
    log = setup_logging(cfg.quiet)

    if not cfg.h5_path.exists():
        log.error("HDF5 file not found: %s", cfg.h5_path)
        sys.exit(1)

    log.info("Opening HDF5:  %s", cfg.h5_path)
    log.info("Target DuckDB: %s  (table: %s)", cfg.db_path, cfg.table_name)

    with duckdb.connect(str(cfg.db_path)) as con:
        init_table(con, cfg)
        with h5py.File(cfg.h5_path, "r") as h5f:
            rows = iter_segments(h5f, cfg, log)
            total = insert_batched(con, cfg.table_name, rows, cfg.batch_size, log)

    log.info("Done — wrote %d rows to %s", total, cfg.db_path)
    return total


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
