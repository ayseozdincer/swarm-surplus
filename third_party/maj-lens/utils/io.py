"""JSON/NPZ writing and a process pool."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

RAW_FIELDS = ('vectors', 'episode', 'agent', 'response')


def numeric(path, fields=RAW_FIELDS):
    """The declared numeric arrays of an .npz, refusing any other field."""
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(fields):
            raise ValueError(f'unexpected numeric fields in {path}: {saved.files}')
        return {key: saved[key] for key in fields}


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def save_json(path, value):
    """Atomic JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def immutable(path, value):
    """Write once; any later write must be identical."""
    value = json.loads(json.dumps(value))
    if Path(path).exists():
        if json.loads(Path(path).read_text()) != value:
            raise ValueError(f'frozen artifact differs: {path}')
    else:
        write_json(path, value)
    return value


def run_jobs(jobs, workers):
    pool = ProcessPoolExecutor(max_workers=workers)
    futures = [pool.submit(*job) for job in jobs]
    try:
        for n, future in enumerate(as_completed(futures), 1):
            print(f'{n}/{len(futures)} {future.result()}', flush=True)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        for process in list((getattr(pool, '_processes', None) or {}).values()):
            process.terminate()
        raise
    pool.shutdown()
