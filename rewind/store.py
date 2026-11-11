"""Reading and writing cached artifacts under EXP_ROOT."""

from __future__ import annotations

import gzip
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import numpy as np

from rewind import config as C


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt") as f:
        return json.load(f)


@contextmanager
def atomic_output(path: Path):
    """Publish a complete artifact atomically; clean temporary files on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_gz(path: Path, obj: Any) -> None:
    with atomic_output(path) as temporary:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(obj, stream)


def save_npz_atomic(path: Path, **arrays) -> None:
    """np.savez through a temporary file, so readers never see a partial cache."""
    with atomic_output(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)


def generation_done(tag: str) -> bool:
    return (C.GEN / f"{tag}.json.gz").exists() and (C.GEN / f"{tag}.tok.npz").exists()


def load_generation(tag: str) -> List[Dict[str, Any]]:
    """Generation records of a tag, each with its token ids attached as `_pids` (prompt)
    and `_oids` (original output)."""
    rows = read_json_gz(C.GEN / f"{tag}.json.gz")
    z = np.load(C.GEN / f"{tag}.tok.npz")
    po = np.concatenate([[0], np.cumsum(z["prompt_len"])])
    oo = np.concatenate([[0], np.cumsum(z["out_len"])])
    for i, r in enumerate(rows):
        r["_pids"] = z["prompt_ids"][po[i] : po[i + 1]].tolist()
        r["_oids"] = z["out_ids"][oo[i] : oo[i + 1]].tolist()
    return rows
