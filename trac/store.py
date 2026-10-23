"""Reading and writing cached artifacts under EXP_ROOT."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from trac import config as C


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def write_json_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def by_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["id"]: r for r in records}


def save_npz_atomic(path: Path, **arrays) -> None:
    """np.savez through a temporary file, so readers never see a partial cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path)[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


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
        r["_pids"] = z["prompt_ids"][po[i]:po[i + 1]].tolist()
        r["_oids"] = z["out_ids"][oo[i]:oo[i + 1]].tolist()
    return rows
