"""All feature blocks of one cell (cache tag), aligned to the response ids of its feature file."""
from __future__ import annotations

from typing import Dict

import numpy as np

from trac import config as C
from trac.analysis.trajectory_features import textpos_block, trajectory_blocks, tsc_block
from trac.store import read_json_gz


def load_cell(tag: str) -> Dict[str, np.ndarray]:
    """Blocks of trac.analysis.features, plus (when their caches exist) the trajectory blocks
    and TrAC-0, TEXTPOS, SC2 (two-sample vote share, trac.analysis.auxiliary), PC (prefix
    consistency of the original trace), and TSC."""
    B = dict(np.load(C.FEATURES / f"{tag}.npz"))
    ids = B["ids"]
    B.update(trajectory_blocks(tag, ids))
    B.update(textpos_block(tag, ids))
    fa = C.AUXILIARY / f"{tag}.npz"
    if fa.exists():
        za = np.load(fa, allow_pickle=True)
        m2 = {str(i): v for i, v in zip(za["ids"], za["sc2"])}
        B["SC2"] = np.array([m2.get(str(q), np.nan) for q in ids], float)[:, None]
    fp = C.PREFIX_CONSISTENCY / f"{tag}.json.gz"
    if fp.exists():
        P = {x["id"]: x["traces"].get("0", {}).get("pc", np.nan) for x in read_json_gz(fp)}
        B["PC"] = np.array([P.get(str(q), np.nan) for q in ids], float)[:, None]
    B.update(tsc_block(tag, ids))
    return B


def load_hidden(tag: str, ids):
    """(N, positions, layers, d) hidden states aligned to `ids`, or None.
    Positions: end_reason, answer_end, mean_reason, cue; layers: 0.5L, 0.75L, L."""
    f = C.HIDDEN / f"{tag}.npz"
    if not f.exists():
        return None
    zh = np.load(f)
    pos = {i: k for k, i in enumerate(zh["ids"])}
    if not all(i in pos for i in ids):
        return None
    return zh["H"][[pos[i] for i in ids]].astype(np.float32)
