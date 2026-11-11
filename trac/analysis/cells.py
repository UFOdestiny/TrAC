"""All feature blocks of one setting (cache tag), aligned to the response ids of its feature file."""

from __future__ import annotations

from typing import Dict

import numpy as np

from trac import config as C
from trac.analysis.features.contrast import contrast_block
from trac.analysis.features.trajectory import trajectory_blocks
from trac.store import read_json_gz


def load_cell(tag: str) -> Dict[str, np.ndarray]:
    """Blocks of trac.analysis.features.build plus, when their caches exist, the trajectory blocks
    (TRAJ*, TFT), the rival blocks (CON, CON_END), the two-sample vote share SC2
    (trac.analysis.features.auxiliary), prefix consistency PC, the math PRM, verbalized top-k
    answers TOPK, and the reward model RM. Rows are unique by question id."""
    B = dict(np.load(C.FEATURES / f"{tag}.npz"))
    ids = np.array([str(q) for q in B["ids"]])
    last = {q: i for i, q in enumerate(ids)}
    if len(last) < len(ids):  # TriviaQA repeats some question ids: keep each id's last row,
        keep = np.array(sorted(last.values()))  # the one every id-keyed probe record refers to
        B = {k: (v[keep] if getattr(v, "ndim", 0) >= 1 and len(v) == len(ids) else v) for k, v in B.items()}
    ids = B["ids"]
    B.update(trajectory_blocks(tag, ids))
    B.update(contrast_block(tag, ids))
    fa = C.AUXILIARY / f"{tag}.npz"
    if fa.exists():
        za = np.load(fa, allow_pickle=True)
        m2 = {str(i): v for i, v in zip(za["ids"], za["sc2"])}
        B["SC2"] = np.array([m2.get(str(q), np.nan) for q in ids], float)[:, None]
    fp = C.PREFIX_CONSISTENCY / f"{tag}.json.gz"
    if fp.exists():
        P = {x["id"]: x["traces"].get("0", {}).get("pc", np.nan) for x in read_json_gz(fp)}
        B["PC"] = np.array([P.get(str(q), np.nan) for q in ids], float)[:, None]
    fp = C.PRM / f"{tag}.npz"
    if fp.exists():
        zp = np.load(fp)
        pos = {str(q): i for i, q in enumerate(zp["ids"])}
        idx = [pos.get(str(q)) for q in ids]
        B["PRM"] = np.array([[zp["prm_min"][i], zp["prm_last"][i]] if i is not None else [np.nan, np.nan] for i in idx], float)
    ft = C.TOPK / f"{tag}.json.gz"
    if ft.exists():
        B["TOPK"] = topk_block(read_json_gz(ft), ids)
    fr = C.REWARD / f"{tag}.npz"
    if fr.exists():
        zr = np.load(fr)
        mr = dict(zip(zr["ids"], zr["reward"]))
        B["RM"] = np.array([mr.get(str(q), np.nan) for q in ids], float)[:, None]
    return B


def topk_block(records, ids):
    """Verbalized top-3 answers (trac.pipeline.topk): confidence of a (0 if a is not listed), whether
    a is listed first, rank of a (4 if absent), confidence of the first answer, number of listed
    answers, and the verbalized confidence of a as a score (1/rank when no confidence was given)."""
    R = {r["id"]: r for r in records}
    out = np.full((len(ids), 6), np.nan)
    for i, q in enumerate(ids):
        r = R.get(str(q))
        if r is None or not r["items"]:
            continue
        its = r["items"]
        rank = next((k + 1 for k, x in enumerate(its) if x["eq_a"]), 4)
        confs = [x["conf"] for x in its]
        ca = next((x["conf"] for x in its if x["eq_a"]), 0.0)
        score = ca if ca is not None else (1.0 / rank if rank < 4 else 0.0)
        out[i] = [
            ca if ca is not None else np.nan,
            float(rank == 1),
            rank,
            confs[0] if confs[0] is not None else np.nan,
            len(its),
            score,
        ]
    return out


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
