"""Answers already written in the trace text, on the trajectory grid (CPU, no model call).

At the grid points of the trajectory (EXP_ROOT/trajectory), the candidate answer already written in the
reasoning prefix is read off the text with the extractor used for final answers
(trac.answers.extract_answer) and compared with the final answer a by the same equivalence check. This
gives a text agreement profile with the summaries of the trajectory (share, convergence point, switches,
distinct candidates) and the share of grid points where the intermediate answer already equals a while
the text does not state it. trac.analysis.evaluation.convergence compares both.

  python -m trac.analysis.features.text --tag T     -> EXP_ROOT/feature_cache/text_trajectory__T.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.analysis.features.trajectory import GRID as _GRID
from trac.store import load_generation, read_json_gz, save_npz_atomic

GRID = _GRID + (1.0,)


def commit(v):
    """Convergence point: earliest grid fraction from which every later entry is 1 (1.1 if the last is 0)."""
    if not len(v) or v[-1] != 1:
        return 1.1
    cp = 1.0
    for j in range(len(v) - 1, -1, -1):
        if v[j] == 1:
            cp = GRID[j]
        else:
            break
    return cp


def run_tag(tag):
    bb = M.from_tag(tag)
    tok = M.load_tokenizer(bb)
    rows = {r["id"]: r for r in load_generation(tag)}
    R2 = {x["id"]: x for x in read_json_gz(C.TRAJECTORY / f"{tag}.json.gz")}
    B = np.load(C.FEATURES / f"{tag}.npz")
    ids = [str(i) for i in B["ids"]]
    TT = np.full((len(ids), len(GRID) + 5), np.nan)  # text profile + share, commit, switches, distinct, latent
    MG = np.full((len(ids), len(GRID)), np.nan)  # model greedy agreement on the same grid
    for i, q in enumerate(ids):
        r, rec = rows.get(q), R2.get(q)
        if r is None or rec is None or r["answer"] is None:
            continue
        body, _ = M.reasoning_prefix(tok, bb, r["_oids"], r["raw"])
        T = rec["T"] or len(body) or 1
        pts = [(c["k"] / T, c["k"], c.get("agree")) for c in rec["ck"]] + [(1.0, T, (rec["end"] or {}).get("agree"))]
        tag_ag, mod_ag, cands = [], [], []
        cache = {}
        for f in GRID:
            _, k, m_ag = min(pts, key=lambda p: abs(p[0] - f))
            if k not in cache:
                cand = A.extract_answer(r, tok.decode(body[:k], skip_special_tokens=True))
                cache[k] = (cand, float(A.equiv(r, cand, r["answer"])))
            cand, ag = cache[k]
            tag_ag.append(ag)
            cands.append(cand)
            mod_ag.append(np.nan if m_ag is None else float(m_ag))
        ta, ma = np.array(tag_ag), np.array(mod_ag)
        sw = sum(1 for x, y in zip(cands, cands[1:]) if x != y)
        latent = float(np.nanmean((ma == 1) & (ta == 0)))
        TT[i] = np.concatenate([ta, [ta.mean(), commit(ta), sw, len({c for c in cands if c is not None}), latent]])
        MG[i] = ma
    save_npz_atomic(C.FEATURE_CACHE / f"text_trajectory__{tag}.npz", TEXTTRAJ=TT, MODELG=MG)
    print("[text]", tag, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    run_tag(ap.parse_args().tag)


if __name__ == "__main__":
    main()
