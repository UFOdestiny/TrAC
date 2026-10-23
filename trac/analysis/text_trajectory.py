"""Model-free text trajectory baseline and latent-commitment analysis (no decoding).

At the same checkpoints as the PCE trajectory (EXP_ROOT/trajectory), the candidate answer already
written in the reasoning prefix r_{<=t} is read off the text with the extractor used for final
answers (trac.answers.extract_answer) and compared with the final answer a by the same
equivalence check. This yields a text agreement profile with the same summaries as the
trajectory (share, commitment point, switches, distinct candidates), computed without any model
call. Latent commitment = checkpoints where the re-elicited answer already equals a while the
text written so far does not state it.

  python -m trac.analysis.text_trajectory --tag T     -> EXP_ROOT/feature_cache/text_trajectory__T.npz
  python -m trac.analysis.text_trajectory --report    (summary over settings)
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.analysis.trajectory_features import GRID as _GRID
from trac.store import load_generation, read_json_gz, save_npz_atomic

GRID = _GRID + (1.0,)


def commit(v):
    """Earliest grid fraction from which every later entry is 1 (1.1 if the last is 0)."""
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
    TT = np.full((len(ids), len(GRID) + 5), np.nan)     # text profile + share, commit, switches, distinct, latent
    MG = np.full((len(ids), len(GRID)), np.nan)         # model greedy agreement on the same grid
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
            tag_ag.append(ag); cands.append(cand); mod_ag.append(np.nan if m_ag is None else float(m_ag))
        ta, ma = np.array(tag_ag), np.array(mod_ag)
        sw = sum(1 for x, y in zip(cands, cands[1:]) if x != y)
        latent = float(np.nanmean((ma == 1) & (ta == 0)))
        TT[i] = np.concatenate([ta, [ta.mean(), commit(ta), sw, len({c for c in cands if c is not None}), latent]])
        MG[i] = ma
    save_npz_atomic(C.FEATURE_CACHE / f"text_trajectory__{tag}.npz", TEXTTRAJ=TT, MODELG=MG)
    print("[text]", tag, flush=True)


def report(min_err=10):
    from sklearn.metrics import roc_auc_score
    G = len(GRID)
    res = {}
    lat_c, lat_w, early_c, early_w = [], [], [], []
    for f in sorted(glob.glob(str(C.FEATURE_CACHE / "text_trajectory__*.npz"))):
        tag = os.path.basename(f)[len("text_trajectory__"):-4]
        B = np.load(C.FEATURES / f"{tag}.npz"); y = B["y"].astype(int)
        z = np.load(f); TT, MG = z["TEXTTRAJ"], z["MODELG"]
        ok = np.isfinite(TT[:, 0])
        if min(y.sum(), (1 - y).sum()) < min_err:
            continue
        ts, mc = TT[:, G], np.array([commit(m) if np.isfinite(m).all() else np.nan for m in MG])
        tc = TT[:, G + 1]
        for k, s in [("text share", ts), ("text commitment", -tc), ("model greedy share", np.nanmean(MG, 1)),
                     ("model commitment", -mc)]:
            s = np.where(np.isfinite(s), s, np.nanmedian(s))
            res.setdefault(k, []).append(roc_auc_score(y, s))
        lat = TT[:, G + 4]
        lat_c.append(np.nanmean(lat[(y == 1) & ok])); lat_w.append(np.nanmean(lat[(y == 0) & ok]))
        both = ok & np.isfinite(mc)
        early = (mc < tc)                                  # model commits before the text does
        early_c.append(np.mean(early[both & (y == 1)])); early_w.append(np.mean(early[both & (y == 0)]))
    print(f"{len(res.get('text share', []))} settings (macro AUROC, training-free)")
    for k, v in res.items():
        print(f"  {k:22s} {np.mean(v):.3f}")
    print(f"latent agreement share (model = a, text != a): correct {np.mean(lat_c):.3f}  wrong {np.mean(lat_w):.3f}")
    print(f"model commits strictly before the text: correct {np.mean(early_c):.3f}  wrong {np.mean(early_w):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.tag:
        run_tag(a.tag)
    if a.report:
        report()


if __name__ == "__main__":
    main()
