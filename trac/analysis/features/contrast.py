"""Rival-answer feature blocks CON and CON_END (CPU), derived from EXP_ROOT/contrast (trac.pipeline.contrast).

Per position (8 trajectory checkpoints + the end of the trace): the share of the final answer a
in the teacher-forced probability mass of the candidate set {a, rivals}, its log-likelihood margin
over the strongest rival, whether the elicited rival equals a, and the elicited rival's
mean token log-probability. CON keeps the value at the end of the trace of each of the four
curves and six summaries (10):
  share@end, margin@end, elicited==a @end, elicited lp @end,
  #rivals, #elicited rivals, mean share, mean margin, mean elicited==a, mean elicited lp
CON_END uses the end of the trace only (minimal configuration and ablation). Blocks are cached
under EXP_ROOT/feature_cache and rebuilt whenever the source or this file is newer.
"""

from __future__ import annotations

import os

import numpy as np

from trac import config as C
from trac.store import read_json_gz, save_npz_atomic


def _seq(v):
    v = [x for x in (v or []) if x is not None]
    return float(np.sum(v)) if v else np.nan


def _mean(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def con_row(rec) -> np.ndarray:
    if rec is None or not rec.get("cands"):
        return np.full(10, np.nan)
    S = np.array([[_seq(p) for p in c] for c in rec["tf"]])  # (candidates, positions)
    share, margin = [], []
    for s in S.T:
        ok = np.isfinite(s)
        if not ok[0]:
            share.append(np.nan)
            margin.append(np.nan)
            continue
        m = s[ok].max()
        share.append(float(np.exp(s[0] - m) / np.exp(s[ok] - m).sum()))
        margin.append(float(s[0] - np.nanmax(s[1:])) if ok[1:].any() else 10.0)
    eq = [float(e["eq_a"]) for e in rec["elic"]]
    elp = [float(np.mean(e["tok_lps"])) if e["tok_lps"] else np.nan for e in rec["elic"]]
    last = lambda v: v[-1] if len(v) else np.nan
    return np.array(
        [
            last(share),
            last(margin),
            last(eq),
            last(elp),
            len(rec["cands"]) - 1,
            sum(c["src"] == "elic" for c in rec["cands"]),
            _mean(share),
            _mean(margin),
            _mean(eq),
            _mean(elp),
        ],
        float,
    )


def end_row(rec) -> np.ndarray:
    """Rival answers at the end of the trace only: candidates a, the trajectory rivals,
    and the rival elicited at the end; share of a, margin, rival == a, rival mean token log-prob."""
    if rec is None or not rec.get("cands"):
        return np.full(4, np.nan)
    end_k = rec["pos"][-1]
    e = next((x for x in rec["elic"] if x["k"] == end_k), None)
    keep = [ci for ci, c in enumerate(rec["cands"]) if c["src"] in ("a", "traj") or (e is not None and c["surf"] == e["surf"])]
    s = np.array([_seq(rec["tf"][ci][-1]) for ci in keep], float)
    if not np.isfinite(s[0]):
        share = margin = np.nan
    else:
        ok = np.isfinite(s)
        m = s[ok].max()
        share = float(np.exp(s[0] - m) / np.exp(s[ok] - m).sum())
        margin = float(s[0] - np.nanmax(s[1:])) if ok[1:].any() else 10.0
    eq = float(e["eq_a"]) if e else np.nan
    elp = float(np.mean(e["tok_lps"])) if e and e["tok_lps"] else np.nan
    return np.array([share, margin, eq, elp], float)


def contrast_block(tag: str, ids):
    """{"CON": (n, 10), "CON_END": (n, 4)} aligned to `ids`; {} without a contrast cache."""
    src = C.CONTRAST / f"{tag}.json.gz"
    if not src.exists():
        return {}
    cache = C.FEATURE_CACHE / f"contrast__{tag}.npz"
    if cache.exists() and cache.stat().st_mtime > max(src.stat().st_mtime, os.path.getmtime(__file__)):
        return dict(np.load(cache))
    R = {r["id"]: r for r in read_json_gz(src)}
    blk = {"CON": np.vstack([con_row(R.get(str(q))) for q in ids]), "CON_END": np.vstack([end_row(R.get(str(q))) for q in ids])}
    save_npz_atomic(cache, **blk)
    return blk
