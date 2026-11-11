"""Runner-up feature blocks RUNNER_UP and RUNNER_UP_END (CPU), derived from the runner-up cache (rewind.pipeline.runner_up).

Per position (8 checkpoints + the end of the trace): the share of the final answer a in the
teacher-forced probability mass of the candidate set {a, other candidates}, its log-likelihood margin
over the strongest other candidate, whether the elicited runner-up equals a, and the elicited
runner-up's mean token log-probability. RUNNER_UP keeps the value at the end of the trace of each of the four
curves and six summaries (10):
  share@end, margin@end, elicited==a @end, elicited lp @end,
  #other candidates, #elicited runner-ups, mean share, mean margin, mean elicited==a, mean elicited lp
RUNNER_UP_END uses the end of the trace only (minimal configuration and ablation). Blocks are cached
under EXP_ROOT/feature_cache and rebuilt whenever the source or this file is newer.
"""

from __future__ import annotations

import os

import numpy as np

from rewind import config as C
from rewind.store import read_json_gz, save_npz_atomic


def _seq(v):
    v = [x for x in (v or []) if x is not None]
    return float(np.sum(v)) if v else np.nan


def _mean(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def runner_up_row(rec) -> np.ndarray:
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
    """Runner-up answers at the end of the trace only: candidates a, the most frequent other intermediate
    answers, and the runner-up elicited at the end; share of a, margin, runner-up == a, runner-up mean
    token log-prob."""
    if rec is None or not rec.get("cands"):
        return np.full(4, np.nan)
    end_k = rec["pos"][-1]
    e = next((x for x in rec["elic"] if x["k"] == end_k), None)
    keep = [ci for ci, c in enumerate(rec["cands"]) if c["src"] in ("a", "intermediate") or (e is not None and c["surf"] == e["surf"])]
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


def runner_up_blocks(tag: str, ids):
    """{"RUNNER_UP": (n, 10), "RUNNER_UP_END": (n, 4)} aligned to `ids`; {} without a runner-up cache."""
    src = C.RUNNER_UP / f"{tag}.json.gz"
    if not src.exists():
        return {}
    cache = C.FEATURE_CACHE / f"runner_up__{tag}.npz"
    if cache.exists() and cache.stat().st_mtime > max(src.stat().st_mtime, os.path.getmtime(__file__)):
        return dict(np.load(cache))
    R = {r["id"]: r for r in read_json_gz(src)}
    blk = {"RUNNER_UP": np.vstack([runner_up_row(R.get(str(q))) for q in ids]), "RUNNER_UP_END": np.vstack([end_row(R.get(str(q))) for q in ids])}
    save_npz_atomic(cache, **blk)
    return blk
