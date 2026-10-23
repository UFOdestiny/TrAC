"""Trajectory feature blocks and TrAC-0 (CPU), derived from EXP_ROOT/trajectory.

Each trajectory record (trac.pipeline.trajectory) is resampled to a fixed grid of 20
fractions of the trace plus the completed prefix (nearest checkpoint per grid point).

  TRAJ     per grid point: greedy agreement with a, and the share of sampled answers equal
           to a; summaries: mean greedy agreement, mean sampled share, commitment points
           (earliest fraction from which every later probe agrees), answer switches,
           distinct answers, mean sampled-answer entropy, level and late-half rise of a's
           teacher-forced log-likelihood, and the log-likelihood of the end answer   (52)
  TRAC0    TrAC-0, the training-free score: share of re-elicited answers (greedy and
           sampled, pooled per grid point) equal to a, averaged over the grid          (1)
  TRAJ_G, TRAC0_G   the same from greedy answers only (ablation)
  TRAJ_8, TRAC0_8   the same from 8 evenly spaced checkpoints (ablation)
  TRAJ_4            the same from 4 checkpoints (ablation)

Passive and sampling comparators built from the same caches:
  TEXTPOS  where and how often the final answer string already appears in the reasoning
           body, and how many distinct boxed candidates it contains (no decoding)       (5)
  TSC      trajectory-weighted consensus over the K = 8 traces                          (3)

Blocks are cached under EXP_ROOT/feature_cache and rebuilt whenever a source or this file
is newer than the cache.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.store import read_json_gz, save_npz_atomic

GRID = tuple(round((j + 1) / 21, 4) for j in range(20))


def _mean(v):
    v = [x for x in (v or []) if x is not None]
    return float(np.mean(v)) if v else np.nan


def _answer_ll(tf):
    """Mean teacher-forced log-likelihood of the answer tokens (closing brace excluded)."""
    if not tf:
        return np.nan
    return _mean(tf[:-1] if len(tf) > 1 else tf)


def thin(rec, n):
    """The same record restricted to n evenly spaced checkpoints."""
    ck = rec["ck"]
    if len(ck) <= n:
        return rec
    keep = sorted({int(round(x)) for x in np.linspace(0, len(ck) - 1, n)})
    return {**rec, "ck": [ck[j] for j in keep]}


def greedy_only(rec):
    """The same record without the sampled answers."""
    strip = lambda c: {k: v for k, v in (c or {}).items() if k != "samp"} if c else c
    return {**rec, "ck": [strip(c) for c in rec["ck"]], "end": strip(rec["end"])}


def _grid_points(rec):
    T = rec["T"] or 1
    pts = [(c["k"] / T, c) for c in rec["ck"]] + [(1.0, rec["end"] or {})]
    return [min(pts, key=lambda p: abs(p[0] - f))[1] or {} for f in list(GRID) + [1.0]]


def trajectory_features(rec) -> np.ndarray:
    """TRAJ row of one record (any number of checkpoints, optional sampled answers)."""
    end = rec["end"] or {}
    g, sft, ent, tf, ans = [], [], [], [], []
    for c in _grid_points(rec):
        g.append(float(c.get("agree", np.nan)) if c else np.nan)
        sm = c.get("samp")
        if sm and sm["agree"]:
            sft.append(np.mean(sm["agree"]))
            _, cnt = np.unique([str(x) for x in sm["ans"]], return_counts=True)
            q = cnt / cnt.sum(); ent.append(float(-(q * np.log(q)).sum()))
        else:
            sft.append(np.nan); ent.append(np.nan)
        tf.append(_answer_ll(c.get("tf")))
        ans.append(c.get("ans"))
    g, sft, ent, tf = map(np.array, (g, sft, ent, tf))

    def commit(v, thr):
        cp = 1.0
        for j in range(len(v) - 1, -1, -1):
            if v[j] >= thr:
                cp = ([*GRID, 1.0])[j]
            else:
                break
        return cp if v[-1] >= thr else 1.1

    sw = sum(1 for x, y in zip(ans, ans[1:]) if x != y)
    t2 = np.where(np.isfinite(tf), tf, np.nanmin(tf) if np.isfinite(tf).any() else -20)
    h = len(t2) // 2
    sh = np.nanmean(sft) if np.isfinite(sft).any() else np.nan
    summ = [np.nanmean(g), sh, commit(g, 1), commit(np.nan_to_num(sft), 0.5), sw,
            len({x for x in ans if x is not None}), np.nanmean(ent) if np.isfinite(ent).any() else np.nan,
            t2.mean(), t2[h:].mean() - t2[:h].mean(), _mean(end.get("tok_lps"))]
    return np.concatenate([g, sft, summ])


def trac0_score(rec) -> float:
    """TrAC-0: the share of re-elicited answers (greedy and sampled, pooled per grid point)
    that equal the final answer, averaged over the trace grid."""
    vals = []
    for c in _grid_points(rec):
        ag = ([float(c["agree"])] if "agree" in c else []) + [float(x) for x in (c.get("samp") or {}).get("agree", [])]
        if ag:
            vals.append(np.mean(ag))
    return float(np.mean(vals)) if vals else np.nan


def _fresh(cache: Path, sources: List[Path]) -> bool:
    return cache.exists() and cache.stat().st_mtime > max(max(s.stat().st_mtime for s in sources),
                                                          os.path.getmtime(__file__))


def trajectory_blocks(tag: str, ids) -> Dict[str, np.ndarray]:
    """TRAJ / TRAC0 blocks (and their ablations) aligned to `ids`; {} without a trajectory."""
    src = C.TRAJECTORY / f"{tag}.json.gz"
    if not src.exists():
        return {}
    cache = C.FEATURE_CACHE / f"trajectory__{tag}.npz"
    if _fresh(cache, [src]):
        return dict(np.load(cache))
    R = {x["id"]: x for x in read_json_gz(src)}
    n = len(ids)
    d = 2 * (len(GRID) + 1) + 10
    T, Tg, T8, T4 = (np.full((n, d), np.nan) for _ in range(4))
    S, Sg, S8 = (np.full(n, np.nan) for _ in range(3))
    for i, q in enumerate(ids):
        r = R.get(str(q))
        if r is None:
            continue
        T[i] = trajectory_features(r); S[i] = trac0_score(r)
        g = greedy_only(r)
        Tg[i] = trajectory_features(g); Sg[i] = trac0_score(g)
        T8[i] = trajectory_features(thin(r, 8)); S8[i] = trac0_score(thin(r, 8))
        T4[i] = trajectory_features(thin(r, 4))
    blk = dict(TRAJ=T, TRAC0=S[:, None], TRAJ_G=Tg, TRAC0_G=Sg[:, None], TRAJ_8=T8, TRAC0_8=S8[:, None], TRAJ_4=T4)
    save_npz_atomic(cache, **blk)
    return blk


def textpos_block(tag: str, ids) -> Dict[str, np.ndarray]:
    """Passive text-position baseline (no decoding): first and last relative position of the
    final answer string in the reasoning body, log(1 + count), whether it appears, and the
    number of distinct boxed candidates in the body. Undefined for option letters (NaN)."""
    cache = C.FEATURE_CACHE / f"textpos__{tag}.npz"
    gf = C.GEN / f"{tag}.json.gz"
    if not gf.exists():                        # probe-temperature variants share another tag's generation
        return {}
    if _fresh(cache, [gf]):
        return dict(np.load(cache))
    G = {r["id"]: r for r in read_json_gz(gf)}
    bb = M.from_tag(tag)
    out = np.full((len(ids), 5), np.nan)
    for i, q in enumerate(ids):
        r = G.get(str(q))
        if r is None:
            continue
        body = M.reasoning_body(bb, r["raw"])
        boxed = set(m.strip() for m in re.findall(r"\\boxed\{([^{}]{1,40})\}", body))
        out[i, 4] = len(boxed)
        a = A.last_boxed(r.get("final") or "") or (str(r["answer"]) if r["answer"] is not None else None)
        if r.get("type") == "mc" or not a or len(a.strip()) < 1:
            continue
        a = a.strip()
        pos = [m.start() for m in re.finditer(re.escape(a), body)]
        L = max(len(body), 1)
        out[i, 0] = pos[0] / L if pos else 1.1
        out[i, 1] = pos[-1] / L if pos else 1.1
        out[i, 2] = np.log1p(len(pos))
        out[i, 3] = float(bool(pos))
    blk = {"TEXTPOS": out}
    save_npz_atomic(cache, **blk)
    return blk


def tsc_block(tag: str, ids) -> Dict[str, np.ndarray]:
    """Trajectory-weighted consensus over the K = 8 traces: trace k votes for its final answer
    with weight w_k = share of its greedy re-elicited answers (8 checkpoints + end) equal to it
    (+0.01). Columns: weighted vote share of the original answer, weighted share among the
    agreeing traces, mean weight. {} without the sample trajectories."""
    fs, f2, gf = C.SAMPLE_TRAJECTORY / f"{tag}.json.gz", C.TRAJECTORY / f"{tag}.json.gz", C.GEN / f"{tag}.json.gz"
    if not (fs.exists() and f2.exists()):
        return {}
    cache = C.FEATURE_CACHE / f"tsc__{tag}.npz"
    if _fresh(cache, [fs, f2, gf]):
        return dict(np.load(cache))
    TS = {x["id"]: x["samples"] for x in read_json_gz(fs)}
    R = {x["id"]: x for x in read_json_gz(f2)}
    G = {r["id"]: r for r in read_json_gz(gf)}
    out = np.full((len(ids), 3), np.nan)
    for i, q in enumerate(ids):
        q = str(q); r = G.get(q); rec = R.get(q)
        if r is None or rec is None:
            continue
        ws, votes = [np.nan_to_num(trac0_score(thin(greedy_only(rec), 8)))], [1.0]
        by_k = {x["k"]: x for x in TS.get(q, [])}
        for k, smp in enumerate(r["samples"]):
            x = by_k.get(k)
            ws.append(float(np.mean(x["agree_own"])) if x and x["agree_own"] else 0.0)
            votes.append(float(A.equiv(r, smp["answer"], r["answer"])))
        ws, votes = np.array(ws) + 0.01, np.array(votes)
        out[i] = [(ws * votes).sum() / ws.sum(), (ws * votes).sum() / max(votes.sum(), 1e-9), ws.mean()]
    blk = {"TSC": out}
    save_npz_atomic(cache, **blk)
    return blk
