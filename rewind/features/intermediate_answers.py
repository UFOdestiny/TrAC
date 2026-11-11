"""Intermediate-answer and answer-likelihood feature blocks (CPU), derived from the intermediate-answer
cache (rewind.pipeline.intermediate_answers).

Each record is resampled to a fixed grid of 20 fractions of the trace plus its end (nearest checkpoint
per grid point).

  INTERMEDIATE     per grid point: greedy agreement with a, and the share of sampled answers equal to a;
           summaries: mean greedy agreement, mean sampled share, convergence points (earliest
           fraction from which every later answer agrees), answer switches, distinct answers,
           mean sampled-answer entropy, level and late-half rise of the likelihood of a, and the
           log-likelihood of the end answer                                                (52)
  TFT      at the end of the trace: likelihood of a (mean, min, first token), its difference to
           the greedy intermediate answer's likelihood, and agreement x that likelihood    (5)
  INTERMEDIATE_4   INTERMEDIATE from 4 evenly spaced checkpoints (the paper's configuration)
  INTERMEDIATE_G   INTERMEDIATE from greedy answers only (ablation)
  INTERMEDIATE_G4  4 checkpoints, greedy answers only (minimal configuration)

Blocks are cached under EXP_ROOT/feature_cache and rebuilt whenever the source or this file is
newer than the cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from rewind import config as C
from rewind.store import read_json_gz, save_npz_atomic

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


def intermediate_answer_features(rec) -> np.ndarray:
    """INTERMEDIATE row of one record (any number of checkpoints, optional sampled answers)."""
    end = rec["end"] or {}
    g, sft, ent, tf, ans = [], [], [], [], []
    for c in _grid_points(rec):
        g.append(float(c.get("agree", np.nan)) if c else np.nan)
        sm = c.get("samp")
        if sm and sm["agree"]:
            sft.append(np.mean(sm["agree"]))
            _, cnt = np.unique([str(x) for x in sm["ans"]], return_counts=True)
            q = cnt / cnt.sum()
            ent.append(float(-(q * np.log(q)).sum()))
        else:
            sft.append(np.nan)
            ent.append(np.nan)
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
    summ = [
        np.nanmean(g),
        sh,
        commit(g, 1),
        commit(np.nan_to_num(sft), 0.5),
        sw,
        len({x for x in ans if x is not None}),
        np.nanmean(ent) if np.isfinite(ent).any() else np.nan,
        t2.mean(),
        t2[h:].mean() - t2[:h].mean(),
        _mean(end.get("tok_lps")),
    ]
    return np.concatenate([g, sft, summ])


def _lp_stats(v):
    v = [x for x in (v or []) if x is not None]
    return [float(np.mean(v)), float(np.min(v)), float(v[0])] if v else [np.nan] * 3


def end_support(rec) -> np.ndarray:
    """TFT row: the likelihood of the final answer at the end of the trace, from the same pass."""
    end = rec["end"] or {}
    tf = end.get("tf") or []
    t = _lp_stats(tf[:-1] if len(tf) > 1 else tf)  # drop the closing-brace token
    m = _lp_stats(end.get("tok_lps"))[0]
    ag = float(end["agree"]) if "agree" in end else np.nan
    return np.array(t + [t[0] - m if np.isfinite(t[0]) and np.isfinite(m) else np.nan, ag * m])


def agreement_profile(rec) -> np.ndarray:
    """Share of intermediate answers (greedy and sampled) equal to a at every grid point and the end."""
    out = []
    for c in _grid_points(rec):
        ag = ([float(c["agree"])] if "agree" in c else []) + [float(x) for x in (c.get("samp") or {}).get("agree", [])]
        out.append(np.mean(ag) if ag else np.nan)
    return np.array(out)


def answer_share(rec) -> float:
    """Mean share of intermediate answers equal to a over the grid (training-free)."""
    v = agreement_profile(rec)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else np.nan


def _fresh(cache: Path, sources: List[Path]) -> bool:
    return cache.exists() and cache.stat().st_mtime > max(max(s.stat().st_mtime for s in sources), os.path.getmtime(__file__))


def intermediate_answer_blocks(tag: str, ids) -> Dict[str, np.ndarray]:
    """INTERMEDIATE, TFT and the INTERMEDIATE variants aligned to `ids`; {} without an intermediate-answer cache."""
    src = C.INTERMEDIATE_ANSWERS / f"{tag}.json.gz"
    if not src.exists():
        return {}
    cache = C.FEATURE_CACHE / f"intermediate_answers__{tag}.npz"
    if _fresh(cache, [src]):
        return dict(np.load(cache))
    R = {x["id"]: x for x in read_json_gz(src)}
    n = len(ids)
    d = 2 * (len(GRID) + 1) + 10
    T, Tg, T4, Tg4 = (np.full((n, d), np.nan) for _ in range(4))
    E = np.full((n, 5), np.nan)
    for i, q in enumerate(ids):
        r = R.get(str(q))
        if r is None:
            continue
        g = greedy_only(r)
        T[i] = intermediate_answer_features(r)
        E[i] = end_support(r)
        Tg[i] = intermediate_answer_features(g)
        T4[i] = intermediate_answer_features(thin(r, 4))
        Tg4[i] = intermediate_answer_features(thin(g, 4))
    blk = dict(INTERMEDIATE=T, TFT=E, INTERMEDIATE_G=Tg, INTERMEDIATE_4=T4, INTERMEDIATE_G4=Tg4)
    save_npz_atomic(cache, **blk)
    return blk
