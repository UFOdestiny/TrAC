"""Per-setting feature blocks (CPU): generation + endpoint caches -> EXP_ROOT/features/{tag}.npz.

Every answer is re-parsed from the stored raw text, so parser fixes never need a GPU rerun.

Blocks (rows aligned with ids; labels from the returned answer section):
  y          correctness (rule-based answer checker; used only for training and evaluation)
  AGREE      intermediate answer at the end of the trace equals a                     (1)
  FLL        its token log-likelihood: mean, min, head (first 2 tokens)               (3)
  TRACE      token log-probability and entropy profile (8 + 8 bins, 11 statistics)    (27)
  ANS        answer section of the original response: mean/min chosen log-prob and mean
             entropy, and mean/min chosen log-prob over the boxed answer               (5)
  META       n_tokens, reasoning closed, finish == length                             (3)
  SC         vote share of a among the K samples, modal vote share, answer entropy     (3)
  SAMP       sample block (9): columns 1, 2, 8 are the discrete and likelihood-weighted answer
             entropy (negated) and the likelihood-weighted vote share of a (REWIND@8); the other
             columns are NLI-based baselines, NaN unless baselines/ computes them

With the baselines package present, its per-response blocks (baselines.features) are added to the same
file. The intermediate-answer and runner-up blocks are derived by rewind.features.intermediate_answers
and rewind.features.runner_up.

  python -m rewind.features.build [--tags math500_qwen8b_k8 ...]   (default: every endpoint cache)
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from rewind import answers as A
from rewind import backbones as M
from rewind import config as C
from rewind.optional import optional_module
from rewind.store import read_json_gz, save_npz_atomic

NBIN = 8


def _binned(a, nbin=NBIN):
    if len(a) == 0:
        return [0.0] * nbin
    idx = np.linspace(0, len(a), nbin + 1).astype(int)
    return [
        float(np.mean(a[idx[i] : idx[i + 1]])) if idx[i + 1] > idx[i] else float(a[min(idx[i], len(a) - 1)]) for i in range(nbin)
    ]


def _slope_r2(a):
    n = len(a)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n)
    b1, b0 = np.polyfit(x, a, 1)
    pred = b1 * x + b0
    ss_res = np.sum((a - pred) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(b1), float(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)


def trace_profile(lp, ent):
    """Token profile: 8 binned log-prob and entropy means, then level, spread, slope, linearity,
    tail, and low-confidence statistics."""
    if len(lp) == 0:
        return [0.0] * (2 * NBIN + 11)
    f = _binned(lp) + (_binned(ent) if len(ent) else [0.0] * NBIN)
    f += [float(np.mean(lp)), float(np.min(lp)), float(np.std(lp))]
    f += [float(np.mean(ent)) if len(ent) else 0.0, float(np.max(ent)) if len(ent) else 0.0]
    s_lp, r_lp = _slope_r2(lp)
    s_e, r_e = _slope_r2(ent) if len(ent) else (0.0, 0.0)
    f += [s_lp, r_lp, s_e, r_e, float(np.mean(lp[-30:])), float(np.mean(lp < -2.0))]
    return f


def clusters(items, same):
    reps, lab = [], []
    for x in items:
        for k, r in enumerate(reps):
            if same(x, r):
                lab.append(k)
                break
        else:
            reps.append(x)
            lab.append(len(reps) - 1)
    return np.array(lab)


def entropy_of(lab, w=None):
    w = np.ones(len(lab)) if w is None else np.asarray(w, float)
    p = np.bincount(lab, weights=w) / w.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def samp_block(g, answers, meanlps, nli=None):
    """Sample block of one question over its K answers (original first); `nli` fills the NLI columns."""
    lab = clusters(answers, lambda a, b: a is not None and b is not None and A.equiv(g, a, b))
    lab = np.where([a is None for a in answers], -1, lab)
    valid = lab >= 0
    se_d = entropy_of(lab[valid]) if valid.any() else 0.0
    w = np.exp(np.nan_to_num(np.asarray(meanlps, float), nan=-5.0))
    se_w = entropy_of(lab[valid], w[valid]) if valid.any() else 0.0
    cw = np.exp((np.nan_to_num(np.asarray(meanlps, float), nan=-5.0) - np.nanmax(meanlps)) / 0.1)
    cisc = float(cw[lab == lab[0]].sum() / cw.sum()) if lab[0] >= 0 else 0.0
    if nli is None:
        return [np.nan, -se_d, -se_w, np.nan, np.nan, np.nan, np.nan, np.nan, cisc]
    selfcheck, se_nli, *graph = nli
    return [selfcheck, -se_d, -se_w, se_nli, *graph, cisc]


def answer_region(tok, bb, out_ids, raw, lp, ent):
    """Token confidence over the returned answer section and over the boxed answer."""
    nan5 = [np.nan] * 5
    if tok is None or len(out_ids) == 0:
        return nan5
    c = M.reasoning_end_char(bb, raw)
    if c is None:
        return nan5
    ids = out_ids.tolist()
    k0 = M.token_cut(tok, ids, c)
    b = raw.rfind("\\boxed{")
    kb = M.token_cut(tok, ids, b) if b >= c else None
    a_lp, a_en = lp[k0:], ent[k0:]
    a_lp, a_en = a_lp[np.isfinite(a_lp)], a_en[np.isfinite(a_en)]
    if len(a_lp) == 0:
        return nan5
    out = [float(a_lp.mean()), float(a_lp.min()), float(a_en.mean()) if len(a_en) else np.nan]
    if kb is not None:
        bl = lp[kb : kb + 12]
        bl = bl[np.isfinite(bl)]
        out += [float(bl.mean()), float(bl.min())] if len(bl) else [np.nan, np.nan]
    else:
        out += [np.nan, np.nan]
    return out


def lp_stats(rec):
    """mean, min, head(first 2), first-token loglik of a query record (or NaN)."""
    if not rec or not rec.get("tok_lps"):
        return [np.nan] * 4
    t = np.asarray([x for x in rec["tok_lps"] if x is not None], float)
    if len(t) == 0:
        return [np.nan] * 4
    return [float(t.mean()), float(t.min()), float(t[:2].mean()), float(t[0])]


def votes(row, answers, primary):
    ans = [a for a in answers if a is not None]
    if not ans:
        return [0.0, 0.0, 0.0]
    reps, cnt = [], []
    for a in ans:
        for k, r in enumerate(reps):
            if A.equiv(row, a, r):
                cnt[k] += 1
                break
        else:
            reps.append(a)
            cnt.append(1)
    tot = len(answers)
    p = np.array(cnt) / tot
    prim = sum(1 for a in answers if A.equiv(row, a, primary)) / tot if primary is not None else 0.0
    return [float(prim), float(p.max()), float(-(p * np.log(p)).sum())]


def build(tag):
    """Features of the generation and endpoint caches of `tag`."""
    gen = read_json_gz(C.GEN / f"{tag}.json.gz")
    z = np.load(C.GEN / f"{tag}.tok.npz")
    oo = np.concatenate([[0], np.cumsum(z["out_len"])])
    ep = {r["id"]: r for r in read_json_gz(C.ENDPOINT / f"{tag}.json.gz")}
    extra = optional_module("baselines.features", "the baseline feature blocks")
    nli = extra.load_nli(tag) if extra else {}
    bb = M.from_tag(tag)
    tok = M.load_tokenizer(bb) if bb is not None else None
    names = ["y", "AGREE", "FLL", "TRACE", "PT", "SC", "BASE", "META", "SD", "SAMP", "ANS"]
    B = {k: [] for k in names if extra or k not in ("PT", "BASE", "SD")}
    ids = []
    for i, g in enumerate(gen):
        e = ep.get(g["id"])
        if e is None:
            continue
        lp = z["chosen_lp"][oo[i] : oo[i + 1]].astype(float)
        ent = z["ent"][oo[i] : oo[i + 1]].astype(float)
        tm = z["topk_mean"][oo[i] : oo[i + 1]].astype(float)
        lp, ent, tm = lp[np.isfinite(lp)], ent[np.isfinite(ent)], tm[np.isfinite(tm)]
        # re-parse everything from stored raw text, so parser fixes never need a GPU rerun
        if bb is not None:
            g = {
                **g,
                "final": M.final_for(bb, g["raw"], g["finish"]),
                "samples": [{**x, "final": M.final_for(bb, x["raw"], x["finish"])} for x in g["samples"]],
            }
        ans = A.extract_answer(g, g["final"])
        end = e.get("end")
        if end:  # re-parse the intermediate answer at the end of the trace
            end = {**end, "agree": bool(A.equiv(g, A.extract_head(g, end["text"]), ans))}
        m, mn, hd, _ = lp_stats(end)
        B["y"].append(A.verify_answer_text(g, g["final"]))
        B["AGREE"].append([float(end["agree"]) if end else np.nan])
        B["FLL"].append([m, mn, hd])
        B["TRACE"].append(trace_profile(lp, ent))
        answers = [ans] + [A.extract_answer(g, s["final"]) for s in g["samples"]]
        B["SC"].append(votes(g, answers, ans))
        ne = nli.get(g["id"])
        B["SAMP"].append(
            samp_block(
                g,
                answers,
                [float(lp.mean()) if len(lp) else np.nan]
                + [s["meanlp"] if s["meanlp"] is not None else np.nan for s in g["samples"]],
                extra.nli_columns(*ne) if ne is not None else None,
            )
        )
        if extra:
            for k, row in extra.response_blocks(g, e, lp, ent, tm, bb).items():
                B[k].append(row)
        B["META"].append([g["n_tokens"], float(g["reason_closed"]), float(g["finish"] == "length")])
        B["ANS"].append(
            answer_region(
                tok, bb, z["out_ids"][oo[i] : oo[i + 1]], g["raw"], z["chosen_lp"][oo[i] : oo[i + 1]], z["ent"][oo[i] : oo[i + 1]]
            )
        )
        ids.append(g["id"])
    out = C.FEATURES / f"{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(out, ids=np.array(ids), **{k: np.asarray(v, float) for k, v in B.items()})
    y = np.asarray(B["y"])
    print(f"[feat] {tag}: n={len(y)} acc={y.mean():.3f} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args()
    tags = args.tags or sorted(os.path.basename(p)[: -len(".json.gz")] for p in glob.glob(str(C.ENDPOINT / "*.json.gz")))
    for t in tags:
        build(t)


if __name__ == "__main__":
    main()
