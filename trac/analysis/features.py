"""Per-cell feature blocks (CPU): generation + endpoint (+ NLI) caches -> EXP_ROOT/features/{tag}.npz.

Every answer is re-parsed from the stored raw text, so parser fixes never need a GPU rerun.

Blocks (rows aligned with ids; labels from the returned answer section):
  y          correctness
  AGREE      PCE agreement e                                  (1)
  FLL        PCE answer loglik: mean, min, head (first 2)      (3)
  FCONF      PCE first-token log-confidence c1                 (1)
  TRACE      TUP profile (8 lp bins, 8 entropy bins, 11 stats) (27)
  CUES       agreement + mean loglik under cues 1, 2           (4)
  QONLY      question-only probe: agreement, mean loglik, c1   (3)
  SHUF       shuffled-trace probe: agreement, mean loglik, c1  (3)
  TFORCE     teacher-forced returned answer: mean, min, first  (3)
  REPEAT     re-answer control: agree frac, mean lp0, n distinct (3)
  PT         P(True)                                           (1)
  SC         primary-answer vote share q_K(a), modal share q_K, answer entropy h_K (3)
  BASE       mean logprob, self-certainty, DeepConf-bottom     (3)
  META       n_tokens, reason_closed, finish==length            (3)
  SD         SelfDoubt inputs: -HVR (hedge/(verify+1)), verbalized confidence V  (2)
  ANS        returned-answer region of the primary: mean/min chosen logprob and mean entropy
             over the answer section, and mean/min chosen logprob over the boxed answer (5)
  CONTRAST   teacher-forced support of the returned answer minus PCE likelihood of the
             re-elicited answer, and agreement x PCE likelihood (2)
  SAMP       confidences (higher = more confident) from the K samples:
             SelfCheckGPT-NLI, -SE (discrete), -SE (likelihood-weighted), -SE (NLI
             clusters), -EigV, Ecc, Deg, -KLE, CISC weighted vote   (9; NaN without nli cache)

The trajectory blocks (TRAJ, TrAC-0) are derived separately by trac.analysis.trajectory_features.

  python -m trac.analysis.features [--tags math500_qwen8b_k8 ...]   (default: every endpoint cache)
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.store import read_json_gz

NBIN = 8


def _binned(a, nbin=NBIN):
    if len(a) == 0:
        return [0.0] * nbin
    idx = np.linspace(0, len(a), nbin + 1).astype(int)
    return [float(np.mean(a[idx[i]:idx[i + 1]])) if idx[i + 1] > idx[i] else float(a[min(idx[i], len(a) - 1)])
            for i in range(nbin)]


def _slope_r2(a):
    n = len(a)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n); b1, b0 = np.polyfit(x, a, 1); pred = b1 * x + b0
    ss_res = np.sum((a - pred) ** 2); ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(b1), float(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)


def trace_profile(lp, ent):
    """TUP profile: 8 binned log-prob and entropy means, then level, spread, slope, linearity,
    tail, and low-confidence statistics."""
    if len(lp) == 0:
        return [0.0] * (2 * NBIN + 11)
    f = _binned(lp) + (_binned(ent) if len(ent) else [0.0] * NBIN)
    f += [float(np.mean(lp)), float(np.min(lp)), float(np.std(lp))]
    f += [float(np.mean(ent)) if len(ent) else 0.0, float(np.max(ent)) if len(ent) else 0.0]
    s_lp, r_lp = _slope_r2(lp); s_e, r_e = _slope_r2(ent) if len(ent) else (0.0, 0.0)
    f += [s_lp, r_lp, s_e, r_e, float(np.mean(lp[-30:])), float(np.mean(lp < -2.0))]
    return f


def deepconf_bottom(conf, frac=0.1):
    n = len(conf)
    if n == 0:
        return -1e9
    w = max(1, int(round(frac * n)))
    if w >= n:
        return float(conf.mean())
    cs = np.cumsum(np.insert(conf, 0, 0.0))
    return float(((cs[w:] - cs[:-w]) / w).min())


# SelfDoubt seed lexicons (Pandey et al., 2026, Table 21); the paper's per-model
# n-gram expansion needs an unlabeled calibration set and is not reproduced.
HEDGE = ["possibly", "seemingly", "maybe", "apparently", "probably", "presumably",
         "perhaps", "likely", "reportedly", "seems"]
VERIFY = ["check", "reassess", "reevaluate", "re-evaluate", "reinspect", "rechecking",
          "verifying", "reconfirming", "recheck", "prove"]
_HRE = re.compile(r"\b(" + "|".join(map(re.escape, sorted(HEDGE, key=len, reverse=True))) + r")\b", re.I)
_VRE = re.compile(r"\b(" + "|".join(map(re.escape, sorted(VERIFY, key=len, reverse=True))) + r")\b", re.I)


def neg_hvr(reasoning):
    return -len(_HRE.findall(reasoning)) / (len(_VRE.findall(reasoning)) + 1)


def clusters(items, same):
    reps, lab = [], []
    for x in items:
        for k, r in enumerate(reps):
            if same(x, r):
                lab.append(k); break
        else:
            reps.append(x); lab.append(len(reps) - 1)
    return np.array(lab)


def entropy_of(lab, w=None):
    w = np.ones(len(lab)) if w is None else np.asarray(w, float)
    p = np.bincount(lab, weights=w) / w.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def graph_confidences(E):
    """EigV, Ecc, Deg (Lin et al.) and KLE (Nikitin et al.) from directional entailment."""
    W = (E + E.T) / 2
    d = W.sum(1)
    L = np.eye(len(W)) - W / np.sqrt(np.outer(d, d))
    lam, U = np.linalg.eigh(L)
    eigv = float(np.clip(1 - lam, 0, None).sum())
    k = max(1, int((lam < 0.9).sum()))
    V = U[:, :k]
    ecc = -float(np.linalg.norm(V[0] - V.mean(0)))
    deg = float(W[0, 1:].mean())
    lam_g, U_g = np.linalg.eigh(np.diag(d) - W)        # KLE: graph Laplacian D - W (Nikitin et al.)
    heat = (U_g * np.exp(-0.3 * lam_g)) @ U_g.T        # heat kernel exp(-tL), t = 0.3
    mu = np.clip(np.linalg.eigvalsh(heat / np.trace(heat)), 1e-12, None)
    kle = float(-(mu * np.log(mu)).sum())
    return -eigv, ecc, deg, -kle


def samp_block(g, answers, meanlps, nli_E, nli_sc):
    lab = clusters(answers, lambda a, b: a is not None and b is not None and A.equiv(g, a, b))
    lab = np.where([a is None for a in answers], -1, lab)
    valid = lab >= 0
    se_d = entropy_of(lab[valid]) if valid.any() else 0.0
    w = np.exp(np.nan_to_num(np.asarray(meanlps, float), nan=-5.0))
    se_w = entropy_of(lab[valid], w[valid]) if valid.any() else 0.0
    cw = np.exp((np.nan_to_num(np.asarray(meanlps, float), nan=-5.0) - np.nanmax(meanlps)) / 0.1)
    cisc = float(cw[lab == lab[0]].sum() / cw.sum()) if lab[0] >= 0 else 0.0
    if nli_E is None:
        return [np.nan, -se_d, -se_w, np.nan, np.nan, np.nan, np.nan, np.nan, cisc]
    E = np.nan_to_num(nli_E, nan=0.0)
    nl = clusters(list(range(len(E))), lambda a, b: E[a, b] > 0.5 and E[b, a] > 0.5)
    return [-float(nli_sc), -se_d, -se_w, -entropy_of(nl), *graph_confidences(E), cisc]


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
        bl = lp[kb:kb + 12]
        bl = bl[np.isfinite(bl)]
        out += [float(bl.mean()), float(bl.min())] if len(bl) else [np.nan, np.nan]
    else:
        out += [np.nan, np.nan]
    return out


def lp_stats(rec):
    """mean, min, head(first 2), first-token loglik of a probe record (or NaN)."""
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
                cnt[k] += 1; break
        else:
            reps.append(a); cnt.append(1)
    tot = len(answers)
    p = np.array(cnt) / tot
    prim = sum(1 for a in answers if A.equiv(row, a, primary)) / tot if primary is not None else 0.0
    return [float(prim), float(p.max()), float(-(p * np.log(p)).sum())]


def build(tag, gen_tag=None):
    """Features for endpoint cache `tag`; `gen_tag` names its generation cache when the
    two differ (e.g. a probe-temperature variant `X_k8_pt0.7` of generation `X_k8`)."""
    gen_tag = gen_tag or tag
    gen = read_json_gz(C.GEN / f"{gen_tag}.json.gz")
    z = np.load(C.GEN / f"{gen_tag}.tok.npz")
    oo = np.concatenate([[0], np.cumsum(z["out_len"])])
    ep = {r["id"]: r for r in read_json_gz(C.ENDPOINT / f"{tag}.json.gz")}
    nf = C.NLI / f"{gen_tag}.npz"
    nli = {}
    if nf.exists():
        zn = np.load(nf)
        nli = {i: (zn["E"][k], zn["selfcheck"][k]) for k, i in enumerate(zn["ids"])}
    bb = M.from_tag(gen_tag)
    tok = M.load_tokenizer(bb) if bb is not None else None
    B = {k: [] for k in ["y", "AGREE", "FLL", "FCONF", "TRACE", "CUES", "QONLY", "SHUF", "TFORCE",
                         "REPEAT", "PT", "SC", "BASE", "META", "SD", "SAMP", "ANS", "CONTRAST"]}
    ids = []
    for i, g in enumerate(gen):
        e = ep.get(g["id"])
        if e is None:
            continue                                   # (PCE may be absent before stage 2: NaN features)
        lp = z["chosen_lp"][oo[i]:oo[i + 1]].astype(float)
        ent = z["ent"][oo[i]:oo[i + 1]].astype(float)
        tm = z["topk_mean"][oo[i]:oo[i + 1]].astype(float)
        lp, ent, tm = lp[np.isfinite(lp)], ent[np.isfinite(ent)], tm[np.isfinite(tm)]
        # re-parse everything from stored raw text, so parser fixes never need a GPU rerun
        if bb is not None:
            g = {**g, "final": M.final_for(bb, g["raw"], g["finish"]),
                 "samples": [{**x, "final": M.final_for(bb, x["raw"], x["finish"])} for x in g["samples"]]}
        ans = A.extract_answer(g, g["final"])
        reparse = lambda rec: None if not rec else {**rec, "agree": bool(A.equiv(g, A.extract_head(g, rec["text"]), ans))}
        e = {**e, "pce": [reparse(x) for x in (e.get("pce") or [None])], "qonly": reparse(e.get("qonly")),
             "shuffled": reparse(e.get("shuffled")),
             "repeat": [{**x, "agree": bool(A.equiv(g, A.extract_head(g, x["text"]), ans))} if "text" in x else x
                        for x in (e.get("repeat") or [])]}
        p0 = e["pce"][0]
        m, mn, hd, c1 = lp_stats(p0)
        B["y"].append(A.verify_answer_text(g, g["final"]))
        B["AGREE"].append([float(p0["agree"]) if p0 else np.nan])
        B["FLL"].append([m, mn, hd]); B["FCONF"].append([c1])
        B["TRACE"].append(trace_profile(lp, ent))
        cues = []
        for c in (1, 2):
            pc = e["pce"][c] if len(e["pce"]) > c else None
            cues += [float(pc["agree"]) if pc else np.nan, lp_stats(pc)[0]]
        B["CUES"].append(cues)
        for key, blk in (("qonly", "QONLY"), ("shuffled", "SHUF")):
            r = e.get(key)
            s = lp_stats(r)
            B[blk].append([float(r["agree"]) if r else np.nan, s[0], s[3]])
        tfr = e.get("tforce")
        if tfr and len(tfr.get("tok_lps") or []) > 1:   # drop the closing-brace token
            tfr = {**tfr, "tok_lps": tfr["tok_lps"][:-1]}
        tf = lp_stats(tfr)
        B["TFORCE"].append([tf[0], tf[1], tf[3]])
        rep = e.get("repeat") or []
        if rep:
            a = [x["agree"] for x in rep]; l0 = [x["lp0"] for x in rep if x["lp0"] is not None]
            nd = len({str(x["answer"]) for x in rep if x["answer"] is not None})
            B["REPEAT"].append([float(np.mean(a)), float(np.mean(l0)) if l0 else np.nan, float(nd)])
        else:
            B["REPEAT"].append([np.nan] * 3)
        B["PT"].append([e.get("ptrue", np.nan)])
        answers = [ans] + [A.extract_answer(g, s["final"]) for s in g["samples"]]
        B["SC"].append(votes(g, answers, ans))
        ne, ns = nli.get(g["id"], (None, None))
        B["SAMP"].append(samp_block(g, answers, [float(lp.mean()) if len(lp) else np.nan]
                                    + [s["meanlp"] if s["meanlp"] is not None else np.nan for s in g["samples"]],
                                    ne, ns))
        c = M.reasoning_end_char(bb, g["raw"]) if bb else None
        vc = (e.get("verbconf") or {}).get("value")
        B["SD"].append([neg_hvr(g["raw"][:c] if c is not None else g["raw"]),
                        vc if vc is not None else np.nan])
        B["BASE"].append([float(lp.mean()) if len(lp) else -1e9,
                          -float(ent.mean()) if len(ent) else 0.0,
                          deepconf_bottom(-tm)])
        B["META"].append([g["n_tokens"], float(g["reason_closed"]), float(g["finish"] == "length")])
        B["ANS"].append(answer_region(tok, bb, z["out_ids"][oo[i]:oo[i + 1]], g["raw"],
                                      z["chosen_lp"][oo[i]:oo[i + 1]], z["ent"][oo[i]:oo[i + 1]]))
        B["CONTRAST"].append([tf[0] - m if np.isfinite(tf[0]) and np.isfinite(m) else np.nan,
                              (float(p0["agree"]) * m) if p0 and np.isfinite(m) else np.nan])
        ids.append(g["id"])
    out = C.FEATURES / f"{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids), **{k: np.asarray(v, float) for k, v in B.items()})
    y = np.asarray(B["y"])
    print(f"[feat] {tag}: n={len(y)} acc={y.mean():.3f} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args()
    tags = args.tags or sorted(os.path.basename(p)[:-len(".json.gz")]
                               for p in glob.glob(str(C.ENDPOINT / "*.json.gz")))
    for t in tags:
        g = t.split("_pt")[0] if "_pt" in t else t
        if (C.GEN / f"{g}.json.gz").exists():
            build(t, g)


if __name__ == "__main__":
    main()
