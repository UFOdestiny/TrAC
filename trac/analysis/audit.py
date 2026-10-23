"""Data-quality audit of finished generation cells (CPU).

Per cell: accuracy (original trace, re-parsed), finish/closure/no-answer rates, token-length
stats, finiteness of token arrays, sample answers and labels, endpoint baseline probe
coverage (P(True), verbalized confidence), hidden-state and NLI cache sanity, and
repetition-loop detection on truncated original traces.

  python -m trac.analysis.audit [--tags ...]      (default: every main-matrix cell)
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.store import read_json_gz


def loopy(text, n=200):
    """True if the tail of a text is a repeated block (degenerate generation)."""
    t = text[-4 * n:]
    return len(t) >= 4 * n and t[-n:] in t[:-n]


def audit(tag):
    rows = read_json_gz(C.GEN / f"{tag}.json.gz")
    bb = M.from_tag(tag)
    z = np.load(C.GEN / f"{tag}.tok.npz")
    fin = [M.final_for(bb, r["raw"], r["finish"]) for r in rows]
    y = np.array([A.verify_answer_text(r, f) for r, f in zip(rows, fin)])
    ans = [A.extract_answer(r, f) for r, f in zip(rows, fin)]
    ntok = np.array([r["n_tokens"] for r in rows])
    trunc = np.array([r["finish"] == "length" for r in rows])
    closed = np.array([M.reasoning_end_char(bb, r["raw"]) is not None for r in rows])
    s_fin = [s["finish"] == "length" for r in rows for s in r["samples"]]
    s_ans = [A.extract_answer(r, M.final_for(bb, s["raw"], s["finish"])) is None for r in rows for s in r["samples"]]
    s_cor = [s["correct"] for r in rows for s in r["samples"]]
    K = 1 + np.mean([len(r["samples"]) for r in rows])
    out = {"n": len(rows), "K": K, "acc": y.mean(), "sample_acc": np.mean(s_cor),
           "trunc": trunc.mean(), "sample_trunc": np.mean(s_fin), "closed": closed.mean(),
           "no_ans": np.mean([a is None for a in ans]), "sample_no_ans": np.mean(s_ans),
           "no_ans_but_finished": np.mean([a is None and not t for a, t in zip(ans, trunc)]),
           "tok_med": float(np.median(ntok)), "tok_p95": float(np.percentile(ntok, 95)), "tok_max": int(ntok.max()),
           "loop_in_trunc": float(np.mean([loopy(r["raw"]) for r, t in zip(rows, trunc) if t])) if trunc.any() else 0.0,
           "lp_finite": float(np.isfinite(z["chosen_lp"]).mean()), "ent_mean": float(np.nanmean(z["ent"])),
           "lp_mean": float(np.nanmean(z["chosen_lp"])), "len_match": bool(z["out_len"].sum() == len(z["out_ids"]))}
    ef = C.ENDPOINT / f"{tag}.json.gz"
    if ef.exists():
        ep = read_json_gz(ef)
        pt = [e.get("ptrue") for e in ep]
        vc = [(e.get("verbconf") or {}).get("value") for e in ep]
        out.update(ep_n=len(ep), ptrue_cov=np.mean([p is not None for p in pt]),
                   ptrue_mean=float(np.mean([p for p in pt if p is not None])) if any(p is not None for p in pt) else np.nan,
                   ptrue_const=(float(np.std([p for p in pt if p is not None]) < 1e-3)
                                if any(p is not None for p in pt) else np.nan),
                   verb_cov=np.mean([v is not None for v in vc]),
                   verb_mean=float(np.mean([v for v in vc if v is not None])) if any(v is not None for v in vc) else np.nan,
                   pce_present=np.mean([bool(e.get("pce") and e["pce"][0]) for e in ep]))
    hf = C.HIDDEN / f"{tag}.npz"
    if hf.exists():
        h = np.load(hf)
        H = h["H"].astype(np.float32)
        out.update(hs_shape=str(H.shape), hs_finite=bool(np.isfinite(H).all()),
                   hs_zero_rows=int((np.abs(H).reshape(len(H), -1).sum(1) == 0).sum()),
                   hs_ids_ok=bool(list(h["ids"]) == [r["id"] for r in rows]))
    nf = C.NLI / f"{tag}.npz"
    if nf.exists():
        n = np.load(nf)
        out.update(nli_E_nan=float(np.isnan(n["E"]).mean()), nli_sc_mean=float(np.mean(n["selfcheck"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args()
    tags = args.tags or sorted(os.path.basename(p)[:-len(".json.gz")]
                               for p in glob.glob(str(C.GEN / "*_k8.json.gz")))
    for t in tags:
        o = audit(t)
        print(f"== {t}")
        for k, v in o.items():
            print(f"   {k:20s} {v:.3f}" if isinstance(v, (float, np.floating)) else f"   {k:20s} {v}")


if __name__ == "__main__":
    main()
