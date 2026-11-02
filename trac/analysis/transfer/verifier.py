"""Fine-tuned text verifiers against and with TrAC under label-free transfer (CPU).

The verifiers (trac.pipeline.verifier) are fine-tuned on the other backbones' labeled responses to the
other datasets (LOMDO) and read either the answer a user sees or the full reasoning trace. The full-trace
verifier is combined with the TrAC classifier transferred under the same protocol by averaging the two
scores' within-setting ranks, which needs no label and no fitted weight.

  python -m trac.analysis.transfer.verifier      -> EXP_ROOT/reports/verifier_transfer.json
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.stats import hboot
from trac.analysis.transfer.common import load_all
from trac.benchmarks import tag_of

PAIRS = [("TrAC", "Verifier, full trace"), ("TrAC + verifier", "TrAC"), ("TrAC + verifier", "Verifier, full trace")]


def verifier_scores(kind, tag, ids):
    """Scores of a fine-tuned verifier (`answer` or `full`) aligned to `ids`, or None."""
    f = C.VERIFIER / kind / f"{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    pos = {str(q): i for i, q in enumerate(z["ids"])}
    return z["score"][[pos[str(q)] for q in ids]].astype(float)


def main():
    S = {}
    for (d, m), B in load_all().items():
        tag = tag_of(d, m)
        answer, full = verifier_scores("answer", tag, B["ids"]), verifier_scores("full", tag, B["ids"])
        if answer is None or full is None:
            continue
        t = np.load(C.TRANSFER / "lomdo_rank" / f"{tag}.npz")
        assert (t["y"] == B["y"].astype(int)).all()
        S[tag] = {
            "y": t["y"],
            "TrAC": t["TrAC"],
            "Verifier, answer": answer,
            "Verifier, full trace": full,
            "TrAC + verifier": rankdata(t["TrAC"]) + rankdata(full),
        }
    names = [n for n in next(iter(S.values())) if n != "y"]
    per = {t: {n: float(roc_auc_score(s["y"], s[n])) for n in names} for t, s in S.items()}
    res = {"n": len(S), "per": per, "macro": {n: float(np.mean([x[n] for x in per.values()])) for n in names}, "pairs": {}}
    for a, b in PAIRS:
        mean, lo, hi = hboot([(s["y"], s[a], s[b]) for s in S.values()], n=1000)
        res["pairs"][f"{a} - {b}"] = {"diff": mean, "lo": lo, "hi": hi, "wins": int(sum(x[a] > x[b] for x in per.values()))}
    print(res["n"], {n: round(v, 3) for n, v in res["macro"].items()})
    for k, v in res["pairs"].items():
        print("  ", k, {n: round(x, 3) for n, x in v.items()})
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(C.REPORTS / "verifier_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
