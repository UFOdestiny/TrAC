"""Auxiliary per-response statistics used by the paper's tables (CPU).

Per question of a cell: SC@k (k = 2, 4) vote share of the returned answer among the first k
traces, the error taxonomy of the completed-prefix re-elicited answer, whether the returned
answer string already appears in the reasoning body, and the output length.

  python -m trac.analysis.auxiliary [--tags T ...]   -> EXP_ROOT/auxiliary/{T}.npz
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as C
from trac.store import read_json_gz

# taxonomy of the re-elicited answer of an incorrect response (-1 for correct responses)
TAXONOMY = ("re-elicited correct", "same wrong answer", "different wrong answer", "unparsable")


def build(tag):
    gen = read_json_gz(C.GEN / f"{tag}.json.gz")
    ep = {r["id"]: r for r in read_json_gz(C.ENDPOINT / f"{tag}.json.gz")}
    bb = M.from_tag(tag)
    out = defaultdict(list)
    for g in gen:
        fin = M.final_for(bb, g["raw"], g["finish"])
        ans = A.extract_answer(g, fin)
        sans = [A.extract_answer(g, M.final_for(bb, s["raw"], s["finish"])) for s in g["samples"]]
        for k in (2, 4):
            pool = [ans] + sans[: k - 1]
            out[f"sc{k}"].append(sum(A.equiv(g, a, ans) for a in pool) / k if ans is not None else 0.0)
        e = ep.get(g["id"]) or {}
        p0 = (e.get("pce") or [None])[0]
        pa = A.extract_head(g, p0["text"]) if p0 else None
        if A.verify_answer_text(g, fin):
            cat = -1
        elif pa is None:
            cat = 3
        elif A.verify_answer_text(g, "\\boxed{" + str(pa) + "}"):
            cat = 0
        elif ans is not None and A.equiv(g, pa, ans):
            cat = 1
        else:
            cat = 2
        out["tax"].append(cat)
        body = M.reasoning_body(bb, g["raw"])
        out["in_trace"].append(float(ans is not None and len(str(ans)) > 0 and str(ans) in body))
        out["ntok"].append(g["n_tokens"])
        out["ids"].append(g["id"])
    C.AUXILIARY.mkdir(parents=True, exist_ok=True)
    np.savez(C.AUXILIARY / f"{tag}.npz", **{k: np.array(v) for k, v in out.items()})
    print(f"[aux] {tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=None, help="default: every main-matrix cell (*_k8)")
    args = ap.parse_args()
    for t in args.tags or sorted(os.path.basename(p)[:-len(".json.gz")]
                                 for p in glob.glob(str(C.ENDPOINT / "*_k8.json.gz"))):
        build(t)


if __name__ == "__main__":
    main()
