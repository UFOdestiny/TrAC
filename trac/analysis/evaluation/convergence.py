"""When does the model commit to its answer? (CPU)

backbones    per backbone, the median convergence point of correct and incorrect responses on the
             20-checkpoint grid (earliest fraction from which the greedy intermediate answer stays
             equal to a; mean of per-setting medians), with and without responses whose end-of-trace
             intermediate answer is unparsable, and the share of unparsable end answers
text         intermediate answers against the answers already written in the trace text
             (trac.analysis.features.text): training-free AUROC of both, and how often the
             model converges strictly before the text states a
within       the eight traces of each question (EXP_ROOT/sample_trajectory: greedy intermediate
             answers at 8 checkpoints), which controls for question difficulty: convergence points
             of correct and incorrect traces of the same question, and per-question AUROC of the
             share of intermediate answers equal to a, mean token log-probability, and output length

python -m trac.analysis.evaluation.convergence      -> EXP_ROOT/reports/convergence.json
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from trac import config as C
from trac.analysis.cells import load_cell
from trac.analysis.features.text import GRID as TEXT_GRID
from trac.analysis.features.text import commit as text_commit
from trac.analysis.features.trajectory import answer_share, greedy_only, thin
from trac.benchmarks import DATASET_KEYS, MIN_CLASS_COUNT, MODEL_KEYS, tag_of
from trac.store import read_json_gz

CONVERGENCE = 44  # column of TRAJ: convergence point of the greedy intermediate answers
N_CK = 8  # checkpoints of the sample trajectories


def eligible(y):
    return min(y.sum(), (1 - y).sum()) >= MIN_CLASS_COUNT


def backbones(cells):
    out = {}
    for m in MODEL_KEYS:
        rows = {"correct": [], "incorrect": [], "correct_parsed": [], "incorrect_parsed": [], "unparsable": []}
        for (d, mm), B in cells.items():
            if mm != m:
                continue
            y = B["y"].astype(int)
            cp = B["TRAJ"][:, CONVERGENCE]
            R = {x["id"]: x for x in read_json_gz(C.TRAJECTORY / f"{tag_of(d, m)}.json.gz")}
            unp = np.array([str(q) in R and (R[str(q)]["end"] or {}).get("ans") is None for q in B["ids"]])
            rows["correct"].append(np.median(cp[y == 1]))
            rows["incorrect"].append(np.median(cp[y == 0]))
            rows["correct_parsed"].append(np.median(cp[(y == 1) & ~unp]))
            rows["incorrect_parsed"].append(np.median(cp[(y == 0) & ~unp]))
            rows["unparsable"].append(unp.mean())
        out[m] = {k: float(np.mean(v)) for k, v in rows.items()} | {"settings": len(rows["correct"])}
    return out


def text(cells):
    """Intermediate answers against the text written so far, on the same grid."""
    G = len(TEXT_GRID)
    auroc, early = {}, {"correct": [], "incorrect": []}
    for (d, m), B in cells.items():
        f = C.FEATURE_CACHE / f"text_trajectory__{tag_of(d, m)}.npz"
        if not f.exists():
            continue
        y = np.load(C.FEATURES / f"{tag_of(d, m)}.npz")["y"].astype(int)
        z = np.load(f)
        TT, MG = z["TEXTTRAJ"], z["MODELG"]
        ok = np.isfinite(TT[:, 0])
        model_cp = np.array([text_commit(g) if np.isfinite(g).all() else np.nan for g in MG])
        text_cp = TT[:, G + 1]
        for k, s in [
            ("text share", TT[:, G]),
            ("text convergence", -text_cp),
            ("model share", np.nanmean(MG, 1)),
            ("model convergence", -model_cp),
        ]:
            s = np.where(np.isfinite(s), s, np.nanmedian(s))
            auroc.setdefault(k, []).append(roc_auc_score(y, s))
        both = ok & np.isfinite(model_cp)
        before = model_cp < text_cp
        early["correct"].append(np.mean(before[both & (y == 1)]))
        early["incorrect"].append(np.mean(before[both & (y == 0)]))
    return {
        "settings": len(early["correct"]),
        "auroc": {k: float(np.mean(v)) for k, v in auroc.items()},
        "model_before_text": {k: float(np.mean(v)) for k, v in early.items()},
    }


def sample_convergence(agree):
    """Earliest checkpoint fraction j/9 from which every later checkpoint returns the trace's own answer."""
    a = [bool(x) for x in agree[:N_CK]]
    if not a or not a[-1]:
        return 1.0
    j = len(a) - 1
    while j > 0 and a[j - 1]:
        j -= 1
    return (j + 1) / (N_CK + 1)


def per_question_auroc(groups):
    """Mean AUROC over questions whose traces have both labels (ties count half)."""
    v = [roc_auc_score(y, s) for y, s in groups if 0 < y.sum() < len(y)]
    return float(np.mean(v)) if v else np.nan


def within(cells):
    per = {}
    for (d, m), B in cells.items():
        tag = tag_of(d, m)
        f = C.SAMPLE_TRAJECTORY / f"{tag}.json.gz"
        if not f.exists():
            continue
        G = {r["id"]: r for r in read_json_gz(C.GEN / f"{tag}.json.gz")}
        records = read_json_gz(f)
        TS = {x["id"]: x["samples"] for x in records}
        R = {x["id"]: x for x in read_json_gz(C.TRAJECTORY / f"{tag}.json.gz")}
        gaps, scores = [], {"share": [], "mean log-prob": [], "length": []}
        for x in records:
            g = G.get(x["id"])
            if g is None:
                continue
            cp = {True: [], False: []}
            for s in x["samples"]:
                if s.get("agree_own") and s["k"] < len(g["samples"]):
                    cp[bool(g["samples"][s["k"]]["correct"])].append(sample_convergence(s["agree_own"]))
            if cp[True] and cp[False]:
                gaps.append(np.mean(cp[False]) - np.mean(cp[True]))
        for i, q in enumerate(str(x) for x in B["ids"]):
            g = G.get(q)
            if g is None or q not in R:
                continue
            labels = np.array([int(g["correct"])] + [int(s["correct"]) for s in g["samples"]])
            by_k = {s["k"]: s for s in TS.get(q, [])}
            share = [answer_share(thin(greedy_only(R[q]), N_CK))]
            share += [
                float(np.mean(by_k[k]["agree_own"])) if k in by_k and by_k[k]["agree_own"] else np.nan
                for k in range(len(g["samples"]))
            ]
            mean_lp = [float(B["BASE"][i, 0])] + [s.get("meanlp", np.nan) for s in g["samples"]]
            length = [-float(g["n_tokens"])] + [-float(s.get("n_tokens", np.nan)) for s in g["samples"]]
            for k, s in [("share", share), ("mean log-prob", mean_lp), ("length", length)]:
                s = np.array(s, float)
                if np.isfinite(s).all():
                    scores[k].append((labels, s))
        per[tag] = {
            "mixed_questions": len(gaps),
            "mean_gap": float(np.mean(gaps)) if gaps else np.nan,
            "correct_earlier": float(np.mean([x > 0 for x in gaps])) if gaps else np.nan,
            "ties": float(np.mean([x == 0 for x in gaps])) if gaps else np.nan,
            "auroc": {k: per_question_auroc(v) for k, v in scores.items()},
        }
    ok = {t: v for t, v in per.items() if v["mixed_questions"] >= 10}
    return {
        "settings": len(ok),
        "mixed_questions": int(sum(v["mixed_questions"] for v in ok.values())),
        "incorrect_later": int(sum(v["mean_gap"] > 0 for v in ok.values())),
        **{k: float(np.mean([v[k] for v in ok.values()])) for k in ("mean_gap", "correct_earlier", "ties")},
        "auroc": {k: float(np.nanmean([v["auroc"][k] for v in ok.values()])) for k in ("share", "mean log-prob", "length")},
        "share_above_mean_log_prob": int(sum(v["auroc"]["share"] > v["auroc"]["mean log-prob"] for v in ok.values())),
        "per": per,
    }


def main():
    cells = {}
    for d in DATASET_KEYS:
        for m in MODEL_KEYS:
            if (C.FEATURES / f"{tag_of(d, m)}.npz").exists():
                B = load_cell(tag_of(d, m))
                if eligible(B["y"]) and "TRAJ" in B:
                    cells[(d, m)] = B
    res = {"backbones": backbones(cells), "text": text(cells), "within": within(cells)}
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "convergence.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "within"} | {"within": {k: v for k, v in res["within"].items() if k != "per"}},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
