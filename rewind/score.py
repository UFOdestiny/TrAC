"""Fit and score REWIND on the feature blocks of a setting (CPU).

  python -m rewind.score --tag math500_qwen8b_k8                in-domain, out of fold
  python -m rewind.score --tag math500_qwen8b_k8 --label-free   label-free transfer to the setting

In-domain: the stacked classifier is trained and scored out of fold on the setting's own labels
(5 shuffled 5-fold partitions, scores averaged; rewind.classifier.stacked_oof) for REWIND, REWIND-Lite,
and, when the setting has K > 1 traces per question, REWIND@8 (sample-consensus features as
meta-classifier inputs).

Label-free: the classifier is fitted on the labeled source settings of the main matrix that share neither
the dataset nor the model of the target (which may be a model outside the matrix), with every feature
replaced by its rank within its own setting (rewind.transfer.rank_norm); the target's labels are not
used, except to report AUROC.

Writes EXP_ROOT/rewind_scores/{tag}.npz with the labels y and one score array per variant (in-domain
variants by name, label-free ones with the suffix " (label-free)"); variants scored earlier are kept.
Run single-threaded (OMP_NUM_THREADS=1).
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from rewind import config as C
from rewind.benchmarks import parse_tag
from rewind.classifier import SEEDS, stack_fit_predict, stacked_oof
from rewind.features.setting import load_setting
from rewind.metrics import aurc, risk_at_coverage
from rewind.store import save_npz_atomic
from rewind.transfer import load_settings, rank_norm, sources
from rewind.views import consensus_f, lite_f, rewind_f

VARIANTS = {"REWIND": (rewind_f, ()), "REWIND-Lite": (lite_f, ("RUNNER_UP_END",))}
NEEDS = ("INTERMEDIATE_4", "TFT", "RUNNER_UP")  # blocks of the intermediate-answer and runner-up caches


def traces_per_question(tag):
    return int(tag.rsplit("_k", 1)[1])


def in_domain(tag, B, y, seeds=SEEDS):
    """{variant: out-of-fold scores} of one setting."""
    out = {}
    for name, (view, needs) in VARIANTS.items():
        if all(k in B for k in needs):
            X = view(B)
            out[name] = stacked_oof([(X, "lr"), (X, "gb")], None, y, seeds)
    if traces_per_question(tag) > 1:
        X = rewind_f(B)
        out["REWIND@8"] = stacked_oof([(X, "lr"), (X, "gb")], consensus_f(B), y, seeds)
    return {k: v for k, v in out.items() if v is not None}


def label_free(tag, B):
    """{variant (label-free): scores} of one setting, from the LOMDO source settings with rank features."""
    d, m = parse_tag(tag)
    cells = load_settings()
    src = sources(cells, d, m, "lomdo")
    if not src:
        raise SystemExit(f"no source settings for {tag}: build the features of the main matrix first")
    out = {}
    for name, (view, needs) in VARIANTS.items():
        if all(k in B for k in needs) and all(k in cells[s] for s in src for k in needs):
            Xtr = np.vstack([rank_norm(view(cells[s])) for s in src])
            ytr = np.concatenate([cells[s]["y"] for s in src]).astype(int)
            out[f"{name} (label-free)"] = stack_fit_predict(Xtr, ytr, rank_norm(view(B)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="cache tag of the setting, e.g. math500_qwen8b_k8")
    ap.add_argument("--label-free", action="store_true", help="transfer from the source settings instead")
    ap.add_argument("--quick", action="store_true", help="in-domain: one outer partition instead of five")
    args = ap.parse_args()
    B = load_setting(args.tag)
    missing = [k for k in NEEDS if k not in B]
    if missing:
        raise SystemExit(f"{args.tag}: missing blocks {missing}; run intermediate_answers and runner_up first")
    y = B["y"].astype(int)
    scores = label_free(args.tag, B) if args.label_free else in_domain(args.tag, B, y, SEEDS[:1] if args.quick else SEEDS)
    if not scores:
        raise SystemExit(f"{args.tag}: too few correct or incorrect responses to score out of fold")
    path = C.REWIND_SCORES / f"{args.tag}.npz"
    out = dict(np.load(path)) if path.exists() else {}
    out.update({"y": y, **scores})
    save_npz_atomic(path, **out)
    for name, s in scores.items():
        auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
        print(f"[score] {args.tag} {name}: AUROC {auc:.3f}  AURC {aurc(s, y):.3f}  Risk@10% {risk_at_coverage(s, y, 0.1):.3f}")
    print(f"[score] saved -> {path}")


if __name__ == "__main__":
    main()
