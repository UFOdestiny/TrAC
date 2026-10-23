"""NLI evidence for the sampling-based baselines (DeBERTa-large-MNLI, one GPU).

Per question, over the K returned answers (original first, then K-1 samples), each
rendered as "<question> Answer: <extracted answer>" (the answer section tail when no
answer can be extracted), as in Kuhn et al. / Lin et al.:
  E[i, j], C[i, j]  P(entailment), P(contradiction) for premise i -> hypothesis j
  selfcheck         SelfCheckGPT-NLI (Manakul et al., 2023): mean over the original's
                    sentences s and samples j of P(contra) / (P(contra) + P(entail))
                    with premise = sample j, hypothesis = s
Semantic entropy, EigV / Ecc / Deg (Lin et al., 2024) and KLE (Nikitin et al., 2024)
are computed from E in trac.analysis.features.

Writes EXP_ROOT/nli/{tag}.npz.

  python -m trac.pipeline.nli --gen-tag math500_qwen8b_k8
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np

from trac import answers as A
from trac import backbones as M
from trac import config as cfg
from trac.config import model_path
from trac.store import read_json_gz

ANSWER_CHARS = 600      # tail of the answer section (keeps the boxed conclusion)
MAX_SENTS = 8           # SelfCheck: last sentences of the original's answer section


def render(q, ans):
    return f"{q.strip()[:1000]} Answer: {ans.strip()[-ANSWER_CHARS:]}"


def sentences(text):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if len(x.strip()) > 3]
    return s[-MAX_SENTS:] if s else [text.strip()[-ANSWER_CHARS:] or "(empty)"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--nli-model", default="deberta-large-mnli")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    out = cfg.NLI / f"{args.gen_tag}.npz"
    if out.exists():
        print(f"[nli] exists {out}"); return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path(args.nli_model))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path(args.nli_model), dtype=torch.float32).cuda().eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    ie, ic = lab["entailment"], lab["contradiction"]

    rows = read_json_gz(cfg.GEN / f"{args.gen_tag}.json.gz")
    K = 1 + max(len(r["samples"]) for r in rows)
    pairs, index = [], []          # (premise, hypothesis), (row, kind, a, b)
    bb = M.from_tag(args.gen_tag)
    for n, r in enumerate(rows):
        if bb is not None:
            r["final"] = M.final_for(bb, r["raw"], r["finish"])
            for x in r["samples"]:
                x["final"] = M.final_for(bb, x["raw"], x["finish"])
        texts = [r["final"] or ""] + [s["final"] or "" for s in r["samples"]]
        answers = [A.extract_answer(r, t) for t in texts]
        rend = [render(r["question"], str(a) if a is not None else t) for a, t in zip(answers, texts)]
        for a in range(len(rend)):
            for b in range(len(rend)):
                if a != b:
                    pairs.append((rend[a], rend[b])); index.append((n, 0, a, b))
        for si, sent in enumerate(sentences(texts[0])):
            for j in range(1, len(texts)):
                pairs.append((texts[j].strip()[-ANSWER_CHARS:] or "(empty)", sent)); index.append((n, 1, j, si))

    t0 = time.time()
    probs = np.zeros((len(pairs), 3), np.float32)
    order = np.argsort([len(p[0]) + len(p[1]) for p in pairs])
    with torch.inference_mode():
        for s0 in range(0, len(order), args.batch):
            sel = order[s0:s0 + args.batch]
            enc = tok([pairs[k][0] for k in sel], [pairs[k][1] for k in sel], truncation=True,
                      max_length=512, padding=True, return_tensors="pt").to("cuda")
            probs[sel] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
    E = np.full((len(rows), K, K), np.nan, np.float32); C = E.copy()
    sc_num = np.zeros(len(rows)); sc_den = np.zeros(len(rows))
    for (n, kind, a, b), p in zip(index, probs):
        if kind == 0:
            E[n, a, b], C[n, a, b] = p[ie], p[ic]
        else:
            sc_num[n] += p[ic] / max(p[ic] + p[ie], 1e-6); sc_den[n] += 1
    for n in range(len(rows)):
        np.fill_diagonal(E[n], 1.0); np.fill_diagonal(C[n], 0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array([r["id"] for r in rows]), E=E, C=C,
             selfcheck=sc_num / np.maximum(sc_den, 1))
    print(f"[nli] {len(pairs)} pairs in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
