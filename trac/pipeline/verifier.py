"""Fine-tuned text verifier for label-free transfer (one GPU), in the spirit of auxiliary calibrator
models (Ulmer et al., 2024): Qwen3-1.7B with a classification head is fine-tuned for one epoch to predict
correctness from the question and the response, on the labeled original responses of the other backbones
of the main matrix for the other datasets (LOMDO), and scores the held-out backbone's responses to the
held-out dataset. It reads the answer a user sees (as the reward model does), or with --full the whole
response including the reasoning trace (truncated to --max-len tokens, keeping its beginning and end).

  python -m trac.pipeline.verifier --model qwen8b --dataset gsm8k [--full]   -> EXP_ROOT/verifier/{answer,full}/{tag}.npz
  python -m trac.pipeline.verifier --model qwen32b --k 1 --dataset gsm8k --full
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from trac import backbones as M
from trac import config as C
from trac.benchmarks import DATASET_KEYS, MODEL_KEYS, tag_of
from trac.config import model_path
from trac.pipeline.reward import TRAILERS, response_text
from trac.store import read_json_gz, save_npz_atomic


def full_text(r):
    text = r["raw"]
    for t in TRAILERS:
        text = text.replace(t, "")
    return text.strip() or "(empty)"


def examples(tok, m, max_len, datasets, full=False, k=8):
    out = []
    bb = M.get(m)
    head = 512 if full else 200
    for d in datasets:
        tag = tag_of(d, m, k)
        rows = {r["id"]: r for r in read_json_gz(C.GEN / f"{tag}.json.gz")}  # last row per id
        for r in rows.values():
            resp = tok.encode(full_text(r) if full else response_text(bb, r), add_special_tokens=False)
            q = tok.encode(r["question"], add_special_tokens=False)[:400]
            ids = q + tok.encode("\n\nResponse:\n", add_special_tokens=False) + resp
            ids = ids[:head] + ids[-(max_len - head) :] if len(ids) > max_len else ids
            out.append((tag, str(r["id"]), ids, int(r["correct"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="held-out backbone (target)")
    ap.add_argument("--dataset", required=True, help="held-out dataset (target)")
    ap.add_argument("--k", type=int, default=8, help="traces per question of the target's generation cache")
    ap.add_argument("--base", default="Qwen3-1.7B")
    ap.add_argument("--full", action="store_true", help="read the reasoning trace too")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    out = C.VERIFIER / ("full" if a.full else "answer") / f"{tag_of(a.dataset, a.model, a.k)}.npz"
    if out.exists():
        print(f"[verifier] exists {out}")
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    random.seed(0)
    tok = AutoTokenizer.from_pretrained(model_path(a.base))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_path(a.base), num_labels=1, dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    model.cuda().train()
    if a.full:
        model.gradient_checkpointing_enable()
    sources = [d for d in DATASET_KEYS if d != a.dataset]
    train = [e for m in MODEL_KEYS if m != a.model for e in examples(tok, m, a.max_len, sources, a.full)]
    test = examples(tok, a.model, a.max_len, [a.dataset], a.full, a.k)
    random.shuffle(train)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(train) + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    pos = np.mean([e[3] for e in train])
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos).cuda())

    def batches(data, bs):
        for i in range(0, len(data), bs):
            b = data[i : i + bs]
            L = max(len(e[2]) for e in b)
            ids = torch.full((len(b), L), tok.pad_token_id)
            att = torch.zeros((len(b), L), dtype=torch.long)
            for k, e in enumerate(b):
                ids[k, : len(e[2])] = torch.tensor(e[2])
                att[k, : len(e[2])] = 1
            yield b, ids.cuda(), att.cuda()

    t0 = time.time()
    for s, (b, ids, att) in enumerate(batches(train, a.batch)):
        logit = model(input_ids=ids, attention_mask=att).logits[:, 0].float()
        loss = lossf(logit, torch.tensor([e[3] for e in b], dtype=torch.float32).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if s % 100 == 0:
            print(f"[verifier] step {s}/{steps} loss {loss.item():.3f} {(time.time() - t0) / 60:.1f} min", flush=True)
    model.eval()
    scores = {}
    test_sorted = sorted(test, key=lambda e: len(e[2]))
    with torch.no_grad():
        for b, ids, att in batches(test_sorted, 32):
            logits = model(input_ids=ids, attention_mask=att).logits[:, 0].float().cpu().numpy()
            for e, o in zip(b, logits):
                scores.setdefault(e[0], {})[e[1]] = float(o)
    d = scores[out.stem]
    save_npz_atomic(out, ids=np.array(list(d)), score=np.array(list(d.values()), np.float32))
    print(f"[verifier] {out.stem}: trained on {len(train)}, scored {len(test)} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
