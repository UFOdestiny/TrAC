"""External reward-model baseline (one GPU): a model-agnostic judge scores the original response
of every question with Skywork-Reward-V2-Llama-3.1-8B, without any information from the backbone.

The judged response is what a user sees: the answer section after the reasoning for thinking
backbones (Qwen3 `</think>`, gpt-oss `final` channel), the whole response for instruct backbones.
Responses longer than --max-response tokens keep their tail (the conclusion).

Writes EXP_ROOT/reward/{tag}.npz with ids, the scalar reward, and the judge's last hidden state at the
final token (float16), the response embedding of the transferred judge classifier
(trac.analysis.transfer.judge).

  python -m trac.pipeline.reward --gen-tag math500_qwen8b_k8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from trac import backbones as M
from trac import config as C
from trac.config import model_path
from trac.store import read_json_gz, save_npz_atomic

TRAILERS = ["<|eot_id|>", "<|im_end|>", "<|endoftext|>", "<|return|>", "<|end|>", "</s>"]


def response_text(bb, r):
    if bb.fmt == "plain":
        text = r["raw"]
    else:
        text = r["final"] or M.reasoning_body(bb, r["raw"])[-4000:]
    for t in TRAILERS:
        text = text.replace(t, "")
    return text.strip() or "(empty)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--rm", default="Skywork-Reward-V2-Llama-3.1-8B")
    ap.add_argument("--max-response", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    out = C.REWARD / f"{a.gen_tag}.npz"
    if out.exists():
        print(f"[reward] exists {out}")
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    bb = M.from_tag(a.gen_tag)
    tok = AutoTokenizer.from_pretrained(model_path(a.rm))
    rm = (
        AutoModelForSequenceClassification.from_pretrained(
            model_path(a.rm), dtype=torch.bfloat16, attn_implementation="sdpa", num_labels=1
        )
        .cuda()
        .eval()
    )
    rows = read_json_gz(C.GEN / f"{a.gen_tag}.json.gz")
    seqs = []
    for r in rows:
        resp = tok.encode(response_text(bb, r), add_special_tokens=False)[-a.max_response :]
        conv = [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": tok.decode(resp)}]
        seqs.append(tok.apply_chat_template(conv, tokenize=True, return_dict=False))
    order = np.argsort([len(s) for s in seqs])
    scores = np.zeros(len(seqs), np.float32)
    emb = np.zeros((len(seqs), rm.config.hidden_size), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(order), a.batch):
            idx = order[i : i + a.batch]
            L = max(len(seqs[j]) for j in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for k, j in enumerate(idx):  # right padding; the head reads the last real token
                ids[k, : len(seqs[j])] = torch.tensor(seqs[j])
                att[k, : len(seqs[j])] = 1
            o = rm(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            scores[idx] = o.logits[:, 0].float().cpu().numpy()
            last = att.sum(1) - 1  # final real token of each sequence
            h = o.hidden_states[-1][torch.arange(len(idx)), last.cuda()]
            emb[idx] = h.float().cpu().numpy().astype(np.float16)
    save_npz_atomic(out, ids=np.array([str(r["id"]) for r in rows]), reward=scores, emb=emb)
    print(f"[reward] {a.gen_tag}: {len(rows)} responses in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
